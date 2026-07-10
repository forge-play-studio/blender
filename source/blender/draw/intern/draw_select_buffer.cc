/* SPDX-FileCopyrightText: 2019 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup draw_engine
 *
 * Utilities to read id buffer created in select_engine.
 */

#include <cfloat>

#include "BLI_math_matrix.hh"
#include "MEM_guardedalloc.h"

#include "BLI_array.hh"
#include "ED_view3d.hh"
#include "BLI_array_utils_c.hh"
#include "BLI_bitmap.hh"
#include "BLI_bitmap_draw_2d.hh"
#include "BLI_math_matrix_c.hh"
#include "BLI_rect.hh"

#include "DNA_layer_types.h"
#include "DNA_scene_types.h"
#include "DNA_screen_types.h"
#include "DNA_view3d_types.h"

#include "GPU_framebuffer.hh"
#include "GPU_select.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_query.hh"

#include "DRW_engine.hh"
#include "DRW_render.hh"
#include "DRW_select_buffer.hh"

#include "../engines/select/select_engine.hh"

#ifdef __EMSCRIPTEN__
#  include "BKE_editmesh.hh"
#  include "bmesh.hh"
#endif

namespace blender {

bool SELECTID_Context::is_dirty(Depsgraph *depsgraph, RegionView3D *rv3d)
{
  uint64_t last_update = this->depsgraph_last_update;
  this->depsgraph_last_update = DEG_get_update_count(depsgraph);

  /* Check if the viewport has changed.
   * This can happen when triggering the selection operator *while* playing back animation and
   * looking through an animated camera. */
  if (!math::is_equal(this->persmat, float4x4(rv3d->persmat), FLT_EPSILON)) {
    return true;
  }
  /* Check if any of the drawn objects have been transformed.
   * This can happen when triggering the selection operator *while* playing back animation on an
   * edited mesh. */
  for (Object *obj_eval : this->objects) {
    if (obj_eval->runtime->last_update_transform > last_update) {
      return true;
    }
  }
  return false;
}

/* -------------------------------------------------------------------- */
/** \name Buffer of select ID's
 * \{ */

#ifdef __EMSCRIPTEN__
/* The WebGPU backend cannot read the select-id texture back synchronously (no
 * JSPI): GPU_framebuffer_read_color returns zeros and edit-mode element picking
 * would select nothing. Rasterize the ids on the CPU into `buf` instead:
 * project each BMesh element with the region's perspective matrix and splat its
 * select id, in the engine's priority order (faces, then edges, then verts, so
 * verts win ties like the real id pass). No occlusion — hidden elements are
 * pickable, matching x-ray select-through behavior. */
static void select_buffer_cpu_fallback(SELECTID_Context *select_ctx,
                                       ARegion *region,
                                       View3D *v3d,
                                       const rcti *rect,
                                       uint *buf)
{
  RegionView3D *rv3d = static_cast<RegionView3D *>(region->regiondata);
  const float4x4 persmat(rv3d->persmat);
  const int rect_w = BLI_rcti_size_x(rect);
  const int rect_h = BLI_rcti_size_y(rect);
  const float2 half_size(region->winx * 0.5f, region->winy * 0.5f);

  /* Project a local-space point to region pixels (+ NDC depth); false when
   * behind the eye. */
  auto project = [&](const float4x4 &obmat, const float3 &co, float3 &r_px) {
    const float4 hp = persmat * (obmat * float4(co, 1.0f));
    if (hp.w < 1e-6f) {
      return false;
    }
    r_px = float3((float2(hp.x, hp.y) / hp.w + 1.0f) * half_size, hp.z / hp.w);
    return true;
  };

  /* Occlusion: rasterize the (fan-triangulated) faces' NDC depth into a
   * z-buffer, then depth-test the element splats — the GPU id pass does the
   * same with a real depth buffer. X-ray shading keeps select-through. */
  const bool use_occlusion = (v3d == nullptr) ? true :
                                                !SHADING_XRAY_FLAG_ENABLED(v3d->shading);
  blender::Array<float> zbuf;
  if (use_occlusion) {
    zbuf = blender::Array<float>(size_t(rect_w) * rect_h, FLT_MAX);
    auto raster_tri = [&](const float3 &a, const float3 &b, const float3 &c) {
      const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      if (fabsf(area) < 1e-9f) {
        return;
      }
      /* Polygon-offset style bias: one pixel of z slope + constant. */
      const float dzdx = ((b.z - a.z) * (c.y - a.y) - (c.z - a.z) * (b.y - a.y)) / area;
      const float dzdy = ((c.z - a.z) * (b.x - a.x) - (b.z - a.z) * (c.x - a.x)) / area;
      const float bias = fabsf(dzdx) + fabsf(dzdy) + 1e-4f;
      const int x0 = std::max(int(floorf(std::min({a.x, b.x, c.x}))) - rect->xmin, 0);
      const int x1 = std::min(int(ceilf(std::max({a.x, b.x, c.x}))) - rect->xmin, rect_w - 1);
      const int y0 = std::max(int(floorf(std::min({a.y, b.y, c.y}))) - rect->ymin, 0);
      const int y1 = std::min(int(ceilf(std::max({a.y, b.y, c.y}))) - rect->ymin, rect_h - 1);
      const float inv_area = 1.0f / area;
      for (int y = y0; y <= y1; y++) {
        for (int x = x0; x <= x1; x++) {
          const float px = float(x + rect->xmin) + 0.5f;
          const float py = float(y + rect->ymin) + 0.5f;
          const float w0 = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) * inv_area;
          const float w1 = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) * inv_area;
          const float w2 = 1.0f - w0 - w1;
          if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f) {
            continue;
          }
          const float z = w0 * a.z + w1 * b.z + w2 * c.z + bias;
          float &zb = zbuf[size_t(y) * rect_w + x];
          zb = std::min(zb, z);
        }
      }
    };
    for (Object *ob : select_ctx->objects) {
      BMEditMesh *em_z = BKE_editmesh_from_object(ob);
      if (em_z == nullptr || em_z->bm == nullptr) {
        continue;
      }
      const float4x4 obmat = float4x4(ob->object_to_world());
      BMFace *f;
      BMIter iter;
      BM_ITER_MESH (f, &iter, em_z->bm, BM_FACES_OF_MESH) {
        if (BM_elem_flag_test(f, BM_ELEM_HIDDEN)) {
          continue;
        }
        BMLoop *l_first = BM_FACE_FIRST_LOOP(f);
        float3 p0;
        if (!project(obmat, l_first->v->co, p0)) {
          continue; /* Crosses the near plane: conservatively occludes nothing. */
        }
        for (BMLoop *l = l_first->next; l->next != l_first; l = l->next) {
          float3 p1, p2;
          if (project(obmat, l->v->co, p1) && project(obmat, l->next->v->co, p2)) {
            raster_tri(p0, p1, p2);
          }
        }
      }
    }
  }

  auto splat = [&](const float3 &px, uint id) {
    const int x = int(px.x) - rect->xmin;
    const int y = int(px.y) - rect->ymin;
    if (x >= 0 && x < rect_w && y >= 0 && y < rect_h) {
      if (use_occlusion && px.z > zbuf[size_t(y) * rect_w + x]) {
        return; /* Hidden behind a face. */
      }
      buf[y * rect_w + x] = id;
    }
  };

  for (Object *ob : select_ctx->objects) {
    BMEditMesh *em = BKE_editmesh_from_object(ob);
    if (em == nullptr || em->bm == nullptr) {
      continue;
    }
    BMesh *bm = em->bm;
    const ElemIndexRanges ranges = select_ctx->elem_ranges.lookup_default(ob, ElemIndexRanges{});
    const float4x4 obmat = float4x4(ob->object_to_world());
    BMIter iter;

    if ((select_ctx->select_mode & SCE_SELECT_FACE) && !ranges.face.is_empty()) {
      BMFace *f;
      int i = 0;
      BM_ITER_MESH_INDEX (f, &iter, bm, BM_FACES_OF_MESH, i) {
        if (BM_elem_flag_test(f, BM_ELEM_HIDDEN)) {
          continue;
        }
        float3 center;
        BM_face_calc_center_median(f, center);
        float3 px;
        if (project(obmat, center, px)) {
          /* Small disk so face clicks don't need to hit the exact center px. */
          const uint id = uint(ranges.face.start()) + uint(i);
          for (int dy = -3; dy <= 3; dy++) {
            for (int dx = -3; dx <= 3; dx++) {
              if (dx * dx + dy * dy <= 9) {
                splat(px + float3(float(dx), float(dy), 0.0f), id);
              }
            }
          }
        }
      }
    }
    if ((select_ctx->select_mode & SCE_SELECT_EDGE) && !ranges.edge.is_empty()) {
      BMEdge *e;
      int i = 0;
      BM_ITER_MESH_INDEX (e, &iter, bm, BM_EDGES_OF_MESH, i) {
        if (BM_elem_flag_test(e, BM_ELEM_HIDDEN)) {
          continue;
        }
        float3 p0, p1;
        if (project(obmat, e->v1->co, p0) && project(obmat, e->v2->co, p1)) {
          const uint id = uint(ranges.edge.start()) + uint(i);
          const float len = math::distance(p0.xy(), p1.xy());
          const int steps = math::clamp(int(len), 1, 512);
          for (int s = 0; s <= steps; s++) {
            splat(math::interpolate(p0, p1, float(s) / steps), id);
          }
        }
      }
    }
    if ((select_ctx->select_mode & SCE_SELECT_VERTEX) && !ranges.vert.is_empty()) {
      BMVert *v;
      int i = 0;
      BM_ITER_MESH_INDEX (v, &iter, bm, BM_VERTS_OF_MESH, i) {
        if (BM_elem_flag_test(v, BM_ELEM_HIDDEN)) {
          continue;
        }
        float3 px;
        if (project(obmat, v->co, px)) {
          splat(px, uint(ranges.vert.start()) + uint(i));
        }
      }
    }
  }
}
#endif

uint *DRW_select_buffer_read(
    Depsgraph *depsgraph, ARegion *region, View3D *v3d, const rcti *rect, uint *r_buf_len)
{
  uint *buf = nullptr;
  uint buf_len = 0;

  /* Clamp rect. */
  rcti r{};
  r.xmin = 0;
  r.xmax = region->winx;
  r.ymin = 0;
  r.ymax = region->winy;

  /* Make sure that the rect is within the bounds of the viewport.
   * Some GPUs have problems reading pixels off limits. */
  rcti rect_clamp = *rect;
  if (BLI_rcti_isect(&r, &rect_clamp, &rect_clamp) && !BLI_rcti_is_empty(&rect_clamp)) {
    SELECTID_Context *select_ctx = DRW_select_engine_context_get();
    RegionView3D *rv3d = static_cast<RegionView3D *>(region->regiondata);

    DRW_gpu_context_enable();

    if (select_ctx->is_dirty(depsgraph, rv3d)) {
      /* Update drawing. */
      DRW_draw_select_id(depsgraph, region, v3d);
    }

    if (select_ctx->max_index_drawn_len > 1) {
      BLI_assert(region->winx == GPU_texture_width(DRW_engine_select_texture_get()) &&
                 region->winy == GPU_texture_height(DRW_engine_select_texture_get()));

      /* Read the UI32 pixels. */
      buf_len = BLI_rcti_size_x(rect) * BLI_rcti_size_y(rect);
      buf = MEM_new_array_uninitialized<uint>(buf_len, __func__);

      gpu::FrameBuffer *select_id_fb = DRW_engine_select_framebuffer_get();
      GPU_framebuffer_bind(select_id_fb);
      GPU_framebuffer_read_color(select_id_fb,
                                 rect_clamp.xmin,
                                 rect_clamp.ymin,
                                 BLI_rcti_size_x(&rect_clamp),
                                 BLI_rcti_size_y(&rect_clamp),
                                 1,
                                 0,
                                 GPU_DATA_UINT,
                                 buf);

      if (!BLI_rcti_compare(rect, &rect_clamp)) {
        /* The rect has been clamped so we need to realign the buffer and fill in the blanks */
        GPU_select_buffer_stride_realign(rect, &rect_clamp, buf);
      }

#ifdef __EMSCRIPTEN__
      /* The GPU read above returned zeros (deferred readback) — rasterize the
       * ids on the CPU so edit-mode picking works. */
      memset(buf, 0, sizeof(uint) * buf_len);
      select_buffer_cpu_fallback(select_ctx, region, v3d, rect, buf);
#endif
    }

    GPU_framebuffer_restore();
    DRW_gpu_context_disable();
  }

  if (r_buf_len) {
    *r_buf_len = buf_len;
  }

  return buf;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Select Bitmap from ID's
 *
 * Given a buffer of select ID's, fill in a booleans (true/false) per index.
 * #BLI_bitmap is used for memory efficiency.
 *
 * \{ */

uint *DRW_select_buffer_bitmap_from_rect(
    Depsgraph *depsgraph, ARegion *region, View3D *v3d, const rcti *rect, uint *r_bitmap_len)
{
  SELECTID_Context *select_ctx = DRW_select_engine_context_get();

  rcti rect_px = *rect;
  rect_px.xmax += 1;
  rect_px.ymax += 1;

  uint buf_len;
  uint *buf = DRW_select_buffer_read(depsgraph, region, v3d, &rect_px, &buf_len);
  if (buf == nullptr) {
    return nullptr;
  }

  BLI_assert(select_ctx->max_index_drawn_len > 0);
  const uint bitmap_len = select_ctx->max_index_drawn_len - 1;

  BLI_bitmap *bitmap_buf = BLI_BITMAP_NEW(bitmap_len, __func__);
  const uint *buf_iter = buf;
  while (buf_len--) {
    const uint index = *buf_iter - 1;
    if (index < bitmap_len) {
      BLI_BITMAP_ENABLE(bitmap_buf, index);
    }
    buf_iter++;
  }
  MEM_delete(buf);

  if (r_bitmap_len) {
    *r_bitmap_len = bitmap_len;
  }

  return bitmap_buf;
}

uint *DRW_select_buffer_bitmap_from_circle(Depsgraph *depsgraph,
                                           ARegion *region,
                                           View3D *v3d,
                                           const int center[2],
                                           const int radius,
                                           uint *r_bitmap_len)
{
  SELECTID_Context *select_ctx = DRW_select_engine_context_get();

  rcti rect{};
  rect.xmin = center[0] - radius;
  rect.xmax = center[0] + radius + 1;
  rect.ymin = center[1] - radius;
  rect.ymax = center[1] + radius + 1;

  const uint *buf = DRW_select_buffer_read(depsgraph, region, v3d, &rect, nullptr);

  if (buf == nullptr) {
    return nullptr;
  }

  BLI_assert(select_ctx->max_index_drawn_len > 0);
  const uint bitmap_len = select_ctx->max_index_drawn_len - 1;

  BLI_bitmap *bitmap_buf = BLI_BITMAP_NEW(bitmap_len, __func__);
  const uint *buf_iter = buf;
  const int radius_sq = radius * radius;
  for (int yc = -radius; yc <= radius; yc++) {
    for (int xc = -radius; xc <= radius; xc++, buf_iter++) {
      if (xc * xc + yc * yc < radius_sq) {
        /* Intentionally wrap to max value if this is zero. */
        const uint index = *buf_iter - 1;
        if (index < bitmap_len) {
          BLI_BITMAP_ENABLE(bitmap_buf, index);
        }
      }
    }
  }
  MEM_delete(buf);

  if (r_bitmap_len) {
    *r_bitmap_len = bitmap_len;
  }

  return bitmap_buf;
}

struct PolyMaskData {
  BLI_bitmap *px;
  int width;
};

static void drw_select_mask_px_cb(int x, int x_end, int y, void *user_data)
{
  PolyMaskData *data = static_cast<PolyMaskData *>(user_data);
  BLI_bitmap *px = data->px;
  int i = (y * data->width) + x;
  do {
    BLI_BITMAP_ENABLE(px, i);
    i++;
  } while (++x != x_end);
}

uint *DRW_select_buffer_bitmap_from_poly(Depsgraph *depsgraph,
                                         ARegion *region,
                                         View3D *v3d,
                                         const Span<int2> poly,
                                         const rcti *rect,
                                         uint *r_bitmap_len)
{
  SELECTID_Context *select_ctx = DRW_select_engine_context_get();

  rcti rect_px = *rect;
  rect_px.xmax += 1;
  rect_px.ymax += 1;

  uint buf_len;
  uint *buf = DRW_select_buffer_read(depsgraph, region, v3d, &rect_px, &buf_len);
  if (buf == nullptr) {
    return nullptr;
  }

  BLI_bitmap *buf_mask = BLI_BITMAP_NEW(buf_len, __func__);

  PolyMaskData poly_mask_data;
  poly_mask_data.px = buf_mask;
  poly_mask_data.width = (rect->xmax - rect->xmin) + 1;

  BLI_bitmap_draw_2d_poly_v2i_n(rect_px.xmin,
                                rect_px.ymin,
                                rect_px.xmax,
                                rect_px.ymax,
                                poly,
                                drw_select_mask_px_cb,
                                &poly_mask_data);

  BLI_assert(select_ctx->max_index_drawn_len > 0);
  const uint bitmap_len = select_ctx->max_index_drawn_len - 1;

  BLI_bitmap *bitmap_buf = BLI_BITMAP_NEW(bitmap_len, __func__);
  const uint *buf_iter = buf;
  int i = 0;
  while (buf_len--) {
    const uint index = *buf_iter - 1;
    if (index < bitmap_len && BLI_BITMAP_TEST(buf_mask, i)) {
      BLI_BITMAP_ENABLE(bitmap_buf, index);
    }
    buf_iter++;
    i++;
  }
  MEM_delete(buf);
  MEM_delete(buf_mask);

  if (r_bitmap_len) {
    *r_bitmap_len = bitmap_len;
  }

  return bitmap_buf;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Find Single Select ID's
 *
 * Given a buffer of select ID's, find the a single select id.
 *
 * \{ */

uint DRW_select_buffer_sample_point(Depsgraph *depsgraph,
                                    ARegion *region,
                                    View3D *v3d,
                                    const int center[2])
{
  uint ret = 0;

  rcti rect{};
  rect.xmin = center[0];
  rect.xmax = center[0] + 1;
  rect.ymin = center[1];
  rect.ymax = center[1] + 1;

  uint buf_len;
  uint *buf = DRW_select_buffer_read(depsgraph, region, v3d, &rect, &buf_len);
  if (buf) {
    BLI_assert(0 != buf_len);
    ret = buf[0];
    MEM_delete(buf);
  }

  return ret;
}

struct SelectReadData {
  const void *val_ptr;
  uint id_min;
  uint id_max;
  uint r_index;
};

static bool select_buffer_test_fn(const void *__restrict value, void *__restrict userdata)
{
  SelectReadData *data = static_cast<SelectReadData *>(userdata);
  uint hit_id = *static_cast<uint *>(const_cast<void *>(value));
  if (hit_id && hit_id >= data->id_min && hit_id < data->id_max) {
    /* Start at 1 to confirm. */
    data->val_ptr = value;
    data->r_index = (hit_id - data->id_min) + 1;
    return true;
  }
  return false;
}

uint DRW_select_buffer_find_nearest_to_point(Depsgraph *depsgraph,
                                             ARegion *region,
                                             View3D *v3d,
                                             const int center[2],
                                             const uint id_min,
                                             const uint id_max,
                                             uint *dist)
{
  /* Create region around center (typically the mouse cursor).
   * This must be square and have an odd width. */

  rcti rect;
  BLI_rcti_init_pt_radius(&rect, center, *dist);
  rect.xmax += 1;
  rect.ymax += 1;

  int width = BLI_rcti_size_x(&rect);
  int height = width;

  /* Read from selection framebuffer. */

  uint buf_len;
  const uint *buf = DRW_select_buffer_read(depsgraph, region, v3d, &rect, &buf_len);

  if (buf == nullptr) {
    return 0;
  }

  const int shape[2] = {height, width};
  const int center_yx[2] = {(height - 1) / 2, (width - 1) / 2};
  SelectReadData data = {nullptr, id_min, id_max, 0};
  BLI_array_iter_spiral_square(buf, shape, center_yx, select_buffer_test_fn, &data);

  if (data.val_ptr) {
    size_t offset = (size_t(data.val_ptr) - size_t(buf)) / sizeof(*buf);
    int hit_x = offset % width;
    int hit_y = offset / width;
    *dist = uint(abs(hit_y - center_yx[0]) + abs(hit_x - center_yx[1]));
  }

  MEM_delete(buf);
  return data.r_index;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Object Utils
 * \{ */

bool DRW_select_buffer_elem_get(const uint sel_id,
                                uint &r_elem,
                                uint &r_base_index,
                                char &r_elem_type)
{
  SELECTID_Context *select_ctx = DRW_select_engine_context_get();

  for (const auto &item : select_ctx->elem_ranges.items()) {
    const ElemIndexRanges &ranges = item.value;
    Object *ob = item.key;
    if (!ranges.total.contains(sel_id)) {
      continue;
    }
    if (ranges.face.contains(sel_id)) {
      r_elem = sel_id - ranges.face.start();
      r_elem_type = SCE_SELECT_FACE;
      r_base_index = select_ctx->objects.first_index_of_try(ob);
      return r_base_index != -1;
    }
    if (ranges.edge.contains(sel_id)) {
      r_elem = sel_id - ranges.edge.start();
      r_elem_type = SCE_SELECT_EDGE;
      r_base_index = select_ctx->objects.first_index_of_try(ob);
      return r_base_index != -1;
    }
    if (ranges.vert.contains(sel_id)) {
      r_elem = sel_id - ranges.vert.start();
      r_elem_type = SCE_SELECT_VERTEX;
      r_base_index = select_ctx->objects.first_index_of_try(ob);
      return r_base_index != -1;
    }
  }
  return false;
}

uint DRW_select_buffer_context_offset_for_object_elem(Depsgraph *depsgraph,
                                                      Object *object,
                                                      char elem_type)
{
  SELECTID_Context *select_ctx = DRW_select_engine_context_get();

  Object *ob_eval = DEG_get_evaluated(depsgraph, object);

  const ElemIndexRanges base_ofs = select_ctx->elem_ranges.lookup_default(ob_eval,
                                                                          ElemIndexRanges{});

  if (elem_type == SCE_SELECT_VERTEX) {
    return base_ofs.vert.start();
  }
  if (elem_type == SCE_SELECT_EDGE) {
    return base_ofs.edge.start();
  }
  if (elem_type == SCE_SELECT_FACE) {
    return base_ofs.face.start();
  }
  BLI_assert(0);
  return 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Context
 * \{ */

void DRW_select_buffer_context_create(Depsgraph *depsgraph,
                                      const Span<Base *> bases,
                                      short select_mode)
{
  SELECTID_Context *select_ctx = DRW_select_engine_context_get();

  select_ctx->objects.reinitialize(bases.size());

  for (const int i : bases.index_range()) {
    Object *obj = bases[i]->object;
    select_ctx->objects[i] = DEG_get_evaluated(depsgraph, obj);
  }

  select_ctx->select_mode = select_mode;
  select_ctx->persmat = float4x4::zero();
}

/** \} */

}  // namespace blender
