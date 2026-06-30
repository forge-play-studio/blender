/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstdio>
#include <cstring>
#include <vector>

#include "GPU_vertex_format.hh"

#include "webgpu_batch.hh"
#include "webgpu_context.hh"
#include "webgpu_framebuffer.hh"
#include "webgpu_index_buffer.hh"
#include "webgpu_shader.hh"
#include "webgpu_shader_interface.hh"
#include "webgpu_texture.hh"
#include "webgpu_vertex_buffer.hh"

namespace blender::gpu {

static WGPUPrimitiveTopology to_wgpu_topology(GPUPrimType prim)
{
  switch (prim) {
    case GPU_PRIM_POINTS:
      return WGPUPrimitiveTopology_PointList;
    case GPU_PRIM_LINES:
      return WGPUPrimitiveTopology_LineList;
    case GPU_PRIM_LINE_STRIP:
      return WGPUPrimitiveTopology_LineStrip;
    case GPU_PRIM_TRI_STRIP:
      return WGPUPrimitiveTopology_TriangleStrip;
    case GPU_PRIM_TRIS:
    case GPU_PRIM_TRI_FAN: /* No fan in WebGPU; treated as list (approximate). */
    default:
      return WGPUPrimitiveTopology_TriangleList;
  }
}

static bool is_strip(GPUPrimType prim)
{
  return prim == GPU_PRIM_LINE_STRIP || prim == GPU_PRIM_TRI_STRIP;
}

/* Map a Blender vertex attribute (component type/count/fetch) to a WGPU vertex
 * format. Covers the cases EEVEE/overlay batches actually use; unknowns fall
 * back to Float32x4 so pipeline creation does not hard-fail. */
static WGPUVertexFormat to_wgpu_vertex_format(GPUVertCompType comp,
                                              int len,
                                              GPUVertFetchMode fetch)
{
  const bool norm = (fetch == GPU_FETCH_INT_TO_FLOAT_UNIT);
  switch (comp) {
    case GPU_COMP_F32:
      return len == 1 ? WGPUVertexFormat_Float32 :
             len == 2 ? WGPUVertexFormat_Float32x2 :
             len == 3 ? WGPUVertexFormat_Float32x3 :
                        WGPUVertexFormat_Float32x4;
    case GPU_COMP_I32:
      return len == 1 ? WGPUVertexFormat_Sint32 :
             len == 2 ? WGPUVertexFormat_Sint32x2 :
             len == 3 ? WGPUVertexFormat_Sint32x3 :
                        WGPUVertexFormat_Sint32x4;
    case GPU_COMP_U32:
      return len == 1 ? WGPUVertexFormat_Uint32 :
             len == 2 ? WGPUVertexFormat_Uint32x2 :
             len == 3 ? WGPUVertexFormat_Uint32x3 :
                        WGPUVertexFormat_Uint32x4;
    case GPU_COMP_I16:
      if (len <= 2) {
        return norm ? WGPUVertexFormat_Snorm16x2 : WGPUVertexFormat_Sint16x2;
      }
      return norm ? WGPUVertexFormat_Snorm16x4 : WGPUVertexFormat_Sint16x4;
    case GPU_COMP_U16:
      if (len <= 2) {
        return norm ? WGPUVertexFormat_Unorm16x2 : WGPUVertexFormat_Uint16x2;
      }
      return norm ? WGPUVertexFormat_Unorm16x4 : WGPUVertexFormat_Uint16x4;
    case GPU_COMP_I8:
      if (len <= 2) {
        return norm ? WGPUVertexFormat_Snorm8x2 : WGPUVertexFormat_Sint8x2;
      }
      return norm ? WGPUVertexFormat_Snorm8x4 : WGPUVertexFormat_Sint8x4;
    case GPU_COMP_U8:
      if (len <= 2) {
        return norm ? WGPUVertexFormat_Unorm8x2 : WGPUVertexFormat_Uint8x2;
      }
      return norm ? WGPUVertexFormat_Unorm8x4 : WGPUVertexFormat_Uint8x4;
    default:
      return WGPUVertexFormat_Float32x4;
  }
}

static void hash_append(uint64_t &h, const void *data, size_t len)
{
  const uint8_t *p = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; i++) {
    h = (h ^ p[i]) * 1099511628211ull;
  }
}

void WebGPUBatch::draw(int vertex_first, int vertex_count, int instance_first, int instance_count)
{
  WebGPUContext *ctx = WebGPUContext::get();
  static int s_entry_log = 0;
  if (s_entry_log < 12) {
    WebGPUShader *es = ctx ? static_cast<WebGPUShader *>(ctx->shader) : nullptr;
    fprintf(stderr,
            "WGPU_BATCH::draw entry #%d ctx=%p dev=%p shader=%p valid=%d fb=%p\n",
            s_entry_log++,
            (void *)ctx,
            ctx ? (void *)ctx->device() : nullptr,
            (void *)es,
            es ? int(es->is_valid()) : -1,
            ctx ? (void *)ctx->active_fb : nullptr);
    fflush(stderr);
  }
  if (ctx == nullptr || ctx->device() == nullptr) {
    return;
  }
  WebGPUShader *shader = static_cast<WebGPUShader *>(ctx->shader);
  if (shader == nullptr || !shader->is_valid() || shader->vertex_module() == nullptr ||
      shader->fragment_module() == nullptr || shader->interface == nullptr)
  {
    return;
  }
  WebGPUFrameBuffer *fb = static_cast<WebGPUFrameBuffer *>(ctx->active_fb);
  if (fb == nullptr) {
    return;
  }
  WebGPUShaderInterface *iface = static_cast<WebGPUShaderInterface *>(shader->interface);

  if (instance_count == 0) {
    instance_count = 1;
  }

  /* --- Vertex buffer layouts from the batch's vertex formats. --- */
  std::vector<WGPUVertexBufferLayout> vb_layouts;
  std::vector<std::vector<WGPUVertexAttribute>> vb_attrs; /* stable storage */
  std::vector<WGPUBuffer> vb_buffers;
  for (int vi = 0; vi < GPU_BATCH_VBO_MAX_LEN; vi++) {
    VertBuf *vbo = verts_(vi);
    if (vbo == nullptr) {
      continue;
    }
    WGPUBuffer buf = static_cast<WebGPUVertexBuffer *>(vbo)->wgpu_buffer();
    if (buf == nullptr) {
      continue;
    }
    const GPUVertFormat &fmt = vbo->format;
    std::vector<WGPUVertexAttribute> attrs;
    for (uint ai = 0; ai < fmt.attr_len; ai++) {
      const GPUVertAttr &attr = fmt.attrs[ai];
      /* Match the first attribute name against the shader interface. */
      const char *name = GPU_vertformat_attr_name_get(&fmt, &attr, 0);
      const ShaderInput *in = iface->attr_get(name);
      if (in == nullptr || in->location < 0) {
        continue;
      }
      WGPUVertexAttribute wa = {};
      wa.format = to_wgpu_vertex_format(
          attr.type.comp_type(), attr.type.comp_len(), attr.type.fetch_mode());
      wa.offset = attr.offset;
      wa.shaderLocation = uint32_t(in->location);
      attrs.push_back(wa);
    }
    if (attrs.empty()) {
      continue;
    }
    vb_attrs.push_back(std::move(attrs));
    vb_buffers.push_back(buf);
    WGPUVertexBufferLayout layout = {};
    layout.arrayStride = fmt.stride;
    layout.stepMode = WGPUVertexStepMode_Vertex;
    layout.attributeCount = vb_attrs.back().size();
    layout.attributes = vb_attrs.back().data();
    vb_layouts.push_back(layout);
  }

  /* --- Pipeline cache key. --- */
  const int color_count = fb->color_attachment_count();
  WGPUTextureFormat color_fmt[GPU_FB_MAX_COLOR_ATTACHMENT] = {};
  for (int i = 0; i < color_count; i++) {
    color_fmt[i] = fb->color_format(i);
  }
  const WGPUTextureFormat depth_fmt = fb->depth_format();
  const WGPUPrimitiveTopology topo = to_wgpu_topology(prim_type);

  uint64_t key = 1469598103934665603ull;
  void *sh_ptr = shader;
  hash_append(key, &sh_ptr, sizeof(sh_ptr));
  hash_append(key, &topo, sizeof(topo));
  hash_append(key, &color_count, sizeof(color_count));
  hash_append(key, color_fmt, sizeof(WGPUTextureFormat) * color_count);
  hash_append(key, &depth_fmt, sizeof(depth_fmt));
  for (const WGPUVertexBufferLayout &l : vb_layouts) {
    hash_append(key, &l.arrayStride, sizeof(l.arrayStride));
    for (size_t a = 0; a < l.attributeCount; a++) {
      hash_append(key, &l.attributes[a], sizeof(WGPUVertexAttribute));
    }
  }

  WGPURenderPipeline pipeline = ctx->render_pipeline_get(key);
  if (pipeline == nullptr) {
    WGPUColorTargetState targets[GPU_FB_MAX_COLOR_ATTACHMENT] = {};
    for (int i = 0; i < color_count; i++) {
      targets[i].format = color_fmt[i];
      targets[i].writeMask = WGPUColorWriteMask_All;
    }
    WGPUFragmentState frag = {};
    frag.module = shader->fragment_module();
    frag.entryPoint = {"main", WGPU_STRLEN};
    frag.targetCount = uint32_t(color_count);
    frag.targets = color_count ? targets : nullptr;

    WGPUDepthStencilState ds = {};
    bool has_depth = depth_fmt != WGPUTextureFormat_Undefined;
    if (has_depth) {
      ds.format = depth_fmt;
      ds.depthWriteEnabled = WGPUOptionalBool_True;
      ds.depthCompare = WGPUCompareFunction_LessEqual;
    }

    /* Explicit pipeline layout from the parsed WGSL bindings so the layout has
     * exactly those bindings (auto-layout would strip declared-but-unused ones,
     * mismatching our complete bind group). */
    WGPUBindGroupLayout bgl_explicit = ctx->make_bind_group_layout(shader->render_bindings(),
                                                                   false);
    WGPUPipelineLayout pipe_layout = ctx->make_pipeline_layout(bgl_explicit);

    WGPURenderPipelineDescriptor rpd = {};
    rpd.layout = pipe_layout;
    rpd.vertex.module = shader->vertex_module();
    rpd.vertex.entryPoint = {"main", WGPU_STRLEN};
    rpd.vertex.bufferCount = uint32_t(vb_layouts.size());
    rpd.vertex.buffers = vb_layouts.empty() ? nullptr : vb_layouts.data();
    rpd.primitive.topology = topo;
    if (is_strip(prim_type) && elem_() != nullptr) {
      rpd.primitive.stripIndexFormat =
          static_cast<WebGPUIndexBuf *>(elem_())->wgpu_index_format();
    }
    rpd.multisample.count = 1;
    rpd.multisample.mask = 0xFFFFFFFFu;
    rpd.fragment = &frag;
    rpd.depthStencil = has_depth ? &ds : nullptr;

    pipeline = wgpuDeviceCreateRenderPipeline(ctx->device(), &rpd);
    if (pipe_layout) {
      wgpuPipelineLayoutRelease(pipe_layout);
    }
    if (bgl_explicit) {
      wgpuBindGroupLayoutRelease(bgl_explicit);
    }
    if (pipeline == nullptr) {
      fprintf(stderr, "WGPU_BATCH pipeline creation failed for '%s'\n",
              shader->name_get().c_str());
      return;
    }
    ctx->render_pipeline_put(key, pipeline);
    fprintf(stderr,
            "WGPU_BATCH pipeline created shader='%s' colors=%d depth=%d vbos=%zu\n",
            shader->name_get().c_str(),
            color_count,
            int(depth_fmt != WGPUTextureFormat_Undefined),
            vb_layouts.size());
    fflush(stderr);
  }

  /* --- Begin render pass + record draw. --- */
  ctx->render_pass_ensure(*fb);
  WGPURenderPassEncoder pass = ctx->render_pass();
  if (pass == nullptr) {
    return;
  }

  WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
  WGPUBindGroup bg = ctx->build_bind_group(shader, shader->render_bindings(), bgl);

  if (strstr(shader->name_get().c_str(), "World") != nullptr) {
    static int s_wlog = 0;
    if (s_wlog++ < 3) {
      fprintf(stderr,
              "WGPU_WORLDDRAW shader='%s' bg=%p bindings=%zu verts=%d fb='%s'\n",
              shader->name_get().c_str(),
              (void *)bg,
              shader->render_bindings().size(),
              vertex_count,
              fb->name_get());
      fflush(stderr);
    }
  }

  static int s_draw_log = 0;
  if (s_draw_log < 8) {
    fprintf(stderr,
            "WGPU_BATCH draw #%d shader='%s' verts=%d inst=%d indexed=%d bg=%p\n",
            s_draw_log++,
            shader->name_get().c_str(),
            vertex_count,
            instance_count,
            int(elem_() != nullptr),
            (void *)bg);
    fflush(stderr);
  }

  wgpuRenderPassEncoderSetPipeline(pass, pipeline);
  if (bg) {
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
  }
  for (size_t i = 0; i < vb_buffers.size(); i++) {
    wgpuRenderPassEncoderSetVertexBuffer(pass, uint32_t(i), vb_buffers[i], 0, WGPU_WHOLE_SIZE);
  }

  if (elem_() != nullptr) {
    WebGPUIndexBuf *ibo = static_cast<WebGPUIndexBuf *>(elem_());
    WGPUBuffer ibuf = ibo->wgpu_buffer();
    if (ibuf) {
      wgpuRenderPassEncoderSetIndexBuffer(
          pass, ibuf, ibo->wgpu_index_format(), 0, WGPU_WHOLE_SIZE);
      const uint32_t index_count = (vertex_count > 0) ? uint32_t(vertex_count) :
                                                        ibo->index_len_get();
      wgpuRenderPassEncoderDrawIndexed(
          pass, index_count, uint32_t(instance_count), 0, 0, uint32_t(instance_first));
    }
  }
  else {
    uint32_t v_count = (vertex_count > 0) ? uint32_t(vertex_count) :
                                            (verts_(0) ? verts_(0)->vertex_len : 0);
    if (v_count > 0) {
      wgpuRenderPassEncoderDraw(
          pass, v_count, uint32_t(instance_count), uint32_t(vertex_first), uint32_t(instance_first));
    }
  }

  /* Debug capture: remember the color target of a real content draw so it can be
   * read back from JS after the render. EEVEE's final "combined" output is
   * assembled by COMPUTE film passes (which can't run on SwiftShader), so target
   * the world/background draw (WOWorld_world_world) — actual rendered color — or
   * any non-clear draw, whichever we can read. Only on a valid bind group. */
  if (bg) {
    const char *shname = shader->name_get().c_str();
    const char *fbname = fb->name_get();
    /* Capture any content draw (procedural or geometry) to a main-resolution
     * color attachment (width >= 32 excludes tiny probe/tile targets); last wins. */
    gpu::Texture *ct = fb->color_tex(0);
    const uint32_t v = (vertex_count > 0) ? uint32_t(vertex_count)
                                          : (verts_(0) ? verts_(0)->vertex_len : 0);
    if (ct && ct->width_get() >= 32 && v > 0) {
      ctx->set_capture_target(static_cast<WebGPUTexture *>(ct));
      static int s_cap_log = 0;
      if (s_cap_log < 16) {
        fprintf(stderr,
                "WGPU_CAPTURE target fb='%s' shader='%s' %dx%d verts=%u\n",
                fbname ? fbname : "?",
                shname ? shname : "?",
                ct->width_get(),
                ct->height_get(),
                v);
        fflush(stderr);
        s_cap_log++;
      }
    }
  }

  if (bg) {
    wgpuBindGroupRelease(bg);
  }
  wgpuBindGroupLayoutRelease(bgl);
}

}  // namespace blender::gpu
