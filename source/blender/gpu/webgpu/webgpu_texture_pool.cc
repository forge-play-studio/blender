/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cmath>

#include "gpu_backend.hh"

#include "webgpu_texture.hh"
#include "webgpu_texture_pool.hh"

namespace blender::gpu {

WebGPUTexturePool::~WebGPUTexturePool()
{
  for (FreeTexture &ft : pool_) {
    delete ft.texture;
  }
  pool_.clear();
  for (auto item : acquired_.keys()) {
    delete item;
  }
  acquired_.clear();
}

Texture *WebGPUTexturePool::acquire_texture_impl(int3 extent,
                                                 int mip_len,
                                                 GPUTextureType type,
                                                 TextureFormat format,
                                                 eGPUTextureUsage usage,
                                                 const char *name)
{
  /* Clamp mip count to the maximum supported by the largest dimension. */
  const int max_dim = std::max({extent.x, extent.y, extent.z});
  const int mip_len_max = 1 + int(std::floor(std::log2(float(std::max(max_dim, 1)))));
  mip_len = std::min(mip_len, mip_len_max);

  /* Reuse a compatible free texture. */
  for (int64_t i : pool_.index_range()) {
    Texture *t = pool_[i].texture;
    if (t->format_get() == format && t->width_get() == extent.x &&
        t->height_get() == extent.y && t->depth_get() == extent.z &&
        t->mip_count() == mip_len)
    {
      pool_.remove_and_reorder(i);
      t->usage_set(usage | GPU_TEXTURE_USAGE_FORMAT_VIEW);
      acquired_.add(t, 1);
      return t;
    }
  }

  /* Allocate a fresh one. */
  Texture *texture = GPUBackend::get()->texture_alloc(name ? name : "TexFromPool");
  texture->usage_set(usage | GPU_TEXTURE_USAGE_FORMAT_VIEW);
  switch (type) {
    case GPU_TEXTURE_1D:
    case GPU_TEXTURE_1D_ARRAY:
      texture->init_1D(extent.x, extent.y, mip_len, format);
      break;
    case GPU_TEXTURE_2D:
    case GPU_TEXTURE_2D_ARRAY:
      texture->init_2D(extent.x, extent.y, extent.z, mip_len, format);
      break;
    case GPU_TEXTURE_3D:
      texture->init_3D(extent.x, extent.y, extent.z, mip_len, format);
      break;
    case GPU_TEXTURE_CUBE:
    case GPU_TEXTURE_CUBE_ARRAY:
      texture->init_cubemap(extent.x, extent.y, mip_len, format);
      break;
    default:
      break;
  }
  acquired_.add(texture, 1);
  return texture;
}

void WebGPUTexturePool::offset_users_count(Texture *tex, int offset)
{
  int *users = acquired_.lookup_ptr(tex);
  if (users == nullptr) {
    return;
  }
  *users += offset;
  if (*users <= 0) {
    acquired_.remove(tex);
    pool_.append({tex, 0});
  }
}

void WebGPUTexturePool::release_texture(Texture *tex)
{
  offset_users_count(tex, -1);
}

void WebGPUTexturePool::reset(bool force_free)
{
  for (int64_t i = pool_.size() - 1; i >= 0; i--) {
    pool_[i].unused_cycles++;
    if (force_free || pool_[i].unused_cycles > max_unused_cycles_) {
      delete pool_[i].texture;
      pool_.remove_and_reorder(i);
    }
  }
}

}  // namespace blender::gpu
