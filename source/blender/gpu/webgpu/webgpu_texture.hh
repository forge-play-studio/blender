/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU texture. Creates a real WGPUTexture (+ default view) when a device is
 * present; stays a metadata-only object in the device-less headless/diagnostic
 * mode (node without a browser device). Format/usage/dimension are mapped from
 * Blender's TextureFormat / eGPUTextureUsage / GPUTextureType.
 */

#pragma once

#include "gpu_texture_private.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUTexture : public Texture {
 private:
  WGPUTexture texture_ = nullptr;
  WGPUTextureView view_ = nullptr;
  WGPUTextureFormat wgpu_format_ = WGPUTextureFormat_RGBA8Unorm;

 public:
  WebGPUTexture(const char *name);
  ~WebGPUTexture() override;

  void update_sub(int mip,
                  int offset[3],
                  int extent[3],
                  eGPUDataFormat format,
                  const void *data,
                  uint unpack_row_length = 0) override;
  void update_sub(int /*offset*/[3],
                  int /*extent*/[3],
                  eGPUDataFormat /*format*/,
                  GPUPixelBuffer * /*pixbuf*/) override
  {
  }

  void generate_mipmap() override {}
  void copy_to(Texture *dst, IndexRange mip_levels) override;
  void clear(const double4 data) override;
  void swizzle_set(const char /*swizzle_mask*/[4]) override {}
  void mip_range_set(int /*min*/, int /*max*/) override {}
  void read(int mip, eGPUDataFormat format, void *dst) override;

  WGPUTexture wgpu_texture() const
  {
    return texture_;
  }
  WGPUTextureView wgpu_view() const
  {
    return view_;
  }

 protected:
  bool init_internal() override;
  bool init_internal(VertBuf *vbo) override;
  bool init_internal(gpu::Texture *src,
                     int mip_offset,
                     int layer_offset,
                     bool use_stencil) override;
};

WGPUTextureFormat webgpu_texture_format(TextureFormat format);
uint32_t webgpu_format_bytes_per_pixel(WGPUTextureFormat f);

}  // namespace blender::gpu
