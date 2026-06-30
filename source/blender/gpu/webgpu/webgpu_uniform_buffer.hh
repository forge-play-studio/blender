/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU uniform buffer (UBO). Like the storage buffer: CPU shadow + a real
 * WGPUBuffer (Uniform usage) created lazily once a device exists. Device-less
 * headless mode keeps only the shadow.
 */

#pragma once

#include "gpu_uniform_buffer_private.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUUniformBuf : public UniformBuf {
 private:
  WGPUBuffer buffer_ = nullptr;
  size_t alloc_size_ = 0;
  int slot_ = -1;

  void ensure_buffer();

 public:
  WebGPUUniformBuf(size_t size, const char *name);
  ~WebGPUUniformBuf() override;

  void update(const void *data) override;
  void clear_to_zero() override;
  void bind(int slot) override;
  void bind_as_ssbo(int slot) override;
  void unbind() override;

  WGPUBuffer buffer()
  {
    ensure_buffer();
    return buffer_;
  }
};

}  // namespace blender::gpu
