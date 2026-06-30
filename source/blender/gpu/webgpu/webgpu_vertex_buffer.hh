/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU vertex buffer: a CPU shadow (data_) plus a real WGPUBuffer (VERTEX |
 * COPY_DST | STORAGE) created+uploaded on upload_data(). Device-less mode keeps
 * only the shadow.
 */

#pragma once

#include "BLI_sys_types.hh"

#include "MEM_guardedalloc.h"

#include "GPU_vertex_buffer.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUVertexBuffer : public VertBuf {
 private:
  WGPUBuffer buffer_ = nullptr;
  size_t buffer_size_ = 0;

 public:
  ~WebGPUVertexBuffer() override;

  void bind_as_ssbo(uint binding) override;
  void bind_as_texture(uint /*binding*/) override {}
  void wrap_handle(uint64_t /*handle*/) override {}
  void update_sub(uint start, uint len, const void *data) override;
  void read(void *data) const override;

  WGPUBuffer wgpu_buffer()
  {
    ensure_buffer();
    return buffer_;
  }

 protected:
  void acquire_data() override;
  void resize_data() override;
  void release_data() override;
  void upload_data() override;

 private:
  void ensure_buffer();
};

}  // namespace blender::gpu
