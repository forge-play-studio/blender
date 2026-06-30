/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU index buffer: CPU shadow (data_, owned by base) + a WGPUBuffer
 * (Index | CopyDst | Storage) uploaded on demand.
 */

#pragma once

#include "GPU_index_buffer.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUIndexBuf : public IndexBuf {
 private:
  WGPUBuffer buffer_ = nullptr;
  size_t buffer_size_ = 0;

 public:
  ~WebGPUIndexBuf() override;

  void upload_data() override;
  void bind_as_ssbo(uint binding) override;
  void read(uint32_t *data) const override;
  void update_sub(uint start, uint len, const void *data) override;

  WGPUBuffer wgpu_buffer()
  {
    ensure_buffer();
    return buffer_;
  }
  WGPUIndexFormat wgpu_index_format() const
  {
    return is_32bit() ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16;
  }

 private:
  void ensure_buffer();
  void strip_restart_indices() override {}
};

}  // namespace blender::gpu
