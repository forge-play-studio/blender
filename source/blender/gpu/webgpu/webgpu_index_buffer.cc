/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstring>

#include "webgpu_context.hh"
#include "webgpu_index_buffer.hh"

namespace blender::gpu {

WebGPUIndexBuf::~WebGPUIndexBuf()
{
  if (buffer_) {
    wgpuBufferRelease(buffer_);
  }
}

void WebGPUIndexBuf::ensure_buffer()
{
  const size_t needed = this->size_get();
  if (buffer_ && buffer_size_ >= needed) {
    return;
  }
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx == nullptr || ctx->device() == nullptr || needed == 0 || data_ == nullptr) {
    return;
  }
  if (buffer_) {
    wgpuBufferRelease(buffer_);
    buffer_ = nullptr;
  }
  buffer_size_ = (needed + 3) & ~size_t(3);
  WGPUBufferDescriptor desc = {};
  desc.size = buffer_size_;
  desc.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage |
               WGPUBufferUsage_CopySrc;
  buffer_ = wgpuDeviceCreateBuffer(ctx->device(), &desc);
}

void WebGPUIndexBuf::upload_data()
{
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue() && data_) {
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, 0, data_, (this->size_get() + 3) & ~size_t(3));
  }
}

void WebGPUIndexBuf::bind_as_ssbo(uint binding)
{
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx) {
    ctx->bind_ssbo(int(binding), wgpu_buffer());
  }
}

void WebGPUIndexBuf::read(uint32_t *data) const
{
  if (data_ && data) {
    memcpy(data, data_, this->size_get());
  }
}

void WebGPUIndexBuf::update_sub(uint start, uint len, const void *data)
{
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue()) {
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, start, data, (len + 3) & ~size_t(3));
  }
}

}  // namespace blender::gpu
