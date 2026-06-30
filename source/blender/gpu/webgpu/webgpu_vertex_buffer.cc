/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstring>

#include "webgpu_context.hh"
#include "webgpu_vertex_buffer.hh"

namespace blender::gpu {

WebGPUVertexBuffer::~WebGPUVertexBuffer()
{
  if (buffer_) {
    wgpuBufferRelease(buffer_);
  }
  MEM_SAFE_DELETE(data_);
}

void WebGPUVertexBuffer::acquire_data()
{
  if (usage_ == GPU_USAGE_DEVICE_ONLY) {
    return;
  }
  MEM_SAFE_DELETE(data_);
  data_ = MEM_new_array_uninitialized<uchar>(this->size_alloc_get(), __func__);
}

void WebGPUVertexBuffer::resize_data()
{
  if (usage_ == GPU_USAGE_DEVICE_ONLY) {
    return;
  }
  data_ = static_cast<uchar *>(
      MEM_realloc_uninitialized(data_, sizeof(uchar) * this->size_alloc_get()));
}

void WebGPUVertexBuffer::release_data()
{
  if (buffer_) {
    wgpuBufferRelease(buffer_);
    buffer_ = nullptr;
    buffer_size_ = 0;
  }
  MEM_SAFE_DELETE(data_);
}

void WebGPUVertexBuffer::ensure_buffer()
{
  const size_t needed = this->size_used_get();
  if (buffer_ && buffer_size_ >= needed) {
    return;
  }
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx == nullptr || ctx->device() == nullptr || needed == 0) {
    return;
  }
  if (buffer_) {
    wgpuBufferRelease(buffer_);
    buffer_ = nullptr;
  }
  /* WGPU requires buffer size be a multiple of 4. */
  buffer_size_ = (needed + 3) & ~size_t(3);
  WGPUBufferDescriptor desc = {};
  desc.size = buffer_size_;
  desc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage |
               WGPUBufferUsage_CopySrc;
  buffer_ = wgpuDeviceCreateBuffer(ctx->device(), &desc);
}

void WebGPUVertexBuffer::upload_data()
{
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue() && data_) {
    const size_t used = this->size_used_get();
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, 0, data_, (used + 3) & ~size_t(3));
  }
}

void WebGPUVertexBuffer::update_sub(uint start, uint len, const void *data)
{
  if (data_) {
    memcpy(data_ + start, data, len);
  }
  ensure_buffer();
  WebGPUContext *ctx = WebGPUContext::get();
  if (buffer_ && ctx && ctx->queue()) {
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, start, data, (len + 3) & ~size_t(3));
  }
}

void WebGPUVertexBuffer::bind_as_ssbo(uint binding)
{
  WebGPUContext *ctx = WebGPUContext::get();
  if (ctx) {
    ctx->bind_ssbo(int(binding), wgpu_buffer());
  }
}

void WebGPUVertexBuffer::read(void *data) const
{
  /* GPU->host readback needs an async map (no JSPI). Return the CPU shadow. */
  if (data_ && data) {
    memcpy(data, data_, this->size_used_get());
  }
}

}  // namespace blender::gpu
