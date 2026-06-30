/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstring>

#include "MEM_guardedalloc.h"

#include "webgpu_context.hh"
#include "webgpu_storage_buffer.hh"

namespace blender::gpu {

static WebGPUContext *webgpu_context_get()
{
  return static_cast<WebGPUContext *>(Context::get());
}

WebGPUStorageBuf::WebGPUStorageBuf(size_t size, GPUUsageType usage, const char *name)
    : StorageBuf(size, name)
{
  usage_ = usage;
  /* WebGPU buffer sizes must be a multiple of 4 bytes. */
  alloc_size_ = (size + 3) & ~size_t(3);
}

WebGPUStorageBuf::~WebGPUStorageBuf()
{
  if (buffer_) {
    wgpuBufferRelease(buffer_);
    buffer_ = nullptr;
  }
  MEM_SAFE_DELETE_VOID(data_);
}

void WebGPUStorageBuf::ensure_buffer()
{
  if (data_ == nullptr) {
    data_ = MEM_new_zeroed(alloc_size_, "WebGPUStorageBuf");
  }
  if (buffer_ != nullptr) {
    return;
  }
  WebGPUContext *ctx = webgpu_context_get();
  if (ctx == nullptr || ctx->device() == nullptr) {
    /* No device yet (e.g. headless node). Stay CPU-only; created lazily later. */
    return;
  }
  WGPUBufferDescriptor desc = {};
  desc.size = alloc_size_;
  desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
               WGPUBufferUsage_Indirect | WGPUBufferUsage_Vertex;
  desc.mappedAtCreation = false;
  buffer_ = wgpuDeviceCreateBuffer(ctx->device(), &desc);
  /* Upload whatever the CPU shadow currently holds. */
  if (buffer_) {
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, 0, data_, alloc_size_);
  }
}

void WebGPUStorageBuf::update(const void *data)
{
  fprintf(stderr, "WGPU_SSBO update this=%p size=%zu alloc=%zu data=%p\n",
          (void *)this, size_in_bytes_, alloc_size_, data);
  fflush(stderr);
  if (size_in_bytes_ > (size_t(1) << 31)) {
    fprintf(stderr, "WGPU_SSBO BAD size_in_bytes_=%zu (skipping update)\n", size_in_bytes_);
    fflush(stderr);
    return;
  }
  ensure_buffer();
  fprintf(stderr, "WGPU_SSBO post-ensure data_=%p\n", data_); fflush(stderr);
  if (data && data_) {
    memcpy(data_, data, size_in_bytes_);
  }
  fprintf(stderr, "WGPU_SSBO post-memcpy ok\n"); fflush(stderr);
  WebGPUContext *ctx = webgpu_context_get();
  if (buffer_ && ctx && ctx->queue()) {
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, 0, data_, alloc_size_);
  }
}

void WebGPUStorageBuf::bind(int slot)
{
  slot_ = slot;
  ensure_buffer();
  /* Record into the context binding table; the bind group is assembled at
   * draw/dispatch time from these + the pipeline's auto layout. */
  WebGPUContext *ctx = webgpu_context_get();
  if (ctx) {
    ctx->bind_ssbo(slot, buffer_);
  }
}

void WebGPUStorageBuf::unbind()
{
  WebGPUContext *ctx = webgpu_context_get();
  if (ctx && slot_ >= 0) {
    ctx->unbind_ssbo(slot_);
  }
  slot_ = -1;
}

void WebGPUStorageBuf::clear(uint32_t clear_value)
{
  ensure_buffer();
  if (data_) {
    /* Fill the CPU shadow with the repeated 32-bit clear value. */
    uint32_t *words = static_cast<uint32_t *>(data_);
    const size_t word_count = alloc_size_ / sizeof(uint32_t);
    for (size_t i = 0; i < word_count; i++) {
      words[i] = clear_value;
    }
  }
  WebGPUContext *ctx = webgpu_context_get();
  if (buffer_ && ctx && ctx->queue()) {
    wgpuQueueWriteBuffer(ctx->queue(), buffer_, 0, data_, alloc_size_);
  }
}

void WebGPUStorageBuf::copy_sub(VertBuf * /*src*/,
                                uint /*dst_offset*/,
                                uint /*src_offset*/,
                                uint /*copy_size*/)
{
  /* TODO: GPU-side buffer-to-buffer copy via a command encoder. Requires the
   * VertBuf WebGPU buffer handle. */
}

void WebGPUStorageBuf::read(void *data)
{
  /* CPU shadow holds the last host-written contents. True GPU readback needs an
   * async buffer map (wgpuBufferMapAsync) routed through the queue; deferred
   * until compute results need to round-trip to the host. */
  if (data && data_) {
    memcpy(data, data_, size_in_bytes_);
  }
}

void WebGPUStorageBuf::async_flush_to_host()
{
  /* No-op until async readback is wired (see read()). */
}

void WebGPUStorageBuf::sync_as_indirect_buffer()
{
  ensure_buffer();
  /* The buffer is already created with WGPUBufferUsage_Indirect. */
}

}  // namespace blender::gpu
