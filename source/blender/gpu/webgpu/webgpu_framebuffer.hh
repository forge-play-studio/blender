/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU framebuffer. Wraps the base FrameBuffer's color/depth attachments and
 * begins a WGPURenderPass over them. Clears are deferred into the render pass's
 * load ops (WebGPU has no mid-pass clear). Pixel readback (read()) needs an async
 * buffer map, which cannot complete while wasm holds the browser main thread, so
 * it records the copy but the data is only valid once control returns to JS.
 */

#pragma once

#include "BLI_math_vector_types.hh"

#include "gpu_framebuffer_private.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUFrameBuffer : public FrameBuffer {
 private:
  bool enabled_srgb_ = false;

  /* Deferred clear state, consumed by the next begin_render_pass(). */
  bool color_clear_pending_[GPU_FB_MAX_COLOR_ATTACHMENT] = {};
  double4 clear_color_[GPU_FB_MAX_COLOR_ATTACHMENT] = {};
  bool depth_clear_pending_ = false;
  float clear_depth_ = 1.0f;

 public:
  WebGPUFrameBuffer(const char *name) : FrameBuffer(name) {}

  void bind(bool enabled_srgb) override;
  bool check(char err_out[256]) override;
  void clear(GPUFrameBufferBits buffers,
             const double4 clear_color,
             float clear_depth,
             uint clear_stencil) override;
  void clear_multi(Span<double4> clear_cols) override;
  void clear_attachment(GPUAttachmentType type, const double4 clear_value) override;
  void attachment_set_loadstore_op(GPUAttachmentType type, GPULoadStore ls) override;
  void read(GPUFrameBufferBits planes,
            eGPUDataFormat format,
            const int area[4],
            int channel_len,
            int slot,
            void *r_data) override;
  void blit_to(GPUFrameBufferBits planes,
               int src_slot,
               FrameBuffer *dst,
               int dst_slot,
               int dst_offset_x,
               int dst_offset_y) override;

  /* Begin a render pass over this framebuffer's attachments on `enc`. Consumes
   * pending clears (Clear load-op) else Load. Returns null if no usable
   * attachments / no device. */
  WGPURenderPassEncoder begin_render_pass(WGPUCommandEncoder enc);

  /* Pipeline-state queries (color/depth formats of the bound attachments). */
  int color_attachment_count() const;
  WGPUTextureFormat color_format(int slot) const;
  WGPUTextureFormat depth_format() const; /* Undefined if none. */

 protected:
  void subpass_transition_impl(const GPUAttachmentState depth_attachment_state,
                               Span<GPUAttachmentState> color_attachment_states) override;
};

}  // namespace blender::gpu
