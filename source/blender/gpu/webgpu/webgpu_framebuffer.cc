/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstdio>

#include "webgpu_context.hh"
#include "webgpu_framebuffer.hh"
#include "webgpu_texture.hh"

namespace blender::gpu {

void WebGPUFrameBuffer::bind(bool enabled_srgb)
{
  static int s_bind_log = 0;
  if (s_bind_log < 12) {
    fprintf(stderr,
            "WGPU_FB::bind #%d '%s' colors=%d depth=%d\n",
            s_bind_log++,
            name_,
            color_attachment_count(),
            int(depth_tex() != nullptr));
    fflush(stderr);
  }
  enabled_srgb_ = enabled_srgb;
  Context *ctx = Context::get();
  if (ctx) {
    /* A change of framebuffer ends the current render pass; the next draw
     * re-opens one over this framebuffer's attachments. */
    WebGPUContext::get()->render_pass_end();
    ctx->active_fb = this;
  }
  /* Derive size from the first present attachment. */
  for (int i = 0; i < GPU_FB_MAX_ATTACHMENT; i++) {
    if (attachments_[i].tex) {
      gpu::Texture *t = attachments_[i].tex;
      size_set(t->width_get(), std::max(t->height_get(), 1));
      break;
    }
  }
  Shader::set_framebuffer_srgb_target(enabled_srgb);
}

bool WebGPUFrameBuffer::check(char /*err_out*/[256])
{
  return true;
}

void WebGPUFrameBuffer::clear(GPUFrameBufferBits buffers,
                              const double4 clear_color,
                              float clear_depth,
                              uint /*clear_stencil*/)
{
  if (buffers & GPU_COLOR_BIT) {
    for (int i = 0; i < GPU_FB_MAX_COLOR_ATTACHMENT; i++) {
      color_clear_pending_[i] = true;
      clear_color_[i] = clear_color;
    }
  }
  if (buffers & GPU_DEPTH_BIT) {
    depth_clear_pending_ = true;
    clear_depth_ = clear_depth;
  }
}

void WebGPUFrameBuffer::clear_multi(Span<double4> clear_cols)
{
  for (int i = 0; i < GPU_FB_MAX_COLOR_ATTACHMENT && i < clear_cols.size(); i++) {
    color_clear_pending_[i] = true;
    clear_color_[i] = clear_cols[i];
  }
}

void WebGPUFrameBuffer::clear_attachment(GPUAttachmentType type, const double4 clear_value)
{
  if (type >= GPU_FB_COLOR_ATTACHMENT0) {
    const int slot = type - GPU_FB_COLOR_ATTACHMENT0;
    if (slot < GPU_FB_MAX_COLOR_ATTACHMENT) {
      color_clear_pending_[slot] = true;
      clear_color_[slot] = clear_value;
    }
  }
  else {
    depth_clear_pending_ = true;
    clear_depth_ = float(clear_value.x);
  }
}

void WebGPUFrameBuffer::attachment_set_loadstore_op(GPUAttachmentType /*type*/,
                                                    GPULoadStore /*ls*/)
{
}

void WebGPUFrameBuffer::subpass_transition_impl(
    const GPUAttachmentState /*depth_attachment_state*/,
    Span<GPUAttachmentState> /*color_attachment_states*/)
{
}

int WebGPUFrameBuffer::color_attachment_count() const
{
  int count = 0;
  for (int i = 0; i < GPU_FB_MAX_COLOR_ATTACHMENT; i++) {
    if (attachments_[GPU_FB_COLOR_ATTACHMENT0 + i].tex) {
      count = i + 1;
    }
  }
  return count;
}

WGPUTextureFormat WebGPUFrameBuffer::color_format(int slot) const
{
  gpu::Texture *t = color_tex(slot);
  return t ? webgpu_texture_format(static_cast<WebGPUTexture *>(t)->format_get()) :
             WGPUTextureFormat_Undefined;
}

WGPUTextureFormat WebGPUFrameBuffer::depth_format() const
{
  gpu::Texture *t = depth_tex();
  return t ? webgpu_texture_format(static_cast<WebGPUTexture *>(t)->format_get()) :
             WGPUTextureFormat_Undefined;
}

WGPURenderPassEncoder WebGPUFrameBuffer::begin_render_pass(WGPUCommandEncoder enc)
{
  if (enc == nullptr) {
    return nullptr;
  }
  WGPURenderPassColorAttachment color_att[GPU_FB_MAX_COLOR_ATTACHMENT] = {};
  int color_count = 0;
  for (int i = 0; i < GPU_FB_MAX_COLOR_ATTACHMENT; i++) {
    gpu::Texture *t = color_tex(i);
    if (!t) {
      continue;
    }
    WGPUTextureView view = static_cast<WebGPUTexture *>(t)->wgpu_view();
    if (!view) {
      continue;
    }
    WGPURenderPassColorAttachment &ca = color_att[color_count++];
    ca.view = view;
    ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
    ca.loadOp = color_clear_pending_[i] ? WGPULoadOp_Clear : WGPULoadOp_Load;
    ca.storeOp = WGPUStoreOp_Store;
    ca.clearValue = {clear_color_[i].x, clear_color_[i].y, clear_color_[i].z, clear_color_[i].w};
    color_clear_pending_[i] = false;
  }

  WGPURenderPassDepthStencilAttachment depth_att = {};
  bool has_depth = false;
  if (gpu::Texture *dt = depth_tex()) {
    if (WGPUTextureView dview = static_cast<WebGPUTexture *>(dt)->wgpu_view()) {
      depth_att.view = dview;
      depth_att.depthLoadOp = depth_clear_pending_ ? WGPULoadOp_Clear : WGPULoadOp_Load;
      depth_att.depthStoreOp = WGPUStoreOp_Store;
      depth_att.depthClearValue = clear_depth_;
      has_depth = true;
      depth_clear_pending_ = false;
    }
  }

  if (color_count == 0 && !has_depth) {
    return nullptr;
  }

  WGPURenderPassDescriptor desc = {};
  desc.colorAttachmentCount = color_count;
  desc.colorAttachments = color_att;
  desc.depthStencilAttachment = has_depth ? &depth_att : nullptr;
  return wgpuCommandEncoderBeginRenderPass(enc, &desc);
}

void WebGPUFrameBuffer::read(GPUFrameBufferBits /*planes*/,
                             eGPUDataFormat /*format*/,
                             const int /*area*/[4],
                             int /*channel_len*/,
                             int /*slot*/,
                             void * /*r_data*/)
{
  /* GPU->host readback requires an async buffer map that cannot complete while
   * wasm holds the browser main thread (no JSPI). Left unimplemented until the
   * harness drives readback from the JS event loop after the render returns. */
}

void WebGPUFrameBuffer::blit_to(GPUFrameBufferBits /*planes*/,
                                int /*src_slot*/,
                                FrameBuffer * /*dst*/,
                                int /*dst_slot*/,
                                int /*dst_offset_x*/,
                                int /*dst_offset_y*/)
{
}

}  // namespace blender::gpu
