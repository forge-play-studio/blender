/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstdlib>
#include <cstring>
#include <vector>

#include "webgpu_context.hh"
#include "webgpu_shader.hh"
#include "webgpu_shader_interface.hh"
#include "webgpu_state.hh"
#include "webgpu_texture.hh"

#ifdef __EMSCRIPTEN__
#  include <emscripten/emscripten.h>
/* Declared in webgpu/webgpu.h on Emscripten; returns the device created in JS
 * (Module.preinitializedWebGPUDevice). */
#endif

/* --- Debug pixel capture --------------------------------------------------
 * Copy a rendered color texture into a persistent (never-released) buffer during
 * the render, so JS can map+read it AFTER Blender quits (the wasm runtime stays
 * alive; synchronous readback is impossible mid-render with no JSPI). The C
 * exports below are driven by the harness after "Blender quit". */
namespace {
WGPUBuffer g_capture_buf = nullptr;
size_t g_capture_cap = 0;
uint32_t g_capture_w = 0, g_capture_h = 0, g_capture_bpr = 0, g_capture_bpp = 0;
uint8_t *g_capture_pixels = nullptr;
volatile int g_capture_ready = 0; /* 0 = pending, 1 = ok, -1 = fail */
}  // namespace

namespace blender::gpu {

static uint32_t align256(uint32_t v)
{
  return (v + 255u) & ~255u;
}

WebGPUContext::WebGPUContext(GHOST_IWindow * /*ghost_window*/,
                             GHOST_IContext * /*ghost_context*/)
{
#ifdef __EMSCRIPTEN__
  /* Acquire the JS-provided device ONCE and share it across all WebGPU contexts
   * (Blender creates more than one; emscripten_webgpu_get_device only hands the
   * device over on the first/right thread, and Module.preinitializedWebGPUDevice
   * isn't visible on every thread under PROXY_TO_PTHREAD). The WGPUDevice handle
   * lives in shared wasm memory, so reusing it across contexts/threads works.
   * Probe guards against the device-less headless case (plain node, no device). */
  static WGPUDevice s_shared_device = nullptr;
  if (s_shared_device == nullptr) {
    const int has_device = EM_ASM_INT({
      return (typeof Module !== "undefined" && Module["preinitializedWebGPUDevice"]) ? 1 : 0;
    });
    if (has_device) {
      s_shared_device = emscripten_webgpu_get_device();
    }
  }
  device_ = s_shared_device;
#endif
  if (device_) {
    queue_ = wgpuDeviceGetQueue(device_);
    fprintf(stderr, "WEBGPU_CONTEXT device=%p queue=%p (real device acquired)\n",
            (void *)device_, (void *)queue_);
  }
  else {
    fprintf(stderr, "WEBGPU_CONTEXT no device (degraded headless mode)\n");
  }
  fflush(stderr);
  /* TODO: a default framebuffer wrapping the canvas swapchain. For headless
   * render-to-texture (Cycles-style output, Workbench bring-up) this stays a
   * dummy until surface presentation is wired. */
  state_manager = new WebGPUStateManager();
  back_left = active_fb = new WebGPUFrameBuffer("WebGPUBackbuffer");
}

WebGPUContext::~WebGPUContext()
{
  render_pass_end();
  if (encoder_) {
    wgpuCommandEncoderRelease(encoder_);
  }
  if (default_sampler_) {
    wgpuSamplerRelease(default_sampler_);
  }
  if (capture_src_) {
    wgpuTextureRelease(capture_src_);
  }
  for (auto &kv : render_pipelines_) {
    if (kv.second) {
      wgpuRenderPipelineRelease(kv.second);
    }
  }
  for (auto &kv : compute_pipelines_) {
    if (kv.second) {
      wgpuComputePipelineRelease(kv.second);
    }
  }
  free_resources();
  delete state_manager;
  state_manager = nullptr;
  if (queue_) {
    wgpuQueueRelease(queue_);
  }
  /* device_ is the shared, JS-owned device (see ctor) — do NOT release it here;
   * other contexts share it and JS still holds the reference. */
  if (false && device_) {
    wgpuDeviceRelease(device_);
  }
}

void WebGPUContext::set_capture_target(WebGPUTexture *t)
{
  if (capture_done_ || capture_src_ != nullptr || t == nullptr) {
    return;
  }
  WGPUTexture h = t->wgpu_texture();
  if (h == nullptr) {
    return;
  }
  /* AddRef now (texture is alive at draw time) so the handle survives the pooled
   * gpu::Texture being released/reused before we record the copy. */
  wgpuTextureAddRef(h);
  capture_src_ = h;
  capture_w_ = uint32_t(std::max(t->width_get(), 1));
  capture_h_ = uint32_t(std::max(t->height_get(), 1));
  capture_fmt_ = webgpu_texture_format(t->format_get());
}

WGPUCommandEncoder WebGPUContext::ensure_encoder()
{
  if (encoder_ == nullptr && device_ != nullptr) {
    WGPUCommandEncoderDescriptor desc = {};
    encoder_ = wgpuDeviceCreateCommandEncoder(device_, &desc);
  }
  return encoder_;
}

void WebGPUContext::render_pass_ensure(WebGPUFrameBuffer &fb)
{
  if (render_pass_ != nullptr && render_pass_fb_ == &fb) {
    return;
  }
  render_pass_end();
  if (ensure_encoder() == nullptr) {
    return;
  }
  render_pass_ = fb.begin_render_pass(encoder_);
  render_pass_fb_ = render_pass_ ? &fb : nullptr;
}

void WebGPUContext::render_pass_end()
{
  if (render_pass_) {
    wgpuRenderPassEncoderEnd(render_pass_);
    wgpuRenderPassEncoderRelease(render_pass_);
    render_pass_ = nullptr;
    render_pass_fb_ = nullptr;

    /* Debug capture: copy the first content framebuffer's color into the
     * persistent readback buffer NOW, right after its render pass ends — the
     * texture is still alive (capture targets like the lightprobe "Capture.Combine"
     * are pooled and get freed/reused soon after, so a deferred copy at the final
     * submit would read garbage). Recorded into the same encoder, so it executes
     * after the just-ended pass's draws. */
    if (encoder_ && capture_src_ && !capture_done_ && device_) {
      const uint32_t w = capture_w_, h = capture_h_;
      const uint32_t bpp = webgpu_format_bytes_per_pixel(capture_fmt_);
      const uint32_t bpr = align256(w * bpp);
      const size_t need = size_t(bpr) * h;
      if (g_capture_buf == nullptr || g_capture_cap < need) {
        if (g_capture_buf) {
          wgpuBufferRelease(g_capture_buf);
        }
        WGPUBufferDescriptor bd = {};
        bd.size = need;
        bd.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        g_capture_buf = wgpuDeviceCreateBuffer(device_, &bd);
        g_capture_cap = need;
      }
      if (g_capture_buf) {
        WGPUTexelCopyTextureInfo src = {};
        src.texture = capture_src_;
        src.aspect = WGPUTextureAspect_All;
        WGPUTexelCopyBufferInfo dst = {};
        dst.buffer = g_capture_buf;
        dst.layout.bytesPerRow = bpr;
        dst.layout.rowsPerImage = h;
        WGPUExtent3D ext = {w, h, 1};
        wgpuCommandEncoderCopyTextureToBuffer(encoder_, &src, &dst, &ext);
        g_capture_w = w;
        g_capture_h = h;
        g_capture_bpr = bpr;
        g_capture_bpp = bpp;
        capture_done_ = true;
      }
    }
  }
}

void WebGPUContext::submit()
{
  render_pass_end();
  if (encoder_ == nullptr) {
    return;
  }
  WGPUCommandBufferDescriptor cb_desc = {};
  WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder_, &cb_desc);
  if (queue_ && cmd) {
    wgpuQueueSubmit(queue_, 1, &cmd);
  }
  if (cmd) {
    wgpuCommandBufferRelease(cmd);
  }
  wgpuCommandEncoderRelease(encoder_);
  encoder_ = nullptr;
}

WGPUSampler WebGPUContext::default_sampler()
{
  if (default_sampler_ == nullptr && device_ != nullptr) {
    WGPUSamplerDescriptor d = {};
    d.addressModeU = WGPUAddressMode_Repeat;
    d.addressModeV = WGPUAddressMode_Repeat;
    d.addressModeW = WGPUAddressMode_Repeat;
    d.magFilter = WGPUFilterMode_Linear;
    d.minFilter = WGPUFilterMode_Linear;
    d.mipmapFilter = WGPUMipmapFilterMode_Linear;
    d.maxAnisotropy = 1;
    d.lodMaxClamp = 32.0f;
    default_sampler_ = wgpuDeviceCreateSampler(device_, &d);
  }
  return default_sampler_;
}

WGPUBindGroupLayout WebGPUContext::make_bind_group_layout(
    const std::vector<WgslResourceBinding> &bindings, bool is_compute)
{
  if (device_ == nullptr) {
    return nullptr;
  }
  const WGPUShaderStage base_vis = is_compute ?
                                       WGPUShaderStage_Compute :
                                       WGPUShaderStage(WGPUShaderStage_Vertex |
                                                       WGPUShaderStage_Fragment);
  std::vector<WGPUBindGroupLayoutEntry> entries;
  for (const WgslResourceBinding &b : bindings) {
    WGPUBindGroupLayoutEntry e = {};
    e.binding = b.binding;
    /* Writable storage is illegal in the vertex stage. */
    e.visibility = (b.writable && !is_compute) ? WGPUShaderStage_Fragment : base_vis;
    switch (b.kind) {
      case WgslResourceBinding::UBO:
      case WgslResourceBinding::PUSH_CONST:
        e.buffer.type = WGPUBufferBindingType_Uniform;
        break;
      case WgslResourceBinding::SSBO:
        e.buffer.type = b.buffer_type;
        break;
      case WgslResourceBinding::TEXTURE:
        e.texture.sampleType = b.tex_sample;
        e.texture.viewDimension = b.view_dim;
        e.texture.multisampled = false;
        break;
      case WgslResourceBinding::SAMPLER:
        e.sampler.type = b.sampler_type;
        break;
      case WgslResourceBinding::STORAGE_TEXTURE:
        e.storageTexture.access = b.storage_access;
        e.storageTexture.format = b.storage_format;
        e.storageTexture.viewDimension = b.view_dim;
        break;
    }
    entries.push_back(e);
  }
  WGPUBindGroupLayoutDescriptor desc = {};
  desc.entryCount = entries.size();
  desc.entries = entries.empty() ? nullptr : entries.data();
  return wgpuDeviceCreateBindGroupLayout(device_, &desc);
}

WGPUPipelineLayout WebGPUContext::make_pipeline_layout(WGPUBindGroupLayout bgl)
{
  if (device_ == nullptr) {
    return nullptr;
  }
  WGPUPipelineLayoutDescriptor desc = {};
  desc.bindGroupLayoutCount = bgl ? 1 : 0;
  desc.bindGroupLayouts = bgl ? &bgl : nullptr;
  return wgpuDeviceCreatePipelineLayout(device_, &desc);
}

WGPUBindGroup WebGPUContext::build_bind_group(WebGPUShader *shader,
                                              const std::vector<WgslResourceBinding> &bindings,
                                              WGPUBindGroupLayout bgl)
{
  if (device_ == nullptr || shader == nullptr || shader->interface == nullptr) {
    return nullptr;
  }
  ShaderInterface *iface = shader->interface;
  std::vector<WGPUBindGroupEntry> entries;
  bool incomplete = false;

  const bool dbg = strstr(shader->name_get().c_str(), "World") != nullptr;
  static int s_dbg_n = 0;
  const bool dbg_log = dbg && (s_dbg_n++ < 3);

  for (const WgslResourceBinding &b : bindings) {
    WGPUBindGroupEntry e = {};
    e.binding = b.binding;
    switch (b.kind) {
      case WgslResourceBinding::PUSH_CONST: {
        e.buffer = shader->push_const_buffer();
        e.size = WGPU_WHOLE_SIZE;
        break;
      }
      case WgslResourceBinding::UBO: {
        const ShaderInput *in = iface->ubo_get(StringRefNull(b.res_name.c_str()));
        int slot = in ? in->binding : -1;
        e.buffer = (slot >= 0 && slot < WEBGPU_MAX_UBO) ? bound_ubo_[slot] : nullptr;
        e.size = WGPU_WHOLE_SIZE;
        break;
      }
      case WgslResourceBinding::SSBO: {
        const ShaderInput *in = iface->ssbo_get(StringRefNull(b.res_name.c_str()));
        int slot = in ? in->binding : -1;
        e.buffer = (slot >= 0 && slot < WEBGPU_MAX_SSBO) ? bound_ssbo_[slot] : nullptr;
        e.size = WGPU_WHOLE_SIZE;
        break;
      }
      case WgslResourceBinding::TEXTURE: {
        const ShaderInput *in = iface->uniform_get(StringRefNull(b.res_name.c_str()));
        int slot = in ? in->binding : -1;
        WebGPUTexture *t = (slot >= 0 && slot < WEBGPU_MAX_TEX) ? bound_tex_[slot] : nullptr;
        e.textureView = t ? t->wgpu_view() : nullptr;
        break;
      }
      case WgslResourceBinding::STORAGE_TEXTURE: {
        const ShaderInput *in = iface->uniform_get(StringRefNull(b.res_name.c_str()));
        int slot = in ? in->binding : -1;
        WebGPUTexture *t = (slot >= 0 && slot < WEBGPU_MAX_IMAGE) ? bound_image_[slot] : nullptr;
        e.textureView = t ? t->wgpu_view() : nullptr;
        break;
      }
      case WgslResourceBinding::SAMPLER: {
        e.sampler = default_sampler();
        break;
      }
    }
    /* A binding the layout requires but we cannot resolve makes the whole bind
     * group invalid; skip building it (that draw is then skipped). */
    const bool resolved = e.buffer || e.textureView || e.sampler;
    if (!resolved) {
      incomplete = true;
    }
    if (dbg_log) {
      fprintf(stderr,
              "WGPU_BG '%s' @%u kind=%d res='%s' -> buf=%p view=%p samp=%p\n",
              shader->name_get().c_str(),
              b.binding,
              int(b.kind),
              b.res_name.c_str(),
              (void *)e.buffer,
              (void *)e.textureView,
              (void *)e.sampler);
      fflush(stderr);
    }
    entries.push_back(e);
  }

  if (incomplete) {
    return nullptr;
  }

  WGPUBindGroupDescriptor desc = {};
  desc.layout = bgl;
  desc.entryCount = entries.size();
  desc.entries = entries.empty() ? nullptr : entries.data();
  return wgpuDeviceCreateBindGroup(device_, &desc);
}

void WebGPUContext::flush()
{
  submit();
}

void WebGPUContext::finish()
{
  /* No synchronous GPU wait is possible without JSPI; submit and rely on the
   * browser event loop to drain the queue once wasm yields. */
  submit();
}

}  // namespace blender::gpu

/* --- JS-facing capture exports (called by the harness after "Blender quit") ---
 * The render submitted a copyTextureToBuffer into g_capture_buf; here we map it
 * (async, on the now-free browser event loop) and stage the pixels for readback. */
#ifdef __EMSCRIPTEN__
static void webgpu_capture_map_cb(WGPUMapAsyncStatus status,
                                  WGPUStringView /*msg*/,
                                  void * /*u1*/,
                                  void * /*u2*/)
{
  if (status != WGPUMapAsyncStatus_Success || g_capture_buf == nullptr) {
    g_capture_ready = -1;
    return;
  }
  const size_t size = size_t(g_capture_bpr) * g_capture_h;
  const void *p = wgpuBufferGetConstMappedRange(g_capture_buf, 0, size);
  if (p == nullptr) {
    g_capture_ready = -1;
    return;
  }
  free(g_capture_pixels);
  g_capture_pixels = static_cast<uint8_t *>(malloc(size));
  memcpy(g_capture_pixels, p, size);
  wgpuBufferUnmap(g_capture_buf);
  g_capture_ready = 1;
}

extern "C" {
EMSCRIPTEN_KEEPALIVE int wgpu_capture_w() { return int(g_capture_w); }
EMSCRIPTEN_KEEPALIVE int wgpu_capture_h() { return int(g_capture_h); }
EMSCRIPTEN_KEEPALIVE int wgpu_capture_bpr() { return int(g_capture_bpr); }
EMSCRIPTEN_KEEPALIVE int wgpu_capture_bpp() { return int(g_capture_bpp); }
EMSCRIPTEN_KEEPALIVE int wgpu_capture_ready() { return g_capture_ready; }
EMSCRIPTEN_KEEPALIVE uint8_t *wgpu_capture_ptr() { return g_capture_pixels; }
EMSCRIPTEN_KEEPALIVE void wgpu_capture_map()
{
  if (g_capture_buf == nullptr) {
    g_capture_ready = -1;
    return;
  }
  g_capture_ready = 0;
  WGPUBufferMapCallbackInfo ci = {};
  ci.mode = WGPUCallbackMode_AllowSpontaneous;
  ci.callback = webgpu_capture_map_cb;
  wgpuBufferMapAsync(
      g_capture_buf, WGPUMapMode_Read, 0, size_t(g_capture_bpr) * g_capture_h, ci);
}
}
#endif
