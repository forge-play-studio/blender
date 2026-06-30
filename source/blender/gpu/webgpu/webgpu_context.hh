/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU backend context. Holds the WGPUDevice/queue plus the per-frame command
 * encoder, the active render pass, the resource binding tables (UBO/SSBO/texture/
 * image — WebGPU binds resources through bind groups at draw time, not via global
 * state), and a render-pipeline cache. On Emscripten the device is created in JS
 * and handed over via emscripten_webgpu_get_device() (no JSPI).
 */

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "gpu_context_private.hh"

#include <webgpu/webgpu.h>

#include "webgpu_framebuffer.hh"
#include "webgpu_shader.hh"

namespace blender::gpu {

class WebGPUTexture;

/* WebGPU baseline per-stage binding counts. */
#define WEBGPU_MAX_UBO 16
#define WEBGPU_MAX_SSBO 16
#define WEBGPU_MAX_TEX 16
#define WEBGPU_MAX_IMAGE 8

class WebGPUContext : public Context {
 private:
  WGPUDevice device_ = nullptr;
  WGPUQueue queue_ = nullptr;

  /* Per-frame command recording. Created lazily; flushed on submit(). */
  WGPUCommandEncoder encoder_ = nullptr;
  WGPURenderPassEncoder render_pass_ = nullptr;
  WebGPUFrameBuffer *render_pass_fb_ = nullptr;

  /* Resource binding tables (group 0). Populated by *Buf::bind / texture_bind,
   * consumed when a bind group is assembled at draw/dispatch time. */
  WGPUBuffer bound_ubo_[WEBGPU_MAX_UBO] = {};
  WGPUBuffer bound_ssbo_[WEBGPU_MAX_SSBO] = {};
  WebGPUTexture *bound_tex_[WEBGPU_MAX_TEX] = {};
  WebGPUTexture *bound_image_[WEBGPU_MAX_IMAGE] = {};

  /* Render-pipeline cache keyed by (shader, vertex/format/prim/target) hash. */
  std::unordered_map<uint64_t, WGPURenderPipeline> render_pipelines_;
  std::unordered_map<uint64_t, WGPUComputePipeline> compute_pipelines_;

  WGPUSampler default_sampler_ = nullptr;

  /* Debug capture: the WGPU handle (AddRef'd at draw time so it survives the
   * pooled texture being freed/reused) of the first content color texture; copied
   * to a persistent readback buffer right after its render pass ends, so JS can
   * map it after the render (see the webgpu_capture_* exports). */
  WGPUTexture capture_src_ = nullptr;
  uint32_t capture_w_ = 0, capture_h_ = 0;
  WGPUTextureFormat capture_fmt_ = WGPUTextureFormat_Undefined;
  bool capture_done_ = false;

 public:
  WebGPUContext(GHOST_IWindow *ghost_window, GHOST_IContext *ghost_context);
  ~WebGPUContext() override;

  static WebGPUContext *get() { return static_cast<WebGPUContext *>(Context::get()); }

  void activate() override {}
  void deactivate() override {}
  void begin_frame() override {}
  void end_frame() override {}
  void flush() override;
  void finish() override;

  void memory_statistics_get(int *r_total_mem, int *r_free_mem) override
  {
    *r_total_mem = 0;
    *r_free_mem = 0;
  }

  void debug_group_begin(const char * /*name*/, int /*index*/) override {}
  void debug_group_end() override {}
  bool debug_capture_begin(const char * /*title*/) override { return false; }
  void debug_capture_end() override {}
  void *debug_capture_scope_create(const char * /*name*/) override { return nullptr; }
  bool debug_capture_scope_begin(void * /*scope*/) override { return false; }
  void debug_capture_scope_end(void * /*scope*/) override {}
  void debug_unbind_all_ubo() override {}
  void debug_unbind_all_ssbo() override {}

  WGPUDevice device() const { return device_; }
  WGPUQueue queue() const { return queue_; }

  /* --- command / render-pass lifecycle --- */
  WGPUCommandEncoder ensure_encoder();
  /* Begin a render pass on `fb` if one is not already active for it (ends any
   * other active pass first). No-op without a device. */
  void render_pass_ensure(WebGPUFrameBuffer &fb);
  void render_pass_end();
  WGPURenderPassEncoder render_pass() const { return render_pass_; }
  /* Submit recorded commands to the queue (ends any active pass first). */
  void submit();

  /* Debug capture: remember the FIRST content color texture (first-wins). AddRefs
   * the WGPU handle so it stays valid even after the pooled gpu::Texture is freed.
   * Implemented in the .cc (needs the full WebGPUTexture type). */
  void set_capture_target(WebGPUTexture *t);
  bool has_capture_target() const { return capture_src_ != nullptr; }

  /* Assemble a group-0 bind group for `shader` from the current binding tables,
   * driven by the WGSL-parsed `bindings` (the exact set the pipeline expects),
   * using the pipeline's auto-derived layout `bgl`. Returns null if no device or
   * if a required resource is unbound (the draw is then skipped). */
  WGPUBindGroup build_bind_group(WebGPUShader *shader,
                                 const std::vector<WgslResourceBinding> &bindings,
                                 WGPUBindGroupLayout bgl);

  /* A cached default linear/repeat sampler used for all sampler bindings until
   * per-binding sampler state is wired. */
  WGPUSampler default_sampler();

  /* Build an explicit bind-group layout (+ pipeline layout wrapping it) from the
   * WGSL-parsed bindings, so the pipeline's layout contains EXACTLY those bindings
   * (auto-layout strips unused ones → mismatch with our complete bind group). The
   * caller owns the returned objects and should release them after pipeline
   * creation (the pipeline retains its own refs). */
  WGPUBindGroupLayout make_bind_group_layout(const std::vector<WgslResourceBinding> &bindings,
                                             bool is_compute);
  WGPUPipelineLayout make_pipeline_layout(WGPUBindGroupLayout bgl);

  /* --- resource binding tables --- */
  void bind_ubo(int slot, WGPUBuffer buf)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_UBO) {
      bound_ubo_[slot] = buf;
    }
  }
  void unbind_ubo(int slot)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_UBO) {
      bound_ubo_[slot] = nullptr;
    }
  }
  void bind_ssbo(int slot, WGPUBuffer buf)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_SSBO) {
      bound_ssbo_[slot] = buf;
    }
  }
  void unbind_ssbo(int slot)
  {
    if (slot >= 0 && slot < WEBGPU_MAX_SSBO) {
      bound_ssbo_[slot] = nullptr;
    }
  }
  void bind_texture(int unit, WebGPUTexture *tex)
  {
    if (unit >= 0 && unit < WEBGPU_MAX_TEX) {
      bound_tex_[unit] = tex;
    }
  }
  void bind_image(int unit, WebGPUTexture *tex)
  {
    if (unit >= 0 && unit < WEBGPU_MAX_IMAGE) {
      bound_image_[unit] = tex;
    }
  }
  WGPUBuffer ubo_at(int slot) const { return bound_ubo_[slot]; }
  WGPUBuffer ssbo_at(int slot) const { return bound_ssbo_[slot]; }
  WebGPUTexture *tex_at(int unit) const { return bound_tex_[unit]; }
  WebGPUTexture *image_at(int unit) const { return bound_image_[unit]; }

  /* --- pipeline cache --- */
  WGPURenderPipeline render_pipeline_get(uint64_t key) const
  {
    auto it = render_pipelines_.find(key);
    return it == render_pipelines_.end() ? nullptr : it->second;
  }
  void render_pipeline_put(uint64_t key, WGPURenderPipeline p) { render_pipelines_[key] = p; }
  WGPUComputePipeline compute_pipeline_get(uint64_t key) const
  {
    auto it = compute_pipelines_.find(key);
    return it == compute_pipelines_.end() ? nullptr : it->second;
  }
  void compute_pipeline_put(uint64_t key, WGPUComputePipeline p) { compute_pipelines_[key] = p; }
};

}  // namespace blender::gpu
