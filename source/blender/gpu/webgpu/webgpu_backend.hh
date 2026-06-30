/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU GPU backend (Emscripten / Dawn). Targets the browser's native WebGPU
 * via the emdawnwebgpu port — the path that lets EEVEE-Next run in the browser
 * (compute shaders + storage buffers, which WebGL2 lacks). WebGPU primitives
 * (compute, render-to-texture, no JSPI) were verified standalone before this
 * scaffold; see the smoke/wgpu_*.c probes and the webgpu-verified memory.
 *
 * Status: SCAFFOLD. Object allocators that return nullptr are TODO, to be filled
 * in incrementally: triangle → Workbench → EEVEE → GUI. Shader path will be
 * GLSL→SPIR-V (shaderc, as the Vulkan backend) → WGSL (Tint), or WGSL generated
 * from ShaderCreateInfo like the Metal backend generates MSL.
 */

#pragma once

#include <cstdio>

#include "GPU_capabilities.hh"
#include "GPU_worker.hh"

#include "gpu_backend.hh"
#include "gpu_capabilities_private.hh"
#include "gpu_platform_private.hh"
#include "gpu_shader_private.hh"

#include "webgpu_batch.hh"
#include "webgpu_context.hh"
#include "webgpu_framebuffer.hh"
#include "webgpu_index_buffer.hh"
#include "webgpu_shader.hh"
#include "webgpu_storage_buffer.hh"
#include "webgpu_texture.hh"
#include "webgpu_texture_pool.hh"
#include "webgpu_uniform_buffer.hh"
#include "webgpu_vertex_buffer.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUBackend : public GPUBackend {
 private:
  WGPUInstance instance_ = nullptr;

 public:
  WebGPUBackend()
  {
    instance_ = wgpuCreateInstance(nullptr);
    GPG.init(GPU_DEVICE_ANY,
             GPU_OS_ANY,
             GPU_DRIVER_ANY,
             GPU_SUPPORT_LEVEL_SUPPORTED,
             GPU_BACKEND_WEBGPU,
             "WebGPU",
             "",
             "",
             GPU_ARCHITECTURE_IMR);
  }
  ~WebGPUBackend() override
  {
    if (instance_) {
      wgpuInstanceRelease(instance_);
    }
  }

  void init_resources() override
  {
    /* Force synchronous, main-thread shader compilation: this leaves
     * ShaderCompiler::compilation_worker_ null so async_compilation() compiles
     * inline on the calling thread. Avoids a GPUWorker background thread racing
     * the main thread (which corrupted the heap), and sidesteps WebGPU device
     * sharing across worker threads — neither is wired for wasm yet. */
    GCaps.use_main_context_workaround = true;

    /* WebGPU device capabilities. The device is created later (in the context),
     * so we cannot query it here; use values matching Dawn's common per-stage
     * limits, which the browser then re-validates at pipeline creation. Without
     * this, every limit is 0 and the GPU module rejects any shader that uses a
     * sampler ("too many samplers"), a storage buffer, etc. */
    GCaps.max_texture_size = 8192;
    GCaps.max_texture_3d_size = 2048;
    GCaps.max_texture_layers = 256;
    GCaps.max_textures = 16;        /* maxSampledTexturesPerShaderStage */
    GCaps.max_images = 8;           /* maxStorageTexturesPerShaderStage */
    GCaps.max_work_group_count[0] = 65535;
    GCaps.max_work_group_count[1] = 65535;
    GCaps.max_work_group_count[2] = 65535;
    GCaps.max_work_group_size[0] = 256;
    GCaps.max_work_group_size[1] = 256;
    GCaps.max_work_group_size[2] = 64;
    GCaps.max_uniforms_vert = 1024;
    GCaps.max_uniforms_frag = 1024;
    GCaps.max_batch_indices = 1 << 24;
    GCaps.max_batch_vertices = 1 << 24;
    GCaps.max_vertex_attribs = 16;
    GCaps.max_varying_floats = 60;  /* maxInterStageShaderComponents */
    GCaps.max_shader_storage_buffer_bindings = 8;
    GCaps.max_compute_shader_storage_blocks = 8;
    GCaps.max_uniform_buffer_size = 65536;
    GCaps.max_storage_buffer_size = size_t(1) << 27; /* 128 MB */
    GCaps.storage_buffer_alignment = 256;
    GCaps.max_parallel_compilations = 0; /* synchronous main-thread compilation */
    GCaps.mem_stats_support = false;
    GCaps.geometry_shader_support = false;
    /* The generic ShaderCompiler drives shader compilation; without it
     * get_compiler() returns the uninitialized compiler_ pointer and any
     * compilation aborts. */
    compiler_ = MEM_new<ShaderCompiler>(
        __func__, 1, GPUWorker::ContextType::Main);
  }
  void delete_resources() override
  {
    if (compiler_) {
      MEM_delete(compiler_);
      compiler_ = nullptr;
    }
  }

  void compute_dispatch(int groups_x_len, int groups_y_len, int groups_z_len) override
  {
    WebGPUContext *ctx = WebGPUContext::get();
    {
      static int s_disp_entry = 0;
      if (s_disp_entry < 12) {
        WebGPUShader *es = ctx ? static_cast<WebGPUShader *>(ctx->shader) : nullptr;
        fprintf(stderr,
                "WGPU_DISPATCH entry #%d ctx=%p dev=%p shader=%p cmod=%p\n",
                s_disp_entry++,
                (void *)ctx,
                ctx ? (void *)ctx->device() : nullptr,
                (void *)es,
                es ? (void *)es->compute_module() : nullptr);
        fflush(stderr);
      }
    }
    if (ctx == nullptr || ctx->device() == nullptr) {
      return;
    }
    WebGPUShader *sh = static_cast<WebGPUShader *>(ctx->shader);
    if (sh == nullptr || !sh->is_valid() || sh->compute_module() == nullptr ||
        sh->interface == nullptr)
    {
      return;
    }
    const uint64_t key = uint64_t(uintptr_t(sh));
    WGPUComputePipeline pipe = ctx->compute_pipeline_get(key);
    if (pipe == nullptr) {
      WGPUBindGroupLayout bgl_explicit = ctx->make_bind_group_layout(sh->compute_bindings(), true);
      WGPUPipelineLayout pipe_layout = ctx->make_pipeline_layout(bgl_explicit);
      WGPUComputePipelineDescriptor d = {};
      d.layout = pipe_layout;
      d.compute.module = sh->compute_module();
      d.compute.entryPoint = {"main", WGPU_STRLEN};
      pipe = wgpuDeviceCreateComputePipeline(ctx->device(), &d);
      if (pipe_layout) {
        wgpuPipelineLayoutRelease(pipe_layout);
      }
      if (bgl_explicit) {
        wgpuBindGroupLayoutRelease(bgl_explicit);
      }
      if (pipe == nullptr) {
        return;
      }
      ctx->compute_pipeline_put(key, pipe);
    }
    /* Compute work cannot be recorded inside a render pass. */
    ctx->render_pass_end();
    WGPUCommandEncoder enc = ctx->ensure_encoder();
    if (enc == nullptr) {
      return;
    }
    WGPUComputePassEncoder cpass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipe, 0);
    WGPUBindGroup bg = ctx->build_bind_group(sh, sh->compute_bindings(), bgl);
    static int s_dispatch_log = 0;
    if (s_dispatch_log < 8) {
      fprintf(stderr,
              "WGPU_DISPATCH #%d shader='%s' groups=%d,%d,%d bg=%p\n",
              s_dispatch_log++,
              sh->name_get().c_str(),
              groups_x_len,
              groups_y_len,
              groups_z_len,
              (void *)bg);
      fflush(stderr);
    }
    wgpuComputePassEncoderSetPipeline(cpass, pipe);
    if (bg) {
      wgpuComputePassEncoderSetBindGroup(cpass, 0, bg, 0, nullptr);
    }
    wgpuComputePassEncoderDispatchWorkgroups(
        cpass, uint32_t(groups_x_len), uint32_t(groups_y_len), uint32_t(groups_z_len));
    wgpuComputePassEncoderEnd(cpass);
    wgpuComputePassEncoderRelease(cpass);
    if (bg) {
      wgpuBindGroupRelease(bg);
    }
    wgpuBindGroupLayoutRelease(bgl);
  }
  void compute_dispatch_indirect(StorageBuf * /*indirect_buf*/) override {}

  Context *context_alloc(GHOST_IWindow *ghost_window, GHOST_IContext *ghost_context) override
  {
    return new WebGPUContext(ghost_window, ghost_context);
  }

  Batch *batch_alloc() override { return new WebGPUBatch; }
  FrameBuffer *framebuffer_alloc(const char *name) override { return new WebGPUFrameBuffer(name); }
  VertBuf *vertbuf_alloc() override { return new WebGPUVertexBuffer; }
  Shader *shader_alloc(const char *name) override { return new WebGPUShader(name); }
  Texture *texture_alloc(const char *name) override { return new WebGPUTexture(name); }
  TexturePool *texturepool_alloc() override { return new WebGPUTexturePool(); }
  UniformBuf *uniformbuf_alloc(size_t size, const char *name) override
  {
    return new WebGPUUniformBuf(size, name);
  }
  StorageBuf *storagebuf_alloc(size_t size, GPUUsageType usage, const char *name) override
  {
    return new WebGPUStorageBuf(size, usage, name);
  }

  IndexBuf *indexbuf_alloc() override { return new WebGPUIndexBuf; }

  /* TODO: still stubs — implement as the render path starts exercising them. */
  Fence *fence_alloc() override { return nullptr; }
  PixelBuffer *pixelbuf_alloc(size_t /*size*/) override { return nullptr; }
  QueryPool *querypool_alloc() override { return nullptr; }

  void shader_cache_dir_clear_old() override {}
  void render_begin() override {}
  void render_end() override
  {
    /* Flush any recorded commands so the final frame (and the debug capture copy)
     * actually executes on the GPU. */
    if (WebGPUContext *ctx = WebGPUContext::get()) {
      ctx->submit();
    }
  }
  void render_step(bool /*force_resource_release*/) override
  {
    /* EEVEE calls GPU_render_step() after each sample; WebGPU needs an explicit
     * submit for recorded draws/dispatches to run (it doesn't call GPU_flush for
     * our backend). Submit here so work is paced per sample, not buffered into one
     * giant command buffer. */
    if (WebGPUContext *ctx = WebGPUContext::get()) {
      ctx->submit();
    }
  }

  WGPUInstance instance() const { return instance_; }
};

}  // namespace blender::gpu
