/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU state manager. Pipeline state in WebGPU is baked into render-pipeline
 * objects rather than set via global calls, and texture/image binding happens
 * through bind groups at draw time. So this tracks the requested state but the
 * no-op bind/unbind methods are placeholders until bind-group assembly lands.
 */

#pragma once

#include "gpu_state_private.hh"

namespace blender::gpu {

class WebGPUStateManager : public StateManager {
 public:
  WebGPUStateManager() = default;

  void apply_state() override {}
  void force_state() override {}

  void issue_barrier(GPUBarrier /*barrier_bits*/) override {}

  /* Route texture/image binds into the context binding tables (consumed when a
   * bind group is assembled at draw/dispatch time). Implemented in
   * webgpu_state.cc to avoid pulling the context into this header. */
  void texture_bind(Texture *tex, GPUSamplerState sampler, int unit) override;
  void texture_unbind(Texture * /*tex*/) override {}
  void texture_unbind_all() override {}

  void image_bind(Texture *tex, int unit) override;
  void image_unbind(Texture * /*tex*/) override {}
  void image_unbind_all() override {}
};

}  // namespace blender::gpu
