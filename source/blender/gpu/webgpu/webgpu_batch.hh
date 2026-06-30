/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * WebGPU batch. Builds a WGPURenderPipeline from the bound shader's modules, the
 * active framebuffer's color/depth formats, and the batch's vertex format(s),
 * then records a draw into the active render pass. Pipelines are cached on the
 * context.
 */

#pragma once

#include "GPU_batch.hh"

#include <webgpu/webgpu.h>

namespace blender::gpu {

class WebGPUBatch : public Batch {
 public:
  void draw(int vertex_first, int vertex_count, int instance_first, int instance_count) override;
  void draw_indirect(StorageBuf * /*indirect_buf*/, intptr_t /*offset*/) override {}
  void multi_draw_indirect(StorageBuf * /*indirect_buf*/,
                           int /*count*/,
                           intptr_t /*offset*/,
                           intptr_t /*stride*/) override {}
};

}  // namespace blender::gpu
