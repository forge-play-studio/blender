/* SPDX-FileCopyrightText: 2020-2022 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "infos/workbench_effect_outline_infos.hh"

FRAGMENT_SHADER_CREATE_INFO(workbench_effect_outline)

#ifdef GPU_WEBGPU
/* WGSL cannot SAMPLE integer textures — fetch the nearest texel instead
 * (identical to nearest-sampling an id buffer). */
uint object_id_sample(float2 uv)
{
  int2 sz = textureSize(object_id_buffer, 0).xy;
  int2 texel = clamp(int2(uv * float2(sz)), int2(0), sz - 1);
  return texelFetch(object_id_buffer, texel, 0).r;
}
#  define OBJECT_ID_SAMPLE(uv) object_id_sample(uv)
#else
#  define OBJECT_ID_SAMPLE(uv) texture(object_id_buffer, uv).r
#endif

void main()
{
  float2 uv = gl_FragCoord.xy / float2(textureSize(object_id_buffer, 0));

  float3 offset = float3(world_data.viewport_size_inv, 0.0f) * world_data.ui_scale;

  uint center_id = OBJECT_ID_SAMPLE(uv);
  uint4 adjacent_ids = uint4(OBJECT_ID_SAMPLE(uv + offset.zy),
                             OBJECT_ID_SAMPLE(uv - offset.zy),
                             OBJECT_ID_SAMPLE(uv + offset.xz),
                             OBJECT_ID_SAMPLE(uv - offset.xz));

  float outline_opacity = 1.0f - dot(float4(equal(uint4(center_id), adjacent_ids)), float4(0.25f));

  frag_color = world_data.object_outline_color * outline_opacity;
}
