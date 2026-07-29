/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/* Based on Frosbite Unified Volumetric.
 * https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite */

/* Store volumetric properties into the froxel textures. */

#pragma once

#include "infos/eevee_geom_infos.hh"
#include "infos/eevee_nodetree_infos.hh"

#ifdef GLSL_CPP_STUBS
#  define MAT_VOLUME
#endif

FRAGMENT_SHADER_CREATE_INFO(eevee_nodetree)

#include "eevee_volume_lib.bsl.hh"

/* Needed includes for shader nodes. */
#include "eevee_attributes_volume_lib.glsl"
#include "eevee_nodetree_frag_lib.glsl"
#include "eevee_occupancy_lib.bsl.hh"
#include "eevee_sampling_lib.bsl.hh"

GlobalData init_globals(const ViewMatrices view, float3 wP)
{
  GlobalData surf;
  surf.P = wP;
  surf.N = float3(0.0f);
  surf.Ng = float3(0.0f);
  surf.is_strand = false;
  surf.hair_diameter = 0.0f;
  surf.hair_strand_id = 0;
  surf.barycentric_coords = float2(0.0f);
  surf.barycentric_dists = float3(0.0f);
  surf.ray_type = RAY_TYPE_CAMERA;
  surf.ray_depth = 0.0f;
  surf.ray_length = distance(surf.P, view.position());
  return surf;
}

namespace eevee {

struct VolumeProperties {
  float3 scattering;
  float3 absorption;
  float3 emission;
  float anisotropy;
};

struct SurfVolume {
  [[compilation_constant]] bool is_homogenous;
  [[compilation_constant]] bool is_volume_object;
  [[compilation_constant]] bool is_world;

  [[legacy_info]] ShaderCreateInfo draw_modelmat_common;
  [[legacy_info]] ShaderCreateInfo eevee_geom_iface_info;

  /* SSBO instead of R32UI image — see occupancy_buf_index (WGSL image atomics). */
  [[storage(OCCUPANCY_BUF_SLOT, read)]] const uint (&occupancy_buf)[];
  /* Accumulation mirror of the prop images: WebGPU forbids read_write storage
   * on RG11B10/R16F, so reads come from this buffer and the images stay
   * write-only (the last store per froxel holds the accumulated value).
   * Layout: VOLUME_PROP_BUF_STRIDE floats per froxel —
   * scatter.xyz, extinction.xyz, emission.xyz, phase, phase_weight, pad. */
  [[storage(VOLUME_PROP_BUF_SLOT, read_write)]] float (&volume_prop_buf)[];

  /* WRITE-only: reads go through volume_prop_buf (WebGPU forbids read_write
   * storage on RG11B10/R16F formats). */
  [[image(VOLUME_PROP_SCATTERING_IMG_SLOT, write, UFLOAT_11_11_10)]] image3D out_scattering_img;
  [[image(VOLUME_PROP_EXTINCTION_IMG_SLOT, write, UFLOAT_11_11_10)]] image3D out_extinction_img;
  [[image(VOLUME_PROP_EMISSION_IMG_SLOT, write, UFLOAT_11_11_10)]] image3D out_emissive_img;
  [[image(VOLUME_PROP_PHASE_IMG_SLOT, write, SFLOAT_16)]] image3D out_phase_img;
  [[image(VOLUME_PROP_PHASE_WEIGHT_IMG_SLOT, write, SFLOAT_16)]] image3D out_phase_weight_img;

  void write_froxel(int3 froxel, VolumeProperties prop)
  {
    float2 phase = float2(prop.anisotropy, 1.0f);

    /* Do not add phase weight if there's no scattering. */
    if (all(equal(prop.scattering, float3(0.0f)))) {
      phase = float2(0.0f);
    }

    float3 extinction = prop.scattering + prop.absorption;

    const int2 vts = imageSize(out_scattering_img).xy;
    const int accum = ((froxel.z * vts.y + froxel.y) * vts.x + froxel.x) *
                      VOLUME_PROP_BUF_STRIDE;
    if (!is_world) [[static_branch]] {
      /* Additive Blending. No race condition since we have a barrier between each conflicting
       * invocations. Reads go through the accumulation buffer: the images stay
       * write-only (WebGPU forbids read_write storage on these formats). */
      prop.scattering += float3(
          volume_prop_buf[accum + 0], volume_prop_buf[accum + 1], volume_prop_buf[accum + 2]);
      extinction += float3(
          volume_prop_buf[accum + 3], volume_prop_buf[accum + 4], volume_prop_buf[accum + 5]);
      prop.emission += float3(
          volume_prop_buf[accum + 6], volume_prop_buf[accum + 7], volume_prop_buf[accum + 8]);
      phase.x += volume_prop_buf[accum + 9];
      phase.y += volume_prop_buf[accum + 10];
    }

    volume_prop_buf[accum + 0] = prop.scattering.x;
    volume_prop_buf[accum + 1] = prop.scattering.y;
    volume_prop_buf[accum + 2] = prop.scattering.z;
    volume_prop_buf[accum + 3] = extinction.x;
    volume_prop_buf[accum + 4] = extinction.y;
    volume_prop_buf[accum + 5] = extinction.z;
    volume_prop_buf[accum + 6] = prop.emission.x;
    volume_prop_buf[accum + 7] = prop.emission.y;
    volume_prop_buf[accum + 8] = prop.emission.z;
    volume_prop_buf[accum + 9] = phase.x;
    volume_prop_buf[accum + 10] = phase.y;

    imageStoreFast(out_scattering_img, froxel, prop.scattering.xyzz);
    imageStoreFast(out_extinction_img, froxel, extinction.xyzz);
    imageStoreFast(out_emissive_img, froxel, prop.emission.xyzz);
    imageStoreFast(out_phase_img, froxel, phase.xxxx);
    imageStoreFast(out_phase_weight_img, froxel, phase.yyyy);
  }

  VolumeProperties eval_froxel([[resource_table]] const Uniform &uni,
                               const ViewMatrices view,
                               const ObjectMatrices obj,
                               const ObjectInfos ob_infos,
                               int3 froxel,
                               float jitter)
  {
    float3 uvw = (float3(froxel) + float3(0.5f, 0.5f, 0.5f - jitter)) *
                 uni.uniform_buf.volumes.inv_tex_size;

    float3 vP = volume_jitter_to_view(uni, view, uvw);
    float3 wP = view.point_view_to_world(vP);
    float3 lP = obj.point_world_to_object(wP);
    /* Compute Original Coordinate (ORCO). */
    float3 lP_orco = lP * ob_infos.orco_mul + ob_infos.orco_add;

    g_data = init_globals(view, wP);
    attrib_load(VolumePoint{lP, lP_orco});
    nodetree_volume();

    if (is_volume_object) [[static_branch]] {
      const auto &drw_volume = buffer_get(draw_volume_infos, drw_volume);
      g_volume_scattering *= drw_volume.density_scale;
      g_volume_absorption *= drw_volume.density_scale;
      g_emission *= drw_volume.density_scale;
    }

    VolumeProperties prop;
    prop.scattering = g_volume_scattering;
    prop.absorption = g_volume_absorption;
    prop.emission = g_emission;
    prop.anisotropy = g_volume_anisotropy;
    return prop;
  }
};

/* Note: Only the front fragments have to be invoked. */
[[fragment]] [[early_fragment_tests]] [[texture_atomic]]
void surf_volume([[resource_table]] SurfVolume &srt,
                 [[resource_table]] const Uniform &uni,
                 [[resource_table]] const draw::Model &models,
                 [[resource_table]] const draw::View &views,
                 [[resource_table]] const draw::Infos &infos,
                 [[resource_table]] const Sampling &sampling,
                 [[resource_table]] const UtilityTexture & /*util_tx*/,
                 [[frag_coord]] const float4 frag_co,
                 [[front_facing]] const bool /*front_face*/ /* Needed for nodes. */)
{
  int3 froxel = int3(int2(frag_co.xy), 0);
  float offset = sampling.rng_1D_get(SAMPLING_VOLUME_W);
  float jitter = volume_froxel_jitter(froxel.xy, offset);

  auto &interp_flat = interface_get(eevee_geom_iface_info, interp_flat);
  draw::ID id{interp_flat.resource_id_raw};
  const uint resource_id = id.resource_id<1>();
  const ObjectMatrices obj = models.get(resource_id);
  const ObjectInfos ob_infos = infos.get(resource_id);
  const ViewMatrices view = views.get(0);

  VolumeProperties prop;

  if (srt.is_homogenous) [[static_branch]] {
    /* Homogenous volumes only evaluate properties at volume entrance and write the same values for
     * each froxel. */
    prop = srt.eval_froxel(uni, view, obj, ob_infos, froxel, jitter);
  }

  occupancy::Bits occupancy;

  if (!srt.is_world) [[static_branch]] {
    /* The buffer only holds ceil(tex_size.z / 32) words per froxel column —
     * unlike the old image, out-of-range reads are NOT defined-zero. */
    const int layer_len = (int(uni.uniform_buf.volumes.tex_size.z) + 31) / 32;
    for (int j = 0; j < 8; j++) {
      occupancy.bits[j] = (j < layer_len) ?
                              srt.occupancy_buf[::occupancy::occupancy_buf_index(
                                  froxel.xy, j, int2(uni.uniform_buf.volumes.tex_size.xy))] :
                              0u;
    }
  }

  /* Check all occupancy bits. */
  for (int j = 0; j < 8; j++) {
    for (int i = 0; i < 32; i++) {
      froxel.z = j * 32 + i;

      if (froxel.z >= imageSize(srt.out_scattering_img).z) {
        break;
      }

      if (!srt.is_world) [[static_branch]] {
        if (((occupancy.bits[j] >> i) & 1u) == 0) {
          continue;
        }
      }

      if (!srt.is_homogenous) [[static_branch]] {
        /* Heterogeneous volumes evaluate properties at every froxel position. */
        prop = srt.eval_froxel(uni, view, obj, ob_infos, froxel, jitter);
      }
      srt.write_froxel(froxel, prop);
    }
  }
}

}  // namespace eevee
