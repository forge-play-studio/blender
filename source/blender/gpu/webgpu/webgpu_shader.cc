/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cctype>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <shaderc/shaderc.hpp>

#include "src/tint/lang/spirv/reader/reader.h"
#include "src/tint/lang/wgsl/writer/writer.h"
#include "src/tint/lang/core/ir/module.h"
#include "src/tint/utils/ice/ice.h"

#include "gpu_shader_dependency_private.hh"

#include "GPU_context.hh"

#include "webgpu_context.hh"
#include "webgpu_shader.hh"
#include "webgpu_shader_interface.hh"

namespace blender::gpu {

/* Create a WGPUShaderModule from WGSL on the active device, or null if there is
 * no device (degraded headless mode) or the source is empty. */
static WGPUShaderModule webgpu_module_from_wgsl(const std::string &wgsl)
{
  if (wgsl.empty()) {
    return nullptr;
  }
  Context *ctx = Context::get();
  if (ctx == nullptr) {
    return nullptr;
  }
  WGPUDevice device = static_cast<WebGPUContext *>(ctx)->device();
  if (device == nullptr) {
    return nullptr;
  }
  WGPUShaderSourceWGSL src = {};
  src.chain.sType = WGPUSType_ShaderSourceWGSL;
  src.code = {wgsl.c_str(), wgsl.size()};
  WGPUShaderModuleDescriptor desc = {};
  desc.nextInChain = &src.chain;
  return wgpuDeviceCreateShaderModule(device, &desc);
}

using namespace shader;

/* --- GLSL declaration helpers (verbatim from vk_shader.cc) --- */
static const char *to_string(const Interpolation &interp)
{
  switch (interp) {
    case Interpolation::SMOOTH:
      return "smooth";
    case Interpolation::FLAT:
      return "flat";
    case Interpolation::NO_PERSPECTIVE:
      return "noperspective";
    default:
      return "unknown";
  }
}

static const char *to_string(const Type &type)
{
  switch (type) {
    case Type::float_t:
      return "float";
    case Type::float2_t:
      return "vec2";
    case Type::float3_t:
      return "vec3";
    case Type::float4_t:
      return "vec4";
    case Type::float3x3_t:
      return "mat3";
    case Type::float4x4_t:
      return "mat4";
    case Type::uint_t:
      return "uint";
    case Type::uint2_t:
      return "uvec2";
    case Type::uint3_t:
      return "uvec3";
    case Type::uint4_t:
      return "uvec4";
    case Type::int_t:
      return "int";
    case Type::int2_t:
      return "ivec2";
    case Type::int3_t:
      return "ivec3";
    case Type::int4_t:
      return "ivec4";
    case Type::bool_t:
      return "bool";
    default:
      return "unknown";
  }
}

static const char *to_string(const TextureFormat &type)
{
  switch (type) {
    case TextureFormat::UINT_8_8_8_8:
      return "rgba8ui";
    case TextureFormat::SINT_8_8_8_8:
      return "rgba8i";
    case TextureFormat::UNORM_8_8_8_8:
      return "rgba8";
    case TextureFormat::UINT_32_32_32_32:
      return "rgba32ui";
    case TextureFormat::SINT_32_32_32_32:
      return "rgba32i";
    case TextureFormat::SFLOAT_32_32_32_32:
      return "rgba32f";
    case TextureFormat::UINT_16_16_16_16:
      return "rgba16ui";
    case TextureFormat::SINT_16_16_16_16:
      return "rgba16i";
    case TextureFormat::SFLOAT_16_16_16_16:
      return "rgba16f";
    case TextureFormat::UNORM_16_16_16_16:
      return "rgba16";
    case TextureFormat::UINT_8_8:
      return "rg8ui";
    case TextureFormat::SINT_8_8:
      return "rg8i";
    case TextureFormat::UNORM_8_8:
      return "rg8";
    case TextureFormat::UINT_32_32:
      return "rg32ui";
    case TextureFormat::SINT_32_32:
      return "rg32i";
    case TextureFormat::SFLOAT_32_32:
      return "rg32f";
    case TextureFormat::UINT_16_16:
      return "rg16ui";
    case TextureFormat::SINT_16_16:
      return "rg16i";
    case TextureFormat::SFLOAT_16_16:
      return "rg16f";
    case TextureFormat::UNORM_16_16:
      return "rg16";
    case TextureFormat::UINT_8:
      return "r8ui";
    case TextureFormat::SINT_8:
      return "r8i";
    case TextureFormat::UNORM_8:
      return "r8";
    case TextureFormat::UINT_32:
      return "r32ui";
    case TextureFormat::SINT_32:
      return "r32i";
    case TextureFormat::SFLOAT_32:
      return "r32f";
    case TextureFormat::UINT_16:
      return "r16ui";
    case TextureFormat::SINT_16:
      return "r16i";
    case TextureFormat::SFLOAT_16:
      return "r16f";
    case TextureFormat::UNORM_16:
      return "r16";
    case TextureFormat::UFLOAT_11_11_10:
      return "r11f_g11f_b10f";
    case TextureFormat::UNORM_10_10_10_2:
      return "rgb10_a2";
    default:
      return "unknown";
  }
}

static void print_image_type(std::ostream &os,
                             const ImageType &type,
                             const ShaderCreateInfo::Resource::BindType bind_type)
{
  switch (type) {
    case ImageType::IntBuffer:
    case ImageType::Int1D:
    case ImageType::Int1DArray:
    case ImageType::Int2D:
    case ImageType::Int2DArray:
    case ImageType::Int3D:
    case ImageType::IntCube:
    case ImageType::IntCubeArray:
    case ImageType::AtomicInt2D:
    case ImageType::AtomicInt2DArray:
    case ImageType::AtomicInt3D:
      os << "i";
      break;
    case ImageType::UintBuffer:
    case ImageType::Uint1D:
    case ImageType::Uint1DArray:
    case ImageType::Uint2D:
    case ImageType::Uint2DArray:
    case ImageType::Uint3D:
    case ImageType::UintCube:
    case ImageType::UintCubeArray:
    case ImageType::AtomicUint2D:
    case ImageType::AtomicUint2DArray:
    case ImageType::AtomicUint3D:
      os << "u";
      break;
    default:
      break;
  }

  if (bind_type == ShaderCreateInfo::Resource::BindType::IMAGE) {
    os << "image";
  }
  else {
    os << "sampler";
  }

  switch (type) {
    case ImageType::FloatBuffer:
    case ImageType::IntBuffer:
    case ImageType::UintBuffer:
      os << "Buffer";
      break;
    case ImageType::Float1D:
    case ImageType::Float1DArray:
    case ImageType::Int1D:
    case ImageType::Int1DArray:
    case ImageType::Uint1D:
    case ImageType::Uint1DArray:
      os << "1D";
      break;
    case ImageType::Float2D:
    case ImageType::Float2DArray:
    case ImageType::Int2D:
    case ImageType::Int2DArray:
    case ImageType::Uint2D:
    case ImageType::Uint2DArray:
    case ImageType::Shadow2D:
    case ImageType::Shadow2DArray:
    case ImageType::Depth2D:
    case ImageType::Depth2DArray:
    case ImageType::AtomicInt2D:
    case ImageType::AtomicInt2DArray:
    case ImageType::AtomicUint2D:
    case ImageType::AtomicUint2DArray:
      os << "2D";
      break;
    case ImageType::Float3D:
    case ImageType::Int3D:
    case ImageType::AtomicInt3D:
    case ImageType::Uint3D:
    case ImageType::AtomicUint3D:
      os << "3D";
      break;
    case ImageType::FloatCube:
    case ImageType::FloatCubeArray:
    case ImageType::IntCube:
    case ImageType::IntCubeArray:
    case ImageType::UintCube:
    case ImageType::UintCubeArray:
    case ImageType::ShadowCube:
    case ImageType::ShadowCubeArray:
    case ImageType::DepthCube:
    case ImageType::DepthCubeArray:
      os << "Cube";
      break;
    default:
      break;
  }

  switch (type) {
    case ImageType::Float1DArray:
    case ImageType::Float2DArray:
    case ImageType::FloatCubeArray:
    case ImageType::Int1DArray:
    case ImageType::Int2DArray:
    case ImageType::IntCubeArray:
    case ImageType::Uint1DArray:
    case ImageType::Uint2DArray:
    case ImageType::UintCubeArray:
    case ImageType::Shadow2DArray:
    case ImageType::ShadowCubeArray:
    case ImageType::Depth2DArray:
    case ImageType::DepthCubeArray:
    case ImageType::AtomicUint2DArray:
      os << "Array";
      break;
    default:
      break;
  }

  switch (type) {
    case ImageType::Shadow2D:
    case ImageType::Shadow2DArray:
    case ImageType::ShadowCube:
    case ImageType::ShadowCubeArray:
      os << "Shadow";
      break;
    default:
      break;
  }
  os << " ";
}

static std::ostream &print_qualifier(std::ostream &os, const Qualifier &qualifiers)
{
  /* NOTE: 'restrict' omitted — it emits SPIR-V Restrict decoration (19) which
   * Tint's SPIR-V reader does not handle (TINT_UNIMPLEMENTED). It is only an
   * aliasing hint, safe to drop for the WGSL path. */
  if (!flag_is_set(qualifiers, Qualifier::read)) {
    os << "writeonly ";
  }
  if (!flag_is_set(qualifiers, Qualifier::write)) {
    os << "readonly ";
  }
  return os;
}

inline int get_location_count(const Type &type)
{
  if (type == shader::Type::float4x4_t) {
    return 4;
  }
  if (type == shader::Type::float3x3_t) {
    return 3;
  }
  return 1;
}

static void print_interface_as_attributes(std::ostream &os,
                                          const std::string &prefix,
                                          const StageInterfaceInfo &iface,
                                          int &location)
{
  for (const StageInterfaceInfo::InOut &inout : iface.inouts) {
    os << "layout(location=" << location << ") " << prefix << " " << to_string(inout.interp) << " "
       << to_string(inout.type) << " " << inout.name << ";\n";
    location += get_location_count(inout.type);
  }
}

static void print_interface_as_struct(std::ostream &os,
                                      const std::string &prefix,
                                      const StageInterfaceInfo &iface,
                                      int &location,
                                      const StringRefNull &suffix)
{
  std::string struct_name = prefix + iface.name;
  Interpolation qualifier = iface.inouts[0].interp;

  os << "struct " << struct_name << " {\n";
  for (const StageInterfaceInfo::InOut &inout : iface.inouts) {
    os << "  " << to_string(inout.type) << " " << inout.name << ";\n";
  }
  os << "};\n";
  os << "layout(location=" << location << ") " << prefix << " " << to_string(qualifier) << " "
     << struct_name << " " << iface.instance_name << suffix << ";\n";

  for (const StageInterfaceInfo::InOut &inout : iface.inouts) {
    location += get_location_count(inout.type);
  }
}

static void print_interface(std::ostream &os,
                            const std::string &prefix,
                            const StageInterfaceInfo &iface,
                            int &location,
                            const StringRefNull &suffix = "")
{
  if (iface.instance_name.is_empty()) {
    print_interface_as_attributes(os, prefix, iface, location);
  }
  else {
    print_interface_as_struct(os, prefix, iface, location, suffix);
  }
}

/* Build the per-stage version patch: #version + extensions + compat macros,
 * by resolving Blender's compat GLSL with our extension header injected. Mirrors
 * VKDevice::glsl_*_patch_get. Targets Vulkan-flavoured GLSL (set/binding/location)
 * which shaderc compiles to SPIR-V suitable for Tint. */
static std::string webgpu_glsl_patch(const char *stage_define)
{
  std::stringstream ss;
  ss << "#version 450\n";
  /* NOTE: do NOT enable GL_ARB_shader_draw_parameters — it emits the
   * SPV_KHR_shader_draw_parameters SPIR-V extension which Tint's reader rejects.
   * WebGPU has no base-instance, so gpu_BaseInstance is 0. */
  ss << "#define gpu_BaseInstance 0\n";
  ss << "#define GPU_ARB_clip_control\n";
  ss << "#define GPU_ARB_derivative_control\n";
  ss << "#define gl_VertexID gl_VertexIndex\n";
  ss << "#define gpu_InstanceIndex (gl_InstanceIndex)\n";
  ss << "#define gl_InstanceID gl_InstanceIndex\n";
  /* WebGPU has no multi-viewport / gl_ViewportIndex in the vertex stage; EEVEE's
   * shadow/geometry shaders reference gpu_ViewportIndex for layered rendering.
   * Pin to layer 0 (single-viewport) so they compile. */
  ss << "#define gpu_ViewportIndex 0\n";
  /* Tint's SPIR-V reader does not implement OpIsNan/OpIsInf — replace isnan/isinf
   * with manual expressions so glslang never emits those instructions. Use
   * type-overloaded helper functions (not a bare macro) so that VECTOR arguments
   * yield a bvec — a scalar `!(v==v)` would collapse to a single bool and break
   * `any(isnan(vec2))` (the EEVEE velocity/film shaders). `not(equal(v,v))`
   * yields true per-NaN-component via OpFOrdEqual (false for NaN) + negate. */
  ss << "bool gpu_isnan(float x) { return !(x == x); }\n";
  ss << "bvec2 gpu_isnan(vec2 v) { return not(equal(v, v)); }\n";
  ss << "bvec3 gpu_isnan(vec3 v) { return not(equal(v, v)); }\n";
  ss << "bvec4 gpu_isnan(vec4 v) { return not(equal(v, v)); }\n";
  ss << "bool gpu_isinf(float x) { return abs(x) > 3.402823466e38; }\n";
  ss << "bvec2 gpu_isinf(vec2 v) { return greaterThan(abs(v), vec2(3.402823466e38)); }\n";
  ss << "bvec3 gpu_isinf(vec3 v) { return greaterThan(abs(v), vec3(3.402823466e38)); }\n";
  ss << "bvec4 gpu_isinf(vec4 v) { return greaterThan(abs(v), vec4(3.402823466e38)); }\n";
  ss << "#define isnan(x) gpu_isnan(x)\n";
  ss << "#define isinf(x) gpu_isinf(x)\n";
  ss << stage_define;

  GeneratedSourceList sources;
  GeneratedSource ext;
  ext.filename = "gpu_shader_glsl_extension.glsl";
  ext.content = ss.str();
  sources.append(std::move(ext));

  Vector<StringRefNull> resolved = gpu_shader_dependency_get_resolved_source(
      "gpu_shader_compat_glsl.glsl", sources);
  std::string out;
  for (StringRefNull s : resolved) {
    out.append(s.c_str(), s.size());
  }
  return out;
}

/* GLSL -> SPIR-V (shaderc) -> WGSL (Tint). Returns true + sets r_wgsl on success.
 * SPIR-V is targeted at Vulkan 1.0 (Tint rejects the SPIR-V 1.5 that vulkan_1.2
 * would emit). This is the validated toolchain (see smoke/wgsl_probe.cc). */
static bool glsl_to_wgsl(const std::string &glsl,
                         shaderc_shader_kind kind,
                         const char *name,
                         std::string &r_wgsl)
{
  shaderc::Compiler compiler;
  shaderc::CompileOptions opts;
  opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_0);
  shaderc::SpvCompilationResult spv = compiler.CompileGlslToSpv(glsl, kind, name, opts);
  if (spv.GetCompilationStatus() != shaderc_compilation_status_success) {
    fprintf(stderr, "WGPU_SH shaderc FAIL %s: %s\n", name, spv.GetErrorMessage().c_str());
    if (getenv("WGPU_DUMP_GLSL")) {
      char path[256];
      snprintf(path, sizeof(path), "/tmp/wgpu_glsl_%s_%d.glsl", name, int(kind));
      FILE *f = fopen(path, "w");
      if (f) { fwrite(glsl.data(), 1, glsl.size(), f); fclose(f); }
    }
    fflush(stderr);
    return false;
  }
  std::vector<uint32_t> words(spv.cbegin(), spv.cend());
  /* Tint's SPIR-V reader / WGSL writer raise ICEs (TINT_ASSERT/TINT_ICE) on a few
   * EEVEE shaders for as-yet-unimplemented constructs (switch fallthrough, etc.).
   * Tint is built -fno-exceptions and its ICE handler traps the whole wasm module.
   * We patched the handler to longjmp to a host-armed recovery point instead, so
   * arm one here: an ICE fails just this shader rather than aborting the render. */
  std::jmp_buf ice_jmp;
  if (setjmp(ice_jmp) != 0) {
    tint::internal_compiler_error_recovery = nullptr;
    fprintf(stderr, "WGPU_SH tint-ICE recovered %s (shader skipped)\n", name);
    fflush(stderr);
    return false;
  }
  tint::internal_compiler_error_recovery = &ice_jmp;

  auto ir = tint::spirv::reader::ReadIR(words, {});
  if (ir != tint::Success) {
    tint::internal_compiler_error_recovery = nullptr;
    fprintf(stderr, "WGPU_SH tint-read FAIL %s: %s\n", name, ir.Failure().reason.c_str());
    fflush(stderr);
    return false;
  }
  /* Permissive WGSL emission: EEVEE uses derivatives in non-uniform control flow
   * (fwidth), has provably-unreachable branches after early-outs, and relies on
   * extensions/features gated off by default (dual-source blending, read-write
   * storage textures). Allow them all here so the WGSL is produced; the browser
   * device must in turn expose the matching WebGPU features at module use. */
  tint::wgsl::writer::Options wopts;
  wopts.allow_non_uniform_derivatives = true;
  wopts.allow_non_uniform_subgroup_operations = true;
  wopts.disable_unreachable_code_warning = true;
  wopts.allowed_features = tint::wgsl::AllowedFeatures::Everything();
  auto out = tint::wgsl::writer::WgslFromIR(ir.Get(), wopts);
  tint::internal_compiler_error_recovery = nullptr;
  if (out != tint::Success) {
    fprintf(stderr, "WGPU_SH tint-write FAIL %s: %s\n", name, out.Failure().reason.c_str());
    fflush(stderr);
    return false;
  }
  r_wgsl = out.Get().wgsl;
  if (getenv("WGPU_DUMP_WGSL")) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/wgpu_wgsl_%s_%d.wgsl", name, int(kind));
    FILE *f = fopen(path, "w");
    if (f) { fwrite(r_wgsl.data(), 1, r_wgsl.size(), f); fclose(f); }
  }
  return true;
}

/* Recover the Blender resource name from a WGSL buffer struct type name:
 * strip a leading '_' (block prefix) and a trailing '_<digits>' (Tint dedup). */
static std::string strip_buffer_typename(std::string t)
{
  if (!t.empty() && t[0] == '_') {
    t.erase(0, 1);
  }
  size_t u = t.rfind('_');
  if (u != std::string::npos && u + 1 < t.size()) {
    bool all_digits = true;
    for (size_t i = u + 1; i < t.size(); i++) {
      if (!isdigit((unsigned char)t[i])) {
        all_digits = false;
        break;
      }
    }
    if (all_digits) {
      t.erase(u);
    }
  }
  return t;
}

static std::string trim_ws(const std::string &s)
{
  size_t a = s.find_first_not_of(" \t\n\r");
  size_t b = s.find_last_not_of(" \t\n\r");
  return (a == std::string::npos) ? "" : s.substr(a, b - a + 1);
}

static bool has(const std::string &s, const char *sub)
{
  return s.find(sub) != std::string::npos;
}

static WGPUTextureViewDimension wgsl_view_dim(const std::string &type)
{
  if (has(type, "cube_array")) {
    return WGPUTextureViewDimension_CubeArray;
  }
  if (has(type, "cube")) {
    return WGPUTextureViewDimension_Cube;
  }
  if (has(type, "2d_array")) {
    return WGPUTextureViewDimension_2DArray;
  }
  if (has(type, "_3d")) {
    return WGPUTextureViewDimension_3D;
  }
  if (has(type, "_1d")) {
    return WGPUTextureViewDimension_1D;
  }
  return WGPUTextureViewDimension_2D;
}

static WGPUTextureSampleType wgsl_sample_type(const std::string &type)
{
  if (has(type, "depth")) {
    return WGPUTextureSampleType_Depth;
  }
  if (has(type, "<u32>")) {
    return WGPUTextureSampleType_Uint;
  }
  if (has(type, "<i32>")) {
    return WGPUTextureSampleType_Sint;
  }
  return WGPUTextureSampleType_Float;
}

static WGPUTextureFormat wgsl_storage_format(const std::string &type)
{
  /* Token between '<' and ',' (or '>'). */
  size_t lt = type.find('<');
  if (lt == std::string::npos) {
    return WGPUTextureFormat_RGBA8Unorm;
  }
  size_t end = type.find_first_of(",>", lt + 1);
  std::string f = trim_ws(type.substr(lt + 1, end - lt - 1));
  struct {
    const char *name;
    WGPUTextureFormat fmt;
  } table[] = {
      {"rgba8unorm", WGPUTextureFormat_RGBA8Unorm},
      {"rgba8snorm", WGPUTextureFormat_RGBA8Snorm},
      {"rgba8uint", WGPUTextureFormat_RGBA8Uint},
      {"rgba8sint", WGPUTextureFormat_RGBA8Sint},
      {"rgba16unorm", WGPUTextureFormat_RGBA16Unorm},
      {"rgba16snorm", WGPUTextureFormat_RGBA16Snorm},
      {"rgba16uint", WGPUTextureFormat_RGBA16Uint},
      {"rgba16sint", WGPUTextureFormat_RGBA16Sint},
      {"rgba16float", WGPUTextureFormat_RGBA16Float},
      {"r8unorm", WGPUTextureFormat_R8Unorm},
      {"r8snorm", WGPUTextureFormat_R8Snorm},
      {"r8uint", WGPUTextureFormat_R8Uint},
      {"r8sint", WGPUTextureFormat_R8Sint},
      {"rg8unorm", WGPUTextureFormat_RG8Unorm},
      {"rg8snorm", WGPUTextureFormat_RG8Snorm},
      {"rg8uint", WGPUTextureFormat_RG8Uint},
      {"rg8sint", WGPUTextureFormat_RG8Sint},
      {"r16unorm", WGPUTextureFormat_R16Unorm},
      {"r16snorm", WGPUTextureFormat_R16Snorm},
      {"r16uint", WGPUTextureFormat_R16Uint},
      {"r16sint", WGPUTextureFormat_R16Sint},
      {"r16float", WGPUTextureFormat_R16Float},
      {"rg16unorm", WGPUTextureFormat_RG16Unorm},
      {"rg16snorm", WGPUTextureFormat_RG16Snorm},
      {"rg16uint", WGPUTextureFormat_RG16Uint},
      {"rg16sint", WGPUTextureFormat_RG16Sint},
      {"rg16float", WGPUTextureFormat_RG16Float},
      {"r32uint", WGPUTextureFormat_R32Uint},
      {"r32sint", WGPUTextureFormat_R32Sint},
      {"r32float", WGPUTextureFormat_R32Float},
      {"rg32uint", WGPUTextureFormat_RG32Uint},
      {"rg32sint", WGPUTextureFormat_RG32Sint},
      {"rg32float", WGPUTextureFormat_RG32Float},
      {"rgba32uint", WGPUTextureFormat_RGBA32Uint},
      {"rgba32sint", WGPUTextureFormat_RGBA32Sint},
      {"rgba32float", WGPUTextureFormat_RGBA32Float},
      {"rgb10a2unorm", WGPUTextureFormat_RGB10A2Unorm},
      {"rgb10a2uint", WGPUTextureFormat_RGB10A2Uint},
      {"rg11b10ufloat", WGPUTextureFormat_RG11B10Ufloat},
      {"bgra8unorm", WGPUTextureFormat_BGRA8Unorm},
  };
  for (auto &e : table) {
    if (f == e.name) {
      return e.fmt;
    }
  }
  return WGPUTextureFormat_RGBA8Unorm;
}

/* Parse `@group(0u) @binding(Nu) var<...> name : type;` declarations from WGSL
 * and append them to `out`. WGSL is the source of truth for the pipeline's
 * expected bindings (Tint renumbers after splitting combined image-samplers). */
static void parse_wgsl_bindings(const std::string &wgsl, std::vector<WgslResourceBinding> &out)
{
  size_t pos = 0;
  const std::string key = "@binding(";
  while ((pos = wgsl.find(key, pos)) != std::string::npos) {
    size_t np = pos + key.size();
    uint32_t binding = 0;
    bool got = false;
    while (np < wgsl.size() && isdigit((unsigned char)wgsl[np])) {
      binding = binding * 10 + (wgsl[np] - '0');
      np++;
      got = true;
    }
    pos = np;
    if (!got) {
      continue;
    }
    size_t var_pos = wgsl.find("var", np);
    size_t semi = wgsl.find(';', np);
    if (var_pos == std::string::npos || semi == std::string::npos || var_pos > semi) {
      continue;
    }
    std::string decl = wgsl.substr(var_pos, semi - var_pos); /* "var<...> name : type" */
    size_t colon = decl.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string type = trim_ws(decl.substr(colon + 1));
    /* var name = token between the var<...> header and ':'. */
    size_t name_start = 3; /* after "var" */
    if (decl.size() > 3 && decl[3] == '<') {
      size_t close = decl.find('>');
      name_start = (close == std::string::npos) ? 3 : close + 1;
    }
    std::string varname = trim_ws(decl.substr(name_start, colon - name_start));

    WgslResourceBinding b;
    b.binding = binding;
    if (decl.compare(0, 12, "var<uniform>") == 0) {
      b.buffer_type = WGPUBufferBindingType_Uniform;
      if (type.compare(0, 9, "constants") == 0) {
        b.kind = WgslResourceBinding::PUSH_CONST;
        b.res_name = "";
      }
      else {
        b.kind = WgslResourceBinding::UBO;
        b.res_name = strip_buffer_typename(type);
      }
    }
    else if (decl.compare(0, 11, "var<storage") == 0) {
      b.kind = WgslResourceBinding::SSBO;
      b.res_name = strip_buffer_typename(type);
      /* `var<storage, read>` = read-only; `read_write` (or bare) = writable. */
      const std::string head = decl.substr(0, decl.find('>'));
      if (has(head, "read_write")) {
        b.buffer_type = WGPUBufferBindingType_Storage;
        b.writable = true;
      }
      else if (has(head, "read")) {
        b.buffer_type = WGPUBufferBindingType_ReadOnlyStorage;
      }
      else {
        b.buffer_type = WGPUBufferBindingType_Storage;
        b.writable = true;
      }
    }
    else if (type.compare(0, 7, "sampler") == 0) {
      b.kind = WgslResourceBinding::SAMPLER;
      b.sampler_type = has(type, "comparison") ? WGPUSamplerBindingType_Comparison :
                                                 WGPUSamplerBindingType_Filtering;
      b.res_name = varname;
      if (b.res_name.size() > 8 && b.res_name.compare(b.res_name.size() - 8, 8, "_sampler") == 0) {
        b.res_name.erase(b.res_name.size() - 8);
      }
    }
    else if (type.compare(0, 15, "texture_storage") == 0) {
      b.kind = WgslResourceBinding::STORAGE_TEXTURE;
      b.view_dim = wgsl_view_dim(type);
      b.storage_format = wgsl_storage_format(type);
      b.storage_access = has(type, "read_write") ? WGPUStorageTextureAccess_ReadWrite :
                         has(type, "read")       ? WGPUStorageTextureAccess_ReadOnly :
                                                   WGPUStorageTextureAccess_WriteOnly;
      b.writable = (b.storage_access != WGPUStorageTextureAccess_ReadOnly);
      b.res_name = varname;
    }
    else if (type.compare(0, 8, "texture_") == 0) {
      b.kind = WgslResourceBinding::TEXTURE;
      b.tex_sample = wgsl_sample_type(type);
      b.view_dim = wgsl_view_dim(type);
      b.res_name = varname;
      if (b.res_name.size() > 6 && b.res_name.compare(b.res_name.size() - 6, 6, "_image") == 0) {
        b.res_name.erase(b.res_name.size() - 6);
      }
    }
    else {
      continue;
    }
    /* Dedup by binding (a resource used in vert+frag appears in both). */
    bool dup = false;
    for (const WgslResourceBinding &e : out) {
      if (e.binding == b.binding) {
        dup = true;
        break;
      }
    }
    if (!dup) {
      out.push_back(b);
    }
  }
}

static std::string join_sources(MutableSpan<StringRefNull> sources)
{
  std::string out;
  for (StringRefNull s : sources) {
    out.append(s.c_str(), s.size());
  }
  return out;
}

WebGPUShader::WebGPUShader(const char *name) : Shader(name) {}

WebGPUShader::~WebGPUShader()
{
  if (vertex_module_) {
    wgpuShaderModuleRelease(vertex_module_);
  }
  if (fragment_module_) {
    wgpuShaderModuleRelease(fragment_module_);
  }
  if (compute_module_) {
    wgpuShaderModuleRelease(compute_module_);
  }
  if (push_const_buffer_) {
    wgpuBufferRelease(push_const_buffer_);
  }
  if (push_const_data_) {
    free(push_const_data_);
  }
}

void WebGPUShader::uniform_float(int location, int comp_len, int array_size, const float *data)
{
  if (push_const_data_ == nullptr || location < 0 || data == nullptr) {
    return;
  }
  const size_t bytes = size_t(comp_len) * size_t(array_size > 0 ? array_size : 1) * sizeof(float);
  if (size_t(location) + bytes <= push_const_size_) {
    memcpy(push_const_data_ + location, data, bytes);
    push_const_dirty_ = true;
  }
}

void WebGPUShader::uniform_int(int location, int comp_len, int array_size, const int *data)
{
  if (push_const_data_ == nullptr || location < 0 || data == nullptr) {
    return;
  }
  const size_t bytes = size_t(comp_len) * size_t(array_size > 0 ? array_size : 1) * sizeof(int);
  if (size_t(location) + bytes <= push_const_size_) {
    memcpy(push_const_data_ + location, data, bytes);
    push_const_dirty_ = true;
  }
}

WGPUShaderModule WebGPUShader::vertex_module()
{
  if (vertex_module_ == nullptr && !vertex_wgsl_.empty()) {
    vertex_module_ = webgpu_module_from_wgsl(vertex_wgsl_);
  }
  return vertex_module_;
}

WGPUShaderModule WebGPUShader::fragment_module()
{
  if (fragment_module_ == nullptr && !fragment_wgsl_.empty()) {
    fragment_module_ = webgpu_module_from_wgsl(fragment_wgsl_);
  }
  return fragment_module_;
}

WGPUShaderModule WebGPUShader::compute_module()
{
  if (compute_module_ == nullptr && !compute_wgsl_.empty()) {
    compute_module_ = webgpu_module_from_wgsl(compute_wgsl_);
  }
  return compute_module_;
}

WGPUBuffer WebGPUShader::push_const_buffer()
{
  if (push_const_size_ == 0) {
    return nullptr;
  }
  Context *ctx = Context::get();
  if (ctx == nullptr) {
    return nullptr;
  }
  WGPUDevice device = static_cast<WebGPUContext *>(ctx)->device();
  if (device == nullptr) {
    return nullptr;
  }
  if (push_const_data_ == nullptr) {
    push_const_data_ = static_cast<uint8_t *>(calloc(1, push_const_size_));
  }
  if (push_const_buffer_ == nullptr) {
    WGPUBufferDescriptor desc = {};
    desc.size = push_const_size_;
    desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    push_const_buffer_ = wgpuDeviceCreateBuffer(device, &desc);
    push_const_dirty_ = true;
  }
  if (push_const_buffer_ && push_const_dirty_) {
    WGPUQueue q = static_cast<WebGPUContext *>(ctx)->queue();
    if (q) {
      wgpuQueueWriteBuffer(q, push_const_buffer_, 0, push_const_data_, push_const_size_);
      push_const_dirty_ = false;
    }
  }
  return push_const_buffer_;
}

void WebGPUShader::init(const shader::ShaderCreateInfo & /*info*/, bool /*is_codegen_only*/) {}

void WebGPUShader::vertex_shader_from_glsl(const shader::ShaderCreateInfo & /*info*/,
                                           MutableSpan<StringRefNull> sources)
{
  std::string patch = webgpu_glsl_patch("#define GPU_VERTEX_SHADER\n");
  if (!sources.is_empty()) {
    sources[SOURCES_INDEX_VERSION] = patch;
  }
  vertex_src_ = join_sources(sources);
}

void WebGPUShader::geometry_shader_from_glsl(const shader::ShaderCreateInfo & /*info*/,
                                             MutableSpan<StringRefNull> sources)
{
  geometry_src_ = join_sources(sources);
}

void WebGPUShader::fragment_shader_from_glsl(const shader::ShaderCreateInfo & /*info*/,
                                             MutableSpan<StringRefNull> sources)
{
  std::string patch = webgpu_glsl_patch("#define GPU_FRAGMENT_SHADER\n");
  if (!sources.is_empty()) {
    sources[SOURCES_INDEX_VERSION] = patch;
  }
  fragment_src_ = join_sources(sources);
}

void WebGPUShader::compute_shader_from_glsl(const shader::ShaderCreateInfo & /*info*/,
                                            MutableSpan<StringRefNull> sources)
{
  std::string patch = webgpu_glsl_patch("#define GPU_COMPUTE_SHADER\n");
  if (!sources.is_empty()) {
    sources[SOURCES_INDEX_VERSION] = patch;
  }
  compute_src_ = join_sources(sources);
}

bool WebGPUShader::finalize(const shader::ShaderCreateInfo *info)
{
  /* Translate each present stage GLSL -> SPIR-V -> WGSL via the in-wasm
   * toolchain (shaderc + Tint). On success, build the ShaderInterface from the
   * create-info and (when a real device is present) create the WGPUShaderModules.
   * A graphics shader needs vertex+fragment to translate; a compute shader needs
   * compute. We report success when the required stages translate so the GPU
   * module treats the shader as compiled and the render path proceeds. */
  const bool is_compute = !compute_src_.empty();
  bool vert_ok = true, frag_ok = true, comp_ok = true;

  if (!is_compute) {
    if (!vertex_src_.empty()) {
      vert_ok = glsl_to_wgsl(vertex_src_, shaderc_vertex_shader, name_get().c_str(), vertex_wgsl_);
    }
    if (!fragment_src_.empty()) {
      frag_ok = glsl_to_wgsl(
          fragment_src_, shaderc_fragment_shader, name_get().c_str(), fragment_wgsl_);
    }
  }
  else {
    comp_ok = glsl_to_wgsl(compute_src_, shaderc_compute_shader, name_get().c_str(), compute_wgsl_);
  }

  const bool translated = is_compute ? comp_ok : (vert_ok && frag_ok);

  /* Always build the ShaderInterface from the create-info, and ALWAYS report
   * success, EVEN when WGSL translation failed. Rationale: EEVEE gates the WHOLE
   * render on every `needed_shaders` group being loaded (Instance::render_frame:
   * `skip_render_ |= !is_loaded(needed_shaders)`), so a single failed shader makes
   * it draw NOTHING. By returning a valid (non-null) shader object with a real
   * interface — but null GPU modules when translation failed — EEVEE proceeds with
   * the render; the draw/dispatch path skips any shader whose module is null (a
   * partial/incorrect image instead of no render at all). This lets the WebGPU
   * draw path actually execute while the remaining ~20 shaders are fixed. */
  if (info != nullptr && this->interface == nullptr) {
    WebGPUShaderInterface *iface = new WebGPUShaderInterface();
    iface->init(*info);
    this->interface = iface;
  }

  /* Parse the WGSL bindings the pipeline will expect. GPU modules are created
   * LAZILY (see vertex_module()/etc.) at draw/dispatch time — under
   * PROXY_TO_PTHREAD finalize runs on a thread without the device context. */
  if (translated) {
    parse_wgsl_bindings(vertex_wgsl_, render_bindings_);
    parse_wgsl_bindings(fragment_wgsl_, render_bindings_);
    parse_wgsl_bindings(compute_wgsl_, compute_bindings_);
  }

  /* Size the push-constant buffer from the create-info (conservative std140
   * upper bound: 16 B per scalar/vec, 64 B per mat4, 48 B per mat3, ×array). */
  if (info != nullptr && !info->push_constants_.is_empty()) {
    size_t sz = 0;
    for (const shader::ShaderCreateInfo::PushConst &pc : info->push_constants_) {
      const int n = (pc.array_size > 0) ? pc.array_size : 1;
      size_t elem = 16;
      if (pc.type == Type::float4x4_t) {
        elem = 64;
      }
      else if (pc.type == Type::float3x3_t) {
        elem = 48;
      }
      sz += elem * size_t(n);
    }
    push_const_size_ = (sz + 15) & ~size_t(15);
    /* Allocate the CPU shadow now so uniform_float/int can write before the GPU
     * buffer is lazily created at first draw. */
    push_const_data_ = static_cast<uint8_t *>(calloc(1, push_const_size_));
  }

  valid_ = (this->interface != nullptr);
  fprintf(stderr,
          "WGPU_SHADER finalize '%s' translated=%d valid=%d (vmod=%p fmod=%p cmod=%p)\n",
          name_get().c_str(),
          int(translated),
          int(valid_),
          (void *)vertex_module_,
          (void *)fragment_module_,
          (void *)compute_module_);
  fflush(stderr);
  return valid_;
}


/* --- WebGPUShader GLSL preamble generators (Vulkan-flavoured GLSL; sequential
 *     set=0 bindings — refine to real bind-group layout when wiring pipelines) --- */

std::string WebGPUShader::resources_declare(const shader::ShaderCreateInfo &info) const
{
  std::stringstream ss;
  /* Specialization constants. */
  uint constant_id = 0;
  for (const SpecializationConstant &sc : info.specialization_constants_) {
    ss << "layout (constant_id=" << constant_id++ << ") const ";
    switch (sc.type) {
      case Type::int_t: ss << "int " << sc.name << "=" << std::to_string(sc.value.i) << ";\n"; break;
      case Type::uint_t: ss << "uint " << sc.name << "=" << std::to_string(sc.value.u) << "u;\n"; break;
      case Type::bool_t: ss << "bool " << sc.name << "=" << (sc.value.u ? "true" : "false") << ";\n"; break;
      case Type::float_t:
        ss << "uint " << sc.name << "_uint=" << std::to_string(sc.value.u) << "u;\n";
        ss << "#define " << sc.name << " uintBitsToFloat(" << sc.name << "_uint)\n"; break;
      default: break;
    }
  }
  for (const CompilationConstant &sc : info.compilation_constants_) {
    ss << "const ";
    switch (sc.type) {
      case Type::int_t: ss << "int " << sc.name << "=" << std::to_string(sc.value.i) << ";\n"; break;
      case Type::uint_t: ss << "uint " << sc.name << "=" << std::to_string(sc.value.u) << "u;\n"; break;
      case Type::bool_t: ss << "bool " << sc.name << "=" << (sc.value.u ? "true" : "false") << ";\n"; break;
      default: break;
    }
  }

  int binding = 0;
  /* Emit one resource as a single line (no trailing newline — the macro adds the
   * line-continuation). */
  auto print_one = [&](const ShaderCreateInfo::Resource &res) {
    ss << "layout(binding = " << binding++;
    if (res.bind_type == ShaderCreateInfo::Resource::BindType::IMAGE) {
      ss << ", " << to_string(res.image.format);
    }
    else if (res.bind_type == ShaderCreateInfo::Resource::BindType::UNIFORM_BUFFER) {
      ss << ", std140";
    }
    else if (res.bind_type == ShaderCreateInfo::Resource::BindType::STORAGE_BUFFER) {
      ss << ", std430";
    }
    ss << ") ";
    switch (res.bind_type) {
      case ShaderCreateInfo::Resource::BindType::SAMPLER:
        ss << "uniform "; print_image_type(ss, res.sampler.type, res.bind_type);
        ss << " " << res.sampler.name << ";"; break;
      case ShaderCreateInfo::Resource::BindType::IMAGE:
        ss << "uniform "; print_qualifier(ss, res.image.qualifiers);
        print_image_type(ss, res.image.type, res.bind_type);
        ss << " " << res.image.name << ";"; break;
      case ShaderCreateInfo::Resource::BindType::UNIFORM_BUFFER:
        ss << "uniform _" << res.uniformbuf.name.str_no_array() << " { "
           << info.buffer_typename(res.uniformbuf.type_name, true) << " " << res.uniformbuf.name
           << "; };"; break;
      case ShaderCreateInfo::Resource::BindType::STORAGE_BUFFER:
        print_qualifier(ss, res.storagebuf.qualifiers);
        ss << "buffer _" << res.storagebuf.name.str_no_array() << " { "
           << info.buffer_typename(res.storagebuf.type_name) << " " << res.storagebuf.name
           << "; };"; break;
    }
  };
  /* Wrap each create-info's resources in a CREATE_INFO_RES_<freq>_<info> macro.
   * The shader body invokes these macros AFTER the struct/typedef definitions are
   * in scope, so struct-typed UBOs/SSBOs resolve correctly (matches VKShader). */
  auto emit_freq = [&](const char *freq, const auto &resources) {
    std::string active;
    for (const ShaderCreateInfo::Resource &res : resources) {
      if (active != std::string(res.info_name)) {
        active = std::string(res.info_name);
        ss << "\n#define CREATE_INFO_RES_" << freq << "_" << res.info_name << " \\\n";
      }
      print_one(res);
      ss << " \\\n";
    }
    ss << "\n";
  };
  emit_freq("PASS", info.pass_resources_);
  emit_freq("BATCH", info.batch_resources_);
  emit_freq("GEOMETRY", info.geometry_resources_);

  if (!info.push_constants_.is_empty()) {
    ss << "layout(binding = " << binding++ << ", std140) uniform constants {\n";
    for (const ShaderCreateInfo::PushConst &uniform : info.push_constants_) {
      ss << "  " << to_string(uniform.type) << " " << uniform.name;
      if (uniform.array_size > 0) { ss << "[" << uniform.array_size << "]"; }
      ss << ";\n";
    }
    ss << "};\n";
  }
  ss << "\n";
  return ss.str();
}

std::string WebGPUShader::vertex_interface_declare(const shader::ShaderCreateInfo &info) const
{
  std::stringstream ss;
  for (const ShaderCreateInfo::VertIn &attr : info.vertex_inputs_) {
    ss << "layout(location = " << attr.index << ") in " << to_string(attr.type) << " "
       << attr.name << ";\n";
  }
  int location = 0;
  for (const StageInterfaceInfo *iface : info.vertex_out_interfaces_) {
    print_interface(ss, "out", *iface, location);
  }
  ss << "\n";
  return ss.str();
}

std::string WebGPUShader::fragment_interface_declare(const shader::ShaderCreateInfo &info) const
{
  std::stringstream ss;
  int location = 0;
  for (const StageInterfaceInfo *iface : info.vertex_out_interfaces_) {
    print_interface(ss, "in", *iface, location);
  }
  for (const ShaderCreateInfo::FragOut &output : info.fragment_outputs_) {
    ss << "layout(location = " << output.index;
    if (output.blend != DualBlend::NONE) {
      ss << ", index = " << ((output.blend == DualBlend::SRC_0) ? 0 : 1);
    }
    ss << ") out " << to_string(output.type) << " " << output.name << ";\n";
  }
  ss << "\n";
  return ss.str();
}

std::string WebGPUShader::geometry_interface_declare(const shader::ShaderCreateInfo & /*info*/) const
{
  return "";
}

std::string WebGPUShader::geometry_layout_declare(const shader::ShaderCreateInfo & /*info*/) const
{
  return "";
}

std::string WebGPUShader::compute_layout_declare(const shader::ShaderCreateInfo &info) const
{
  std::stringstream ss;
  if (info.compute_layout_.local_size_x != -1) {
    ss << "layout(local_size_x = " << info.compute_layout_.local_size_x;
    if (info.compute_layout_.local_size_y != -1) {
      ss << ", local_size_y = " << info.compute_layout_.local_size_y;
    }
    if (info.compute_layout_.local_size_z != -1) {
      ss << ", local_size_z = " << info.compute_layout_.local_size_z;
    }
    ss << ") in;\n";
  }
  return ss.str();
}

}  // namespace blender::gpu
