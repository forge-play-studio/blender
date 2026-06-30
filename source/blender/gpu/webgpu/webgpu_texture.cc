/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 */

#include <cstring>

#include "webgpu_context.hh"
#include "webgpu_texture.hh"

namespace blender::gpu {

static WebGPUContext *ctx_get()
{
  return static_cast<WebGPUContext *>(Context::get());
}

/* Map Blender's logical texture format to the closest WebGPU format. Covers the
 * formats EEVEE/Workbench commonly use; unmapped formats fall back to RGBA8 so
 * texture creation never hard-fails (correctness for exotic formats TBD). */
WGPUTextureFormat webgpu_texture_format(TextureFormat format)
{
  switch (format) {
    case TextureFormat::UNORM_8_8_8_8:
      return WGPUTextureFormat_RGBA8Unorm;
    case TextureFormat::SRGBA_8_8_8_8:
      return WGPUTextureFormat_RGBA8UnormSrgb;
    case TextureFormat::UNORM_8:
      return WGPUTextureFormat_R8Unorm;
    case TextureFormat::UNORM_8_8:
      return WGPUTextureFormat_RG8Unorm;
    case TextureFormat::UINT_8:
      return WGPUTextureFormat_R8Uint;
    case TextureFormat::SFLOAT_16:
      return WGPUTextureFormat_R16Float;
    case TextureFormat::SFLOAT_16_16:
      return WGPUTextureFormat_RG16Float;
    case TextureFormat::SFLOAT_16_16_16_16:
      return WGPUTextureFormat_RGBA16Float;
    case TextureFormat::SFLOAT_32:
      return WGPUTextureFormat_R32Float;
    case TextureFormat::SFLOAT_32_32:
      return WGPUTextureFormat_RG32Float;
    case TextureFormat::SFLOAT_32_32_32_32:
      return WGPUTextureFormat_RGBA32Float;
    case TextureFormat::UINT_32:
      return WGPUTextureFormat_R32Uint;
    case TextureFormat::UINT_32_32:
      return WGPUTextureFormat_RG32Uint;
    case TextureFormat::UINT_32_32_32_32:
      return WGPUTextureFormat_RGBA32Uint;
    case TextureFormat::SINT_32:
      return WGPUTextureFormat_R32Sint;
    case TextureFormat::UINT_16:
      return WGPUTextureFormat_R16Uint;
    case TextureFormat::UINT_16_16_16_16:
      return WGPUTextureFormat_RGBA16Uint;
    case TextureFormat::UNORM_10_10_10_2:
      return WGPUTextureFormat_RGB10A2Unorm;
    case TextureFormat::UFLOAT_11_11_10:
      return WGPUTextureFormat_RG11B10Ufloat;
    case TextureFormat::UNORM_16_DEPTH:
      return WGPUTextureFormat_Depth16Unorm;
    case TextureFormat::SFLOAT_32_DEPTH:
      return WGPUTextureFormat_Depth32Float;
    case TextureFormat::SFLOAT_32_DEPTH_UINT_8:
      return WGPUTextureFormat_Depth32FloatStencil8;
    default:
      return WGPUTextureFormat_RGBA8Unorm;
  }
}

/* WebGPU only permits STORAGE_BINDING on a fixed set of formats (without optional
 * features). EEVEE liberally tags textures SHADER_WRITE in formats that are not
 * storage-capable in WebGPU (srgb, R16Float, RG11B10Ufloat, 8/16-bit norm, depth,
 * rgb10a2). Adding StorageBinding to those makes the whole texture invalid, which
 * cascades. Gate the flag so the texture is at least created as sampled/attachment.
 * (Correct EEVEE behaviour will need format substitution — tracked separately.) */
static bool webgpu_format_supports_storage(WGPUTextureFormat f)
{
  switch (f) {
    case WGPUTextureFormat_R32Uint:
    case WGPUTextureFormat_R32Sint:
    case WGPUTextureFormat_R32Float:
    case WGPUTextureFormat_RG32Uint:
    case WGPUTextureFormat_RG32Sint:
    case WGPUTextureFormat_RG32Float:
    case WGPUTextureFormat_RGBA8Unorm:
    case WGPUTextureFormat_RGBA8Snorm:
    case WGPUTextureFormat_RGBA8Uint:
    case WGPUTextureFormat_RGBA8Sint:
    case WGPUTextureFormat_RGBA16Uint:
    case WGPUTextureFormat_RGBA16Sint:
    case WGPUTextureFormat_RGBA16Float:
    case WGPUTextureFormat_RGBA32Uint:
    case WGPUTextureFormat_RGBA32Sint:
    case WGPUTextureFormat_RGBA32Float:
      return true;
    default:
      return false;
  }
}

static WGPUTextureUsage webgpu_texture_usage(eGPUTextureUsage usage, WGPUTextureFormat format)
{
  WGPUTextureUsage flags = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
  if (usage & GPU_TEXTURE_USAGE_SHADER_READ) {
    flags |= WGPUTextureUsage_TextureBinding;
  }
  if ((usage & GPU_TEXTURE_USAGE_SHADER_WRITE) && webgpu_format_supports_storage(format)) {
    flags |= WGPUTextureUsage_StorageBinding;
  }
  if (usage & GPU_TEXTURE_USAGE_ATTACHMENT) {
    flags |= WGPUTextureUsage_RenderAttachment;
  }
  return flags;
}

/* Bytes per texel for the WebGPU formats produced by webgpu_texture_format().
 * Used to size queue writes (bytesPerRow). Defaults to 4 for unmapped formats. */
uint32_t webgpu_format_bytes_per_pixel(WGPUTextureFormat f)
{
  switch (f) {
    case WGPUTextureFormat_R8Unorm:
    case WGPUTextureFormat_R8Uint:
      return 1;
    case WGPUTextureFormat_RG8Unorm:
    case WGPUTextureFormat_R16Float:
    case WGPUTextureFormat_R16Uint:
    case WGPUTextureFormat_Depth16Unorm:
      return 2;
    case WGPUTextureFormat_RGBA8Unorm:
    case WGPUTextureFormat_RGBA8UnormSrgb:
    case WGPUTextureFormat_RG16Float:
    case WGPUTextureFormat_R32Float:
    case WGPUTextureFormat_R32Uint:
    case WGPUTextureFormat_R32Sint:
    case WGPUTextureFormat_RGB10A2Unorm:
    case WGPUTextureFormat_RG11B10Ufloat:
    case WGPUTextureFormat_Depth32Float:
      return 4;
    case WGPUTextureFormat_RGBA16Float:
    case WGPUTextureFormat_RGBA16Uint:
    case WGPUTextureFormat_RG32Float:
    case WGPUTextureFormat_RG32Uint:
      return 8;
    case WGPUTextureFormat_RGBA32Float:
    case WGPUTextureFormat_RGBA32Uint:
      return 16;
    default:
      return 4;
  }
}

WebGPUTexture::WebGPUTexture(const char *name) : Texture(name) {}

WebGPUTexture::~WebGPUTexture()
{
  if (view_) {
    wgpuTextureViewRelease(view_);
  }
  if (texture_) {
    wgpuTextureDestroy(texture_);
    wgpuTextureRelease(texture_);
  }
}

bool WebGPUTexture::init_internal()
{
  wgpu_format_ = webgpu_texture_format(format_);

  WebGPUContext *ctx = ctx_get();
  if (ctx == nullptr || ctx->device() == nullptr) {
    /* Device-less (headless diagnostics): metadata-only, succeed so the higher
     * layers proceed. Real GPU resource is created when a device is present. */
    return true;
  }

  WGPUTextureDescriptor desc = {};
  desc.dimension = (type_ & GPU_TEXTURE_3D) ? WGPUTextureDimension_3D :
                   (type_ & GPU_TEXTURE_1D) ? WGPUTextureDimension_1D :
                                              WGPUTextureDimension_2D;
  desc.size.width = std::max(w_, 1);
  desc.size.height = (desc.dimension == WGPUTextureDimension_1D) ? 1 : std::max(h_, 1);
  /* depthOrArrayLayers: 3D depth, or array layer count, min 1. */
  desc.size.depthOrArrayLayers = std::max(d_, 1);
  desc.format = wgpu_format_;
  desc.mipLevelCount = std::max(mipmaps_, 1);
  desc.sampleCount = 1;
  desc.usage = webgpu_texture_usage(gpu_image_usage_flags_, wgpu_format_);

  texture_ = wgpuDeviceCreateTexture(ctx->device(), &desc);
  if (texture_ == nullptr) {
    return false;
  }
  WGPUTextureViewDescriptor vdesc = {};
  vdesc.format = wgpu_format_;
  vdesc.dimension = (desc.dimension == WGPUTextureDimension_3D) ? WGPUTextureViewDimension_3D :
                    (desc.dimension == WGPUTextureDimension_1D) ? WGPUTextureViewDimension_1D :
                    (type_ & GPU_TEXTURE_CUBE)                  ? WGPUTextureViewDimension_Cube :
                    (type_ & GPU_TEXTURE_ARRAY) ? WGPUTextureViewDimension_2DArray :
                                                  WGPUTextureViewDimension_2D;
  vdesc.baseMipLevel = 0;
  vdesc.mipLevelCount = desc.mipLevelCount;
  vdesc.baseArrayLayer = 0;
  vdesc.arrayLayerCount = (desc.dimension == WGPUTextureDimension_2D) ?
                              desc.size.depthOrArrayLayers :
                              1;
  view_ = wgpuTextureCreateView(texture_, &vdesc);
  return true;
}

bool WebGPUTexture::init_internal(VertBuf * /*vbo*/)
{
  /* Texture-from-buffer not implemented yet. */
  wgpu_format_ = webgpu_texture_format(format_);
  return true;
}

bool WebGPUTexture::init_internal(gpu::Texture * /*src*/,
                                  int /*mip_offset*/,
                                  int /*layer_offset*/,
                                  bool /*use_stencil*/)
{
  /* Texture views onto another texture not implemented yet. */
  wgpu_format_ = webgpu_texture_format(format_);
  return true;
}

void WebGPUTexture::update_sub(int mip,
                               int offset[3],
                               int extent[3],
                               eGPUDataFormat /*format*/,
                               const void *data,
                               uint /*unpack_row_length*/)
{
  WebGPUContext *ctx = ctx_get();
  if (!texture_ || !ctx || !ctx->queue() || !data) {
    return;
  }
  WGPUTexelCopyTextureInfo dst = {};
  dst.texture = texture_;
  dst.mipLevel = mip;
  dst.origin = {uint32_t(offset[0]), uint32_t(offset[1]), uint32_t(offset[2])};
  const uint32_t bpp = webgpu_format_bytes_per_pixel(wgpu_format_);
  WGPUTexelCopyBufferLayout layout = {};
  layout.offset = 0;
  layout.bytesPerRow = uint32_t(extent[0]) * bpp;
  layout.rowsPerImage = uint32_t(extent[1]);
  WGPUExtent3D wext = {uint32_t(extent[0]), uint32_t(extent[1]), uint32_t(std::max(extent[2], 1))};
  const size_t byte_size = size_t(layout.bytesPerRow) * size_t(extent[1]) *
                           size_t(std::max(extent[2], 1));
  wgpuQueueWriteTexture(ctx->queue(), &dst, data, byte_size, &layout, &wext);
}

void WebGPUTexture::copy_to(Texture * /*dst*/, IndexRange /*mip_levels*/)
{
  /* GPU texture-to-texture copy: TODO (command encoder CopyTextureToTexture). */
}

void WebGPUTexture::clear(const double4 /*data*/)
{
  /* Clear via a render pass / clear-buffer write: TODO. */
}

void WebGPUTexture::read(int /*mip*/, eGPUDataFormat /*format*/, void * /*dst*/)
{
  /* GPU→host readback needs an async buffer map; not wired yet. Caller owns the
   * destination buffer; we don't know the per-format byte size here, so do
   * nothing rather than risk overflowing dst. */
}

}  // namespace blender::gpu
