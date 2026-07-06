const ChainedStruct = @import("chained_struct.zig").ChainedStruct;

const _misc = @import("misc.zig");
const WGPUFlags = _misc.WGPUFlags;
const WGPUBool = _misc.WGPUBool;
const StringView = _misc.StringView;
const U32_MAX = _misc.U32_MAX;

pub const WGPU_ARRAY_LAYER_COUNT_UNDEFINED = U32_MAX;
pub const WGPU_MIP_LEVEL_COUNT_UNDEFINED = U32_MAX;
pub const WGPU_COPY_STRIDE_UNDEFINED = U32_MAX;

const Buffer = @import("buffer.zig").Buffer;

// TextureFormat values match wgpu-native 29.x / webgpu.h (NOT the old wgpu 0.19 values).
// wgpu 29.x promoted r16_unorm/snorm, rg16_unorm/snorm, rgba16_unorm/snorm from
// native extensions into the core enum, shifting all subsequent values by +4 or more.
pub const TextureFormat = enum(u32) {
    undefined = 0x00000000,
    r8_unorm = 0x00000001,
    r8_snorm = 0x00000002,
    r8_uint = 0x00000003,
    r8_sint = 0x00000004,
    r16_unorm = 0x00000005,
    r16_snorm = 0x00000006,
    r16_uint = 0x00000007,
    r16_sint = 0x00000008,
    r16_float = 0x00000009,
    rg8_unorm = 0x0000000A,
    rg8_snorm = 0x0000000B,
    rg8_uint = 0x0000000C,
    rg8_sint = 0x0000000D,
    r32_float = 0x0000000E,
    r32_uint = 0x0000000F,
    r32_sint = 0x00000010,
    rg16_unorm = 0x00000011,
    rg16_snorm = 0x00000012,
    rg16_uint = 0x00000013,
    rg16_sint = 0x00000014,
    rg16_float = 0x00000015,
    rgba8_unorm = 0x00000016,
    rgba8_unorm_srgb = 0x00000017,
    rgba8_snorm = 0x00000018,
    rgba8_uint = 0x00000019,
    rgba8_sint = 0x0000001A,
    bgra8_unorm = 0x0000001B,
    bgra8_unorm_srgb = 0x0000001C,
    rgb10a2_uint = 0x0000001D,
    rgb10a2_unorm = 0x0000001E,
    rg11b10_ufloat = 0x0000001F,
    rgb9e5_ufloat = 0x00000020,
    rg32_float = 0x00000021,
    rg32_uint = 0x00000022,
    rg32_sint = 0x00000023,
    rgba16_unorm = 0x00000024,
    rgba16_snorm = 0x00000025,
    rgba16_uint = 0x00000026,
    rgba16_sint = 0x00000027,
    rgba16_float = 0x00000028,
    rgba32_float = 0x00000029,
    rgba32_uint = 0x0000002A,
    rgba32_sint = 0x0000002B,
    stencil8 = 0x0000002C,
    depth16_unorm = 0x0000002D,
    depth24_plus = 0x0000002E,
    depth24_plus_stencil8 = 0x0000002F,
    depth32_float = 0x00000030,
    depth32_float_stencil8 = 0x00000031,
    bc1_rgba_unorm = 0x00000032,
    bc1_rgba_unorm_srgb = 0x00000033,
    bc2_rgba_unorm = 0x00000034,
    bc2_rgba_unorm_srgb = 0x00000035,
    bc3_rgba_unorm = 0x00000036,
    bc3_rgba_unorm_srgb = 0x00000037,
    bc4_r_unorm = 0x00000038,
    bc4_r_snorm = 0x00000039,
    bc5_rg_unorm = 0x0000003A,
    bc5_rg_snorm = 0x0000003B,
    bc6_hrgb_ufloat = 0x0000003C,
    bc6_hrgb_float = 0x0000003D,
    bc7_rgba_unorm = 0x0000003E,
    bc7_rgba_unorm_srgb = 0x0000003F,
    etc2_rgb8_unorm = 0x00000040,
    etc2_rgb8_unorm_srgb = 0x00000041,
    etc2_rgb8a1_unorm = 0x00000042,
    etc2_rgb8a1_unorm_srgb = 0x00000043,
    etc2_rgba8_unorm = 0x00000044,
    etc2_rgba8_unorm_srgb = 0x00000045,
    eacr11_unorm = 0x00000046,
    eacr11_snorm = 0x00000047,
    eacrg11_unorm = 0x00000048,
    eacrg11_snorm = 0x00000049,
    astc4x4_unorm = 0x0000004A,
    astc4x4_unorm_srgb = 0x0000004B,
    astc5x4_unorm = 0x0000004C,
    astc5x4_unorm_srgb = 0x0000004D,
    astc5x5_unorm = 0x0000004E,
    astc5x5_unorm_srgb = 0x0000004F,
    astc6x5_unorm = 0x00000050,
    astc6x5_unorm_srgb = 0x00000051,
    astc6x6_unorm = 0x00000052,
    astc6x6_unorm_srgb = 0x00000053,
    astc8x5_unorm = 0x00000054,
    astc8x5_unorm_srgb = 0x00000055,
    astc8x6_unorm = 0x00000056,
    astc8x6_unorm_srgb = 0x00000057,
    astc8x8_unorm = 0x00000058,
    astc8x8_unorm_srgb = 0x00000059,
    astc10x5_unorm = 0x0000005A,
    astc10x5_unorm_srgb = 0x0000005B,
    astc10x6_unorm = 0x0000005C,
    astc10x6_unorm_srgb = 0x0000005D,
    astc10x8_unorm = 0x0000005E,
    astc10x8_unorm_srgb = 0x0000005F,
    astc10x10_unorm = 0x00000060,
    astc10x10_unorm_srgb = 0x00000061,
    astc12x10_unorm = 0x00000062,
    astc12x10_unorm_srgb = 0x00000063,
    astc12x12_unorm = 0x00000064,
    astc12x12_unorm_srgb = 0x00000065,
};

pub const TextureUsage = WGPUFlags;
pub const TextureUsages = struct {
    pub const none = @as(TextureUsage, 0x0000000000000000);
    pub const copy_src = @as(TextureUsage, 0x0000000000000001);
    pub const copy_dst = @as(TextureUsage, 0x0000000000000002);
    pub const texture_binding = @as(TextureUsage, 0x0000000000000004);
    pub const storage_binding = @as(TextureUsage, 0x0000000000000008);
    pub const render_attachment = @as(TextureUsage, 0x0000000000000010);
};

// TODO: Like a lot of things in this file, this breaks from the wrapper code convention by having an unneeded prefix ("Texture")
//       in front of the name, even though "Aspect" is exclusively used in TextureAspect. I've done this because just calling
//       it "Aspect" seems like it'd confuse people thinking it is an aspect ratio or something, but should it just be "Aspect"?
pub const TextureAspect = enum(u32) {
    undefined = 0x00000000, // Indicates no value is passed for this argument.
    all = 0x00000001,
    stencil_only = 0x00000002,
    depth_only = 0x00000003,
};

pub const TextureViewDescriptor = extern struct {
    next_in_chain: ?*const ChainedStruct = null,
    label: StringView = StringView{},
    format: TextureFormat = TextureFormat.undefined,
    dimension: ViewDimension = ViewDimension.undefined,
    base_mip_level: u32 = 0,
    mip_level_count: u32 = WGPU_MIP_LEVEL_COUNT_UNDEFINED,
    base_array_layer: u32 = 0,
    array_layer_count: u32 = WGPU_ARRAY_LAYER_COUNT_UNDEFINED,
    aspect: TextureAspect = TextureAspect.all,
    usage: TextureUsage = TextureUsages.none,
};

pub const TextureViewProcs = struct {
    pub const SetLabel = *const fn (*TextureView, StringView) callconv(.c) void;
    pub const AddRef = *const fn (*TextureView) callconv(.c) void;
    pub const Release = *const fn (*TextureView) callconv(.c) void;
};

extern fn wgpuTextureViewSetLabel(texture_view: *TextureView, label: StringView) void;
extern fn wgpuTextureViewAddRef(texture_view: *TextureView) void;
extern fn wgpuTextureViewRelease(texture_view: *TextureView) void;

pub const TextureView = opaque {
    // Unimplemented as of wgpu-native v25.0.2.1,
    // see https://github.com/gfx-rs/wgpu-native/blob/d8238888998db26ceab41942f269da0fa32b890c/src/unimplemented.rs#L216
    // pub inline fn setLabel(self: *TextureView, label: []const u8) void {
    //     wgpuTextureViewSetLabel(self, StringView.fromSlice(label));
    // }

    pub inline fn addRef(self: *TextureView) void {
        wgpuTextureViewAddRef(self);
    }
    pub inline fn release(self: *TextureView) void {
        wgpuTextureViewRelease(self);
    }
};

// TODO: Should this maybe go in sampler.zig instead?
pub const SampleType = enum(u32) {
    // Indicates that this TextureBindingLayout member of its parent BindGroupLayoutEntry is not used.
    binding_not_used = 0x00000000,

    // Indicates no value is passed for this argument.
    undefined = 0x00000001,

    float = 0x00000002,
    unfilterable_float = 0x00000003,
    depth = 0x00000004,
    s_int = 0x00000005,
    u_int = 0x00000006,
};

pub const ViewDimension = enum(u32) {
    undefined = 0x00000000, // Indicates no value is passed for this argument.
    @"1d" = 0x00000001,
    @"2d" = 0x00000002,
    @"2d_array" = 0x00000003,
    cube = 0x00000004,
    cube_array = 0x00000005,
    @"3d" = 0x00000006,
};

pub const TextureBindingLayout = extern struct {
    next_in_chain: ?*const ChainedStruct = null,
    sample_type: SampleType = SampleType.undefined,
    view_dimension: ViewDimension = ViewDimension.undefined,
    multisampled: WGPUBool = @intFromBool(false),
};

pub const StorageTextureAccess = enum(u32) {
    // Indicates that this StorageTextureBindingLayout member of its parent BindGroupLayoutEntry is not used.
    binding_not_used = 0x00000000,

    // Indicates no value is passed for this argument.
    undefined = 0x00000001,

    write_only = 0x00000002,
    read_only = 0x00000003,
    read_write = 0x00000004,
};

pub const StorageTextureBindingLayout = extern struct {
    next_in_chain: ?*const ChainedStruct = null,
    access: StorageTextureAccess = StorageTextureAccess.undefined,
    format: TextureFormat = TextureFormat.undefined,
    view_dimension: ViewDimension = ViewDimension.undefined,
};

pub const TextureDimension = enum(u32) {
    undefined = 0x00000000, // Indicates no value is passed for this argument.
    @"1d" = 0x00000001,
    @"2d" = 0x00000002,
    @"3d" = 0x00000003,
};

pub const Extent3D = extern struct {
    width: u32 = 1,
    height: u32 = 1,
    depth_or_array_layers: u32 = 1,
};

pub const TextureDescriptor = extern struct {
    next_in_chain: ?*const ChainedStruct = null,
    label: StringView = StringView{},
    usage: TextureUsage,
    dimension: TextureDimension = TextureDimension.@"2d",
    size: Extent3D,
    format: TextureFormat,
    mip_level_count: u32 = 1,
    sample_count: u32 = 1,
    view_format_count: usize = 0,
    view_formats: [*]const TextureFormat = &[_]TextureFormat{},
};

pub const TextureProcs = struct {
    pub const CreateView = *const fn (*Texture, ?*const TextureViewDescriptor) callconv(.c) ?*TextureView;
    pub const Destroy = *const fn (*Texture) callconv(.c) void;
    pub const GetDepthOrArrayLayers = *const fn (*Texture) callconv(.c) u32;
    pub const GetDimension = *const fn (*Texture) callconv(.c) TextureDimension;
    pub const GetFormat = *const fn (*Texture) callconv(.c) TextureFormat;
    pub const GetHeight = *const fn (*Texture) callconv(.c) u32;
    pub const GetMipLevelCount = *const fn (*Texture) callconv(.c) u32;
    pub const GetSampleCount = *const fn (*Texture) callconv(.c) u32;
    pub const GetUsage = *const fn (*Texture) callconv(.c) TextureUsage;
    pub const GetWidth = *const fn (*Texture) callconv(.c) u32;
    pub const SetLabel = *const fn (*Texture, StringView) callconv(.c) void;
    pub const AddRef = *const fn (*Texture) callconv(.c) void;
    pub const Release = *const fn (*Texture) callconv(.c) void;
};

extern fn wgpuTextureCreateView(texture: *Texture, descriptor: ?*const TextureViewDescriptor) ?*TextureView;
extern fn wgpuTextureDestroy(texture: *Texture) void;
extern fn wgpuTextureGetDepthOrArrayLayers(texture: *Texture) u32;
extern fn wgpuTextureGetDimension(texture: *Texture) TextureDimension;
extern fn wgpuTextureGetFormat(texture: *Texture) TextureFormat;
extern fn wgpuTextureGetHeight(texture: *Texture) u32;
extern fn wgpuTextureGetMipLevelCount(texture: *Texture) u32;
extern fn wgpuTextureGetSampleCount(texture: *Texture) u32;
extern fn wgpuTextureGetUsage(texture: *Texture) TextureUsage;
extern fn wgpuTextureGetWidth(texture: *Texture) u32;
extern fn wgpuTextureSetLabel(texture: *Texture, label: StringView) void;
extern fn wgpuTextureAddRef(texture: *Texture) void;
extern fn wgpuTextureRelease(texture: *Texture) void;

pub const Texture = opaque {
    pub inline fn createView(self: *Texture, descriptor: ?*const TextureViewDescriptor) ?*TextureView {
        return wgpuTextureCreateView(self, descriptor);
    }
    pub inline fn destroy(self: *Texture) void {
        wgpuTextureDestroy(self);
    }
    pub inline fn getDepthOrArrayLayers(self: *Texture) u32 {
        return wgpuTextureGetDepthOrArrayLayers(self);
    }
    pub inline fn getDimension(self: *Texture) TextureDimension {
        return wgpuTextureGetDimension(self);
    }
    pub inline fn getFormat(self: *Texture) TextureFormat {
        return wgpuTextureGetFormat(self);
    }
    pub inline fn getHeight(self: *Texture) u32 {
        return wgpuTextureGetHeight(self);
    }
    pub inline fn getMipLevelCount(self: *Texture) u32 {
        return wgpuTextureGetMipLevelCount(self);
    }
    pub inline fn getSampleCount(self: *Texture) u32 {
        return wgpuTextureGetSampleCount(self);
    }
    pub inline fn getUsage(self: *Texture) TextureUsage {
        return wgpuTextureGetUsage(self);
    }
    pub inline fn getWidth(self: *Texture) u32 {
        return wgpuTextureGetWidth(self);
    }

    // Unimplemented as of wgpu-native v25.0.2.1,
    // see https://github.com/gfx-rs/wgpu-native/blob/d8238888998db26ceab41942f269da0fa32b890c/src/unimplemented.rs#L208
    // pub inline fn setLabel(self: *Texture, label: []const u8) void {
    //     wgpuTextureSetLabel(self, StringView.fromSlice(label));
    // }

    pub inline fn addRef(self: *Texture) void {
        wgpuTextureAddRef(self);
    }
    pub inline fn release(self: *Texture) void {
        wgpuTextureRelease(self);
    }
};

pub const Origin3D = extern struct {
    x: u32 = 0,
    y: u32 = 0,
    z: u32 = 0,
};

pub const TexelCopyTextureInfo = extern struct {
    texture: *Texture,
    mip_level: u32 = 0,
    origin: Origin3D,
    aspect: TextureAspect = TextureAspect.all,
};

pub const TexelCopyBufferLayout = extern struct {
    offset: u64 = 0,
    bytes_per_row: u32 = WGPU_COPY_STRIDE_UNDEFINED,
    rows_per_image: u32 = WGPU_COPY_STRIDE_UNDEFINED,
};

// Seems a little weird to put this in texture.zig,
// but it seems to have more to do with images/textures than with buffers.
pub const TexelCopyBufferInfo = extern struct {
    layout: TexelCopyBufferLayout,
    buffer: *Buffer,
};
pub extern fn wgpuTextureGetNativeMetalTexture(texture: *Texture) ?*anyopaque;
