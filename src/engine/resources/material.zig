const std = @import("std");
const math = @import("../math/root.zig");
const Vec4 = math.Vec4;
const Vec3 = math.Vec3;

// opaque_mode: solid. mask: alpha-tested cutout. blend: generic src-alpha transparency (decals,
// stickers, hair, cloth, toon). glass: dedicated transmissive/reflective path (windows, lenses).
pub const AlphaMode = enum { opaque_mode, mask, blend, glass };
pub const RenderEffect = enum(u32) {
    standard = 0,
    crt_tv = 1,
    unlit = 2,
};

/// PBR metallic-roughness material. Textures are optional; factors are used as fallback.
pub const Material = struct {
    name: []const u8 = "",
    base_color_factor: Vec4 = .{ .x = 1, .y = 1, .z = 1, .w = 1 },
    metallic_factor: f32 = 0.0,
    roughness_factor: f32 = 0.5,
    emissive_factor: Vec3 = Vec3.zero,
    alpha_mode: AlphaMode = .opaque_mode,
    alpha_cutoff: f32 = 0.5,
    double_sided: bool = false,
    effect: RenderEffect = .standard,
    // Extended PBR (KHR_materials_clearcoat / _specular). clearcoat_factor 0 = no coat (default);
    // the glossy lacquer lobe is opt-in per material. specular_factor scales the dielectric F0
    // (0.04) — 1.0 = standard.
    clearcoat_factor: f32 = 0.0,
    clearcoat_roughness: f32 = 0.0,
    specular_factor: f32 = 1.0,
    // KHR_materials_ior / _transmission. ior drives the dielectric F0 (((n-1)/(n+1))²) and the
    // glass refraction bend; transmission blends the lit surface toward the transmissive glass
    // response (0 = opaque PBR; the glass alpha mode with transmission unset is treated as 1).
    ior: f32 = 1.5,
    transmission: f32 = 0.0,
    // >0 → the metallic-roughness map's R channel is glTF ORM occlusion; scales ambient.
    occlusion_strength: f32 = 0.0,

    // Optional texture pixel data (RGBA8, owned).
    base_color_pixels: ?[]u8 = null,
    base_color_width: u32 = 0,
    base_color_height: u32 = 0,

    normal_pixels: ?[]u8 = null,
    normal_width: u32 = 0,
    normal_height: u32 = 0,

    // Metallic-roughness map (glTF convention: roughness in G, metallic in B).
    metallic_roughness_pixels: ?[]u8 = null,
    metallic_roughness_width: u32 = 0,
    metallic_roughness_height: u32 = 0,

    // Emissive map (sRGB); multiplied by emissive_factor.
    emissive_pixels: ?[]u8 = null,
    emissive_width: u32 = 0,
    emissive_height: u32 = 0,

    pub fn deinit(self: *Material, allocator: std.mem.Allocator) void {
        if (self.base_color_pixels) |p| allocator.free(p);
        if (self.normal_pixels) |p| allocator.free(p);
        if (self.metallic_roughness_pixels) |p| allocator.free(p);
        if (self.emissive_pixels) |p| allocator.free(p);
    }

    pub fn flat(color: Vec4) Material {
        return .{ .base_color_factor = color };
    }
};
