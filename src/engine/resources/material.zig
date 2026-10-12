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

    pub const Pass = enum(u8) { solid, blend, glass };

    pub fn renderPass(self: *const Material) Pass {
        return switch (self.alpha_mode) {
            .opaque_mode, .mask => .solid,
            .blend => .blend,
            .glass => .glass,
        };
    }

    /// Shared GPU material payload for regular and instanced draws. Keep authored values intact;
    /// sanitize at the upload boundary so malformed assets cannot poison an HDR frame with NaNs.
    pub const Shading = struct {
        base_color: [4]f32,
        metallic_roughness: [4]f32,
        emissive: [4]f32,
        effect: [4]u32,
        surface: [4]f32,
    };

    fn bounded(value: f32, fallback: f32, low: f32, high: f32) f32 {
        return std.math.clamp(if (std.math.isFinite(value)) value else fallback, low, high);
    }

    pub fn shading(self: *const Material, selected: bool) Shading {
        const emission = self.emissive_factor.toArray();
        var color = self.base_color_factor.toArray();
        for (&color) |*c| c.* = bounded(c.*, 1, 0, 1);
        return .{
            .base_color = color,
            .metallic_roughness = .{
                bounded(self.metallic_factor, 0, 0, 1),
                bounded(self.roughness_factor, 0.5, 0, 1),
                bounded(self.clearcoat_factor, 0, 0, 1),
                bounded(self.clearcoat_roughness, 0, 0, 1),
            },
            .emissive = .{
                bounded(emission[0], 0, 0, 65504),
                bounded(emission[1], 0, 0, 65504),
                bounded(emission[2], 0, 0, 65504),
                bounded(self.specular_factor, 1, 0, 1),
            },
            .effect = .{ @backingInt(self.effect), @intFromBool(selected), @backingInt(self.alpha_mode), @bitCast(bounded(self.alpha_cutoff, 0.5, 0, 1)) },
            .surface = .{
                bounded(self.ior, 1.5, 1, 3),
                if (self.alpha_mode == .glass and self.transmission == 0) 1 else bounded(self.transmission, 0, 0, 1),
                bounded(self.occlusion_strength, 0, 0, 1),
                0,
            },
        };
    }

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

test "material payload preserves PBR factors and legacy glass" {
    const m = Material{ .alpha_mode = .glass, .ior = 1.33, .specular_factor = 0.7, .clearcoat_roughness = 0.9 };
    const p = m.shading(true);
    try std.testing.expectEqual(@as(u32, 3), p.effect[2]);
    try std.testing.expectEqual(@as(u32, 1), p.effect[1]);
    try std.testing.expectEqual(@as(f32, 1), p.surface[1]);
    try std.testing.expectEqual(@as(f32, 1.33), p.surface[0]);
    try std.testing.expectEqual(@as(f32, 0.7), p.emissive[3]);
    try std.testing.expectEqual(@as(f32, 0.9), p.metallic_roughness[3]);
    var partial = m;
    partial.transmission = 0.25;
    try std.testing.expectEqual(@as(f32, 0.25), partial.shading(false).surface[1]);
    try std.testing.expectEqual(@as(f32, 0), (Material{}).shading(false).surface[1]);
}

test "material payload bounds invalid authored values without mutating them" {
    const m = Material{
        .roughness_factor = 2,
        .metallic_factor = -1,
        .ior = std.math.nan(f32),
        .specular_factor = std.math.inf(f32),
        .emissive_factor = .{ .x = -2, .y = 12, .z = std.math.nan(f32) },
        .alpha_cutoff = std.math.nan(f32),
    };
    const p = m.shading(false);
    try std.testing.expectEqual(@as(f32, 1), p.metallic_roughness[1]);
    try std.testing.expectEqual(@as(f32, 0), p.metallic_roughness[0]);
    try std.testing.expectEqual(@as(f32, 1.5), p.surface[0]);
    try std.testing.expectEqual([4]f32{ 0, 12, 0, 1 }, p.emissive);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.5))), p.effect[3]);
    try std.testing.expectEqual(@as(f32, 2), m.roughness_factor);
}

test "material pass keeps alpha masks opaque and separates blend from glass" {
    try std.testing.expectEqual(Material.Pass.solid, (Material{}).renderPass());
    try std.testing.expectEqual(Material.Pass.solid, (Material{ .alpha_mode = .mask }).renderPass());
    try std.testing.expectEqual(Material.Pass.blend, (Material{ .alpha_mode = .blend }).renderPass());
    try std.testing.expectEqual(Material.Pass.glass, (Material{ .alpha_mode = .glass }).renderPass());
}
