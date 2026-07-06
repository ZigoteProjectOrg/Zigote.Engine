const std = @import("std");
const wgpu = @import("wgpu");

pub const ResourceKind = enum(u8) {
    swapchain_color = 0,
    scene_color = 1,
    scene_depth = 2,
    backdrop_color = 3,
    ui_color = 4,
    blur_temp_a = 5,
    blur_temp_b = 6,
    picking_id_buffer = 7,
    shadow_depth = 8,
};

pub const TextureDesc = struct {
    width: u32,
    height: u32,
    format: wgpu.TextureFormat,
    usage: u64, // wgpu.TextureUsage (u64 flags)
    sample_count: u32 = 1,
    debug_name: []const u8 = "transient",
};

const TransientEntry = struct {
    texture: *wgpu.Texture,
    view: *wgpu.TextureView,
    desc: TextureDesc,
    in_use: bool = false,
};

/// Pool of transient (per-frame) textures. Reuses compatible allocations across
/// frames to avoid GPU allocations every frame in steady state.
pub const TransientPool = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(TransientEntry) = .empty,
    device: *wgpu.Device,

    pub fn init(allocator: std.mem.Allocator, device: *wgpu.Device) TransientPool {
        return .{ .allocator = allocator, .device = device };
    }

    pub fn deinit(self: *TransientPool) void {
        for (self.entries.items) |e| {
            e.view.release();
            e.texture.release();
        }
        self.entries.deinit(self.allocator);
    }

    fn compat(a: TextureDesc, b: TextureDesc) bool {
        return a.width == b.width and
            a.height == b.height and
            @intFromEnum(a.format) == @intFromEnum(b.format) and
            a.usage == b.usage and
            a.sample_count == b.sample_count;
    }

    pub fn acquire(self: *TransientPool, desc: TextureDesc) !struct {
        texture: *wgpu.Texture,
        view: *wgpu.TextureView,
    } {
        for (self.entries.items) |*e| {
            if (!e.in_use and compat(e.desc, desc)) {
                e.in_use = true;
                return .{ .texture = e.texture, .view = e.view };
            }
        }
        const tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice(desc.debug_name),
            .usage = desc.usage,
            .dimension = .@"2d",
            .size = .{ .width = desc.width, .height = desc.height, .depth_or_array_layers = 1 },
            .format = desc.format,
            .mip_level_count = 1,
            .sample_count = desc.sample_count,
        }) orelse return error.CreateTextureFailed;
        const view = tex.createView(null) orelse {
            tex.release();
            return error.CreateTextureViewFailed;
        };
        try self.entries.append(self.allocator, .{
            .texture = tex,
            .view = view,
            .desc = desc,
            .in_use = true,
        });
        return .{ .texture = tex, .view = view };
    }

    pub fn releaseTexture(self: *TransientPool, texture: *wgpu.Texture) void {
        for (self.entries.items) |*e| {
            if (e.texture == texture) {
                e.in_use = false;
                return;
            }
        }
    }

    /// Remove entries whose size no longer matches (on window resize).
    pub fn evictResized(self: *TransientPool, width: u32, height: u32) void {
        var i: usize = 0;
        while (i < self.entries.items.len) {
            const e = &self.entries.items[i];
            if (e.desc.width != width or e.desc.height != height) {
                e.view.release();
                e.texture.release();
                _ = self.entries.swapRemove(i);
            } else {
                i += 1;
            }
        }
    }

    pub fn resetFrame(self: *TransientPool) void {
        for (self.entries.items) |*e| e.in_use = false;
    }
};
