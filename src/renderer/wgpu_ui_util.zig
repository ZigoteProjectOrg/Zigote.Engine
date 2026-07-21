//! Leaf utility helpers for the 2D UI renderer: vertex-buffer growth, scissor
//! rect application, and clip-rect intersection. No dependency on renderer state.

const std = @import("std");
const wgpu = @import("wgpu");
const zg = @import("../root.zig");

pub fn ensureVertexBuffer(
    device: *wgpu.Device,
    slot: *?*wgpu.Buffer,
    capacity: *usize,
    label: []const u8,
    required_size: usize,
) !*wgpu.Buffer {
    if (slot.*) |buffer| {
        if (capacity.* >= required_size) return buffer;
    }

    const next_size = growBufferCapacity(capacity.*, required_size);
    const next = device.createBuffer(&.{
        .label = wgpu.StringView.fromSlice(label),
        .usage = wgpu.BufferUsages.copy_dst | wgpu.BufferUsages.vertex,
        .size = @intCast(next_size),
    }) orelse return error.WgpuVertexBufferUnavailable;

    releaseBuffer(slot.*);
    slot.* = next;
    capacity.* = next_size;
    return next;
}

pub fn growBufferCapacity(current: usize, required: usize) usize {
    var next = @max(current, 4096);
    while (next < required) {
        next = std.math.mul(usize, next, 2) catch required;
        if (next == required) break;
    }
    return next;
}

pub fn releaseBuffer(buffer: ?*wgpu.Buffer) void {
    if (buffer) |value| {
        value.release();
    }
}

/// Returns false — with NO pass state touched — when the effective scissor is empty, so callers
/// can skip the whole draw (pipeline/bind-group/vertex-buffer setup included) instead of issuing
/// a draw that rasterizes nothing.
pub fn applyScissor(
    pass: *wgpu.RenderPassEncoder,
    clip: ?zg.Rect,
    damage: ?zg.Rect,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
) bool {
    // Effective scissor = the batch clip intersected with the frame damage region (partial repaint).
    // Either may be absent: no batch clip = the whole frame; no damage = full-frame repaint. The
    // intersection of a rect with "everything" is that rect, so this reduces to the plain clip when
    // damage is null (the common, full-repaint case) — no behavioural change there.
    var effective: ?zg.Rect = clip;
    if (damage) |d| {
        effective = if (clip) |c| intersectRects(c, d) else d;
    }

    if (effective) |c| {
        if (!std.math.isFinite(c.x) or !std.math.isFinite(c.y) or !std.math.isFinite(c.width) or !std.math.isFinite(c.height)) {
            return false;
        }
        // Clamp a negative origin by SHRINKING the box, never shifting it: a shifted scissor invades
        // the pixels to the right/below its true extent — on partial repaint that overlaps the
        // neighbouring damage rect's sweep and alpha-blended ops composite twice in the overlap band
        // (a visible lighter/darker strip), and on clips it draws content past the clip edge.
        const sx = @max(0.0, c.x * scale_factor);
        const sy = @max(0.0, c.y * scale_factor);
        const sw = @max(0.0, (c.width + @min(0.0, c.x)) * scale_factor);
        const sh = @max(0.0, (c.height + @min(0.0, c.y)) * scale_factor);

        const x_u32 = @as(u32, @intFromFloat(sx));
        const y_u32 = @as(u32, @intFromFloat(sy));
        const w_u32 = @as(u32, @intFromFloat(sw));
        const h_u32 = @as(u32, @intFromFloat(sh));

        // Clamp to frame bounds
        const cx = @min(x_u32, frame_width);
        const cy = @min(y_u32, frame_height);
        const cw = @min(w_u32, frame_width -% cx);
        const ch = @min(h_u32, frame_height -% cy);

        if (cw > 0 and ch > 0) {
            pass.setScissorRect(cx, cy, cw, ch);
            return true;
        }
        return false;
    }
    pass.setScissorRect(0, 0, frame_width, frame_height);
    return true;
}

pub fn intersectRects(a: zg.Rect, b: zg.Rect) zg.Rect {
    const x0 = @max(a.x, b.x);
    const y0 = @max(a.y, b.y);
    const x1 = @min(a.x + a.width, b.x + b.width);
    const y1 = @min(a.y + a.height, b.y + b.height);
    return .{
        .x = x0,
        .y = y0,
        .width = @max(0.0, x1 - x0),
        .height = @max(0.0, y1 - y0),
    };
}
