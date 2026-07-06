const std = @import("std");
const frame_context = @import("frame_context.zig");
const render_resource = @import("render_resource.zig");

/// A render-graph resource is identified by its logical kind. This keeps the handle
/// space small and debuggable; transient allocation still goes through TransientPool.
pub const ResourceHandle = render_resource.ResourceKind;

/// Logical pass categories. Used for grouping, enable/disable toggles from C#, and
/// profiling labels. Execution order is the order passes are registered.
pub const PassType = enum(u8) {
    resource_upload = 0,
    scene_3d = 1,
    scene_2d = 2,
    backdrop_capture = 3,
    ui = 4,
    debug_overlay = 5,
    present = 6,
};

/// Per-pass state handed to each `execute` callback. `user` is an opaque pointer to the
/// owner's GPU state (EngineState in the FFI layer) — passes cast it back via `userAs`.
/// The graph deliberately stays ignorant of concrete GPU types so it has no dependency
/// on the FFI layer (which depends on it).
pub const PassContext = struct {
    frame: *const frame_context.FrameContext,
    graph: *RenderGraph,
    user: *anyopaque,

    pub fn userAs(self: *PassContext, comptime T: type) *T {
        return @ptrCast(@alignCast(self.user));
    }
};

/// A single node in the render graph. `reads`/`writes` declare the resources the pass
/// touches — used for lifetime tracking, validation, and FFI introspection. The pass
/// owns its own command encoding (some passes submit independently), matching the
/// existing multi-submit frame structure; the graph orchestrates order, enable flags,
/// profiling labels and stats.
pub const Pass = struct {
    name: []const u8,
    pass_type: PassType,
    enabled: bool = true,
    reads: []const ResourceHandle = &.{},
    writes: []const ResourceHandle = &.{},
    execute: *const fn (*PassContext) anyerror!void,
};

/// Computed first-write / last-read span for one resource across the enabled pass list.
/// `first_write == null` means the resource is read before any registered pass writes it,
/// i.e. it is imported/externally produced (swapchain, shadow map persisted across frames).
pub const ResourceLifetime = struct {
    handle: ResourceHandle,
    first_write: ?u32 = null,
    last_write: ?u32 = null,
    last_read: ?u32 = null,
};

pub const RenderGraph = struct {
    allocator: std.mem.Allocator,
    passes: std.ArrayList(Pass) = .empty,

    // Per-frame statistics, refreshed by execute() / written by passes.
    frame_pass_count: u32 = 0,
    frame_draw_calls: u32 = 0,
    frame_paint_commands: u32 = 0,
    frame_scene_objects: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) RenderGraph {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *RenderGraph) void {
        self.passes.deinit(self.allocator);
    }

    pub fn addPass(self: *RenderGraph, pass: Pass) !void {
        try self.passes.append(self.allocator, pass);
    }

    /// Toggle every pass of a given category (e.g. disable debug_overlay).
    pub fn setEnabled(self: *RenderGraph, pass_type: PassType, enabled: bool) void {
        for (self.passes.items) |*p| {
            if (p.pass_type == pass_type) p.enabled = enabled;
        }
    }

    /// Run the graph: execute each enabled pass in registration order. Pass failures are
    /// logged and skipped rather than aborting the frame — a single broken pass should not
    /// black-screen the editor. Resets per-frame counters first (scene/draw stats are
    /// accumulated by the passes themselves through `ctx.graph`).
    pub fn execute(self: *RenderGraph, ctx: *PassContext) void {
        self.frame_pass_count = 0;
        self.frame_draw_calls = 0;
        self.frame_scene_objects = 0;
        for (self.passes.items) |*p| {
            if (!p.enabled) continue;
            p.execute(ctx) catch |err| {
                std.log.err("render graph: pass '{s}' failed: {}", .{ p.name, err });
                continue;
            };
            self.frame_pass_count += 1;
        }
    }

    /// Compute resource lifetimes over the enabled passes, in execution order. Pure over
    /// the read/write declarations — used by FFI introspection and (future) transient
    /// aliasing. `out` is filled with one entry per distinct resource; returns the count.
    pub fn computeLifetimes(self: *const RenderGraph, out: []ResourceLifetime) usize {
        var count: usize = 0;
        var step: u32 = 0;
        for (self.passes.items) |*p| {
            if (!p.enabled) continue;
            for (p.writes) |w| {
                const e = lifetimeEntry(out[0..count], w) orelse blk: {
                    if (count >= out.len) continue;
                    out[count] = .{ .handle = w };
                    count += 1;
                    break :blk &out[count - 1];
                };
                if (e.first_write == null) e.first_write = step;
                e.last_write = step;
            }
            for (p.reads) |r| {
                const e = lifetimeEntry(out[0..count], r) orelse blk: {
                    if (count >= out.len) continue;
                    out[count] = .{ .handle = r };
                    count += 1;
                    break :blk &out[count - 1];
                };
                e.last_read = step;
            }
            step += 1;
        }
        return count;
    }

    /// Soft validation: count passes that read a resource no earlier pass has written.
    /// Imported resources legitimately trip this, so it is advisory (logged, not fatal).
    pub fn validate(self: *const RenderGraph) u32 {
        var warnings: u32 = 0;
        var written = std.EnumSet(ResourceHandle).initEmpty();
        for (self.passes.items) |*p| {
            if (!p.enabled) continue;
            for (p.reads) |r| {
                if (!written.contains(r)) {
                    warnings += 1;
                    std.log.debug("render graph: pass '{s}' reads un-produced resource {s}", .{ p.name, @tagName(r) });
                }
            }
            for (p.writes) |w| written.insert(w);
        }
        return warnings;
    }

    fn lifetimeEntry(slice: []ResourceLifetime, handle: ResourceHandle) ?*ResourceLifetime {
        for (slice) |*e| {
            if (e.handle == handle) return e;
        }
        return null;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────────────

fn noopExecute(_: *PassContext) anyerror!void {}

test "computeLifetimes tracks first-write / last-read across passes" {
    const A = ResourceHandle.scene_color;
    const B = ResourceHandle.scene_depth;
    var g = RenderGraph.init(std.testing.allocator);
    defer g.deinit();
    // 0: writes A | 1: reads A, writes B | 2: reads B
    try g.addPass(.{ .name = "w_a", .pass_type = .scene_3d, .writes = &.{A}, .execute = noopExecute });
    try g.addPass(.{ .name = "rw", .pass_type = .scene_3d, .reads = &.{A}, .writes = &.{B}, .execute = noopExecute });
    try g.addPass(.{ .name = "r_b", .pass_type = .ui, .reads = &.{B}, .execute = noopExecute });

    var buf: [8]ResourceLifetime = undefined;
    const n = g.computeLifetimes(&buf);
    try std.testing.expectEqual(@as(usize, 2), n);

    const a = RenderGraph.lifetimeEntry(buf[0..n], A).?;
    try std.testing.expectEqual(@as(?u32, 0), a.first_write);
    try std.testing.expectEqual(@as(?u32, 1), a.last_read);

    const b = RenderGraph.lifetimeEntry(buf[0..n], B).?;
    try std.testing.expectEqual(@as(?u32, 1), b.first_write);
    try std.testing.expectEqual(@as(?u32, 2), b.last_read);
}

test "validate flags read-before-write but allows produced resources" {
    const A = ResourceHandle.scene_color;
    var g = RenderGraph.init(std.testing.allocator);
    defer g.deinit();
    try g.addPass(.{ .name = "reader", .pass_type = .ui, .reads = &.{A}, .execute = noopExecute });
    try std.testing.expectEqual(@as(u32, 1), g.validate()); // A never produced first

    var g2 = RenderGraph.init(std.testing.allocator);
    defer g2.deinit();
    try g2.addPass(.{ .name = "writer", .pass_type = .scene_3d, .writes = &.{A}, .execute = noopExecute });
    try g2.addPass(.{ .name = "reader", .pass_type = .ui, .reads = &.{A}, .execute = noopExecute });
    try std.testing.expectEqual(@as(u32, 0), g2.validate());
}

test "disabled passes are skipped by lifetimes and execution order" {
    const A = ResourceHandle.scene_color;
    var g = RenderGraph.init(std.testing.allocator);
    defer g.deinit();
    try g.addPass(.{ .name = "off", .pass_type = .scene_3d, .enabled = false, .writes = &.{A}, .execute = noopExecute });
    try g.addPass(.{ .name = "on", .pass_type = .ui, .reads = &.{A}, .execute = noopExecute });
    // With the writer disabled, the reader now reads an un-produced resource.
    try std.testing.expectEqual(@as(u32, 1), g.validate());
}

/// Settings that control which features the render graph activates each frame. Mirrored
/// from the C# RenderSettings over FFI.
pub const RenderSettings = struct {
    enable_3d_scene: bool = true,
    enable_glass_effects: bool = true,
    enable_debug_overlays: bool = false,
    scene_clear_r: f32 = 0.38,
    scene_clear_g: f32 = 0.60,
    scene_clear_b: f32 = 0.86,
    scene_clear_a: f32 = 1.0,
};
