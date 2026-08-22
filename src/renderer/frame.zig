//! Per-frame orchestration: the ordered pass list, the per-frame context handed to each pass,
//! and the per-frame statistics passes accumulate into.
//!
//! This replaces the former `render/render_graph.zig`. That module called itself a graph and
//! carried `reads`/`writes` declarations, a `computeLifetimes` pass-lifetime solver, a
//! `validate()` checker and a `setEnabled` toggle — but execution was plain registration order,
//! `computeLifetimes` had no non-test caller, `validate()`'s result was discarded at the one call
//! site, `setEnabled` had no callers, and 4 of 7 pass categories and 4 of 9 resource kinds were
//! never referenced. Nothing inserted a barrier, aliased a resource, reordered or culled. What is
//! left once the inert metadata goes is an ordered list of function pointers, which is what this
//! is. See docs/v2-design.md §2.2.

const std = @import("std");
const transient = @import("transient.zig");

/// Per-frame state shared across all passes.
pub const FrameContext = struct {
    frame_index: u32,
    surface_width: u32,
    surface_height: u32,
    dpi_scale: f32,
    delta_time: f32,
    transient_pool: *transient.TransientPool,
};

/// Per-frame counters. Passes accumulate into these through `PassContext.stats`; the host reads
/// them back for the debug overlay.
pub const FrameStats = struct {
    pass_count: u32 = 0,
    draw_calls: u32 = 0,
    paint_commands: u32 = 0,
    scene_objects: u32 = 0,

    /// Reset the counters the passes accumulate. `paint_commands` is set by the host before the
    /// run (it knows the submitted list length), so it is deliberately not cleared here.
    pub fn resetPerRun(self: *FrameStats) void {
        self.pass_count = 0;
        self.draw_calls = 0;
        self.scene_objects = 0;
    }
};

/// Handed to each pass. `user` is an opaque pointer to the owner's GPU state (`EngineState` in
/// the FFI layer); passes cast it back via `userAs`. Deliberately ignorant of concrete GPU types
/// so this module has no dependency on the FFI layer that depends on it.
pub const PassContext = struct {
    frame: *const FrameContext,
    stats: *FrameStats,
    user: *anyopaque,

    pub fn userAs(self: *PassContext, comptime T: type) *T {
        return @ptrCast(@alignCast(self.user));
    }
};

/// One pass. The pass owns its own command encoding (some submit independently, matching the
/// existing multi-submit frame structure); this module owns only order and the profiling label.
pub const Pass = struct {
    name: []const u8,
    execute: *const fn (*PassContext) anyerror!void,
};

/// What one `run` did. Reported rather than logged from in here so the caller owns the log
/// (and so the skip-on-failure path is testable without tripping the test runner's
/// errors-were-logged check).
pub const RunResult = struct {
    failed: u32 = 0,
    /// The first pass that failed, for the caller's log line.
    first_failure_name: []const u8 = "",
    first_failure_err: ?anyerror = null,
};

/// Run `passes` in order. A pass failure is recorded and skipped rather than aborting the frame —
/// one broken pass should not black-screen the editor.
pub fn run(passes: []const Pass, ctx: *PassContext) RunResult {
    ctx.stats.resetPerRun();
    var result = RunResult{};
    for (passes) |p| {
        p.execute(ctx) catch |err| {
            if (result.failed == 0) {
                result.first_failure_name = p.name;
                result.first_failure_err = err;
            }
            result.failed += 1;
            continue;
        };
        ctx.stats.pass_count += 1;
    }
    return result;
}

/// Settings controlling which features the frame activates. Mirrored from C# `RenderSettings`.
pub const RenderSettings = struct {
    enable_3d_scene: bool = true,
    enable_glass_effects: bool = true,
    enable_debug_overlays: bool = false,
    scene_clear_r: f32 = 0.38,
    scene_clear_g: f32 = 0.60,
    scene_clear_b: f32 = 0.86,
    scene_clear_a: f32 = 1.0,
};

// ── tests ─────────────────────────────────────────────────────────────────────

const TestState = struct { order: [4]u8 = @splat(0), n: usize = 0, fail_at: ?usize = null };

fn record(comptime id: u8) fn (*PassContext) anyerror!void {
    return struct {
        fn f(ctx: *PassContext) anyerror!void {
            const s = ctx.userAs(TestState);
            if (s.fail_at) |fa| if (fa == s.n) {
                s.n += 1;
                return error.Boom;
            };
            s.order[s.n] = id;
            s.n += 1;
            ctx.stats.draw_calls += 1;
        }
    }.f;
}

test "passes run in registration order and count" {
    var st = TestState{};
    var stats = FrameStats{};
    var pool: transient.TransientPool = undefined;
    const fc = FrameContext{
        .frame_index = 0,
        .surface_width = 1,
        .surface_height = 1,
        .dpi_scale = 1,
        .delta_time = 0,
        .transient_pool = &pool,
    };
    var ctx = PassContext{ .frame = &fc, .stats = &stats, .user = &st };
    const r = run(&.{
        .{ .name = "a", .execute = record(1) },
        .{ .name = "b", .execute = record(2) },
        .{ .name = "c", .execute = record(3) },
    }, &ctx);
    try std.testing.expectEqual(@as(u32, 0), r.failed);
    try std.testing.expectEqual(@as(usize, 3), st.n);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, st.order[0..3]);
    try std.testing.expectEqual(@as(u32, 3), stats.pass_count);
    try std.testing.expectEqual(@as(u32, 3), stats.draw_calls);
}

test "a failing pass is skipped, later passes still run" {
    var st = TestState{ .fail_at = 1 };
    var stats = FrameStats{ .paint_commands = 42 };
    var pool: transient.TransientPool = undefined;
    const fc = FrameContext{
        .frame_index = 0,
        .surface_width = 1,
        .surface_height = 1,
        .dpi_scale = 1,
        .delta_time = 0,
        .transient_pool = &pool,
    };
    var ctx = PassContext{ .frame = &fc, .stats = &stats, .user = &st };
    const r = run(&.{
        .{ .name = "a", .execute = record(1) },
        .{ .name = "boom", .execute = record(2) },
        .{ .name = "c", .execute = record(3) },
    }, &ctx);
    // Two of three succeeded; the third still ran after the failure.
    try std.testing.expectEqual(@as(u32, 2), stats.pass_count);
    try std.testing.expectEqual(@as(u32, 1), r.failed);
    try std.testing.expectEqualStrings("boom", r.first_failure_name);
    try std.testing.expectEqual(@as(?anyerror, error.Boom), r.first_failure_err);
    try std.testing.expectEqual(@as(u8, 3), st.order[2]);
    // resetPerRun must not clear the host-supplied paint_commands.
    try std.testing.expectEqual(@as(u32, 42), stats.paint_commands);
}
