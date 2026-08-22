//! Headless WGSL validation — `zig build check-shaders`.
//!
//! Every `.wgsl` in this engine is compiled by wgpu-native's embedded naga at **runtime**, on the
//! first frame that uses it. A broken shader therefore builds perfectly and fails in front of a
//! user; `zig build shared-lib` proves nothing about it. That is also how
//! `shaders/text_shader_source.wgsl` drifted out of sync with the live inline text shader without
//! anyone noticing.
//!
//! This creates a real wgpu device with no window or surface and compiles every shader source the
//! engine ships — taken from the Zig constants, not the raw files, so what is checked is exactly
//! what runs, comptime prelude concatenation included. Any naga diagnostic is captured through an
//! error scope and reported with the shader's name.
//!
//! Requires a working adapter (a GPU, or a software rasteriser such as lavapipe). Where none is
//! available it exits 0 with a notice rather than failing a build that has nothing to do with it.
//! See docs/v2-design.md §5.3.

const std = @import("std");
const wgpu = @import("wgpu");
const renderer = @import("zigote").renderer;
const ui_shaders = renderer.wgpu_ui_shaders;
const shaders_3d = renderer.wgpu_3d_shaders;
const freetype_text = renderer.freetype_text;
const wgpu_blur = renderer.wgpu_blur;

const Shader = struct { name: []const u8, source: []const u8 };

/// Every shader the engine can compile. A new one must be added here — the list is checked
/// against the shader directory below so a forgotten entry fails rather than going unvalidated.
const shaders = [_]Shader{
    .{ .name = "shape", .source = ui_shaders.shape_shader_source },
    .{ .name = "image", .source = ui_shaders.image_shader_source },
    .{ .name = "liquid_glass", .source = ui_shaders.liquid_glass_shader_source },
    .{ .name = "text", .source = freetype_text.text_shader_source },
    .{ .name = "blur_h", .source = wgpu_blur.h_wgsl },
    .{ .name = "blur_v", .source = wgpu_blur.v_wgsl },
    .{ .name = "mesh", .source = shaders_3d.mesh_shader_source },
    .{ .name = "particle", .source = shaders_3d.particle_shader_source },
    .{ .name = "particle_compute", .source = shaders_3d.particle_compute_source },
    .{ .name = "sprite", .source = shaders_3d.sprite_shader_source },
    .{ .name = "shadow", .source = shaders_3d.shadow_shader_source },
    .{ .name = "shadow_alpha", .source = shaders_3d.shadow_alpha_shader_source },
    .{ .name = "point_shadow", .source = shaders_3d.point_shadow_shader_source },
    .{ .name = "sky", .source = shaders_3d.sky_shader_source },
    .{ .name = "env_bake", .source = shaders_3d.env_bake_shader_source },
    .{ .name = "bloom_down", .source = shaders_3d.bloom_down_shader_source },
    .{ .name = "bloom_up", .source = shaders_3d.bloom_up_shader_source },
    .{ .name = "ssao", .source = shaders_3d.ssao_shader_source },
    .{ .name = "ssr", .source = shaders_3d.ssr_shader_source },
    .{ .name = "taa", .source = shaders_3d.taa_shader_source },
    .{ .name = "tonemap", .source = shaders_3d.tonemap_shader_source },
    .{ .name = "exposure", .source = shaders_3d.exposure_shader_source },
    .{ .name = "dof", .source = shaders_3d.dof_shader_source },
};

var failures: u32 = 0;
var current_shader: []const u8 = "";

fn onUncapturedError(
    device: ?*wgpu.Device,
    error_type: wgpu.ErrorType,
    message: wgpu.StringView,
    userdata1: ?*anyopaque,
    userdata2: ?*anyopaque,
) callconv(.c) void {
    _ = device;
    _ = userdata1;
    _ = userdata2;
    failures += 1;
    std.debug.print("  {s}: wgpu {s}: {s}\n", .{
        current_shader,
        @tagName(error_type),
        message.toSlice() orelse "(no message)",
    });
}

pub fn main() !u8 {
    wgpu.setLogCallback(wgpuLog, null);
    wgpu.setLogLevel(.warn);

    const instance = wgpu.Instance.create(null) orelse {
        std.debug.print("check-shaders: no wgpu instance available — skipping\n", .{});
        return 0;
    };
    defer instance.release();

    // No surface: nothing here rasterises, so the adapter needs no presentation support. This is
    // what lets the check run in CI and over SSH.
    var adapter_opts = wgpu.RequestAdapterOptions{ .compatible_surface = null };
    const adapter_resp = instance.requestAdapterSync(&adapter_opts, 5_000_000);
    const adapter = adapter_resp.adapter orelse {
        std.debug.print("check-shaders: no GPU adapter (need a GPU or lavapipe) — skipping\n", .{});
        return 0;
    };
    defer adapter.release();

    var device_desc = wgpu.DeviceDescriptor{
        .label = wgpu.StringView.fromSlice("shader-check device"),
        .required_limits = null,
        .uncaptured_error_callback_info = .{ .callback = onUncapturedError },
    };
    const device_resp = adapter.requestDeviceSync(instance, &device_desc, 5_000_000);
    const device = device_resp.device orelse {
        std.debug.print("check-shaders: no wgpu device — skipping\n", .{});
        return 0;
    };
    defer device.release();

    for (shaders) |s| {
        current_shader = s.name;
        var desc = wgpu.shaderModuleWGSLDescriptor(.{ .label = s.name, .code = s.source });
        if (device.createShaderModule(&desc)) |mod| {
            mod.release();
        } else {
            failures += 1;
            std.debug.print("  {s}: createShaderModule returned null\n", .{s.name});
        }
        // Diagnostics arrive on the device's error callback, which wgpu-native may deliver
        // asynchronously; drain before moving on so a failure is attributed to the right shader.
        _ = device.poll(false, null);
    }
    current_shader = "";

    if (failures != 0) {
        std.debug.print("check-shaders: {d} shader diagnostic(s) across {d} shaders\n", .{ failures, shaders.len });
        return 1;
    }
    std.debug.print("check-shaders: {d} shaders compiled clean\n", .{shaders.len});
    return 0;
}

fn wgpuLog(level: wgpu.LogLevel, message: wgpu.StringView, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    if (@intFromEnum(level) > @intFromEnum(wgpu.LogLevel.warn)) return;
    std.debug.print("  wgpu [{s}]: {s}\n", .{ @tagName(level), message.toSlice() orelse "" });
}

// A shader file that nothing in `shaders` above references would never be validated — the exact
// gap that let text_shader_source.wgsl rot. Every .wgsl must be reachable from a listed source.
test "every shader file is covered by the validation list" {
    var io_state = std.Io.Threaded.init(std.testing.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    var dir = std.Io.Dir.cwd().openDir(io, "src/renderer/shaders", .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer dir.close(io);

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".wgsl")) continue;
        // The common_* fragments are prepended into other sources rather than compiled alone.
        if (std.mem.startsWith(u8, entry.name, "common_")) continue;

        const stem = entry.name[0 .. entry.name.len - ".wgsl".len];
        const base = if (std.mem.endsWith(u8, stem, "_shader_source"))
            stem[0 .. stem.len - "_shader_source".len]
        else if (std.mem.endsWith(u8, stem, "_source"))
            stem[0 .. stem.len - "_source".len]
        else
            stem;

        const found = for (shaders) |s| {
            if (std.mem.eql(u8, s.name, base)) break true;
        } else false;
        if (!found) {
            std.debug.print("shaders/{s} is never validated — add it to check_shaders.zig\n", .{entry.name});
            return error.ShaderNotValidated;
        }
    }
}
