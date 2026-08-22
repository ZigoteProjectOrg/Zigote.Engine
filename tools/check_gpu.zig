//! Headless GPU checks — `zig build check-gpu`.
//!
//! Two things, on one throwaway device: every WGSL shader compiles, and the texture row-pitch
//! contract three upload paths depend on actually holds.
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
//! The row-pitch check exists because `updateCachedImage`, `createImageTexture` and
//! `uploadMipLevel` each used to repack unaligned images into a 256-byte-aligned staging buffer —
//! an allocation, a per-row memcpy and a memset per upload, for any width that is not a multiple
//! of 64 px. That alignment rule is real, but it governs buffer<->texture COPIES, not
//! `queue.writeTexture`, which takes the row pitch as given. Removing the staging is only safe as
//! long as that stays true, so it is asserted here rather than assumed.
//!
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
        std.debug.print("check-gpu: no wgpu instance available — skipping\n", .{});
        return 0;
    };
    defer instance.release();

    // No surface: nothing here rasterises, so the adapter needs no presentation support. This is
    // what lets the check run in CI and over SSH.
    var adapter_opts = wgpu.RequestAdapterOptions{ .compatible_surface = null };
    const adapter_resp = instance.requestAdapterSync(&adapter_opts, 5_000_000);
    const adapter = adapter_resp.adapter orelse {
        std.debug.print("check-gpu: no GPU adapter (need a GPU or lavapipe) — skipping\n", .{});
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
        std.debug.print("check-gpu: no wgpu device — skipping\n", .{});
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
        std.debug.print("check-gpu: {d} shader diagnostic(s) across {d} shaders\n", .{ failures, shaders.len });
        return 1;
    }
    std.debug.print("check-gpu: {d} shaders compiled clean\n", .{shaders.len});

    const queue = device.getQueue() orelse {
        std.debug.print("check-gpu: no queue — skipping the row-pitch check\n", .{});
        return 0;
    };
    const pitch_mismatches = checkUnalignedRowPitch(device, queue) catch |err| {
        std.debug.print("check-gpu: row-pitch check failed: {}\n", .{err});
        return 1;
    };
    if (failures != 0 or pitch_mismatches != 0) {
        std.debug.print(
            "check-gpu: unaligned writeTexture is NOT safe here ({d} error(s), {d} byte mismatch(es)) —\n" ++
                "  the staging removal in wgpu.zig/wgpu_3d.zig must be reverted\n",
            .{ failures, pitch_mismatches },
        );
        return 1;
    }
    std.debug.print("check-gpu: unaligned writeTexture round-trips byte-exact\n", .{});
    return 0;
}

/// Write a texture whose row pitch is deliberately not a multiple of 256, read it back, and
/// compare every byte. Returns the mismatch count.
fn checkUnalignedRowPitch(device: *wgpu.Device, queue: *wgpu.Queue) !u32 {
    const w: u32 = 100; // 400 bytes/row
    const h: u32 = 10;
    std.debug.assert((w * 4) % 256 != 0);

    const tex = device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("row-pitch probe"),
        .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst | wgpu.TextureUsages.copy_src,
        .dimension = .@"2d",
        .size = .{ .width = w, .height = h, .depth_or_array_layers = 1 },
        .format = .rgba8_unorm,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return error.CreateTextureFailed;
    defer tex.release();

    // Every pixel distinct, so a pitch mistake shows up as shifted data rather than passing.
    var pixels: [w * h * 4]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const i = (y * w + x) * 4;
        pixels[i + 0] = @intCast(x);
        pixels[i + 1] = @intCast(y);
        pixels[i + 2] = @intCast((x *% 7 +% y *% 13) & 0xff);
        pixels[i + 3] = 0xff;
    };
    queue.writeTexture(
        &.{ .texture = tex, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &pixels,
        pixels.len,
        &.{ .offset = 0, .bytes_per_row = w * 4, .rows_per_image = h },
        &.{ .width = w, .height = h, .depth_or_array_layers = 1 },
    );

    // copyTextureToBuffer DOES require the 256-byte pitch, so the readback pads.
    const dst_stride: u32 = std.mem.alignForward(u32, w * 4, 256);
    const buf = device.createBuffer(&.{
        .label = wgpu.StringView.fromSlice("row-pitch readback"),
        .usage = wgpu.BufferUsages.copy_dst | wgpu.BufferUsages.map_read,
        .size = @as(u64, dst_stride) * h,
        .mapped_at_creation = @intFromBool(false),
    }) orelse return error.CreateBufferFailed;
    defer buf.release();

    const enc = device.createCommandEncoder(&.{ .label = wgpu.StringView.fromSlice("row-pitch enc") }) orelse
        return error.CreateEncoderFailed;
    enc.copyTextureToBuffer(
        &.{ .texture = tex, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &.{ .buffer = buf, .layout = .{ .offset = 0, .bytes_per_row = dst_stride, .rows_per_image = h } },
        &.{ .width = w, .height = h, .depth_or_array_layers = 1 },
    );
    const cmd = enc.finish(&.{ .label = wgpu.StringView.fromSlice("row-pitch cmd") }) orelse
        return error.FinishFailed;
    queue.submit(&.{cmd});

    const Ctx = struct {
        var done: bool = false;
        var ok: bool = false;
        fn cb(status: wgpu.MapAsyncStatus, msg: wgpu.StringView, _: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
            _ = msg;
            ok = status == .success;
            done = true;
        }
    };
    _ = buf.mapAsync(wgpu.MapModes.read, 0, @as(usize, dst_stride) * h, .{ .callback = Ctx.cb });
    var guard: u32 = 0;
    while (!Ctx.done and guard < 100_000) : (guard += 1) _ = device.poll(true, null);
    if (!Ctx.ok) return error.MapFailed;

    const mapped = buf.getConstMappedRange(0, @as(usize, dst_stride) * h) orelse return error.MapRangeFailed;
    const bytes: [*]const u8 = @ptrCast(mapped);
    var mismatches: u32 = 0;
    for (0..h) |y| for (0..w * 4) |b| {
        if (bytes[y * dst_stride + b] != pixels[y * w * 4 + b]) mismatches += 1;
    };
    return mismatches;
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
