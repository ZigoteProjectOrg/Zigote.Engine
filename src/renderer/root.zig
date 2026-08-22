//! The wgpu renderer: 2D UI paint path, 3D forward+ pipeline, text, and the frame pass list.
//!
//! Every `.zig` in this directory must be re-exported here AND referenced from the `test` block
//! below, or Zig's lazy analysis silently drops it from the test binary. Seven files were missing
//! before — including `wgpu_particles`, `mesh_cache` and `uniforms` — so a test written in any of
//! them would never have run. The `no module in this directory is missing` test enforces it now.

pub const backend = @import("backend.zig");
pub const bidi = @import("bidi.zig");
pub const bidi_table = @import("bidi_table.zig");
pub const frame = @import("frame.zig");
pub const freetype_text = @import("freetype_text.zig");
pub const gpu_select = @import("gpu_select.zig");
pub const mesh_cache = @import("mesh_cache.zig");
pub const transient = @import("transient.zig");
pub const uniforms = @import("uniforms.zig");
pub const wgpu = @import("wgpu.zig");
pub const wgpu_3d = @import("wgpu_3d.zig");
pub const wgpu_3d_shaders = @import("wgpu_3d_shaders.zig");
pub const wgpu_blur = @import("wgpu_blur.zig");
pub const wgpu_particles = @import("wgpu_particles.zig");
pub const wgpu_sprites = @import("wgpu_sprites.zig");
pub const wgpu_ui_shaders = @import("wgpu_ui_shaders.zig");
pub const wgpu_ui_util = @import("wgpu_ui_util.zig");

test {
    _ = backend;
    _ = bidi;
    _ = bidi_table;
    _ = frame;
    _ = freetype_text;
    _ = gpu_select;
    _ = mesh_cache;
    _ = transient;
    _ = uniforms;
    _ = wgpu;
    _ = wgpu_3d;
    _ = wgpu_3d_shaders;
    _ = wgpu_blur;
    _ = wgpu_particles;
    _ = wgpu_sprites;
    _ = wgpu_ui_shaders;
    _ = wgpu_ui_util;
}

// A file added to this directory but not to the list above compiles fine and its tests never
// run — the failure mode is silent and permanent, which is how seven of them accumulated. So
// compare the declarations against the directory itself.
test "no module in this directory is missing from the re-export list" {
    const std = @import("std");
    const declared = @typeInfo(@This()).@"struct".decls;

    var io_state = std.Io.Threaded.init(std.testing.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    var dir = std.Io.Dir.cwd().openDir(io, "src/renderer", .{ .iterate = true }) catch |err| switch (err) {
        // Tests are run from the build root; if that ever stops being true, skip rather than
        // fail on an unrelated machine detail.
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer dir.close(io);

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;
        const stem = entry.name[0 .. entry.name.len - ".zig".len];
        if (std.mem.eql(u8, stem, "root")) continue;

        const found = inline for (declared) |d| {
            if (std.mem.eql(u8, d.name, stem)) break true;
        } else false;
        if (!found) {
            std.debug.print("src/renderer/{s} is not re-exported from root.zig\n", .{entry.name});
            return error.ModuleNotReExported;
        }
    }
}
