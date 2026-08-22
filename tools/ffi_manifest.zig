//! Emits the FFI wire contract as JSON — `zig build ffi-manifest`.
//!
//! Reflects over every `extern struct` and enum in `src/abi.zig` with `@sizeOf`, `@offsetOf` and
//! `@typeInfo`, so the numbers are whatever the compiler actually laid out, for the target being
//! built. The C# side asserts against this file rather than against hand-copied literals.
//!
//! Why: the C# mirrors (Zigote.Core/Native/ZgStructs.cs, 500 lines) and the offsets pinned in
//! AbiLayoutTests were both written by hand. The Zig-side `comptime @offsetOf` asserts catch a
//! drift on this side, and the startup check in RendererAbiInfo.Validate catches a change in total
//! size — but a field REORDER that preserves total size passes both, and every field past the
//! change is then silently misread. Nothing compared the two layouts field by field until now.
//!
//! Output goes to stdout, and `zig build ffi-manifest` writes it to `zig-out/ffi-manifest.json`.
//! See docs/v2-design.md §2.5.

const std = @import("std");
const abi = @import("zigote_abi");

/// The wire types, in the order they should appear. A type absent from here is not checked on the
/// C# side, so the test below fails if `abi.zig` declares one that is not listed.
const wire_types = [_]type{
    abi.ZgPaintCommand,
    abi.ZgGlyphRunQuad,
    abi.ZgEvent,
    abi.ZgSize,
    abi.ZgAbiInfo,
    abi.ZgRendererCaps,
    abi.ZgResult,
    abi.ZgTextureLoadItem,
    abi.ZgRenderSettings3D,
    abi.ZgEngineStats,
};

fn typeName(comptime T: type) []const u8 {
    const full = @typeName(T);
    // "abi.ZgPaintCommand" -> "ZgPaintCommand"
    return if (std.mem.lastIndexOfScalar(u8, full, '.')) |i| full[i + 1 ..] else full;
}

fn writeManifest(w: *std.Io.Writer) !void {
    try w.writeAll("{\n  \"types\": [\n");
    inline for (wire_types, 0..) |T, ti| {
        const info = @typeInfo(T);
        try w.print("    {{\n      \"name\": \"{s}\",\n      \"size\": {d},\n      \"align\": {d},\n", .{
            typeName(T), @sizeOf(T), @alignOf(T),
        });
        switch (info) {
            .@"struct" => |st| {
                try w.writeAll("      \"kind\": \"struct\",\n      \"fields\": [\n");
                inline for (st.fields, 0..) |f, fi| {
                    try w.print(
                        "        {{ \"name\": \"{s}\", \"offset\": {d}, \"size\": {d}, \"type\": \"{s}\" }}{s}\n",
                        .{ f.name, @offsetOf(T, f.name), @sizeOf(f.type), @typeName(f.type), if (fi + 1 < st.fields.len) "," else "" },
                    );
                }
                try w.writeAll("      ]\n");
            },
            .@"enum" => |en| {
                try w.writeAll("      \"kind\": \"enum\",\n      \"values\": [\n");
                inline for (en.fields, 0..) |f, fi| {
                    try w.print("        {{ \"name\": \"{s}\", \"value\": {d} }}{s}\n", .{
                        f.name, f.value, if (fi + 1 < en.fields.len) "," else "",
                    });
                }
                try w.writeAll("      ]\n");
            },
            else => @compileError("wire_types must hold only structs and enums: " ++ @typeName(T)),
        }
        try w.print("    }}{s}\n", .{if (ti + 1 < wire_types.len) "," else ""});
    }
    try w.writeAll("  ]\n}\n");
}

pub fn main() !void {
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    var buf: [1 << 16]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    try writeManifest(&stdout.interface);
    try stdout.interface.flush();
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "every type declared in abi.zig is in the manifest" {
    const declared = @typeInfo(abi).@"struct".decls;
    var counted: usize = 0;
    inline for (declared) |d| {
        const T = @field(abi, d.name);
        if (@TypeOf(T) != type) continue;
        const info = @typeInfo(T);
        if (info != .@"struct" and info != .@"enum") continue;
        counted += 1;

        const listed = inline for (wire_types) |W| {
            if (W == T) break true;
        } else false;
        if (!listed) {
            std.debug.print("abi.zig declares {s}, which tools/ffi_manifest.zig does not list —" ++
                " it would cross the ABI unchecked\n", .{d.name});
            return error.WireTypeNotInManifest;
        }
    }
    try std.testing.expectEqual(wire_types.len, counted);
}

test "the manifest is well-formed JSON with the sizes the compiler reports" {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    var w = std.Io.Writer.Allocating.fromArrayList(std.testing.allocator, &buf);
    defer buf = w.toArrayList();
    try writeManifest(&w.writer);

    const json = w.written();
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();

    const types = parsed.value.object.get("types").?.array;
    try std.testing.expectEqual(wire_types.len, types.items.len);

    // Spot-check the two structs the whole per-frame path depends on, against the sizes the ABI
    // contract documents. A change here is a deliberate ABI break, not a refactor.
    for (types.items) |t| {
        const name = t.object.get("name").?.string;
        if (std.mem.eql(u8, name, "ZgPaintCommand")) {
            try std.testing.expectEqual(@as(i64, 112), t.object.get("size").?.integer);
        } else if (std.mem.eql(u8, name, "ZgEvent")) {
            try std.testing.expectEqual(@as(i64, 44), t.object.get("size").?.integer);
        } else if (std.mem.eql(u8, name, "ZgRenderSettings3D")) {
            try std.testing.expectEqual(@as(i64, 280), t.object.get("size").?.integer);
        }
    }
}
