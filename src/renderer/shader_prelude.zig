//! Shared WGSL fragments, prepended to shader sources at comptime.
//!
//! The fullscreen-triangle vertex stage was written out eight times (bloom down/up, ssao, ssr,
//! dof, taa, exposure, tonemap), `rounded_clip_coverage` three times (shape, image, and the text
//! shader), and `srgb_decode` twice — every copy byte-identical apart from whitespace. A shader
//! is only checked when wgpu-native's naga compiles it at runtime, so copies drift silently;
//! `shaders/text_shader_source.wgsl` had already diverged from the live inline text shader before
//! it was deleted. See docs/v2-design.md §4.
//!
//! Everything here is concatenated at comptime, so this costs nothing at runtime.

const std = @import("std");

/// Fullscreen-triangle vertex stage plus the `VOut` it produces. Prepend to any post pass.
pub const fullscreen = @embedFile("shaders/common_fullscreen.wgsl") ++ "\n";

/// `srgb_decode`. Prepend to anything that decodes an sRGB-encoded colour.
pub const color = @embedFile("shaders/common_color.wgsl") ++ "\n";

const rounded_clip_template = @embedFile("shaders/common_rounded_clip.wgsl");

/// `RoundedClip` + its uniform binding + `rounded_clip_coverage`, bound to `group`.
/// Shapes bind it at group 0; images and text at group 1.
pub fn roundedClip(comptime group: u32) []const u8 {
    return comptime replace(
        rounded_clip_template,
        "$CLIP_GROUP",
        std.fmt.comptimePrint("{d}", .{group}),
    ) ++ "\n";
}

/// Comptime string replace. `std.mem.replace` needs a runtime output buffer, and the WGSL is
/// full of braces so `comptimePrint` cannot template it.
fn replace(comptime haystack: []const u8, comptime needle: []const u8, comptime with: []const u8) []const u8 {
    comptime {
        var out: []const u8 = "";
        var rest = haystack;
        while (std.mem.indexOf(u8, rest, needle)) |i| {
            out = out ++ rest[0..i] ++ with;
            rest = rest[i + needle.len ..];
        }
        return out ++ rest;
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

test "the clip token is substituted and none is left behind" {
    // Every group the engine actually binds it at.
    inline for (.{ 0, 1 }) |g| {
        const src = roundedClip(g);
        try std.testing.expect(std.mem.indexOf(u8, src, "$CLIP_GROUP") == null);
        try std.testing.expect(std.mem.indexOf(
            u8,
            src,
            std.fmt.comptimePrint("@group({d}) @binding(0) var<uniform> rounded_clip", .{g}),
        ) != null);
    }
    // The two groups must actually differ, or the substitution is a no-op that happens to pass.
    try std.testing.expect(!std.mem.eql(u8, roundedClip(0), roundedClip(1)));
}

test "prelude fragments declare what callers rely on" {
    try std.testing.expect(std.mem.indexOf(u8, fullscreen, "fn vs_main") != null);
    try std.testing.expect(std.mem.indexOf(u8, fullscreen, "struct VOut") != null);
    try std.testing.expect(std.mem.indexOf(u8, color, "fn srgb_decode") != null);
    try std.testing.expect(std.mem.indexOf(u8, roundedClip(0), "fn rounded_clip_coverage") != null);
}

test "replace handles absent, single and repeated needles" {
    try std.testing.expectEqualStrings("abc", comptime replace("abc", "$X", "1"));
    try std.testing.expectEqualStrings("a1c", comptime replace("a$Xc", "$X", "1"));
    try std.testing.expectEqualStrings("1b1", comptime replace("$Xb$X", "$X", "1"));
    try std.testing.expectEqualStrings("", comptime replace("$X", "$X", ""));
}
