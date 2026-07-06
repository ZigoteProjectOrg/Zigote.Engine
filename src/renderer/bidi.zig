//! UAX#9-lite bidirectional run segmentation for the text shaper.
//!
//! HarfBuzz shapes one buffer in one direction, so a mixed-direction string ("Zigote" inside an
//! Arabic sentence, "2026" inside Hebrew) must be split into direction runs, each shaped with an
//! explicit direction, and the runs emitted in visual order. This module is the pure analysis half:
//! classify codepoints (strong RTL / strong LTR / digits / neutral), resolve neutrals against their
//! strong context (matching context → that direction, conflicting → paragraph direction), and merge
//! into maximal same-direction runs. Digits form LTR runs so multi-digit numbers keep their order
//! inside RTL text, and common separators between digits ("05/07/2026", "14:30") stay LTR too.
//!
//! Deliberately two-level (paragraph + counter-direction runs): no nesting depth, no explicit
//! directional formatting characters (LRE/RLE/LRI/RLI), no bracket-pair rule. Paragraph direction
//! comes from the first strong character (UAX#9 P2/P3); a widget-supplied override can be plumbed
//! through later.

const std = @import("std");

pub const Dir = enum { ltr, rtl };

/// A maximal same-direction byte range of the analyzed text, in logical order.
pub const Run = struct {
    start: u32,
    end: u32,
    dir: Dir,
};

const Class = enum { ltr, rtl, number, neutral };

/// Strong right-to-left codepoints (UAX#9 classes R + AL, by block).
/// The Arabic-Indic digit ranges inside these blocks are carved out by classOf (digits are weak).
pub fn isStrongRtl(cp: u21) bool {
    return switch (cp) {
        0x0590...0x08FF => true, // Hebrew, Arabic, Syriac, Arabic Sup, Thaana, NKo, Samaritan, Mandaic, Arabic Ext-B/A
        0x200F => true, // RIGHT-TO-LEFT MARK
        0xFB1D...0xFDFF => true, // Hebrew + Arabic presentation forms A
        0xFE70...0xFEFF => true, // Arabic presentation forms B
        0x10800...0x10FFF => true, // ancient RTL scripts (Phoenician, Imperial Aramaic, ...)
        0x1E800...0x1EFFF => true, // Adlam, Mende Kikakui, Arabic Mathematical
        else => false,
    };
}

fn isDigit(cp: u21) bool {
    return switch (cp) {
        '0'...'9' => true,
        0x0660...0x0669 => true, // Arabic-Indic digits (class AN — also most-significant-first)
        0x06F0...0x06F9 => true, // Extended Arabic-Indic digits
        else => false,
    };
}

fn isNeutralish(cp: u21) bool {
    return switch (cp) {
        0x09, 0x20 => true, // tab, space
        0x21...0x2F => true, // ! " # $ % & ' ( ) * + , - . /
        0x3A...0x40 => true, // : ; < = > ? @
        0x5B...0x60 => true, // [ \ ] ^ _ `
        0x7B...0x7E => true, // { | } ~
        0xA0...0xBF => true, // NBSP + Latin-1 punctuation/symbols
        0xD7, 0xF7 => true, // × ÷
        0x2000...0x206F => true, // general punctuation (ZWJ/ZWNJ, dashes, quotes; RLM/LRM caught earlier)
        0x20A0...0x20CF => true, // currency symbols
        0x2100...0x2BFF => true, // letterlike, arrows, math operators, misc symbols
        else => false,
    };
}

fn classOf(cp: u21) Class {
    if (isDigit(cp)) return .number; // before the RTL blocks: Arabic-Indic digits live inside them
    if (isStrongRtl(cp)) return .rtl;
    if (cp == 0x200E) return .ltr; // LEFT-TO-RIGHT MARK (inside the general-punctuation block)
    if (isNeutralish(cp)) return .neutral;
    return .ltr;
}

fn decode(text: []const u8, i: usize, end_out: *usize) u21 {
    const len: usize = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
    const end = @min(i + len, text.len);
    end_out.* = end;
    return std.unicode.utf8Decode(text[i..end]) catch 0xFFFD;
}

fn appendRange(allocator: std.mem.Allocator, runs: *std.ArrayList(Run), start: u32, end: u32, dir: Dir) !void {
    if (start == end) return;
    if (runs.items.len > 0) {
        const last = &runs.items[runs.items.len - 1];
        if (last.dir == dir and last.end == start) {
            last.end = end;
            return;
        }
    }
    try runs.append(allocator, .{ .start = start, .end = end, .dir = dir });
}

/// Analyze `text` (valid UTF-8 expected; invalid sequences degrade to U+FFFD = neutral-safe).
/// Returns null when the text contains no strong RTL codepoint — the caller should shape the whole
/// string as a single buffer exactly as before (the fast path; keeps pure-LTR output bit-identical).
/// Otherwise fills `runs` with maximal direction runs in LOGICAL order and returns the paragraph
/// direction; the caller emits runs in visual order (paragraph RTL → reversed run sequence).
pub fn analyze(allocator: std.mem.Allocator, text: []const u8, runs: *std.ArrayList(Run)) !?Dir {
    // Pass 1: paragraph direction = first strong character (digits are weak, skipped); note
    // whether any strong RTL exists at all.
    var paragraph: ?Dir = null;
    var has_rtl = false;
    var i: usize = 0;
    while (i < text.len) {
        var end: usize = undefined;
        const cp = decode(text, i, &end);
        switch (classOf(cp)) {
            .rtl => {
                has_rtl = true;
                if (paragraph == null) paragraph = .rtl;
            },
            .ltr => {
                if (paragraph == null) paragraph = .ltr;
            },
            else => {},
        }
        if (has_rtl) break; // paragraph is already decided at (or before) the first RTL char
        i = end;
    }
    if (!has_rtl) return null;
    const para = paragraph.?;

    // Pass 2: resolve each codepoint to a direction and merge into runs. Digits resolve LTR;
    // a neutral span takes the shared direction of its strong neighbours, else the paragraph
    // direction (UAX#9 N1/N2 without the number-context refinements).
    runs.clearRetainingCapacity();
    var prev: Dir = para; // sos = paragraph direction
    var neutral_start: ?u32 = null;
    i = 0;
    while (i < text.len) {
        var end: usize = undefined;
        const cp = decode(text, i, &end);
        const cls = classOf(cp);
        if (cls == .neutral) {
            if (neutral_start == null) neutral_start = @intCast(i);
            i = end;
            continue;
        }
        const dir: Dir = if (cls == .rtl) .rtl else .ltr;
        if (neutral_start) |ns| {
            try appendRange(allocator, runs, ns, @intCast(i), if (prev == dir) dir else para);
            neutral_start = null;
        }
        try appendRange(allocator, runs, @intCast(i), @intCast(end), dir);
        prev = dir;
        i = end;
    }
    if (neutral_start) |ns| // eos = paragraph direction
        try appendRange(allocator, runs, ns, @intCast(text.len), para);

    return para;
}

// ── Tests ────────────────────────────────────────────────────────────────────

fn analyzeExpect(text: []const u8, expected_para: Dir, expected: []const struct { []const u8, Dir }) !void {
    const allocator = std.testing.allocator;
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(allocator);
    const para = (try analyze(allocator, text, &runs)) orelse return error.ExpectedRtlContent;
    try std.testing.expectEqual(expected_para, para);
    try std.testing.expectEqual(expected.len, runs.items.len);
    for (expected, runs.items) |exp, run| {
        try std.testing.expectEqualStrings(exp[0], text[run.start..run.end]);
        try std.testing.expectEqual(exp[1], run.dir);
    }
    // Runs must tile the text exactly.
    try std.testing.expectEqual(@as(u32, 0), runs.items[0].start);
    try std.testing.expectEqual(@as(u32, @intCast(text.len)), runs.items[runs.items.len - 1].end);
    for (runs.items[1..], runs.items[0 .. runs.items.len - 1]) |cur, prev|
        try std.testing.expectEqual(prev.end, cur.start);
}

test "pure LTR, digits and punctuation take the fast path" {
    const allocator = std.testing.allocator;
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(allocator);
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "Hello, world! 2026", &runs));
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "05/07/2026 14:30", &runs));
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "", &runs));
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "Привет мир 你好", &runs));
}

test "pure RTL is a single run" {
    try analyzeExpect("مرحبا بالعالم", .rtl, &.{.{ "مرحبا بالعالم", .rtl }});
    try analyzeExpect("שלום עולם!", .rtl, &.{.{ "שלום עולם!", .rtl }});
}

test "Latin word inside Arabic splits into counter-direction run" {
    try analyzeExpect("مرحبا Zigote الآن", .rtl, &.{
        .{ "مرحبا ", .rtl },
        .{ "Zigote", .ltr },
        .{ " الآن", .rtl },
    });
}

test "numbers inside RTL text form LTR runs" {
    try analyzeExpect("عام 2026 م", .rtl, &.{
        .{ "عام ", .rtl },
        .{ "2026", .ltr },
        .{ " م", .rtl },
    });
    // Separators between digits stay with the number (dates, times).
    try analyzeExpect("التاريخ: 05/07/2026", .rtl, &.{
        .{ "التاريخ: ", .rtl },
        .{ "05/07/2026", .ltr },
    });
    // Arabic-Indic digits are numbers too (most-significant digit first).
    try analyzeExpect("عام ٢٠٢٦ م", .rtl, &.{
        .{ "عام ", .rtl },
        .{ "٢٠٢٦", .ltr },
        .{ " م", .rtl },
    });
}

test "RTL word inside LTR paragraph" {
    try analyzeExpect("Hello عالم again", .ltr, &.{
        .{ "Hello ", .ltr },
        .{ "عالم", .rtl },
        .{ " again", .ltr },
    });
    // Trailing neutrals resolve to the paragraph direction.
    try analyzeExpect("Hello עולם!", .ltr, &.{
        .{ "Hello ", .ltr },
        .{ "עולם", .rtl },
        .{ "!", .ltr },
    });
}

test "leading number in an RTL paragraph stays LTR" {
    try analyzeExpect("2026 عام", .rtl, &.{
        .{ "2026", .ltr },
        .{ " عام", .rtl },
    });
}
