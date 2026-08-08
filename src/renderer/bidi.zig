//! UAX #9 — the Unicode Bidirectional Algorithm, implemented through rule L2.
//!
//! HarfBuzz shapes one buffer in one direction, so a mixed-direction string ("Zigote" inside an
//! Arabic sentence, "2026" inside Hebrew) must be split into direction runs, each shaped with an
//! explicit direction, and the runs emitted in visual order. This module is the pure analysis half:
//! it resolves an embedding level per character and returns maximal same-level byte runs already
//! reordered by rule L2, so the caller shapes them left to right and is done.
//!
//! The full rule set is here — P2/P3, the explicit formatting characters and isolates (X1–X10),
//! the weak and neutral resolutions including bracket pairs (W1–W7, N0–N2), implicit levels
//! (I1/I2), and reordering (L1/L2). Character classes and the paired-bracket table come from the
//! Unicode Character Database via `bidi_table.zig` (see `tools/gen_bidi_table.py`).
//!
//! Out of scope, deliberately: rule P1 (the caller hands us one paragraph — a paragraph separator
//! inside the text is levelled but does not start a new paragraph), and L3/L4 (combining-mark
//! reordering and mirrored glyph selection), both of which belong to the shaper, not to level
//! resolution. Analysis allocates a handful of per-character arrays, which is why the pure-LTR fast
//! path below matters: it allocates nothing.
//!
//! Verified against the UCD's `BidiCharacterTest.txt` (all 91_707 cases, both explicit paragraph
//! directions and auto) — see `tools/gen_bidi_table.py` for where the data comes from.

const std = @import("std");
const table = @import("bidi_table.zig");

pub const Class = table.Class;
pub const Dir = enum { ltr, rtl };

/// A maximal same-level byte range of the analyzed text, in VISUAL order (rule L2 already applied).
pub const Run = struct {
    start: u32,
    end: u32,
    /// Resolved embedding level; odd is right-to-left.
    level: u8,

    pub fn dir(self: Run) Dir {
        return if (self.level & 1 != 0) .rtl else .ltr;
    }
};

/// UAX #9 caps the explicit embedding depth here (BD2).
const max_depth = 125;
/// BD16 stops collecting bracket pairs once this many are open.
const max_bracket_pairs = 63;

/// Analyze `text` (valid UTF-8 expected; invalid sequences degrade to U+FFFD = neutral-safe).
/// `base` forces the paragraph direction (rule P3's "higher-level protocol" — a widget in an RTL
/// locale wants `.rtl` even for Latin content); null lets P2/P3 derive it from the text.
/// Returns null when the text needs no bidi processing at all — the caller should shape the whole
/// string as a single buffer (the fast path; keeps pure-LTR output bit-identical). Otherwise fills
/// `runs` with maximal same-level runs **in visual order** and returns the paragraph direction.
pub fn analyze(
    allocator: std.mem.Allocator,
    text: []const u8,
    base: ?Dir,
    runs: *std.ArrayList(Run),
) !?Dir {
    if (base != .rtl and !needsBidi(text)) return null;

    var a = try Analysis.init(allocator, text);
    defer a.deinit();

    try a.matchIsolates(); // BD9
    a.para = if (base) |b| @intFromBool(b == .rtl) else a.paragraphLevel(0, a.n); // P2/P3
    a.resolveExplicit(); // X1–X8
    try a.resolveSequences(); // X10, then W1–W7 / N0–N2 / I1–I2 per isolating run sequence
    a.resetLevels(); // L1 (+ levels for the characters X9 removed)
    try a.buildRuns(runs); // level runs, reordered by L2

    return if (a.para & 1 != 0) Dir.rtl else Dir.ltr;
}

/// True when `text` contains anything the algorithm could reorder: right-to-left or Arabic-number
/// characters, or an explicit formatting character. Everything else resolves to level 0 throughout.
fn needsBidi(text: []const u8) bool {
    var i: usize = 0;
    while (i < text.len) {
        var end: usize = undefined;
        const cp = decode(text, i, &end);
        switch (table.classOf(cp)) {
            .r, .al, .an, .rle, .lre, .rlo, .lro, .pdf, .rli, .lri, .fsi, .pdi => return true,
            else => {},
        }
        i = end;
    }
    return false;
}

fn decode(text: []const u8, i: usize, end_out: *usize) u21 {
    const len: usize = std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
    const end = @min(i + len, text.len);
    end_out.* = end;
    return std.unicode.utf8Decode(text[i..end]) catch 0xFFFD;
}

/// X9: these are removed from the sequence the weak/neutral rules run over. They keep an index so
/// byte offsets stay stable; `resetLevels` gives them the level of the character they follow.
fn removedByX9(c: Class) bool {
    return switch (c) {
        .rle, .lre, .rlo, .lro, .pdf, .bn => true,
        else => false,
    };
}

/// The neutral-or-isolate types rules N0–N2 operate on (UAX #9 calls them NI).
fn isNeutral(c: Class) bool {
    return switch (c) {
        .b, .s, .ws, .on, .fsi, .lri, .rli, .pdi => true,
        else => false,
    };
}

/// The direction a character contributes to the neutral and bracket rules: numbers count as R.
fn strongDir(c: Class) ?Class {
    return switch (c) {
        .l => .l,
        .r, .en, .an => .r,
        else => null,
    };
}

fn nextOdd(level: u8) u8 {
    return (level + 1) | 1;
}

fn nextEven(level: u8) u8 {
    return (level + 2) & ~@as(u8, 1);
}

fn levelDir(level: u8) Class {
    return if (level & 1 != 0) .r else .l;
}

const Analysis = struct {
    allocator: std.mem.Allocator,
    n: usize,
    /// Byte offset of each character, plus a terminator at `text.len` (length `n + 1`).
    offs: []u32,
    cp: []u21,
    /// Class as the table gives it — the rules read this for L1 and the N0 combining-mark clause.
    orig: []Class,
    /// Class as the rules resolve it.
    class: []Class,
    level: []u8,
    /// BD9: isolate initiator → its matching PDI, and that PDI back to its initiator; `n` when the
    /// character has no match (or is neither).
    match: []u32,
    para: u8 = 0,

    fn init(allocator: std.mem.Allocator, text: []const u8) !Analysis {
        var n: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            var end: usize = undefined;
            _ = decode(text, i, &end);
            i = end;
            n += 1;
        }

        const offs = try allocator.alloc(u32, n + 1);
        errdefer allocator.free(offs);
        const cp = try allocator.alloc(u21, n);
        errdefer allocator.free(cp);
        const orig = try allocator.alloc(Class, n);
        errdefer allocator.free(orig);
        const class = try allocator.alloc(Class, n);
        errdefer allocator.free(class);
        const level = try allocator.alloc(u8, n);
        errdefer allocator.free(level);
        const match = try allocator.alloc(u32, n);
        errdefer allocator.free(match);

        var a: Analysis = .{
            .allocator = allocator,
            .n = n,
            .offs = offs,
            .cp = cp,
            .orig = orig,
            .class = class,
            .level = level,
            .match = match,
        };

        i = 0;
        var k: usize = 0;
        while (i < text.len) : (k += 1) {
            var end: usize = undefined;
            const decoded = decode(text, i, &end);
            a.offs[k] = @intCast(i);
            a.cp[k] = decoded;
            a.orig[k] = table.classOf(decoded);
            a.class[k] = a.orig[k];
            a.level[k] = 0;
            a.match[k] = @intCast(n);
            i = end;
        }
        a.offs[n] = @intCast(text.len);
        return a;
    }

    fn deinit(a: *Analysis) void {
        a.allocator.free(a.offs);
        a.allocator.free(a.cp);
        a.allocator.free(a.orig);
        a.allocator.free(a.class);
        a.allocator.free(a.level);
        a.allocator.free(a.match);
    }

    /// BD9: pair each isolate initiator with its matching PDI, in both directions.
    fn matchIsolates(a: *Analysis) !void {
        if (a.n == 0) return;
        const stack = try a.allocator.alloc(u32, a.n);
        defer a.allocator.free(stack);
        var top: usize = 0;
        for (a.orig, 0..) |c, i| switch (c) {
            .lri, .rli, .fsi => {
                stack[top] = @intCast(i);
                top += 1;
            },
            .pdi => if (top > 0) {
                top -= 1;
                a.match[stack[top]] = @intCast(i);
                a.match[i] = stack[top];
            },
            else => {},
        };
    }

    /// P2/P3 over `[start, end)`: the first strong character decides, isolate runs are skipped.
    /// Also serves rule X5c, which asks the same question about the text inside an FSI.
    fn paragraphLevel(a: *Analysis, start: usize, end: usize) u8 {
        var i = start;
        while (i < end) : (i += 1) {
            switch (a.orig[i]) {
                .l => return 0,
                .r, .al => return 1,
                // Skip to the matching PDI; an unmatched initiator swallows the rest (BD9/P2).
                .lri, .rli, .fsi => i = if (a.match[i] < end) a.match[i] else end,
                else => {},
            }
        }
        return 0;
    }

    /// X1–X8: walk the directional status stack, assigning each character an embedding level and
    /// applying any directional override to its class.
    fn resolveExplicit(a: *Analysis) void {
        const Entry = struct { level: u8, override: ?Class, isolate: bool };
        var stack: [max_depth + 2]Entry = undefined;
        var sp: usize = 0;
        stack[0] = .{ .level = a.para, .override = null, .isolate = false };

        var overflow_isolate: usize = 0;
        var overflow_embedding: usize = 0;
        var valid_isolate: usize = 0;

        for (0..a.n) |i| {
            switch (a.orig[i]) {
                // X2–X5: embeddings and overrides. X9 removes the character itself.
                .rle, .lre, .rlo, .lro => {
                    a.level[i] = stack[sp].level;
                    const c = a.orig[i];
                    const rtl = c == .rle or c == .rlo;
                    const new_level = if (rtl) nextOdd(stack[sp].level) else nextEven(stack[sp].level);
                    if (new_level <= max_depth and overflow_isolate == 0 and overflow_embedding == 0) {
                        sp += 1;
                        stack[sp] = .{
                            .level = new_level,
                            .override = switch (c) {
                                .rlo => .r,
                                .lro => .l,
                                else => null,
                            },
                            .isolate = false,
                        };
                    } else if (overflow_isolate == 0) overflow_embedding += 1;
                },
                // X5a–X5c: isolates. Unlike embeddings these stay in the sequence, so they take the
                // current level and any override before the new level is pushed.
                .rli, .lri, .fsi => {
                    const c = a.orig[i];
                    const rtl = switch (c) {
                        .rli => true,
                        .lri => false,
                        // X5c: an FSI acts as whichever isolate its own content calls for.
                        else => a.paragraphLevel(i + 1, if (a.match[i] < a.n) a.match[i] else a.n) == 1,
                    };
                    a.level[i] = stack[sp].level;
                    if (stack[sp].override) |o| a.class[i] = o;
                    const new_level = if (rtl) nextOdd(stack[sp].level) else nextEven(stack[sp].level);
                    if (new_level <= max_depth and overflow_isolate == 0 and overflow_embedding == 0) {
                        valid_isolate += 1;
                        sp += 1;
                        stack[sp] = .{ .level = new_level, .override = null, .isolate = true };
                    } else overflow_isolate += 1;
                },
                // X6a: a PDI pops back to its initiator's context, then takes that context's level.
                .pdi => {
                    if (overflow_isolate > 0) {
                        overflow_isolate -= 1;
                    } else if (valid_isolate > 0) {
                        overflow_embedding = 0;
                        while (!stack[sp].isolate) sp -= 1;
                        sp -= 1;
                        valid_isolate -= 1;
                    }
                    a.level[i] = stack[sp].level;
                    if (stack[sp].override) |o| a.class[i] = o;
                },
                // X7: a PDF only pops embeddings, never past an isolate.
                .pdf => {
                    a.level[i] = stack[sp].level;
                    if (overflow_isolate > 0) {
                        // An unterminated isolate outranks it; nothing to pop.
                    } else if (overflow_embedding > 0) {
                        overflow_embedding -= 1;
                    } else if (!stack[sp].isolate and sp >= 1) sp -= 1;
                },
                // X8: paragraph separators always come back to the paragraph level.
                .b => a.level[i] = a.para,
                // X6 skips BN along with the formatting characters — no override, just a level.
                .bn => a.level[i] = stack[sp].level,
                else => {
                    a.level[i] = stack[sp].level;
                    if (stack[sp].override) |o| a.class[i] = o;
                },
            }
        }
    }

    /// X10: build each isolating run sequence (BD13) and run the weak, neutral and implicit rules
    /// over it. Sequences are the unit of resolution — an isolate's content is a separate one.
    fn resolveSequences(a: *Analysis) !void {
        // sos/eos read the levels on both sides of a sequence, and I1/I2 raise the levels of the
        // sequences resolved before this one — so the boundaries come from a snapshot taken while
        // every level is still the explicit one X1–X8 assigned.
        const explicit = try a.allocator.dupe(u8, a.level);
        defer a.allocator.free(explicit);

        // The characters X9 did not remove, in logical order, plus a level run index per character.
        const keep = try a.allocator.alloc(u32, a.n);
        defer a.allocator.free(keep);
        const run_of = try a.allocator.alloc(u32, a.n);
        defer a.allocator.free(run_of);
        var m: usize = 0;
        for (0..a.n) |i| {
            if (removedByX9(a.orig[i])) continue;
            keep[m] = @intCast(i);
            m += 1;
        }
        if (m == 0) return;

        // Level runs: maximal same-level stretches of `keep`.
        const LevelRun = struct { start: u32, end: u32 };
        var level_runs: std.ArrayList(LevelRun) = .empty;
        defer level_runs.deinit(a.allocator);
        {
            var start: usize = 0;
            for (1..m + 1) |k| {
                if (k == m or explicit[keep[k]] != explicit[keep[start]]) {
                    try level_runs.append(a.allocator, .{ .start = @intCast(start), .end = @intCast(k) });
                    start = k;
                }
            }
            for (level_runs.items, 0..) |r, ri|
                for (r.start..r.end) |k| {
                    run_of[keep[k]] = @intCast(ri);
                };
        }

        // Scratch for the characters of one sequence, reused across sequences.
        const seq = try a.allocator.alloc(u32, m);
        defer a.allocator.free(seq);

        for (level_runs.items) |first_run| {
            // BD13: a sequence starts at every level run except those continuing an isolate, i.e.
            // those whose first character is a PDI that matched an initiator.
            const first_char = keep[first_run.start];
            if (a.orig[first_char] == .pdi and a.match[first_char] < a.n) continue;

            var len: usize = 0;
            var run = first_run;
            while (true) {
                for (run.start..run.end) |k| {
                    seq[len] = keep[k];
                    len += 1;
                }
                const last = keep[run.end - 1];
                const continues = switch (a.orig[last]) {
                    .lri, .rli, .fsi => a.match[last] < a.n,
                    else => false,
                };
                if (!continues) break;
                run = level_runs.items[run_of[a.match[last]]];
            }

            // X10: sos/eos come from the higher of this sequence's level and the level on the other
            // side of it (the paragraph level where the text, or the isolate, simply ends).
            const seq_level = explicit[seq[0]];
            const before: u8 = if (first_run.start > 0) explicit[keep[first_run.start - 1]] else a.para;
            const last_char = seq[len - 1];
            const after: u8 = blk: {
                // An isolate initiator with no matching PDI runs to the end of the paragraph.
                switch (a.orig[last_char]) {
                    .lri, .rli, .fsi => if (a.match[last_char] >= a.n) break :blk a.para,
                    else => {},
                }
                const last_run = level_runs.items[run_of[last_char]];
                break :blk if (last_run.end < m) explicit[keep[last_run.end]] else a.para;
            };

            const sos = levelDir(@max(seq_level, before));
            const eos = levelDir(@max(seq_level, after));
            a.resolveWeak(seq[0..len], sos);
            try a.resolveNeutral(seq[0..len], sos, eos, seq_level);
            a.resolveImplicit(seq[0..len]);
        }
    }

    /// W1–W7 over one isolating run sequence.
    fn resolveWeak(a: *Analysis, seq: []const u32, sos: Class) void {
        // W1: a combining mark takes the type of what it attaches to; after an isolate it is ON.
        var prev: Class = sos;
        for (seq) |i| {
            if (a.class[i] == .nsm) a.class[i] = switch (prev) {
                .lri, .rli, .fsi, .pdi => .on,
                else => prev,
            };
            prev = a.class[i];
        }

        // W2: a European number in Arabic-letter context is an Arabic number.
        var strong: Class = sos;
        for (seq) |i| switch (a.class[i]) {
            .l, .r, .al => strong = a.class[i],
            .en => if (strong == .al) {
                a.class[i] = .an;
            },
            else => {},
        };

        // W3: Arabic letters are simply strong RTL from here on.
        for (seq) |i| {
            if (a.class[i] == .al) a.class[i] = .r;
        }

        // W4: a lone separator between two numbers of the same kind joins them ("1,000", "١٬٠٠٠").
        if (seq.len >= 3) for (1..seq.len - 1) |k| {
            const i = seq[k];
            const before = a.class[seq[k - 1]];
            const after = a.class[seq[k + 1]];
            switch (a.class[i]) {
                .es => if (before == .en and after == .en) {
                    a.class[i] = .en;
                },
                .cs => {
                    if (before == .en and after == .en) a.class[i] = .en;
                    if (before == .an and after == .an) a.class[i] = .an;
                },
                else => {},
            }
        };

        // W5: a run of terminators next to a European number joins it ("$1", "1%").
        var k: usize = 0;
        while (k < seq.len) {
            if (a.class[seq[k]] != .et) {
                k += 1;
                continue;
            }
            var end = k;
            while (end < seq.len and a.class[seq[end]] == .et) end += 1;
            const attached = (k > 0 and a.class[seq[k - 1]] == .en) or
                (end < seq.len and a.class[seq[end]] == .en);
            if (attached) for (k..end) |j| {
                a.class[seq[j]] = .en;
            };
            k = end;
        }

        // W6: every separator or terminator that did not join a number is neutral.
        for (seq) |i| switch (a.class[i]) {
            .et, .es, .cs => a.class[i] = .on,
            else => {},
        };

        // W7: a European number in Latin context is just Latin text.
        strong = sos;
        for (seq) |i| switch (a.class[i]) {
            .l, .r => strong = a.class[i],
            .en => if (strong == .l) {
                a.class[i] = .l;
            },
            else => {},
        };
    }

    /// N0–N2 over one isolating run sequence: bracket pairs first, then the remaining neutrals.
    fn resolveNeutral(a: *Analysis, seq: []const u32, sos: Class, eos: Class, level: u8) !void {
        const e = levelDir(level);
        try a.resolveBrackets(seq, sos, e);

        // N1: a stretch of neutrals between two like directions takes that direction; N2: whatever
        // is left takes the embedding direction.
        var k: usize = 0;
        while (k < seq.len) {
            if (!isNeutral(a.class[seq[k]])) {
                k += 1;
                continue;
            }
            var end = k;
            while (end < seq.len and isNeutral(a.class[seq[end]])) end += 1;
            const before = if (k > 0) strongDir(a.class[seq[k - 1]]) orelse sos else sos;
            const after = if (end < seq.len) strongDir(a.class[seq[end]]) orelse eos else eos;
            const resolved = if (before == after) before else e;
            for (k..end) |j| a.class[seq[j]] = resolved;
            k = end;
        }
    }

    /// N0 with BD16: a bracket pair takes the direction its contents establish, so "(Zigote)" keeps
    /// its parentheses upright inside an Arabic sentence instead of letting them flip.
    fn resolveBrackets(a: *Analysis, seq: []const u32, sos: Class, e: Class) !void {
        const Pair = struct { open: u32, close: u32 };
        var pairs: std.ArrayList(Pair) = .empty;
        defer pairs.deinit(a.allocator);

        // BD16: match closing brackets against a stack of the opening ones, both taken from the
        // characters still classified ON.
        {
            const Open = struct { close_cp: u21, pos: u32 };
            var stack: [max_bracket_pairs]Open = undefined;
            var top: usize = 0;
            for (seq, 0..) |i, k| {
                if (a.class[i] != .on) continue;
                if (table.pairedBracket(a.cp[i])) |close_cp| {
                    if (top == max_bracket_pairs) break; // stack full: stop BD16 for this sequence
                    stack[top] = .{ .close_cp = table.canonicalBracket(close_cp), .pos = @intCast(k) };
                    top += 1;
                    continue;
                }
                const cp = table.canonicalBracket(a.cp[i]);
                var d = top;
                while (d > 0) {
                    d -= 1;
                    if (stack[d].close_cp != cp) continue;
                    try pairs.append(a.allocator, .{ .open = stack[d].pos, .close = @intCast(k) });
                    top = d; // the pairs it skipped over are discarded
                    break;
                }
            }
            std.mem.sort(Pair, pairs.items, {}, struct {
                fn lessThan(_: void, x: Pair, y: Pair) bool {
                    return x.open < y.open;
                }
            }.lessThan);
        }

        for (pairs.items) |pair| {
            // N0 b/c: the embedding direction inside wins outright; otherwise the opposite
            // direction only wins if the context before the pair agrees with it.
            var found_opposite = false;
            var resolved: ?Class = null;
            for (pair.open + 1..pair.close) |k| {
                const s = strongDir(a.class[seq[k]]) orelse continue;
                if (s == e) {
                    resolved = e;
                    break;
                }
                found_opposite = true;
            }
            if (resolved == null and found_opposite) {
                var context = sos;
                var k = pair.open;
                while (k > 0) {
                    k -= 1;
                    if (strongDir(a.class[seq[k]])) |s| {
                        context = s;
                        break;
                    }
                }
                resolved = if (context == e) e else context;
            }
            // N0 d: no strong type inside — the brackets stay neutral for N1/N2 to sort out.
            const dir = resolved orelse continue;
            a.class[seq[pair.open]] = dir;
            a.class[seq[pair.close]] = dir;
            // Combining marks that followed a bracket track the direction it just took.
            for ([2]u32{ pair.open, pair.close }) |at| {
                var k = at + 1;
                while (k < seq.len and a.orig[seq[k]] == .nsm) : (k += 1) a.class[seq[k]] = dir;
            }
        }
    }

    /// I1/I2: turn resolved classes into level bumps away from the embedding direction.
    fn resolveImplicit(a: *Analysis, seq: []const u32) void {
        for (seq) |i| {
            if (a.level[i] & 1 == 0) {
                switch (a.class[i]) {
                    .r => a.level[i] += 1,
                    .an, .en => a.level[i] += 2,
                    else => {},
                }
            } else switch (a.class[i]) {
                .l, .en, .an => a.level[i] += 1,
                else => {},
            }
        }
    }

    /// Level the characters X9 removed (5.2: they follow the character before them), then apply L1.
    fn resetLevels(a: *Analysis) void {
        var last: u8 = a.para;
        for (0..a.n) |i| {
            if (removedByX9(a.orig[i])) a.level[i] = last else last = a.level[i];
        }

        // L1: separators, and any whitespace or isolate formatting trailing them or the line, come
        // back to the paragraph level — so a trailing space in an RTL line hangs on the right side.
        var at_boundary = true; // the end of the line counts as one
        var i = a.n;
        while (i > 0) {
            i -= 1;
            switch (a.orig[i]) {
                .b, .s => {
                    a.level[i] = a.para;
                    at_boundary = true;
                },
                .ws, .lri, .rli, .fsi, .pdi, .rle, .lre, .rlo, .lro, .pdf, .bn => {
                    if (at_boundary) a.level[i] = a.para;
                },
                else => at_boundary = false,
            }
        }
    }

    /// Collect same-level byte runs, then reorder them by L2.
    fn buildRuns(a: *Analysis, runs: *std.ArrayList(Run)) !void {
        runs.clearRetainingCapacity();
        for (0..a.n) |i| {
            if (runs.items.len > 0) {
                const last = &runs.items[runs.items.len - 1];
                if (last.level == a.level[i]) {
                    last.end = a.offs[i + 1];
                    continue;
                }
            }
            try runs.append(a.allocator, .{ .start = a.offs[i], .end = a.offs[i + 1], .level = a.level[i] });
        }
        if (runs.items.len == 0) return;

        // L2: from the highest level down to the lowest odd level, reverse every contiguous
        // stretch of runs at or above that level.
        var highest: u8 = 0;
        var lowest_odd: u8 = std.math.maxInt(u8);
        for (runs.items) |r| {
            highest = @max(highest, r.level);
            if (r.level & 1 != 0) lowest_odd = @min(lowest_odd, r.level);
        }
        if (lowest_odd > highest) return;

        var level = highest;
        while (level >= lowest_odd) : (level -= 1) {
            var k: usize = 0;
            while (k < runs.items.len) {
                if (runs.items[k].level < level) {
                    k += 1;
                    continue;
                }
                var end = k;
                while (end < runs.items.len and runs.items[end].level >= level) end += 1;
                std.mem.reverse(Run, runs.items[k..end]);
                k = end;
            }
        }
    }
};

// ── Tests ────────────────────────────────────────────────────────────────────

/// Assert the visual run sequence: each entry is the run's text and its direction.
fn analyzeExpect(text: []const u8, expected_para: Dir, expected: []const struct { []const u8, Dir }) !void {
    const allocator = std.testing.allocator;
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(allocator);
    const para = (try analyze(allocator, text, null, &runs)) orelse return error.ExpectedBidiContent;
    try std.testing.expectEqual(expected_para, para);
    try std.testing.expectEqual(expected.len, runs.items.len);
    for (expected, runs.items) |exp, run| {
        try std.testing.expectEqualStrings(exp[0], text[run.start..run.end]);
        try std.testing.expectEqual(exp[1], run.dir());
    }
    // Runs must still tile the text exactly, in whatever order they ended up.
    var covered: usize = 0;
    for (runs.items) |r| covered += r.end - r.start;
    try std.testing.expectEqual(text.len, covered);
}

test "text with nothing to reorder takes the fast path" {
    const allocator = std.testing.allocator;
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(allocator);
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "Hello, world! 2026", null, &runs));
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "05/07/2026 14:30", null, &runs));
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "", null, &runs));
    try std.testing.expectEqual(@as(?Dir, null), try analyze(allocator, "Привет мир 你好", null, &runs));
}

test "pure RTL is a single run" {
    try analyzeExpect("مرحبا بالعالم", .rtl, &.{.{ "مرحبا بالعالم", .rtl }});
    try analyzeExpect("שלום עולם!", .rtl, &.{.{ "שלום עולם!", .rtl }});
}

test "Latin word inside Arabic splits into counter-direction run" {
    // Visual order: the RTL paragraph runs right to left, so the trailing Arabic run comes first.
    try analyzeExpect("مرحبا Zigote الآن", .rtl, &.{
        .{ " الآن", .rtl },
        .{ "Zigote", .ltr },
        .{ "مرحبا ", .rtl },
    });
}

test "numbers inside RTL text form their own runs" {
    try analyzeExpect("عام 2026 م", .rtl, &.{
        .{ " م", .rtl },
        .{ "2026", .ltr },
        .{ "عام ", .rtl },
    });
    // Separators between digits join the number (dates, times).
    try analyzeExpect("التاريخ: 05/07/2026", .rtl, &.{
        .{ "05/07/2026", .ltr },
        .{ "التاريخ: ", .rtl },
    });
    // Arabic-Indic digits are Arabic numbers: level 2 inside RTL, still most-significant-first.
    try analyzeExpect("عام ٢٠٢٦ م", .rtl, &.{
        .{ " م", .rtl },
        .{ "٢٠٢٦", .ltr },
        .{ "عام ", .rtl },
    });
}

test "RTL word inside LTR paragraph" {
    try analyzeExpect("Hello عالم again", .ltr, &.{
        .{ "Hello ", .ltr },
        .{ "عالم", .rtl },
        .{ " again", .ltr },
    });
    // L1 pulls the trailing exclamation back to the paragraph level.
    try analyzeExpect("Hello עולם!", .ltr, &.{
        .{ "Hello ", .ltr },
        .{ "עולם", .rtl },
        .{ "!", .ltr },
    });
}

test "leading number in an RTL paragraph" {
    try analyzeExpect("2026 عام", .rtl, &.{
        .{ " عام", .rtl },
        .{ "2026", .ltr },
    });
}

test "N0: brackets take the direction of what they enclose" {
    // The Latin content is the embedding direction's opposite and the context before the pair is
    // Arabic, so N0 keeps the parentheses with the Arabic text and only "Zigote" flips.
    try analyzeExpect("مرحبا (Zigote) الآن", .rtl, &.{
        .{ ") الآن", .rtl },
        .{ "Zigote", .ltr },
        .{ "مرحبا (", .rtl },
    });
    // Arabic inside brackets in an LTR paragraph: the brackets follow the paragraph, not the content.
    try analyzeExpect("Read (عالم) now", .ltr, &.{
        .{ "Read (", .ltr },
        .{ "عالم", .rtl },
        .{ ") now", .ltr },
    });
}

test "N0: a bracket pair with no strong content stays neutral" {
    // "()" resolves through N1/N2 like any other neutral, so it merges into the Arabic run.
    try analyzeExpect("مرحبا () الآن", .rtl, &.{.{ "مرحبا () الآن", .rtl }});
}

test "explicit embeddings raise the level" {
    // RLE…PDF pushes "b c" to level 2 inside an LTR paragraph — still left to right, but one level
    // in, so it reorders as a unit against anything RTL around it.
    try analyzeExpect("a \u{202B}b c\u{202C} d", .ltr, &.{
        .{ "a \u{202B}", .ltr },
        .{ "b c\u{202C}", .ltr },
        .{ " d", .ltr },
    });
    // An override forces the class outright: inside LRO the Hebrew runs left to right. P2 still
    // reads the Hebrew as the first strong character, so the paragraph itself is RTL.
    try analyzeExpect("\u{202D}שלום\u{202C}", .rtl, &.{
        .{ "\u{202C}", .rtl },
        .{ "שלום", .ltr },
        .{ "\u{202D}", .rtl },
    });
}

test "isolates keep their content from affecting the surrounding text" {
    // The RTL isolate reorders internally; to the outer paragraph it is a single neutral, so the
    // text around it stays put and "c" does not get dragged leftwards.
    try analyzeExpect("a \u{2067}b עברית\u{2069} c", .ltr, &.{
        .{ "a \u{2067}", .ltr },
        .{ " עברית", .rtl },
        .{ "b", .ltr },
        .{ "\u{2069} c", .ltr },
    });
    // FSI picks its direction from its own first strong character (X5c) — here, Hebrew.
    try analyzeExpect("a \u{2068}עברית b\u{2069} c", .ltr, &.{
        .{ "a \u{2068}", .ltr },
        .{ "b", .ltr },
        .{ "עברית ", .rtl },
        .{ "\u{2069} c", .ltr },
    });
}

test "nesting goes deeper than two levels" {
    // Hebrew inside an LTR isolate inside an RTL isolate inside an LTR paragraph: levels 0…3.
    try analyzeExpect("a \u{2067}b \u{2066}c עברית\u{2069} d\u{2069} e", .ltr, &.{
        .{ "a \u{2067}", .ltr },
        .{ "b \u{2066}c ", .ltr },
        .{ "עברית", .rtl },
        .{ "\u{2069} d", .ltr },
        .{ "\u{2069} e", .ltr },
    });
    // Without any explicit formatting the deepest a mixed string goes is level 2.
    try analyzeExpect("עברית a עברית b עברית", .rtl, &.{
        .{ " עברית", .rtl },
        .{ "b", .ltr },
        .{ " עברית ", .rtl },
        .{ "a", .ltr },
        .{ "עברית ", .rtl },
    });
}

test "combining marks inherit the direction they attach to (W1)" {
    // A combining mark after Hebrew is Hebrew; the same mark after Latin is Latin.
    try analyzeExpect("a\u{0301} שלום", .ltr, &.{
        .{ "a\u{0301} ", .ltr },
        .{ "שלום", .rtl },
    });
}

test "table lookups agree with the ranges they came from" {
    try std.testing.expectEqual(Class.l, table.classOf('A'));
    try std.testing.expectEqual(Class.en, table.classOf('7'));
    try std.testing.expectEqual(Class.r, table.classOf(0x05D0)); // HEBREW LETTER ALEF
    try std.testing.expectEqual(Class.al, table.classOf(0x0627)); // ARABIC LETTER ALEF
    try std.testing.expectEqual(Class.an, table.classOf(0x0660)); // ARABIC-INDIC DIGIT ZERO
    try std.testing.expectEqual(Class.nsm, table.classOf(0x064B)); // ARABIC FATHATAN
    try std.testing.expectEqual(Class.rli, table.classOf(0x2067));
    try std.testing.expectEqual(Class.al, table.classOf(0x070E)); // unassigned, defaults to AL
    try std.testing.expectEqual(@as(?u21, ')'), table.pairedBracket('('));
    try std.testing.expectEqual(@as(?u21, 0x3009), table.pairedBracket(0x2329)); // canonical fold
    try std.testing.expectEqual(@as(?u21, null), table.pairedBracket('a'));
}
