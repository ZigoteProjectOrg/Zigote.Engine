const std = @import("std");
const geometry = @import("../geometry.zig");
const text_mod = @import("../text.zig");

pub const Command = union(enum) {
    rect: Rect,
    border: Border,
    text: Text,
    image: Image,
    clip_start: Clip,
    clip_end,
    push_opacity: PushOpacity,
    pop_opacity,
    shadow: Shadow,
    liquid_glass: LiquidGlass,
    shader_effect: ShaderEffect,
    text_layout: TextLayoutDraw,
    glyph_run: GlyphRun,
    bezier: Bezier,
    polygon: Polygon,
    transform_push: Transform2D,
    transform_pop,
};

/// A clip-stack entry. radius = 0 is a plain rectangular clip (scissor). A positive radius rounds
/// the clip's corners: the scissor still applies the bounding rect (coarse cull) and the shape/
/// text/image pipelines additionally multiply fragment coverage by the rounded-box SDF.
pub const Clip = struct {
    rect: geometry.Rect,
    radius: f32 = 0,
};

/// A 2-D affine transform pushed onto the paint transform stack:
/// x' = a·x + c·y + tx; y' = b·x + d·y + ty (SVG/Canvas coefficient naming, logical pixels).
/// Composes with the current stack top; balanced by `transform_pop`.
pub const Transform2D = struct {
    a: f32 = 1,
    b: f32 = 0,
    c: f32 = 0,
    d: f32 = 1,
    tx: f32 = 0,
    ty: f32 = 0,
};

/// A solid convex/simple polygon fill: an ordered ring of points, triangle-fanned at draw time.
/// Coordinates are logical pixels. Used for chart area fills, pie/donut wedges, and marker symbols
/// that a stroke can't express (filled triangles/diamonds) — anything needing a filled path.
pub const Polygon = struct {
    points: []const [2]f32,
    color: geometry.Color,
};

pub const GlyphRunQuad = extern struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
};

pub const TextLayoutDraw = struct {
    handle: u64,
    draw_x: f32,
    draw_y: f32,
    color: geometry.Color,
};

pub const GlyphRun = struct {
    atlas_handle: u64,
    tint: geometry.Color,
    quads: []const GlyphRunQuad,
};

pub const ShaderEffect = struct {
    bounds: geometry.Rect,
    shader_id: u32,
    params: [8]f32 = [_]f32{0} ** 8,
};

pub const LiquidGlass = struct {
    bounds: geometry.Rect,
    color: geometry.Color,
    radius: f32 = 0,
    thickness: f32 = 0,
    glow_x: f32 = 0,
    glow_y: f32 = 0,
    pinch: f32 = 0,
    /// Adaptive luminance: <0 anchors the backdrop dark (for light content),
    /// >0 anchors it light (for dark content), 0 = off. Magnitude is strength.
    adapt: f32 = 0,
};

pub const Shadow = struct {
    bounds: geometry.Rect,
    color: geometry.Color,
    radius: f32 = 0,
    blur_radius: f32 = 0,
    spread: f32 = 0,
};

/// A cubic Bézier stroke: control points (x0,y0)→(x1,y1)→(x2,y2)→(x3,y3) and a stroke width.
/// Coordinates are in logical pixels (the renderer scales to device pixels). Tessellated into a
/// single anti-aliased ribbon at draw time.
pub const Bezier = struct {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    color: geometry.Color,
    width: f32 = 1,
};

pub const PushOpacity = struct {
    bounds: geometry.Rect,
    alpha: f32,
};

pub const Rect = struct {
    bounds: geometry.Rect,
    color: geometry.Color,
    radius: f32 = 0,
};

pub const Border = struct {
    bounds: geometry.Rect,
    color: geometry.Color,
    radius: f32 = 0,
    width: f32 = 1,
};

pub const Text = struct {
    baseline_x: f32,
    baseline_y: f32,
    text: []const u8,
    color: geometry.Color,
    size: f32,
    line_height: f32,
    font_family: ?[]const u8 = null,
    font_family_fallback: []const []const u8 = &.{},
    font_weight: text_mod.FontWeight = .normal,
    font_style: text_mod.FontStyle = .normal,
    letter_spacing: f32 = 0,
    word_spacing: f32 = 0,
    text_baseline: ?text_mod.TextBaseline = null,
    overflow: ?text_mod.TextOverflow = null,
    /// Gaussian-ish blur radius in text units (CSS text-shadow semantics, radius ≈ 2σ).
    /// 0 = sharp. Used for shadow passes; blurred glyphs bake into the atlas per blur bucket.
    blur: f32 = 0,
};

pub const Image = struct {
    bounds: geometry.Rect,
    width: u32,
    height: u32,
    pixels: []const u8,
    cache_key: ?u64 = null,
    u0: f32 = 0.0,
    v0: f32 = 0.0,
    u1: f32 = 1.0,
    v1: f32 = 1.0,
};

pub const PaintList = struct {
    commands: std.ArrayList(Command) = .empty,

    // Per-frame scratch arena for the text / font-family / glyph-quad blobs that must be copied out
    // of the transient FFI side-buffers (which do not outlive the fill call). Reset — retaining its
    // backing pages — in `clearRetainingCapacity`, so steady-state painting does a bump per blob
    // instead of a libc malloc+free per text/glyph command per frame. Lazily created on first owned
    // append; `blobAllocator` seeds it from the caller's allocator. `null` until first text/glyph run.
    blob_arena: ?std.heap.ArenaAllocator = null,

    pub fn deinit(self: *PaintList, allocator: std.mem.Allocator) void {
        if (self.blob_arena) |*arena| arena.deinit();
        self.commands.deinit(allocator);
    }

    pub fn clearRetainingCapacity(self: *PaintList, allocator: std.mem.Allocator) void {
        // Retain the arena's pages across frames — one reset replaces N per-blob frees.
        if (self.blob_arena) |*arena| _ = arena.reset(.retain_capacity);
        self.commands.clearRetainingCapacity();
        _ = allocator;
    }

    pub fn append(self: *PaintList, allocator: std.mem.Allocator, command: Command) !void {
        try self.commands.append(allocator, command);
    }

    /// The scratch arena's allocator, created on first use from `allocator` as the backing.
    fn blobAllocator(self: *PaintList, allocator: std.mem.Allocator) std.mem.Allocator {
        if (self.blob_arena == null) self.blob_arena = std.heap.ArenaAllocator.init(allocator);
        return self.blob_arena.?.allocator();
    }

    pub fn appendOwnedText(self: *PaintList, allocator: std.mem.Allocator, text: Text) !void {
        const scratch = self.blobAllocator(allocator);

        var command_text = text;
        command_text.text = try scratch.dupe(u8, text.text);

        // Own the font-family name too — it may point at a transient FFI side-buffer that does not
        // outlive this call. Lives in the same per-frame arena; reset with the rest each frame.
        if (text.font_family) |family| {
            command_text.font_family = try scratch.dupe(u8, family);
        }

        try self.commands.append(allocator, .{ .text = command_text });
    }

    pub fn appendOwnedGlyphRun(self: *PaintList, allocator: std.mem.Allocator, run: GlyphRun) !void {
        const scratch = self.blobAllocator(allocator);

        var command_run = run;
        command_run.quads = try scratch.dupe(GlyphRunQuad, run.quads);
        try self.commands.append(allocator, .{ .glyph_run = command_run });
    }

    /// Decode packed point bytes (x,y f32 pairs) from the transient FFI side-buffer into an
    /// arena-owned point ring, then append the polygon command. Reads unaligned (the byte buffer is
    /// a marshalled managed array with no guaranteed 4-byte alignment).
    pub fn appendOwnedPolygon(self: *PaintList, allocator: std.mem.Allocator, bytes: []const u8, color: geometry.Color) !void {
        const count = bytes.len / 8;
        if (count < 3) return;
        const scratch = self.blobAllocator(allocator);
        const pts = try scratch.alloc([2]f32, count);
        var i: usize = 0;
        while (i < count) : (i += 1) {
            pts[i] = @as(*align(1) const [2]f32, @ptrCast(bytes[i * 8 ..].ptr)).*;
        }
        try self.commands.append(allocator, .{ .polygon = .{ .points = pts, .color = color } });
    }

    fn appendOwnedPolygonPts(self: *PaintList, allocator: std.mem.Allocator, poly: Polygon) !void {
        const scratch = self.blobAllocator(allocator);
        var command_poly = poly;
        command_poly.points = try scratch.dupe([2]f32, poly.points);
        try self.commands.append(allocator, .{ .polygon = command_poly });
    }

    pub fn cloneShallow(self: *const PaintList, allocator: std.mem.Allocator) !PaintList {
        var clone = PaintList{};
        errdefer clone.deinit(allocator);

        for (self.commands.items) |command| {
            switch (command) {
                .text => |text| try clone.appendOwnedText(allocator, text),
                .glyph_run => |gr| try clone.appendOwnedGlyphRun(allocator, gr),
                .polygon => |pg| try clone.appendOwnedPolygonPts(allocator, pg),
                else => try clone.append(allocator, command),
            }
        }

        return clone;
    }
};
