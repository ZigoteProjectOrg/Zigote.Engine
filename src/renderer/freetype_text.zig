const std = @import("std");
const wgpu = @import("wgpu");
const zg = @import("../root.zig");
const bidi = @import("bidi.zig");

const ft = @cImport({
    @cInclude("ft2build.h");
    @cInclude("freetype/freetype.h");
    @cInclude("freetype/tttables.h");
    @cInclude("freetype/ftoutln.h");
});

const hb = @cImport({
    @cInclude("hb.h");
});

extern fn hb_ft_font_create_referenced(ft_face: ft.FT_Face) ?*hb.hb_font_t;
extern fn hb_ft_font_set_funcs(font: ?*hb.hb_font_t) void;

pub const TextVertex = extern struct {
    position: [2]f32,
    uv: [2]f32,
    // unorm8x4 (4 B, was 16 B) — lossless for the 8-bit source color, delivered to WGSL as a
    // normalized vec4<f32>, so the text shader is unchanged. The shaped-run cache stores geometry
    // (ShapedGlyph), not colored vertices, so color is applied here at emit and this is a clean shrink.
    color: [4]u8,
};

const Atlas = struct {
    // Coverage glyphs are compact: a plain UI app fits comfortably in 1024², while editor
    // workloads touch several font sizes, weights and large Unicode ranges. Start small and
    // grow ×2 on overflow (tryGrowAtlas) up to 4096² — supported by every WebGPU/Metal target —
    // so small apps don't pay a 16 MB CPU + 16 MB GPU backing store they never fill.
    pub const initial: u32 = 1024;
    pub const max: u32 = 4096;
};

const ColorAtlas = struct {
    pub const width: u32 = 1024;
    pub const height: u32 = 1024;
};

/// GPU half of the color emoji atlas, created lazily on first color-glyph bake.
pub const ColorAtlasGpu = struct {
    texture: *wgpu.Texture,
    view: *wgpu.TextureView,
    bind_group: *wgpu.BindGroup,
};

/// Pixel-space quad for a color (BGRA) emoji glyph, ready for the image pipeline.
/// Positions are in device pixels; UV maps into the RGBA color atlas.
pub const PendingColorQuad = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
};

const Glyph = struct {
    atlas_x: u32,
    atlas_y: u32,
    width: u32,
    height: u32,

    bearing_x: f32,
    bearing_y: f32,
    advance_x: f32,
};

/// The style a paint command asked for. The renderer resolves it against what the face actually
/// provides: a face that already is bold/italic (real weight variants registered by the host)
/// renders untouched, while a face that lacks the style gets a synthetic fallback — emboldened
/// outlines for weight, a shear for oblique — so `fontWeight`/`fontStyle` never silently no-op.
pub const FaceStyle = struct {
    weight: u16 = 400,
    italic: bool = false,
};

/// What must actually be synthesized for (face, style): `embolden` is the outline strength in
/// pixels (0 = none), `oblique` a 12° shear. Derived per glyph bake / shaped run — both are cheap
/// table reads.
const Synth = struct {
    embolden: f32 = 0,
    oblique: bool = false,

    fn none(self: Synth) bool {
        return self.embolden == 0 and !self.oblique;
    }

    /// Embolden strength in 26.6 fixed point, quantized — also the value glyph/run cache keys
    /// hash, so two requests that synthesize identically share one raster.
    fn strength266(self: Synth) i32 {
        return @intFromFloat(@round(self.embolden * 64.0));
    }
};

/// The face's own weight class (OS/2 usWeightClass, the field real weight variants like
/// Inter-Medium carry), falling back to the bold style flag for fonts without an OS/2 table.
fn faceWeightClass(face: ft.FT_Face) u16 {
    if (ft.FT_Get_Sfnt_Table(face, ft.FT_SFNT_OS2)) |table| {
        const os2: *const ft.TT_OS2 = @ptrCast(@alignCast(table));
        const w = os2.usWeightClass;
        if (w >= 100 and w <= 1000) return @intCast(w);
    }
    return if ((face.*.style_flags & ft.FT_STYLE_FLAG_BOLD) != 0) 700 else 400;
}

/// Decide what to synthesize: only the style the face cannot deliver itself. A requested weight
/// above the face's own gets outline emboldening proportional to the gap (700 over a 400 face is
/// the classic size/24 full-bold strength); requested italic on a non-italic face gets a shear.
/// Lighter-than-face weights are never synthesized — thinning outlines destroys small sizes.
fn synthFor(face: ft.FT_Face, style: FaceStyle, pixel_size: u16) Synth {
    var synth = Synth{};
    if (style.italic and (face.*.style_flags & ft.FT_STYLE_FLAG_ITALIC) == 0)
        synth.oblique = true;
    // The OS/2 read is gated on the request being bolder than regular, so the overwhelmingly
    // common regular-on-regular path stays table-lookup-free.
    if (style.weight > 400) {
        const own = faceWeightClass(face);
        if (style.weight > own + 50) {
            const gap: f32 = @floatFromInt(style.weight - own);
            const t = @min(gap / 300.0, 5.0 / 3.0);
            synth.embolden = t * @as(f32, @floatFromInt(pixel_size)) / 24.0;
        }
    }
    return synth;
}

/// One glyph's pixel-relative position (origin = draw baseline) and atlas UVs.
/// Stored in TextLayoutEntry so the same layout can be drawn at any position.
pub const TextLayoutGlyph = struct {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
};

/// A visual caret stop produced from HarfBuzz clusters. `text_offset` is a UTF-8 byte offset.
pub const TextCaret = struct {
    text_offset: u32,
    x: f32,
    y: f32,
};

/// Pre-computed, cached text layout — produced by appendTextLayout().
///
/// The glyph quads bake atlas UVs, so they go stale when the coverage atlas is reset (overflow zeroes
/// `atlas_pixels` and re-packs from the origin). Unlike `shaped_run_cache`, layout entries are
/// addressed by long-lived u64 handles held by widgets, so they can't simply be dropped on reset
/// without blanking the holder's text. Instead each entry remembers the atlas `generation` its UVs
/// were baked against plus enough source (`text` + resolved `font_family` + `size`/`line_height`) to
/// re-shape itself: `appendLayoutGlyphs` re-bakes lazily on the next draw whose generation is stale.
/// `text`/`font_family` are owned copies (the originating zg.paint.Text does not outlive the call).
pub const TextLayoutEntry = struct {
    glyphs: []TextLayoutGlyph,
    /// Color-emoji quads baked alongside the coverage glyphs, in the same origin-relative logical
    /// space. Drawn through the image pipeline (see appendLayoutGlyphs); UVs go stale together with
    /// `glyphs` on a generation bump, and re-bake in the same reshape.
    color_quads: []PendingColorQuad = &.{},
    carets: []TextCaret,
    width: f32,
    height: f32,
    text: []const u8,
    font_family: []const u8,
    size: f32,
    line_height: f32,
    letter_spacing: f32 = 0,
    word_spacing: f32 = 0,
    style: FaceStyle = .{},
    raster_scale: f32 = 1,
    generation: u64,
};

const ShapedGlyph = struct {
    glyph_index: u32,
    cluster: u32,
    x_offset: f32,
    y_offset: f32,
    x_advance: f32,
    y_advance: f32,
};

/// HarfBuzz output cached independently of raster size and atlas coordinates. This is important for
/// editors: a run is shaped once, but can be rasterized at 1x/2x after a window moves between
/// displays, and atlas resets never invalidate the expensive shaping result.
const ShapedRun = struct {
    glyphs: []ShapedGlyph,
    advance_x: f32,
    advance_y: f32,
};

pub const AtlasDirtyRect = struct {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
};

pub const FreeTypeTextRenderer = struct {
    allocator: std.mem.Allocator,
    default_font_family: []const u8,

    library: ft.FT_Library = null,
    faces: std.StringHashMap(ft.FT_Face),

    glyphs: std.AutoHashMap(u64, Glyph),

    atlas_pixels: []u8,
    atlas_width: u32 = Atlas.initial,
    atlas_height: u32 = Atlas.initial,
    atlas_dirty: bool = true,
    dirty_min_x: u32 = 0,
    dirty_min_y: u32 = 0,
    dirty_max_x: u32 = Atlas.initial,
    dirty_max_y: u32 = Atlas.initial,

    pen_x: u32 = 1,
    pen_y: u32 = 1,
    row_h: u32 = 0,

    // Bumped every time the coverage atlas is reset (overflow). Cached `TextLayoutEntry`s store the
    // generation their UVs were baked against; a mismatch at draw time triggers a lazy re-shape.
    atlas_generation: u64 = 0,

    // GPU resources — only valid when `gpu_enabled` (the wgpu backend). The Metal backend
    // constructs this renderer CPU-only (initCpu): it reuses the FreeType/HarfBuzz shaping and
    // the coverage atlas_pixels, and uploads the atlas to its own Metal texture.
    gpu_enabled: bool = false,
    // Retained for out-of-frame GPU resource creation: atlas growth and the lazy color atlas can
    // both trigger from zigote_text_layout_create (host measure phase), outside any render call.
    gpu_device: ?*wgpu.Device = null,
    texture: *wgpu.Texture,
    texture_view: *wgpu.TextureView,
    sampler: *wgpu.Sampler,
    bind_group_layout: *wgpu.BindGroupLayout,
    pipeline_layout: *wgpu.PipelineLayout,
    bind_group: *wgpu.BindGroup,
    pipeline: *wgpu.RenderPipeline,

    layout_cache: std.AutoHashMap(u64, TextLayoutEntry),
    next_layout_handle: u64,

    // Content-addressed shaped-run cache: text re-drawn every frame (e.g. an editor rendering
    // continuously) is shaped by HarfBuzz once, then re-emitted from cached glyphs. Keyed by
    // (text, font_family, pixel_size); cleared when a face is (re)registered.
    shaped_run_cache: std.AutoHashMap(u64, ShapedRun),

    // ── Color emoji atlas (RGBA 1024×1024) ────────────────────────────────────
    // Fully lazy: the 4 MB CPU buffer and the GPU texture/view/bind-group are only created when
    // the first color (BGRA) glyph is actually baked — registering an emoji family alone costs
    // nothing, so apps that never render emoji never pay for the atlas (per window).
    color_atlas_pixels: ?[]u8,
    color_atlas_dirty: bool,
    color_pen_x: u32,
    color_pen_y: u32,
    color_row_h: u32,
    color_gpu: ?ColorAtlasGpu,
    color_glyphs: std.AutoHashMap(u64, Glyph),
    emoji_families: std.ArrayList([]const u8),
    pending_color_quads: std.ArrayList(PendingColorQuad),
    // Memoizes emojiHasGlyph(cp): the segmenter probes FT_Get_Char_Index per codepoint every
    // frame, but the answer only changes when a font family is (re)registered — so cache it and
    // invalidate on loadFontFromCPath / addEmojiFontFamily.
    emoji_glyph_cache: std.AutoHashMap(u21, bool),

    // ── Script fallback ───────────────────────────────────────────────────────
    // Families to try, in order, for a codepoint the requested face has no glyph for. A UI font
    // covers the scripts its designer drew and nothing else — Inter is Latin, Greek and Cyrillic —
    // so without this, any app displaying text it did not author (a music player showing tags, a
    // file manager showing filenames) renders Japanese, Korean, Chinese, Arabic and Thai as tofu.
    // The host registers whatever the platform ships; see addFallbackFontFamily.
    fallback_families: std.ArrayList([]const u8),
    // Memoizes routeFor. Keyed on (primary family, codepoint) because coverage is per face, and
    // invalidated whenever a face or fallback is registered.
    route_cache: std.AutoHashMap(u64, i8),

    /// CPU-only construction for native backends (Metal/…): reuses all FreeType/HarfBuzz shaping
    /// and the coverage atlas; the caller uploads `atlasPixels()` to its own GPU texture.
    pub fn initCpu(
        allocator: std.mem.Allocator,
        fonts: []const zg.FontAsset,
        default_font_family: []const u8,
    ) !FreeTypeTextRenderer {
        return init(allocator, null, undefined, null, fonts, default_font_family);
    }

    pub fn init(
        allocator: std.mem.Allocator,
        device: ?*wgpu.Device,
        format: wgpu.TextureFormat,
        // The GpuUi-owned rounded-clip bind group layout, bound at group(1) of the text pipeline
        // (borrowed — GpuUi releases it). Required whenever `device` is non-null.
        clip_bind_group_layout: ?*wgpu.BindGroupLayout,
        fonts: []const zg.FontAsset,
        default_font_family: []const u8,
    ) !FreeTypeTextRenderer {
        var self: FreeTypeTextRenderer = undefined;

        self.allocator = allocator;
        // Own the name: the caller's slice (a marshaled C# string) does not outlive init,
        // so storing it directly would dangle and make every face lookup fail at render time.
        self.default_font_family = try allocator.dupe(u8, default_font_family);
        self.library = null;
        self.faces = std.StringHashMap(ft.FT_Face).init(allocator);
        self.glyphs = std.AutoHashMap(u64, Glyph).init(allocator);

        self.atlas_width = Atlas.initial;
        self.atlas_height = Atlas.initial;
        self.atlas_pixels = try allocator.alloc(u8, @as(usize, self.atlas_width) * self.atlas_height);
        @memset(self.atlas_pixels, 0);

        self.atlas_dirty = true;
        self.dirty_min_x = 0;
        self.dirty_min_y = 0;
        self.dirty_max_x = self.atlas_width;
        self.dirty_max_y = self.atlas_height;
        self.pen_x = 1;
        self.pen_y = 1;
        self.row_h = 0;
        self.atlas_generation = 0;

        self.layout_cache = std.AutoHashMap(u64, TextLayoutEntry).init(allocator);
        self.next_layout_handle = 1;
        self.shaped_run_cache = std.AutoHashMap(u64, ShapedRun).init(allocator);

        self.color_atlas_pixels = null;
        self.color_gpu = null;
        self.color_atlas_dirty = false;
        self.color_pen_x = 1;
        self.color_pen_y = 1;
        self.color_row_h = 0;
        self.color_glyphs = std.AutoHashMap(u64, Glyph).init(allocator);
        self.emoji_families = .empty;
        self.pending_color_quads = .empty;
        self.emoji_glyph_cache = std.AutoHashMap(u21, bool).init(allocator);
        self.fallback_families = .empty;
        self.route_cache = std.AutoHashMap(u64, i8).init(allocator);

        if (ft.FT_Init_FreeType(&self.library) != 0) {
            return error.FreeTypeInitFailed;
        }
        errdefer _ = ft.FT_Done_FreeType(self.library);

        for (fonts) |asset| {
            var face: ft.FT_Face = null;
            switch (asset.source) {
                .platform_path => |path| {
                    if (ft.FT_New_Face(self.library, path.ptr, 0, &face) != 0) {
                        return error.FreeTypeFaceFailed;
                    }
                },
                .embedded => |data| {
                    if (ft.FT_New_Memory_Face(self.library, data.ptr, @intCast(data.len), 0, &face) != 0) {
                        return error.FreeTypeFaceFailed;
                    }
                },
            }
            const owned_name = try allocator.dupe(u8, asset.name);
            errdefer allocator.free(owned_name);
            try self.faces.put(owned_name, face);
        }

        self.gpu_device = device;
        if (device) |dev| {
            self.texture = dev.createTexture(&.{
                .label = wgpu.StringView.fromSlice("zigote text atlas"),
                .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
                .dimension = .@"2d",
                .size = .{
                    .width = self.atlas_width,
                    .height = self.atlas_height,
                    .depth_or_array_layers = 1,
                },
                .format = .r8_unorm,
                .mip_level_count = 1,
                .sample_count = 1,
            }) orelse return error.WgpuTextTextureUnavailable;
            errdefer self.texture.release();

            self.texture_view = self.texture.createView(null) orelse return error.WgpuTextTextureViewUnavailable;
            errdefer self.texture_view.release();

            self.sampler = dev.createSampler(&.{
                .label = wgpu.StringView.fromSlice("zigote text sampler"),
                .address_mode_u = .clamp_to_edge,
                .address_mode_v = .clamp_to_edge,
                .address_mode_w = .clamp_to_edge,
                .mag_filter = .linear,
                .min_filter = .linear,
                .mipmap_filter = .nearest,
            }) orelse return error.WgpuTextSamplerUnavailable;
            errdefer self.sampler.release();

            self.bind_group_layout = createTextBindGroupLayout(dev) orelse {
                return error.WgpuTextBindGroupLayoutUnavailable;
            };
            errdefer self.bind_group_layout.release();

            self.pipeline_layout = createTextPipelineLayout(
                dev,
                self.bind_group_layout,
                clip_bind_group_layout orelse return error.WgpuTextPipelineLayoutUnavailable,
            ) orelse return error.WgpuTextPipelineLayoutUnavailable;
            errdefer self.pipeline_layout.release();

            self.bind_group = createTextBindGroup(
                dev,
                self.bind_group_layout,
                self.texture_view,
                self.sampler,
            ) orelse return error.WgpuTextBindGroupUnavailable;
            errdefer self.bind_group.release();

            self.pipeline = createTextPipeline(
                dev,
                format,
                self.pipeline_layout,
            ) orelse return error.WgpuTextPipelineUnavailable;
            errdefer self.pipeline.release();

            // The color emoji atlas (CPU buffer + GPU texture/view/bind group) is created lazily
            // by ensureColorAtlas() on the first color-glyph bake — see the field comment.

            self.gpu_enabled = true;
        } else {
            // CPU-only (native backend): no wgpu GPU resources.
            self.gpu_enabled = false;
            self.texture = undefined;
            self.texture_view = undefined;
            self.sampler = undefined;
            self.bind_group_layout = undefined;
            self.pipeline_layout = undefined;
            self.bind_group = undefined;
            self.pipeline = undefined;
        }

        return self;
    }

    // ── CPU atlas accessors for native backends ────────────────────────────────
    /// The R8 grayscale-coverage glyph atlas backing buffer (atlas_width × atlas_height).
    pub fn atlasPixels(self: *const FreeTypeTextRenderer) []const u8 {
        return self.atlas_pixels;
    }

    /// GPU bytes held by the two text atlases: the R8 coverage atlas (1 B/texel, grows 1024²→4096²) and
    /// the optional RGBA8 color-emoji atlas (0 until the first color glyph bakes). Diagnostic only.
    pub const AtlasBytes = struct { coverage: u64, emoji: u64 };
    pub fn atlasBytes(self: *const FreeTypeTextRenderer) AtlasBytes {
        return .{
            .coverage = @as(u64, self.atlas_width) * self.atlas_height,
            .emoji = if (self.color_gpu != null) @as(u64, ColorAtlas.width) * ColorAtlas.height * 4 else 0,
        };
    }
    pub fn atlasIsDirty(self: *const FreeTypeTextRenderer) bool {
        return self.atlas_dirty;
    }
    pub fn atlasDirtyRect(self: *const FreeTypeTextRenderer) ?AtlasDirtyRect {
        if (!self.atlas_dirty or self.dirty_max_x <= self.dirty_min_x or self.dirty_max_y <= self.dirty_min_y) return null;
        return .{
            .x = self.dirty_min_x,
            .y = self.dirty_min_y,
            .width = self.dirty_max_x - self.dirty_min_x,
            .height = self.dirty_max_y - self.dirty_min_y,
        };
    }
    pub fn markAtlasClean(self: *FreeTypeTextRenderer) void {
        self.atlas_dirty = false;
        self.dirty_min_x = self.atlas_width;
        self.dirty_min_y = self.atlas_height;
        self.dirty_max_x = 0;
        self.dirty_max_y = 0;
    }

    pub fn deinit(self: *FreeTypeTextRenderer) void {
        if (self.gpu_enabled) {
            self.pipeline.release();
            self.bind_group.release();
            self.pipeline_layout.release();
            self.bind_group_layout.release();
            self.sampler.release();
            self.texture_view.release();
            self.texture.release();
        }
        if (self.color_gpu) |cg| {
            cg.bind_group.release();
            cg.view.release();
            cg.texture.release();
        }
        self.color_glyphs.deinit();
        if (self.color_atlas_pixels) |p| self.allocator.free(p);
        for (self.emoji_families.items) |name| self.allocator.free(name);
        self.emoji_families.deinit(self.allocator);
        for (self.fallback_families.items) |name| self.allocator.free(name);
        self.fallback_families.deinit(self.allocator);
        self.pending_color_quads.deinit(self.allocator);
        self.emoji_glyph_cache.deinit();
        self.route_cache.deinit();

        self.glyphs.deinit();
        self.allocator.free(self.atlas_pixels);
        self.allocator.free(self.default_font_family);

        var it = self.faces.iterator();
        while (it.next()) |entry| {
            _ = ft.FT_Done_Face(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.faces.deinit();

        if (self.library != null) {
            _ = ft.FT_Done_FreeType(self.library);
        }

        var layout_it = self.layout_cache.iterator();
        while (layout_it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*.glyphs);
            self.allocator.free(entry.value_ptr.*.color_quads);
            self.allocator.free(entry.value_ptr.*.carets);
            self.allocator.free(entry.value_ptr.*.text);
            self.allocator.free(entry.value_ptr.*.font_family);
        }
        self.layout_cache.deinit();

        var run_it = self.shaped_run_cache.valueIterator();
        while (run_it.next()) |run| self.allocator.free(run.glyphs);
        self.shaped_run_cache.deinit();
    }

    pub fn bindGroupLayout(self: *const FreeTypeTextRenderer) *wgpu.BindGroupLayout {
        return self.bind_group_layout;
    }

    pub fn getSampler(self: *const FreeTypeTextRenderer) *wgpu.Sampler {
        return self.sampler;
    }

    /// Register an emoji font by family name (must already be loaded via faces).
    /// The name is duped; caller does not need to keep it alive.
    /// Load a font face from a null-terminated C path and register it under `name`.
    /// Exposed for FFI (ffi/root.zig) since `ft` is private to this file.
    pub fn loadFontFromCPath(self: *FreeTypeTextRenderer, name: []const u8, path_cstr: [*c]const u8) !void {
        var face: ft.FT_Face = null;
        if (ft.FT_New_Face(self.library, path_cstr, 0, &face) != 0)
            return error.FreeTypeFaceFailed;
        errdefer _ = ft.FT_Done_Face(face);
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        try self.faces.put(owned_name, face);

        // A (re)registered face — e.g. a live language switch that swaps the UI face under a fixed
        // family name ("ui": Pixelify -> Inter) — must FULLY reset the caches, not just the shaped runs.
        // The glyph cache is keyed by (family, glyph_index, size), NOT by face identity, so any glyph
        // index the new face reuses would hit a stale bitmap from the old face and render as the wrong
        // font until a restart.
        self.resetAllTextCaches();
    }

    /// Drop EVERY text cache and zero both atlases: shaped runs, rasterized coverage glyphs,
    /// colour/emoji glyphs. Handle-held layout entries re-shape lazily via the atlas generation
    /// bump. Used by face (re)registration, and exported (zigote_text_reset_caches) so hosts can
    /// force a clean re-shape after a wholesale text sizing change (a live UI font-scale switch).
    pub fn resetAllTextCaches(self: *FreeTypeTextRenderer) void {
        self.clearShapedRunCache();
        self.resetAtlas();
        self.resetColorAtlas();
        self.emoji_glyph_cache.clearRetainingCapacity();
        self.route_cache.clearRetainingCapacity();
    }

    /// Lazily create the color emoji atlas (CPU buffer, and — on the wgpu path — the GPU
    /// texture/view/bind group). Called on the first color-glyph bake; on failure everything
    /// already created stays valid and the caller simply skips the glyph (retried next bake).
    fn ensureColorAtlas(self: *FreeTypeTextRenderer) !void {
        if (self.color_atlas_pixels == null) {
            const pixels = try self.allocator.alloc(u8, ColorAtlas.width * ColorAtlas.height * 4);
            @memset(pixels, 0);
            self.color_atlas_pixels = pixels;
        }
        if (!self.gpu_enabled or self.color_gpu != null) return;
        const dev = self.gpu_device orelse return error.WgpuTextTextureUnavailable;
        const tex = dev.createTexture(&.{
            .label = wgpu.StringView.fromSlice("zigote color emoji atlas"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{
                .width = ColorAtlas.width,
                .height = ColorAtlas.height,
                .depth_or_array_layers = 1,
            },
            .format = .rgba8_unorm,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.WgpuTextTextureUnavailable;
        errdefer tex.release();
        const view = tex.createView(null) orelse return error.WgpuTextTextureViewUnavailable;
        errdefer view.release();
        // Same bind group layout as the text atlas so it is compatible with the image pipeline
        // (which uses text.bindGroupLayout()).
        const bg = createTextBindGroup(dev, self.bind_group_layout, view, self.sampler) orelse
            return error.WgpuTextBindGroupUnavailable;
        self.color_gpu = .{ .texture = tex, .view = view, .bind_group = bg };
    }

    /// True if `name` resolves to a face the color path can actually draw — one that produces
    /// BGRA bitmaps under FT_LOAD_COLOR (CBDT/sbix strikes). COLR-outline-only fonts load fine
    /// but render monochrome; registering one as THE emoji font would capture every emoji
    /// codepoint and then draw nothing, which is strictly worse than the monochrome fallback.
    pub fn emojiFamilyRendersColor(self: *FreeTypeTextRenderer, name: []const u8) bool {
        const face = self.faces.get(name) orelse return false;
        // Probe a handful of universal emoji — any single one may be absent from a given font.
        const probes = [_]u21{ 0x1F600, 0x1F389, 0x1F680, 0x2764 };
        for (probes) |cp| {
            const index = ft.FT_Get_Char_Index(face, @as(ft.FT_ULong, cp));
            if (index == 0) continue;
            _ = emojiSelectSize(face, 32) catch return false;
            if (ft.FT_Load_Glyph(face, index, ft.FT_LOAD_COLOR | ft.FT_LOAD_RENDER) != 0) continue;
            return face.*.glyph.*.bitmap.pixel_mode == ft.FT_PIXEL_MODE_BGRA;
        }
        return false;
    }

    pub fn addEmojiFontFamily(self: *FreeTypeTextRenderer, name: []const u8) void {
        const duped = self.allocator.dupe(u8, name) catch return;
        self.emoji_families.append(self.allocator, duped) catch {
            self.allocator.free(duped);
            return;
        };
        self.emoji_glyph_cache.clearRetainingCapacity(); // new family may now cover codepoints cached as false
        self.route_cache.clearRetainingCapacity();
    }

    /// A stretch of text that one face can shape on its own.
    const FaceRun = struct { text: []const u8, family: []const u8, emoji: bool };

    /// Splits text into runs by which face renders them.
    ///
    /// Every shaping entry point — paint, measure and cached layout — goes through this, because a
    /// run boundary the painter sees but the measurer does not means text laid out at one width and
    /// drawn at another. (Emoji segmentation used to live only in the paint path, so emoji were
    /// already measured against the wrong face.)
    ///
    /// Combining marks never start a run: they belong to the base character before them, and
    /// splitting a mark onto a different face than its base would shape both wrongly.
    const RunSplitter = struct {
        renderer: *FreeTypeTextRenderer,
        text: []const u8,
        primary: []const u8,
        pos: usize = 0,

        /// Presentation the marks *after* a base character request: U+FE0F (VS16) and U+20E3
        /// (combining keycap) pull an otherwise-text base like '1' or '#' onto the emoji face —
        /// the keycap/emoji form is a ligature that only exists there. U+FE0E (VS15) is the
        /// opposite: an explicit request for text presentation.
        fn markPresentation(self: *const RunSplitter, from: usize) Presentation {
            var pos = from;
            while (pos < self.text.len) {
                const len = std.unicode.utf8ByteSequenceLength(self.text[pos]) catch 1;
                const end = @min(pos + len, self.text.len);
                const cp: u21 = std.unicode.utf8Decode(self.text[pos..end]) catch 0xFFFD;
                if (!isCombining(cp)) break;
                if (cp == 0xFE0F or cp == 0x20E3) return .emoji;
                if (cp == 0xFE0E) return .text;
                pos = end;
            }
            return .default;
        }

        fn next(self: *RunSplitter) ?FaceRun {
            if (self.pos >= self.text.len) return null;

            const start = self.pos;
            var current: ?Route = null;

            while (self.pos < self.text.len) {
                const len = std.unicode.utf8ByteSequenceLength(self.text[self.pos]) catch 1;
                const end = @min(self.pos + len, self.text.len);
                const cp: u21 = std.unicode.utf8Decode(self.text[self.pos..end]) catch 0xFFFD;

                if (current) |run| {
                    if (!isCombining(cp)) {
                        const route = self.renderer.routeForCluster(cp, self.primary, self.markPresentation(end));
                        if (route.emoji != run.emoji or !std.mem.eql(u8, route.family, run.family))
                            break;
                    }
                } else {
                    current = self.renderer.routeForCluster(cp, self.primary, self.markPresentation(end));
                }

                self.pos = end;
            }

            const run = current orelse return null;
            return .{ .text = self.text[start..self.pos], .family = run.family, .emoji = run.emoji };
        }
    };

    fn splitRuns(self: *FreeTypeTextRenderer, text: []const u8, primary: []const u8) RunSplitter {
        return .{ .renderer = self, .text = text, .primary = primary };
    }

    /// Register a family to fall back to for codepoints the requested face cannot render. Order is
    /// priority order, so the host should register the closest-matching face first. The family must
    /// already be loaded (see loadFontFromCPath).
    pub fn addFallbackFontFamily(self: *FreeTypeTextRenderer, name: []const u8) void {
        // Ignore an unknown or duplicate family rather than storing a name that resolves to nothing
        // on every lookup for the life of the process.
        if (!self.faces.contains(name)) return;
        for (self.fallback_families.items) |existing| {
            if (std.mem.eql(u8, existing, name)) return;
        }

        const duped = self.allocator.dupe(u8, name) catch return;
        self.fallback_families.append(self.allocator, duped) catch {
            self.allocator.free(duped);
            return;
        };
        self.route_cache.clearRetainingCapacity(); // codepoints cached as tofu may now resolve
    }

    /// Where a codepoint should be drawn from. `emoji` takes the colour path; otherwise `family`
    /// names the face to shape with.
    const Route = struct { family: []const u8, emoji: bool };

    /// What a cluster's trailing variation-selector/keycap marks ask for (see markPresentation).
    const Presentation = enum { default, emoji, text };

    /// Cached values of routeFor, packed into the i8 the cache stores.
    const route_primary: i8 = -1;
    const route_emoji: i8 = -2;

    /// Resolve which face renders `cp`: the requested one if it has the glyph, else the first
    /// fallback that does, else the requested one again (so it renders as that face's .notdef —
    /// a visible box is the honest answer when nothing on the system can draw the character).
    fn routeFor(self: *FreeTypeTextRenderer, cp: u21, primary: []const u8) Route {
        // Latin-1 stays on the default UI face without consulting the map — it is the
        // overwhelming majority of what gets measured, and the default face always covers it.
        // The shortcut must NOT extend to other primaries: an icon or symbol face typically has
        // no space/letter glyphs at all, and pinning Latin-1 to it renders every separator in an
        // icon string as .notdef boxes. Those go through the (memoized) resolver like anything else.
        if (cp < 0x0100 and std.mem.eql(u8, primary, self.default_font_family))
            return .{ .family = primary, .emoji = false };

        var hasher = std.hash.Wyhash.init(cp);
        hasher.update(primary);
        const key = hasher.final();

        const slot: i8 = if (self.route_cache.get(key)) |cached| cached else blk: {
            const resolved = self.resolveRoute(cp, primary);
            self.route_cache.put(key, resolved) catch {};
            break :blk resolved;
        };

        if (slot == route_emoji) return .{ .family = self.emoji_families.items[0], .emoji = true };
        if (slot == route_primary or slot >= self.fallback_families.items.len)
            return .{ .family = primary, .emoji = false };
        return .{ .family = self.fallback_families.items[@intCast(slot)], .emoji = false };
    }

    fn resolveRoute(self: *FreeTypeTextRenderer, cp: u21, primary: []const u8) i8 {
        if (self.emoji_families.items.len > 0 and isEmojiCodepoint(cp) and self.emojiHasGlyph(cp))
            return route_emoji;
        return self.resolveTextRoute(cp, primary);
    }

    fn resolveTextRoute(self: *FreeTypeTextRenderer, cp: u21, primary: []const u8) i8 {
        if (self.faces.get(primary)) |face| {
            if (ft.FT_Get_Char_Index(face, @as(ft.FT_ULong, cp)) != 0) return route_primary;
        }

        for (self.fallback_families.items, 0..) |name, index| {
            const face = self.faces.get(name) orelse continue;
            if (ft.FT_Get_Char_Index(face, @as(ft.FT_ULong, cp)) != 0) return @intCast(index);
        }

        return route_primary;
    }

    /// routeFor, refined by the presentation the base's trailing marks asked for: VS16/keycap
    /// promotes a text-face base ('1', '#', '©') onto the emoji face — the emoji form is a
    /// ligature that only exists there — and VS15 demotes an emoji-block codepoint back onto a
    /// text face. Both variants are rarer than the mark-less default, so only the default is
    /// answered from (and stored in) the route cache.
    fn routeForCluster(
        self: *FreeTypeTextRenderer,
        cp: u21,
        primary: []const u8,
        presentation: Presentation,
    ) Route {
        switch (presentation) {
            .emoji => if (self.emoji_families.items.len > 0 and self.emojiHasGlyph(cp))
                return .{ .family = self.emoji_families.items[0], .emoji = true },
            .text => {
                const slot = self.resolveTextRoute(cp, primary);
                if (slot == route_primary or slot >= self.fallback_families.items.len)
                    return .{ .family = primary, .emoji = false };
                return .{ .family = self.fallback_families.items[@intCast(slot)], .emoji = false };
            },
            .default => {},
        }
        return self.routeFor(cp, primary);
    }

    /// True only if an emoji font actually contains a glyph for this codepoint. Many symbols
    /// fall in "emoji-ish" Unicode blocks but aren't in the color-emoji font (e.g. ✕ ⌘);
    /// those must stay on the main text font instead of rendering as tofu.
    fn emojiHasGlyph(self: *FreeTypeTextRenderer, cp: u21) bool {
        if (self.emoji_glyph_cache.get(cp)) |hit| return hit;
        var found = false;
        for (self.emoji_families.items) |name| {
            const face = self.faces.get(name) orelse continue;
            if (ft.FT_Get_Char_Index(face, @as(ft.FT_ULong, cp)) != 0) {
                found = true;
                break;
            }
        }
        self.emoji_glyph_cache.put(cp, found) catch {};
        return found;
    }

    /// Upload the RGBA color emoji atlas to the GPU if it has changed.
    pub fn uploadColorAtlasIfDirty(self: *FreeTypeTextRenderer, queue: *wgpu.Queue) void {
        if (!self.gpu_enabled) return; // CPU-only (native backend) uploads its own atlas
        if (!self.color_atlas_dirty) return;
        const gpu = self.color_gpu orelse return;
        const pixels = self.color_atlas_pixels orelse return;
        queue.writeTexture(
            &.{
                .texture = gpu.texture,
                .mip_level = 0,
                .origin = .{ .x = 0, .y = 0, .z = 0 },
                .aspect = .all,
            },
            pixels.ptr,
            pixels.len,
            &.{
                .offset = 0,
                .bytes_per_row = ColorAtlas.width * 4,
                .rows_per_image = ColorAtlas.height,
            },
            &.{
                .width = ColorAtlas.width,
                .height = ColorAtlas.height,
                .depth_or_array_layers = 1,
            },
        );
        self.color_atlas_dirty = false;
    }

    pub fn appendText(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        vertices: *std.ArrayList(TextVertex),
        text: zg.paint.Text,
        offset_x: f32,
        offset_y: f32,
        frame_width: f32,
        frame_height: f32,
    ) !void {
        if (text.text.len == 0) return;

        const pixel_size = fontPixelSize(text.size);
        const line_height = if (text.line_height > 0) text.line_height else text.size * 1.25;
        const font_family = effectiveFontFamily(self, text);
        const color = colorToU8(text.color);
        const style = styleOf(text);

        var pen_x = text.baseline_x;
        var baseline_y = text.baseline_y;

        _ = std.unicode.Utf8View.init(text.text) catch {
            std.debug.print("Invalid UTF-8 Text: {any}\n", .{text.text});
            return error.InvalidUtf8Text;
        };

        self.pending_color_quads.clearRetainingCapacity();

        // Only control characters are split here — they move the pen rather than drawing. Which
        // face renders what is decided inside flushTextSegment, so paint, measure and cached
        // layout all get the same answer from the same code.
        var segment_start: usize = 0;
        for (text.text, 0..) |byte, i| {
            if (byte != '\n' and byte != '\r' and byte != '\t') continue;

            try self.flushTextSegment(allocator, vertices, font_family, pixel_size, style, color, offset_x, offset_y, frame_width, frame_height, &pen_x, &baseline_y, text.text[segment_start..i], text.letter_spacing, text.word_spacing, text.blur);
            switch (byte) {
                '\n' => {
                    pen_x = text.baseline_x;
                    baseline_y += line_height;
                },
                '\t' => pen_x += text.size * 2.0,
                else => {},
            }

            segment_start = i + 1;
        }

        try self.flushTextSegment(allocator, vertices, font_family, pixel_size, style, color, offset_x, offset_y, frame_width, frame_height, &pen_x, &baseline_y, text.text[segment_start..], text.letter_spacing, text.word_spacing, text.blur);
    }

    fn flushTextSegment(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        vertices: *std.ArrayList(TextVertex),
        font_family: []const u8,
        pixel_size: u16,
        style: FaceStyle,
        color: [4]u8,
        offset_x: f32,
        offset_y: f32,
        frame_width: f32,
        frame_height: f32,
        pen_x: *f32,
        baseline_y: *f32,
        segment: []const u8,
        letter_spacing: f32,
        word_spacing: f32,
        blur: f32,
    ) !void {
        if (segment.len == 0) return;

        var runs = self.splitRuns(segment, font_family);
        while (runs.next()) |run| {
            if (run.emoji) {
                // Color emoji cast no text-shadow (matches browsers); a blurred pass skips them
                // but must still advance the pen so following glyphs keep their positions.
                if (blur > 0) {
                    var dx: f32 = 0;
                    var dy: f32 = 0;
                    try self.appendEmojiShapedSegment(run.text, run.family, pixel_size, offset_x, offset_y, frame_width, frame_height, &dx, &dy);
                    self.pending_color_quads.clearRetainingCapacity();
                    pen_x.* += dx;
                    baseline_y.* += dy;
                } else {
                    try self.appendEmojiShapedSegment(run.text, run.family, pixel_size, offset_x, offset_y, frame_width, frame_height, pen_x, baseline_y);
                }
            } else {
                try self.appendShapedSegment(allocator, vertices, run.text, run.family, pixel_size, style, color, offset_x, offset_y, frame_width, frame_height, pen_x, baseline_y, letter_spacing, word_spacing, blur);
            }
        }
    }

    fn clearShapedRunCache(self: *FreeTypeTextRenderer) void {
        var it = self.shaped_run_cache.valueIterator();
        while (it.next()) |run| self.allocator.free(run.glyphs);
        self.shaped_run_cache.clearRetainingCapacity();
    }

    /// Shape `text` once and cache glyph IDs and positions. Atlas UVs deliberately do not live here:
    /// shaping remains valid across atlas resets and device-scale changes.
    ///
    /// `style` participates through what it synthesizes: emboldened glyphs are wider, so the
    /// strength joins the advances (and the cache key) — measure, carets and paint all read the
    /// same widened run. `spaced` (letter-spacing active) shapes without optional ligatures:
    /// tracking pulls letters apart, and an "ffi" fused into one glyph cannot be tracked.
    fn getShapedRun(
        self: *FreeTypeTextRenderer,
        face: ft.FT_Face,
        text: []const u8,
        font_family: []const u8,
        pixel_size: u16,
        style: FaceStyle,
        spaced: bool,
    ) !*const ShapedRun {
        const synth = synthFor(face, style, pixel_size);

        var hasher = std.hash.Wyhash.init(@as(u64, pixel_size));
        hasher.update(font_family);
        hasher.update(&[_]u8{0}); // separator so (family, text) can't alias across the boundary
        const strength: i32 = synth.strength266(); // oblique changes no advance — not keyed
        hasher.update(std.mem.asBytes(&strength));
        hasher.update(&[_]u8{@intFromBool(spaced)});
        hasher.update(text);
        const key = hasher.final();

        if (self.shaped_run_cache.getPtr(key)) |run| return run;

        // Bound the cache; a wholesale clear is fine — entries rebuild lazily when next drawn.
        if (self.shaped_run_cache.count() >= 8192) self.clearShapedRunCache();

        if (ft.FT_Set_Pixel_Sizes(face, 0, @intCast(pixel_size)) != 0)
            return error.FreeTypeSetPixelSizeFailed;

        const hb_font = hb_ft_font_create_referenced(face) orelse return error.HarfBuzzFontCreateFailed;
        defer hb.hb_font_destroy(hb_font);
        // NOTE: no hb_ft_font_set_funcs here — create_referenced already installed the FT funcs
        // on THIS face; the extra call cloned a second face from the font blob per shaped run
        // (an FT_New_Memory_Face each time) for identical unhinted metrics.

        var shaped_list: std.ArrayList(ShapedGlyph) = .empty;
        defer shaped_list.deinit(self.allocator);

        // UAX #9: text with anything to reorder is split into embedding-level runs, each shaped with
        // an explicit direction. `analyze` hands them back already in VISUAL order (rule L2), so
        // appending them in order is what the sequential left-to-right emitters want. Text with
        // nothing to reorder takes the single-buffer path exactly as before.
        var dir_runs: std.ArrayList(bidi.Run) = .empty;
        defer dir_runs.deinit(self.allocator);
        // null base: let P2/P3 derive the paragraph direction from the text itself. Nothing at this
        // level knows the widget's locale, which is the only thing that would justify forcing it.
        if (try bidi.analyze(self.allocator, text, null, &dir_runs)) |_| {
            for (dir_runs.items) |r|
                try self.shapeRunInto(hb_font, text[r.start..r.end], r.start, r.dir(), spaced, &shaped_list);
        } else {
            try self.shapeRunInto(hb_font, text, 0, null, spaced, &shaped_list);
        }

        const shaped = try self.allocator.dupe(ShapedGlyph, shaped_list.items);
        errdefer self.allocator.free(shaped);
        var advance_x: f32 = 0;
        var advance_y: f32 = 0;
        for (shaped) |*g| {
            // A synthetically emboldened glyph is wider by the stroke growth; folding it into the
            // cached advance keeps letterfit correct and every consumer consistent. Zero-advance
            // glyphs (combining marks) ride their base and must not push the pen.
            if (synth.embolden > 0 and g.x_advance != 0)
                g.x_advance += synth.embolden;
            advance_x += g.x_advance;
            advance_y -= g.y_advance;
        }

        const run = ShapedRun{
            .glyphs = shaped,
            .advance_x = advance_x,
            .advance_y = advance_y,
        };
        try self.shaped_run_cache.put(key, run);
        return self.shaped_run_cache.getPtr(key).?;
    }

    /// Shape one direction run and append its glyphs. `cluster_base` re-anchors HarfBuzz's
    /// slice-relative clusters to the full segment text (clusterIsSpace / caret stops rely on it).
    /// `direction == null` lets HarfBuzz guess from content (the pre-bidi single-run behavior).
    fn shapeRunInto(
        self: *FreeTypeTextRenderer,
        hb_font: *hb.hb_font_t,
        run_text: []const u8,
        cluster_base: u32,
        direction: ?bidi.Dir,
        spaced: bool,
        out: *std.ArrayList(ShapedGlyph),
    ) !void {
        if (run_text.len == 0) return;

        const buffer = hb.hb_buffer_create() orelse return error.HarfBuzzBufferCreateFailed;
        defer hb.hb_buffer_destroy(buffer);
        hb.hb_buffer_add_utf8(buffer, run_text.ptr, @intCast(run_text.len), 0, @intCast(run_text.len));
        // Guess script/language from content, then pin the direction when the bidi pass decided it
        // (guessing direction from the first strong character is exactly what mis-orders mixed runs).
        hb.hb_buffer_guess_segment_properties(buffer);
        if (direction) |d|
            hb.hb_buffer_set_direction(buffer, if (d == .rtl) hb.HB_DIRECTION_RTL else hb.HB_DIRECTION_LTR);
        // Code faces rely on both standard and contextual ligatures (for example !=, =>, ffi).
        // Specify them explicitly instead of depending on backend/font defaults so cached editor
        // layouts and immediate text runs shape identically on every platform. Under active
        // letter-spacing the optional ligatures (liga/clig) are turned OFF instead — tracking pulls
        // letters apart, and a fused "ffi" cannot be tracked — while calt stays on (it substitutes
        // per-glyph forms without fusing clusters).
        const lig: [*c]const u8 = if (spaced) "liga=0" else "liga=1";
        const clig: [*c]const u8 = if (spaced) "clig=0" else "clig=1";
        var features: [3]hb.hb_feature_t = undefined;
        _ = hb.hb_feature_from_string(lig, -1, &features[0]);
        _ = hb.hb_feature_from_string(clig, -1, &features[1]);
        _ = hb.hb_feature_from_string("calt=1", -1, &features[2]);
        hb.hb_shape(hb_font, buffer, &features, features.len);

        var info_count: c_uint = 0;
        const infos_ptr = hb.hb_buffer_get_glyph_infos(buffer, &info_count) orelse return error.HarfBuzzGlyphInfoFailed;
        var position_count: c_uint = 0;
        const positions_ptr = hb.hb_buffer_get_glyph_positions(buffer, &position_count) orelse return error.HarfBuzzGlyphPositionFailed;
        const glyph_count: usize = @intCast(@min(info_count, position_count));

        try out.ensureUnusedCapacity(self.allocator, glyph_count);
        for (infos_ptr[0..glyph_count], positions_ptr[0..glyph_count]) |info, pos| {
            out.appendAssumeCapacity(.{
                .glyph_index = info.codepoint,
                .cluster = info.cluster + cluster_base,
                .x_offset = hbPositionToPixels(pos.x_offset),
                .y_offset = hbPositionToPixels(pos.y_offset),
                .x_advance = hbPositionToPixels(pos.x_advance),
                .y_advance = hbPositionToPixels(pos.y_advance),
            });
        }
    }

    fn appendShapedSegment(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        vertices: *std.ArrayList(TextVertex),
        text: []const u8,
        font_family: []const u8,
        pixel_size: u16,
        style: FaceStyle,
        color: [4]u8,
        offset_x: f32,
        offset_y: f32,
        frame_width: f32,
        frame_height: f32,
        pen_x: *f32,
        baseline_y: *f32,
        letter_spacing: f32,
        word_spacing: f32,
        blur: f32,
    ) !void {
        if (text.len == 0) return;

        const face = self.resolveFace(font_family) orelse return error.FreeTypeFontNotFound;
        const spaced = letter_spacing != 0;
        const run = try self.getShapedRun(face, text, font_family, pixel_size, style, spaced);
        for (run.glyphs, 0..) |shaped, i| {
            const glyph = try self.getGlyph(face, font_family, shaped.glyph_index, pixel_size, style, blur);
            if (glyph.width > 0 and glyph.height > 0) {
                // Snap the quad origin to the physical pixel grid: each glyph exists as ONE
                // integer-origin raster in the atlas, so a fractional origin makes the linear
                // sampler resample every texel — uniform blur, worst at 1x DPI. The pen itself
                // stays fractional (shaping accuracy); only the emitted quad is snapped, giving
                // a 1:1 texel↔pixel mapping.
                const x0 = @round(pen_x.* + shaped.x_offset + glyph.bearing_x);
                const y0 = @round(baseline_y.* - shaped.y_offset - glyph.bearing_y);
                const x1 = x0 + @as(f32, @floatFromInt(glyph.width));
                const y1 = y0 + @as(f32, @floatFromInt(glyph.height));
                const atlas_w: f32 = @floatFromInt(self.atlas_width);
                const atlas_h: f32 = @floatFromInt(self.atlas_height);
                try appendGlyphQuad(
                    allocator,
                    vertices,
                    offset_x,
                    offset_y,
                    frame_width,
                    frame_height,
                    x0,
                    y0,
                    x1,
                    y1,
                    @as(f32, @floatFromInt(glyph.atlas_x)) / atlas_w,
                    @as(f32, @floatFromInt(glyph.atlas_y)) / atlas_h,
                    @as(f32, @floatFromInt(glyph.atlas_x + glyph.width)) / atlas_w,
                    @as(f32, @floatFromInt(glyph.atlas_y + glyph.height)) / atlas_h,
                    color,
                );
            }

            pen_x.* += shaped.x_advance;
            baseline_y.* -= shaped.y_advance;
            // Tracking applies per CLUSTER, after its final glyph — never between a base and the
            // combining marks that ride it (a mark pushed sideways by letter-spacing detaches
            // from its base) and never inside a ligature.
            if (clusterEndsAt(run.glyphs, i)) {
                var spacing = letter_spacing;
                if (clusterIsSpace(text, shaped.cluster)) spacing += word_spacing;
                pen_x.* += spacing;
            }
        }
    }

    /// Build and cache a text layout. Returns an opaque handle (> 0).
    /// Caller must call releaseTextLayout(handle) when done.
    pub fn appendTextLayout(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        text: zg.paint.Text,
    ) !u64 {
        const pixel_size = fontPixelSize(text.size);
        const line_height = if (text.line_height > 0) text.line_height else text.size * 1.25;
        const font_family = effectiveFontFamily(self, text);
        const style = styleOf(text);

        var glyph_list: std.ArrayList(TextLayoutGlyph) = .empty;
        defer glyph_list.deinit(allocator);
        var color_list: std.ArrayList(PendingColorQuad) = .empty;
        defer color_list.deinit(allocator);

        var width: f32 = 0.0;
        var height: f32 = 0.0;
        try self.buildLayoutGlyphs(
            allocator,
            &glyph_list,
            &color_list,
            text.text,
            font_family,
            style,
            pixel_size,
            pixel_size,
            1.0,
            text.size,
            line_height,
            text.letter_spacing,
            text.word_spacing,
            &width,
            &height,
        );

        // Own the source + resolved style so the entry can re-shape itself after an atlas reset
        // invalidates its baked UVs (the originating zg.paint.Text does not outlive this call).
        const owned_text = try self.allocator.dupe(u8, text.text);
        errdefer self.allocator.free(owned_text);
        const owned_family = try self.allocator.dupe(u8, font_family);
        errdefer self.allocator.free(owned_family);
        const owned_glyphs = try self.allocator.dupe(TextLayoutGlyph, glyph_list.items);
        errdefer self.allocator.free(owned_glyphs);
        const owned_colors = try self.allocator.dupe(PendingColorQuad, color_list.items);
        errdefer self.allocator.free(owned_colors);
        const owned_carets = try self.buildCaretStops(text.text, font_family, pixel_size, style, line_height, text.letter_spacing, text.word_spacing);
        errdefer self.allocator.free(owned_carets);

        const entry = TextLayoutEntry{
            .glyphs = owned_glyphs,
            .color_quads = owned_colors,
            .carets = owned_carets,
            .width = width,
            .height = height,
            .text = owned_text,
            .font_family = owned_family,
            .size = text.size,
            .line_height = line_height,
            .letter_spacing = text.letter_spacing,
            .word_spacing = text.word_spacing,
            .style = style,
            .raster_scale = 1.0,
            // Read after building: a large layout can itself overflow and reset the atlas mid-build.
            .generation = self.atlas_generation,
        };

        const handle = self.next_layout_handle;
        self.next_layout_handle += 1;
        try self.layout_cache.put(handle, entry);
        return handle;
    }

    /// Shape `text` into baseline-relative layout glyphs (handling \n, \r, \t) at origin (0,0),
    /// appending to `glyph_list` and reporting the laid-out bounding box. Shared by appendTextLayout
    /// (initial bake) and reshapeLayoutEntry (re-bake after an atlas reset) so geometry is identical.
    fn buildLayoutGlyphs(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        glyph_list: *std.ArrayList(TextLayoutGlyph),
        color_list: *std.ArrayList(PendingColorQuad),
        text: []const u8,
        font_family: []const u8,
        style: FaceStyle,
        shape_pixel_size: u16,
        raster_pixel_size: u16,
        raster_scale: f32,
        size: f32,
        line_height: f32,
        letter_spacing: f32,
        word_spacing: f32,
        out_w: *f32,
        out_h: *f32,
    ) !void {
        var pen_x: f32 = 0.0;
        var baseline_y: f32 = 0.0;
        var max_x: f32 = 0.0;

        var segment_start: usize = 0;
        var i: usize = 0;
        while (i < text.len) : (i += 1) {
            switch (text[i]) {
                '\n' => {
                    try self.appendShapedSegmentLayout(allocator, glyph_list, color_list, text[segment_start..i], font_family, style, shape_pixel_size, raster_pixel_size, raster_scale, letter_spacing, word_spacing, &pen_x, &baseline_y);
                    max_x = @max(max_x, pen_x);
                    pen_x = 0.0;
                    baseline_y += line_height;
                    segment_start = i + 1;
                },
                '\r' => {
                    try self.appendShapedSegmentLayout(allocator, glyph_list, color_list, text[segment_start..i], font_family, style, shape_pixel_size, raster_pixel_size, raster_scale, letter_spacing, word_spacing, &pen_x, &baseline_y);
                    max_x = @max(max_x, pen_x);
                    segment_start = i + 1;
                },
                '\t' => {
                    try self.appendShapedSegmentLayout(allocator, glyph_list, color_list, text[segment_start..i], font_family, style, shape_pixel_size, raster_pixel_size, raster_scale, letter_spacing, word_spacing, &pen_x, &baseline_y);
                    max_x = @max(max_x, pen_x);
                    pen_x += size * 2.0;
                    segment_start = i + 1;
                },
                else => {},
            }
        }
        try self.appendShapedSegmentLayout(allocator, glyph_list, color_list, text[segment_start..], font_family, style, shape_pixel_size, raster_pixel_size, raster_scale, letter_spacing, word_spacing, &pen_x, &baseline_y);
        max_x = @max(max_x, pen_x);

        out_w.* = max_x;
        out_h.* = baseline_y + line_height;
    }

    pub fn releaseTextLayout(self: *FreeTypeTextRenderer, handle: u64) void {
        if (self.layout_cache.fetchRemove(handle)) |kv| {
            self.allocator.free(kv.value.glyphs);
            self.allocator.free(kv.value.color_quads);
            self.allocator.free(kv.value.carets);
            self.allocator.free(kv.value.text);
            self.allocator.free(kv.value.font_family);
        }
    }

    pub fn measureTextLayout(self: *const FreeTypeTextRenderer, handle: u64, out_w: *f32, out_h: *f32) void {
        if (self.layout_cache.get(handle)) |entry| {
            out_w.* = entry.width;
            out_h.* = entry.height;
        } else {
            out_w.* = 0;
            out_h.* = 0;
        }
    }

    /// Build visual caret stops from HarfBuzz cluster boundaries. Multiple Unicode scalars that
    /// shape as one cluster deliberately expose only the cluster edges, preventing the caret from
    /// entering a ligature, combining sequence, surrogate pair, or emoji ZWJ sequence.
    fn buildCaretStops(
        self: *FreeTypeTextRenderer,
        text: []const u8,
        font_family: []const u8,
        pixel_size: u16,
        style: FaceStyle,
        line_height: f32,
        letter_spacing: f32,
        word_spacing: f32,
    ) ![]TextCaret {
        var result: std.ArrayList(TextCaret) = .empty;
        defer result.deinit(self.allocator);

        var cluster_scratch: std.ArrayList(u32) = .empty;
        defer cluster_scratch.deinit(self.allocator);

        const spaced = letter_spacing != 0;

        var line_start: usize = 0;
        var line_index: usize = 0;
        while (line_start <= text.len) {
            const relative_end = std.mem.indexOfScalar(u8, text[line_start..], '\n');
            const line_end = if (relative_end) |n| line_start + n else text.len;
            const line = text[line_start..line_end];
            const y = @as(f32, @floatFromInt(line_index)) * line_height;

            if (line.len == 0) {
                try result.append(self.allocator, .{
                    .text_offset = @intCast(line_start),
                    .x = 0,
                    .y = y,
                });
            } else {
                // Walk the line the way the painter does — split into face runs, shape each with
                // its own face — so caret geometry lands exactly where the glyphs were drawn.
                // A whole-line shape against the primary face put every caret after a
                // fallback-rendered (CJK, emoji) stretch at the wrong x.
                var pen_x: f32 = 0;
                var runs = self.splitRuns(line, font_family);
                while (runs.next()) |face_run| {
                    const run_offset = @intFromPtr(face_run.text.ptr) - @intFromPtr(line.ptr);
                    const run_opt: ?*const ShapedRun = if (face_run.emoji) blk: {
                        const face = self.faces.get(face_run.family) orelse break :blk null;
                        break :blk self.getEmojiShapedRun(face, face_run.text, face_run.family, pixel_size) catch null;
                    } else blk: {
                        const face = self.resolveFace(face_run.family) orelse break :blk null;
                        break :blk self.getShapedRun(face, face_run.text, face_run.family, pixel_size, style, spaced) catch null;
                    };
                    const run = run_opt orelse continue;

                    const glyphs = run.glyphs;
                    const rtl = glyphs.len > 1 and glyphs[0].cluster > glyphs[glyphs.len - 1].cluster;

                    // One sorted sweep of the run's cluster values replaces a per-group rescan of
                    // every glyph: a group's logical end is the smallest cluster value above its own.
                    cluster_scratch.clearRetainingCapacity();
                    for (glyphs) |g| {
                        if (cluster_scratch.items.len == 0 or cluster_scratch.items[cluster_scratch.items.len - 1] != g.cluster)
                            try cluster_scratch.append(self.allocator, g.cluster);
                    }
                    std.mem.sort(u32, cluster_scratch.items, {}, std.sort.asc(u32));

                    var glyph_index: usize = 0;
                    while (glyph_index < glyphs.len) {
                        const cluster = glyphs[glyph_index].cluster;
                        var group_end = glyph_index;
                        var advance: f32 = 0;
                        while (group_end < glyphs.len and glyphs[group_end].cluster == cluster) : (group_end += 1) {
                            advance += glyphs[group_end].x_advance;
                        }
                        // Per-cluster tracking, matching the emitters (never applied to emoji runs).
                        if (!face_run.emoji) {
                            advance += letter_spacing;
                            if (clusterIsSpace(face_run.text, cluster)) advance += word_spacing;
                        }

                        const logical_end: usize = if (nextClusterAbove(cluster_scratch.items, cluster)) |next|
                            @min(face_run.text.len, @as(usize, next))
                        else
                            face_run.text.len;

                        const base = line_start + run_offset;
                        try result.append(self.allocator, .{
                            .text_offset = @intCast(base + if (rtl) logical_end else cluster),
                            .x = pen_x,
                            .y = y,
                        });
                        pen_x += advance;
                        try result.append(self.allocator, .{
                            .text_offset = @intCast(base + if (rtl) cluster else logical_end),
                            .x = pen_x,
                            .y = y,
                        });
                        glyph_index = group_end;
                    }
                }
            }

            if (line_end == text.len) break;
            line_start = line_end + 1;
            line_index += 1;
        }

        return self.allocator.dupe(TextCaret, result.items);
    }

    pub fn hitTestTextLayout(self: *const FreeTypeTextRenderer, handle: u64, x: f32, y: f32) u32 {
        const entry = self.layout_cache.get(handle) orelse return 0;
        if (entry.carets.len == 0) return 0;
        var best = entry.carets[0];
        var best_distance = @abs(best.x - x) + @abs(best.y - y) * 10000;
        for (entry.carets[1..]) |caret| {
            const distance = @abs(caret.x - x) + @abs(caret.y - y) * 10000;
            if (distance < best_distance) {
                best = caret;
                best_distance = distance;
            }
        }
        return best.text_offset;
    }

    pub fn caretPositionTextLayout(
        self: *const FreeTypeTextRenderer,
        handle: u64,
        text_offset: u32,
        out_x: *f32,
        out_y: *f32,
        out_h: *f32,
    ) bool {
        const entry = self.layout_cache.get(handle) orelse return false;
        for (entry.carets) |caret| {
            if (caret.text_offset == text_offset) {
                out_x.* = caret.x;
                out_y.* = caret.y;
                out_h.* = entry.line_height;
                return true;
            }
        }
        return false;
    }

    pub fn moveCaretTextLayout(self: *const FreeTypeTextRenderer, handle: u64, text_offset: u32, direction: i32) u32 {
        const entry = self.layout_cache.get(handle) orelse return text_offset;
        if (entry.carets.len == 0) return text_offset;
        var found: usize = 0;
        for (entry.carets, 0..) |caret, i| {
            if (caret.text_offset == text_offset) {
                found = i;
                break;
            }
        }
        if (direction < 0) {
            var target = found;
            while (target > 0) {
                target -= 1;
                if (entry.carets[target].text_offset != text_offset)
                    return entry.carets[target].text_offset;
            }
            return text_offset;
        }
        var target = found;
        while (target + 1 < entry.carets.len) {
            target += 1;
            if (entry.carets[target].text_offset != text_offset)
                return entry.carets[target].text_offset;
        }
        return text_offset;
    }

    /// One-shot accurate measurement of a string's bounding box, honouring `text.font_family`
    /// (resolved to a registered face, falling back to the default). Mirrors appendTextLayout's
    /// HarfBuzz advance math. Newlines/tabs are handled; max-width wrapping is not (callers needing
    /// wrapping use the coarse estimate). Measurement warms only the shaped-run cache and does not
    /// rasterize glyphs or mutate the atlas.
    pub fn measureText(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        text: zg.paint.Text,
        out_w: *f32,
        out_h: *f32,
    ) !void {
        _ = allocator;
        const line_height = if (text.line_height > 0) text.line_height else text.size * 1.25;
        out_w.* = 0;
        out_h.* = line_height;
        if (text.text.len == 0) return;

        const pixel_size = fontPixelSize(text.size);
        const font_family = effectiveFontFamily(self, text);
        const style = styleOf(text);

        var pen_x: f32 = 0.0;
        var baseline_y: f32 = 0.0;
        var max_x: f32 = 0.0;

        var segment_start: usize = 0;
        var i: usize = 0;
        while (i < text.text.len) : (i += 1) {
            switch (text.text[i]) {
                '\n' => {
                    try self.advanceShapedSegment(text.text[segment_start..i], font_family, pixel_size, style, text.letter_spacing, text.word_spacing, &pen_x, &baseline_y);
                    max_x = @max(max_x, pen_x);
                    pen_x = 0.0;
                    baseline_y += line_height;
                    segment_start = i + 1;
                },
                '\r' => {
                    try self.advanceShapedSegment(text.text[segment_start..i], font_family, pixel_size, style, text.letter_spacing, text.word_spacing, &pen_x, &baseline_y);
                    max_x = @max(max_x, pen_x);
                    segment_start = i + 1;
                },
                '\t' => {
                    try self.advanceShapedSegment(text.text[segment_start..i], font_family, pixel_size, style, text.letter_spacing, text.word_spacing, &pen_x, &baseline_y);
                    max_x = @max(max_x, pen_x);
                    pen_x += text.size * 2.0;
                    segment_start = i + 1;
                },
                else => {},
            }
        }
        try self.advanceShapedSegment(text.text[segment_start..], font_family, pixel_size, style, text.letter_spacing, text.word_spacing, &pen_x, &baseline_y);
        max_x = @max(max_x, pen_x);

        out_w.* = max_x;
        out_h.* = baseline_y + line_height;
    }

    /// Advance the pen across a segment without drawing, splitting it by face exactly as the
    /// painter does — otherwise a string containing anything outside the primary face measures
    /// against a face that will not draw it, and the layout is wrong before anything is painted.
    fn advanceShapedSegment(
        self: *FreeTypeTextRenderer,
        text: []const u8,
        font_family: []const u8,
        pixel_size: u16,
        style: FaceStyle,
        letter_spacing: f32,
        word_spacing: f32,
        pen_x: *f32,
        baseline_y: *f32,
    ) !void {
        if (text.len == 0) return;

        const spaced = letter_spacing != 0;
        var runs = self.splitRuns(text, font_family);
        while (runs.next()) |face_run| {
            // Emoji runs advance by the emoji face's own (strike-scaled) metrics, exactly like
            // the painter — measuring them against a text face put every mixed label off by the
            // difference before a single glyph was drawn.
            if (face_run.emoji) {
                if (self.faces.get(face_run.family)) |face| {
                    if (self.getEmojiShapedRun(face, face_run.text, face_run.family, pixel_size)) |run| {
                        pen_x.* += run.advance_x;
                        baseline_y.* += run.advance_y;
                    } else |_| {}
                }
                continue;
            }

            const face = self.resolveFace(face_run.family) orelse return error.FreeTypeFontNotFound;
            const run = try self.getShapedRun(face, face_run.text, face_run.family, pixel_size, style, spaced);
            if (letter_spacing == 0 and word_spacing == 0) {
                pen_x.* += run.advance_x;
                baseline_y.* += run.advance_y;
                continue;
            }

            for (run.glyphs, 0..) |shaped, i| {
                pen_x.* += shaped.x_advance;
                baseline_y.* -= shaped.y_advance;
                // Per-cluster tracking — see appendShapedSegment.
                if (clusterEndsAt(run.glyphs, i)) {
                    var spacing = letter_spacing;
                    if (clusterIsSpace(face_run.text, shaped.cluster)) spacing += word_spacing;
                    pen_x.* += spacing;
                }
            }
        }
    }

    /// Emit TextVertex quads for a cached layout at the given draw position. If the atlas has been
    /// reset since the entry was baked, its UVs are stale — re-shape it first (lazily, so only
    /// layouts actually drawn after a reset pay the cost). The u64 handle the caller holds stays
    /// valid throughout, so widgets never have to rebuild on reset.
    pub fn appendLayoutGlyphs(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        entry: *TextLayoutEntry,
        vertices: *std.ArrayList(TextVertex),
        draw_x: f32,
        draw_y: f32,
        scale_factor: f32,
        frame_width: f32,
        frame_height: f32,
        color: [4]u8,
    ) !void {
        const raster_scale = @max(scale_factor, 0.01);
        if (entry.generation != self.atlas_generation or @abs(entry.raster_scale - raster_scale) > 0.001)
            try self.reshapeLayoutEntry(allocator, entry, raster_scale);
        // Color emoji baked into the layout ride the pending queue exactly like the immediate
        // path; the caller flushes them into the image pipeline after this command.
        for (entry.color_quads) |q| {
            try self.pending_color_quads.append(self.allocator, .{
                .x = (q.x + draw_x) * scale_factor,
                .y = (q.y + draw_y) * scale_factor,
                .w = q.w * scale_factor,
                .h = q.h * scale_factor,
                .u0 = q.u0,
                .v0 = q.v0,
                .u1 = q.u1,
                .v1 = q.v1,
            });
        }
        for (entry.glyphs) |g| {
            // Same pixel-grid snap as the immediate path: the origin lands on a physical pixel
            // and the quad keeps the raster's exact integer size ((x1-x0)·scale is the glyph
            // bitmap width up to float fuzz), restoring the 1:1 texel↔pixel mapping that the
            // linear atlas sampler needs to stay crisp — the cached-layout path is what the
            // code editor and every CachedText widget render through.
            const x0 = @round((g.x0 + draw_x) * scale_factor);
            const y0 = @round((g.y0 + draw_y) * scale_factor);
            const x1 = x0 + @round((g.x1 - g.x0) * scale_factor);
            const y1 = y0 + @round((g.y1 - g.y0) * scale_factor);
            try appendGlyphQuad(
                allocator,
                vertices,
                0.0,
                0.0,
                frame_width,
                frame_height,
                x0,
                y0,
                x1,
                y1,
                g.u0,
                g.v0,
                g.u1,
                g.v1,
                color,
            );
        }
    }

    /// Re-shape a cached layout whose baked UVs were invalidated by an atlas reset, replacing its
    /// glyphs in place from the stored source and re-warming the atlas. The u64 handle is unaffected.
    /// On error the entry is left untouched (old glyphs stay valid and freeable, generation stays
    /// stale) so the next draw retries rather than the renderer leaking or double-freeing.
    fn reshapeLayoutEntry(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        entry: *TextLayoutEntry,
        raster_scale: f32,
    ) !void {
        const shape_pixel_size = fontPixelSize(entry.size);
        const raster_pixel_size = fontPixelSize(entry.size * raster_scale);

        var glyph_list: std.ArrayList(TextLayoutGlyph) = .empty;
        defer glyph_list.deinit(allocator);
        var color_list: std.ArrayList(PendingColorQuad) = .empty;
        defer color_list.deinit(allocator);

        var width: f32 = 0.0;
        var height: f32 = 0.0;
        try self.buildLayoutGlyphs(
            allocator,
            &glyph_list,
            &color_list,
            entry.text,
            entry.font_family,
            entry.style,
            shape_pixel_size,
            raster_pixel_size,
            raster_scale,
            entry.size,
            entry.line_height,
            entry.letter_spacing,
            entry.word_spacing,
            &width,
            &height,
        );

        const new_glyphs = try self.allocator.dupe(TextLayoutGlyph, glyph_list.items);
        errdefer self.allocator.free(new_glyphs);
        const new_colors = try self.allocator.dupe(PendingColorQuad, color_list.items);
        self.allocator.free(entry.glyphs);
        self.allocator.free(entry.color_quads);
        entry.glyphs = new_glyphs;
        entry.color_quads = new_colors;
        entry.width = width;
        entry.height = height;
        entry.raster_scale = raster_scale;
        // Read after rebuilding: re-shaping can itself overflow and bump the generation again.
        entry.generation = self.atlas_generation;
    }

    fn appendShapedSegmentLayout(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        glyphs_out: *std.ArrayList(TextLayoutGlyph),
        colors_out: *std.ArrayList(PendingColorQuad),
        text: []const u8,
        font_family: []const u8,
        style: FaceStyle,
        shape_pixel_size: u16,
        raster_pixel_size: u16,
        raster_scale: f32,
        letter_spacing: f32,
        word_spacing: f32,
        pen_x: *f32,
        baseline_y: *f32,
    ) !void {
        if (text.len == 0) return;
        const inv_raster_scale = 1.0 / @max(raster_scale, 0.01);

        // Split by face like the painter and the measurer. A cached layout that shaped everything
        // against the primary face would bake the wrong glyphs and the wrong advances into the
        // handle, and every later draw would replay them.
        var runs = self.splitRuns(text, font_family);
        while (runs.next()) |face_run| {
            if (face_run.emoji) {
                try self.appendEmojiLayoutRun(
                    allocator,
                    colors_out,
                    face_run,
                    shape_pixel_size,
                    raster_pixel_size,
                    inv_raster_scale,
                    pen_x,
                    baseline_y,
                );
            } else {
                try self.appendLayoutRun(
                    allocator,
                    glyphs_out,
                    face_run,
                    style,
                    shape_pixel_size,
                    raster_pixel_size,
                    inv_raster_scale,
                    letter_spacing,
                    word_spacing,
                    pen_x,
                    baseline_y,
                );
            }
        }
    }

    fn appendLayoutRun(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        glyphs_out: *std.ArrayList(TextLayoutGlyph),
        face_run: FaceRun,
        style: FaceStyle,
        shape_pixel_size: u16,
        raster_pixel_size: u16,
        inv_raster_scale: f32,
        letter_spacing: f32,
        word_spacing: f32,
        pen_x: *f32,
        baseline_y: *f32,
    ) !void {
        const text = face_run.text;
        const font_family = face_run.family;

        const face = self.resolveFace(font_family) orelse return error.FreeTypeFontNotFound;
        const spaced = letter_spacing != 0;
        const run = try self.getShapedRun(face, text, font_family, shape_pixel_size, style, spaced);

        for (run.glyphs, 0..) |shaped, i| {
            const glyph = try self.getGlyph(face, font_family, shaped.glyph_index, raster_pixel_size, style, 0);

            if (glyph.width > 0 and glyph.height > 0) {
                const x0 = pen_x.* + shaped.x_offset + glyph.bearing_x * inv_raster_scale;
                const y0 = baseline_y.* - shaped.y_offset - glyph.bearing_y * inv_raster_scale;
                const x1 = x0 + @as(f32, @floatFromInt(glyph.width)) * inv_raster_scale;
                const y1 = y0 + @as(f32, @floatFromInt(glyph.height)) * inv_raster_scale;

                const uv0_x = @as(f32, @floatFromInt(glyph.atlas_x)) / @as(f32, @floatFromInt(self.atlas_width));
                const uv0_y = @as(f32, @floatFromInt(glyph.atlas_y)) / @as(f32, @floatFromInt(self.atlas_height));
                const uv1_x = @as(f32, @floatFromInt(glyph.atlas_x + glyph.width)) / @as(f32, @floatFromInt(self.atlas_width));
                const uv1_y = @as(f32, @floatFromInt(glyph.atlas_y + glyph.height)) / @as(f32, @floatFromInt(self.atlas_height));

                try glyphs_out.append(allocator, .{
                    .x0 = x0,
                    .y0 = y0,
                    .x1 = x1,
                    .y1 = y1,
                    .u0 = uv0_x,
                    .v0 = uv0_y,
                    .u1 = uv1_x,
                    .v1 = uv1_y,
                });
            }

            pen_x.* += shaped.x_advance;
            baseline_y.* -= shaped.y_advance;
            // Per-cluster tracking — see appendShapedSegment.
            if (clusterEndsAt(run.glyphs, i)) {
                var spacing = letter_spacing;
                if (clusterIsSpace(text, shaped.cluster)) spacing += word_spacing;
                pen_x.* += spacing;
            }
        }
    }

    /// The cached-layout counterpart of appendEmojiShapedSegment: bakes the run's color glyphs and
    /// appends origin-relative logical-space quads to `colors_out`. Emoji that cannot bake (no
    /// BGRA bitmap) still advance the pen, so surrounding text never shifts.
    fn appendEmojiLayoutRun(
        self: *FreeTypeTextRenderer,
        allocator: std.mem.Allocator,
        colors_out: *std.ArrayList(PendingColorQuad),
        face_run: FaceRun,
        shape_pixel_size: u16,
        raster_pixel_size: u16,
        inv_raster_scale: f32,
        pen_x: *f32,
        baseline_y: *f32,
    ) !void {
        if (face_run.text.len == 0) return;
        // Direct lookup only — no fallback to a text face (it has no color glyphs).
        const face = self.faces.get(face_run.family) orelse return;

        const run = self.getEmojiShapedRun(face, face_run.text, face_run.family, shape_pixel_size) catch return;

        for (run.glyphs) |shaped| {
            if (self.rasterEmojiGlyph(face, face_run.family, shaped.glyph_index, raster_pixel_size)) |raster| {
                const glyph = raster.glyph;
                if (glyph.width > 0 and glyph.height > 0) {
                    // Strike px → logical: the raster scale maps strike to device px, and
                    // inv_raster_scale maps device to layout units.
                    const to_logical = raster.scale * inv_raster_scale;
                    const cw_f = @as(f32, @floatFromInt(ColorAtlas.width));
                    const ch_f = @as(f32, @floatFromInt(ColorAtlas.height));
                    try colors_out.append(allocator, .{
                        .x = pen_x.* + shaped.x_offset + glyph.bearing_x * to_logical,
                        .y = baseline_y.* - shaped.y_offset - glyph.bearing_y * to_logical,
                        .w = @as(f32, @floatFromInt(glyph.width)) * to_logical,
                        .h = @as(f32, @floatFromInt(glyph.height)) * to_logical,
                        .u0 = @as(f32, @floatFromInt(glyph.atlas_x)) / cw_f,
                        .v0 = @as(f32, @floatFromInt(glyph.atlas_y)) / ch_f,
                        .u1 = @as(f32, @floatFromInt(glyph.atlas_x + glyph.width)) / cw_f,
                        .v1 = @as(f32, @floatFromInt(glyph.atlas_y + glyph.height)) / ch_f,
                    });
                }
            }
            pen_x.* += shaped.x_advance;
            baseline_y.* -= shaped.y_advance;
        }
    }

    pub fn uploadAtlasIfDirty(self: *FreeTypeTextRenderer, queue: *wgpu.Queue) void {
        if (!self.gpu_enabled) return; // CPU-only (native backend) uploads its own atlas
        const dirty = self.atlasDirtyRect() orelse return;
        const src_offset = @as(usize, dirty.y) * @as(usize, self.atlas_width) + dirty.x;
        const src_len = (@as(usize, dirty.height) - 1) * @as(usize, self.atlas_width) + dirty.width;

        queue.writeTexture(
            &.{
                .texture = self.texture,
                .mip_level = 0,
                .origin = .{
                    .x = dirty.x,
                    .y = dirty.y,
                    .z = 0,
                },
                .aspect = .all,
            },
            self.atlas_pixels.ptr + src_offset,
            src_len,
            &.{
                .offset = 0,
                .bytes_per_row = self.atlas_width,
                .rows_per_image = dirty.height,
            },
            &.{
                .width = dirty.width,
                .height = dirty.height,
                .depth_or_array_layers = 1,
            },
        );

        self.markAtlasClean();
    }

    fn getGlyph(
        self: *FreeTypeTextRenderer,
        face: ft.FT_Face,
        font_family: []const u8,
        glyph_index: u32,
        size: u16,
        style: FaceStyle,
        blur: f32,
    ) !Glyph {
        // Keyed by what is actually rasterized (the synth result), not by the request: every
        // (weight, italic) combination the face satisfies natively shares one atlas entry.
        // Blur is quantized to quarter-pixel buckets so shadow passes bake once per radius.
        const synth = synthFor(face, style, size);
        const blur_q: u16 = if (blur > 0) @intFromFloat(@min(@round(blur * 4.0), 4096.0)) else 0;
        const key = glyphKey(font_family, glyph_index, size, synth, blur_q);

        if (self.glyphs.get(key)) |glyph| {
            return glyph;
        }

        // UI text is rasterized at its physical display size into a grayscale coverage atlas.
        // Light hinting preserves horizontal metrics while sharpening vertical stems, closely
        // matching CoreText/Skia behavior at editor sizes without SDF's small-text blur.
        if (ft.FT_Set_Pixel_Sizes(face, 0, @intCast(size)) != 0)
            return error.FreeTypeSetPixelSizeFailed;
        // Synthetic styles transform the outline, so hinting would fight the shear/embolden —
        // load unhinted in that case (matches how browsers rasterize faux bold/italic).
        const target: c_long = if (synth.none()) ft.FT_LOAD_TARGET_LIGHT else ft.FT_LOAD_NO_HINTING;
        const load_flags: i32 = @intCast(ft.FT_LOAD_DEFAULT | ft.FT_LOAD_NO_BITMAP | target);
        if (ft.FT_Load_Glyph(face, @intCast(glyph_index), load_flags) != 0)
            return error.FreeTypeLoadGlyphFailed;

        const slot = face.*.glyph;
        if (!synth.none() and slot.*.format == ft.FT_GLYPH_FORMAT_OUTLINE) {
            if (synth.embolden > 0) {
                const s = synth.strength266();
                _ = ft.FT_Outline_EmboldenXY(&slot.*.outline, s, s);
            }
            if (synth.oblique) {
                // ~12° shear (tan ≈ 0.2126 in 16.16), the same slant FT_GlyphSlot_Oblique uses.
                var shear = ft.FT_Matrix{ .xx = 0x10000, .xy = 0x0366A, .yx = 0, .yy = 0x10000 };
                ft.FT_Outline_Transform(&slot.*.outline, &shear);
            }
        }
        if (ft.FT_Render_Glyph(slot, ft.FT_RENDER_MODE_NORMAL) != 0)
            return error.FreeTypeLoadGlyphFailed;
        const bitmap = slot.*.bitmap;

        const bitmap_width: u32 = bitmap.width;
        const bitmap_height: u32 = bitmap.rows;

        if (blur_q > 0 and bitmap_width > 0 and bitmap_height > 0 and
            bitmap.pixel_mode == ft.FT_PIXEL_MODE_GRAY and bitmap.buffer != null)
        {
            const glyph = try self.bakeBlurredGlyph(bitmap, @as(f32, @floatFromInt(blur_q)) / 4.0, slot);
            try self.glyphs.put(key, glyph);
            return glyph;
        }

        // Overflow first grows the atlas ×2 (up to Atlas.max) — glyphs re-bake lazily into the
        // larger store via the generation bump — and only at max size falls back to the
        // destructive reset-and-repack.
        const atlas_pos = blk: {
            while (true) {
                if (self.reserveAtlas(bitmap_width, bitmap_height)) |pos| {
                    break :blk pos;
                } else |_| {}
                if (!self.tryGrowAtlas())
                    break :blk try self.reserveAtlasAfterReset(bitmap_width, bitmap_height);
            }
        };

        if (bitmap_width > 0 and bitmap_height > 0) {
            try self.copyBitmapToAtlas(bitmap, atlas_pos.x, atlas_pos.y);
        }

        const glyph = Glyph{
            .atlas_x = atlas_pos.x,
            .atlas_y = atlas_pos.y,
            .width = bitmap_width,
            .height = bitmap_height,

            .bearing_x = @floatFromInt(slot.*.bitmap_left),
            .bearing_y = @floatFromInt(slot.*.bitmap_top),
            .advance_x = @as(f32, @floatFromInt(slot.*.advance.x)) / 64.0,
        };

        try self.glyphs.put(key, glyph);
        return glyph;
    }

    /// Bake a blurred copy of a freshly rendered glyph into the coverage atlas. The bitmap is
    /// zero-padded so the halo fits, box-blurred 3× (separable, ≈ Gaussian with σ = blur/2 —
    /// CSS text-shadow radius semantics), and stored with pad-adjusted bearings so quad
    /// emission downstream needs no changes.
    // ponytail: per-glyph blur double-darkens where adjacent halos overlap (vs whole-run blur);
    // invisible at radii ≤ ~8px — switch to a run-level mask bake if large radii are ever needed.
    fn bakeBlurredGlyph(self: *FreeTypeTextRenderer, bitmap: ft.FT_Bitmap, blur: f32, slot: ft.FT_GlyphSlot) !Glyph {
        const sigma = blur * 0.5;
        // Box radius r such that 3 passes ≈ Gaussian σ: 3 · r(r+1)/3 = σ².
        const rf = (@sqrt(1.0 + 4.0 * sigma * sigma) - 1.0) * 0.5;
        const r: u32 = @max(1, @as(u32, @intFromFloat(@round(rf))));
        const pad: u32 = 3 * r; // each pass spreads at most r per side
        const w: u32 = bitmap.width + 2 * pad;
        const h: u32 = bitmap.rows + 2 * pad;
        const len = @as(usize, w) * @as(usize, h);

        const buf = try self.allocator.alloc(u8, len);
        defer self.allocator.free(buf);
        const tmp = try self.allocator.alloc(u8, len);
        defer self.allocator.free(tmp);
        @memset(buf, 0);

        const pitch_abs: usize = if (bitmap.pitch < 0) @intCast(-bitmap.pitch) else @intCast(bitmap.pitch);
        const src_base: [*]const u8 = @ptrCast(bitmap.buffer);
        var row: usize = 0;
        while (row < bitmap.rows) : (row += 1) {
            const src_row_index = if (bitmap.pitch < 0) bitmap.rows - 1 - row else row;
            const src_row = src_base + src_row_index * pitch_abs;
            const dst_off = (@as(usize, pad) + row) * w + pad;
            @memcpy(buf[dst_off .. dst_off + bitmap.width], src_row[0..bitmap.width]);
        }

        var pass: usize = 0;
        while (pass < 3) : (pass += 1) {
            boxBlurRows(buf, tmp, w, h, r);
            boxBlurCols(tmp, buf, w, h, r);
        }

        const atlas_pos = blk: {
            while (true) {
                if (self.reserveAtlas(w, h)) |pos| {
                    break :blk pos;
                } else |_| {}
                if (!self.tryGrowAtlas())
                    break :blk try self.reserveAtlasAfterReset(w, h);
            }
        };

        row = 0;
        while (row < h) : (row += 1) {
            const dst_offset = (@as(usize, atlas_pos.y) + row) * @as(usize, self.atlas_width) + @as(usize, atlas_pos.x);
            @memcpy(self.atlas_pixels[dst_offset .. dst_offset + w], buf[row * @as(usize, w) ..][0..w]);
        }
        self.markAtlasDirty(atlas_pos.x, atlas_pos.y, w, h);

        return Glyph{
            .atlas_x = atlas_pos.x,
            .atlas_y = atlas_pos.y,
            .width = w,
            .height = h,
            .bearing_x = @as(f32, @floatFromInt(slot.*.bitmap_left)) - @as(f32, @floatFromInt(pad)),
            .bearing_y = @as(f32, @floatFromInt(slot.*.bitmap_top)) + @as(f32, @floatFromInt(pad)),
            .advance_x = @as(f32, @floatFromInt(slot.*.advance.x)) / 64.0,
        };
    }

    fn resolveFace(self: *const FreeTypeTextRenderer, font_family: []const u8) ?ft.FT_Face {
        if (self.faces.get(font_family)) |face| return face;
        return self.faces.get(self.default_font_family);
    }

    fn resetAtlas(self: *FreeTypeTextRenderer) void {
        self.glyphs.clearRetainingCapacity();
        // Shaped runs contain only glyph IDs and logical positions, not UVs, so they remain valid.
        // Handle-addressed layout entries (layout_cache) bake the same atlas UVs but are referenced by
        // long-lived u64 handles held by widgets, so they can't be dropped here without blanking the
        // holder. Instead bump the generation: each entry re-shapes itself lazily on its next draw
        // (see appendLayoutGlyphs / reshapeLayoutEntry) while keeping the handle valid.
        self.atlas_generation +%= 1;
        @memset(self.atlas_pixels, 0);
        self.atlas_dirty = true;
        self.dirty_min_x = 0;
        self.dirty_min_y = 0;
        self.dirty_max_x = self.atlas_width;
        self.dirty_max_y = self.atlas_height;
        self.pen_x = 1;
        self.pen_y = 1;
        self.row_h = 0;
    }

    /// Double the coverage atlas (clamped to Atlas.max) and reset it so glyphs re-bake lazily into
    /// the larger store. The new CPU buffer and GPU resources are fully created BEFORE anything old
    /// is released, so any failure leaves the renderer exactly as it was (returns false → the caller
    /// degrades to the destructive reset-and-repack at the current size). Returns false at max size.
    fn tryGrowAtlas(self: *FreeTypeTextRenderer) bool {
        if (self.atlas_width >= Atlas.max and self.atlas_height >= Atlas.max) return false;
        const new_w = @min(self.atlas_width * 2, Atlas.max);
        const new_h = @min(self.atlas_height * 2, Atlas.max);

        const new_pixels = self.allocator.alloc(u8, @as(usize, new_w) * new_h) catch return false;

        if (self.gpu_enabled) {
            const dev = self.gpu_device orelse {
                self.allocator.free(new_pixels);
                return false;
            };
            const new_tex = dev.createTexture(&.{
                .label = wgpu.StringView.fromSlice("zigote text atlas"),
                .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
                .dimension = .@"2d",
                .size = .{ .width = new_w, .height = new_h, .depth_or_array_layers = 1 },
                .format = .r8_unorm,
                .mip_level_count = 1,
                .sample_count = 1,
            }) orelse {
                self.allocator.free(new_pixels);
                return false;
            };
            const new_view = new_tex.createView(null) orelse {
                new_tex.release();
                self.allocator.free(new_pixels);
                return false;
            };
            const new_bg = createTextBindGroup(dev, self.bind_group_layout, new_view, self.sampler) orelse {
                new_view.release();
                new_tex.release();
                self.allocator.free(new_pixels);
                return false;
            };
            // wgpu keeps the old texture alive until in-flight GPU work that references it completes.
            self.bind_group.release();
            self.texture_view.release();
            self.texture.release();
            self.texture = new_tex;
            self.texture_view = new_view;
            self.bind_group = new_bg;
        }

        self.allocator.free(self.atlas_pixels);
        self.atlas_pixels = new_pixels;
        self.atlas_width = new_w;
        self.atlas_height = new_h;
        // Must run AFTER the buffer/dims swap: it memsets atlas_pixels and stamps the dirty
        // extents from the new dims (full first upload of the grown texture).
        self.resetAtlas();
        std.log.info("zigote: glyph atlas grown to {d}x{d}", .{ new_w, new_h });
        return true;
    }

    fn reserveAtlasAfterReset(
        self: *FreeTypeTextRenderer,
        width: u32,
        height: u32,
    ) !Pos {
        self.resetAtlas();
        return self.reserveAtlas(width, height);
    }

    const Pos = struct { x: u32, y: u32 };

    fn reserveAtlas(
        self: *FreeTypeTextRenderer,
        width: u32,
        height: u32,
    ) !Pos {
        if (width == 0 or height == 0) {
            return .{
                .x = 0,
                .y = 0,
            };
        }

        const padded_w = width + 2;
        const padded_h = height + 2;

        if (self.pen_x + padded_w >= self.atlas_width) {
            self.pen_x = 1;
            self.pen_y += self.row_h + 1;
            self.row_h = 0;
        }

        if (self.pen_y + padded_h >= self.atlas_height) {
            return error.FreeTypeAtlasFull;
        }

        const out = Pos{
            .x = self.pen_x,
            .y = self.pen_y,
        };

        self.pen_x += padded_w;
        self.row_h = @max(self.row_h, padded_h);

        return out;
    }

    fn copyBitmapToAtlas(
        self: *FreeTypeTextRenderer,
        bitmap: ft.FT_Bitmap,
        dst_x: u32,
        dst_y: u32,
    ) !void {
        if (bitmap.pixel_mode != ft.FT_PIXEL_MODE_GRAY) {
            return error.FreeTypeUnsupportedPixelMode;
        }

        if (bitmap.buffer == null) return;

        const width: usize = bitmap.width;
        const height: usize = bitmap.rows;

        const pitch_abs: usize = if (bitmap.pitch < 0)
            @intCast(-bitmap.pitch)
        else
            @intCast(bitmap.pitch);

        const src_base: [*]const u8 = @ptrCast(bitmap.buffer);

        var row: usize = 0;
        while (row < height) : (row += 1) {
            const src_row_index = if (bitmap.pitch < 0)
                height - 1 - row
            else
                row;

            const src_row = src_base + src_row_index * pitch_abs;

            const dst_offset =
                (@as(usize, dst_y) + row) * @as(usize, self.atlas_width) +
                @as(usize, dst_x);

            @memcpy(
                self.atlas_pixels[dst_offset .. dst_offset + width],
                src_row[0..width],
            );
        }

        self.markAtlasDirty(dst_x, dst_y, bitmap.width, bitmap.rows);
    }

    fn markAtlasDirty(self: *FreeTypeTextRenderer, x: u32, y: u32, width: u32, height: u32) void {
        if (width == 0 or height == 0) return;
        self.atlas_dirty = true;
        self.dirty_min_x = @min(self.dirty_min_x, x);
        self.dirty_min_y = @min(self.dirty_min_y, y);
        self.dirty_max_x = @max(self.dirty_max_x, x + width);
        self.dirty_max_y = @max(self.dirty_max_y, y + height);
    }

    /// A strike (or exact size) selected on an emoji face: `actual_px` is what FreeType will
    /// rasterize, `scale` maps that to the requested size. Scalable faces select exactly
    /// (scale 1); bitmap-only faces (Apple Color Emoji sbix, Noto CBDT) carry fixed strikes,
    /// where FT_Set_Pixel_Sizes simply fails — the closest strike is selected and drawn scaled.
    const EmojiSize = struct { actual_px: u16, scale: f32 };

    fn emojiSelectSize(face: ft.FT_Face, want: u16) !EmojiSize {
        if ((face.*.face_flags & ft.FT_FACE_FLAG_SCALABLE) != 0) {
            if (ft.FT_Set_Pixel_Sizes(face, 0, @intCast(want)) != 0)
                return error.FreeTypeSetPixelSizeFailed;
            return .{ .actual_px = want, .scale = 1 };
        }

        const count: usize = @intCast(face.*.num_fixed_sizes);
        if (count == 0) return error.FreeTypeSetPixelSizeFailed;
        // Smallest strike at or above the request (downscaling keeps edges clean); if the request
        // exceeds every strike, the largest available.
        var best: usize = 0;
        var best_h: i32 = @intCast(face.*.available_sizes[0].height);
        for (1..count) |i| {
            const h: i32 = @intCast(face.*.available_sizes[i].height);
            const want_i: i32 = @intCast(want);
            const better = if (best_h >= want_i)
                h >= want_i and h < best_h
            else
                h > best_h;
            if (better) {
                best = i;
                best_h = h;
            }
        }
        if (ft.FT_Select_Size(face, @intCast(best)) != 0)
            return error.FreeTypeSetPixelSizeFailed;
        return .{
            .actual_px = @intCast(best_h),
            .scale = @as(f32, @floatFromInt(want)) / @as(f32, @floatFromInt(best_h)),
        };
    }

    /// Shape an emoji segment once and cache it in shaped_run_cache. Own key namespace
    /// (separator byte 1 vs getShapedRun's 0): emoji segments are shaped against the directly
    /// looked-up emoji face with HarfBuzz default features, so they must never alias a regular
    /// run of the same (family, text, size) shaped via resolveFace with explicit features.
    /// Cached advances are already scaled to the requested pixel size, so fixed-strike faces
    /// measure and draw identically at every size.
    fn getEmojiShapedRun(
        self: *FreeTypeTextRenderer,
        face: ft.FT_Face,
        text: []const u8,
        emoji_family: []const u8,
        pixel_size: u16,
    ) !*const ShapedRun {
        var hasher = std.hash.Wyhash.init(@as(u64, pixel_size));
        hasher.update(emoji_family);
        hasher.update(&[_]u8{1});
        hasher.update(text);
        const key = hasher.final();

        if (self.shaped_run_cache.getPtr(key)) |run| return run;

        // Bound the cache; a wholesale clear is fine — entries rebuild lazily when next drawn.
        if (self.shaped_run_cache.count() >= 8192) self.clearShapedRunCache();

        const sel = try emojiSelectSize(face, pixel_size);

        const hb_font = hb_ft_font_create_referenced(face) orelse return error.HarfBuzzFontCreateFailed;
        defer hb.hb_font_destroy(hb_font);
        // NOTE: no hb_ft_font_set_funcs here — create_referenced already installed the FT funcs
        // on THIS face; the extra call cloned a second face from the font blob per shaped run
        // (an FT_New_Memory_Face each time) for identical unhinted metrics.

        const buffer = hb.hb_buffer_create() orelse return error.HarfBuzzBufferCreateFailed;
        defer hb.hb_buffer_destroy(buffer);
        hb.hb_buffer_add_utf8(buffer, text.ptr, @intCast(text.len), 0, @intCast(text.len));
        hb.hb_buffer_guess_segment_properties(buffer);
        hb.hb_shape(hb_font, buffer, null, 0);

        var info_count: c_uint = 0;
        const infos_ptr = hb.hb_buffer_get_glyph_infos(buffer, &info_count) orelse return error.HarfBuzzGlyphInfoFailed;
        var pos_count: c_uint = 0;
        const pos_ptr = hb.hb_buffer_get_glyph_positions(buffer, &pos_count) orelse return error.HarfBuzzGlyphPositionFailed;
        const glyph_count: usize = @intCast(@min(info_count, pos_count));

        const shaped = try self.allocator.alloc(ShapedGlyph, glyph_count);
        errdefer self.allocator.free(shaped);
        var advance_x: f32 = 0;
        var advance_y: f32 = 0;
        for (shaped, infos_ptr[0..glyph_count], pos_ptr[0..glyph_count]) |*out, info, pos| {
            out.* = .{
                .glyph_index = info.codepoint,
                .cluster = info.cluster,
                .x_offset = hbPositionToPixels(pos.x_offset) * sel.scale,
                .y_offset = hbPositionToPixels(pos.y_offset) * sel.scale,
                .x_advance = hbPositionToPixels(pos.x_advance) * sel.scale,
                .y_advance = hbPositionToPixels(pos.y_advance) * sel.scale,
            };
            advance_x += out.x_advance;
            advance_y -= out.y_advance;
        }

        try self.shaped_run_cache.put(key, .{
            .glyphs = shaped,
            .advance_x = advance_x,
            .advance_y = advance_y,
        });
        return self.shaped_run_cache.getPtr(key).?;
    }

    /// One baked color glyph plus the factor mapping its strike pixels to the requested size.
    const EmojiRaster = struct { glyph: Glyph, scale: f32 };

    /// Bake (or fetch) a color glyph at the strike closest to `want_px`. The cache and the atlas
    /// hold ONE image per (family, glyph, strike) — every requested size maps to a strike and
    /// scales the quad, so a size ramp doesn't multiply 128×128 bakes. Returns null when the
    /// face can't produce a BGRA bitmap for the glyph (the caller advances the pen regardless);
    /// a null is remembered as an empty marker so a CBDT PNG decode isn't retried every frame.
    fn rasterEmojiGlyph(
        self: *FreeTypeTextRenderer,
        face: ft.FT_Face,
        emoji_family: []const u8,
        glyph_index: u32,
        want_px: u16,
    ) ?EmojiRaster {
        const sel = emojiSelectSize(face, want_px) catch return null;
        const key = glyphKey(emoji_family, glyph_index, sel.actual_px, .{}, 0);

        if (self.color_glyphs.get(key)) |glyph| {
            return .{ .glyph = glyph, .scale = sel.scale };
        }

        const load_flags = ft.FT_LOAD_COLOR | ft.FT_LOAD_RENDER;
        if (ft.FT_Load_Glyph(face, @intCast(glyph_index), load_flags) != 0) return null;

        const slot = face.*.glyph;
        const bitmap = slot.*.bitmap;
        const bw: u32 = bitmap.width;
        const bh: u32 = bitmap.rows;

        if (bitmap.pixel_mode == ft.FT_PIXEL_MODE_BGRA and bw > 0 and bh > 0) {
            // First color glyph materializes the atlas; on failure nothing is cached and the
            // next bake retries.
            self.ensureColorAtlas() catch return null;
            // A full atlas resets and repacks (mirroring the coverage atlas): live layouts and
            // this frame's earlier quads go stale, but the generation bump re-bakes them — far
            // better than a permanently emoji-less session that re-decodes PNGs every frame.
            const atlas_pos = self.reserveColorAtlas(bw, bh) catch blk: {
                self.resetColorAtlas();
                break :blk self.reserveColorAtlas(bw, bh) catch return null;
            };
            self.copyColorBitmapToAtlas(bitmap, atlas_pos.x, atlas_pos.y) catch {};
            const glyph = Glyph{
                .atlas_x = atlas_pos.x,
                .atlas_y = atlas_pos.y,
                .width = bw,
                .height = bh,
                .bearing_x = @floatFromInt(slot.*.bitmap_left),
                .bearing_y = @floatFromInt(slot.*.bitmap_top),
                .advance_x = @as(f32, @floatFromInt(slot.*.advance.x)) / 64.0,
            };
            self.color_glyphs.put(key, glyph) catch {};
            return .{ .glyph = glyph, .scale = sel.scale };
        }

        // Monochrome or zero-sized: never drawn (no fallback to the text atlas to avoid font
        // confusion). Cache an empty marker so the load isn't repeated every frame.
        self.color_glyphs.put(key, .{
            .atlas_x = 0,
            .atlas_y = 0,
            .width = 0,
            .height = 0,
            .bearing_x = 0,
            .bearing_y = 0,
            .advance_x = 0,
        }) catch {};
        return null;
    }

    /// Zero the color atlas and forget every baked color glyph, bumping the shared atlas
    /// generation so cached layouts re-bake their (now stale) color UVs on next draw.
    fn resetColorAtlas(self: *FreeTypeTextRenderer) void {
        self.color_glyphs.clearRetainingCapacity();
        if (self.color_atlas_pixels) |pixels| {
            @memset(pixels, 0);
            self.color_atlas_dirty = true;
        }
        self.color_pen_x = 1;
        self.color_pen_y = 1;
        self.color_row_h = 0;
        self.atlas_generation +%= 1;
    }

    /// Shape and rasterize a run of emoji codepoints using the given emoji font face.
    /// Color glyphs (FT_PIXEL_MODE_BGRA) are copied to the RGBA color atlas and appended
    /// to self.pending_color_quads (renderer-lifetime storage) for wgpu.zig to flush.
    /// Monochrome/missing glyphs are skipped — no fallback to the text atlas.
    fn appendEmojiShapedSegment(
        self: *FreeTypeTextRenderer,
        text: []const u8,
        emoji_family: []const u8,
        pixel_size: u16,
        offset_x: f32,
        offset_y: f32,
        frame_width: f32,
        frame_height: f32,
        pen_x: *f32,
        baseline_y: *f32,
    ) !void {
        _ = frame_width;
        _ = frame_height;
        if (text.len == 0) return;

        // Direct lookup only — no fallback to default font. If the emoji face isn't
        // loaded, bail silently rather than using a font that has no color glyphs.
        const face = self.faces.get(emoji_family) orelse return;

        const run = self.getEmojiShapedRun(face, text, emoji_family, pixel_size) catch return;

        for (run.glyphs) |shaped| {
            if (self.rasterEmojiGlyph(face, emoji_family, shaped.glyph_index, pixel_size)) |raster| {
                const glyph = raster.glyph;
                if (glyph.width > 0 and glyph.height > 0) {
                    const x0 = pen_x.* + shaped.x_offset + glyph.bearing_x * raster.scale;
                    const y0 = baseline_y.* - shaped.y_offset - glyph.bearing_y * raster.scale;
                    const cw_f = @as(f32, @floatFromInt(ColorAtlas.width));
                    const ch_f = @as(f32, @floatFromInt(ColorAtlas.height));
                    self.pending_color_quads.append(self.allocator, .{
                        .x = x0 - offset_x,
                        .y = y0 - offset_y,
                        .w = @as(f32, @floatFromInt(glyph.width)) * raster.scale,
                        .h = @as(f32, @floatFromInt(glyph.height)) * raster.scale,
                        .u0 = @as(f32, @floatFromInt(glyph.atlas_x)) / cw_f,
                        .v0 = @as(f32, @floatFromInt(glyph.atlas_y)) / ch_f,
                        .u1 = @as(f32, @floatFromInt(glyph.atlas_x + glyph.width)) / cw_f,
                        .v1 = @as(f32, @floatFromInt(glyph.atlas_y + glyph.height)) / ch_f,
                    }) catch {};
                }
            }

            pen_x.* += shaped.x_advance;
            baseline_y.* -= shaped.y_advance;
        }
    }

    /// Reserve a region in the RGBA color atlas. Returns error.AtlasFull on overflow.
    fn reserveColorAtlas(self: *FreeTypeTextRenderer, width: u32, height: u32) !Pos {
        if (width == 0 or height == 0) return .{ .x = 0, .y = 0 };

        const padded_w = width + 2;
        const padded_h = height + 2;

        if (self.color_pen_x + padded_w >= ColorAtlas.width) {
            self.color_pen_x = 1;
            self.color_pen_y += self.color_row_h + 1;
            self.color_row_h = 0;
        }

        if (self.color_pen_y + padded_h >= ColorAtlas.height) {
            return error.AtlasFull;
        }

        const out = Pos{ .x = self.color_pen_x, .y = self.color_pen_y };
        self.color_pen_x += padded_w;
        self.color_row_h = @max(self.color_row_h, padded_h);
        return out;
    }

    /// Copy a BGRA pre-multiplied FreeType bitmap to the RGBA color atlas, un-premultiplying alpha.
    fn copyColorBitmapToAtlas(self: *FreeTypeTextRenderer, bitmap: ft.FT_Bitmap, dst_x: u32, dst_y: u32) !void {
        if (bitmap.buffer == null) return;
        const pixels = self.color_atlas_pixels orelse return;
        const width: usize = bitmap.width;
        const height: usize = bitmap.rows;
        const pitch_abs: usize = if (bitmap.pitch < 0) @intCast(-bitmap.pitch) else @intCast(bitmap.pitch);
        const src_base: [*]const u8 = @ptrCast(bitmap.buffer);

        var row: usize = 0;
        while (row < height) : (row += 1) {
            const src_row_index: usize = if (bitmap.pitch < 0) height - 1 - row else row;
            const src_row = src_base + src_row_index * pitch_abs;

            var col: usize = 0;
            while (col < width) : (col += 1) {
                // FreeType BGRA: src[0]=B, src[1]=G, src[2]=R, src[3]=A (pre-multiplied)
                const b = src_row[col * 4 + 0];
                const g = src_row[col * 4 + 1];
                const r = src_row[col * 4 + 2];
                const a = src_row[col * 4 + 3];

                const dst_offset = ((@as(usize, dst_y) + row) * ColorAtlas.width + (@as(usize, dst_x) + col)) * 4;
                if (a == 0) {
                    pixels[dst_offset + 0] = 0;
                    pixels[dst_offset + 1] = 0;
                    pixels[dst_offset + 2] = 0;
                    pixels[dst_offset + 3] = 0;
                } else {
                    // Un-premultiply: divide by alpha
                    pixels[dst_offset + 0] = @intCast(@min(255, @as(u32, r) * 255 / @as(u32, a)));
                    pixels[dst_offset + 1] = @intCast(@min(255, @as(u32, g) * 255 / @as(u32, a)));
                    pixels[dst_offset + 2] = @intCast(@min(255, @as(u32, b) * 255 / @as(u32, a)));
                    pixels[dst_offset + 3] = a;
                }
            }
        }

        self.color_atlas_dirty = true;
    }
};

fn isEmojiCodepoint(cp: u21) bool {
    return switch (cp) {
        0x1F000...0x1F02F => true, // Mahjong Tiles
        0x1F030...0x1F09F => true, // Domino Tiles
        0x1F0A0...0x1F0FF => true, // Playing Cards
        0x1F100...0x1F1FF => true, // Enclosed Alphanumeric Supplement / Regional Indicators
        0x1F200...0x1F2FF => true, // Enclosed Ideographic Supplement
        0x1F300...0x1F5FF => true, // Misc Symbols & Pictographs
        0x1F600...0x1F64F => true, // Emoticons
        0x1F650...0x1F67F => true, // Ornamental Dingbats
        0x1F680...0x1F6FF => true, // Transport & Map
        0x1F700...0x1F77F => true, // Alchemical Symbols
        0x1F780...0x1F7FF => true, // Geometric Shapes Extended
        0x1F800...0x1F8FF => true, // Supplemental Arrows-C
        0x1F900...0x1F9FF => true, // Supplemental Symbols & Pictographs
        0x1FA00...0x1FA6F => true, // Chess Symbols
        0x1FA70...0x1FAFF => true, // Symbols & Pictographs Extended-A
        0x2300...0x23FF => true, // Miscellaneous Technical
        0x2600...0x26FF => true, // Miscellaneous Symbols
        0x2700...0x27BF => true, // Dingbats
        0x2B00...0x2BFF => true, // Miscellaneous Symbols and Arrows
        // Combining characters — must stay in the emoji segment so HarfBuzz shapes them
        // together with the base emoji (e.g. ❤️ = U+2764 + U+FE0F, 👨‍💻 = ZWJ sequence)
        0x200D => true, // Zero Width Joiner (ZWJ sequences)
        0x20E3 => true, // Combining Enclosing Keycap (1️⃣ etc.)
        0xFE00...0xFE0F => true, // Variation Selectors 1-16 (emoji vs text presentation)
        0xE0020...0xE007F => true, // Tags (used in country flag sequences)
        // BMP emoji not covered by block ranges above
        0x00A9 => true, // © Copyright
        0x00AE => true, // ® Registered
        0x203C => true, // ‼ Double Exclamation
        0x2049 => true, // ⁉ Exclamation Question
        0x2122 => true, // ™ Trade Mark
        0x2139 => true, // ℹ Information
        0x2194...0x2199 => true, // ↔↕↖↗↘↙ Arrows
        0x21A9...0x21AA => true, // ↩↪ Curving arrows
        0x24C2 => true, // Ⓜ Circled M
        0x25AA...0x25AB => true, // ▪▫ Small squares
        0x25B6 => true, // ▶ Play button
        0x25C0 => true, // ◀ Reverse button
        0x25FB...0x25FE => true, // ◻◼◽◾ Medium/large squares
        0x2934...0x2935 => true, // ⤴⤵ Curving arrows
        0x3030 => true, // 〰 Wavy Dash
        0x303D => true, // 〽 Part Alternation Mark
        0x3297 => true, // ㊗ Circled Ideograph Congratulation
        0x3299 => true, // ㊙ Circled Ideograph Secret
        else => false,
    };
}

/// One horizontal box-blur pass (window 2r+1, out-of-range treated as zero — matches the
/// zero-padded scratch buffer bakeBlurredGlyph builds). Running-sum, O(w·h).
fn boxBlurRows(src: []const u8, dst: []u8, w: u32, h: u32, r: u32) void {
    const wu: usize = w;
    const win: u32 = 2 * r + 1;
    var y: usize = 0;
    while (y < h) : (y += 1) {
        const row = src[y * wu ..][0..wu];
        const out = dst[y * wu ..][0..wu];
        var sum: u32 = 0;
        var i: usize = 0;
        while (i < @min(r + 1, wu)) : (i += 1) sum += row[i];
        var x: usize = 0;
        while (x < wu) : (x += 1) {
            out[x] = @intCast(sum / win);
            if (x + r + 1 < wu) sum += row[x + r + 1];
            if (x >= r) sum -= row[x - r];
        }
    }
}

/// Vertical counterpart of boxBlurRows.
fn boxBlurCols(src: []const u8, dst: []u8, w: u32, h: u32, r: u32) void {
    const wu: usize = w;
    const hu: usize = h;
    const win: u32 = 2 * r + 1;
    var x: usize = 0;
    while (x < wu) : (x += 1) {
        var sum: u32 = 0;
        var i: usize = 0;
        while (i < @min(r + 1, hu)) : (i += 1) sum += src[i * wu + x];
        var y: usize = 0;
        while (y < hu) : (y += 1) {
            dst[y * wu + x] = @intCast(sum / win);
            if (y + r + 1 < hu) sum += src[(y + r + 1) * wu + x];
            if (y >= r) sum -= src[(y - r) * wu + x];
        }
    }
}

fn glyphKey(font_family: []const u8, glyph_index: u32, size: u16, synth: Synth, blur_q: u16) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(font_family);
    hasher.update(std.mem.asBytes(&size));
    hasher.update(std.mem.asBytes(&glyph_index));
    if (!synth.none()) {
        const strength: i32 = synth.strength266();
        hasher.update(std.mem.asBytes(&strength));
        hasher.update(&[_]u8{@intFromBool(synth.oblique)});
    }
    if (blur_q > 0) hasher.update(std.mem.asBytes(&blur_q));
    return hasher.final();
}

fn hbPositionToPixels(value: c_int) f32 {
    return @as(f32, @floatFromInt(value)) / 64.0;
}

fn fontPixelSize(size: f32) u16 {
    return @intFromFloat(@round(std.math.clamp(size, 1.0, 65535.0)));
}

/// First value in `sorted` strictly greater than `cluster`, or null if none.
fn nextClusterAbove(sorted: []const u32, cluster: u32) ?u32 {
    var lo: usize = 0;
    var hi: usize = sorted.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] <= cluster) lo = mid + 1 else hi = mid;
    }
    return if (lo < sorted.len) sorted[lo] else null;
}

/// True when glyph `i` is the last glyph of its HarfBuzz cluster — the only place per-cluster
/// tracking (letter/word spacing) may advance the pen. Works for LTR and RTL runs alike: either
/// way, a cluster's glyphs are contiguous and share one cluster value.
fn clusterEndsAt(glyphs: []const ShapedGlyph, i: usize) bool {
    return i + 1 >= glyphs.len or glyphs[i + 1].cluster != glyphs[i].cluster;
}

/// The FaceStyle a paint command carries (paint.Text stores weight/style as enums).
fn styleOf(text: zg.paint.Text) FaceStyle {
    return .{
        .weight = @intFromEnum(text.font_weight),
        .italic = text.font_style == .italic,
    };
}

fn clusterIsSpace(text: []const u8, cluster: u32) bool {
    const start: usize = @intCast(cluster);
    if (start >= text.len) return false;
    const sequence_len = std.unicode.utf8ByteSequenceLength(text[start]) catch return false;
    const end = @min(start + sequence_len, text.len);
    const cp = std.unicode.utf8Decode(text[start..end]) catch return false;
    return switch (cp) {
        0x0009...0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2000...0x200A,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
        => true,
        else => false,
    };
}

/// Marks and joiners that attach to the character before them: combining diacriticals (and their
/// supplement/extended blocks), the symbol marks, the half-marks, variation selectors and the
/// zero-width joiner. These must stay in their base character's run — a mark shaped against a
/// different face than its base lands in the wrong place, if it lands at all.
fn isCombining(cp: u21) bool {
    return switch (cp) {
        0x0300...0x036F => true, // combining diacritical marks
        0x1AB0...0x1AFF => true, // …extended
        0x1DC0...0x1DFF => true, // …supplement
        0x20D0...0x20FF => true, // combining marks for symbols
        0xFE00...0xFE0F => true, // variation selectors
        0xFE20...0xFE2F => true, // combining half marks
        0x200D => true, // zero-width joiner (emoji sequences)
        else => false,
    };
}

fn effectiveFontFamily(self: *const FreeTypeTextRenderer, text: zg.paint.Text) []const u8 {
    if (text.font_family) |family| return family;
    if (text.font_family_fallback.len > 0) return text.font_family_fallback[0];
    return self.default_font_family;
}

fn appendGlyphQuad(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(TextVertex),
    offset_x: f32,
    offset_y: f32,
    frame_width: f32,
    frame_height: f32,
    x0_px: f32,
    y0_px: f32,
    x1_px: f32,
    y1_px: f32,
    uv0_x: f32,
    uv0_y: f32,
    uv1_x: f32,
    uv1_y: f32,
    color: [4]u8,
) !void {
    const x0 = (x0_px - offset_x) / frame_width * 2.0 - 1.0;
    const y0 = 1.0 - (y0_px - offset_y) / frame_height * 2.0;
    const x1 = (x1_px - offset_x) / frame_width * 2.0 - 1.0;
    const y1 = 1.0 - (y1_px - offset_y) / frame_height * 2.0;

    // Corners TL,TR,BL,BR — the order wgpu.zig's shared quad index pattern (ensureQuadIndexBuffer)
    // expands into the two triangles the old 6-vertex emission produced.
    try vertices.appendSlice(allocator, &.{
        .{
            .position = .{ x0, y0 },
            .uv = .{ uv0_x, uv0_y },
            .color = color,
        },
        .{
            .position = .{ x1, y0 },
            .uv = .{ uv1_x, uv0_y },
            .color = color,
        },
        .{
            .position = .{ x0, y1 },
            .uv = .{ uv0_x, uv1_y },
            .color = color,
        },
        .{
            .position = .{ x1, y1 },
            .uv = .{ uv1_x, uv1_y },
            .color = color,
        },
    });
}

fn colorToU8(color: zg.Color) [4]u8 {
    return .{ color.r, color.g, color.b, color.a };
}

const text_shader_source =
    \\@group(0) @binding(0) var text_tex: texture_2d<f32>;
    \\@group(0) @binding(1) var text_sampler: sampler;
    \\
    \\// Per-batch rounded clip — see shape_shader_source.wgsl. Slot 0 (all zeros) = disabled, so
    \\// unclipped text multiplies coverage by exactly 1.0 and stays byte-identical.
    \\struct RoundedClip {
    \\  rect: vec4<f32>,
    \\  radius: vec4<f32>,
    \\};
    \\
    \\@group(1) @binding(0) var<uniform> rounded_clip: RoundedClip;
    \\
    \\fn rounded_clip_coverage(frag_pos: vec2<f32>) -> f32 {
    \\  if (rounded_clip.radius.x <= 0.0) {
    \\    return 1.0;
    \\  }
    \\  let q = abs(frag_pos - rounded_clip.rect.xy) - rounded_clip.rect.zw + vec2<f32>(rounded_clip.radius.x);
    \\  let dist = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rounded_clip.radius.x;
    \\  return clamp(0.5 - dist, 0.0, 1.0);
    \\}
    \\
    \\struct VertexOut {
    \\  @builtin(position) position: vec4<f32>,
    \\  @location(0) uv: vec2<f32>,
    \\  @location(1) color: vec4<f32>,
    \\};
    \\
    \\@vertex
    \\fn vs_main(
    \\  @location(0) position: vec2<f32>,
    \\  @location(1) uv: vec2<f32>,
    \\  @location(2) color: vec4<f32>,
    \\) -> VertexOut {
    \\  var out: VertexOut;
    \\  out.position = vec4<f32>(position, 0.0, 1.0);
    \\  out.uv = uv;
    \\  out.color = color;
    \\  return out;
    \\}
    \\
    \\@fragment
    \\fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    \\  // Coverage gamma: the swapchain is sRGB, so straight alpha blending happens in non-linear
    \\  // space, and the error is asymmetric — light-on-dark text loses its antialiased edges and
    \\  // reads thin and washy, while dark-on-light text gains ink and reads smudged. One fixed
    \\  // exponent can't serve both themes, so it follows the text's own luminance: bright glyphs
    \\  // boost coverage (~1/1.2), dark glyphs thin it slightly (~1.08), mid-tones blend. pow
    \\  // leaves 0 and 1 fixed either way, so solid pixels never fatten and holes never fill.
    \\  // Quad origins are pixel-snapped at emit, so the linear filter only ever smooths genuine
    \\  // sub-glyph detail, not placement error.
    \\  let lum = dot(in.color.rgb, vec3<f32>(0.299, 0.587, 0.114));
    \\  let exponent = mix(1.08, 1.0 / 1.2, lum);
    \\  let coverage = pow(textureSample(text_tex, text_sampler, in.uv).r, exponent);
    \\  return vec4<f32>(in.color.rgb, in.color.a * coverage * rounded_clip_coverage(in.position.xy));
    \\}
;

fn createTextBindGroupLayout(device: *wgpu.Device) ?*wgpu.BindGroupLayout {
    const entries = [_]wgpu.BindGroupLayoutEntry{
        .{
            .binding = 0,
            .visibility = wgpu.ShaderStages.fragment,
            .texture = .{
                .sample_type = .float,
                .view_dimension = .@"2d",
                .multisampled = 0,
            },
        },
        .{
            .binding = 1,
            .visibility = wgpu.ShaderStages.fragment,
            .sampler = .{
                .type = .filtering,
            },
        },
    };

    return device.createBindGroupLayout(&.{
        .label = wgpu.StringView.fromSlice("zigote text bind group layout"),
        .entry_count = entries.len,
        .entries = &entries,
    });
}

fn createTextBindGroup(
    device: *wgpu.Device,
    layout: *wgpu.BindGroupLayout,
    texture_view: *wgpu.TextureView,
    sampler: *wgpu.Sampler,
) ?*wgpu.BindGroup {
    const entries = [_]wgpu.BindGroupEntry{
        .{
            .binding = 0,
            .texture_view = texture_view,
        },
        .{
            .binding = 1,
            .sampler = sampler,
        },
    };

    return device.createBindGroup(&.{
        .label = wgpu.StringView.fromSlice("zigote text bind group"),
        .layout = layout,
        .entry_count = entries.len,
        .entries = &entries,
    });
}

fn createTextPipelineLayout(
    device: *wgpu.Device,
    bind_group_layout: *wgpu.BindGroupLayout,
    clip_bind_group_layout: *wgpu.BindGroupLayout,
) ?*wgpu.PipelineLayout {
    const bind_group_layouts = [_]*wgpu.BindGroupLayout{
        bind_group_layout,
        clip_bind_group_layout,
    };

    return device.createPipelineLayout(&.{
        .label = wgpu.StringView.fromSlice("zigote text pipeline layout"),
        .bind_group_layout_count = bind_group_layouts.len,
        .bind_group_layouts = &bind_group_layouts,
    });
}

fn createTextPipeline(
    device: *wgpu.Device,
    format: wgpu.TextureFormat,
    pipeline_layout: *wgpu.PipelineLayout,
) ?*wgpu.RenderPipeline {
    var shader_desc = wgpu.shaderModuleWGSLDescriptor(.{
        .label = "zigote text shader",
        .code = text_shader_source,
    });

    const shader = device.createShaderModule(&shader_desc) orelse return null;
    defer shader.release();

    const attributes = [_]wgpu.VertexAttribute{
        .{
            .format = .float32x2,
            .offset = @offsetOf(TextVertex, "position"),
            .shader_location = 0,
        },
        .{
            .format = .float32x2,
            .offset = @offsetOf(TextVertex, "uv"),
            .shader_location = 1,
        },
        .{
            .format = .unorm8x4,
            .offset = @offsetOf(TextVertex, "color"),
            .shader_location = 2,
        },
    };

    const vertex_buffer_layout = wgpu.VertexBufferLayout{
        .array_stride = @sizeOf(TextVertex),
        .attribute_count = attributes.len,
        .attributes = &attributes,
    };

    const color_target = wgpu.ColorTargetState{
        .format = format,
        .blend = &wgpu.BlendState.alpha_blending,
    };

    const fragment = wgpu.FragmentState{
        .module = shader,
        .entry_point = wgpu.StringView.fromSlice("fs_main"),
        .target_count = 1,
        .targets = @ptrCast(&color_target),
    };

    return device.createRenderPipeline(&.{
        .label = wgpu.StringView.fromSlice("zigote text pipeline"),
        .layout = pipeline_layout,
        .vertex = .{
            .module = shader,
            .entry_point = wgpu.StringView.fromSlice("vs_main"),
            .buffer_count = 1,
            .buffers = @ptrCast(&vertex_buffer_layout),
        },
        .primitive = .{},
        .multisample = .{},
        .fragment = &fragment,
    });
}

test "handle-addressed layout survives an atlas reset and re-shapes lazily" {
    const allocator = std.testing.allocator;

    // CPU-only renderer; an empty-text layout needs no font face or shaping, so no fonts are loaded.
    var renderer = try FreeTypeTextRenderer.initCpu(allocator, &.{}, "default");
    defer renderer.deinit();

    // Stand in for what appendTextLayout produces: a handle-addressed entry whose UVs were baked at
    // the current atlas generation. Built by hand so the test needs no glyph rasterisation.
    const handle: u64 = 1;
    try renderer.layout_cache.put(handle, .{
        .glyphs = try allocator.dupe(TextLayoutGlyph, &[_]TextLayoutGlyph{}),
        .carets = try allocator.dupe(TextCaret, &[_]TextCaret{}),
        .width = 0,
        .height = 0,
        .text = try allocator.dupe(u8, ""),
        .font_family = try allocator.dupe(u8, "default"),
        .size = 14,
        .line_height = 18,
        .generation = renderer.atlas_generation,
    });

    // An overflow resets the atlas. It must NOT drop the entry (the widget still holds the handle),
    // but it must bump the generation so the baked UVs are recognised as stale.
    const gen_before = renderer.atlas_generation;
    renderer.resetAtlas();
    try std.testing.expect(renderer.atlas_generation != gen_before);
    try std.testing.expect(renderer.layout_cache.contains(handle));

    const entry = renderer.layout_cache.getPtr(handle).?;
    try std.testing.expect(entry.generation != renderer.atlas_generation); // now stale

    // Drawing through the handle re-shapes the entry in place and re-stamps it with the live
    // generation, so the holder keeps rendering correctly without ever rebuilding.
    var vertices: std.ArrayList(TextVertex) = .empty;
    defer vertices.deinit(allocator);
    try renderer.appendLayoutGlyphs(allocator, entry, &vertices, 0, 0, 1, 800, 600, .{ 1, 1, 1, 1 });

    try std.testing.expectEqual(renderer.atlas_generation, entry.generation);

    // A display-scale change re-bakes coverage at physical resolution without replacing the handle.
    try renderer.appendLayoutGlyphs(allocator, entry, &vertices, 0, 0, 2, 1600, 1200, .{ 1, 1, 1, 1 });
    try std.testing.expectEqual(@as(f32, 2), entry.raster_scale);
}

test "coverage atlas grows x2 up to max, then falls back to reset" {
    const allocator = std.testing.allocator;
    var renderer = try FreeTypeTextRenderer.initCpu(allocator, &.{}, "default");
    defer renderer.deinit();

    try std.testing.expectEqual(Atlas.initial, renderer.atlas_width);
    try std.testing.expectEqual(@as(usize, Atlas.initial) * Atlas.initial, renderer.atlas_pixels.len);

    // Each growth doubles the store, bumps the generation (stale UVs re-bake lazily), and
    // resets the packer so the triggering glyph lands in the fresh buffer.
    var gen = renderer.atlas_generation;
    while (renderer.atlas_width < Atlas.max) {
        try std.testing.expect(renderer.tryGrowAtlas());
        try std.testing.expect(renderer.atlas_generation != gen);
        gen = renderer.atlas_generation;
        try std.testing.expectEqual(@as(usize, renderer.atlas_width) * renderer.atlas_height, renderer.atlas_pixels.len);
        try std.testing.expectEqual(@as(u32, 1), renderer.pen_x);
    }
    try std.testing.expectEqual(Atlas.max, renderer.atlas_width);
    try std.testing.expectEqual(Atlas.max, renderer.atlas_height);
    try std.testing.expect(!renderer.tryGrowAtlas()); // at max: caller degrades to reset-and-repack
}

test "color emoji atlas is lazy until the first bake" {
    const allocator = std.testing.allocator;
    var renderer = try FreeTypeTextRenderer.initCpu(allocator, &.{}, "default");
    defer renderer.deinit();

    // Registering a family and resetting caches must not materialize the atlas.
    renderer.addEmojiFontFamily("emoji");
    renderer.resetAllTextCaches();
    try std.testing.expect(renderer.color_atlas_pixels == null);
    try std.testing.expect(renderer.color_gpu == null);
    try std.testing.expect(!renderer.color_atlas_dirty);

    // First bake allocates the CPU buffer (GPU half is skipped on the CPU-only path).
    try renderer.ensureColorAtlas();
    try std.testing.expect(renderer.color_atlas_pixels != null);
    try std.testing.expectEqual(@as(usize, ColorAtlas.width) * ColorAtlas.height * 4, renderer.color_atlas_pixels.?.len);
    try std.testing.expect(renderer.color_gpu == null);
}

test "coverage atlas tracks a minimal upload region" {
    const allocator = std.testing.allocator;
    var renderer = try FreeTypeTextRenderer.initCpu(allocator, &.{}, "default");
    defer renderer.deinit();

    renderer.markAtlasClean();
    try std.testing.expect(renderer.atlasDirtyRect() == null);

    renderer.markAtlasDirty(10, 20, 4, 6);
    renderer.markAtlasDirty(8, 25, 3, 2);
    const dirty = renderer.atlasDirtyRect().?;
    try std.testing.expectEqual(@as(u32, 8), dirty.x);
    try std.testing.expectEqual(@as(u32, 20), dirty.y);
    try std.testing.expectEqual(@as(u32, 6), dirty.width);
    try std.testing.expectEqual(@as(u32, 7), dirty.height);
}

test "nextClusterAbove returns the smallest cluster strictly greater" {
    const sorted = [_]u32{ 0, 3, 3, 7, 12 };
    try std.testing.expectEqual(@as(?u32, 3), nextClusterAbove(&sorted, 0));
    try std.testing.expectEqual(@as(?u32, 3), nextClusterAbove(&sorted, 1));
    try std.testing.expectEqual(@as(?u32, 7), nextClusterAbove(&sorted, 3));
    try std.testing.expectEqual(@as(?u32, 12), nextClusterAbove(&sorted, 7));
    try std.testing.expectEqual(@as(?u32, null), nextClusterAbove(&sorted, 12));
    try std.testing.expectEqual(@as(?u32, null), nextClusterAbove(&[_]u32{}, 5));
}

// ── tests ─────────────────────────────────────────────────────────────────────

// The first of these paths that exists, or null. Finding a CJK face this way avoids hardcoding one
// distribution's packaging — the same Noto ships under several names.
fn testExistingPath(candidates: []const []const u8) ?[]const u8 {
    var io_state = std.Io.Threaded.init(std.testing.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    for (candidates) |candidate| {
        std.Io.Dir.accessAbsolute(io, candidate, .{}) catch continue;
        return candidate;
    }
    return null;
}

// Fallback routing is invisible until someone reads a language the bundled font does not cover, so
// it is checked against real faces: Latin must stay on the primary face (a fallback that also
// carries Latin would otherwise silently replace the UI font), and a codepoint the primary cannot
// draw must reach the fallback rather than render as .notdef.
test "routeFor keeps Latin on the primary face and sends CJK to a fallback" {
    const allocator = std.testing.allocator;

    const cjk = testExistingPath(&.{
        "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    }) orelse return error.SkipZigTest; // no CJK font here; nothing to assert against

    var renderer = try FreeTypeTextRenderer.initCpu(allocator, &.{}, "primary");
    defer renderer.deinit();

    const cjk_z = try allocator.dupeZ(u8, cjk);
    defer allocator.free(cjk_z);

    // Relative to the build's working directory, which is this package's root.
    renderer.loadFontFromCPath("primary", "../Zigote.UI/Fonts/Inter/static/Inter_18pt-Regular.ttf") catch
        return error.SkipZigTest;
    try renderer.loadFontFromCPath("cjk", cjk_z.ptr);
    renderer.addFallbackFontFamily("cjk");

    // Latin, Latin-1, Cyrillic and Greek are all Inter's, and must not migrate to a fallback that
    // happens to carry them too.
    try std.testing.expectEqualStrings("primary", renderer.routeFor('A', "primary").family);
    try std.testing.expectEqualStrings("primary", renderer.routeFor(0x00E9, "primary").family);
    try std.testing.expectEqualStrings("primary", renderer.routeFor(0x0416, "primary").family);
    try std.testing.expectEqualStrings("primary", renderer.routeFor(0x03A9, "primary").family);

    // Hiragana, katakana and Han are not.
    try std.testing.expectEqualStrings("cjk", renderer.routeFor(0x3042, "primary").family);
    try std.testing.expectEqualStrings("cjk", renderer.routeFor(0x30A2, "primary").family);
    try std.testing.expectEqualStrings("cjk", renderer.routeFor(0x4E00, "primary").family);

    // Asking twice must agree — the second answer comes from the memo, and a key that collided
    // would hand back whatever was cached for a different codepoint.
    try std.testing.expectEqualStrings("primary", renderer.routeFor('A', "primary").family);
    try std.testing.expectEqualStrings("cjk", renderer.routeFor(0x3042, "primary").family);

    // Mixed text splits at the script boundary and loses nothing across the seam.
    const mixed = "Kid A \u{3042}\u{3042} live";
    var runs = renderer.splitRuns(mixed, "primary");
    var covered: usize = 0;
    var saw_fallback = false;
    while (runs.next()) |run| {
        covered += run.text.len;
        if (std.mem.eql(u8, run.family, "cjk")) saw_fallback = true;
    }
    try std.testing.expectEqual(mixed.len, covered);
    try std.testing.expect(saw_fallback);
}

// The splitter is what keeps paint and measure agreeing, so its boundaries have to be exact.
test "RunSplitter covers its input and keeps marks with their base" {
    const allocator = std.testing.allocator;

    var renderer = try FreeTypeTextRenderer.initCpu(allocator, &.{}, "primary");
    defer renderer.deinit();

    // Nothing registered: everything routes to the primary, so mixed text is a single run and no
    // byte is dropped or repeated.
    const mixed = "abc\u{6F22}\u{5B57}def";
    var runs = renderer.splitRuns(mixed, "primary");
    var total: usize = 0;
    var count: usize = 0;
    while (runs.next()) |run| {
        total += run.text.len;
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expectEqual(mixed.len, total);

    // A combining acute must never start a run of its own.
    var marks = renderer.splitRuns("e\u{0301}", "primary");
    const first = marks.next().?;
    try std.testing.expectEqual("e\u{0301}".len, first.text.len);
    try std.testing.expect(marks.next() == null);

    // Empty input yields nothing rather than an empty run.
    var empty = renderer.splitRuns("", "primary");
    try std.testing.expect(empty.next() == null);

    // A fallback naming a face that was never loaded is ignored, rather than kept and probed on
    // every lookup for the life of the process.
    renderer.addFallbackFontFamily("nonexistent");
    try std.testing.expectEqual(@as(usize, 0), renderer.fallback_families.items.len);
}
