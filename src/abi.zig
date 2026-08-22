//! The FFI wire contract: every `extern struct` and enum that crosses the C ABI.
//!
//! Collected here, out of the 7k-line ffi/root.zig, for two reasons. It is the contract, so it
//! should be readable on its own. And it is the single input to `zig build ffi-manifest`, which
//! reflects over these types and emits their real sizes and field offsets — the C# side asserts
//! against that manifest instead of against hand-copied literals.
//!
//! That mattered: the C# mirrors in Zigote.Core/Native/ZgStructs.cs and the offsets pinned in
//! AbiLayoutTests were both written by hand, so a Zig field reorder that preserved total size
//! would pass the C# tests AND the startup size check in RendererAbiInfo.Validate, and silently
//! misread every field past the change. The comptime asserts below catch it on the Zig side; the
//! manifest is what carries the same truth across to C#.
//!
//! Imports nothing but std, by design — see docs/v2-design.md §2.1 and §4.

const std = @import("std");

/// Glyph quad for CMD_GLYPH_RUN — matches C# ZgGlyphQuad (32 bytes).
pub const ZgGlyphRunQuad = extern struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
};


/// Flat C-ABI paint command. Layout must match ZgStructs.cs ZgPaintCommand.
/// Total size: 112 bytes on 64-bit.
/// Fields are ordered large→small (8-byte pointers first, then f32/u32, then the small ints) so the
/// struct packs with a single 3-byte hole instead of the ~11 padding bytes the natural declaration
/// order used to force — 120→112 B, ~8 B saved on every one of the hundreds–thousands of commands a
/// painted frame streams to fillPaintList. All fields keep their meaning; Zig reads them by name, so
/// the reorder is transparent to the renderer. The comptime block below pins every offset — a drift
/// on either side of the ABI now fails the Zig build (C# side is pinned by AbiLayoutTests).
pub const ZgPaintCommand = extern struct {
    kind: u8, // offset   0
    font_style: u8, // offset   1  (0=normal, 1=italic)
    font_weight: u16, // offset   2  (100..900)
    has_cache_key: u8, // offset   4
    /// CMD_SHADER_EFFECT only: this effect is a FILTER and must see the previous effect's
    /// output, so the backdrop capture is refreshed before it. Off = the effect shares the
    /// capture with its neighbours, which is what Liquid Glass and backdrop blur want.
    chains_backdrop: u8, // offset   5
    pad0: [2]u8 = .{0} ** 2,
    text_ptr: [*c]const u8, // offset  8  (also the GlyphRunQuad array pointer for CMD_GLYPH_RUN)
    pixels_ptr: [*c]const u8, // offset 16  (image pixels / font-family bytes / polygon points)
    rect_x: f32, // offset  24
    rect_y: f32, // offset  28
    rect_w: f32, // offset  32
    rect_h: f32, // offset  36
    color_r: f32, // offset  40
    color_g: f32, // offset  44
    color_b: f32, // offset  48
    color_a: f32, // offset  52
    radius: f32, // offset  56  (aliased: image u0 / shader id via @bitCast)
    border_width: f32, // offset  60  (image v0)
    baseline_x: f32, // offset  64  (image u1)
    baseline_y: f32, // offset  68  (image v1)
    font_size: f32, // offset  72
    line_height: f32, // offset  76
    letter_spacing: f32, // offset  80
    word_spacing: f32, // offset  84
    img_pixel_w: u32, // offset  88
    img_pixel_h: u32, // offset  92
    cache_key_lo: u32, // offset  96
    cache_key_hi: u32, // offset 100
    text_len: u32, // offset 104
    pixels_len: u32, // offset 108
    // total: 112 bytes
};

comptime {
    std.debug.assert(@sizeOf(ZgPaintCommand) == 112);
    std.debug.assert(@offsetOf(ZgPaintCommand, "kind") == 0);
    std.debug.assert(@offsetOf(ZgPaintCommand, "font_style") == 1);
    std.debug.assert(@offsetOf(ZgPaintCommand, "font_weight") == 2);
    std.debug.assert(@offsetOf(ZgPaintCommand, "has_cache_key") == 4);
    std.debug.assert(@offsetOf(ZgPaintCommand, "chains_backdrop") == 5);
    std.debug.assert(@offsetOf(ZgPaintCommand, "text_ptr") == 8);
    std.debug.assert(@offsetOf(ZgPaintCommand, "pixels_ptr") == 16);
    std.debug.assert(@offsetOf(ZgPaintCommand, "rect_x") == 24);
    std.debug.assert(@offsetOf(ZgPaintCommand, "color_r") == 40);
    std.debug.assert(@offsetOf(ZgPaintCommand, "radius") == 56);
    std.debug.assert(@offsetOf(ZgPaintCommand, "border_width") == 60);
    std.debug.assert(@offsetOf(ZgPaintCommand, "baseline_x") == 64);
    std.debug.assert(@offsetOf(ZgPaintCommand, "baseline_y") == 68);
    std.debug.assert(@offsetOf(ZgPaintCommand, "font_size") == 72);
    std.debug.assert(@offsetOf(ZgPaintCommand, "line_height") == 76);
    std.debug.assert(@offsetOf(ZgPaintCommand, "letter_spacing") == 80);
    std.debug.assert(@offsetOf(ZgPaintCommand, "word_spacing") == 84);
    std.debug.assert(@offsetOf(ZgPaintCommand, "img_pixel_w") == 88);
    std.debug.assert(@offsetOf(ZgPaintCommand, "img_pixel_h") == 92);
    std.debug.assert(@offsetOf(ZgPaintCommand, "cache_key_lo") == 96);
    std.debug.assert(@offsetOf(ZgPaintCommand, "cache_key_hi") == 100);
    std.debug.assert(@offsetOf(ZgPaintCommand, "text_len") == 104);
    std.debug.assert(@offsetOf(ZgPaintCommand, "pixels_len") == 108);
}


/// Flat C-ABI input event. Layout must match ZgStructs.cs ZgEvent.
/// Total size: 44 bytes. The text_input / text_editing UTF-8 payload is stored OUT OF BAND: it is
/// appended to the engine's per-poll `poll_text` buffer and the event carries only (text_off, text_len)
/// into it — so the common flood of mouse/key events (which have no text) costs 44 B, not 288 B. The
/// out-of-band buffer is unbounded, so IME pre-edit is never truncated. C# reads it via
/// zigote_poll_text_ptr right after polling (valid until the next poll; single-threaded drain-decode).
pub const ZgEvent = extern struct {
    kind: u8, // offset  0
    button: u8, // offset  1  (mouse button; for key events: 1 = OS auto-repeat)
    modifiers: u8, // offset  2
    key_char: u8, // offset  3  (ASCII, 0 if not a printable key)
    key_scancode: u32, // offset  4  (raw SDL scancode)
    x: f32, // offset  8
    y: f32, // offset 12
    scroll_x: f32, // offset 16
    scroll_y: f32, // offset 20
    resize_w: u32, // offset 24  (text_editing: IME composition start)
    resize_h: u32, // offset 28  (text_editing: IME composition length)
    text_off: u32, // offset 32  (text_input / text_editing: byte offset into poll_text)
    text_len: u32, // offset 36  (text_input / text_editing: byte length in poll_text)
    window_id: u32, // offset 40  (SDL window id; 0 = unknown → treated as the main window)
    // total: 44 bytes
};


/// Result of zigote_measure_text.
pub const ZgSize = extern struct {
    width: f32,
    height: f32,
};


/// ABI compatibility info returned by zigote_get_renderer_abi_info().
/// C# must call this at startup and verify sizes match its compile-time @sizeOf values.
pub const ZgAbiInfo = extern struct {
    abi_version: u32, // offset  0  — bump when breaking ABI changes occur
    paint_command_size: u32, // offset  4  — must equal sizeof(ZgPaintCommand) on C# side
    event_size: u32, // offset  8  — must equal sizeof(ZgEvent) on C# side
    handle_size: u32, // offset 12  — size of an opaque resource handle (usize)
    render_settings_3d_size: u32, // offset 16  — must equal sizeof(ZgRenderSettings3D) on C# side
    // total: 20 bytes
};


/// Runtime renderer capabilities returned by zigote_get_renderer_caps() AFTER init.
/// Reports the backend actually selected plus optional native features (vendor upscalers /
/// hardware ray tracing). Kept separate from ZgAbiInfo (whose 16-byte size is a fixed
/// compile-time ABI guard). Mirrors backend.Caps and C# ZgRendererCaps. Total: 12 bytes.
pub const ZgRendererCaps = extern struct {
    active_backend: u32, // offset 0 — BackendId actually in use (auto may fall back)
    upscalers: u32, // offset 4 — bitset of backend.UpscalerKind (0 = none)
    raytracing: u8, // offset 8 — hardware ray tracing available
    raytracing_from_render: u8, // offset 9 — RT usable from fragment shaders (not only compute)
    pad: [2]u8 = .{0} ** 2, // offset 10 — pad to 12 bytes
    // total: 12 bytes
};


/// Typed result code returned by fallible FFI functions.
/// Replaces raw i32 0/-1 returns so the C# side has a typed enum to check.
pub const ZgResult = enum(i32) {
    ok = 0,
    err = -1,
};


/// One entry for the parallel batch texture loader. `base_color_path` / `mr_path` / `normal_path`
/// / `emissive_path` are optional (null = skip that map). Layout must match ZgTextureLoadItem in C#.
pub const ZgTextureLoadItem = extern struct {
    node_handle: u64,
    base_color_path: [*c]const u8,
    mr_path: [*c]const u8,
    normal_path: [*c]const u8,
    emissive_path: [*c]const u8,
};


/// Flat C-ABI mirror of wgpu_3d.Settings3D. Layout must match ZgRenderSettings3D in C#.
/// All fields f32 (70 floats); colours are linear rgb, sun angles in degrees.
pub const ZgRenderSettings3D = extern struct {
    ambient_intensity: f32,
    sky_horizon_r: f32,
    sky_horizon_g: f32,
    sky_horizon_b: f32,
    sky_zenith_r: f32,
    sky_zenith_g: f32,
    sky_zenith_b: f32,
    sky_ground_r: f32,
    sky_ground_g: f32,
    sky_ground_b: f32,
    env_avg_r: f32,
    env_avg_g: f32,
    env_avg_b: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    overhead: f32,
    horizon_glow: f32,
    sun_sharpness: f32,
    exposure: f32,
    contrast: f32,
    saturation: f32,
    shadow_strength: f32,
    shadow_bias: f32,
    shadow_softness: f32,
    clearcoat: f32,
    bloom_threshold: f32,
    bloom_knee: f32,
    bloom_intensity: f32,
    ssao_radius: f32,
    ssao_bias: f32,
    ssao_strength: f32,
    ssao_power: f32,
    ssr_intensity: f32,
    ssr_max_distance: f32,
    ssr_thickness: f32,
    ssr_steps: f32,
    taa_enabled: f32,
    taa_feedback: f32,
    diagnostic_mode: f32,
    debug_view: f32,
    // Depth of field: enable + fallback focus distance + aperture + max blur radius.
    dof_enabled: f32,
    dof_focus_distance: f32,
    dof_f_stop: f32,
    dof_max_coc: f32,
    // Wireframe render debug mode (0/1): draw all geometry as flat line edges.
    wireframe: f32,
    // Atmospheric fog: density (0 = off), colour rgb, height base, height falloff, sun in-scatter, anisotropy g.
    fog_density: f32,
    fog_color_r: f32,
    fog_color_g: f32,
    fog_color_b: f32,
    fog_height: f32,
    fog_height_falloff: f32,
    fog_sun_inscatter: f32,
    fog_anisotropy: f32,
    // Auto-exposure: enabled (0/1), key value, min/max metered luminance, adaptation speed.
    auto_exposure_enabled: f32,
    auto_exposure_key: f32,
    auto_exposure_min: f32,
    auto_exposure_max: f32,
    auto_exposure_speed: f32,
    // Photographic grade (post-AgX look). Exposed so film-stock emulation + the physical camera can drive
    // them (previously baked as Settings3D defaults). Consumed by the tonemap shader.
    agx_look: f32,
    wb_temperature: f32,
    wb_tint: f32,
    vignette_strength: f32,
    vignette_softness: f32,
    grain_amount: f32,
    chromatic_aberration: f32,
    // Lens optics (physical-camera native effects). Radial distortion applied as a UV remap in tonemap.
    lens_distortion_k1: f32,
    lens_distortion_k2: f32,
    // Aperture bokeh shape (extends the DoF gather): blade count (0/<3 = circular) + anamorphic squeeze.
    bokeh_blades: f32,
    bokeh_anamorphic: f32,
};

comptime {
    // ABI guard: this struct is passed BY VALUE to C# (ZgRenderSettings3D in ZgStructs.cs) with no
    // per-field check, so a field inserted on one side silently shifts every downstream setting. Pin
    // the size here and report it through ZgAbiInfo (abi_version 7) so a mismatch fails loudly at
    // startup instead of corrupting the render settings. Keep 70 f32 in lockstep with the C# struct.
    if (@sizeOf(ZgRenderSettings3D) != 70 * @sizeOf(f32))
        @compileError("ZgRenderSettings3D must be 70 f32 (280 bytes) — keep field count in sync with C#");
}


/// Per-frame engine statistics for the debug overlay/profiler (design doc §14.1). Cheap snapshot —
/// no allocation, no GPU stall. Mirrors ZgEngineStats in C#.
pub const ZgEngineStats = extern struct {
    frame_index: u64,
    draw_calls: u32,
    triangles: u32,
    render_passes: u32,
    visible_objects: u32,
    gpu_buffer_memory: u64,
    gpu_texture_memory: u64,
};
