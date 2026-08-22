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

/// Glyph quad for the glyph_run paint op.
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


// The flat 112-byte ZgPaintCommand that every kind shared lived here. It is gone: see the
// "Paint command stream" section below for the tagged records that replaced it, and why.

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
    paint_op_header_size: u32, // offset  4  — must equal sizeof(ZgPaintOpHeader) on C# side
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


/// The one status code every fallible export returns.
///
/// Before this there were five conventions at once: 178 `void` exports that swallowed failure
/// silently, `u64`/`u32` sentinels where 0 meant "failed", `bool`, `u8` standing in for bool, and
/// `i32` ladders in the dialog code — with `ZgResult` (ok/err) used by 8 of ~290. A caller could
/// not tell "not ready yet" from "you passed a dead handle" from "the GPU refused", and most of the
/// time could not tell anything happened at all.
///
/// `ok` is 0 and `err` is -1, so the old two-value `ZgResult` encoding is a subset: a host that
/// only checks `!= 0` keeps working. Everything else is negative and more specific.
pub const ZgStatus = enum(i32) {
    ok = 0,
    /// Unspecified failure. Prefer a specific code; this is for genuinely unclassifiable errors.
    err = -1,
    /// The engine, or a resource handle, is not live — never created, already destroyed, or from a
    /// previous occupant of a reused slot.
    invalid_handle = -2,
    /// A parameter was out of range, null where it may not be, or otherwise malformed.
    invalid_argument = -3,
    out_of_memory = -4,
    /// The platform or the active backend cannot do this.
    unsupported = -5,
    /// The subsystem exists but is not initialised yet (audio device not opened, 3D not created).
    not_ready = -6,
    /// The GPU rejected the work, or a device resource could not be created.
    gpu_failure = -7,
    /// A file or stream operation failed.
    io_failure = -8,

    pub fn isOk(self: ZgStatus) bool {
        return self == .ok;
    }
};

/// Retained name for the previous two-value result. Identical encoding; `ZgStatus` is the contract.
pub const ZgResult = ZgStatus;


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

// ── Scene command stream ──────────────────────────────────────────────────────
//
// The scene used to be driven by fourteen per-node setters — transform, light, camera, colour,
// roughness, surface, emissive, effect, alpha mode, double-sided, volume, occlusion, visibility,
// primitive — each its own export and its own P/Invoke. The host dirty-gates them, so a static
// node costs nothing per frame, but a scene load or rebuild pays three to six transitions per node
// and the surface is fourteen functions wide for what is really one idea: "apply these changes".
//
// `zigote_scene_apply` takes a flat byte stream of records instead. Every record starts with
// `ZgSceneOpHeader`; `size` is the WHOLE record including the header, so a decoder that meets an
// op it does not know can skip it rather than lose sync — which is what lets the stream grow
// without another ABI break. See docs/v2-design.md §2.3.

pub const ZgSceneOp = enum(u32) {
    transform = 1,
    light = 2,
    camera = 3,
    material = 4,
    visibility = 5,
    primitive = 6,
};

pub const ZgSceneOpHeader = extern struct {
    /// A `ZgSceneOp`. Unknown values are skipped using `size`.
    kind: u32,
    /// Total bytes of this record, header included. Always a multiple of 8.
    size: u32,
};

pub const ZgSceneTransform = extern struct {
    header: ZgSceneOpHeader,
    node: u64,
    x: f32,
    y: f32,
    z: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    sx: f32,
    sy: f32,
    sz: f32,
    pad: f32 = 0,
};

pub const ZgSceneLight = extern struct {
    header: ZgSceneOpHeader,
    node: u64,
    kind: u32,
    cast_shadows: u32,
    r: f32,
    g: f32,
    b: f32,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    pad: f32 = 0,
};

pub const ZgSceneCamera = extern struct {
    header: ZgSceneOpHeader,
    node: u64,
    fovy_degrees: f32,
    near: f32,
    far: f32,
    pad: f32 = 0,
};

/// The whole PBR factor set in one record — the eight material setters it replaces were each a
/// separate call that ended up mutating adjacent fields of the same material.
pub const ZgSceneMaterial = extern struct {
    header: ZgSceneOpHeader,
    node: u64,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    metallic: f32,
    roughness: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    specular: f32,
    emissive_r: f32,
    emissive_g: f32,
    emissive_b: f32,
    ior: f32,
    transmission: f32,
    occlusion_strength: f32,
    alpha_cutoff: f32,
    effect: u32,
    alpha_mode: u32,
    double_sided: u32,
    pad: u32 = 0,
};

pub const ZgSceneVisibility = extern struct {
    header: ZgSceneOpHeader,
    node: u64,
    visible: u32,
    pad: u32 = 0,
};

pub const ZgScenePrimitive = extern struct {
    header: ZgSceneOpHeader,
    node: u64,
    prim_type: u32,
    pad: u32 = 0,
};

comptime {
    // Every record must be 8-byte aligned and a multiple of 8 bytes, so a stream of them can be
    // walked by adding `size` without ever landing mid-field.
    for ([_]type{
        ZgSceneTransform, ZgSceneLight,      ZgSceneCamera,
        ZgSceneMaterial,  ZgSceneVisibility, ZgScenePrimitive,
    }) |T| {
        if (@sizeOf(T) % 8 != 0) @compileError("scene op " ++ @typeName(T) ++ " must be a multiple of 8 bytes");
        if (@alignOf(T) != 8) @compileError("scene op " ++ @typeName(T) ++ " must be 8-byte aligned");
        if (@offsetOf(T, "header") != 0) @compileError("scene op " ++ @typeName(T) ++ " must start with its header");
    }
}

// ── Paint command stream ──────────────────────────────────────────────────────
//
// `ZgPaintCommand` is one flat 112-byte struct shared by all 20 command kinds, which made it a
// union in all but name. `radius` is also an image's u0 AND a shader id (`@bitCast` from f32).
// `img_pixel_w` is also a text shadow's blur radius. `text_len` is also a GLYPH_RUN's quad COUNT.
// A text shadow packs its colour into `rect_x/y/w/h` — the rectangle fields — while its geometry
// lives in `baseline_x/y`. Every one of those is documented, and every one is a place where
// reading the wrong field yields a plausible number instead of an error.
//
// The V2 stream is tagged and variable-size: `ZgPaintOpHeader` then a struct that names its own
// fields. A rect is 48 bytes rather than 112, and nothing aliases. `size` covers the whole record,
// so an unknown op is skipped rather than desynchronising the stream. See docs/v2-design.md §2.1.
//
// Blobs (text bytes, font family, pixel data, polygon points, glyph quads) stay as pointer+len
// rather than moving into a side buffer as §2.1 proposed. The host already memoises UTF-8 text on
// the pinned object heap and passes stable pointers, so today those cross at zero copy; a side
// buffer would copy every string every frame. The benchmark in §11 puts C# list building at ~26%
// of a paint frame, which is the wrong number to add copying to.

pub const ZgPaintOp = enum(u32) {
    rect = 0,
    border = 1,
    text = 2,
    image = 3,
    clip_start = 4,
    clip_end = 5,
    push_opacity = 6,
    pop_opacity = 7,
    shadow = 8,
    liquid_glass = 9,
    shader_effect = 10,
    text_layout = 11,
    glyph_run = 12,
    render_texture_begin = 13,
    render_texture_end = 14,
    blur = 15,
    bezier = 16,
    polygon = 17,
    transform_push = 18,
    transform_pop = 19,
};

pub const ZgPaintOpHeader = extern struct {
    /// A `ZgPaintOp`. Unknown values are skipped using `size`.
    kind: u32,
    /// Total bytes of this record, header included. Always a multiple of 8.
    size: u32,
};

/// Straight (non-premultiplied) RGBA, 0..1 — the same range the flat struct used.
pub const ZgRgba = extern struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

pub const ZgXywh = extern struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
};

pub const ZgPaintRect = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    color: ZgRgba,
    radius: f32,
    pad: f32 = 0,
};

pub const ZgPaintBorder = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    color: ZgRgba,
    radius: f32,
    width: f32,
};

pub const ZgPaintShadow = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    color: ZgRgba,
    radius: f32,
    blur_radius: f32,
    spread: f32,
    pad: f32 = 0,
};

pub const ZgPaintLiquidGlass = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    color: ZgRgba,
    radius: f32,
    thickness: f32,
    glow_x: f32,
    glow_y: f32,
    pinch: f32,
    adapt: f32,
};

/// One text run. `is_shadow` selects the drop-shadow variant, which used to be signalled by
/// stuffing the colour into the rect fields and the blur into `img_pixel_w`.
pub const ZgPaintText = extern struct {
    header: ZgPaintOpHeader,
    text_ptr: [*c]const u8,
    text_len: u32,
    family_len: u32,
    family_ptr: [*c]const u8,
    color: ZgRgba,
    baseline_x: f32,
    baseline_y: f32,
    font_size: f32,
    line_height: f32,
    letter_spacing: f32,
    word_spacing: f32,
    font_weight: u32,
    /// 0 = normal, 1 = italic.
    font_style: u32,
    /// Non-zero draws the run as a drop shadow, offset by (shadow_dx, shadow_dy).
    is_shadow: u32,
    shadow_blur: f32,
    shadow_dx: f32,
    shadow_dy: f32,
};

pub const ZgPaintImage = extern struct {
    header: ZgPaintOpHeader,
    pixels_ptr: [*c]const u8,
    pixels_len: u32,
    pixel_w: u32,
    pixel_h: u32,
    has_cache_key: u32,
    cache_key: u64,
    bounds: ZgXywh,
    tint: ZgRgba,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
};

pub const ZgPaintClipStart = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    radius: f32,
    pad: f32 = 0,
};

/// clip_end, pop_opacity, render_texture_end and transform_pop carry nothing but their header.
pub const ZgPaintBare = extern struct {
    header: ZgPaintOpHeader,
};

pub const ZgPaintPushOpacity = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    alpha: f32,
    pad: f32 = 0,
};

pub const ZgPaintShaderEffect = extern struct {
    header: ZgPaintOpHeader,
    bounds: ZgXywh,
    /// A real u32 id, not a float reinterpreted through @bitCast.
    shader_id: u32,
    has_cache_key: u32,
    cache_key: u64,
    chains_backdrop: u32,
    pad: u32 = 0,
    params: [8]f32,
};

pub const ZgPaintTextLayout = extern struct {
    header: ZgPaintOpHeader,
    layout: u64,
    color: ZgRgba,
    draw_x: f32,
    draw_y: f32,
};

pub const ZgPaintGlyphRun = extern struct {
    header: ZgPaintOpHeader,
    /// Points at `quad_count` ZgGlyphRunQuad. Was `text_ptr`, with the count in `text_len`.
    quads_ptr: [*c]const u8,
    quad_count: u32,
    pad: u32 = 0,
    atlas: u64,
    color: ZgRgba,
};

pub const ZgPaintRenderTextureBegin = extern struct {
    header: ZgPaintOpHeader,
    rt_handle: u64,
    bounds: ZgXywh,
};

pub const ZgPaintBlur = extern struct {
    header: ZgPaintOpHeader,
    src_handle: u64,
    sigma: f32,
    pad: f32 = 0,
};

pub const ZgPaintBezier = extern struct {
    header: ZgPaintOpHeader,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    color: ZgRgba,
    width: f32,
    pad: f32 = 0,
};

pub const ZgPaintPolygon = extern struct {
    header: ZgPaintOpHeader,
    points_ptr: [*c]const u8,
    points_len: u32,
    pad: u32 = 0,
    color: ZgRgba,
};

pub const ZgPaintTransformPush = extern struct {
    header: ZgPaintOpHeader,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    tx: f32,
    ty: f32,
};

comptime {
    for ([_]type{
        ZgPaintRect,          ZgPaintBorder,             ZgPaintShadow,   ZgPaintLiquidGlass,
        ZgPaintText,          ZgPaintImage,              ZgPaintClipStart, ZgPaintBare,
        ZgPaintPushOpacity,   ZgPaintShaderEffect,       ZgPaintTextLayout, ZgPaintGlyphRun,
        ZgPaintRenderTextureBegin, ZgPaintBlur,          ZgPaintBezier,   ZgPaintPolygon,
        ZgPaintTransformPush,
    }) |T| {
        if (@sizeOf(T) % 8 != 0) @compileError("paint op " ++ @typeName(T) ++ " must be a multiple of 8 bytes");
        if (@offsetOf(T, "header") != 0) @compileError("paint op " ++ @typeName(T) ++ " must start with its header");
    }
    // The whole point: the common case got smaller.
    if (@sizeOf(ZgPaintRect) >= 112) @compileError("ZgPaintRect should be far smaller than the old flat command");
}
