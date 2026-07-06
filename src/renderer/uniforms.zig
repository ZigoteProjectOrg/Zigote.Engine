//! Backend-neutral GPU uniform-buffer (UBO) and pass-parameter structs for the 3D renderer.
//!
//! These are plain `extern struct`s of `[N]f32`/`[N]u32` — pure ABI, no GPU-API types — shared by
//! the wgpu renderer and (eventually) any future backend. Field order is **load-bearing**: the
//! WGSL shaders bind these buffers and several declare a *prefix* of the struct, so fields marked
//! "appended last for offset stability" must stay at the end. The `comptime` offset asserts below
//! pin those invariants so a reorder fails the build instead of silently corrupting a bind group.

const std = @import("std");

// ── Per-pass parameter UBOs ─────────────────────────────────────────────────────

/// Uniform params for the mip-chain bloom downsample pass.
pub const BloomDownParams = extern struct {
    cfg: [4]f32, // x=is_first (1=prefilter+karis), y=threshold, z=knee, w unused
};

/// Uniform params for the tonemap/grade pass.
pub const TonemapParams = extern struct {
    cfg: [4]f32, // x=exposure, y=contrast, z=saturation, w=bloom_intensity
    debug: [4]u32, // x = debug view code (DebugView enum, 0 = off); yzw unused
    // Appended last (UBO-growth pattern): everything below is the photographic look grade.
    look: [4]f32, // x=look(0=Default,1=Punchy,2=Golden), y=vignette_strength, z=vignette_softness, w=grain_amount
    wb: [4]f32, // x=wb_temperature, y=wb_tint, z=chromatic_aberration, w=frame_seed
    lens: [4]f32, // x=lens_distortion_k1, y=lens_distortion_k2, zw unused (appended: UBO-growth pattern)
};

/// Uniform params for the depth-of-field gather pass (gather bokeh on linear HDR, pre-tonemap).
pub const DofParams = extern struct {
    texel: [2]f32, // 1/width, 1/height (full-res)
    cfg: [2]f32, // x=focus_distance (view units, +), y=max_coc (pixels)
    cfg2: [4]f32, // x=coc_scale (ramp rate), y=unused, z=near_blur_boost, w=enabled (0/1)
    bokeh: [4]f32, // x=blades (0/<3 = circular), y=anamorphic (1 = round), zw unused
};

/// Uniform params for the SSAO pass (also drives the folded-in SSGI temporal accumulation).
pub const SsaoParams = extern struct {
    proj: [16]f32, // view → clip
    cfg: [4]f32, // x=radius, y=bias, z=strength, w=power
    contact: [4]f32, // xyz = view-space dir to the sun, w = contact-shadow strength
    contact2: [4]f32, // x=length (view units), y=steps, z=thickness, w=ssgi_strength
    // SSGI temporal reprojection (mirrors the TAA reprojection): blend this frame's GI gather with the
    // previous accumulated GI, reprojected by camera motion.
    prev_view_proj: [16]f32,
    cur_view_proj: [16]f32, // current UNJITTERED — for jitter-free motion vectors
    inv_view: [16]f32,
    gi: [4]f32, // x = history_valid (0/1), y = history feedback weight, zw unused
};

/// Uniform params for the SSR (screen-space reflections) pass.
pub const SsrParams = extern struct {
    proj: [16]f32, // view → clip
    cfg: [4]f32, // x=intensity, y=max_distance, z=thickness, w=step_count
};

/// Uniform params for the TAA resolve pass.
pub const TaaParams = extern struct {
    prev_view_proj: [16]f32,
    cur_view_proj: [16]f32,
    inv_view: [16]f32,
    cfg: [4]f32, // x=feedback, y=enabled, z=history_valid, w unused
};

/// Uniform params for the auto-exposure metering pass (average scene luminance → adapted multiplier,
/// temporally smoothed against the previous frame's value read from a 1×1 history texture).
pub const ExposureParams = extern struct {
    cfg: [4]f32, // x = key value (~0.18), y = min luminance, z = max luminance, w = adaptation speed (per-frame blend)
    cfg2: [4]f32, // x = history_valid (0/1, 0 → snap to target), yzw unused
};

/// Per-cascade shadow matrix, written once per cascade into a dynamic-offset UBO and read by the
/// shadow-pass vertex shader (which renders scene depth from each cascade's light view).
pub const ShadowCascadeParams = extern struct {
    view_proj: [16]f32,
};

/// Per-cube-face params for the omnidirectional point-light shadow pass (dynamic-offset UBO).
pub const PointShadowParams = extern struct {
    view_proj: [16]f32,
    light_pos_range: [4]f32, // xyz = light world position, w = range
};

/// Per-draw bake parameters, written once per cube face/mip into a dynamic-offset UBO.
pub const EnvBakeParams = extern struct {
    face_mode_rough: [4]f32, // x=face, y=mode (0 procedural, 1 hdri), z=roughness
    sky_horizon: [4]f32,
    sky_zenith: [4]f32,
    sky_ground: [4]f32,
    sun_dir: [4]f32,
    studio: [4]f32,
};

// ── Camera / light / model UBOs ─────────────────────────────────────────────────

pub const CameraUniforms = extern struct {
    view_proj: [16]f32,
    view: [16]f32,
    camera_pos: [4]f32,
    // inverse(view_proj) — lets the sky pass reconstruct a world-space view ray per pixel to
    // sample the environment cubemap as the visible backdrop. Appended last so the mesh/shadow
    // shaders (which declare the smaller struct) keep their offsets and bind the same buffer.
    inv_view_proj: [16]f32,
};

pub const LightData = extern struct {
    position_or_dir: [4]f32,
    color_range: [4]f32,
};

/// Extended per-light data (parallel to `LightData`), appended to `LightUniforms` so the base
/// `lights` array keeps its offset. Carries the spot cone + the per-light shadow-map index.
pub const LightExt = extern struct {
    spot_dir: [4]f32, // xyz = spot forward direction (world), w = cos(outer cone half-angle)
    spot_cone: [4]f32, // x = cos(inner cone half-angle), y = shadow layer index (<0 = none), zw unused
};

pub const LightUniforms = extern struct {
    view_proj: [16]f32,
    ambient_color: [4]f32, // rgb ambient tint, .a = ambient/IBL intensity
    light_count: [4]u32,
    // ── Render settings (driven by the editor Settings tab via RenderSettings3D) ──
    sky_horizon: [4]f32, // reflected/ambient sky near horizon (rgb)
    sky_zenith: [4]f32, // reflected/ambient sky overhead (rgb)
    sky_ground: [4]f32, // lower-hemisphere bounce (rgb)
    env_avg: [4]f32, // prefiltered average env for rough surfaces (rgb)
    sun_dir: [4]f32, // studio key reflection dir (xyz), .w = key intensity
    studio: [4]f32, // x=overhead softbox, y=horizon glow, z=key sharpness, w=clearcoat
    post: [4]f32, // x=exposure (legacy, unused here), y=contrast, z=saturation, w=env max LOD
    shadow: [4]f32, // x=shadow strength, y=bias, z=softness, w unused
    lights: [16]LightData,
    // x = debug view code (DebugView enum, 0 = off), y = diagnostic mode flag (0/1),
    // zw unused. Appended last so the sky/shadow shaders (which omit this field) keep their
    // existing offsets and still bind the same, larger buffer.
    debug: [4]u32,
    // Reflection-probe box: xyz = world centre, w = enabled (0/1); extents xyz = half-size.
    // Appended after `debug` for the same offset-stability reason. Only the mesh shader reads these.
    probe_center: [4]f32,
    probe_extents: [4]f32,
    // ── Cascaded shadow maps (appended last for offset stability; only the mesh shader reads these) ──
    // Four cascade light view-proj matrices for directional CSM. The fragment shader picks the
    // tightest cascade whose projection contains the surface, then samples that depth-array layer.
    csm_view_proj: [4][16]f32,
    // x = active cascade count, y = shadow-map resolution (drives the PCF texel radius), zw unused.
    csm_info: [4]f32,
    // ── Per-light extended data + spot shadow matrices (appended last; only the mesh shader reads) ──
    light_ext: [16]LightExt,
    // Perspective light view-proj per shadow-casting spot light, indexed by (shadow layer - cascade count).
    // Array length must match wgpu_3d.MAX_SPOT_SHADOWS.
    spot_view_proj: [4][16]f32,
    // ── Atmospheric fog (appended last for offset stability; only the mesh shader reads these) ──
    // Height-based exponential fog + analytic sun in-scatter (aerial perspective + god-ray glow).
    fog_color: [4]f32, // rgb fog/in-scatter base colour, .w = density (0 = fog off)
    fog_params: [4]f32, // x = height falloff, y = height base (world Y), z = sun in-scatter strength, w = anisotropy g
};

pub const ModelUniforms = extern struct {
    model: [16]f32,
    // transpose(inverse(model)) — the normal matrix. Stored as a full mat4 for std140 alignment;
    // only the upper-left 3×3 is used. Required so normals/tangents transform correctly under
    // non-uniform scale (using `model` directly skews them → distorted "clay" highlights).
    normal_mat: [16]f32,
    base_color: [4]f32,
    metallic_roughness: [4]f32,
    emissive: [4]f32,
    effect: [4]u32,
    // Extended surface params: x = IOR (index of refraction, 1.5 default), y = transmission
    // factor (0..1; glass path blends the lit surface toward the transmissive response),
    // z = occlusion strength (>0 → the MR map's R channel is glTF ORM occlusion), w = reserved.
    surface: [4]f32,
};

// ── Layout guards — pin the offset-stability invariants the shaders depend on ────
comptime {
    // CameraUniforms: shadow/mesh shaders bind a prefix ending before inv_view_proj.
    std.debug.assert(@offsetOf(CameraUniforms, "inv_view_proj") == 144);
    std.debug.assert(@sizeOf(CameraUniforms) == 208);
    // LightUniforms: sky/shadow shaders bind a prefix ending before `debug`; the mesh shader
    // additionally reads the probe fields at the very end.
    std.debug.assert(@offsetOf(LightUniforms, "lights") == 224);
    std.debug.assert(@offsetOf(LightUniforms, "debug") == 736);
    std.debug.assert(@offsetOf(LightUniforms, "probe_center") == 752);
    std.debug.assert(@offsetOf(LightUniforms, "csm_view_proj") == 784);
    std.debug.assert(@offsetOf(LightUniforms, "csm_info") == 1040);
    std.debug.assert(@offsetOf(LightUniforms, "light_ext") == 1056);
    std.debug.assert(@offsetOf(LightUniforms, "spot_view_proj") == 1568);
    std.debug.assert(@offsetOf(LightUniforms, "fog_color") == 1824);
    std.debug.assert(@offsetOf(LightUniforms, "fog_params") == 1840);
    std.debug.assert(@sizeOf(LightUniforms) == 1856);
    // ModelUniforms / TonemapParams: pinned so a field insert can't silently shift the rest.
    std.debug.assert(@offsetOf(ModelUniforms, "surface") == 192);
    std.debug.assert(@sizeOf(ModelUniforms) == 208);
    std.debug.assert(@sizeOf(TonemapParams) == 80);
}
