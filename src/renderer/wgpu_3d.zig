/// 3D render pass.
///
/// Renders a World's scene graph into the surface before the 2D UI pass.
/// Architecture:
///   - One shared depth texture per resolution
///   - One mesh pipeline with optional texture sampling and CRT effect support
///   - Camera UBO updated once per frame
///   - Per-object model matrix uploaded as a small uniform buffer
///   - Mesh GPU buffers cached by MeshHandle in a HashMap

const std = @import("std");
const wgpu = @import("wgpu");
const zpool = @import("zpool");
const zmo = @import("zmesh_opt");
const engine = @import("zigote_engine");
const math = engine.math;
const scene_mod = engine.scene;
const resources_mod = engine.resources;
const shaders3d = @import("wgpu_3d_shaders.zig");
const uniforms = @import("uniforms.zig");
const particles_mod = @import("wgpu_particles.zig");
const sprites_mod = @import("wgpu_sprites.zig");
const MeshCache = @import("mesh_cache.zig").MeshCache;

const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;

/// MSAA sample count for the 3D scene pass (sky + meshes). The shadow pass stays 1x.
// Geometry-pass MSAA sample count. Defaults to 4 — the only count the WebGPU spec guarantees for
// EVERY renderable format (rgba16float in particular). `root.zig` lowers it to 2 at device init on
// Metal adapters that grant TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES (2× rgba16float MSAA is
// adapter-specific, not spec-baseline — Apple Silicon supports it; Mesa's llvmpipe grants the same
// feature but supports [1, 4, 8], and asking it for 2 aborts the process in pipeline validation).
// 2× halves every multisampled target (HDR color, depth, and the 3 MSAA G-buffers); TAA runs on top
// and reclaims most of the edge antialiasing.
// Written exactly once, before the first Gpu3d is created; read-only thereafter (pipelines + targets
// both read this same value, so they always agree on the sample count).
pub var MSAA_SAMPLES: u32 = 4;

/// The 3D scene renders into a linear HDR buffer of this format; a post-processing pass
/// (bloom + ACES tonemap) converts it to the LDR surface format the UI samples.
const SCENE_HDR_FORMAT: wgpu.TextureFormat = .rgba16_float;

// Per-pass UBO + camera/light/model uniform structs live in the backend-neutral uniforms.zig;
// re-exported here so the rest of this file refers to them by their short names.
const BloomDownParams = uniforms.BloomDownParams;
const TonemapParams = uniforms.TonemapParams;
const DofParams = uniforms.DofParams;
const SsaoParams = uniforms.SsaoParams;
const SsrParams = uniforms.SsrParams;
const TaaParams = uniforms.TaaParams;
const ExposureParams = uniforms.ExposureParams;
const EnvBakeParams = uniforms.EnvBakeParams;
const CameraUniforms = uniforms.CameraUniforms;
const LightData = uniforms.LightData;
const LightUniforms = uniforms.LightUniforms;
const ModelUniforms = uniforms.ModelUniforms;

/// Halton low-discrepancy sequence — sub-pixel jitter offsets for TAA, in [0,1).
fn halton(index: u32, base: u32) f32 {
    var f: f32 = 1.0;
    var r: f32 = 0.0;
    var i = index + 1;
    const bf: f32 = @floatFromInt(base);
    while (i > 0) {
        f /= bf;
        r += f * @as(f32, @floatFromInt(i % base));
        i /= base;
    }
    return r;
}

/// Prefiltered environment cubemap for image-based lighting. Each mip is a rougher reflection
/// level; the mesh shader selects a mip by surface roughness. HDR (rgba16float) so bright
/// studio highlights / HDRI hotspots survive into reflections.
// Mip-0 reflection resolution (128 was too soft for chrome mirrors). A var, not a const:
// root.zig shrinks it at device init on SOFTWARE Vulkan adapters (the Android emulator's only
// Vulkan is SwiftShader), where the 512²×6-face×128-sample GGX bake takes long enough that the
// emulator's fence watchdog declares the device lost. Written once, before any Gpu3d exists.
pub var ENV_SIZE: u32 = 512;
const ENV_MIPS: u32 = 6; // 128 → 4 px; max LOD index = ENV_MIPS - 1
const BLOOM_MIPS: u32 = 6; // mip-chain bloom levels (mip0 full-res, unused; chain output at mip1)
const ENV_CUBE_FORMAT: wgpu.TextureFormat = .rgba16_float;
/// Uniform offset alignment for the per-face/mip bake params (wgpu min is 256).
const ENV_SLOT: u32 = 256;

/// Per-cascade directional shadow map resolution. The mesh shader reads the resolution from
/// `light.csm_info.y` to size its PCF texel radius, so this is no longer hard-coded shader-side.
const SHADOW_MAP_SIZE: u32 = 2048;
/// Number of directional shadow cascades. The shadow map is a depth-2d-array with this many layers;
/// the mesh shader picks the tightest cascade that contains each surface.
const NUM_CASCADES: u32 = 4;
/// Per-cascade orthographic half-extent (world units). Concentric, increasing boxes centred ahead
/// of the camera: cascade 0 is tight (crisp near shadows), higher indices extend the range. At
/// extent 20 the geometry reproduces the former single-box directional shadow exactly.
const CASCADE_EXTENTS = [NUM_CASCADES]f32{ 7.0, 18.0, 45.0, 110.0 };
/// Maximum number of shadow-casting spot lights per frame (must match `uniforms.spot_view_proj` len).
const MAX_SPOT_SHADOWS: u32 = 4;
/// Total layers in the shadow depth-2d-array: directional cascades + per-spot perspective maps.
const TOTAL_SHADOW_LAYERS: u32 = NUM_CASCADES + MAX_SPOT_SHADOWS;
/// Uniform offset alignment for the per-slice shadow matrix UBO (wgpu min is 256).
const SHADOW_SLOT: u32 = 256;
/// Omnidirectional point-light shadows: a depth cube-array. Each shadow-casting point light claims one
/// cube (6 faces); the mesh shader samples it by direction with a hardware depth comparison.
const MAX_POINT_SHADOWS: u32 = if (shaders3d.cube_array_supported) 2 else 1;
/// Sampled view over the point-shadow faces: a cube array normally, a single cube on the one
/// target without cube-array support (see shaders3d.cube_array_supported).
const POINT_SHADOW_VIEW_DIM: wgpu.ViewDimension =
    if (shaders3d.cube_array_supported) .cube_array else .cube;
const POINT_SHADOW_SIZE: u32 = 1024; // per cube-face resolution
const POINT_CUBE_LAYERS: u32 = 6 * MAX_POINT_SHADOWS;
/// Cube-face look directions + up vectors (WebGPU/D3D cube convention: +X,-X,+Y,-Y,+Z,-Z). Used to
/// build the 6 per-face view matrices for point-light shadows so a direction sample hits the face the
/// renderer drew it into.
const CUBE_FACE_DIRS = [6]Vec3{
    .{ .x = 1, .y = 0, .z = 0 },  .{ .x = -1, .y = 0, .z = 0 },
    .{ .x = 0, .y = 1, .z = 0 },  .{ .x = 0, .y = -1, .z = 0 },
    .{ .x = 0, .y = 0, .z = 1 },  .{ .x = 0, .y = 0, .z = -1 },
};
const CUBE_FACE_UPS = [6]Vec3{
    .{ .x = 0, .y = -1, .z = 0 }, .{ .x = 0, .y = -1, .z = 0 },
    .{ .x = 0, .y = 0, .z = 1 },  .{ .x = 0, .y = 0, .z = -1 },
    .{ .x = 0, .y = -1, .z = 0 }, .{ .x = 0, .y = -1, .z = 0 },
};

// ── GPU buffer types ──────────────────────────────────────────────────────────

/// Tunable 3D render settings, exposed to the editor's Settings tab. Defaults reproduce the
/// built-in studio look. Colours are linear rgb; sun direction is given as azimuth/elevation.
pub const Settings3D = struct {
    // Ambient = diffuse fill only now (specular reflections are full-strength). The sky has strong
    // VERTICAL CONTRAST — bright zenith, mid horizon, dark ground — so a smooth metallic body shows
    // a reflection gradient (bright roof/upper, dark sills) instead of a uniform flat-grey "clay"
    // reflection. This is the studio/EEVEE-world look that makes metals read as glossy.
    ambient_intensity: f32 = 0.6, // EEVEE-like world fill: the cosine-convolved env irradiance lights all orientations so albedo reads (was 0.18, which buried diffuse colour under the env reflection)
    sky_horizon: [3]f32 = .{ 0.34, 0.30, 0.26 }, // warm tan, darker → steeper glossy reflection ramp
    sky_zenith: [3]f32 = .{ 0.66, 0.64, 0.56 }, // warm, kills the cool/blue studio cast
    sky_ground: [3]f32 = .{ 0.26, 0.25, 0.23 }, // soft warm-grey lower hemisphere (EEVEE-like world fill) so down-facing normals aren't black (was 0.085, which left objects unlit underneath)
    env_avg: [3]f32 = .{ 0.38, 0.42, 0.50 }, // legacy; unused now the cubemap drives reflections
    sun_azimuth_deg: f32 = 48.0,
    sun_elevation_deg: f32 = 50.0,
    sun_intensity: f32 = 6.0, // brighter direct key → crisper specular glint + stronger form (was 5.0)
    // Stronger studio softboxes → crisp bright streaks across glossy paint/chrome (the studio
    // highlight that reads as a reflective surface rather than matte clay).
    overhead: f32 = 3.2, // stronger softbox streaks (glossy-clearcoat cue)
    horizon_glow: f32 = 0.95,
    sun_sharpness: f32 = 150.0, // tighter reflected sun disc → crisp glint
    // AgX provides the filmic contrast curve; the grade below pushes it toward a punchy photographic look.
    exposure: f32 = 1.10, // pulled down from 1.22 → deeper blacks, less washed midtones
    contrast: f32 = 0.34, // punchier midtones (was 0.28)
    saturation: f32 = 1.20, // restore chroma AgX flattens (was 1.12)
    // ── Photographic look (post-AgX). ──
    // look: 0 = Default (neutral AgX), 1 = Punchy (steeper slope + sat), 2 = Golden (warm, lifted).
    agx_look: f32 = 1.0, // Punchy — counters AgX's washed/desaturated midtones (was 0 = Default)
    // White balance in LINEAR space pre-AgX. temperature>0 warms (boost R, cut B); tint>0 -> magenta.
    wb_temperature: f32 = 0.10, // slight warm → kills the cool shadow cast (was 0.0)
    wb_tint: f32 = 0.0,
    // Vignette: radial darkening from screen centre. strength 0 = off; softness = falloff width.
    vignette_strength: f32 = 0.18,
    vignette_softness: f32 = 0.55,
    // Film grain amount (added in LDR, animated, luma-modulated). Very low default.
    grain_amount: f32 = 0.015,
    // Chromatic aberration at frame edges (per-channel UV split, scales with radius^2). Very low.
    chromatic_aberration: f32 = 0.0015,
    // Radial lens distortion (physical camera): r' = r*(1 + k1*r^2 + k2*r^4). k1<0 barrel, >0 pincushion.
    lens_distortion_k1: f32 = 0.0,
    lens_distortion_k2: f32 = 0.0,
    // Aperture bokeh shape (extends the DoF gather): blade count (0/<3 = circular), anamorphic squeeze.
    bokeh_blades: f32 = 0.0,
    bokeh_anamorphic: f32 = 1.0,
    shadow_strength: f32 = 0.55, // deeper contact/under-car shadow → grounds the car (was 0.45)
    shadow_bias: f32 = 0.006,
    shadow_softness: f32 = 1.5, // PCF kernel radius scale (texels)
    clearcoat: f32 = 1.0,
    // Bloom (HDR bright-pass). threshold/knee are in linear radiance; intensity scales the
    // additive bloom in the tonemap pass. Threshold sits above the sky's peak radiance so
    // only genuine highlights (sun hotspots on metal/clearcoat, emissives) bloom — not the
    // broad sky, which otherwise hazes the whole frame.
    bloom_threshold: f32 = 0.7,
    bloom_knee: f32 = 0.4,
    bloom_intensity: f32 = 0.45,
    // SSAO. radius/bias are in view-space units; strength scales the darkening (0 = off).
    // Tuned DOWN: the old radius/strength over-occluded the whole (convex, open) car body to a dark
    // silhouette in the AO debug view, and the tonemap multiplies AO into the full image. AO should
    // only deepen genuine creases/contact, not darken broad panels.
    ssao_radius: f32 = 0.35,
    ssao_bias: f32 = 0.03,
    ssao_strength: f32 = 0.5,
    ssao_power: f32 = 1.0,
    // SSGI (screen-space indirect diffuse). One bounce of colour bled from nearby lit surfaces,
    // gathered in the same GTAO horizon pass and added in the tonemap. 0 = off. Kept modest — it is an
    // approximation (no albedo separation) and rides the AO denoise.
    ssgi_strength: f32 = 3.0,
    // SSR (screen-space reflections). intensity 0 = off. max_distance/thickness in view units.
    // Lowered (0.5 → 0.3): SSR should refine glossy reflections, not add a white haze around the
    // car. It is already roughness-faded in the SSR shader (smooth surfaces only).
    ssr_intensity: f32 = 0.5,
    ssr_max_distance: f32 = 8.0,
    ssr_thickness: f32 = 0.6,
    ssr_steps: f32 = 32.0,
    // TAA. enabled 0/1; feedback = history blend weight (higher = smoother but more ghosting).
    taa_enabled: f32 = 1.0,
    taa_feedback: f32 = 0.9,
    // ── Depth of field (gather bokeh, EEVEE-style). Runs on linear HDR before tonemap so
    // out-of-focus highlights bloom as round bokeh. CoC ramps from the loc1 view-space depth vs
    // the focus distance; f-stop scales the ramp (lower = shallower DoF); max_coc caps the radius.
    dof_enabled: f32 = 0.0, // off by default — opt in via the Settings tab (saved per-project)
    dof_focus_distance: f32 = 8.0, // view-space distance the lens is focused at (orbit target ≈ 8)
    dof_f_stop: f32 = 2.8, // aperture; lower = more background separation / bigger bokeh
    dof_max_coc: f32 = 18.0, // clamp the blur radius to this many pixels (perf + sanity)
    // ── Diagnostics (Renderer Diagnostic Mode + debug views) ──
    // diagnostic_mode (0/1): forces a stable physically-plausible baseline — disables bloom,
    // SSR, SSAO, TAA, clearcoat, selection rim and the editor grid, neutralises exposure/grade,
    // drops ambient/IBL low, and lights the scene with a single synthetic directional light.
    // Lets material/colour/normal bugs be inspected without post-processing hiding them.
    diagnostic_mode: f32 = 0.0,
    // debug_view: which channel to visualise (see DebugView). 0 = normal shaded output.
    debug_view: f32 = 0.0,
    // wireframe (0/1): draw all geometry as line-list edges (a dedicated wireframe pipeline +
    // per-primitive edge index buffer) shaded a flat unlit colour. wgpu-first; Metal ignores it.
    wireframe: f32 = 0.0,
    // ── Atmospheric fog (height-based exponential + analytic sun in-scatter) ──
    // density 0 = off. The fog colour defaults to the horizon so distant geometry fades into the
    // sky (aerial perspective); sun_inscatter brightens fog toward the sun (god-ray glow), shaped
    // by the Henyey-Greenstein anisotropy g (0 = isotropic, →1 = forward/sun-hugging).
    fog_density: f32 = 0.0,
    fog_color: [3]f32 = .{ 0.55, 0.60, 0.68 },
    fog_height: f32 = 0.0, // world Y where fog is densest
    fog_height_falloff: f32 = 0.15, // how fast density decays with height (0 = uniform)
    fog_sun_inscatter: f32 = 0.6,
    fog_anisotropy: f32 = 0.72,
    // ── Auto-exposure / eye adaptation ──
    // enabled 0/1. The metering pass takes the frame's log-average luminance, maps it to `key` (middle
    // grey), clamps the metered luminance to [min,max], and eases toward it at `speed` per frame. The
    // manual `exposure` above remains an EV bias multiplied on top.
    auto_exposure_enabled: f32 = 0.0,
    auto_exposure_key: f32 = 0.18,
    auto_exposure_min: f32 = 0.03, // darkest average luminance the eye adapts to (caps brightening)
    auto_exposure_max: f32 = 8.0, // brightest average luminance (caps darkening)
    auto_exposure_speed: f32 = 0.08, // per-frame adaptation blend (higher = snappier)
};

/// Render/debug visualisation channels. Codes 1..10 are produced by the mesh fragment shader
/// (material/geometry channels); codes 11..15 are produced by the tonemap pass from the
/// post-processing buffers. Keep in sync with DebugView in C# and the WGSL switch blocks.
pub const DebugView = enum(u32) {
    none = 0,
    base_color = 1,
    world_normal = 2,
    view_normal = 3,
    roughness = 4,
    metallic = 5,
    alpha = 6,
    emissive = 7,
    depth = 8,
    view_position = 9,
    shadow_factor = 10,
    ao = 11,
    ssr_contribution = 12,
    ssr_hit = 13,
    bloom = 14,
    hdr_luminance = 15,
};

/// Per-frame renderer statistics for the debug overlay/profiler (design doc §7.2). Cheap counters
/// incremented during the frame; reset each frame in `beginScene`. No allocation, no GPU stalls.
pub const Stats3D = extern struct {
    frame_index: u64 = 0,
    draw_calls: u32 = 0,
    triangles: u32 = 0,
    render_passes: u32 = 0,
    visible_objects: u32 = 0,
};


const TextureViewGpu = struct {
    texture: *wgpu.Texture,
    texture_view: *wgpu.TextureView,

    pub fn deinit(self: *TextureViewGpu) void {
        self.texture_view.release();
        self.texture.release();
    }
};

const MaterialGpu = struct {
    base_color_tex: ?*wgpu.Texture = null,
    base_color_view: ?*wgpu.TextureView = null,
    normal_tex: ?*wgpu.Texture = null,
    normal_view: ?*wgpu.TextureView = null,
    mr_tex: ?*wgpu.Texture = null,
    mr_view: ?*wgpu.TextureView = null,
    emissive_tex: ?*wgpu.Texture = null,
    emissive_view: ?*wgpu.TextureView = null,
    bind_group: *wgpu.BindGroup,

    pub fn deinit(self: *MaterialGpu) void {
        self.bind_group.release();
        if (self.base_color_view) |v| v.release();
        if (self.base_color_tex) |t| t.release();
        if (self.normal_view) |v| v.release();
        if (self.normal_tex) |t| t.release();
        if (self.mr_view) |v| v.release();
        if (self.mr_tex) |t| t.release();
        if (self.emissive_view) |v| v.release();
        if (self.emissive_tex) |t| t.release();
    }
};

fn createSolidTextureView(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    rgba: [4]u8,
    is_srgb: bool,
) !TextureViewGpu {
    const texture = device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote solid texture"),
        .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
        .dimension = .@"2d",
        .size = .{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
        .format = if (is_srgb) .rgba8_unorm_srgb else .rgba8_unorm,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return error.WgpuTextureCreateFailed;
    errdefer texture.release();

    const texture_view = texture.createView(null) orelse return error.WgpuTextureViewUnavailable;
    errdefer texture_view.release();

    var upload: [256]u8 = .{0} ** 256;
    upload[0] = rgba[0]; upload[1] = rgba[1]; upload[2] = rgba[2]; upload[3] = rgba[3];
    queue.writeTexture(
        &.{ .texture = texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &upload, upload.len,
        &.{ .offset = 0, .bytes_per_row = 256, .rows_per_image = 1 },
        &.{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
    );
    return .{ .texture = texture, .texture_view = texture_view };
}

/// Box-filter downsample a tightly-packed RGBA8 image to roughly half size.
/// Averaging happens in storage (gamma) space — a standard, good-enough approximation.
fn boxDownsampleRgba8(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
    var y: u32 = 0;
    while (y < dst_h) : (y += 1) {
        const sy0 = @min(y * 2, src_h - 1);
        const sy1 = @min(sy0 + 1, src_h - 1);
        var x: u32 = 0;
        while (x < dst_w) : (x += 1) {
            const sx0 = @min(x * 2, src_w - 1);
            const sx1 = @min(sx0 + 1, src_w - 1);
            const p00 = (@as(usize, sy0) * src_w + sx0) * 4;
            const p01 = (@as(usize, sy0) * src_w + sx1) * 4;
            const p10 = (@as(usize, sy1) * src_w + sx0) * 4;
            const p11 = (@as(usize, sy1) * src_w + sx1) * 4;
            const di = (@as(usize, y) * dst_w + x) * 4;
            var c: usize = 0;
            while (c < 4) : (c += 1) {
                const sum = @as(u32, src[p00 + c]) + src[p01 + c] + src[p10 + c] + src[p11 + c];
                dst[di + c] = @intCast(sum / 4);
            }
        }
    }
}

// Per-entity GPU instancing state for an instanced mesh draw. `cpu` holds the flat,
// column-major 4x4 model matrices (16 f32 each) the game submits each frame; the renderer
// uploads them into `gpu` (grown on demand) and issues one drawIndexed(idx, count) instead
// of `count` single draws. count==0 means the node draws nothing (an instanced node is never
// rendered as a single fallback draw — see the geometry/shadow loops).
const InstanceGpu = struct {
    cpu: std.ArrayListUnmanaged(f32) = .{ .items = &.{}, .capacity = 0 },
    gpu: ?*wgpu.Buffer = null,
    capacity: u32 = 0, // instances the gpu buffer can hold
    count: u32 = 0, // live instance count this frame
    // Stored so the pool can auto-free `cpu` on slot release (zpool calls the 0-arg
    // deinit). Null for a default-constructed value that was never populated.
    alloc: ?std.mem.Allocator = null,

    pub fn deinit(self: *InstanceGpu) void {
        if (self.gpu) |b| b.release();
        if (self.alloc) |a| self.cpu.deinit(a);
    }
};

// Generational handle pools backing the per-entity GPU resource caches. Storage is
// contiguous and capacity-reservable (no value-rehash spikes when entity counts grow
// under load), and a handle whose slot was freed/reused fails isLiveHandle instead of
// silently aliasing a different entity's resource. 20 index bits (≤1M live) + 12 cycle
// bits pack into a u32 handle id, stored in the small Entity→id index maps below.
const InstancePool = zpool.Pool(20, 12, InstanceGpu, struct { gpu: InstanceGpu });

// A blended renderable tagged with its squared distance to the camera so the
// transparent pass can draw farthest-first.
const TransparentItem = struct {
    renderable: scene_mod.Renderable,
    dist_sq: f32,
};

// One opaque/masked draw, bucketed then sorted by (double_sided, material) so the draw loop can
// skip redundant pipeline + material-bind-group switches for runs of the same material.
const OpaqueItem = struct {
    renderable: scene_mod.Renderable,
    material: scene_mod.MaterialHandle,
    double_sided: u8,
};

// A shadow caster classified ONCE per frame (drawShadowCasters ran the gizmo/instancing/alpha-mode
// classification per renderable per slice — 4..20× redundant). The per-slice loop now only frustum-
// tests and draws these. `model_off` memoizes the caster's slice-invariant ModelUniforms ring slot:
// the first slice that draws it pushes the slot, later slices reuse the offset (its light matrix
// rides group 2's dynamic offset, not group 1). NO_MODEL_OFF = not yet pushed this frame.
const NO_MODEL_OFF: u32 = 0xFFFF_FFFF;
const ShadowCaster = struct {
    renderable: scene_mod.Renderable,
    mesh_data: *const resources_mod.Mesh,
    model_off: u32 = NO_MODEL_OFF,
    // Masked-only (foliage/decal cut-out shadows); opaque casters leave these null. The material
    // pointer is stable for the frame and lets drawShadowCasters reconstruct the exact ModelUniforms.
    material: ?*const resources_mod.Material = null,
    mat_bg: ?*wgpu.BindGroup = null,
};

// ── Gpu3d ─────────────────────────────────────────────────────────────────────

pub const Gpu3d = struct {
    allocator: std.mem.Allocator,
    device: *wgpu.Device,
    surface_format: wgpu.TextureFormat,

    pipeline: *wgpu.RenderPipeline,
    // Same as `pipeline` but with back-face culling off (`cull_mode = .none`); selected per-draw
    // when a material has `double_sided` set. Two-sided foliage/cloth otherwise loses its back faces.
    pipeline_double_sided: *wgpu.RenderPipeline,
    // Same layout as `pipeline` but with depth writes disabled, for back-to-front
    // alpha-blended geometry. Opaque depth still occludes it via the depth test.
    transparent_pipeline: *wgpu.RenderPipeline,
    transparent_pipeline_ds: *wgpu.RenderPipeline, // no-cull variant for double-sided materials
    glass_pipeline: *wgpu.RenderPipeline, // transparent variant with depth-write ON (glass self-sorts)
    glass_pipeline_ds: *wgpu.RenderPipeline, // no-cull glass for double-sided windshields/panes
    // Same bind-group layout as `pipeline`, plus a second vertex buffer (step_mode=instance)
    // carrying a per-instance model matrix. One drawIndexed(idx, N) draws a whole batch.
    instanced_pipeline: *wgpu.RenderPipeline,
    // Wireframe variants of `pipeline` / `instanced_pipeline`: identical layout/shaders but with
    // line-list topology (and no back-face cull). Bound when settings.wireframe is set; they draw
    // the per-primitive edge index buffer. The mesh shader emits a flat colour for these.
    wireframe_pipeline: *wgpu.RenderPipeline,
    wireframe_instanced_pipeline: *wgpu.RenderPipeline,
    shadow_pipeline: *wgpu.RenderPipeline,
    // Alpha-tested cascade/spot shadow pipeline (cull-none, fragment discards below cutoff) — lets
    // masked casters (foliage/decals) render cut-out shadows. Layout adds the material texture group.
    shadow_alpha_pipeline: *wgpu.RenderPipeline,
    sky_pipeline: *wgpu.RenderPipeline,
    camera_bgl: *wgpu.BindGroupLayout,
    model_bgl: *wgpu.BindGroupLayout,
    texture_bgl: *wgpu.BindGroupLayout,
    shadow_map_bgl: *wgpu.BindGroupLayout,

    camera_buf: *wgpu.Buffer,
    light_buf: *wgpu.Buffer,
    camera_bg: *wgpu.BindGroup,

    shadow_map_bg: *wgpu.BindGroup,
    // Variant of shadow_map_bg for the GLASS pass only: binding 6 = the view-position G-buffer (read for
    // thickness-aware refraction). The opaque pass WRITES that G-buffer, so it can't also bind it there;
    // shadow_map_bg carries a harmless dummy at binding 6 for the opaque/sky passes.
    shadow_map_bg_glass: *wgpu.BindGroup,
    shadow_texture: *wgpu.Texture,
    shadow_view: *wgpu.TextureView, // depth-2d-array view, all layers — sampled by the mesh shader
    shadow_layer_views: [TOTAL_SHADOW_LAYERS]*wgpu.TextureView, // per-layer depth render targets
    shadow_sampler: *wgpu.Sampler,
    // Per-slice light view-proj matrix (cascade or spot), bound to the shadow pass via a dynamic-offset UBO.
    shadow_cascade_bgl: *wgpu.BindGroupLayout,
    shadow_cascade_buf: *wgpu.Buffer, // one SHADOW_SLOT per shadow layer
    shadow_cascade_bg: *wgpu.BindGroup,
    active_spot_shadows: u32 = 0, // shadow-casting spot lights this frame (extra layers after the cascades)
    // Omnidirectional point-light shadows: a depth cube-array + the per-face pass pipeline/UBO.
    point_shadow_texture: *wgpu.Texture,
    point_shadow_view: *wgpu.TextureView, // cube-array view, sampled by the mesh shader
    point_shadow_face_views: [POINT_CUBE_LAYERS]*wgpu.TextureView, // per-face depth render targets
    point_shadow_pipeline: *wgpu.RenderPipeline,
    point_shadow_bgl: *wgpu.BindGroupLayout,
    point_shadow_buf: *wgpu.Buffer, // one SHADOW_SLOT per cube face
    point_shadow_bg: *wgpu.BindGroup,
    active_point_shadows: u32 = 0, // shadow-casting point lights this frame
    // Lazy shadow allocation: the directional depth-array and the point cube-array are sized to the
    // scene's ACTUAL shadow-caster counts (grow-on-demand), not the compile-time maxima. A scene with
    // no spot/point shadows (the common case, e.g. one directional sun) therefore pays only for
    // NUM_CASCADES directional layers + a 1×1×6 point placeholder instead of TOTAL_SHADOW_LAYERS +
    // POINT_CUBE_LAYERS at full resolution. `ensureShadowCapacity` grows these (never shrinks) as
    // spots/points appear; the shader never samples an unallocated layer (guarded by per-light indices),
    // so the reduced allocation is bit-for-bit identical in output.
    shadow_layers_alloc: u32 = NUM_CASCADES, // directional array layers currently allocated (>= NUM_CASCADES)
    point_cubes_alloc: u32 = 0, // full-res point cubes currently allocated; 0 = the 1×1×6 placeholder
    // Per-shadow-slice view-proj matrices (CPU copy), for per-slice frustum culling of casters so each
    // cascade / spot / cube-face only redraws the geometry it can actually see. shadow_slice_vp holds
    // cascades [0..NUM_CASCADES) then spots [NUM_CASCADES..); point_face_vp is one per cube face.
    shadow_slice_vp: [TOTAL_SHADOW_LAYERS]Mat4 = undefined,
    point_face_vp: [POINT_CUBE_LAYERS]Mat4 = undefined,

    // Environment IBL: prefiltered cubemap + the bake pipeline that fills it.
    env_cube_texture: *wgpu.Texture,
    env_cube_view: *wgpu.TextureView, // cube view, all mips — sampled by the mesh shader
    env_face_views: [ENV_MIPS][6]*wgpu.TextureView, // 2D render targets, one per face/mip
    env_sampler: *wgpu.Sampler,
    env_bake_pipeline: *wgpu.RenderPipeline,
    env_bake_bgl: *wgpu.BindGroupLayout,
    env_bake_buf: *wgpu.Buffer, // dynamic-offset UBO, one ENV_SLOT per face/mip
    env_bake_bg: *wgpu.BindGroup,
    env_equirect_tex: *wgpu.Texture, // 1×1 dummy until an HDRI is supplied
    env_equirect_view: *wgpu.TextureView,
    env_mode: u32 = 0, // 0 = procedural studio, 1 = HDRI
    env_dirty: bool = true, // rebake on next render
    // Last diagnostic_mode the environment was baked with. Toggling diagnostic mode swaps the
    // sky/studio inputs the cubemap is baked from, so a change forces a rebake.
    env_baked_diagnostic: bool = false,

    // ── Reflection probe (box-projected env cubemap) ──
    // A finite box volume (EEVEE reflection-probe model): reflections are parallax-corrected to hit
    // the box walls instead of an infinitely-distant sky, so the env appears anchored to the room.
    // extents all-zero ⇒ disabled (default = global infinite env, no regression).
    probe_center: [3]f32 = .{ 0, 0, 0 },
    probe_extents: [3]f32 = .{ 0, 0, 0 },

    default_base_color: TextureViewGpu,
    default_normal: TextureViewGpu,
    default_mr: TextureViewGpu,
    default_sampler: *wgpu.Sampler,
    default_material: MaterialGpu,

    depth_texture: ?*wgpu.Texture = null,
    depth_view: ?*wgpu.TextureView = null,
    depth_width: u32 = 0,
    depth_height: u32 = 0,
    // Resolution of the SSAO/SSGI/SSR effect targets (ao/gi_history/ssr). Equals the frame size at
    // post_scale 1 (default); at post_scale 2 these run at half resolution (¼ the invocations) and
    // the tonemap upsamples them via its linear sampler. Set in ensurePostTargets; the temporal
    // gi_history copy uses this extent (the G-buffers it reads stay full-res).
    post_width: u32 = 0,
    post_height: u32 = 0,
    // 4x MSAA color target; the scene resolves into the (single-sample) HDR scene view.
    msaa_color_texture: ?*wgpu.Texture = null,
    msaa_color_view: ?*wgpu.TextureView = null,

    // VFX particle billboards, drawn in the geometry pass after the transparent meshes. Lazy +
    // failure-isolated, so it stays a no-op (and never threatens boot) until the host uploads particles.
    particles: particles_mod.ParticleSystem = .{},

    // 2D sprite renderer: scene stage drawn after the geometry pass (own pass on the resolved HDR
    // target), overlay stage after post (exact LDR colors). Lazy + failure-isolated like particles.
    sprites: sprites_mod.SpriteSystem = .{},

    // ── HDR pipeline + post-processing ─────────────────────────────────────────
    // Linear HDR scene buffer the 3D content resolves into (sky + meshes).
    scene_hdr_texture: ?*wgpu.Texture = null,
    scene_hdr_view: ?*wgpu.TextureView = null,
    // Refraction source: a single-sample copy of the opaque scene_hdr taken before the glass pass, so
    // glass can sample the scene behind it for screen-space refraction. Bound at shadow group binding 5;
    // shadow_map_bg is rebuilt on resize to point at the current frame-sized copy.
    refraction_src_texture: ?*wgpu.Texture = null,
    refraction_src_view: ?*wgpu.TextureView = null,
    // Depth-of-field output (linear HDR, full-res). The tonemap reads this instead of scene_hdr
    // when DoF is active. Gather bokeh from scene_hdr + the view-position G-buffer.
    dof_texture: ?*wgpu.Texture = null,
    dof_view: ?*wgpu.TextureView = null,
    // Mip-chain bloom (rgba16float): a single texture with per-mip single-level views.
    // Downsample scene_hdr → mip1 → … → mip(n-1); additive tent upsample back to mip1; tonemap
    // reads mip1. mip0 (full-res) is allocated but unused (kept so mip indices match resolution).
    bloom_chain_texture: ?*wgpu.Texture = null,
    bloom_mip_views: [BLOOM_MIPS]?*wgpu.TextureView = .{null} ** BLOOM_MIPS,
    bloom_mip_count: u32 = 0,
    // bloom_down_bgs[i] samples the source for the pass writing mip i (scene_hdr for i=1, else
    // mip i-1). bloom_up_bgs[i] samples mip i for the pass adding into mip i-1.
    bloom_down_bgs: [BLOOM_MIPS]?*wgpu.BindGroup = .{null} ** BLOOM_MIPS,
    bloom_up_bgs: [BLOOM_MIPS]?*wgpu.BindGroup = .{null} ** BLOOM_MIPS,
    // SSAO G-buffer: view-space position (MSAA target + resolved single-sample) + AO result.
    msaa_pos_texture: ?*wgpu.Texture = null,
    msaa_pos_view: ?*wgpu.TextureView = null,
    gbuf_pos_texture: ?*wgpu.Texture = null,
    gbuf_pos_view: ?*wgpu.TextureView = null,
    msaa_normal_texture: ?*wgpu.Texture = null,
    msaa_normal_view: ?*wgpu.TextureView = null,
    gbuf_normal_texture: ?*wgpu.Texture = null,
    gbuf_normal_view: ?*wgpu.TextureView = null,
    // Albedo (base colour) G-buffer — MRT location 3. Lets the tonemap tint the SSGI indirect bounce by
    // the RECEIVER's albedo (a green floor reflects green), instead of adding untinted light.
    msaa_albedo_texture: ?*wgpu.Texture = null,
    msaa_albedo_view: ?*wgpu.TextureView = null,
    gbuf_albedo_texture: ?*wgpu.Texture = null,
    gbuf_albedo_view: ?*wgpu.TextureView = null,
    ao_texture: ?*wgpu.Texture = null,
    ao_view: ?*wgpu.TextureView = null,
    // Auto-exposure: 1×1 adapted-multiplier output (sampled by tonemap) + its history (copied each
    // frame, read by next frame's metering pass for temporal adaptation). exposure_bg binds scene_hdr.
    exposure_out_texture: ?*wgpu.Texture = null,
    exposure_out_view: ?*wgpu.TextureView = null,
    exposure_hist_texture: ?*wgpu.Texture = null,
    exposure_hist_view: ?*wgpu.TextureView = null,
    exposure_bg: ?*wgpu.BindGroup = null,
    exposure_valid: bool = false, // history populated (cleared on resize / first frame)
    // Previous frame's accumulated SSGI/AO buffer (copy of ao_tex), reprojected + blended by the SSAO
    // pass for temporal accumulation. gi_history_valid gates the first frame / post-resize.
    gi_history_texture: ?*wgpu.Texture = null,
    gi_history_view: ?*wgpu.TextureView = null,
    gi_history_valid: bool = false,
    ssr_texture: ?*wgpu.Texture = null,
    ssr_view: ?*wgpu.TextureView = null,
    // TAA: tonemapped current frame (LDR) + resolved output (LDR, owned) + accumulated history.
    // The resolve writes to taa_output (renderer-owned), which is then copied to BOTH the history
    // and the destination — so TAA never depends on the destination texture's usage flags.
    taa_input_texture: ?*wgpu.Texture = null,
    taa_input_view: ?*wgpu.TextureView = null,
    taa_output_texture: ?*wgpu.Texture = null,
    taa_output_view: ?*wgpu.TextureView = null,
    taa_history_texture: ?*wgpu.Texture = null,
    taa_history_view: ?*wgpu.TextureView = null,
    // True only when the active render path can run the TAA resolve (offscreen destination that
    // can receive the resolved copy). The legacy `render()` wrapper clears it. Gates camera jitter
    // so the image never shimmers on a path where the resolve won't run (Task: jitter↔resolve).
    taa_path_supported: bool = true,
    // Projection matrix of the last world-3D layer, used to project SSAO/SSR samples.
    last_proj: [16]f32 = [_]f32{0} ** 16,
    last_view: [16]f32 = [_]f32{0} ** 16, // world→view of the last world-3D layer (for contact shadows)
    // TAA reprojection state (unjittered): current view-proj (becomes prev next frame),
    // previous view-proj, and current inverse view (view_pos → world).
    cur_view_proj: [16]f32 = [_]f32{0} ** 16,
    prev_view_proj: [16]f32 = [_]f32{0} ** 16,
    inv_view: [16]f32 = [_]f32{0} ** 16,
    taa_frame: u32 = 0,
    taa_valid: bool = false, // history populated (cleared on resize / first frame)
    // Post pipelines + layouts + param buffers (created once in init).
    bloom_down_pipeline: *wgpu.RenderPipeline, // 13-tap COD downsample (replace blend)
    bloom_up_pipeline: *wgpu.RenderPipeline, // 9-tap tent upsample (additive blend)
    dof_pipeline: *wgpu.RenderPipeline, // gather-bokeh depth of field
    tonemap_pipeline: *wgpu.RenderPipeline,
    ssao_pipeline: *wgpu.RenderPipeline,
    ssr_pipeline: *wgpu.RenderPipeline,
    taa_pipeline: *wgpu.RenderPipeline,
    exposure_pipeline: *wgpu.RenderPipeline, // auto-exposure metering (scene → 1×1 adapted multiplier)
    exposure_bgl: *wgpu.BindGroupLayout, // scene + sampler + history + params
    exposure_buf: *wgpu.Buffer,
    post_bgl: *wgpu.BindGroupLayout, // 1 texture + sampler + params (bloom downsample)
    ssao_bgl: *wgpu.BindGroupLayout, // 2 textures (pos + normal) + sampler + params (ssao)
    bloom_up_bgl: *wgpu.BindGroupLayout, // 1 texture + sampler (upsample; no UBO)
    dof_bgl: *wgpu.BindGroupLayout, // 2 textures + sampler + params (dof)
    tonemap_bgl: *wgpu.BindGroupLayout, // 4 textures + sampler + params (tonemap)
    ssr_bgl: *wgpu.BindGroupLayout, // 2 textures + sampler + params (ssr)
    taa_bgl: *wgpu.BindGroupLayout, // 3 textures + sampler + params (taa)
    post_sampler: *wgpu.Sampler,
    bloom_down_first_buf: *wgpu.Buffer, // DownParams with is_first=1 (prefilter+karis)
    bloom_down_rest_buf: *wgpu.Buffer, // DownParams with is_first=0
    dof_buf: *wgpu.Buffer,
    tonemap_buf: *wgpu.Buffer,
    ssao_buf: *wgpu.Buffer,
    ssr_buf: *wgpu.Buffer,
    taa_buf: *wgpu.Buffer,
    // Per-size bind groups (recreated on resize in ensureDepthTexture). Bloom down/up bind
    // groups live in bloom_down_bgs / bloom_up_bgs above.
    dof_bg: ?*wgpu.BindGroup = null, // scene_hdr + gbuf_pos → dof
    tonemap_dof_bg: ?*wgpu.BindGroup = null, // tonemap reading dof_view at binding 0
    tonemap_bg: ?*wgpu.BindGroup = null,
    ssao_bg: ?*wgpu.BindGroup = null,
    ssr_bg: ?*wgpu.BindGroup = null,
    taa_bg: ?*wgpu.BindGroup = null,

    mesh_cache: MeshCache,
    // GPU material resources, indexed DIRECTLY by material handle (a dense u32 index into
    // World.materials) rather than an AutoHashMap: the per-draw lookup is a bounds-check + load
    // instead of hashing a u32 and probing. `null` = not yet uploaded. Every ensureMaterialGpu call
    // is guarded by a successful world.getMaterial (the null_material sentinel never reaches here), so
    // the handle is always < materials count → the array stays dense and bounded.
    material_gpu_cache: std.ArrayListUnmanaged(?MaterialGpu) = .empty,
    render_list: std.ArrayListUnmanaged(scene_mod.Renderable) = .{ .items = &.{}, .capacity = 0 },
    // The world-3D renderable set, collected ONCE per frame in beginScene and reused by both the
    // shadow pass (read-only) and the geometry pass (copied into render_list, then frustum-culled).
    // Avoids re-walking the whole scene tree 2× per frame for the identical set. (Renderable is two
    // pointers, so the per-frame copy into render_list is far cheaper than a second tree traversal.)
    world3d_list: std.ArrayListUnmanaged(scene_mod.Renderable) = .{ .items = &.{}, .capacity = 0 },
    // Frustum culling of the world-3D geometry pass (set via zigote_render_set_frustum_cull). On by
    // default; drops draws whose world-space bounding sphere is fully outside the camera frustum.
    frustum_cull: bool = true,
    // Scratch buffer for depth-sorting blended renderables each frame (back-to-front).
    transparent_scratch: std.ArrayListUnmanaged(TransparentItem) = .{ .items = &.{}, .capacity = 0 },
    // Glass renderables, deferred to a second pass (after copying the opaque scene) for refraction.
    glass_scratch: std.ArrayListUnmanaged(TransparentItem) = .{ .items = &.{}, .capacity = 0 },
    // Reused per-frame light list (cleared, not freed, each frame) so collectLights doesn't
    // alloc+free on the steady path — the 3D scene renders every frame.
    light_scratch: std.ArrayListUnmanaged(scene_mod.ActiveLight) = .{ .items = &.{}, .capacity = 0 },
    // Instanced renderables collected during the opaque loop, drawn together with the
    // instanced pipeline afterwards.
    instanced_scratch: std.ArrayListUnmanaged(scene_mod.Renderable) = .{ .items = &.{}, .capacity = 0 },
    // Opaque draws for one layer, sorted by (double_sided, material) before drawing to minimise
    // pipeline/material-bind state changes (the model group still rebinds per object).
    opaque_scratch: std.ArrayListUnmanaged(OpaqueItem) = .{ .items = &.{}, .capacity = 0 },
    // Shadow casters classified ONCE per frame (opaque vs alpha-masked), reused across every shadow
    // slice so the per-slice loop only frustum-tests + draws. Rebuilt in classifyShadowCasters.
    shadow_casters_opaque: std.ArrayListUnmanaged(ShadowCaster) = .{ .items = &.{}, .capacity = 0 },
    shadow_casters_masked: std.ArrayListUnmanaged(ShadowCaster) = .{ .items = &.{}, .capacity = 0 },
    // Per-entity model UBO+bind-group, stored in a generational pool; the index map resolves
    // the entity-keyed FFI ops (invalidateEntity) to a handle.

    // Shared dynamic-offset model UBO ("ring"): ONE buffer of 256-byte slots + ONE bind group,
    // replacing the old per-entity model buffer + bind-group churn. Each draw writes its slot once at
    // a distinct cursor offset (disjoint across the shadow + geometry passes that share a command
    // buffer) and binds it via a dynamic offset. The cursor resets each frame; the buffer only grows,
    // sized to the per-frame high-water mark of model slots used.
    model_ring: ?*wgpu.Buffer = null,
    model_ring_bg: ?*wgpu.BindGroup = null,
    model_ring_slots: u32 = 0,
    model_cursor: u32 = 0,
    model_hwm: u32 = 0,
    // CPU mirror of the model ring: pushModel writes slots here and flushModelRing uploads the
    // written range in ONE queue.writeBuffer per stage (shadow / geometry) instead of one per draw.
    model_staging: []u8 = &.{},
    model_flushed: u32 = 0,
    // Per-entity instance buffers (set via FFI zigote_scene_set_mesh_instances), same scheme.
    instance_pool: InstancePool,
    instance_index: std.AutoHashMapUnmanaged(scene_mod.Entity, u32) = .{},
    // Entities already reported (once) as falling back to the default material — so the renderer
    // can flag "this primitive is rendering with the grey/silver default" without log spam.
    logged_default_mat: std.AutoHashMapUnmanaged(scene_mod.Entity, void) = .{},
    selected_node_ptr: u64 = 0,
    settings: Settings3D = .{},
    stats: Stats3D = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        surface_format: wgpu.TextureFormat,
    ) !Gpu3d {
        @setEvalBranchQuota(4000); // large pipeline-init body (shadows/post/env/point-shadows)
        const shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.mesh_shader_source),
            }),
        }) orelse return error.ShaderCreateFailed;
        defer shader.release();

        // Camera uniform buffer (persistent, updated each frame)
        const camera_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("camera ubo"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(CameraUniforms),
        }) orelse return error.BufferCreateFailed;

        // Light uniform buffer
        const light_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("light ubo"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(LightUniforms),
        }) orelse return error.BufferCreateFailed;

        // Camera bind group layout (group 0)
        const camera_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{
                .binding = 0,
                .visibility = wgpu.ShaderStages.vertex | wgpu.ShaderStages.fragment,
                .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(CameraUniforms) },
            },
            .{
                .binding = 1,
                .visibility = wgpu.ShaderStages.vertex | wgpu.ShaderStages.fragment,
                .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(LightUniforms) },
            },
        };
        const camera_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("global bgl"),
            .entry_count = camera_bgl_entries.len,
            .entries = &camera_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;

        const camera_bg_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .buffer = camera_buf, .size = @sizeOf(CameraUniforms) },
            .{ .binding = 1, .buffer = light_buf, .size = @sizeOf(LightUniforms) },
        };
        const camera_bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("global bg"),
            .layout = camera_bgl,
            .entry_count = camera_bg_entries.len,
            .entries = &camera_bg_entries,
        }) orelse return error.BindGroupCreateFailed;

        // Model bind group layout (group 1)
        const model_bgl_entries = [_]wgpu.BindGroupLayoutEntry{.{
            .binding = 0,
            .visibility = wgpu.ShaderStages.vertex | wgpu.ShaderStages.fragment,
            .buffer = .{ .type = .uniform, .has_dynamic_offset = 1, .min_binding_size = @sizeOf(ModelUniforms) },
        }};
        const model_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("model bgl"),
            .entry_count = model_bgl_entries.len,
            .entries = &model_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;

        const texture_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
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
                .sampler = .{ .type = .filtering },
            },
            .{
                .binding = 2,
                .visibility = wgpu.ShaderStages.fragment,
                .texture = .{
                    .sample_type = .float,
                    .view_dimension = .@"2d",
                    .multisampled = 0,
                },
            },
            .{
                .binding = 3,
                .visibility = wgpu.ShaderStages.fragment,
                .sampler = .{ .type = .filtering },
            },
            .{
                .binding = 4,
                .visibility = wgpu.ShaderStages.fragment,
                .texture = .{
                    .sample_type = .float,
                    .view_dimension = .@"2d",
                    .multisampled = 0,
                },
            },
            .{
                .binding = 5,
                .visibility = wgpu.ShaderStages.fragment,
                .sampler = .{ .type = .filtering },
            },
            // Emissive map (sRGB) + sampler.
            .{
                .binding = 6,
                .visibility = wgpu.ShaderStages.fragment,
                .texture = .{
                    .sample_type = .float,
                    .view_dimension = .@"2d",
                    .multisampled = 0,
                },
            },
            .{
                .binding = 7,
                .visibility = wgpu.ShaderStages.fragment,
                .sampler = .{ .type = .filtering },
            },
        };
        const texture_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("texture bgl"),
            .entry_count = texture_bgl_entries.len,
            .entries = &texture_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;

        const shadow_map_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .depth, .view_dimension = .@"2d_array" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .comparison } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .cube } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 4, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .depth, .view_dimension = POINT_SHADOW_VIEW_DIM } },
            // binding 5: refraction source (opaque scene copy) for glass screen-space refraction.
            // Rides in this group because WebGPU caps pipelines at 4 bind groups (0..3).
            .{ .binding = 5, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            // binding 6: view-space position G-buffer (unfilterable f16) — glass reads the depth of the
            // surface behind it to make refraction thickness-aware (thin flat glass barely refracts).
            .{ .binding = 6, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .unfilterable_float, .view_dimension = .@"2d" } },
        };
        const shadow_map_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("shadow map bgl"),
            .entry_count = shadow_map_bgl_entries.len,
            .entries = &shadow_map_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;

        const bgls = [_]*wgpu.BindGroupLayout{ camera_bgl, model_bgl, texture_bgl, shadow_map_bgl };
        const pipeline_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d pipeline layout"),
            .bind_group_layout_count = bgls.len,
            .bind_group_layouts = &bgls,
        }) orelse return error.PipelineLayoutFailed;
        defer pipeline_layout.release();

        // Vertex buffer layout: matches resources.GpuVertex (28 bytes) — position(float32x3),
        // normal(snorm8x4), uv(float32x2), tangent(snorm8x4). snorm8 normal/tangent are decoded to
        // vec3/vec4<f32> in [-1,1] for the shader's existing vec3/vec4 inputs (the extra normal.w
        // component is dropped), so no shader change is needed. Offsets are pinned by comptime
        // asserts on GpuVertex in mesh.zig. This one layout feeds every mesh pipeline (opaque/
        // transparent/glass/instanced/shadow), so shadow pipelines that read only a subset of the
        // locations get the same stride automatically.
        const vertex_attrs = [_]wgpu.VertexAttribute{
            .{ .format = .float32x3, .offset = 0,  .shader_location = 0 }, // position
            .{ .format = .snorm8x4,  .offset = 12, .shader_location = 1 }, // normal (xyz; w unused)
            .{ .format = .float32x2, .offset = 16, .shader_location = 2 }, // uv
            .{ .format = .snorm8x4,  .offset = 24, .shader_location = 3 }, // tangent (xyz + w handedness)
        };
        const vertex_buf_layout = wgpu.VertexBufferLayout{
            .array_stride = @sizeOf(resources_mod.GpuVertex),
            .step_mode = .vertex,
            .attribute_count = vertex_attrs.len,
            .attributes = &vertex_attrs,
        };

        const depth_stencil_state = wgpu.DepthStencilState{
            .format = .depth32_float,
            .depth_write_enabled = .true,
            .depth_compare = .less,
            .stencil_front = .{ .compare = .always, .fail_op = .keep, .depth_fail_op = .keep, .pass_op = .keep },
            .stencil_back = .{ .compare = .always, .fail_op = .keep, .depth_fail_op = .keep, .pass_op = .keep },
        };

        // Opaque colour target: NO blending. Opaque surfaces must be solid regardless of the
        // alpha their shader happens to emit — relying on alpha == 1.0 to avoid accidental
        // transparency is fragile (unlit/CRT/debug materials could blend a hole into the scene).
        const opaque_color_target = wgpu.ColorTargetState{
            .format = SCENE_HDR_FORMAT,
            .blend = null,
            .write_mask = wgpu.ColorWriteMasks.all,
        };
        // Transparent colour target: standard src-alpha over blending (used only by the
        // dedicated transparent/glass pass).
        const transparent_color_target = wgpu.ColorTargetState{
            .format = SCENE_HDR_FORMAT,
            .blend = &wgpu.BlendState{
                .color = .{ .src_factor = .src_alpha, .dst_factor = .one_minus_src_alpha, .operation = .add },
                .alpha = .{ .src_factor = .one, .dst_factor = .zero, .operation = .add },
            },
            .write_mask = wgpu.ColorWriteMasks.all,
        };
        // MRT location 1 = view-space position (SSAO/SSR), location 2 = view normal + roughness
        // (SSR). Opaque writes both G-buffer targets; transparent (glass) writes neither
        // (write_mask none) so the opaque surface behind the glass remains in the G-buffer.
        const opaque_targets = [_]wgpu.ColorTargetState{
            opaque_color_target,
            .{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all },
            .{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all },
            .{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all }, // location 3: albedo
        };
        const transparent_targets = [_]wgpu.ColorTargetState{
            transparent_color_target,
            .{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.none },
            .{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.none },
            .{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.none }, // location 3: albedo (glass/blend don't write)
        };

        const pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d pipeline"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .back,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = opaque_targets.len,
                .targets = &opaque_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Double-sided opaque variant: identical to `pipeline` but with NO back-face cull, selected
        // per-draw when a material sets `double_sided` (two-sided foliage/cloth/leaves/paper). The flag
        // was stored on Material but never honored before; single-sided geometry keeps `pipeline`.
        const pipeline_double_sided = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d pipeline (double-sided)"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = opaque_targets.len,
                .targets = &opaque_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Transparent variant: identical to `pipeline` but it does NOT write depth, so
        // blended surfaces don't occlude the geometry behind them. Depth testing stays
        // on (compare .less) so opaque geometry still hides transparent fragments.
        const transparent_depth_state = wgpu.DepthStencilState{
            .format = .depth32_float,
            .depth_write_enabled = .false,
            .depth_compare = .less,
            .stencil_front = .{ .compare = .always, .fail_op = .keep, .depth_fail_op = .keep, .pass_op = .keep },
            .stencil_back = .{ .compare = .always, .fail_op = .keep, .depth_fail_op = .keep, .pass_op = .keep },
        };
        const transparent_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d transparent pipeline"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .back,
            },
            .depth_stencil = &transparent_depth_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = transparent_targets.len,
                .targets = &transparent_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Glass variant: like the transparent pipeline (MRT masked off) but with depth-WRITE ON.
        // Glass composites the refracted background itself → opaque output, so it must depth-sort like
        // opaque geometry; otherwise a bottle's far/inner glass surfaces (neck, base rim, back wall)
        // draw over the near ones with no ordering. Depth-write on → only the nearest glass surface
        // per pixel survives. Reuses the opaque depth state (write on, compare less).
        const glass_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d glass pipeline"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .back,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = transparent_targets.len,
                .targets = &transparent_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Double-sided variants of the transparent + glass pipelines (no cull), selected per-draw
        // when the material sets `double_sided` — windshields, leaves and cloth are authored
        // two-sided and must shade their back faces just like the opaque double-sided variant.
        const transparent_pipeline_ds = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d transparent pipeline (double-sided)"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .depth_stencil = &transparent_depth_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = transparent_targets.len,
                .targets = &transparent_targets,
            },
        }) orelse return error.PipelineCreateFailed;
        const glass_pipeline_ds = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d glass pipeline (double-sided)"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = transparent_targets.len,
                .targets = &transparent_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Instanced variant: same bind-group layout, but a second vertex buffer (step_mode=
        // instance) supplies a per-instance 4x4 model matrix as four float32x4 columns at
        // locations 4..7. `vs_instanced` reads it; one drawIndexed(idx, N) covers the batch.
        const instance_attrs = [_]wgpu.VertexAttribute{
            .{ .format = .float32x4, .offset = 0, .shader_location = 4 },
            .{ .format = .float32x4, .offset = 16, .shader_location = 5 },
            .{ .format = .float32x4, .offset = 32, .shader_location = 6 },
            .{ .format = .float32x4, .offset = 48, .shader_location = 7 },
        };
        const instance_buf_layout = wgpu.VertexBufferLayout{
            .array_stride = 64,
            .step_mode = .instance,
            .attribute_count = instance_attrs.len,
            .attributes = &instance_attrs,
        };
        const instanced_vbufs = [_]wgpu.VertexBufferLayout{ vertex_buf_layout, instance_buf_layout };
        const instanced_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d instanced pipeline"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_instanced"),
                .buffer_count = instanced_vbufs.len,
                .buffers = &instanced_vbufs,
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .back,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = opaque_targets.len,
                .targets = &opaque_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Wireframe pipelines: clones of `pipeline` / `instanced_pipeline` with line-list topology
        // and no back-face cull (a line has no winding). They draw the per-primitive edge index
        // buffer; the mesh shader's wireframe branch (light.debug.z) emits a flat colour.
        const wireframe_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d wireframe pipeline"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .line_list,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = opaque_targets.len,
                .targets = &opaque_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        const wireframe_instanced_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote 3d wireframe instanced pipeline"),
            .layout = pipeline_layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_instanced"),
                .buffer_count = instanced_vbufs.len,
                .buffers = &instanced_vbufs,
            },
            .primitive = .{
                .topology = .line_list,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = opaque_targets.len,
                .targets = &opaque_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        const shadow_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote shadow shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.shadow_shader_source),
            }),
        }) orelse return error.ShaderCreateFailed;
        defer shadow_shader.release();

        // Shadow map: a depth-2d-array of NUM_CASCADES layers, one per directional cascade. Each
        // layer is rendered in its own pass (per-cascade light view); the mesh shader samples the
        // array, picking the tightest cascade that contains the surface and softening with PCF.
        // Allocate only the directional cascade layers up front; spot layers are added on demand by
        // `ensureShadowCapacity` when the scene actually has shadow-casting spot lights.
        const shadow_tex = device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("shadow map"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = SHADOW_MAP_SIZE, .height = SHADOW_MAP_SIZE, .depth_or_array_layers = NUM_CASCADES },
            .format = .depth32_float,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.DepthTextureCreateFailed;
        const shadow_view = shadow_tex.createView(&.{
            .dimension = .@"2d_array",
            .base_array_layer = 0,
            .array_layer_count = NUM_CASCADES,
            .aspect = .depth_only,
        }) orelse return error.DepthViewCreateFailed;
        var shadow_layer_views: [TOTAL_SHADOW_LAYERS]*wgpu.TextureView = undefined;
        for (0..NUM_CASCADES) |layer| {
            shadow_layer_views[layer] = shadow_tex.createView(&.{
                .dimension = .@"2d",
                .base_array_layer = @intCast(layer),
                .array_layer_count = 1,
                .aspect = .depth_only,
            }) orelse return error.DepthViewCreateFailed;
        }
        const shadow_sampler = device.createSampler(&.{
            .compare = .less,
            .mag_filter = .linear,
            .min_filter = .linear,
        }) orelse return error.WgpuSamplerUnavailable;

        // Point-light shadows: a depth cube-array (MAX_POINT_SHADOWS cubes × 6 faces). The point
        // shadow pass writes linear distance-to-light per face; the mesh shader samples by direction.
        // Start as a 1×1×6 placeholder (one cube) so the cube-array binding is valid; grown to
        // POINT_SHADOW_SIZE × (6 × N) by `ensureShadowCapacity` when N point lights actually cast shadows.
        const point_shadow_tex = device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("point shadow cube array"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = 1, .height = 1, .depth_or_array_layers = 6 },
            .format = .depth32_float,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.DepthTextureCreateFailed;
        const point_shadow_view = point_shadow_tex.createView(&.{
            .dimension = POINT_SHADOW_VIEW_DIM,
            .base_array_layer = 0,
            .array_layer_count = 6,
            .aspect = .depth_only,
        }) orelse return error.DepthViewCreateFailed;
        var point_shadow_face_views: [POINT_CUBE_LAYERS]*wgpu.TextureView = undefined;
        for (0..6) |layer| {
            point_shadow_face_views[layer] = point_shadow_tex.createView(&.{
                .dimension = .@"2d",
                .base_array_layer = @intCast(layer),
                .array_layer_count = 1,
                .aspect = .depth_only,
            }) orelse return error.DepthViewCreateFailed;
        }

        // ── Environment IBL cubemap ──────────────────────────────────────────────────────
        const env_cube_texture = device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("env cubemap"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = ENV_SIZE, .height = ENV_SIZE, .depth_or_array_layers = 6 },
            .format = ENV_CUBE_FORMAT,
            .mip_level_count = ENV_MIPS,
            .sample_count = 1,
        }) orelse return error.DepthTextureCreateFailed;
        const env_cube_view = env_cube_texture.createView(&.{
            .format = ENV_CUBE_FORMAT,
            .dimension = .cube,
            .base_mip_level = 0,
            .mip_level_count = ENV_MIPS,
            .base_array_layer = 0,
            .array_layer_count = 6,
        }) orelse return error.DepthViewCreateFailed;
        var env_face_views: [ENV_MIPS][6]*wgpu.TextureView = undefined;
        for (0..ENV_MIPS) |mip| {
            for (0..6) |face| {
                env_face_views[mip][face] = env_cube_texture.createView(&.{
                    .format = ENV_CUBE_FORMAT,
                    .dimension = .@"2d",
                    .base_mip_level = @intCast(mip),
                    .mip_level_count = 1,
                    .base_array_layer = @intCast(face),
                    .array_layer_count = 1,
                }) orelse return error.DepthViewCreateFailed;
            }
        }
        const env_sampler = device.createSampler(&.{
            .address_mode_u = .clamp_to_edge,
            .address_mode_v = .clamp_to_edge,
            .address_mode_w = .clamp_to_edge,
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_filter = .linear,
        }) orelse return error.WgpuSamplerUnavailable;

        // 1×1 dummy panorama; replaced when an HDRI is supplied. Procedural mode ignores it.
        const env_equirect = try createSolidTextureView(device, queue, .{ 128, 128, 128, 255 }, false);

        const env_bake_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .has_dynamic_offset = 1, .min_binding_size = @sizeOf(EnvBakeParams) } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
        };
        const env_bake_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("env bake bgl"),
            .entry_count = env_bake_bgl_entries.len,
            .entries = &env_bake_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;

        const env_bake_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("env bake ubo"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @as(u64, ENV_SLOT) * 6 * ENV_MIPS,
        }) orelse return error.BufferCreateFailed;

        const env_bake_bg_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .buffer = env_bake_buf, .size = @sizeOf(EnvBakeParams) },
            .{ .binding = 1, .texture_view = env_equirect.texture_view },
            .{ .binding = 2, .sampler = env_sampler },
        };
        const env_bake_bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("env bake bg"),
            .layout = env_bake_bgl,
            .entry_count = env_bake_bg_entries.len,
            .entries = &env_bake_bg_entries,
        }) orelse return error.BindGroupCreateFailed;

        const env_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("env bake shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.env_bake_shader_source),
            }),
        }) orelse return error.ShaderCreateFailed;
        defer env_shader.release();
        const env_bake_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("env bake layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&env_bake_bgl),
        }) orelse return error.PipelineLayoutFailed;
        defer env_bake_layout.release();
        const env_color_target = wgpu.ColorTargetState{ .format = ENV_CUBE_FORMAT, .write_mask = wgpu.ColorWriteMasks.all };
        const env_bake_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("env bake pipeline"),
            .layout = env_bake_layout,
            .vertex = .{ .module = env_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list, .front_face = .ccw, .cull_mode = .none },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = env_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&env_color_target),
            },
        }) orelse return error.PipelineCreateFailed;

        // 1×1 placeholder for the refraction source until ensurePostTargets makes the real frame-sized
        // copy and rebuilds this bind group. (Binding 5 must reference a valid view at init.)
        const refraction_placeholder = try createSolidTextureView(device, queue, .{ 0, 0, 0, 255 }, false);
        const shadow_map_bg_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .texture_view = shadow_view },
            .{ .binding = 1, .sampler = shadow_sampler },
            .{ .binding = 2, .texture_view = env_cube_view },
            .{ .binding = 3, .sampler = env_sampler },
            .{ .binding = 4, .texture_view = point_shadow_view },
            .{ .binding = 5, .texture_view = refraction_placeholder.texture_view },
            .{ .binding = 6, .texture_view = refraction_placeholder.texture_view },
        };
        const shadow_map_bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("shadow map bg"),
            .layout = shadow_map_bgl,
            .entry_count = shadow_map_bg_entries.len,
            .entries = &shadow_map_bg_entries,
        }) orelse return error.BindGroupCreateFailed;
        // Glass-pass variant (same layout); ensurePostTargets points its binding 6 at the real G-buffer.
        const shadow_map_bg_glass = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("shadow map bg (glass)"),
            .layout = shadow_map_bgl,
            .entry_count = shadow_map_bg_entries.len,
            .entries = &shadow_map_bg_entries,
        }) orelse return error.BindGroupCreateFailed;

        // Per-cascade shadow matrix UBO (dynamic offset, one SHADOW_SLOT per cascade). Bound at
        // group 2 of the shadow pipeline; the shadow pass selects the cascade by its dynamic offset.
        const shadow_cascade_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.vertex, .buffer = .{ .type = .uniform, .has_dynamic_offset = 1, .min_binding_size = @sizeOf(uniforms.ShadowCascadeParams) } },
        };
        const shadow_cascade_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("shadow cascade bgl"),
            .entry_count = shadow_cascade_bgl_entries.len,
            .entries = &shadow_cascade_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;
        const shadow_cascade_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("shadow cascade ubo"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @as(u64, SHADOW_SLOT) * TOTAL_SHADOW_LAYERS,
        }) orelse return error.BufferCreateFailed;
        const shadow_cascade_bg_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .buffer = shadow_cascade_buf, .size = @sizeOf(uniforms.ShadowCascadeParams) },
        };
        const shadow_cascade_bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("shadow cascade bg"),
            .layout = shadow_cascade_bgl,
            .entry_count = shadow_cascade_bg_entries.len,
            .entries = &shadow_cascade_bg_entries,
        }) orelse return error.BindGroupCreateFailed;

        // Shadow pipeline. Group 0 (camera/light) stays in the layout for binding-slot parity with
        // the geometry passes but is unused by the shadow shader; group 1 = model, group 2 = cascade.
        const shadow_bgls = [_]*wgpu.BindGroupLayout{ camera_bgl, model_bgl, shadow_cascade_bgl };
        const shadow_pipeline_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("shadow pipeline layout"),
            .bind_group_layout_count = shadow_bgls.len,
            .bind_group_layouts = &shadow_bgls,
        }) orelse return error.PipelineLayoutFailed;
        defer shadow_pipeline_layout.release();

        const shadow_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("shadow pipeline"),
            .layout = shadow_pipeline_layout,
            .vertex = .{
                .module = shadow_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .back,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
        }) orelse return error.PipelineCreateFailed;

        // Alpha-tested shadow pipeline: same camera/model/cascade groups + the material texture group (3),
        // a fragment stage that discards below the cutoff, and NO back-face cull (masked foliage is
        // usually two-sided). Depth-only (no colour targets). Used for masked casters in the cascade/spot
        // shadow slices; the point-cube pass keeps skipping masked casters (its own frag_depth shader).
        const no_color_targets = [_]wgpu.ColorTargetState{}; // depth-only fragment (shadow alpha + point)
        const shadow_alpha_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote shadow alpha shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.shadow_alpha_shader_source),
            }),
        }) orelse return error.ShaderCreateFailed;
        defer shadow_alpha_shader.release();
        const shadow_alpha_bgls = [_]*wgpu.BindGroupLayout{ camera_bgl, model_bgl, shadow_cascade_bgl, texture_bgl };
        const shadow_alpha_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("shadow alpha pipeline layout"),
            .bind_group_layout_count = shadow_alpha_bgls.len,
            .bind_group_layouts = &shadow_alpha_bgls,
        }) orelse return error.PipelineLayoutFailed;
        defer shadow_alpha_pl_layout.release();
        const shadow_alpha_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("shadow alpha pipeline"),
            .layout = shadow_alpha_pl_layout,
            .vertex = .{
                .module = shadow_alpha_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .none,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shadow_alpha_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 0,
                .targets = &no_color_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // ── Point-light shadow pipeline (omnidirectional depth, linear distance via frag_depth) ──
        const point_shadow_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.vertex | wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .has_dynamic_offset = 1, .min_binding_size = @sizeOf(uniforms.PointShadowParams) } },
        };
        const point_shadow_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("point shadow bgl"),
            .entry_count = point_shadow_bgl_entries.len,
            .entries = &point_shadow_bgl_entries,
        }) orelse return error.BindGroupLayoutFailed;
        const point_shadow_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("point shadow ubo"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @as(u64, SHADOW_SLOT) * POINT_CUBE_LAYERS,
        }) orelse return error.BufferCreateFailed;
        const point_shadow_bg_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .buffer = point_shadow_buf, .size = @sizeOf(uniforms.PointShadowParams) },
        };
        const point_shadow_bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("point shadow bg"),
            .layout = point_shadow_bgl,
            .entry_count = point_shadow_bg_entries.len,
            .entries = &point_shadow_bg_entries,
        }) orelse return error.BindGroupCreateFailed;
        const point_shadow_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote point shadow shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.point_shadow_shader_source),
            }),
        }) orelse return error.ShaderCreateFailed;
        defer point_shadow_shader.release();
        const point_shadow_bgls = [_]*wgpu.BindGroupLayout{ camera_bgl, model_bgl, point_shadow_bgl };
        const point_shadow_pipeline_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("point shadow pipeline layout"),
            .bind_group_layout_count = point_shadow_bgls.len,
            .bind_group_layouts = &point_shadow_bgls,
        }) orelse return error.PipelineLayoutFailed;
        defer point_shadow_pipeline_layout.release();
        const point_shadow_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("point shadow pipeline"),
            .layout = point_shadow_pipeline_layout,
            .vertex = .{
                .module = point_shadow_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buf_layout),
            },
            .primitive = .{
                .topology = .triangle_list,
                .front_face = .ccw,
                .cull_mode = .back,
            },
            .depth_stencil = &depth_stencil_state,
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            // Fragment writes only @builtin(frag_depth) (linear distance); no colour targets.
            .fragment = &.{
                .module = point_shadow_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 0,
                .targets = &no_color_targets,
            },
        }) orelse return error.PipelineCreateFailed;

        // Sky gradient pipeline (fullscreen triangle, no vertex buffer, no depth)
        const sky_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote sky shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.sky_shader_source),
            }),
        }) orelse return error.ShaderCreateFailed;
        defer sky_shader.release();

        // Sky reads the camera+light UBO (group 0) and the environment cubemap (group 1, reusing
        // shadow_map_bgl which carries env_cube at binding 2 / env_samp at binding 3) so the visible
        // backdrop IS the lit environment — consistent with the reflections on the car.
        const sky_bgls = [_]*wgpu.BindGroupLayout{ camera_bgl, shadow_map_bgl };
        const sky_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("sky pipeline layout"),
            .bind_group_layout_count = sky_bgls.len,
            .bind_group_layouts = &sky_bgls,
        }) orelse return error.PipelineLayoutFailed;
        defer sky_layout.release();

        const sky_color_target = wgpu.ColorTargetState{
            .format = SCENE_HDR_FORMAT,
            .write_mask = wgpu.ColorWriteMasks.all,
        };
        const empty_vbl: [0]wgpu.VertexBufferLayout = .{};
        const sky_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote sky pipeline"),
            .layout = sky_layout,
            .vertex = .{
                .module = sky_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_sky"),
                .buffer_count = 0,
                .buffers = &empty_vbl,
            },
            .primitive = .{ .topology = .triangle_list },
            .depth_stencil = null,
            .multisample = .{ .count = MSAA_SAMPLES, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = sky_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_sky"),
                .target_count = 1,
                .targets = @ptrCast(&sky_color_target),
            },
        }) orelse return error.PipelineCreateFailed;

        // Camera/Light uniform buffers are already created above

        const default_base_color = try createSolidTextureView(device, queue, .{ 255, 255, 255, 255 }, true);
        const default_normal = try createSolidTextureView(device, queue, .{ 128, 128, 255, 255 }, false);
        // Metallic-roughness default: white (linear) so untextured materials use their
        // factor unchanged (metallic = factor * 1, roughness = factor * 1).
        const default_mr = try createSolidTextureView(device, queue, .{ 255, 255, 255, 255 }, false);

        const default_sampler = device.createSampler(&.{
            .label = wgpu.StringView.fromSlice("zigote default sampler"),
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_filter = .linear,
            // Anisotropic filtering: keeps grazing-angle textures (roads, floors receding to the
            // horizon) sharp instead of blurring to the coarsest mip. wgpu enables aniso when
            // max_anisotropy > 1 and all three filters are linear (they are). 16× matches UE's default.
            .max_anisotropy = 16,
        }) orelse return error.WgpuSamplerUnavailable;

        const default_bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("zigote default material bg"),
            .layout = texture_bgl,
            .entry_count = 8,
            .entries = &[_]wgpu.BindGroupEntry{
                .{ .binding = 0, .texture_view = default_base_color.texture_view },
                .{ .binding = 1, .sampler = default_sampler },
                .{ .binding = 2, .texture_view = default_normal.texture_view },
                .{ .binding = 3, .sampler = default_sampler },
                .{ .binding = 4, .texture_view = default_mr.texture_view },
                .{ .binding = 5, .sampler = default_sampler },
                // Emissive default: the shared 1×1 white — factor passes through untextured.
                .{ .binding = 6, .texture_view = default_base_color.texture_view },
                .{ .binding = 7, .sampler = default_sampler },
            },
        }) orelse return error.BindGroupCreateFailed;

        const default_material = MaterialGpu{
            .bind_group = default_bg,
        };

        // ── Post-processing pipelines (bloom + tonemap) ────────────────────────
        const post_sampler = device.createSampler(&.{
            .label = wgpu.StringView.fromSlice("post sampler"),
            .address_mode_u = .clamp_to_edge,
            .address_mode_v = .clamp_to_edge,
            .address_mode_w = .clamp_to_edge,
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_filter = .nearest,
        }) orelse return error.WgpuSamplerUnavailable;

        // Bloom/SSAO bind group layout: source texture + sampler + params UBO. min_binding_size
        // is 0 (validated by the bound buffer size at draw) so this single layout can back both
        // the bloom params (BloomParams) and the larger SSAO params (SsaoParams).
        const post_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = 0 } },
        };
        const post_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("post bgl"),
            .entry_count = post_bgl_entries.len,
            .entries = &post_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;

        // Tonemap bind group layout: scene HDR + bloom + AO + SSR + params + two G-buffer metadata
        // textures (bindings 9/10 — lighting/scene meta; their slot numbers are kept to avoid a
        // shader renumber after the Metal-RT inputs at 6/7/8 were removed).
        const tonemap_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 4, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 5, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(TonemapParams) } },
            .{ .binding = 9, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 10, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 11, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } }, // albedo G-buffer (SSGI receiver tint)
            .{ .binding = 12, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } }, // 1×1 auto-exposure multiplier
        };
        const tonemap_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("tonemap bgl"),
            .entry_count = tonemap_bgl_entries.len,
            .entries = &tonemap_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;

        // SSR bind group layout: scene colour + view-position + view-normal + sampler + params.
        const ssr_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 4, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(SsrParams) } },
        };
        const ssr_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("ssr bgl"),
            .entry_count = ssr_bgl_entries.len,
            .entries = &ssr_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;

        // SSAO bind group layout: view-position + view-normal G-buffers + sampler + params. The
        // normal comes from the G-buffer (same target SSR reads) instead of cross(dpdx,dpdy).
        const ssao_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(SsaoParams) } },
            // binding 4: the lit scene colour — sampled at the horizon occluders for SSGI (indirect bounce).
            .{ .binding = 4, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            // binding 5: previous accumulated GI/AO (history) — reprojected + blended for temporal SSGI.
            .{ .binding = 5, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
        };
        const ssao_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("ssao bgl"),
            .entry_count = ssao_bgl_entries.len,
            .entries = &ssao_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;

        // DoF bind group layout: scene colour + view-position G-buffer + sampler + params.
        const dof_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(DofParams) } },
        };
        const dof_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("dof bgl"),
            .entry_count = dof_bgl_entries.len,
            .entries = &dof_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;

        // Upsample bind group layout: just the coarser-mip texture + sampler (no UBO).
        const bloom_up_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
        };
        const bloom_up_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("bloom up bgl"),
            .entry_count = bloom_up_bgl_entries.len,
            .entries = &bloom_up_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;

        const bloom_down_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote bloom down shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.bloom_down_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer bloom_down_shader.release();
        const bloom_up_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote bloom up shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.bloom_up_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer bloom_up_shader.release();
        const tonemap_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote tonemap shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.tonemap_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer tonemap_shader.release();

        // Downsample pipeline: replace-blend, reuses post_bgl (texture + sampler + DownParams UBO).
        const bloom_down_target = wgpu.ColorTargetState{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all };
        const bloom_down_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("bloom down pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&post_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer bloom_down_pl_layout.release();
        const bloom_down_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("bloom down pipeline"),
            .layout = bloom_down_pl_layout,
            .vertex = .{ .module = bloom_down_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = bloom_down_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&bloom_down_target),
            },
        }) orelse return error.PipelineCreateFailed;
        // Upsample pipeline: additive (one,one) blend so each tent accumulates into the finer mip.
        const bloom_up_target = wgpu.ColorTargetState{
            .format = SCENE_HDR_FORMAT,
            .write_mask = wgpu.ColorWriteMasks.all,
            .blend = &wgpu.BlendState{
                .color = .{ .src_factor = .one, .dst_factor = .one, .operation = .add },
                .alpha = .{ .src_factor = .one, .dst_factor = .one, .operation = .add },
            },
        };
        const bloom_up_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("bloom up pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&bloom_up_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer bloom_up_pl_layout.release();
        const bloom_up_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("bloom up pipeline"),
            .layout = bloom_up_pl_layout,
            .vertex = .{ .module = bloom_up_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = bloom_up_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&bloom_up_target),
            },
        }) orelse return error.PipelineCreateFailed;

        // Depth-of-field gather pipeline (own bgl: scene + view-position + sampler + DofParams).
        const dof_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote dof shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.dof_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer dof_shader.release();
        const dof_target = wgpu.ColorTargetState{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all };
        const dof_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("dof pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&dof_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer dof_pl_layout.release();
        const dof_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("dof pipeline"),
            .layout = dof_pl_layout,
            .vertex = .{ .module = dof_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = dof_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&dof_target),
            },
        }) orelse return error.PipelineCreateFailed;

        const tonemap_target = wgpu.ColorTargetState{ .format = surface_format, .write_mask = wgpu.ColorWriteMasks.all };
        const tonemap_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("tonemap pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&tonemap_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer tonemap_pl_layout.release();
        const tonemap_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("tonemap pipeline"),
            .layout = tonemap_pl_layout,
            .vertex = .{ .module = tonemap_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = tonemap_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&tonemap_target),
            },
        }) orelse return error.PipelineCreateFailed;

        // Auto-exposure metering pipeline: reads the scene HDR + the 1×1 history + params, writes the
        // 1×1 adapted-multiplier target (SCENE_HDR_FORMAT so the value isn't clamped to [0,1]).
        const exposure_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote exposure shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.exposure_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer exposure_shader.release();
        const exposure_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(ExposureParams) } },
        };
        const exposure_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("exposure bgl"),
            .entry_count = exposure_bgl_entries.len,
            .entries = &exposure_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;
        const exposure_target = wgpu.ColorTargetState{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all };
        const exposure_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("exposure pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&exposure_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer exposure_pl_layout.release();
        const exposure_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("exposure pipeline"),
            .layout = exposure_pl_layout,
            .vertex = .{ .module = exposure_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = exposure_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&exposure_target),
            },
        }) orelse return error.PipelineCreateFailed;
        const exposure_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("exposure params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(ExposureParams),
        }) orelse return error.BufferCreateFailed;

        const bloom_down_first_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("bloom down first params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(BloomDownParams),
        }) orelse return error.BufferCreateFailed;
        const bloom_down_rest_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("bloom down rest params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(BloomDownParams),
        }) orelse return error.BufferCreateFailed;
        const dof_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("dof params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(DofParams),
        }) orelse return error.BufferCreateFailed;
        const tonemap_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("tonemap params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(TonemapParams),
        }) orelse return error.BufferCreateFailed;

        // SSAO pipeline — uses ssao_bgl (pos + normal textures + sampler + uniform); outputs an AO factor.
        const ssao_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote ssao shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.ssao_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer ssao_shader.release();
        const ssao_target = wgpu.ColorTargetState{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all };
        const ssao_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("ssao pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&ssao_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer ssao_pl_layout.release();
        const ssao_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("ssao pipeline"),
            .layout = ssao_pl_layout,
            .vertex = .{ .module = ssao_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = ssao_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&ssao_target),
            },
        }) orelse return error.PipelineCreateFailed;
        const ssao_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("ssao params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(SsaoParams),
        }) orelse return error.BufferCreateFailed;

        // SSR pipeline — reads scene colour + view-position; outputs reflection colour.
        const ssr_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote ssr shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.ssr_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer ssr_shader.release();
        const ssr_target = wgpu.ColorTargetState{ .format = SCENE_HDR_FORMAT, .write_mask = wgpu.ColorWriteMasks.all };
        const ssr_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("ssr pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&ssr_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer ssr_pl_layout.release();
        const ssr_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("ssr pipeline"),
            .layout = ssr_pl_layout,
            .vertex = .{ .module = ssr_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = ssr_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&ssr_target),
            },
        }) orelse return error.PipelineCreateFailed;
        const ssr_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("ssr params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(SsrParams),
        }) orelse return error.BufferCreateFailed;

        // TAA pipeline — current LDR + history LDR + view-position → resolved LDR.
        const taa_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
            .{ .binding = 4, .visibility = wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .min_binding_size = @sizeOf(TaaParams) } },
        };
        const taa_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("taa bgl"),
            .entry_count = taa_bgl_entries.len,
            .entries = &taa_bgl_entries,
        }) orelse return error.BindGroupLayoutCreateFailed;
        const taa_shader = device.createShaderModule(&.{
            .label = wgpu.StringView.fromSlice("zigote taa shader"),
            .next_in_chain = @ptrCast(&wgpu.ShaderSourceWGSL{
                .code = wgpu.StringView.fromSlice(shaders3d.taa_shader_source),
            }),
        }) orelse return error.ShaderModuleCreateFailed;
        defer taa_shader.release();
        const taa_target = wgpu.ColorTargetState{ .format = surface_format, .write_mask = wgpu.ColorWriteMasks.all };
        const taa_pl_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("taa pl layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&taa_bgl),
        }) orelse return error.PipelineLayoutCreateFailed;
        defer taa_pl_layout.release();
        const taa_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("taa pipeline"),
            .layout = taa_pl_layout,
            .vertex = .{ .module = taa_shader, .entry_point = wgpu.StringView.fromSlice("vs_main") },
            .primitive = .{ .topology = .triangle_list },
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = taa_shader,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&taa_target),
            },
        }) orelse return error.PipelineCreateFailed;
        const taa_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("taa params"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @sizeOf(TaaParams),
        }) orelse return error.BufferCreateFailed;

        return .{
            .allocator = allocator,
            .device = device,
            .surface_format = surface_format,
            .mesh_cache = MeshCache.init(device, allocator),
            .instance_pool = InstancePool.init(allocator),
            .bloom_down_pipeline = bloom_down_pipeline,
            .bloom_up_pipeline = bloom_up_pipeline,
            .dof_pipeline = dof_pipeline,
            .tonemap_pipeline = tonemap_pipeline,
            .ssao_pipeline = ssao_pipeline,
            .ssr_pipeline = ssr_pipeline,
            .taa_pipeline = taa_pipeline,
            .exposure_pipeline = exposure_pipeline,
            .exposure_bgl = exposure_bgl,
            .exposure_buf = exposure_buf,
            .post_bgl = post_bgl,
            .ssao_bgl = ssao_bgl,
            .bloom_up_bgl = bloom_up_bgl,
            .dof_bgl = dof_bgl,
            .tonemap_bgl = tonemap_bgl,
            .ssr_bgl = ssr_bgl,
            .taa_bgl = taa_bgl,
            .post_sampler = post_sampler,
            .bloom_down_first_buf = bloom_down_first_buf,
            .bloom_down_rest_buf = bloom_down_rest_buf,
            .dof_buf = dof_buf,
            .tonemap_buf = tonemap_buf,
            .ssao_buf = ssao_buf,
            .ssr_buf = ssr_buf,
            .taa_buf = taa_buf,
            .pipeline = pipeline,
            .pipeline_double_sided = pipeline_double_sided,
            .transparent_pipeline = transparent_pipeline,
            .transparent_pipeline_ds = transparent_pipeline_ds,
            .glass_pipeline = glass_pipeline,
            .glass_pipeline_ds = glass_pipeline_ds,
            .instanced_pipeline = instanced_pipeline,
            .wireframe_pipeline = wireframe_pipeline,
            .wireframe_instanced_pipeline = wireframe_instanced_pipeline,
            .shadow_pipeline = shadow_pipeline,
            .shadow_alpha_pipeline = shadow_alpha_pipeline,
            .sky_pipeline = sky_pipeline,
            .camera_bgl = camera_bgl,
            .model_bgl = model_bgl,
            .texture_bgl = texture_bgl,
            .shadow_map_bgl = shadow_map_bgl,
            .refraction_src_texture = refraction_placeholder.texture,
            .refraction_src_view = refraction_placeholder.texture_view,
            .camera_buf = camera_buf,
            .light_buf = light_buf,
            .camera_bg = camera_bg,
            .shadow_map_bg = shadow_map_bg,
            .shadow_map_bg_glass = shadow_map_bg_glass,
            .shadow_texture = shadow_tex,
            .shadow_view = shadow_view,
            .shadow_layer_views = shadow_layer_views,
            .shadow_sampler = shadow_sampler,
            .shadow_cascade_bgl = shadow_cascade_bgl,
            .shadow_cascade_buf = shadow_cascade_buf,
            .shadow_cascade_bg = shadow_cascade_bg,
            .point_shadow_texture = point_shadow_tex,
            .point_shadow_view = point_shadow_view,
            .point_shadow_face_views = point_shadow_face_views,
            .point_shadow_pipeline = point_shadow_pipeline,
            .point_shadow_bgl = point_shadow_bgl,
            .point_shadow_buf = point_shadow_buf,
            .point_shadow_bg = point_shadow_bg,
            .env_cube_texture = env_cube_texture,
            .env_cube_view = env_cube_view,
            .env_face_views = env_face_views,
            .env_sampler = env_sampler,
            .env_bake_pipeline = env_bake_pipeline,
            .env_bake_bgl = env_bake_bgl,
            .env_bake_buf = env_bake_buf,
            .env_bake_bg = env_bake_bg,
            .env_equirect_tex = env_equirect.texture,
            .env_equirect_view = env_equirect.texture_view,
            .default_base_color = default_base_color,
            .default_normal = default_normal,
            .default_mr = default_mr,
            .default_sampler = default_sampler,
            .default_material = default_material,
        };
    }

    /// Release the per-size mip-chain bloom resources (texture, per-mip views, down/up bind
    /// groups). Shared by deinit and the resize path so neither leaks a TextureView per resize.
    fn releaseBloomTargets(self: *Gpu3d) void {
        for (&self.bloom_mip_views) |*mv| if (mv.*) |v| {
            v.release();
            mv.* = null;
        };
        for (&self.bloom_down_bgs) |*bg| if (bg.*) |b| {
            b.release();
            bg.* = null;
        };
        for (&self.bloom_up_bgs) |*bg| if (bg.*) |b| {
            b.release();
            bg.* = null;
        };
        if (self.bloom_chain_texture) |t| {
            t.release();
            self.bloom_chain_texture = null;
        }
        self.bloom_mip_count = 0;
    }

    /// Free every per-scene GPU resource — mesh geometry (prim_buffers), material textures
    /// (material_gpu_cache), per-entity model UBOs (model_gpu_cache) and instancing buffers
    /// (instance_data) — while keeping the persistent renderer state (pipelines, env, default
    /// textures, render targets). Called by zigote_scene_clear so loading a new scene or switching
    /// projects releases the previous scene's geometry + textures instead of leaking them until app
    /// shutdown. clearRetainingCapacity keeps the maps allocated for immediate reuse.
    pub fn clearScene(self: *Gpu3d) void {
        self.mesh_cache.clear();
        for (self.material_gpu_cache.items) |*slot| if (slot.*) |*g| g.deinit();
        self.material_gpu_cache.clearRetainingCapacity();
        // clear() releases every slot, which auto-calls each value's 0-arg deinit
        // (InstanceGpu) — no manual free loop, no double-free.
        self.instance_pool.clear();
        self.instance_index.clearRetainingCapacity();
    }

    pub fn deinit(self: *Gpu3d) void {
        self.particles.deinit();
        self.sprites.deinit();
        self.mesh_cache.deinit();
        for (self.material_gpu_cache.items) |*slot| if (slot.*) |*g| g.deinit();
        self.material_gpu_cache.deinit(self.allocator);
        // pool.deinit() clears first (auto-deinits live values) then frees storage.
        if (self.model_ring_bg) |bg| bg.release();
        if (self.model_ring) |b| b.release();
        if (self.model_staging.len != 0) self.allocator.free(self.model_staging);
        self.instance_pool.deinit();
        self.instance_index.deinit(self.allocator);
        self.logged_default_mat.deinit(self.allocator);
        self.render_list.deinit(self.allocator);
        self.world3d_list.deinit(self.allocator);
        self.transparent_scratch.deinit(self.allocator);
        self.glass_scratch.deinit(self.allocator);
        self.opaque_scratch.deinit(self.allocator);
        self.shadow_casters_opaque.deinit(self.allocator);
        self.shadow_casters_masked.deinit(self.allocator);
        self.light_scratch.deinit(self.allocator);
        self.instanced_scratch.deinit(self.allocator);
        self.default_base_color.deinit();
        self.default_normal.deinit();
        self.default_mr.deinit();
        self.default_sampler.release();
        if (self.depth_view) |v| v.release();
        if (self.depth_texture) |t| t.release();
        if (self.msaa_color_view) |v| v.release();
        if (self.msaa_color_texture) |t| t.release();
        self.camera_bg.release();
        self.camera_buf.release();
        self.shadow_map_bg.release();
        self.shadow_map_bg_glass.release();
        self.light_buf.release();
        self.shadow_sampler.release();
        for (self.shadow_layer_views[0..self.shadow_layers_alloc]) |v| v.release();
        self.shadow_view.release();
        self.shadow_texture.release();
        self.shadow_cascade_bg.release();
        self.shadow_cascade_buf.release();
        self.shadow_cascade_bgl.release();
        for (self.point_shadow_face_views[0..pointFaceViewCount(self.point_cubes_alloc)]) |v| v.release();
        self.point_shadow_view.release();
        self.point_shadow_texture.release();
        self.point_shadow_bg.release();
        self.point_shadow_buf.release();
        self.point_shadow_bgl.release();
        self.point_shadow_pipeline.release();
        for (0..ENV_MIPS) |mip| {
            for (0..6) |face| self.env_face_views[mip][face].release();
        }
        self.env_cube_view.release();
        self.env_cube_texture.release();
        self.env_sampler.release();
        self.env_bake_bg.release();
        self.env_bake_buf.release();
        self.env_bake_bgl.release();
        self.env_bake_pipeline.release();
        self.env_equirect_view.release();
        self.env_equirect_tex.release();
        self.texture_bgl.release();
        self.model_bgl.release();
        self.camera_bgl.release();
        self.shadow_map_bgl.release();
        self.pipeline.release();
        self.transparent_pipeline.release();
        self.transparent_pipeline_ds.release();
        self.glass_pipeline.release();
        self.glass_pipeline_ds.release();
        self.instanced_pipeline.release();
        self.wireframe_pipeline.release();
        self.wireframe_instanced_pipeline.release();
        self.shadow_pipeline.release();
        self.shadow_alpha_pipeline.release();
        self.sky_pipeline.release();
        if (self.scene_hdr_view) |v| v.release();
        if (self.scene_hdr_texture) |t| t.release();
        if (self.refraction_src_view) |v| v.release();
        if (self.refraction_src_texture) |t| t.release();
        self.releaseDofTargets();
        self.releaseBloomTargets();
        if (self.msaa_pos_view) |v| v.release();
        if (self.msaa_pos_texture) |t| t.release();
        if (self.gbuf_pos_view) |v| v.release();
        if (self.gbuf_pos_texture) |t| t.release();
        if (self.msaa_normal_view) |v| v.release();
        if (self.msaa_normal_texture) |t| t.release();
        if (self.gbuf_normal_view) |v| v.release();
        if (self.gbuf_normal_texture) |t| t.release();
        if (self.msaa_albedo_view) |v| v.release();
        if (self.msaa_albedo_texture) |t| t.release();
        if (self.gbuf_albedo_view) |v| v.release();
        if (self.gbuf_albedo_texture) |t| t.release();
        if (self.ao_view) |v| v.release();
        if (self.ao_texture) |t| t.release();
        if (self.gi_history_view) |v| v.release();
        if (self.gi_history_texture) |t| t.release();
        if (self.exposure_out_view) |v| v.release();
        if (self.exposure_out_texture) |t| t.release();
        if (self.exposure_hist_view) |v| v.release();
        if (self.exposure_hist_texture) |t| t.release();
        if (self.ssr_view) |v| v.release();
        if (self.ssr_texture) |t| t.release();
        if (self.taa_input_view) |v| v.release();
        if (self.taa_input_texture) |t| t.release();
        if (self.taa_output_view) |v| v.release();
        if (self.taa_output_texture) |t| t.release();
        if (self.taa_history_view) |v| v.release();
        if (self.taa_history_texture) |t| t.release();
        if (self.tonemap_bg) |bg| bg.release();
        if (self.ssao_bg) |bg| bg.release();
        if (self.exposure_bg) |bg| bg.release();
        if (self.ssr_bg) |bg| bg.release();
        if (self.taa_bg) |bg| bg.release();
        self.bloom_down_pipeline.release();
        self.bloom_up_pipeline.release();
        self.dof_pipeline.release();
        self.tonemap_pipeline.release();
        self.ssao_pipeline.release();
        self.ssr_pipeline.release();
        self.taa_pipeline.release();
        self.exposure_pipeline.release();
        self.post_bgl.release();
        self.ssao_bgl.release();
        self.bloom_up_bgl.release();
        self.dof_bgl.release();
        self.tonemap_bgl.release();
        self.ssr_bgl.release();
        self.taa_bgl.release();
        self.exposure_bgl.release();
        self.post_sampler.release();
        self.bloom_down_first_buf.release();
        self.bloom_down_rest_buf.release();
        self.dof_buf.release();
        self.tonemap_buf.release();
        self.ssao_buf.release();
        self.ssr_buf.release();
        self.taa_buf.release();
        self.exposure_buf.release();
    }

    /// Ensure the depth texture, MSAA target, HDR scene buffer and bloom targets match the
    /// current framebuffer size, rebuilding the post-processing bind groups on resize.
    fn ensureDepthTexture(self: *Gpu3d, width: u32, height: u32) !void {
        if (self.depth_texture != null and self.depth_width == width and self.depth_height == height) return;
        if (self.depth_view) |v| v.release();
        if (self.depth_texture) |t| t.release();
        if (self.msaa_color_view) |v| v.release();
        if (self.msaa_color_texture) |t| t.release();
        if (self.scene_hdr_view) |v| v.release();
        if (self.scene_hdr_texture) |t| t.release();
        if (self.refraction_src_view) |v| v.release();
        if (self.refraction_src_texture) |t| t.release();
        self.releaseDofTargets();
        self.releaseBloomTargets();
        if (self.msaa_pos_view) |v| v.release();
        if (self.msaa_pos_texture) |t| t.release();
        if (self.gbuf_pos_view) |v| v.release();
        if (self.gbuf_pos_texture) |t| t.release();
        if (self.msaa_normal_view) |v| v.release();
        if (self.msaa_normal_texture) |t| t.release();
        if (self.gbuf_normal_view) |v| v.release();
        if (self.gbuf_normal_texture) |t| t.release();
        if (self.msaa_albedo_view) |v| v.release();
        if (self.msaa_albedo_texture) |t| t.release();
        if (self.gbuf_albedo_view) |v| v.release();
        if (self.gbuf_albedo_texture) |t| t.release();
        if (self.ao_view) |v| v.release();
        if (self.ao_texture) |t| t.release();
        if (self.gi_history_view) |v| v.release();
        if (self.gi_history_texture) |t| t.release();
        if (self.exposure_out_view) |v| v.release();
        if (self.exposure_out_texture) |t| t.release();
        if (self.exposure_hist_view) |v| v.release();
        if (self.exposure_hist_texture) |t| t.release();
        if (self.ssr_view) |v| v.release();
        if (self.ssr_texture) |t| t.release();
        if (self.taa_input_view) |v| v.release();
        if (self.taa_input_texture) |t| t.release();
        if (self.taa_output_view) |v| v.release();
        if (self.taa_output_texture) |t| t.release();
        if (self.taa_history_view) |v| v.release();
        if (self.taa_history_texture) |t| t.release();
        if (self.tonemap_bg) |bg| bg.release();
        if (self.ssao_bg) |bg| bg.release();
        if (self.exposure_bg) |bg| bg.release();
        if (self.ssr_bg) |bg| bg.release();
        if (self.taa_bg) |bg| bg.release();

        // Depth must match the MSAA sample count of the pipelines that use it.
        const tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("depth texture"),
            .usage = wgpu.TextureUsages.render_attachment,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .depth32_float,
            .mip_level_count = 1,
            .sample_count = MSAA_SAMPLES,
        }) orelse return error.DepthTextureCreateFailed;

        const view = tex.createView(&.{
            .label = wgpu.StringView.fromSlice("depth view"),
            .format = .depth32_float,
            .dimension = .@"2d",
            .base_mip_level = 0,
            .mip_level_count = 1,
            .base_array_layer = 0,
            .array_layer_count = 1,
            .aspect = .depth_only,
        }) orelse {
            tex.release();
            return error.DepthViewCreateFailed;
        };

        // Multisampled HDR color target; the scene render passes resolve it into the
        // single-sample HDR scene buffer the post-processing pass consumes.
        const msaa_tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("msaa color texture"),
            .usage = wgpu.TextureUsages.render_attachment,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = MSAA_SAMPLES,
        }) orelse {
            view.release();
            tex.release();
            return error.DepthTextureCreateFailed;
        };
        const msaa_view = msaa_tex.createView(null) orelse {
            msaa_tex.release();
            view.release();
            tex.release();
            return error.DepthViewCreateFailed;
        };

        self.depth_texture = tex;
        self.depth_view = view;
        self.msaa_color_texture = msaa_tex;
        self.msaa_color_view = msaa_view;
        self.depth_width = width;
        self.depth_height = height;

        try self.ensurePostTargets(width, height);
    }

    /// Depth-of-field render targets (dof output + the DoF pass bind group + the DoF tonemap variant),
    /// allocated on demand — only while DoF is enabled — so the common DoF-off case never pays for a
    /// full-res HDR buffer. Idempotent. Requires the post targets it reads (scene_hdr / gbuffers / ao /
    /// ssr / bloom / exposure) to already exist — true once ensurePostTargets has run for this size.
    fn ensureDofTargets(self: *Gpu3d) !void {
        if (self.dof_view != null) return;
        const dof_tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("dof"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = self.depth_width, .height = self.depth_height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const dof_view = dof_tex.createView(null) orelse {
            dof_tex.release();
            return error.SceneHdrViewCreateFailed;
        };
        const dof_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("dof bg"),
            .layout = self.dof_bgl,
            .entry_count = 4,
            .entries = &.{
                .{ .binding = 0, .texture_view = self.scene_hdr_view },
                .{ .binding = 1, .texture_view = self.gbuf_pos_view },
                .{ .binding = 2, .sampler = self.post_sampler },
                .{ .binding = 3, .buffer = self.dof_buf, .size = @sizeOf(DofParams) },
            },
        }) orelse {
            dof_view.release();
            dof_tex.release();
            return error.BindGroupCreateFailed;
        };
        // Same bindings as tonemap_bg but binding 0 = dof_view (the tonemap samples DoF'd colour).
        const bloom_mip1_view = self.bloom_mip_views[1] orelse self.bloom_mip_views[0].?;
        const tonemap_dof_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("tonemap dof bg"),
            .layout = self.tonemap_bgl,
            .entry_count = 10,
            .entries = &.{
                .{ .binding = 0, .texture_view = dof_view },
                .{ .binding = 1, .sampler = self.post_sampler },
                .{ .binding = 2, .texture_view = bloom_mip1_view },
                .{ .binding = 3, .texture_view = self.ao_view },
                .{ .binding = 4, .texture_view = self.ssr_view },
                .{ .binding = 5, .buffer = self.tonemap_buf, .size = @sizeOf(TonemapParams) },
                .{ .binding = 9, .texture_view = self.gbuf_pos_view },
                .{ .binding = 10, .texture_view = self.scene_hdr_view },
                .{ .binding = 11, .texture_view = self.gbuf_albedo_view },
                .{ .binding = 12, .texture_view = self.exposure_out_view },
            },
        }) orelse {
            dof_bg.release();
            dof_view.release();
            dof_tex.release();
            return error.BindGroupCreateFailed;
        };
        self.dof_texture = dof_tex;
        self.dof_view = dof_view;
        self.dof_bg = dof_bg;
        self.tonemap_dof_bg = tonemap_dof_bg;
    }

    fn releaseDofTargets(self: *Gpu3d) void {
        if (self.tonemap_dof_bg) |b| b.release();
        if (self.dof_bg) |b| b.release();
        if (self.dof_view) |v| v.release();
        if (self.dof_texture) |t| t.release();
        self.tonemap_dof_bg = null;
        self.dof_bg = null;
        self.dof_view = null;
        self.dof_texture = null;
    }

    /// (Re)create the single-sample HDR scene buffer + half-res bloom targets and the
    /// post-processing bind groups for a given framebuffer size.
    fn ensurePostTargets(self: *Gpu3d, width: u32, height: u32) !void {
        const hdr_usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding;

        // Optional half-resolution SSAO/SSGI/SSR (ZIGOTE_POST_SCALE=2). Default 1 → these targets
        // are frame-sized, so the render is unchanged. At 2 the three effect passes run at ¼ the
        // pixels; SSAO/SSR are UV-space (resolution-independent) and the tonemap upsamples their
        // outputs with its linear sampler. The full-res G-buffers they SAMPLE are unaffected.
        const post_scale: u32 = blk: {
            if (std.c.getenv("ZIGOTE_POST_SCALE")) |v| {
                if (std.mem.eql(u8, std.mem.span(v), "2")) break :blk 2;
            }
            break :blk 1;
        };
        const post_w = @max(1, (width + post_scale - 1) / post_scale);
        const post_h = @max(1, (height + post_scale - 1) / post_scale);
        self.post_width = post_w;
        self.post_height = post_h;

        const scene_hdr = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("scene hdr"),
            // copy_src so the opaque result can be copied to refraction_src for the glass pass.
            .usage = hdr_usage | wgpu.TextureUsages.copy_src,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const scene_hdr_view = scene_hdr.createView(null) orelse {
            scene_hdr.release();
            return error.SceneHdrViewCreateFailed;
        };

        // Refraction source: a sampleable copy of the opaque scene_hdr (copy_dst). The glass pass copies
        // scene_hdr → here, then samples it (shadow_map_bg binding 5) for screen-space refraction.
        const refraction_src = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("refraction src"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const refraction_src_view = refraction_src.createView(null) orelse return error.SceneHdrViewCreateFailed;
        self.refraction_src_texture = refraction_src;
        self.refraction_src_view = refraction_src_view;
        // shadow_map_bg is rebuilt at the END of this function (it also references the G-buffer position
        // created below, for thickness-aware glass refraction).

        // Depth-of-field targets (dof output + dof_bg + tonemap_dof_bg) are allocated on demand by
        // `ensureDofTargets` at the end of this function — only when DoF is actually enabled — so the
        // common DoF-off case never pays for the full-res dof buffer. See the DoF section below.

        // Mip-chain bloom: one rgba16float texture with per-mip single-level views. mip0 is full-res
        // (allocated, unused); the chain is built scene_hdr → mip1 → … → mip(n-1) then tent-upsampled
        // back to mip1, which the tonemap reads. Clamp the mip count so the smallest mip stays ≳16px.
        self.releaseBloomTargets();
        // The two tonemap bind groups sample bloom mip1, so the chain MUST have >= 2 levels. Floor the
        // chain texture to 2px so a 2-level chain is physically creatable on a tiny/collapsed viewport
        // (a 2x2 texture has exactly 2 mips); real viewports are unaffected (mip count already > 2).
        // Fixes the latent null-unwrap of bloom_mip_views[1] when @min(width,height) <= 16.
        const bloom_w = @max(2, width);
        const bloom_h = @max(2, height);
        var max_mip: u32 = 1;
        {
            var d = @min(bloom_w, bloom_h);
            while (d > 16 and max_mip < BLOOM_MIPS) : (max_mip += 1) d /= 2;
        }
        max_mip = @max(2, max_mip);
        const bloom_chain = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("bloom chain"),
            .usage = hdr_usage,
            .dimension = .@"2d",
            .size = .{ .width = bloom_w, .height = bloom_h, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = max_mip,
            .sample_count = 1,
        }) orelse return error.BloomCreateFailed;
        {
            var i: u32 = 0;
            while (i < max_mip) : (i += 1) {
                self.bloom_mip_views[i] = bloom_chain.createView(&.{
                    .format = SCENE_HDR_FORMAT,
                    .dimension = .@"2d",
                    .base_mip_level = i,
                    .mip_level_count = 1,
                    .base_array_layer = 0,
                    .array_layer_count = 1,
                    .aspect = .all,
                }) orelse return error.BloomViewCreateFailed;
            }
        }
        self.bloom_chain_texture = bloom_chain;
        self.bloom_mip_count = max_mip;

        // SSAO G-buffer: MSAA view-position target resolving into a single-sample buffer,
        // plus the AO result the tonemap pass consumes.
        const msaa_pos = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("msaa view-pos"),
            .usage = wgpu.TextureUsages.render_attachment,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = MSAA_SAMPLES,
        }) orelse return error.SceneHdrCreateFailed;
        const msaa_pos_view = msaa_pos.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const gbuf_pos = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("gbuf view-pos"),
            .usage = hdr_usage,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const gbuf_pos_view = gbuf_pos.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const msaa_normal = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("msaa view-normal"),
            .usage = wgpu.TextureUsages.render_attachment,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = MSAA_SAMPLES,
        }) orelse return error.SceneHdrCreateFailed;
        const msaa_normal_view = msaa_normal.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const gbuf_normal = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("gbuf view-normal"),
            .usage = hdr_usage,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const gbuf_normal_view = gbuf_normal.createView(null) orelse return error.SceneHdrViewCreateFailed;
        // Albedo G-buffer (MRT location 3): base colour, for receiver-albedo tinting of SSGI.
        const msaa_albedo = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("msaa albedo"),
            .usage = wgpu.TextureUsages.render_attachment,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = MSAA_SAMPLES,
        }) orelse return error.SceneHdrCreateFailed;
        const msaa_albedo_view = msaa_albedo.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const gbuf_albedo = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("gbuf albedo"),
            .usage = hdr_usage,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const gbuf_albedo_view = gbuf_albedo.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const ao_tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("ssao"),
            // copy_src: the accumulated SSGI/AO is copied to gi_history each frame for temporal reuse.
            .usage = hdr_usage | wgpu.TextureUsages.storage_binding | wgpu.TextureUsages.copy_src,
            .dimension = .@"2d",
            .size = .{ .width = post_w, .height = post_h, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const ao_view = ao_tex.createView(null) orelse return error.SceneHdrViewCreateFailed;
        // GI/AO temporal history — copy target + sampled by the SSAO pass next frame.
        const gi_history = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("gi history"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = post_w, .height = post_h, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const gi_history_view = gi_history.createView(null) orelse return error.SceneHdrViewCreateFailed;
        // Auto-exposure: 1×1 adapted-multiplier output (render target + sampled by tonemap + copied to
        // history) and its 1×1 history (read by next frame's metering pass). Resolution-independent.
        const exposure_out = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("exposure out"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_src,
            .dimension = .@"2d",
            .size = .{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const exposure_out_view = exposure_out.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const exposure_hist = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("exposure hist"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const exposure_hist_view = exposure_hist.createView(null) orelse return error.SceneHdrViewCreateFailed;
        self.exposure_valid = false; // metering history is stale after a resize
        const ssr_tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("ssr"),
            .usage = hdr_usage | wgpu.TextureUsages.storage_binding,
            .dimension = .@"2d",
            .size = .{ .width = post_w, .height = post_h, .depth_or_array_layers = 1 },
            .format = SCENE_HDR_FORMAT,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const ssr_view = ssr_tex.createView(null) orelse return error.SceneHdrViewCreateFailed;
        // TAA input (tonemap target when TAA is on) + accumulated history, both LDR.
        const taa_input = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("taa input"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = self.surface_format,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const taa_input_view = taa_input.createView(null) orelse return error.SceneHdrViewCreateFailed;
        // Renderer-owned resolve target: the TAA pass writes here, then we copy it to both the
        // history and the destination. copy_src lets it feed both copies; render_attachment lets
        // the resolve pass draw into it.
        const taa_output = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("taa output"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.copy_src,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = self.surface_format,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const taa_output_view = taa_output.createView(null) orelse return error.SceneHdrViewCreateFailed;
        const taa_history = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("taa history"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = self.surface_format,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.SceneHdrCreateFailed;
        const taa_history_view = taa_history.createView(null) orelse return error.SceneHdrViewCreateFailed;
        self.taa_valid = false; // history is stale after a resize

        // SSAO: reads the view-position + view-normal G-buffers.
        const ssao_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("ssao bg"),
            .layout = self.ssao_bgl,
            .entry_count = 6,
            .entries = &.{
                .{ .binding = 0, .texture_view = gbuf_pos_view },
                .{ .binding = 1, .texture_view = gbuf_normal_view },
                .{ .binding = 2, .sampler = self.post_sampler },
                .{ .binding = 3, .buffer = self.ssao_buf, .size = @sizeOf(SsaoParams) },
                .{ .binding = 4, .texture_view = scene_hdr_view },
                .{ .binding = 5, .texture_view = gi_history_view },
            },
        }) orelse return error.BindGroupCreateFailed;

        // SSR: reads the HDR scene colour + view-position + view-normal G-buffers.
        const ssr_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("ssr bg"),
            .layout = self.ssr_bgl,
            .entry_count = 5,
            .entries = &.{
                .{ .binding = 0, .texture_view = scene_hdr_view },
                .{ .binding = 1, .texture_view = gbuf_pos_view },
                .{ .binding = 2, .texture_view = gbuf_normal_view },
                .{ .binding = 3, .sampler = self.post_sampler },
                .{ .binding = 4, .buffer = self.ssr_buf, .size = @sizeOf(SsrParams) },
            },
        }) orelse return error.BindGroupCreateFailed;

        // TAA: current LDR + history LDR + view-position G-buffer.
        const taa_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("taa bg"),
            .layout = self.taa_bgl,
            .entry_count = 5,
            .entries = &.{
                .{ .binding = 0, .texture_view = taa_input_view },
                .{ .binding = 1, .texture_view = taa_history_view },
                .{ .binding = 2, .texture_view = gbuf_pos_view },
                .{ .binding = 3, .sampler = self.post_sampler },
                .{ .binding = 4, .buffer = self.taa_buf, .size = @sizeOf(TaaParams) },
            },
        }) orelse return error.BindGroupCreateFailed;

        // Mip-chain bloom bind groups. Downsample bg[i] feeds the pass writing mip i: source is
        // scene_hdr for i=1 (with the prefilter/Karis "first" UBO), else mip i-1 (plain UBO).
        {
            var i: u32 = 1;
            while (i < self.bloom_mip_count) : (i += 1) {
                const src_view = if (i == 1) scene_hdr_view else self.bloom_mip_views[i - 1].?;
                const ubo = if (i == 1) self.bloom_down_first_buf else self.bloom_down_rest_buf;
                self.bloom_down_bgs[i] = self.device.createBindGroup(&.{
                    .label = wgpu.StringView.fromSlice("bloom down bg"),
                    .layout = self.post_bgl,
                    .entry_count = 3,
                    .entries = &.{
                        .{ .binding = 0, .texture_view = src_view },
                        .{ .binding = 1, .sampler = self.post_sampler },
                        .{ .binding = 2, .buffer = ubo, .size = @sizeOf(BloomDownParams) },
                    },
                }) orelse return error.BindGroupCreateFailed;
            }
        }
        // Upsample bg[i] feeds the pass reading mip i and adding into mip i-1 (coarse → mip1).
        {
            var i: u32 = 2;
            while (i < self.bloom_mip_count) : (i += 1) {
                self.bloom_up_bgs[i] = self.device.createBindGroup(&.{
                    .label = wgpu.StringView.fromSlice("bloom up bg"),
                    .layout = self.bloom_up_bgl,
                    .entry_count = 2,
                    .entries = &.{
                        .{ .binding = 0, .texture_view = self.bloom_mip_views[i].? },
                        .{ .binding = 1, .sampler = self.post_sampler },
                    },
                }) orelse return error.BindGroupCreateFailed;
            }
        }
        // Auto-exposure metering: scene HDR + 1×1 history + params → 1×1 adapted multiplier.
        const exposure_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("exposure bg"),
            .layout = self.exposure_bgl,
            .entry_count = 4,
            .entries = &.{
                .{ .binding = 0, .texture_view = scene_hdr_view },
                .{ .binding = 1, .sampler = self.post_sampler },
                .{ .binding = 2, .texture_view = exposure_hist_view },
                .{ .binding = 3, .buffer = self.exposure_buf, .size = @sizeOf(ExposureParams) },
            },
        }) orelse return error.BindGroupCreateFailed;

        // bloom mip1 always exists (max_mip floored to >= 2 above); fall back to mip0 defensively so a
        // future mip-count regression degrades to un-bloomed output instead of a null-unwrap abort.
        const bloom_mip1_view = self.bloom_mip_views[1] orelse self.bloom_mip_views[0].?;
        // tonemap: HDR scene + bloom (chain mip1) + ao + ssr → LDR
        const tonemap_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("tonemap bg"),
            .layout = self.tonemap_bgl,
            .entry_count = 10,
            .entries = &.{
                .{ .binding = 0, .texture_view = scene_hdr_view },
                .{ .binding = 1, .sampler = self.post_sampler },
                .{ .binding = 2, .texture_view = bloom_mip1_view },
                .{ .binding = 3, .texture_view = ao_view },
                .{ .binding = 4, .texture_view = ssr_view },
                .{ .binding = 5, .buffer = self.tonemap_buf, .size = @sizeOf(TonemapParams) },
                .{ .binding = 9, .texture_view = gbuf_pos_view },
                .{ .binding = 10, .texture_view = scene_hdr_view },
                .{ .binding = 11, .texture_view = gbuf_albedo_view },
                .{ .binding = 12, .texture_view = exposure_out_view },
            },
        }) orelse return error.BindGroupCreateFailed;

        self.scene_hdr_texture = scene_hdr;
        self.scene_hdr_view = scene_hdr_view;
        self.msaa_pos_texture = msaa_pos;
        self.msaa_pos_view = msaa_pos_view;
        self.gbuf_pos_texture = gbuf_pos;
        self.gbuf_pos_view = gbuf_pos_view;
        self.msaa_normal_texture = msaa_normal;
        self.msaa_normal_view = msaa_normal_view;
        self.gbuf_normal_texture = gbuf_normal;
        self.gbuf_normal_view = gbuf_normal_view;
        self.msaa_albedo_texture = msaa_albedo;
        self.msaa_albedo_view = msaa_albedo_view;
        self.gbuf_albedo_texture = gbuf_albedo;
        self.gbuf_albedo_view = gbuf_albedo_view;
        self.ao_texture = ao_tex;
        self.ao_view = ao_view;
        self.gi_history_texture = gi_history;
        self.gi_history_view = gi_history_view;
        self.gi_history_valid = false; // fresh (garbage) history after (re)create → skip reproject once
        self.exposure_out_texture = exposure_out;
        self.exposure_out_view = exposure_out_view;
        self.exposure_hist_texture = exposure_hist;
        self.exposure_hist_view = exposure_hist_view;
        self.exposure_bg = exposure_bg;
        self.ssr_texture = ssr_tex;
        self.ssr_view = ssr_view;
        self.taa_input_texture = taa_input;
        self.taa_input_view = taa_input_view;
        self.taa_output_texture = taa_output;
        self.taa_output_view = taa_output_view;
        self.taa_history_texture = taa_history;
        self.taa_history_view = taa_history_view;
        self.tonemap_bg = tonemap_bg;
        self.ssao_bg = ssao_bg;
        self.ssr_bg = ssr_bg;
        self.taa_bg = taa_bg;

        // Rebuild both shadow/env bind groups for the new frame-sized targets. binding 5 = refraction
        // copy (sampled by glass; never a render target, so safe in every pass). binding 6 = the
        // view-position G-buffer, needed ONLY by the glass pass — the opaque pass WRITES that G-buffer,
        // so its variant uses the refraction copy there as a harmless placeholder instead. (Shared with
        // the shadow-capacity grow path, which rebuilds these when the shadow textures change.)
        try self.rebuildShadowMapBindGroups();

        // Depth-of-field targets only when DoF is enabled (off by default). The per-frame reconcile in
        // beginScene keeps this in sync when DoF is toggled at runtime.
        if (self.effectiveSettings().dof_enabled != 0.0) try self.ensureDofTargets();
    }


    fn createTextureViewFromPixels(
        self: *Gpu3d,
        queue: *wgpu.Queue,
        label: []const u8,
        pixels: []const u8,
        width: u32,
        height: u32,
        is_srgb: bool,
    ) !TextureViewGpu {
        // Tightly packed RGBA8 pixels for the current mip level (level 0 = source).
        var cur = try self.allocator.dupe(u8, pixels);
        defer self.allocator.free(cur);
        var cur_w = width;
        var cur_h = height;

        // Cap oversized source textures (this model ships 4K maps). The editor viewport
        // never needs that much, and large textures dominate load time + GPU memory and
        // alias to white at distance. Downscale to fit before building the mip chain.
        const max_texture_dim: u32 = 1024;
        while (cur_w > max_texture_dim or cur_h > max_texture_dim) {
            const nw = @max(@as(u32, 1), cur_w / 2);
            const nh = @max(@as(u32, 1), cur_h / 2);
            const next = try self.allocator.alloc(u8, @as(usize, nw) * nh * 4);
            boxDownsampleRgba8(cur, cur_w, cur_h, next, nw, nh);
            self.allocator.free(cur);
            cur = next;
            cur_w = nw;
            cur_h = nh;
        }

        // Full mip chain so distant/minified surfaces resolve smoothly (avoids the
        // aliased "white speckle" look from sampling a single high-res level).
        const max_dim = @max(cur_w, cur_h);
        const mip_count: u32 = std.math.log2_int(u32, max_dim) + 1;

        const texture = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice(label),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = cur_w, .height = cur_h, .depth_or_array_layers = 1 },
            .format = if (is_srgb) .rgba8_unorm_srgb else .rgba8_unorm,
            .mip_level_count = mip_count,
            .sample_count = 1,
        }) orelse return error.WgpuTextureCreateFailed;
        errdefer texture.release();

        const texture_view = texture.createView(null) orelse return error.WgpuTextureViewUnavailable;
        errdefer texture_view.release();

        var level: u32 = 0;
        while (true) : (level += 1) {
            try self.uploadMipLevel(queue, texture, level, cur_w, cur_h, cur);
            if (level + 1 >= mip_count) break;

            const nw = @max(@as(u32, 1), cur_w / 2);
            const nh = @max(@as(u32, 1), cur_h / 2);
            const next = try self.allocator.alloc(u8, @as(usize, nw) * nh * 4);
            boxDownsampleRgba8(cur, cur_w, cur_h, next, nw, nh);
            self.allocator.free(cur);
            cur = next;
            cur_w = nw;
            cur_h = nh;
        }

        return .{ .texture = texture, .texture_view = texture_view };
    }

    /// Upload one tightly-packed RGBA8 mip level, honouring wgpu's 256-byte row alignment.
    fn uploadMipLevel(
        self: *Gpu3d,
        queue: *wgpu.Queue,
        texture: *wgpu.Texture,
        level: u32,
        width: u32,
        height: u32,
        pixels: []const u8,
    ) !void {
        const bytes_per_row = std.mem.alignForward(usize, @as(usize, width) * 4, 256);
        const upload = try self.allocator.alloc(u8, bytes_per_row * @as(usize, height));
        defer self.allocator.free(upload);
        @memset(upload, 0);

        const src_stride = @as(usize, width) * 4;
        var row: usize = 0;
        while (row < height) : (row += 1) {
            const src_start = row * src_stride;
            const dst_start = row * bytes_per_row;
            @memcpy(upload[dst_start .. dst_start + src_stride], pixels[src_start .. src_start + src_stride]);
        }

        queue.writeTexture(
            &.{ .texture = texture, .mip_level = level, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
            upload.ptr,
            upload.len,
            &.{ .offset = 0, .bytes_per_row = @intCast(bytes_per_row), .rows_per_image = height },
            &.{ .width = width, .height = height, .depth_or_array_layers = 1 },
        );
    }

    fn ensureMaterialGpu(
        self: *Gpu3d,
        queue: *wgpu.Queue,
        material_handle: u32,
        mat: *const resources_mod.Material,
    ) !*MaterialGpu {
        const cache = &self.material_gpu_cache;
        if (material_handle < cache.items.len) {
            if (cache.items[material_handle]) |*cached| return cached;
        }

        var mat_gpu = MaterialGpu{ .bind_group = undefined };

        const base_view = if (mat.base_color_pixels) |pixels| blk: {
            const tv = try self.createTextureViewFromPixels(queue, mat.name, pixels, mat.base_color_width, mat.base_color_height, true);
            mat_gpu.base_color_tex = tv.texture;
            mat_gpu.base_color_view = tv.texture_view;
            break :blk tv.texture_view;
        } else self.default_base_color.texture_view;

        const normal_view = if (mat.normal_pixels) |pixels| blk: {
            const tv = try self.createTextureViewFromPixels(queue, mat.name, pixels, mat.normal_width, mat.normal_height, false);
            mat_gpu.normal_tex = tv.texture;
            mat_gpu.normal_view = tv.texture_view;
            break :blk tv.texture_view;
        } else self.default_normal.texture_view;

        // Metallic-roughness map is linear data (not sRGB).
        const mr_view = if (mat.metallic_roughness_pixels) |pixels| blk: {
            const tv = try self.createTextureViewFromPixels(queue, mat.name, pixels, mat.metallic_roughness_width, mat.metallic_roughness_height, false);
            mat_gpu.mr_tex = tv.texture;
            mat_gpu.mr_view = tv.texture_view;
            break :blk tv.texture_view;
        } else self.default_mr.texture_view;

        // Emissive map is colour (sRGB). Default is the shared 1×1 white so the emissive factor
        // passes through unchanged for untextured materials.
        const emissive_view = if (mat.emissive_pixels) |pixels| blk: {
            const tv = try self.createTextureViewFromPixels(queue, mat.name, pixels, mat.emissive_width, mat.emissive_height, true);
            mat_gpu.emissive_tex = tv.texture;
            mat_gpu.emissive_view = tv.texture_view;
            break :blk tv.texture_view;
        } else self.default_base_color.texture_view;

        mat_gpu.bind_group = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("material bg"),
            .layout = self.texture_bgl,
            .entry_count = 8,
            .entries = &[_]wgpu.BindGroupEntry{
                .{ .binding = 0, .texture_view = base_view },
                .{ .binding = 1, .sampler = self.default_sampler },
                .{ .binding = 2, .texture_view = normal_view },
                .{ .binding = 3, .sampler = self.default_sampler },
                .{ .binding = 4, .texture_view = mr_view },
                .{ .binding = 5, .sampler = self.default_sampler },
                .{ .binding = 6, .texture_view = emissive_view },
                .{ .binding = 7, .sampler = self.default_sampler },
            },
        }) orelse {
            mat_gpu.deinit();
            return error.BindGroupCreateFailed;
        };

        if (cache.items.len <= material_handle)
            try cache.appendNTimes(self.allocator, null, material_handle + 1 - cache.items.len);
        cache.items[material_handle] = mat_gpu;
        return &cache.items[material_handle].?;
    }

    /// Settings actually used for rendering. Identical to the authored `self.settings` unless
    /// diagnostic mode is on, in which case it returns a stable baseline: all stylised/unstable
    /// post effects off, neutral grade, low ambient, neutral sky/studio. `self.settings` itself
    /// is left untouched so the editor reads the user's values back unchanged.
    fn effectiveSettings(self: *const Gpu3d) Settings3D {
        var s = self.settings;
        // Wireframe lines are 1px; TAA jitter+history would ghost/shimmer them. Force it off so the
        // edges stay crisp (this also disables the camera jitter via taaActive()).
        if (s.wireframe != 0.0) s.taa_enabled = 0.0;
        if (s.diagnostic_mode == 0.0) return s;
        s.bloom_intensity = 0.0;
        s.ssr_intensity = 0.0;
        s.ssao_strength = 0.0;
        s.taa_enabled = 0.0;
        s.fog_density = 0.0; // fog would tint the inspection image
        s.auto_exposure_enabled = 0.0; // fixed exposure for a stable inspection baseline
        s.clearcoat = 0.0;
        s.exposure = 1.0;
        s.contrast = 0.0;
        s.saturation = 1.0;
        s.ambient_intensity = 0.15;
        // Neutral, low-contrast sky so reflections/ambient don't tint or blow out the inspection.
        s.sky_horizon = .{ 0.20, 0.20, 0.20 };
        s.sky_zenith = .{ 0.20, 0.20, 0.20 };
        s.sky_ground = .{ 0.10, 0.10, 0.10 };
        s.overhead = 0.0;
        s.horizon_glow = 0.0;
        // Neutralise the photographic look so the diagnostic baseline is a clean inspection image.
        s.agx_look = 0.0;
        s.wb_temperature = 0.0;
        s.wb_tint = 0.0;
        s.vignette_strength = 0.0;
        s.grain_amount = 0.0;
        s.chromatic_aberration = 0.0;
        s.lens_distortion_k1 = 0.0;
        s.lens_distortion_k2 = 0.0;
        return s;
    }

    /// Whether the TAA resolve will actually run this frame. Camera jitter is applied only when
    /// this is true, so non-TAA paths (legacy wrapper, diagnostic mode, missing resources) never
    /// shimmer. Reads effective settings so diagnostic mode (which forces TAA off) also disables
    /// jitter.
    fn taaActive(self: *const Gpu3d) bool {
        return self.effectiveSettings().taa_enabled != 0.0 and
            self.taa_path_supported and
            self.taa_input_view != null and
            self.taa_output_view != null and
            self.taa_history_texture != null and
            self.taa_bg != null;
    }

    fn temporalJitterActive(self: *const Gpu3d) bool {
        return self.taaActive();
    }

    pub fn currentJitter(self: *const Gpu3d) [2]f32 {
        return .{ halton(self.taa_frame, 2) - 0.5, halton(self.taa_frame, 3) - 0.5 };
    }

    /// GPU render-target memory breakdown, in bytes, computed from the live target dimensions +
    /// the allocation constants above. Mirrors exactly what `ensurePostTargets`/`ensureDepthTexture`/
    /// the shadow+env init allocate, so it attributes the device-wide Metal `currentAllocatedSize`
    /// (which also counts meshes/textures/UI/swapchain) to the renderer's own persistent targets.
    /// Diagnostic only — no allocation, no GPU access.
    pub const TargetMem = struct {
        shadow: u64 = 0,
        point_shadow: u64 = 0,
        env: u64 = 0,
        depth_msaa: u64 = 0,
        hdr: u64 = 0,
        gbuffer: u64 = 0,
        post: u64 = 0,
        taa: u64 = 0,
        total: u64 = 0,
    };

    pub fn targetMemoryBytes(self: *const Gpu3d) TargetMem {
        const px: u64 = @as(u64, self.depth_width) * self.depth_height;
        const ppx: u64 = @as(u64, self.post_width) * self.post_height;
        const samples: u64 = MSAA_SAMPLES;
        const hdr_bpp: u64 = 8; // rgba16float
        const surf_bpp: u64 = 4; // bgra8/rgba8 surface format

        const shadow: u64 = @as(u64, SHADOW_MAP_SIZE) * SHADOW_MAP_SIZE * 4 * self.shadow_layers_alloc;
        const point_shadow: u64 = if (self.point_cubes_alloc == 0)
            @as(u64, 1) * 1 * 4 * 6 // 1×1×6 placeholder
        else
            @as(u64, POINT_SHADOW_SIZE) * POINT_SHADOW_SIZE * 4 * (self.point_cubes_alloc * 6);
        var env: u64 = 0;
        {
            var mip: u32 = 0;
            while (mip < ENV_MIPS) : (mip += 1) {
                const s: u64 = @as(u64, ENV_SIZE) >> @as(u5, @intCast(mip));
                env += s * s * hdr_bpp * 6;
            }
        }
        // depth (MSAA depth32) + msaa_color (MSAA hdr)
        const depth_msaa: u64 = px * 4 * samples + px * hdr_bpp * samples;
        // scene_hdr + refraction_src (each full-res hdr) + bloom mip-chain (~4/3 full-res hdr) + dof
        // (full-res hdr, allocated only while DoF is enabled — see ensureDofTargets).
        var hdr: u64 = px * hdr_bpp * 2 + (px * hdr_bpp * 4 / 3);
        if (self.dof_view != null) hdr += px * hdr_bpp;
        // 3 G-buffers (pos/normal/albedo), each an MSAA target + a single-sample resolve
        const gbuffer: u64 = 3 * (px * hdr_bpp * samples + px * hdr_bpp);
        // ao + gi_history + ssr at post resolution (hdr); exposure 1×1 negligible
        const post: u64 = ppx * hdr_bpp * 3;
        // taa input + output + history at full res (surface format)
        const taa: u64 = px * surf_bpp * 3;
        return .{
            .shadow = shadow,
            .point_shadow = point_shadow,
            .env = env,
            .depth_msaa = depth_msaa,
            .hdr = hdr,
            .gbuffer = gbuffer,
            .post = post,
            .taa = taa,
            .total = shadow + point_shadow + env + depth_msaa + hdr + gbuffer + post + taa,
        };
    }

    /// Total GPU bytes held by cached mesh vertex/index/edge buffers (for engine stats).
    pub fn meshBufferBytes(self: *const Gpu3d) u64 {
        _ = self;
        return @import("mesh_cache.zig").gpuMeshBytes();
    }

    fn pointFaceViewCount(cubes: u32) u32 {
        return if (cubes == 0) 6 else cubes * 6;
    }

    /// Rebuild the two combined shadow/env bind groups (g3) from the CURRENT shadow, env, refraction
    /// and G-buffer-position views. Called after a resize (targets changed) OR a shadow-capacity grow
    /// (shadow_view / point_shadow_view changed). Requires refraction_src_view + gbuf_pos_view to be
    /// valid already — ensurePostTargets sets them before any scene renders.
    fn rebuildShadowMapBindGroups(self: *Gpu3d) !void {
        self.shadow_map_bg.release();
        self.shadow_map_bg_glass.release();
        const opaque_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .texture_view = self.shadow_view },
            .{ .binding = 1, .sampler = self.shadow_sampler },
            .{ .binding = 2, .texture_view = self.env_cube_view },
            .{ .binding = 3, .sampler = self.env_sampler },
            .{ .binding = 4, .texture_view = self.point_shadow_view },
            .{ .binding = 5, .texture_view = self.refraction_src_view },
            .{ .binding = 6, .texture_view = self.refraction_src_view }, // dummy — opaque pass never reads it
        };
        const glass_entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .texture_view = self.shadow_view },
            .{ .binding = 1, .sampler = self.shadow_sampler },
            .{ .binding = 2, .texture_view = self.env_cube_view },
            .{ .binding = 3, .sampler = self.env_sampler },
            .{ .binding = 4, .texture_view = self.point_shadow_view },
            .{ .binding = 5, .texture_view = self.refraction_src_view },
            .{ .binding = 6, .texture_view = self.gbuf_pos_view }, // read for thickness-aware refraction
        };
        self.shadow_map_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("shadow map bg"),
            .layout = self.shadow_map_bgl,
            .entry_count = opaque_entries.len,
            .entries = &opaque_entries,
        }) orelse return error.BindGroupCreateFailed;
        self.shadow_map_bg_glass = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("shadow map bg (glass)"),
            .layout = self.shadow_map_bgl,
            .entry_count = glass_entries.len,
            .entries = &glass_entries,
        }) orelse return error.BindGroupCreateFailed;
    }

    /// Grow the shadow textures to fit this frame's active spot/point shadow casters (never shrinks —
    /// high-water mark, like the glyph atlas). Called at the end of beginScene, before the shadow +
    /// geometry passes. No-op in the common case (counts unchanged): a couple of comparisons.
    fn ensureShadowCapacity(self: *Gpu3d) !void {
        const need_dir = @min(NUM_CASCADES + self.active_spot_shadows, TOTAL_SHADOW_LAYERS);
        const need_pts = @min(self.active_point_shadows, MAX_POINT_SHADOWS);
        var rebuild = false;
        if (need_dir > self.shadow_layers_alloc) {
            try self.reallocDirectionalShadow(need_dir);
            rebuild = true;
        }
        if (need_pts > self.point_cubes_alloc) {
            try self.reallocPointShadow(need_pts);
            rebuild = true;
        }
        if (rebuild) try self.rebuildShadowMapBindGroups();
    }

    fn reallocDirectionalShadow(self: *Gpu3d, layers: u32) !void {
        const tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("shadow map"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = SHADOW_MAP_SIZE, .height = SHADOW_MAP_SIZE, .depth_or_array_layers = layers },
            .format = .depth32_float,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.DepthTextureCreateFailed;
        const view = tex.createView(&.{
            .dimension = .@"2d_array",
            .base_array_layer = 0,
            .array_layer_count = layers,
            .aspect = .depth_only,
        }) orelse {
            tex.release();
            return error.DepthViewCreateFailed;
        };
        var new_views: [TOTAL_SHADOW_LAYERS]*wgpu.TextureView = undefined;
        for (0..layers) |layer| {
            new_views[layer] = tex.createView(&.{
                .dimension = .@"2d",
                .base_array_layer = @intCast(layer),
                .array_layer_count = 1,
                .aspect = .depth_only,
            }) orelse {
                for (0..layer) |j| new_views[j].release();
                view.release();
                tex.release();
                return error.DepthViewCreateFailed;
            };
        }
        for (self.shadow_layer_views[0..self.shadow_layers_alloc]) |v| v.release();
        self.shadow_view.release();
        self.shadow_texture.release();
        self.shadow_texture = tex;
        self.shadow_view = view;
        for (0..layers) |i| self.shadow_layer_views[i] = new_views[i];
        self.shadow_layers_alloc = layers;
    }

    fn reallocPointShadow(self: *Gpu3d, cubes: u32) !void {
        const face_count = cubes * 6; // cubes >= 1 here (grown from the 1×1 placeholder)
        const tex = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("point shadow cube array"),
            .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding,
            .dimension = .@"2d",
            .size = .{ .width = POINT_SHADOW_SIZE, .height = POINT_SHADOW_SIZE, .depth_or_array_layers = face_count },
            .format = .depth32_float,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.DepthTextureCreateFailed;
        const view = tex.createView(&.{
            .dimension = POINT_SHADOW_VIEW_DIM,
            .base_array_layer = 0,
            .array_layer_count = face_count,
            .aspect = .depth_only,
        }) orelse {
            tex.release();
            return error.DepthViewCreateFailed;
        };
        var new_views: [POINT_CUBE_LAYERS]*wgpu.TextureView = undefined;
        for (0..face_count) |layer| {
            new_views[layer] = tex.createView(&.{
                .dimension = .@"2d",
                .base_array_layer = @intCast(layer),
                .array_layer_count = 1,
                .aspect = .depth_only,
            }) orelse {
                for (0..layer) |j| new_views[j].release();
                view.release();
                tex.release();
                return error.DepthViewCreateFailed;
            };
        }
        for (self.point_shadow_face_views[0..pointFaceViewCount(self.point_cubes_alloc)]) |v| v.release();
        self.point_shadow_view.release();
        self.point_shadow_texture.release();
        self.point_shadow_texture = tex;
        self.point_shadow_view = view;
        for (0..face_count) |i| self.point_shadow_face_views[i] = new_views[i];
        self.point_cubes_alloc = cubes;
    }

    /// Re-bake the environment cubemap (procedural studio or HDRI) into all 6 faces × ENV_MIPS
    /// roughness levels. Cheap enough to run on demand; gated behind `env_dirty` so it only fires
    /// at startup and when settings or the HDRI change.
    fn rebuildEnvironment(self: *Gpu3d, queue: *wgpu.Queue, encoder: *wgpu.CommandEncoder) void {
        const s = self.effectiveSettings();
        const az = s.sun_azimuth_deg * std.math.pi / 180.0;
        const el = s.sun_elevation_deg * std.math.pi / 180.0;
        const ce = @cos(el);
        const sun = [4]f32{ ce * @cos(az), @sin(el), ce * @sin(az), s.sun_intensity };

        // One params slot per (mip, face). All queue writes land before any pass executes, so
        // each pass must read its own slot through a distinct dynamic offset.
        for (0..ENV_MIPS) |mip| {
            const rough: f32 = if (ENV_MIPS <= 1) 0.0 else @as(f32, @floatFromInt(mip)) / @as(f32, @floatFromInt(ENV_MIPS - 1));
            for (0..6) |face| {
                const params = EnvBakeParams{
                    .face_mode_rough = .{ @floatFromInt(face), @floatFromInt(self.env_mode), rough, 0 },
                    .sky_horizon = .{ s.sky_horizon[0], s.sky_horizon[1], s.sky_horizon[2], 0 },
                    .sky_zenith = .{ s.sky_zenith[0], s.sky_zenith[1], s.sky_zenith[2], 0 },
                    .sky_ground = .{ s.sky_ground[0], s.sky_ground[1], s.sky_ground[2], 0 },
                    .sun_dir = sun,
                    .studio = .{ s.overhead, s.horizon_glow, s.sun_sharpness, s.clearcoat },
                };
                const slot: u64 = (@as(u64, @intCast(mip)) * 6 + @as(u64, @intCast(face))) * ENV_SLOT;
                const bytes = std.mem.asBytes(&params);
                queue.writeBuffer(self.env_bake_buf, slot, bytes.ptr, bytes.len);
            }
        }

        for (0..ENV_MIPS) |mip| {
            for (0..6) |face| {
                const color_attachment = wgpu.ColorAttachment{
                    .view = self.env_face_views[mip][face],
                    .load_op = .clear,
                    .store_op = .store,
                    .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
                };
                const pass_desc = wgpu.RenderPassDescriptor{
                    .color_attachment_count = 1,
                    .color_attachments = @ptrCast(&color_attachment),
                    .depth_stencil_attachment = null,
                };
                const pass = encoder.beginRenderPass(&pass_desc) orelse return;
                pass.setPipeline(self.env_bake_pipeline);
                const offset: u32 = @intCast((mip * 6 + face) * ENV_SLOT);
                pass.setBindGroup(0, self.env_bake_bg, 1, @ptrCast(&offset));
                pass.draw(3, 1, 0, 0);
                pass.end();
                pass.release();
            }
        }
    }

    /// Switch IBL to a prefiltered HDRI panorama. `pixels` is tightly packed RGBA8 (width×height);
    /// it is uploaded as an equirectangular source the bake pass GGX-prefilters into the cubemap.
    pub fn setEnvironmentHdri(self: *Gpu3d, queue: *wgpu.Queue, pixels: []const u8, width: u32, height: u32) !void {
        const tv = try self.createTextureViewFromPixels(queue, "env equirect", pixels, width, height, true);
        self.env_equirect_view.release();
        self.env_equirect_tex.release();
        self.env_equirect_tex = tv.texture;
        self.env_equirect_view = tv.texture_view;

        // Repoint the bake bind group at the new panorama.
        self.env_bake_bg.release();
        const entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .buffer = self.env_bake_buf, .size = @sizeOf(EnvBakeParams) },
            .{ .binding = 1, .texture_view = self.env_equirect_view },
            .{ .binding = 2, .sampler = self.env_sampler },
        };
        self.env_bake_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("env bake bg"),
            .layout = self.env_bake_bgl,
            .entry_count = entries.len,
            .entries = &entries,
        }) orelse return error.BindGroupCreateFailed;
        self.env_mode = 1;
        self.env_dirty = true;
    }

    /// Set a TRUE HDR equirectangular environment from rgba16-float pixels (8 bytes/texel). Used by
    /// the Radiance .hdr loader so bright sources (sun, sky) survive into reflections — unlike the
    /// LDR path which clamps to [0,1]. The bake pass GGX-prefilters it into the cubemap as usual.
    pub fn setEnvironmentHdriFloat(self: *Gpu3d, queue: *wgpu.Queue, pixels: []const u8, width: u32, height: u32) !void {
        const texture = self.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("env equirect (hdr)"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = .rgba16_float,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.WgpuTextureCreateFailed;
        errdefer texture.release();
        const view = texture.createView(null) orelse return error.WgpuTextureViewUnavailable;
        errdefer view.release();

        queue.writeTexture(
            &.{ .texture = texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
            pixels.ptr,
            pixels.len,
            &.{ .offset = 0, .bytes_per_row = width * 8, .rows_per_image = height },
            &.{ .width = width, .height = height, .depth_or_array_layers = 1 },
        );

        self.env_equirect_view.release();
        self.env_equirect_tex.release();
        self.env_equirect_tex = texture;
        self.env_equirect_view = view;

        self.env_bake_bg.release();
        const entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .buffer = self.env_bake_buf, .size = @sizeOf(EnvBakeParams) },
            .{ .binding = 1, .texture_view = self.env_equirect_view },
            .{ .binding = 2, .sampler = self.env_sampler },
        };
        self.env_bake_bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("env bake bg"),
            .layout = self.env_bake_bgl,
            .entry_count = entries.len,
            .entries = &entries,
        }) orelse return error.BindGroupCreateFailed;
        self.env_mode = 1;
        self.env_dirty = true;
    }

    /// Revert IBL to the built-in procedural studio environment.
    pub fn setEnvironmentProcedural(self: *Gpu3d) void {
        self.env_mode = 0;
        self.env_dirty = true;
    }

    /// Define (or clear) the reflection-probe box. Passing all-zero extents disables it, reverting
    /// to the default infinite-distance environment. Cheap — just stores the box for beginScene.
    pub fn setReflectionProbe(self: *Gpu3d, center: [3]f32, extents: [3]f32) void {
        self.probe_center = center;
        self.probe_extents = extents;
    }

    pub fn renderSkyPass(
        self: *Gpu3d,
        encoder: *wgpu.CommandEncoder,
        geometry_will_resolve: bool,
    ) !void {
        self.stats.render_passes += 1;
        // The sky renders into the MSAA colour buffer and is stored (store_op=.store). When the
        // geometry pass runs (active 3D camera present) it LOADs that same MSAA buffer and re-resolves
        // sky+geometry into scene_hdr, so the sky's own full-res 4x MSAA resolve here is dead work —
        // skip it. Only when no geometry pass will resolve (no active camera) does the sky resolve
        // itself, as the fallback that keeps scene_hdr defined.
        const color_attachment = wgpu.ColorAttachment{
            .view = self.msaa_color_view.?,
            .resolve_target = if (geometry_will_resolve) null else self.scene_hdr_view.?,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
        };
        const pass_desc = wgpu.RenderPassDescriptor{
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&color_attachment),
            .depth_stencil_attachment = null,
        };
        const pass = encoder.beginRenderPass(&pass_desc) orelse return error.RenderPassFailed;
        defer pass.release();
        pass.setPipeline(self.sky_pipeline);
        pass.setBindGroup(0, self.camera_bg, 0, null);
        pass.setBindGroup(1, self.shadow_map_bg, 0, null);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    /// Render the 3D scene. Clears the color buffer (so call before UI pass).
    ///
    /// Thin wrapper kept for the legacy `zigote_render_3d` FFI path. The render-graph
    /// path (`ffi/root.zig`) instead drives `beginScene` → `renderShadowPass` →
    /// `renderSkyPass` → `renderSceneGeometry` → `renderPostProcess` as individually-declared
    /// graph passes. The scene renders into the internal HDR buffer; post-processing tonemaps
    /// it into `color_view` (the LDR target the UI samples).
    pub fn render(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        color_view: *wgpu.TextureView,
        frame_width: u32,
        frame_height: u32,
        clear_color: wgpu.Color,
        selected_node_ptr: u64,
    ) !void {
        _ = clear_color;
        // Legacy path has no offscreen destination for the TAA history copy → no resolve, so
        // disable TAA (and thus jitter) for this frame.
        self.taa_path_supported = false;
        try self.beginScene(world, queue, encoder, frame_width, frame_height, selected_node_ptr);
        try self.renderShadowPass(world, queue, encoder);
        try self.renderSkyPass(encoder, world.active_camera != null);
        _ = try self.renderSceneGeometry(world, queue, encoder, frame_width, frame_height);
        self.renderPostProcess(queue, encoder, color_view, null);
    }

    /// Scene-3D upload/prepare step: refreshes the depth/MSAA targets, gathers active
    /// lights, packs the shared light+settings UBO (including the shadow light_view_proj),
    /// and rebakes the environment cubemap when dirty. Must run before the shadow pass —
    /// the shadow vertex shader reads `light.view_proj` from the buffer written here.
    pub fn beginScene(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        frame_width: u32,
        frame_height: u32,
        selected_node_ptr: u64,
    ) !void {
        self.selected_node_ptr = selected_node_ptr;
        self.taa_frame +%= 1;
        // Reset per-frame debug stats (frame_index keeps counting).
        self.stats = .{ .frame_index = self.stats.frame_index +% 1 };
        self.ensureModelRing(); // reset the shared model-UBO cursor + grow the ring if needed
        try self.ensureDepthTexture(frame_width, frame_height);

        self.light_scratch.clearRetainingCapacity();
        try world.collectLights(self.allocator, &self.light_scratch);
        const active_lights = &self.light_scratch;

        // Collect the world-3D renderables once for the whole frame; the shadow and geometry passes
        // both read this instead of each re-walking the scene tree. beginScene is guaranteed to run
        // before either pass (both the legacy render() and the v2 render-graph order it first).
        self.world3d_list.clearRetainingCapacity();
        try world.collectRenderables(self.allocator, .world_3d, &self.world3d_list);

        var light_data = std.mem.zeroes(LightUniforms);
        const s = self.effectiveSettings();
        const diagnostic = s.diagnostic_mode != 0.0;
        light_data.ambient_color = .{ 0.08, 0.10, 0.14, s.ambient_intensity };
        light_data.light_count[0] = @min(16, @as(u32, @intCast(active_lights.items.len)));
        light_data.debug = .{ @intFromFloat(@max(self.settings.debug_view, 0.0)), if (diagnostic) 1 else 0, if (self.settings.wireframe != 0.0) 1 else 0, 0 };

        // Pack editor-tunable render settings into the light UBO (read by the mesh shader).
        light_data.sky_horizon = .{ s.sky_horizon[0], s.sky_horizon[1], s.sky_horizon[2], 0 };
        light_data.sky_zenith = .{ s.sky_zenith[0], s.sky_zenith[1], s.sky_zenith[2], 0 };
        light_data.sky_ground = .{ s.sky_ground[0], s.sky_ground[1], s.sky_ground[2], 0 };
        light_data.env_avg = .{ s.env_avg[0], s.env_avg[1], s.env_avg[2], 0 };
        const az = s.sun_azimuth_deg * std.math.pi / 180.0;
        const el = s.sun_elevation_deg * std.math.pi / 180.0;
        const ce = @cos(el);
        light_data.sun_dir = .{ ce * @cos(az), @sin(el), ce * @sin(az), s.sun_intensity };
        light_data.studio = .{ s.overhead, s.horizon_glow, s.sun_sharpness, s.clearcoat };
        // post.w carries the env cubemap max-LOD so the shader can map roughness → mip level.
        light_data.post = .{ s.exposure, s.contrast, s.saturation, @floatFromInt(ENV_MIPS - 1) };
        light_data.shadow = .{ s.shadow_strength, s.shadow_bias, s.shadow_softness, 0 };
        // Reflection probe: enabled when any extent is positive. Disabled in diagnostic mode so
        // material inspection sees the raw infinite env.
        const probe_on = !diagnostic and (self.probe_extents[0] > 0.0 or self.probe_extents[1] > 0.0 or self.probe_extents[2] > 0.0);
        light_data.probe_center = .{ self.probe_center[0], self.probe_center[1], self.probe_center[2], if (probe_on) 1.0 else 0.0 };
        light_data.probe_extents = .{ self.probe_extents[0], self.probe_extents[1], self.probe_extents[2], 0.0 };
        // Atmospheric fog (height exponential + analytic sun in-scatter). density 0 → shader skips it.
        light_data.fog_color = .{ s.fog_color[0], s.fog_color[1], s.fog_color[2], s.fog_density };
        light_data.fog_params = .{ s.fog_height_falloff, s.fog_height, s.fog_sun_inscatter, s.fog_anisotropy };


        var shadow_dir = (Vec3{ .x = 1.0, .y = 2.0, .z = 1.5 }).normalize();
        var has_shadow_light = false;

        if (diagnostic) {
            // Diagnostic baseline: exactly one directional light, derived from the sun direction,
            // so material/colour/normal bugs read against pure direct + base PBR.
            const d = Vec3{ .x = light_data.sun_dir[0], .y = light_data.sun_dir[1], .z = light_data.sun_dir[2] };
            const dir = if (d.lengthSq() > 0.00001) d.normalize() else shadow_dir;
            light_data.light_count[0] = 1;
            light_data.lights[0] = .{
                .position_or_dir = .{ dir.x, dir.y, dir.z, 0.0 },
                .color_range = .{ 3.0, 3.0, 3.0, 0.0 },
            };
            shadow_dir = dir;
            has_shadow_light = true;
        }

        // Spot shadow-casters claim extra layers after the directional cascades; point shadow-casters
        // claim cubes in the point cube-array. Both capped (MAX_SPOT_SHADOWS / MAX_POINT_SHADOWS).
        var spot_shadow_count: u32 = 0;
        var point_shadow_count: u32 = 0;
        for (active_lights.items, 0..) |al, i| {
            if (diagnostic) break;
            if (i >= 16) break;
            const is_dir = al.light.kind == .directional;
            const is_spot = al.light.kind == .spot;
            const w_pos = al.node.world_transform.position;
            const w_mat = al.node.worldMatrix();
            // -Z is forward in our local space
            const fwd = w_mat.mulDirection(.{ .x = 0, .y = 0, .z = -1 }).normalize();

            if (is_dir and !has_shadow_light) {
                // For directional lights, forward is the direction the light points
                shadow_dir = fwd.scale(-1.0); // lookAt target needs to be opposite
                has_shadow_light = true;
            }

            // position_or_dir.w is the light type: 0 = directional, 1 = point, 2 = spot.
            const type_w: f32 = if (is_dir) 0.0 else if (is_spot) 2.0 else 1.0;
            light_data.lights[i] = .{
                .position_or_dir = if (is_dir)
                    .{ -fwd.x, -fwd.y, -fwd.z, type_w }
                else
                    .{ w_pos.x, w_pos.y, w_pos.z, type_w },
                .color_range = .{
                    al.light.color.x * al.light.intensity,
                    al.light.color.y * al.light.intensity,
                    al.light.color.z * al.light.intensity,
                    al.light.range,
                },
            };

            if (is_spot) {
                const cos_outer = @cos(al.light.outer_angle);
                const cos_inner = @cos(al.light.inner_angle);
                var shadow_layer: f32 = -1.0;
                if (al.light.cast_shadows and spot_shadow_count < MAX_SPOT_SHADOWS) {
                    const slot = spot_shadow_count;
                    const layer = NUM_CASCADES + slot;
                    // Perspective depth from the spot apex along its cone axis (full cone = 2·outer angle).
                    const up = if (@abs(fwd.y) > 0.99) Vec3.forward else Vec3.up;
                    const view = Mat4.lookAt(w_pos, w_pos.add(fwd), up);
                    const fov = @min(@max(al.light.outer_angle * 2.0, 0.2), 3.0);
                    const proj = Mat4.perspectiveRhZo(fov, 1.0, 0.1, @max(al.light.range, 1.0));
                    const spot_vp = proj.mul(view);
                    self.shadow_slice_vp[layer] = spot_vp; // for per-spot caster culling
                    const vp_arr = spot_vp.toArray();
                    light_data.spot_view_proj[slot] = vp_arr;
                    const sp = uniforms.ShadowCascadeParams{ .view_proj = vp_arr };
                    const sb = std.mem.asBytes(&sp);
                    queue.writeBuffer(self.shadow_cascade_buf, @as(u64, SHADOW_SLOT) * @as(u64, layer), sb.ptr, sb.len);
                    shadow_layer = @floatFromInt(layer);
                    spot_shadow_count += 1;
                }
                // spot_cone.z = -1 (spots don't use the point cube path).
                light_data.light_ext[i] = .{
                    .spot_dir = .{ fwd.x, fwd.y, fwd.z, cos_outer },
                    .spot_cone = .{ cos_inner, shadow_layer, -1.0, 0 },
                };
            } else if (!is_dir) {
                // Point light: optional omnidirectional cube shadow (6 faces written here).
                var cube_idx: f32 = -1.0;
                if (al.light.cast_shadows and point_shadow_count < MAX_POINT_SHADOWS) {
                    const ci = point_shadow_count;
                    const range = @max(al.light.range, 1.0);
                    for (0..6) |f| {
                        const view = Mat4.lookAt(w_pos, w_pos.add(CUBE_FACE_DIRS[f]), CUBE_FACE_UPS[f]);
                        const proj = Mat4.perspectiveRhZo(std.math.pi / 2.0, 1.0, 0.1, range);
                        const face_vp = proj.mul(view);
                        self.point_face_vp[ci * 6 + f] = face_vp; // for per-face caster culling
                        const pp = uniforms.PointShadowParams{
                            .view_proj = face_vp.toArray(),
                            .light_pos_range = .{ w_pos.x, w_pos.y, w_pos.z, range },
                        };
                        const pb = std.mem.asBytes(&pp);
                        const slot: u64 = (@as(u64, ci) * 6 + @as(u64, @intCast(f)));
                        queue.writeBuffer(self.point_shadow_buf, @as(u64, SHADOW_SLOT) * slot, pb.ptr, pb.len);
                    }
                    cube_idx = @floatFromInt(ci);
                    point_shadow_count += 1;
                }
                light_data.light_ext[i] = .{
                    .spot_dir = .{ 0, 0, 0, 0 },
                    .spot_cone = .{ 0, -1.0, cube_idx, 0 },
                };
            }
        }
        self.active_spot_shadows = spot_shadow_count;
        self.active_point_shadows = point_shadow_count;
        // Grow the shadow textures (never shrink) to cover this frame's spot/point casters, before the
        // shadow + geometry passes below use them. No-op unless a new caster type just appeared.
        try self.ensureShadowCapacity();

        // Reconcile the on-demand DoF targets with the DoF setting (handles a runtime toggle with no
        // resize). Runs before the post passes; when DoF is off the targets stay freed (dof_active =
        // false in renderPostProcess skips the pass and picks the non-DoF tonemap bind group).
        const dof_want = self.effectiveSettings().dof_enabled != 0.0;
        if (dof_want and self.dof_view == null) {
            try self.ensureDofTargets();
        } else if (!dof_want and self.dof_view != null) {
            self.releaseDofTargets();
        }

        // ── Directional cascaded shadow maps ────────────────────────────────────────────────
        // Build NUM_CASCADES concentric, texel-snapped ortho boxes centred ahead of the camera.
        // Each cascade's light view-proj is uploaded both into the light UBO (for the mesh shader to
        // sample) and into the per-cascade shadow UBO (for the shadow pass to render that layer).
        // Centring the boxes on the camera (rather than the world origin) keeps shadows present as
        // the camera pans; the increasing extents trade near crispness for far range.
        var cam_pos = Vec3.zero;
        var cam_fwd = Vec3.forward;
        if (world.active_camera) |cam| {
            cam_pos = cam.world_transform.position;
            cam_fwd = cam.world_transform.rotation.rotateVec(Vec3.forward);
        }
        light_data.csm_info = .{ @floatFromInt(NUM_CASCADES), @floatFromInt(SHADOW_MAP_SIZE), 0, 0 };
        for (0..NUM_CASCADES) |ci| {
            const e = CASCADE_EXTENTS[ci];
            // Focus the cascade ahead of the camera, proportional to its extent (≈0.75·e reproduces
            // the former single-box SHADOW_FOCUS_DIST=15 at e=20).
            var center = cam_pos.add(cam_fwd.scale(e * 0.75));
            // Texel-snap the centre in light space so projected texels don't crawl as the camera moves.
            const off = e; // eye offset along the light dir (at e=20 this is the former "20").
            const view0 = Mat4.lookAt(center.add(shadow_dir.scale(off)), center, Vec3.up);
            const texel_world = (e * 2.0) / @as(f32, @floatFromInt(SHADOW_MAP_SIZE));
            const center_ls = view0.mulPoint(center);
            const snapped_ls = Vec3{
                .x = @round(center_ls.x / texel_world) * texel_world,
                .y = @round(center_ls.y / texel_world) * texel_world,
                .z = center_ls.z,
            };
            center = view0.inverse().mulPoint(snapped_ls);
            const light_pos = center.add(shadow_dir.scale(off));
            const light_view = Mat4.lookAt(light_pos, center, Vec3.up);
            // Depth range scales with extent (at e=20 → [-20, 50], the former single-box range).
            const light_proj = Mat4.orthographicRhZo(-e, e, -e, e, -e, 2.5 * e);
            const light_vp = light_proj.mul(light_view);
            self.shadow_slice_vp[ci] = light_vp; // for per-cascade caster culling
            const vp_arr = light_vp.toArray();
            light_data.csm_view_proj[ci] = vp_arr;
            if (ci == 0) light_data.view_proj = vp_arr; // back-compat prefix (sky/legacy readers)
            const cascade_params = uniforms.ShadowCascadeParams{ .view_proj = vp_arr };
            const cb = std.mem.asBytes(&cascade_params);
            queue.writeBuffer(self.shadow_cascade_buf, @as(u64, SHADOW_SLOT) * @as(u64, @intCast(ci)), cb.ptr, cb.len);
        }
        const light_bytes = std.mem.asBytes(&light_data);
        queue.writeBuffer(self.light_buf, 0, light_bytes.ptr, light_bytes.len);

        // Diagnostic mode swaps the sky/studio inputs the cubemap bakes from, so a toggle must
        // force a rebake even when the authored settings haven't otherwise changed.
        if (diagnostic != self.env_baked_diagnostic) self.env_dirty = true;
        if (self.env_dirty) {
            self.rebuildEnvironment(queue, encoder);
            self.env_dirty = false;
            self.env_baked_diagnostic = diagnostic;
        }
    }

    /// Opaque + transparent geometry for the world-3D layer (and the optional 2D scene
    /// layer on top). Reads the shadow map produced by `renderShadowPass` and composites
    /// over the sky already in the HDR scene buffer. Returns the number of renderables drawn
    /// so the render graph can report scene-object stats.
    pub fn renderSceneGeometry(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        frame_width: u32,
        frame_height: u32,
    ) !u32 {
        const color_attachment = wgpu.ColorAttachment{
            .view = self.scene_hdr_view.?,
            .load_op = .load,
            .store_op = .store,
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
        };
        const depth_attachment = wgpu.DepthStencilAttachment{
            .view = self.depth_view.?,
            .depth_store_op = .store,
            .depth_clear_value = 1.0,
        };

        // GPU particle simulation: dispatch the compute kernel onto this encoder BEFORE the geometry
        // render pass reads each emitter's (compute-written) instance buffer. wgpu barriers the two passes.
        self.particles.computeStep(self.device, queue, encoder);

        try self.renderLayer(world, queue, encoder, &color_attachment, &depth_attachment, frame_width, frame_height, .world_3d, true);
        var drawn: u32 = @intCast(self.render_list.items.len);
        if (world.active_camera_2d != null) {
            try self.renderLayer(world, queue, encoder, &color_attachment, &depth_attachment, frame_width, frame_height, .scene2d, true);
            drawn += @intCast(self.render_list.items.len);
        }
        self.flushModelRing(queue); // one writeBuffer for every model slot staged this stage

        // 2D sprites, scene stage: painter's-order pass onto the resolved single-sample HDR target
        // (AFTER the MSAA geometry pass, so no re-resolve overwrites it). Skipped when no batches.
        const sprite_draws = self.sprites.renderScene(self.device, queue, encoder, self.scene_hdr_view.?, SCENE_HDR_FORMAT);
        self.stats.draw_calls += sprite_draws;
        if (sprite_draws > 0) self.stats.render_passes += 1;

        self.stats.visible_objects = drawn;
        self.stats.render_passes += 1; // geometry pass (sky/shadow counted in their own fns)
        return drawn;
    }

    /// Post-processing: extract a bloom from the HDR scene buffer (soft-knee bright-pass +
    /// separable gaussian at half resolution), then tonemap+grade the HDR scene with the
    /// bloom added, writing the final LDR image into `dst_view` (the texture the UI samples).
    /// Records three full-screen passes into `encoder`; the caller submits.
    pub fn renderPostProcess(
        self: *Gpu3d,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        dst_view: *wgpu.TextureView,
        dst_texture: ?*wgpu.Texture,
    ) void {
        const s = self.effectiveSettings();

        // Mip-chain bloom downsample params: "first" enables the prefilter + Karis firefly clamp on
        // the brightest mip; "rest" is a plain energy-preserving downsample.
        const down_first = BloomDownParams{ .cfg = .{ 1.0, s.bloom_threshold, s.bloom_knee, 0.0 } };
        const down_rest = BloomDownParams{ .cfg = .{ 0.0, s.bloom_threshold, s.bloom_knee, 0.0 } };
        const auto_exp_on = s.auto_exposure_enabled != 0.0;
        const tp = TonemapParams{
            .cfg = .{ s.exposure, s.contrast, s.saturation, s.bloom_intensity },
            // debug.y = auto-exposure flag: the tonemap multiplies by the metered 1×1 exposure when set.
            .debug = .{ @intFromFloat(@max(self.settings.debug_view, 0.0)), if (auto_exp_on) 1 else 0, 0, 0 },
            .look = .{ s.agx_look, s.vignette_strength, s.vignette_softness, s.grain_amount },
            .wb = .{ s.wb_temperature, s.wb_tint, s.chromatic_aberration, @as(f32, @floatFromInt(self.taa_frame & 0xFFFF)) },
            .lens = .{ s.lens_distortion_k1, s.lens_distortion_k2, 0.0, 0.0 },
        };
        const ep = ExposureParams{
            .cfg = .{ s.auto_exposure_key, s.auto_exposure_min, s.auto_exposure_max, s.auto_exposure_speed },
            .cfg2 = .{ if (self.exposure_valid) 1.0 else 0.0, 0, 0, 0 },
        };
        // Contact shadows march toward the sun in VIEW space. Build the view-space sun direction
        // from the (effective) sun angles via the last world→view matrix. Disabled in diagnostic mode.
        const az = s.sun_azimuth_deg * std.math.pi / 180.0;
        const el = s.sun_elevation_deg * std.math.pi / 180.0;
        const ce = @cos(el);
        const sun_world = Vec3{ .x = ce * @cos(az), .y = @sin(el), .z = ce * @sin(az) };
        const view_mat = Mat4.fromArray(self.last_view);
        const sun_vs = view_mat.mulDirection(sun_world).normalize();
        const contact_strength: f32 = if (s.diagnostic_mode != 0.0) 0.0 else 0.6;
        const sp = SsaoParams{
            .proj = self.last_proj,
            .cfg = .{ s.ssao_radius, s.ssao_bias, s.ssao_strength, s.ssao_power },
            .contact = .{ sun_vs.x, sun_vs.y, sun_vs.z, contact_strength },
            .contact2 = .{ 0.25, 8.0, 0.5, if (s.diagnostic_mode != 0.0) 0.0 else s.ssgi_strength }, // length, steps, thickness, ssgi_strength
            // Temporal SSGI reprojection (same matrices the TAA pass uses). history_valid gates frame 1 /
            // resize; feedback 0.9 accumulates the low-frequency GI heavily.
            .prev_view_proj = self.prev_view_proj,
            .cur_view_proj = self.cur_view_proj,
            .inv_view = self.inv_view,
            .gi = .{ if (self.gi_history_valid) 1.0 else 0.0, 0.9, 0.0, 0.0 },
        };
        const rp = SsrParams{ .proj = self.last_proj, .cfg = .{ s.ssr_intensity, s.ssr_max_distance, s.ssr_thickness, s.ssr_steps } };
        queue.writeBuffer(self.bloom_down_first_buf, 0, std.mem.asBytes(&down_first).ptr, @sizeOf(BloomDownParams));
        queue.writeBuffer(self.bloom_down_rest_buf, 0, std.mem.asBytes(&down_rest).ptr, @sizeOf(BloomDownParams));
        queue.writeBuffer(self.tonemap_buf, 0, std.mem.asBytes(&tp).ptr, @sizeOf(TonemapParams));
        queue.writeBuffer(self.ssao_buf, 0, std.mem.asBytes(&sp).ptr, @sizeOf(SsaoParams));
        queue.writeBuffer(self.ssr_buf, 0, std.mem.asBytes(&rp).ptr, @sizeOf(SsrParams));
        queue.writeBuffer(self.exposure_buf, 0, std.mem.asBytes(&ep).ptr, @sizeOf(ExposureParams));

        // Depth-of-field params. CoC ramps from the loc1 view-space depth vs the focus distance;
        // f-stop scales the ramp (lower = more blur), clamped to max_coc pixels. Off in diagnostic.
        const dof_enabled: f32 = if (s.diagnostic_mode != 0.0) 0.0 else s.dof_enabled;
        const fw: f32 = @floatFromInt(@max(self.depth_width, 1));
        const fh: f32 = @floatFromInt(@max(self.depth_height, 1));
        const coc_scale = s.dof_max_coc * std.math.clamp(2.8 / @max(s.dof_f_stop, 0.1), 0.25, 2.5);
        const dp = DofParams{
            .texel = .{ 1.0 / fw, 1.0 / fh },
            .cfg = .{ s.dof_focus_distance, s.dof_max_coc },
            .cfg2 = .{ coc_scale, 0.0, 1.0, dof_enabled },
            .bokeh = .{ s.bokeh_blades, s.bokeh_anamorphic, 0.0, 0.0 },
        };
        queue.writeBuffer(self.dof_buf, 0, std.mem.asBytes(&dp).ptr, @sizeOf(DofParams));

        const ao_view = self.ao_view orelse return;
        const ssr_view = self.ssr_view orelse return;
        const tonemap_bg = self.tonemap_bg orelse return;
        const ssao_bg = self.ssao_bg orelse return;
        const ssr_bg = self.ssr_bg orelse return;

        // Keep the mature raster effects as the stable baseline. Native RT writes separate
        // auxiliary textures and is blended conservatively by the tonemap pass.
        //
        // SSAO always runs: even at ssao_strength==0 this pass also produces the contact shadows
        // (contact_strength above) the tonemap relies on, so it is NOT gateable on ssao_strength.
        self.fullscreenPass(encoder, ao_view, self.ssao_pipeline, ssao_bg);
        // Temporal SSGI: stash the just-accumulated GI/AO as next frame's history (the SSAO pass above
        // already blended in the previous history when gi_history_valid). Copy AFTER the pass writes ao.
        if (self.ao_texture) |ao_t| {
            if (self.gi_history_texture) |hist_t| {
                encoder.copyTextureToTexture(
                    &.{ .texture = ao_t, .origin = .{ .x = 0, .y = 0, .z = 0 } },
                    &.{ .texture = hist_t, .origin = .{ .x = 0, .y = 0, .z = 0 } },
                    &.{ .width = self.post_width, .height = self.post_height, .depth_or_array_layers = 1 },
                );
                self.gi_history_valid = true;
            }
        }
        // SSR is purely additive in the tonemap (color + ssr.rgb), so a zero ssr_view is the
        // identity. When SSR is disabled, skip the per-pixel ray-march (ssr_steps taps each) and
        // just clear ssr_view to 0 — a cheap full-screen clear instead of a heavy fragment pass.
        if (s.ssr_intensity > 0.0) {
            self.fullscreenPass(encoder, ssr_view, self.ssr_pipeline, ssr_bg);
        } else {
            self.clearPass(encoder, ssr_view, .{ .r = 0, .g = 0, .b = 0, .a = 0 });
        }
        // ── Mip-chain bloom: downsample scene_hdr → mip1 → … → mip(n-1) (13-tap COD, prefilter +
        // Karis on mip1), then additive tent upsample coarse → mip1 (read by the tonemap pass).
        // Each per-mip view is a disjoint subresource, so read mip i-1 / write mip i is valid.
        const bn = self.bloom_mip_count;
        if (bn >= 3 and s.bloom_intensity > 0.0) {
            var i: u32 = 1;
            while (i < bn) : (i += 1) {
                self.bloomPass(encoder, self.bloom_mip_views[i].?, self.bloom_down_pipeline, self.bloom_down_bgs[i].?, true);
            }
            i = bn - 1;
            while (i >= 2) : (i -= 1) {
                self.bloomPass(encoder, self.bloom_mip_views[i - 1].?, self.bloom_up_pipeline, self.bloom_up_bgs[i].?, false);
            }
        }

        // ── Depth of field: gather bokeh on the sharp linear-HDR scene → dof_view. Runs AFTER bloom
        // (so bloom still comes from true highlights) and BEFORE tonemap/TAA (so the resolve
        // stabilises the bokeh). When active, the tonemap samples dof_view via tonemap_dof_bg.
        const dof_active = dof_enabled != 0.0 and self.dof_bg != null and self.tonemap_dof_bg != null and self.dof_view != null;
        if (dof_active) {
            self.fullscreenPass(encoder, self.dof_view.?, self.dof_pipeline, self.dof_bg.?);
        }

        // ── Auto-exposure metering: measure scene luminance → 1×1 adapted multiplier (sampled by the
        // tonemap when enabled), then snapshot it into the history for next frame's temporal adaptation.
        if (auto_exp_on) {
            if (self.exposure_bg) |ebg| if (self.exposure_out_view) |eout| {
                self.fullscreenPass(encoder, eout, self.exposure_pipeline, ebg);
                if (self.exposure_out_texture) |eot| if (self.exposure_hist_texture) |eht| {
                    encoder.copyTextureToTexture(
                        &.{ .texture = eot, .origin = .{ .x = 0, .y = 0, .z = 0 } },
                        &.{ .texture = eht, .origin = .{ .x = 0, .y = 0, .z = 0 } },
                        &.{ .width = 1, .height = 1, .depth_or_array_layers = 1 },
                    );
                    self.exposure_valid = true;
                };
            };
        }

        const tm_bg = if (dof_active) self.tonemap_dof_bg.? else tonemap_bg;

        // TAA ownership (Task: history ownership): the resolve writes to the renderer-owned
        // taa_output, which is copied to BOTH the history and the destination. This means TAA
        // never reads back from — nor requires any usage flags on — the external destination
        // texture; history is only ever sourced from a renderer-owned, copy_src texture, and is
        // valid only after a resolve actually ran. `taaActive()` (which reads effective settings)
        // is the single predicate shared with the camera-jitter decision, so jitter and resolve
        // can never disagree.
        if (self.taaActive() and dst_texture != null) {
            // Pass 3 — tonemap+grade → taa_input (LDR, sampleable). tm_bg = DoF'd scene when active.
            self.fullscreenPass(encoder, self.taa_input_view.?, self.tonemap_pipeline, tm_bg);
            // Pass 4 — TAA resolve: taa_input + history + view-pos → taa_output (owned)
            const taa = TaaParams{
                .prev_view_proj = self.prev_view_proj,
                .cur_view_proj = self.cur_view_proj,
                .inv_view = self.inv_view,
                .cfg = .{ s.taa_feedback, 1.0, if (self.taa_valid) @as(f32, 1.0) else 0.0, 0.0 },
            };
            queue.writeBuffer(self.taa_buf, 0, std.mem.asBytes(&taa).ptr, @sizeOf(TaaParams));
            self.fullscreenPass(encoder, self.taa_output_view.?, self.taa_pipeline, self.taa_bg.?);
            const extent = wgpu.Extent3D{ .width = self.depth_width, .height = self.depth_height, .depth_or_array_layers = 1 };
            const out_src = wgpu.TexelCopyTextureInfo{ .texture = self.taa_output_texture.?, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all };
            // Snapshot the resolved frame as next frame's history (owned → owned).
            encoder.copyTextureToTexture(
                &out_src,
                &.{ .texture = self.taa_history_texture.?, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
                &extent,
            );
            // Present the resolved frame into the destination (owned → external).
            encoder.copyTextureToTexture(
                &out_src,
                &.{ .texture = dst_texture.?, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
                &extent,
            );
            self.taa_valid = true;
        } else {
            // Pass 3 — tonemap+grade: (DoF'd) scene + bloom + ao + ssr → dst (LDR)
            self.fullscreenPass(encoder, dst_view, self.tonemap_pipeline, tm_bg);
            self.taa_valid = false; // history goes stale while TAA is off
        }

        // The current (unjittered) view-proj becomes the reprojection source next frame.
        self.prev_view_proj = self.cur_view_proj;

        // 2D sprites, overlay stage: recorded on this encoder strictly AFTER the tonemap/TAA
        // writes above (including the TAA copyTextureToTexture into dst), so overlay sprites
        // composite over the final LDR image with exact colors. Also ends the sprite frame.
        if (dst_texture) |dst_tex| {
            const overlay_draws = self.sprites.renderOverlay(self.device, queue, encoder, dst_view, dst_tex.getFormat());
            self.stats.draw_calls += overlay_draws;
            if (overlay_draws > 0) self.stats.render_passes += 1;
        }
    }

    /// Helper: one full-screen-triangle pass writing `target`, with `pipeline` + group(0)=`bg`.
    fn fullscreenPass(
        self: *Gpu3d,
        encoder: *wgpu.CommandEncoder,
        target: *wgpu.TextureView,
        pipeline: *wgpu.RenderPipeline,
        bg: *wgpu.BindGroup,
    ) void {
        self.stats.render_passes += 1;
        const color_attachment = wgpu.ColorAttachment{
            .view = target,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
        };
        const pass = encoder.beginRenderPass(&.{
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&color_attachment),
            .depth_stencil_attachment = null,
        }) orelse return;
        defer pass.release();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg, 0, null);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    /// Clear a target view to a constant with no fragment work — the cheap stand-in for a
    /// full-screen effect pass whose result is the identity (e.g. additive SSR when disabled).
    fn clearPass(
        self: *Gpu3d,
        encoder: *wgpu.CommandEncoder,
        target: *wgpu.TextureView,
        clear_value: wgpu.Color,
    ) void {
        self.stats.render_passes += 1;
        const color_attachment = wgpu.ColorAttachment{
            .view = target,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = clear_value,
        };
        const pass = encoder.beginRenderPass(&.{
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&color_attachment),
            .depth_stencil_attachment = null,
        }) orelse return;
        defer pass.release();
        pass.end();
    }

    /// Like fullscreenPass but with a selectable load op — the mip-chain bloom upsample must
    /// LOAD (preserve the downsampled finer mip) so its additive blend accumulates, while the
    /// downsample CLEARs. Each call is its own render pass, so wgpu serialises the mip read/write.
    fn bloomPass(
        self: *Gpu3d,
        encoder: *wgpu.CommandEncoder,
        target: *wgpu.TextureView,
        pipeline: *wgpu.RenderPipeline,
        bg: *wgpu.BindGroup,
        clear: bool,
    ) void {
        self.stats.render_passes += 1;
        const color_attachment = wgpu.ColorAttachment{
            .view = target,
            .load_op = if (clear) .clear else .load,
            .store_op = .store,
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
        };
        const pass = encoder.beginRenderPass(&.{
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&color_attachment),
            .depth_stencil_attachment = null,
        }) orelse return;
        defer pass.release();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg, 0, null);
        pass.draw(3, 1, 0, 0);
        pass.end();
    }

    const MODEL_SLOT_STRIDE: u32 = 256; // wgpu min uniform-buffer dynamic-offset alignment

    /// (Re)size the shared model UBO ring to the per-frame high-water mark and reset the cursor.
    /// Called once per frame at the top of beginScene. The ring only ever grows (capacity high-water),
    /// matching the instance-buffer strategy — avoids per-frame reallocation churn.
    fn ensureModelRing(self: *Gpu3d) void {
        const want = std.math.ceilPowerOfTwo(u32, @max(self.model_hwm, 256)) catch self.model_hwm;
        self.model_cursor = 0;
        self.model_flushed = 0;
        if (self.model_ring != null and self.model_ring_slots >= want) return;
        if (self.model_ring_bg) |bg| bg.release();
        if (self.model_ring) |b| b.release();
        self.model_ring = null;
        self.model_ring_bg = null;
        self.model_ring_slots = 0;
        const buf = self.device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("model ring"),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
            .size = @as(u64, want) * MODEL_SLOT_STRIDE,
        }) orelse return;
        const entry = wgpu.BindGroupEntry{ .binding = 0, .buffer = buf, .offset = 0, .size = @sizeOf(ModelUniforms) };
        const bg = self.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("model ring bg"),
            .layout = self.model_bgl,
            .entry_count = 1,
            .entries = @ptrCast(&entry),
        }) orelse {
            buf.release();
            return;
        };
        self.model_ring = buf;
        self.model_ring_bg = bg;
        self.model_ring_slots = want;
        // Grow the CPU staging mirror with the ring. On allocation failure pushModel falls back to
        // the per-draw writeBuffer path, so rendering still works (just slower).
        if (self.model_staging.len != 0) self.allocator.free(self.model_staging);
        self.model_staging = self.allocator.alloc(u8, @as(usize, want) * MODEL_SLOT_STRIDE) catch &.{};
    }

    /// Write one object's ModelUniforms into the next ring slot; return its dynamic byte offset (a
    /// multiple of MODEL_SLOT_STRIDE) for setBindGroup, or null if the ring isn't ready. Each call
    /// advances the cursor so slots are disjoint within a frame (shadow + geometry never collide).
    /// The write lands in the CPU staging mirror; flushModelRing uploads the whole written range in
    /// one queue.writeBuffer per stage (a queue write always executes before the frame's submit, so
    /// every pass in the command buffer sees it — including ones encoded before the flush).
    fn pushModel(self: *Gpu3d, queue: *wgpu.Queue, model_data: ModelUniforms) ?u32 {
        const ring = self.model_ring orelse return null;
        const demand = self.model_cursor;
        self.model_cursor += 1;
        if (self.model_cursor > self.model_hwm) self.model_hwm = self.model_cursor;
        // Clamp to the last slot if this frame exceeded capacity (the high-water grew); the ring
        // resizes to fit next frame, so this only affects a single growth frame.
        const slot = if (demand < self.model_ring_slots) demand else self.model_ring_slots - 1;
        const offset = slot * MODEL_SLOT_STRIDE;
        const bytes = std.mem.asBytes(&model_data);
        if (self.model_staging.len >= @as(usize, offset) + bytes.len and slot >= self.model_flushed) {
            @memcpy(self.model_staging[offset..][0..bytes.len], bytes);
        } else {
            // Staging unavailable (alloc failure) or slot already flushed this frame (capacity
            // overflow re-using the clamped last slot) — write it directly.
            queue.writeBuffer(ring, offset, bytes.ptr, bytes.len);
        }
        return offset;
    }

    /// Upload every staged model slot written since the last flush in a single writeBuffer.
    /// Called at the end of the shadow and geometry stages (the two pushModel producers).
    fn flushModelRing(self: *Gpu3d, queue: *wgpu.Queue) void {
        const ring = self.model_ring orelse return;
        const upto = @min(self.model_cursor, self.model_ring_slots);
        if (upto <= self.model_flushed) return;
        const start = @as(usize, self.model_flushed) * MODEL_SLOT_STRIDE;
        const end = @as(usize, upto) * MODEL_SLOT_STRIDE;
        if (self.model_staging.len < end) return; // fallback path already wrote directly
        const range = self.model_staging[start..end];
        queue.writeBuffer(ring, @intCast(start), range.ptr, range.len);
        self.model_flushed = upto;
    }


    pub fn renderShadowPass(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
    ) !void {
        // Classify casters once for the whole pass (frame-constant gizmo/instancing/alpha-mode +
        // material-bind resolve), so each slice below only frustum-tests + draws.
        self.classifyShadowCasters(world, queue);

        // One depth pass per shadow slice: NUM_CASCADES directional cascades, then one perspective
        // layer per shadow-casting spot light. Each binds its matrix through the dynamic offset on
        // group 2 and renders (frustum-culled) scene depth into its array layer.
        const slice_count = NUM_CASCADES + self.active_spot_shadows;
        for (0..slice_count) |ci| {
            self.stats.render_passes += 1;
            const depth_attachment = wgpu.DepthStencilAttachment{
                .view = self.shadow_layer_views[ci],
                .depth_load_op = .clear,
                .depth_store_op = .store,
                .depth_clear_value = 1.0,
            };
            const pass_desc = wgpu.RenderPassDescriptor{
                .color_attachment_count = 0,
                .color_attachments = undefined,
                .depth_stencil_attachment = &depth_attachment,
            };

            const pass = encoder.beginRenderPass(&pass_desc) orelse return error.RenderPassFailed;
            defer pass.release();

            pass.setPipeline(self.shadow_pipeline);
            pass.setBindGroup(0, self.camera_bg, 0, null); // unused by the shadow shader; layout parity
            const cascade_off: u32 = @as(u32, @intCast(ci)) * SHADOW_SLOT;
            pass.setBindGroup(2, self.shadow_cascade_bg, 1, @as([*]const u32, @ptrCast(&cascade_off)));
            self.drawShadowCasters(queue, pass, math.Frustum.fromViewProj(self.shadow_slice_vp[ci]), self.shadow_alpha_pipeline);
            pass.end();
        }

        // Omnidirectional point-light shadows: 6 cube faces per shadow-casting point light, each
        // writing linear distance-to-light into its cube-array layer.
        for (0..self.active_point_shadows) |ci| {
            for (0..6) |f| {
                self.stats.render_passes += 1;
                const layer = ci * 6 + f;
                const depth_attachment = wgpu.DepthStencilAttachment{
                    .view = self.point_shadow_face_views[layer],
                    .depth_load_op = .clear,
                    .depth_store_op = .store,
                    .depth_clear_value = 1.0,
                };
                const pass = encoder.beginRenderPass(&.{
                    .color_attachment_count = 0,
                    .color_attachments = undefined,
                    .depth_stencil_attachment = &depth_attachment,
                }) orelse return error.RenderPassFailed;
                defer pass.release();
                pass.setPipeline(self.point_shadow_pipeline);
                pass.setBindGroup(0, self.camera_bg, 0, null); // unused; layout parity
                const off: u32 = @as(u32, @intCast(layer)) * SHADOW_SLOT;
                pass.setBindGroup(2, self.point_shadow_bg, 1, @as([*]const u32, @ptrCast(&off)));
                // Point-cube pass: group-2 layout differs from the alpha pipeline's, so masked casters
                // are skipped here (null) — they still cast cascade/spot shadows.
                self.drawShadowCasters(queue, pass, math.Frustum.fromViewProj(self.point_face_vp[layer]), null);
                pass.end();
            }
        }
        self.flushModelRing(queue); // one writeBuffer for every model slot staged by the shadow slices
    }

    /// Per-slice frustum cull: is this caster's bounding sphere inside the shadow slice's frustum?
    /// Keeps shadow cost sane — otherwise every cascade / cube face redraws the whole scene.
    fn shadowCasterVisible(renderable: scene_mod.Renderable, mesh_data: *const resources_mod.Mesh, frustum: math.Frustum) bool {
        const wt = renderable.node.world_transform;
        const center = wt.position.add(wt.rotation.rotateVec(mesh_data.bounds_center.mul(wt.scale)));
        const smax = @max(@abs(wt.scale.x), @max(@abs(wt.scale.y), @abs(wt.scale.z)));
        const radius = mesh_data.bounds_radius * smax * 1.05 + 0.05;
        return frustum.intersectsSphere(center, radius);
    }

    fn drawShadowMesh(self: *Gpu3d, queue: *wgpu.Queue, pass: *wgpu.RenderPassEncoder, mesh_handle: scene_mod.MeshHandle, mesh_data: *const resources_mod.Mesh) void {
        for (mesh_data.primitives, 0..) |*prim, prim_idx| {
            const buffers = self.mesh_cache.getOrUpload(queue, mesh_handle, @intCast(prim_idx), prim) orelse continue;
            pass.setVertexBuffer(0, buffers.vertex_buf, 0, wgpu.WGPU_WHOLE_SIZE);
            pass.setIndexBuffer(buffers.index_buf, .uint32, 0, wgpu.WGPU_WHOLE_SIZE);
            pass.drawIndexed(buffers.index_count, 1, 0, 0, 0);
        }
    }

    /// Classify the frame's world-3D shadow casters ONCE into opaque + alpha-masked lists. The
    /// gizmo/instancing/alpha-mode tests, mesh lookup and (for masked casters) the material bind-group
    /// resolve are all frame-constant, so hoisting them here turns each shadow slice's per-renderable
    /// work into a bare frustum-test + draw. Blended/glass never cast, so they land in neither list.
    fn classifyShadowCasters(self: *Gpu3d, world: *const scene_mod.World, queue: *wgpu.Queue) void {
        self.shadow_casters_opaque.clearRetainingCapacity();
        self.shadow_casters_masked.clearRetainingCapacity();
        for (self.world3d_list.items) |renderable| {
            if (std.mem.startsWith(u8, renderable.node.name, "__Gizmo")) continue;
            if (self.instanceFor(renderable.node.entity)) |_| continue;
            const mr = renderable.mesh_renderer;
            const mesh_data = world.getMesh(mr.mesh) orelse continue;
            if (mr.material != scene_mod.null_material) {
                if (world.getMaterial(mr.material)) |m| {
                    if (m.alpha_mode == .blend or m.alpha_mode == .glass) continue; // never cast
                    if (m.alpha_mode == .mask) {
                        const mat_bg = (self.ensureMaterialGpu(queue, mr.material, m) catch continue).bind_group;
                        self.shadow_casters_masked.append(self.allocator, .{
                            .renderable = renderable,
                            .mesh_data = mesh_data,
                            .material = m,
                            .mat_bg = mat_bg,
                        }) catch {};
                        continue;
                    }
                }
            }
            self.shadow_casters_opaque.append(self.allocator, .{
                .renderable = renderable,
                .mesh_data = mesh_data,
            }) catch {};
        }
    }

    /// Draw the pre-classified shadow casters into the currently-bound shadow pass. Opaque casters use
    /// the bound depth pipeline (dummy white material). Masked casters (foliage/decals) draw in a
    /// second pass with `alpha_pipeline` (alpha-tested cut-out shadows) when the caller supplies one —
    /// the cascade/spot slices do; the point-cube pass passes null (its group-2 layout differs), so
    /// masked casters still don't cast point shadows. Shared by all slices (the caller binds group 0 +
    /// the per-slice matrix on group 2). Each caster's ModelUniforms is slice-invariant, so its ring
    /// slot is pushed on the first slice that draws it and the offset is reused by later slices.
    fn drawShadowCasters(self: *Gpu3d, queue: *wgpu.Queue, pass: *wgpu.RenderPassEncoder, frustum: math.Frustum, alpha_pipeline: ?*wgpu.RenderPipeline) void {
        // Pass 1 — opaque casters (dummy white material, no alpha test).
        for (self.shadow_casters_opaque.items) |*rec| {
            if (!shadowCasterVisible(rec.renderable, rec.mesh_data, frustum)) continue;
            if (rec.model_off == NO_MODEL_OFF) {
                const model_matrix = rec.renderable.node.worldMatrix();
                const model_data = ModelUniforms{
                    .model = model_matrix.toArray(),
                    .normal_mat = model_matrix.toArray(),
                    .base_color = .{ 1, 1, 1, 1 },
                    .metallic_roughness = .{ 0, 0, 0, 0 },
                    .emissive = .{ 0, 0, 0, 0 },
                    .effect = .{ 0, 0, 0, 0 },
                    .surface = .{ 0, 0, 0, 0 },
                };
                rec.model_off = self.pushModel(queue, model_data) orelse continue;
            }
            pass.setBindGroup(1, self.model_ring_bg.?, 1, @as([*]const u32, @ptrCast(&rec.model_off)));
            self.drawShadowMesh(queue, pass, rec.renderable.mesh_renderer.mesh, rec.mesh_data);
        }

        // Pass 2 — masked casters (alpha-tested cut-out). Groups 0/2 stay bound across the pipeline
        // switch (camera/model/cascade layouts are shared); this adds the material texture group (3).
        const ap = alpha_pipeline orelse return;
        var switched = false;
        for (self.shadow_casters_masked.items) |*rec| {
            if (!shadowCasterVisible(rec.renderable, rec.mesh_data, frustum)) continue;
            if (!switched) {
                pass.setPipeline(ap);
                switched = true;
            }
            if (rec.model_off == NO_MODEL_OFF) {
                const m = rec.material.?;
                const model_matrix = rec.renderable.node.worldMatrix();
                const model_data = ModelUniforms{
                    .model = model_matrix.toArray(),
                    .normal_mat = model_matrix.toArray(),
                    .base_color = m.base_color_factor.toArray(),
                    .metallic_roughness = .{ 0, 0, 0, 0 },
                    .emissive = .{ 0, 0, 0, 0 },
                    // effect.z = alpha mode (mask), .w = alpha cutoff (bitcast) — read by the alpha shader.
                    .effect = .{ 0, 0, @intFromEnum(m.alpha_mode), @bitCast(m.alpha_cutoff) },
                    .surface = .{ 0, 0, 0, 0 },
                };
                rec.model_off = self.pushModel(queue, model_data) orelse continue;
            }
            pass.setBindGroup(1, self.model_ring_bg.?, 1, @as([*]const u32, @ptrCast(&rec.model_off)));
            pass.setBindGroup(3, rec.mat_bg.?, 0, null);
            self.drawShadowMesh(queue, pass, rec.renderable.mesh_renderer.mesh, rec.mesh_data);
        }
    }

    fn renderLayer(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        color_attachment: *const wgpu.ColorAttachment,
        depth_attachment: *const wgpu.DepthStencilAttachment,
        frame_width: u32,
        frame_height: u32,
        layer: scene_mod.RenderLayer,
        clear_depth: bool,
    ) !void {
        const cam_node = switch (layer) {
            .world_3d => world.active_camera orelse return,
            .scene2d => world.active_camera_2d orelse return,
        };
        const cam_union = cam_node.getComponent(.camera) orelse return;
        const cam_comp = &cam_union.camera;
        const aspect = @as(f32, @floatFromInt(frame_width)) / @as(f32, @floatFromInt(frame_height));
        var proj = cam_comp.projMatrix(aspect);
        const view = Mat4.lookAt(
            cam_node.world_transform.position,
            cam_node.world_transform.position.add(cam_node.world_transform.rotation.rotateVec(Vec3.forward)),
            Vec3.up,
        );
        if (layer == .world_3d) {
            // Stash the world-3D projection (SSAO/SSR sample projection) and the unjittered
            // view-proj + inverse view for TAA reprojection BEFORE applying the jitter.
            self.last_proj = proj.toArray();
            self.last_view = view.toArray();
            self.cur_view_proj = proj.mul(view).toArray();
            self.inv_view = view.inverse().toArray();
            // Per-frame sub-pixel projection jitter — applied ONLY when the TAA resolve will
            // actually run this frame (taaActive). Jittering without a resolve makes edges
            // wobble frame-to-frame; that bug appeared on the legacy path and in diagnostic mode.
            if (self.temporalJitterActive()) {
                const fw: f32 = @floatFromInt(frame_width);
                const fh: f32 = @floatFromInt(frame_height);
                const jx = (halton(self.taa_frame, 2) - 0.5) * 2.0 / fw;
                const jy = (halton(self.taa_frame, 3) - 0.5) * 2.0 / fh;
                proj.cols[2].x += jx;
                proj.cols[2].y += jy;
            }
        }
        const view_proj = proj.mul(view);

        const cam_pos = cam_node.world_transform.position;
        const cam_data = CameraUniforms{
            .view_proj = view_proj.toArray(),
            .view = view.toArray(),
            .camera_pos = .{ cam_pos.x, cam_pos.y, cam_pos.z, 1.0 },
            .inv_view_proj = view_proj.inverse().toArray(),
        };
        const cam_bytes = std.mem.asBytes(&cam_data);
        queue.writeBuffer(self.camera_buf, 0, cam_bytes.ptr, cam_bytes.len);

        self.render_list.clearRetainingCapacity();
        if (layer == .world_3d) {
            // world-3D was already collected once in beginScene — copy it (two-pointer structs) and
            // cull below, instead of walking the tree a third time this frame.
            try self.render_list.appendSlice(self.allocator, self.world3d_list.items);
        } else {
            try world.collectRenderables(self.allocator, layer, &self.render_list);
        }

        // Frustum cull (world-3D only): drop renderables whose world-space bounding sphere is fully
        // outside the camera frustum, so off-screen meshes cost no draw call. Filtering render_list
        // in place culls the opaque, instanced and transparent loops below in one pass. Instanced
        // nodes are never culled here — their extent is the whole instance set, not one mesh.
        if (layer == .world_3d and self.frustum_cull) {
            const fr = math.Frustum.fromViewProj(view_proj);
            var w: usize = 0;
            for (self.render_list.items) |r| {
                var keep = true;
                const instanced = if (self.instanceFor(r.node.entity)) |inst| inst.count > 0 else false;
                if (!instanced) {
                    if (world.getMesh(r.mesh_renderer.mesh)) |mesh| {
                        const wt = r.node.world_transform;
                        const center = wt.position.add(wt.rotation.rotateVec(mesh.bounds_center.mul(wt.scale)));
                        const smax = @max(@abs(wt.scale.x), @max(@abs(wt.scale.y), @abs(wt.scale.z)));
                        const radius = mesh.bounds_radius * smax * 1.05 + 0.05; // margin: keep borderline meshes
                        keep = fr.intersectsSphere(center, radius);
                    }
                }
                if (keep) {
                    self.render_list.items[w] = r;
                    w += 1;
                }
            }
            self.render_list.shrinkRetainingCapacity(w);
        }
        const renderables = self.render_list;

        const layer_depth_attachment = wgpu.DepthStencilAttachment{
            .view = depth_attachment.view,
            .depth_load_op = if (clear_depth) .clear else .load,
            .depth_store_op = .store,
            .depth_clear_value = depth_attachment.depth_clear_value,
        };
        const layer_color_attachments = [_]wgpu.ColorAttachment{
            .{
                .view = self.msaa_color_view.?,
                .resolve_target = color_attachment.view,
                .load_op = if (layer == .world_3d) color_attachment.load_op else .load,
                .store_op = color_attachment.store_op,
                .clear_value = color_attachment.clear_value,
            },
            // View-position G-buffer (MRT location 1). Cleared to w=0 on the first layer so
            // background pixels read as "no geometry"; the 2D layer loads to keep 3D positions.
            .{
                .view = self.msaa_pos_view.?,
                .resolve_target = self.gbuf_pos_view.?,
                .load_op = if (layer == .world_3d) .clear else .load,
                .store_op = .store,
                .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
            },
            // View-normal + roughness G-buffer (MRT location 2), for roughness-aware SSR.
            .{
                .view = self.msaa_normal_view.?,
                .resolve_target = self.gbuf_normal_view.?,
                .load_op = if (layer == .world_3d) .clear else .load,
                .store_op = .store,
                .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
            },
            // Albedo G-buffer (MRT location 3), for receiver-albedo tinting of SSGI.
            .{
                .view = self.msaa_albedo_view.?,
                .resolve_target = self.gbuf_albedo_view.?,
                .load_op = if (layer == .world_3d) .clear else .load,
                .store_op = .store,
                .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
            },
        };
        const pass_desc = wgpu.RenderPassDescriptor{
            .color_attachment_count = layer_color_attachments.len,
            .color_attachments = &layer_color_attachments,
            .depth_stencil_attachment = &layer_depth_attachment,
        };

        const pass = encoder.beginRenderPass(&pass_desc) orelse return error.RenderPassFailed;
        defer pass.release();

        pass.setBindGroup(0, self.camera_bg, 0, null);
        pass.setBindGroup(3, self.shadow_map_bg, 0, null); // includes the refraction source at binding 5

        // Wireframe debug mode swaps every geometry pass to the line-list pipeline + edge index
        // buffer; the drawRenderable helpers pick the line index buffer off the same flag.
        const wf = self.settings.wireframe != 0.0;

        // Pass 1 — bucket geometry: opaque/masked → opaque_scratch (sorted below), blended → pass 2,
        // glass → post-copy refraction pass, instanced → batch pass.
        self.transparent_scratch.clearRetainingCapacity();
        self.glass_scratch.clearRetainingCapacity();
        self.instanced_scratch.clearRetainingCapacity();
        self.opaque_scratch.clearRetainingCapacity();
        for (renderables.items) |renderable| {
            // Instanced nodes (asteroid belt etc.) draw in one batch after the opaque loop. A node with
            // an instance entry is handled ONLY here — even with 0 live instances (an empty LOD bucket)
            // it must never fall through to a single full-size draw at the node origin.
            if (self.instanceFor(renderable.node.entity)) |inst| {
                if (inst.count > 0) self.instanced_scratch.append(self.allocator, renderable) catch {};
                continue;
            }
            const mat_data = if (renderable.mesh_renderer.material != scene_mod.null_material)
                world.getMaterial(renderable.mesh_renderer.material)
            else
                null;
            if (mat_data) |m| {
                if (m.alpha_mode == .blend or m.alpha_mode == .glass) {
                    // Blended → pass 2 (same pass). Glass → the post-copy refraction pass so it can
                    // sample the opaque scene behind it. Both tagged with camera distance for sorting.
                    const p = renderable.node.world_transform.position;
                    const d = p.sub(cam_pos);
                    const item = TransparentItem{ .renderable = renderable, .dist_sq = d.dot(d) };
                    if (m.alpha_mode == .glass and layer == .world_3d) {
                        self.glass_scratch.append(self.allocator, item) catch {};
                    } else {
                        self.transparent_scratch.append(self.allocator, item) catch {};
                    }
                    continue;
                }
            }
            self.opaque_scratch.append(self.allocator, .{
                .renderable = renderable,
                .material = renderable.mesh_renderer.material,
                .double_sided = if (mat_data) |m| @intFromBool(m.double_sided) else 0,
            }) catch {};
        }

        // Sort opaque draws by (double_sided, material handle) so identical materials are adjacent.
        // Opaque geometry is order-independent (depth-write on, depth-test .less), so reordering
        // produces a byte-identical image — but lets the loop below rebind the material group (2) and
        // swap the cull pipeline only at run boundaries instead of every draw (state-change cut).
        std.mem.sort(OpaqueItem, self.opaque_scratch.items, {}, struct {
            fn less(_: void, a: OpaqueItem, b: OpaqueItem) bool {
                if (a.double_sided != b.double_sided) return a.double_sided < b.double_sided;
                return a.material < b.material;
            }
        }.less);

        // Pass 1 draw loop — dedup the pipeline + material bind group across the sorted runs. The
        // per-object model group (1, dynamic offset) still rebinds every draw.
        var cur_pipeline: ?*wgpu.RenderPipeline = null;
        var cur_mat_bg: ?*wgpu.BindGroup = null;
        for (self.opaque_scratch.items) |item| {
            const want_pipeline = if (wf)
                self.wireframe_pipeline
            else if (item.double_sided != 0)
                self.pipeline_double_sided
            else
                self.pipeline;
            if (cur_pipeline != want_pipeline) {
                pass.setPipeline(want_pipeline);
                cur_pipeline = want_pipeline;
            }
            const mat_data = if (item.material != scene_mod.null_material)
                world.getMaterial(item.material)
            else
                null;
            const material_bg = if (mat_data) |mat|
                (self.ensureMaterialGpu(queue, item.material, mat) catch continue).bind_group
            else
                self.default_material.bind_group;
            const rebind = cur_mat_bg != material_bg;
            if (rebind) cur_mat_bg = material_bg;
            self.drawRenderableDedup(world, queue, pass, item.renderable, mat_data, material_bg, rebind);
        }

        // Instanced pass — one drawIndexed(idx, N) per instanced node (shared mesh + material).
        if (self.instanced_scratch.items.len > 0) {
            pass.setPipeline(if (wf) self.wireframe_instanced_pipeline else self.instanced_pipeline);
            for (self.instanced_scratch.items) |renderable| {
                const mat_data = if (renderable.mesh_renderer.material != scene_mod.null_material)
                    world.getMaterial(renderable.mesh_renderer.material)
                else
                    null;
                self.drawRenderableInstanced(world, queue, pass, renderable, mat_data);
            }
        }

        // Pass 2 — blended geometry, farthest-first, with depth writes off so overlapping
        // transparent surfaces don't punch holes in each other or in the opaque scene.
        if (self.transparent_scratch.items.len > 0) {
            std.mem.sort(TransparentItem, self.transparent_scratch.items, {}, struct {
                fn farther(_: void, a: TransparentItem, b: TransparentItem) bool {
                    return a.dist_sq > b.dist_sq;
                }
            }.farther);

            var trans_cur: ?*wgpu.RenderPipeline = null;
            for (self.transparent_scratch.items) |item| {
                const mat_data = world.getMaterial(item.renderable.mesh_renderer.material);
                const want = if (wf)
                    self.wireframe_pipeline
                else if (mat_data != null and mat_data.?.double_sided)
                    self.transparent_pipeline_ds
                else
                    self.transparent_pipeline;
                if (trans_cur != want) {
                    pass.setPipeline(want);
                    trans_cur = want;
                }
                self.drawRenderable(world, queue, pass, item.renderable, mat_data);
            }
        }

        // VFX particles — camera-facing billboards over the lit + blended scene. World-3D layer only
        // (the 2D scene layer has its own ortho camera). No-op until the host uploads particles.
        if (layer == .world_3d)
            self.particles.render(self.device, self.camera_bgl, SCENE_HDR_FORMAT, MSAA_SAMPLES, pass, self.camera_bg);

        pass.end();

        // Glass pass — after the opaque (+blend+particle) result is resolved to scene_hdr, copy it to
        // refraction_src and draw glass in a second MSAA pass that samples it for screen-space
        // refraction. Loads the stored MSAA colour + depth so glass composites over and depth-tests
        // against the opaque scene; the glass pipeline WRITES depth so a bottle's own overlapping glass
        // surfaces self-sort (nearest wins). Resolves the combined result back to scene_hdr.
        if (layer == .world_3d and self.glass_scratch.items.len > 0) {
            encoder.copyTextureToTexture(
                &.{ .texture = self.scene_hdr_texture.?, .origin = .{ .x = 0, .y = 0, .z = 0 } },
                &.{ .texture = self.refraction_src_texture.?, .origin = .{ .x = 0, .y = 0, .z = 0 } },
                &.{ .width = frame_width, .height = frame_height, .depth_or_array_layers = 1 },
            );
            std.mem.sort(TransparentItem, self.glass_scratch.items, {}, struct {
                fn farther(_: void, a: TransparentItem, b: TransparentItem) bool {
                    return a.dist_sq > b.dist_sq;
                }
            }.farther);
            const glass_color_attachments = [_]wgpu.ColorAttachment{
                .{ .view = self.msaa_color_view.?, .resolve_target = self.scene_hdr_view.?, .load_op = .load, .store_op = .store, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 } },
                .{ .view = self.msaa_pos_view.?, .load_op = .load, .store_op = .store, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 } },
                .{ .view = self.msaa_normal_view.?, .load_op = .load, .store_op = .store, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 } },
                .{ .view = self.msaa_albedo_view.?, .load_op = .load, .store_op = .store, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 } },
            };
            const glass_depth = wgpu.DepthStencilAttachment{
                .view = depth_attachment.view,
                .depth_load_op = .load,
                .depth_store_op = .store,
                .depth_clear_value = depth_attachment.depth_clear_value,
            };
            const glass_pass = encoder.beginRenderPass(&.{
                .color_attachment_count = glass_color_attachments.len,
                .color_attachments = &glass_color_attachments,
                .depth_stencil_attachment = &glass_depth,
            }) orelse return error.RenderPassFailed;
            defer glass_pass.release();
            glass_pass.setBindGroup(0, self.camera_bg, 0, null);
            glass_pass.setBindGroup(3, self.shadow_map_bg_glass, 0, null); // refraction src (5) + G-buffer pos (6)
            var glass_cur: ?*wgpu.RenderPipeline = null;
            for (self.glass_scratch.items) |item| {
                const mat_data = world.getMaterial(item.renderable.mesh_renderer.material);
                const want = if (wf)
                    self.wireframe_pipeline
                else if (mat_data != null and mat_data.?.double_sided)
                    self.glass_pipeline_ds
                else
                    self.glass_pipeline;
                if (glass_cur != want) {
                    glass_pass.setPipeline(want);
                    glass_cur = want;
                }
                self.drawRenderable(world, queue, glass_pass, item.renderable, mat_data);
            }
            glass_pass.end();
        }
    }

    /// Emit draw calls for a single renderable. `mat_data` is the resolved material (may be
    /// null for the default grey). Caller has already bound the active pipeline and the
    /// camera/shadow bind groups; this resolves + binds the per-model (1) and material (2) groups.
    /// Used by the transparent/glass passes (few items, no dedup); the opaque pass uses the sorted
    /// `drawRenderableDedup` path instead.
    fn drawRenderable(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        pass: *wgpu.RenderPassEncoder,
        renderable: scene_mod.Renderable,
        mat_data: ?*const resources_mod.Material,
    ) void {
        const material_bg = if (mat_data) |mat|
            (self.ensureMaterialGpu(queue, renderable.mesh_renderer.material, mat) catch return).bind_group
        else
            self.default_material.bind_group;
        self.drawRenderableDedup(world, queue, pass, renderable, mat_data, material_bg, true);
    }

    /// Draw a renderable with an externally-resolved material bind group, optionally skipping the
    /// group-2 rebind when the caller knows it is unchanged from the previous draw (opaque sorted
    /// path). Binds the per-object model group (1) unconditionally, then draws each primitive.
    fn drawRenderableDedup(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        pass: *wgpu.RenderPassEncoder,
        renderable: scene_mod.Renderable,
        mat_data: ?*const resources_mod.Material,
        material_bg: *wgpu.BindGroup,
        bind_material: bool,
    ) void {
        const mr = renderable.mesh_renderer;
        const mesh_data = world.getMesh(mr.mesh) orelse return;

        // Diagnostic: report (once per entity) primitives rendering with the default material —
        // i.e. no material assigned. A car whose body/chrome/tyres all look the same grey/silver
        // usually means most primitives fell back here.
        if (mat_data == null and !self.logged_default_mat.contains(renderable.node.entity)) {
            self.logged_default_mat.put(self.allocator, renderable.node.entity, {}) catch {};
            std.log.warn("zigote 3d: node '{s}' is rendering with the DEFAULT material (no material assigned)", .{renderable.node.name});
        }

        const model_matrix = renderable.node.worldMatrix();
        const base_color = if (mat_data) |m| m.base_color_factor.toArray() else [4]f32{ 0.8, 0.8, 0.8, 1.0 };
        const metallic = if (mat_data) |m| m.metallic_factor else 0.0;
        const roughness = if (mat_data) |m| m.roughness_factor else 0.5;
        const emissive = if (mat_data) |m| m.emissive_factor.toArray() else [3]f32{ 0, 0, 0 };
        // Extended PBR: clearcoat factor/roughness ride in metallic_roughness.zw, specular in
        // emissive.w. Defaults (0/0/1) reproduce a plain dielectric/metal with no coat.
        const clearcoat = if (mat_data) |m| m.clearcoat_factor else 0.0;
        const clearcoat_rough = if (mat_data) |m| m.clearcoat_roughness else 0.0;
        const specular = if (mat_data) |m| m.specular_factor else 1.0;
        const effect = if (mat_data) |m| @intFromEnum(m.effect) else 0;
        // Alpha mode (0=opaque, 1=mask, 2=blend) + cutoff ride in effect.z/.w. The shader uses
        // it to `discard` masked texels (foliage/decals) and to give blended surfaces a
        // Fresnel glass response (opaque/reflective at grazing angles).
        const alpha_code: u32 = if (mat_data) |m| @intFromEnum(m.alpha_mode) else 0;
        const cutoff_bits: u32 = if (mat_data) |m| @bitCast(m.alpha_cutoff) else @bitCast(@as(f32, 0.5));
        const is_selected: u32 = if (self.selected_node_ptr != 0 and
            @intFromPtr(renderable.node) == self.selected_node_ptr) 1 else 0;
        // Extended surface params (surface vec4): IOR + transmission + ORM occlusion strength.
        // Legacy glass (alpha mode glass, transmission never set) is treated as fully transmissive.
        const ior = if (mat_data) |m| m.ior else 1.5;
        const transmission = if (mat_data) |m|
            (if (m.alpha_mode == .glass and m.transmission <= 0.0) 1.0 else m.transmission)
        else
            0.0;
        const occlusion = if (mat_data) |m| m.occlusion_strength else 0.0;

        const model_data = ModelUniforms{
            .model = model_matrix.toArray(),
            .normal_mat = renderable.node.normalMatrix().toArray(),
            .base_color = base_color,
            .metallic_roughness = .{ metallic, roughness, clearcoat, clearcoat_rough },
            .emissive = .{ emissive[0], emissive[1], emissive[2], specular },
            .effect = .{ effect, is_selected, alpha_code, cutoff_bits },
            .surface = .{ ior, transmission, occlusion, 0 },
        };

        const model_off = self.pushModel(queue, model_data) orelse return;
        pass.setBindGroup(1, self.model_ring_bg.?, 1, @as([*]const u32, @ptrCast(&model_off)));

        if (bind_material) pass.setBindGroup(2, material_bg, 0, null);

        for (mesh_data.primitives, 0..) |*prim, prim_idx| {
            const buffers = self.mesh_cache.getOrUpload(queue, mr.mesh, @intCast(prim_idx), prim) orelse continue;

            pass.setVertexBuffer(0, buffers.vertex_buf, 0, wgpu.WGPU_WHOLE_SIZE);
            if (self.settings.wireframe != 0.0) {
                const lb = self.mesh_cache.ensureLineBuffer(queue, mr.mesh, @intCast(prim_idx), prim) orelse continue;
                pass.setIndexBuffer(lb.buf, .uint32, 0, wgpu.WGPU_WHOLE_SIZE);
                pass.drawIndexed(lb.count, 1, 0, 0, 0);
            } else {
                pass.setIndexBuffer(buffers.index_buf, .uint32, 0, wgpu.WGPU_WHOLE_SIZE);
                pass.drawIndexed(buffers.index_count, 1, 0, 0, 0);
            }
            self.stats.draw_calls += 1;
            self.stats.triangles += buffers.index_count / 3;
        }
    }

    /// Draw an instanced renderable: the shared mesh + material, with per-instance model
    /// matrices from the node's instance buffer. One drawIndexed(idx, count) per primitive.
    /// The g1 model UBO supplies only the shared material factors — vs_instanced reads the
    /// transform from vertex buffer 1, so the UBO model matrix is identity/unused.
    fn drawRenderableInstanced(
        self: *Gpu3d,
        world: *const scene_mod.World,
        queue: *wgpu.Queue,
        pass: *wgpu.RenderPassEncoder,
        renderable: scene_mod.Renderable,
        mat_data: ?*const resources_mod.Material,
    ) void {
        const mr = renderable.mesh_renderer;
        const mesh_data = world.getMesh(mr.mesh) orelse return;
        const inst = self.instanceFor(renderable.node.entity) orelse return;
        if (inst.count == 0) return;

        const base_color = if (mat_data) |m| m.base_color_factor.toArray() else [4]f32{ 0.8, 0.8, 0.8, 1.0 };
        const metallic = if (mat_data) |m| m.metallic_factor else 0.0;
        const roughness = if (mat_data) |m| m.roughness_factor else 0.5;
        const emissive = if (mat_data) |m| m.emissive_factor.toArray() else [3]f32{ 0, 0, 0 };
        const clearcoat = if (mat_data) |m| m.clearcoat_factor else 0.0;
        const clearcoat_rough = if (mat_data) |m| m.clearcoat_roughness else 0.0;
        const specular = if (mat_data) |m| m.specular_factor else 1.0;
        const effect = if (mat_data) |m| @intFromEnum(m.effect) else 0;
        const alpha_code: u32 = if (mat_data) |m| @intFromEnum(m.alpha_mode) else 0;
        const cutoff_bits: u32 = if (mat_data) |m| @bitCast(m.alpha_cutoff) else @bitCast(@as(f32, 0.5));
        const ior = if (mat_data) |m| m.ior else 1.5;
        const transmission = if (mat_data) |m|
            (if (m.alpha_mode == .glass and m.transmission <= 0.0) 1.0 else m.transmission)
        else
            0.0;
        const occlusion = if (mat_data) |m| m.occlusion_strength else 0.0;

        const ident = Mat4.identity.toArray();
        const model_data = ModelUniforms{
            .model = ident,
            .normal_mat = ident,
            .base_color = base_color,
            .metallic_roughness = .{ metallic, roughness, clearcoat, clearcoat_rough },
            .emissive = .{ emissive[0], emissive[1], emissive[2], specular },
            .effect = .{ effect, 0, alpha_code, cutoff_bits },
            .surface = .{ ior, transmission, occlusion, 0 },
        };
        const model_off = self.pushModel(queue, model_data) orelse return;
        pass.setBindGroup(1, self.model_ring_bg.?, 1, @as([*]const u32, @ptrCast(&model_off)));

        const material_bg = if (mat_data) |mat|
            (self.ensureMaterialGpu(queue, mr.material, mat) catch return).bind_group
        else
            self.default_material.bind_group;
        pass.setBindGroup(2, material_bg, 0, null);

        // Ensure / grow the GPU instance buffer, then upload this frame's matrices. Grow with
        // headroom (next power of two): the per-frame LOD/cull count fluctuates, so sizing the
        // buffer to the exact count would reallocate it almost every frame as the running max
        // ratchets up — churning VRAM through wgpu's deferred buffer deallocation and stalling
        // the present queue over time. Capacity only ever grows; reallocation converges quickly.
        const needed_bytes: u64 = @as(u64, inst.count) * 64;
        if (inst.gpu == null or inst.capacity < inst.count) {
            if (inst.gpu) |b| b.release();
            const new_cap = std.math.ceilPowerOfTwo(u32, @max(inst.count, 256)) catch inst.count;
            inst.gpu = self.device.createBuffer(&.{
                .label = wgpu.StringView.fromSlice("instance buffer"),
                .usage = wgpu.BufferUsages.vertex | wgpu.BufferUsages.copy_dst,
                .size = @as(u64, new_cap) * 64,
            }) orelse {
                inst.gpu = null;
                inst.capacity = 0;
                return;
            };
            inst.capacity = new_cap;
        }
        const data_bytes = std.mem.sliceAsBytes(inst.cpu.items);
        queue.writeBuffer(inst.gpu.?, 0, data_bytes.ptr, @min(data_bytes.len, needed_bytes));

        for (mesh_data.primitives, 0..) |*prim, prim_idx| {
            const buffers = self.mesh_cache.getOrUpload(queue, mr.mesh, @intCast(prim_idx), prim) orelse continue;

            pass.setVertexBuffer(0, buffers.vertex_buf, 0, wgpu.WGPU_WHOLE_SIZE);
            pass.setVertexBuffer(1, inst.gpu.?, 0, needed_bytes);
            if (self.settings.wireframe != 0.0) {
                const lb = self.mesh_cache.ensureLineBuffer(queue, mr.mesh, @intCast(prim_idx), prim) orelse continue;
                pass.setIndexBuffer(lb.buf, .uint32, 0, wgpu.WGPU_WHOLE_SIZE);
                pass.drawIndexed(lb.count, inst.count, 0, 0, 0);
            } else {
                pass.setIndexBuffer(buffers.index_buf, .uint32, 0, wgpu.WGPU_WHOLE_SIZE);
                pass.drawIndexed(buffers.index_count, inst.count, 0, 0, 0);
            }
            self.stats.draw_calls += 1;
            self.stats.triangles += (buffers.index_count / 3) * inst.count;
        }
    }

    /// Invalidate GPU buffers for an entity when it is removed.
    /// Resolve a node's live per-instance buffer, or null if it has none. The generational
    /// handle guards against a stale index entry pointing at a freed/reused pool slot.
    fn instanceFor(self: *Gpu3d, entity_id: scene_mod.Entity) ?*InstanceGpu {
        // Fast path: the overwhelming majority of scenes have no instanced nodes, yet this is
        // probed 2-3× per renderable per frame (shadow + cull + opaque loops). Skip the hash of
        // the entity key entirely when the map is empty — a single integer compare instead.
        if (self.instance_index.count() == 0) return null;
        const id = self.instance_index.get(entity_id) orelse return null;
        return self.instance_pool.getColumnPtrIfLive(.{ .id = id }, .gpu);
    }

    pub fn invalidateEntity(self: *Gpu3d, entity_id: scene_mod.Entity) void {
        // removeIfLive releases the slot, auto-calling the value's 0-arg deinit.
        if (self.instance_index.fetchRemove(entity_id)) |entry| {
            _ = self.instance_pool.removeIfLive(.{ .id = entry.value });
        }
    }

    /// Submit (or update) the per-instance model matrices for a node's mesh. `ptr` points at
    /// count*16 column-major f32 matrix values. count==0 reverts the node to a normal draw.
    /// Stored CPU-side here; uploaded to the GPU during the next render.
    // ptr is a nullable C pointer: C# passes null when count == 0 (clearing instancing for a node,
    // e.g. an empty LOD bucket). Coercing a null [*c] to a non-null [*] would panic, so accept [*c]
    // and only read it when count > 0 and it is non-null.
    pub fn setInstances(self: *Gpu3d, entity_id: scene_mod.Entity, ptr: [*c]const f32, count: u32) void {
        const inst = self.instanceFor(entity_id) orelse blk: {
            const handle = self.instance_pool.add(.{ .gpu = .{ .alloc = self.allocator } }) catch return;
            self.instance_index.put(self.allocator, entity_id, handle.id) catch {
                self.instance_pool.removeAssumeLive(handle);
                return;
            };
            break :blk self.instance_pool.getColumnPtrAssumeLive(handle, .gpu);
        };
        inst.cpu.clearRetainingCapacity();
        if (count > 0 and ptr != null) {
            inst.cpu.appendSlice(self.allocator, ptr[0 .. count * 16]) catch {
                inst.count = 0;
                return;
            };
            inst.count = count;
        } else {
            inst.count = 0;
        }
    }
};
