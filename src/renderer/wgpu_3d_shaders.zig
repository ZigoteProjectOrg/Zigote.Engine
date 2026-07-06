//! WGSL shader sources for the 3D renderer (wgpu_3d.zig).
//! Extracted verbatim so the renderer module stays focused on pipeline wiring.

pub const mesh_shader_source = @embedFile("shaders/mesh_shader_source.wgsl");

// VFX particle billboards. Vertex-pulled camera-facing quads (additive / premultiplied-alpha),
// drawn in the geometry pass after the transparent meshes. See wgpu_particles.zig.
pub const particle_shader_source = @embedFile("shaders/particle_shader_source.wgsl");

// VFX GPU particle simulation compute kernel (spawn + update); writes the billboard instance buffer.
pub const particle_compute_source = @embedFile("shaders/particle_compute_source.wgsl");

// 2D sprite quads (default sprite material + the custom-shader contract). See wgpu_sprites.zig.
pub const sprite_shader_source = @embedFile("shaders/sprite_shader_source.wgsl");

pub const shadow_shader_source = @embedFile("shaders/shadow_shader_source.wgsl");

// Alpha-tested cascade/spot shadow depth pass — masked casters (foliage/decals) cast cut-out shadows.
pub const shadow_alpha_shader_source = @embedFile("shaders/shadow_alpha_shader_source.wgsl");

// Omnidirectional point-light shadows. Renders the scene into the 6 faces of a depth cube (per
// shadow-casting point light), writing LINEAR distance-to-light (normalised by range) to frag_depth
// so the mesh shader can hardware-PCF-compare a cube sample by direction. See wgpu_3d.zig.
pub const point_shadow_shader_source = @embedFile("shaders/point_shadow_shader_source.wgsl");

pub const sky_shader_source = @embedFile("shaders/sky_shader_source.wgsl");

// Environment cubemap bake. Renders one cube face/mip per draw into an HDR cubemap that the
// mesh shader samples for image-based lighting. mode 0 = procedural studio (from the same
// sky/studio params as the analytic path); mode 1 = GGX-prefiltered from a bound equirect HDRI.
pub const env_bake_shader_source = @embedFile("shaders/env_bake_shader_source.wgsl");

// ── Post-processing ─────────────────────────────────────────────────────────────
// The 3D scene renders into a linear rgba16float buffer; these passes turn that HDR
// buffer into the final LDR image. Bloom: a soft-knee bright-pass fused with a separable
// gaussian (mode 0 = prefilter+horizontal from the scene buffer, mode 1 = vertical from
// the half-res bloom buffer). Tonemap: exposure → +bloom → ACES → gamma → grade.

// Mip-chain bloom (COD/Siggraph-2014 style). A downsample chain (13-tap, soft-knee prefilter +
// Karis luma-average firefly clamp on the first mip) followed by an additive tent upsample chain.
// Each pass binds a single-level view of one mip, so textureDimensions(src_tex, 0) gives that
// mip's size → no per-pass texel uniform. Output is read by the tonemap pass at chain mip1.

pub const bloom_down_shader_source = @embedFile("shaders/bloom_down_shader_source.wgsl");

pub const bloom_up_shader_source = @embedFile("shaders/bloom_up_shader_source.wgsl");

// Screen-space ambient occlusion. Reads the view-space position G-buffer (rgba16float,
// w=1 on geometry, 0 on background), reconstructs the face normal from screen derivatives,
// and accumulates occlusion over a hemisphere kernel rotated by a per-pixel hash. Output is
// a single AO factor in [0,1] (1 = unoccluded) written to all channels.
pub const ssao_shader_source = @embedFile("shaders/ssao_shader_source.wgsl");

// Screen-space reflections. Reflects the lit HDR scene off the view-position G-buffer:
// reconstructs the face normal, reflects the view ray, ray-marches in view space, and on a
// depth intersection samples the scene colour at the hit. Output is the (pre-weighted)
// reflection colour in rgb + confidence in a, added by the tonemap pass.
pub const ssr_shader_source = @embedFile("shaders/ssr_shader_source.wgsl");

// Temporal anti-aliasing. Reconstructs each pixel's world position from the view-position
// G-buffer, reprojects it through the PREVIOUS frame's view-projection to fetch the history
// colour, neighborhood-clamps that history to the current 3x3 colour range (kills ghosting),
// and blends. Combined with a per-frame sub-pixel projection jitter, accumulated frames
// anti-alias the image (including the post-MSAA bits: SSR, specular, softbox highlights).
pub const taa_shader_source = @embedFile("shaders/taa_shader_source.wgsl");

pub const tonemap_shader_source = @embedFile("shaders/tonemap_shader_source.wgsl");

// Auto-exposure metering: average scene luminance → temporally-adapted exposure multiplier (1×1).
pub const exposure_shader_source = @embedFile("shaders/exposure_shader_source.wgsl");

// Depth of field — gather bokeh on the linear-HDR scene before tonemap. Per-pixel circle of
// confusion is derived from the view-position G-buffer depth vs the focus distance; a 32-tap
// golden-angle disk is gathered with a scatter-as-gather weight (a sample contributes only if its
// own CoC disc reaches the offset) so a sharp foreground doesn't smear onto a blurred background.
pub const dof_shader_source = @embedFile("shaders/dof_shader_source.wgsl");
