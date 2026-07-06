struct CameraUniforms {
  view_proj: mat4x4<f32>,
  view: mat4x4<f32>,
  camera_pos: vec4<f32>,
  inv_view_proj: mat4x4<f32>,
}
struct LightData { position_or_dir: vec4<f32>, color_range: vec4<f32> }
struct LightUniforms {
  view_proj: mat4x4<f32>,
  ambient_color: vec4<f32>,
  light_count: vec4<u32>,
  sky_horizon: vec4<f32>,
  sky_zenith: vec4<f32>,
  sky_ground: vec4<f32>,
  env_avg: vec4<f32>,
  sun_dir: vec4<f32>,
  studio: vec4<f32>,
  post: vec4<f32>,
  shadow: vec4<f32>,
  lights: array<LightData, 16>,
  debug: vec4<u32>,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;
@group(1) @binding(2) var env_cube: texture_cube<f32>;
@group(1) @binding(3) var env_samp: sampler;

struct SkyOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) vi: u32) -> SkyOut {
  let x = select(-1.0, 3.0, vi == 1u);
  let y = select(-1.0, 3.0, vi == 2u);
  var out: SkyOut;
  out.pos = vec4<f32>(x, y, 1.0, 1.0);
  out.ndc = vec2<f32>(x, y);
  return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
  // Reconstruct the world-space view ray per pixel (inverse view-proj of the far plane) and
  // sample the environment cubemap as the visible backdrop. The background is then the SAME lit
  // environment (procedural studio or HDRI) the car reflects, so they agree — the EEVEE "world"
  // look instead of a flat painted gradient. A gentle mip bias softens busy HDRIs into a clean,
  // slightly defocused studio backdrop. The sky writes linear HDR; AgX tonemaps it downstream.
  let far = camera.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
  let world = far.xyz / far.w;
  let dir = normalize(world - camera.camera_pos.xyz);
  let env_max_lod = light.post.w;
  let col = textureSampleLevel(env_cube, env_samp, dir, env_max_lod * 0.30).rgb;
  return vec4<f32>(col, 1.0);
}
