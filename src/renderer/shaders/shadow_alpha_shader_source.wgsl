// Alpha-tested variant of the directional/spot cascade depth pass. Masked materials (foliage, decals,
// chain-link) must cast cut-out shadows: the plain depth-only `shadow_shader` can't, so those casters
// were skipped entirely. This variant carries UV through to a fragment stage that samples the base
// colour and `discard`s where the (factor × texture) alpha is below the cutoff, so only the opaque
// texels write depth. Layout matches `shadow_shader` (camera/model/cascade) plus the material texture
// group (3) reused from the main pipeline's `texture_bgl`, so a caster's existing material bind group
// binds directly.
struct ModelUniforms {
  model: mat4x4<f32>,
  normal_mat: mat4x4<f32>,
  base_color: vec4<f32>,
  metallic_roughness: vec4<f32>,
  emissive: vec4<f32>,
  effect: vec4<u32>, // .z = alpha mode, .w = alpha cutoff (bitcast f32)
}
struct ShadowCascade {
  view_proj: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> model_data: ModelUniforms;
@group(2) @binding(0) var<uniform> cascade: ShadowCascade;
@group(3) @binding(0) var base_color_tex: texture_2d<f32>;
@group(3) @binding(1) var base_color_sampler: sampler;

struct VOut {
  @builtin(position) clip_pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
  let world_pos = model_data.model * vec4<f32>(position, 1.0);
  var out: VOut;
  out.clip_pos = cascade.view_proj * world_pos;
  out.uv = uv;
  return out;
}

@fragment
fn fs_main(in: VOut) {
  let a = model_data.base_color.a * textureSample(base_color_tex, base_color_sampler, in.uv).a;
  let cutoff = bitcast<f32>(model_data.effect.w);
  if (a < cutoff) { discard; }
}
