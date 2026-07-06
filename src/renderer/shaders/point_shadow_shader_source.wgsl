// Omnidirectional point-light shadow pass. The renderer issues 6 passes per shadow-casting point
// light (one per cube face), each binding that face's view-proj + the light position/range through a
// dynamic-offset UBO (group 2). The fragment writes the LINEAR distance from the light (normalised by
// range) into frag_depth, so the mesh shader can sample the depth cube by direction and hardware-PCF
// compare against `distance(frag, light)/range`. Group 0 (camera/light) stays in the layout for
// binding parity but is unused here; group 1 = model.
struct ModelUniforms {
  model: mat4x4<f32>,
  normal_mat: mat4x4<f32>,
  base_color: vec4<f32>,
  metallic_roughness: vec4<f32>,
  emissive: vec4<f32>,
  effect: vec4<u32>,
}
struct PointShadow {
  view_proj: mat4x4<f32>,
  light_pos_range: vec4<f32>, // xyz = light world position, w = range
}
@group(1) @binding(0) var<uniform> model_data: ModelUniforms;
@group(2) @binding(0) var<uniform> ps: PointShadow;

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world: vec3<f32>,
}

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VOut {
  let wp = (model_data.model * vec4<f32>(position, 1.0)).xyz;
  var o: VOut;
  o.clip = ps.view_proj * vec4<f32>(wp, 1.0);
  o.world = wp;
  return o;
}

@fragment
fn fs_main(in: VOut) -> @builtin(frag_depth) f32 {
  let d = length(in.world - ps.light_pos_range.xyz) / max(ps.light_pos_range.w, 0.001);
  return clamp(d, 0.0, 1.0);
}
