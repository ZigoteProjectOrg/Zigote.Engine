// Directional cascaded-shadow depth pass. The renderer issues one pass per cascade, each binding a
// different cascade matrix through a dynamic-offset UBO (group 2) and rendering scene depth into that
// cascade's array layer. Only the model transform (group 1) and the cascade matrix are needed; the
// camera/light UBO (group 0) stays in the pipeline layout but is unused here.
struct ModelUniforms {
  model: mat4x4<f32>,
  normal_mat: mat4x4<f32>,
  base_color: vec4<f32>,
  metallic_roughness: vec4<f32>,
  emissive: vec4<f32>,
  effect: vec4<u32>,
}
struct ShadowCascade {
  view_proj: mat4x4<f32>,
}
@group(1) @binding(0) var<uniform> model_data: ModelUniforms;
@group(2) @binding(0) var<uniform> cascade: ShadowCascade;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
  let world_pos = model_data.model * vec4<f32>(position, 1.0);
  return cascade.view_proj * world_pos;
}
