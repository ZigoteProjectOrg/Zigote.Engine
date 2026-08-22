@group(0) @binding(0) var image_tex: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) position: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.uv = uv;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  // The sampled texel is already linear (sRGB texture view decodes on sample); the tint arrives
  // sRGB-encoded like every UI color, so decode it before multiplying (see shape shader).
  let c = textureSample(image_tex, image_sampler, in.uv);
  let tint = pow(max(in.color.rgb, vec3<f32>(0.0)), vec3<f32>(2.2));
  var out = vec4<f32>(c.rgb * tint, c.a * in.color.a);
  out.a = out.a * rounded_clip_coverage(in.position.xy);
  return out;
}
