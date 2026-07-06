@group(0) @binding(0) var text_tex: texture_2d<f32>;
@group(0) @binding(1) var text_sampler: sampler;

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
  // Atlas texels are grayscale coverage rasterized at the output's physical pixel size.
  // Coverage gamma (~1/1.2): the swapchain is sRGB and alpha is blended in non-linear space, which
  // makes edge pixels perceptually too dark — light text reads thin, dark text heavy. Lifting the
  // coverage counters that stem-thinning (the standard cheap fix when stuck with sRGB-space AA).
  let coverage = pow(textureSample(text_tex, text_sampler, in.uv).r, 1.0 / 1.2);
  return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
