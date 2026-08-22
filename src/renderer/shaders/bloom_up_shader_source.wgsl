@group(0) @binding(0) var src_tex: texture_2d<f32>; // the coarser (smaller) mip
@group(0) @binding(1) var src_samp: sampler;
fn san(c: vec3<f32>) -> vec3<f32> {
  var v = select(c, vec3<f32>(0.0), c != c);
  return clamp(v, vec3<f32>(0.0), vec3<f32>(65000.0));
}
@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let o = 0.5 / vec2<f32>(textureDimensions(src_tex, 0)); // half-texel offset: 4 bilinear taps == 3x3 tent
  let t00 = textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-o.x, -o.y), 0.0).rgb;
  let t10 = textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( o.x, -o.y), 0.0).rgb;
  let t01 = textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-o.x,  o.y), 0.0).rgb;
  let t11 = textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( o.x,  o.y), 0.0).rgb;
  let col = (t00 + t10 + t01 + t11) * 0.25;
  return vec4<f32>(san(col), 1.0); // ADDED into the finer mip by the one-one blend state
}
