struct DownParams { cfg: vec4<f32> } // x=is_first(1=prefilter+karis), y=threshold, z=knee
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> p: DownParams;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
  let x = f32((vid << 1u) & 2u); let y = f32(vid & 2u);
  var o: VOut; o.uv = vec2<f32>(x, y);
  o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0); return o;
}
// NaN/Inf guard: drop non-finite, clamp below the fp16 max so one firefly can't poison a tile.
fn san(c: vec3<f32>) -> vec3<f32> {
  var v = select(c, vec3<f32>(0.0), c != c);
  return clamp(v, vec3<f32>(0.0), vec3<f32>(65000.0));
}
fn prefilter(c: vec3<f32>) -> vec3<f32> {
  let br = max(c.r, max(c.g, c.b));
  let knee = max(p.cfg.z, 0.0001); let t = p.cfg.y;
  let soft = clamp(br - t + knee, 0.0, 2.0 * knee);
  let soft2 = soft * soft / (4.0 * knee + 0.0001);
  let contrib = max(soft2, br - t) / max(br, 0.0001);
  return c * max(contrib, 0.0);
}
fn karis(c: vec3<f32>) -> f32 {
  let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return 1.0 / (1.0 + l);
}
@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let t = 1.0 / vec2<f32>(textureDimensions(src_tex, 0));
  let a = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-2.0*t.x,  2.0*t.y), 0.0).rgb);
  let b = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 0.0,      2.0*t.y), 0.0).rgb);
  let c = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 2.0*t.x,  2.0*t.y), 0.0).rgb);
  let d = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-2.0*t.x,  0.0    ), 0.0).rgb);
  let e = san(textureSampleLevel(src_tex, src_samp, in.uv,                                  0.0).rgb);
  let f = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 2.0*t.x,  0.0    ), 0.0).rgb);
  let g = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-2.0*t.x, -2.0*t.y), 0.0).rgb);
  let h = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 0.0,     -2.0*t.y), 0.0).rgb);
  let ii = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 2.0*t.x, -2.0*t.y), 0.0).rgb);
  let j = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-1.0*t.x,  1.0*t.y), 0.0).rgb);
  let k = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 1.0*t.x,  1.0*t.y), 0.0).rgb);
  let l = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>(-1.0*t.x, -1.0*t.y), 0.0).rgb);
  let m = san(textureSampleLevel(src_tex, src_samp, in.uv + vec2<f32>( 1.0*t.x, -1.0*t.y), 0.0).rgb);
  var col: vec3<f32>;
  if (p.cfg.x > 0.5) {
    // First mip: Karis luma-average per box-group to kill fireflies, then soft-knee prefilter.
    let g0 = (j + k + l + m) * 0.25;
    let g1 = (a + b + d + e) * 0.25;
    let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25;
    let g4 = (e + f + h + ii) * 0.25;
    let kw0 = 0.5   * karis(g0);
    let kw1 = 0.125 * karis(g1);
    let kw2 = 0.125 * karis(g2);
    let kw3 = 0.125 * karis(g3);
    let kw4 = 0.125 * karis(g4);
    let sumw = kw0 + kw1 + kw2 + kw3 + kw4;
    col = (g0*kw0 + g1*kw1 + g2*kw2 + g3*kw3 + g4*kw4) / max(sumw, 0.0001);
    col = prefilter(col);
  } else {
    col  = e * 0.125;
    col += (a + c + g + ii) * 0.03125;
    col += (b + d + f + h) * 0.0625;
    col += (j + k + l + m) * 0.125;
  }
  return vec4<f32>(san(col), 1.0);
}
