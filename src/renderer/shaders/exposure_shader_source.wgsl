// Auto-exposure metering pass. Measures the frame's geometric-mean (log-average) luminance by
// sampling a stratified grid across the linear-HDR scene, converts it to an exposure multiplier
// (key / avg), clamps the metered luminance to a sensible range, and temporally adapts toward the
// target against the previous frame's value (1×1 history texture) for smooth eye-adaptation.
// Writes a 1×1 target: r = adapted exposure multiplier, g = raw metered luminance (debug).
struct ExposureParams {
  cfg: vec4<f32>,  // x=key, y=min_lum, z=max_lum, w=speed (per-frame blend)
  cfg2: vec4<f32>, // x=history_valid, yzw unused
}
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var hist_tex: texture_2d<f32>; // 1×1, previous adapted multiplier in .r
@group(0) @binding(3) var<uniform> p: ExposureParams;

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  // 16×16 stratified taps over the whole frame. Center-weighted (Gaussian-ish) so the subject the
  // camera is pointed at drives exposure more than the periphery — standard center-weighted metering.
  var log_sum = 0.0;
  var wsum = 0.0;
  let N = 16;
  for (var yy = 0; yy < N; yy = yy + 1) {
    for (var xx = 0; xx < N; xx = xx + 1) {
      let uv = (vec2<f32>(f32(xx), f32(yy)) + 0.5) / f32(N);
      let d = uv - vec2<f32>(0.5);
      let w = exp(-dot(d, d) * 3.0); // center weight
      let c = textureSampleLevel(scene_tex, samp, uv, 0.0).rgb;
      let lum = max(dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-5);
      log_sum = log_sum + log2(lum) * w;
      wsum = wsum + w;
    }
  }
  let avg_lum = exp2(log_sum / max(wsum, 1e-4)); // weighted geometric mean
  let clamped = clamp(avg_lum, p.cfg.y, p.cfg.z);
  let target_mult = p.cfg.x / max(clamped, 1e-4);
  let prev = textureSampleLevel(hist_tex, samp, vec2<f32>(0.5), 0.0).r;
  // Snap to target on the first valid frame / after a resize; otherwise ease toward it.
  let blend = select(1.0, clamp(p.cfg.w, 0.0, 1.0), p.cfg2.x > 0.5);
  let adapted = mix(prev, target_mult, blend);
  return vec4<f32>(adapted, avg_lum, 0.0, 1.0);
}
