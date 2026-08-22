struct TaaParams {
  prev_view_proj: mat4x4<f32>,
  cur_view_proj: mat4x4<f32>,  // current frame UNJITTERED — for jitter-free motion vectors
  inv_view: mat4x4<f32>,
  cfg: vec4<f32>, // x=feedback, y=enabled, z=history_valid, w unused
}
@group(0) @binding(0) var cur_tex: texture_2d<f32>;
@group(0) @binding(1) var history_tex: texture_2d<f32>;
@group(0) @binding(2) var pos_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> p: TaaParams;

// Sharp 5-tap Catmull-Rom (bicubic) history sample (Karis 2014). Sampling reprojected history with
// plain bilinear softens the image a little every frame — under camera/object motion that compounds
// into the "TAA is blurry" look. Catmull-Rom reconstructs the history with a sharpening kernel so the
// accumulated image stays crisp. 5 bilinear fetches via the standard weight factorisation.
fn sampleHistoryCR(uv: vec2<f32>, tex_size: vec2<f32>) -> vec3<f32> {
  let sample_pos = uv * tex_size;
  let tex_pos1 = floor(sample_pos - 0.5) + 0.5;
  let f = sample_pos - tex_pos1;
  let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
  let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
  let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
  let w3 = f * f * (-0.5 + 0.5 * f);
  let w12 = w1 + w2;
  let offset12 = w2 / w12;
  let p0 = (tex_pos1 - 1.0) / tex_size;
  let p3 = (tex_pos1 + 2.0) / tex_size;
  let p12 = (tex_pos1 + offset12) / tex_size;
  var r = vec3<f32>(0.0);
  r += textureSampleLevel(history_tex, samp, vec2<f32>(p12.x, p0.y), 0.0).rgb * (w12.x * w0.y);
  r += textureSampleLevel(history_tex, samp, vec2<f32>(p0.x, p12.y), 0.0).rgb * (w0.x * w12.y);
  r += textureSampleLevel(history_tex, samp, vec2<f32>(p12.x, p12.y), 0.0).rgb * (w12.x * w12.y);
  r += textureSampleLevel(history_tex, samp, vec2<f32>(p3.x, p12.y), 0.0).rgb * (w3.x * w12.y);
  r += textureSampleLevel(history_tex, samp, vec2<f32>(p12.x, p3.y), 0.0).rgb * (w12.x * w3.y);
  let wsum = (w12.x * w0.y) + (w0.x * w12.y) + (w12.x * w12.y) + (w3.x * w12.y) + (w12.x * w3.y);
  return max(r / max(wsum, 0.0001), vec3<f32>(0.0));
}

// YCoCg is a cheap luma/chroma space; clipping the history AABB here (rather than in RGB) keeps a
// tight bound on luma while tolerating more chroma drift — far less colour fringing on gradients.
fn rgb2ycocg(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn ycocg2rgb(c: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
// Clip history `q` into the [aabb_min, aabb_max] box along the ray from the box centre (a hard
// "clip" toward the current-frame distribution, not a per-channel clamp — preserves hue better).
fn clipAABB(aabb_min: vec3<f32>, aabb_max: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
  let center = 0.5 * (aabb_max + aabb_min);
  let extent = 0.5 * (aabb_max - aabb_min) + vec3<f32>(1e-5);
  let v = q - center;
  let uv = v / extent;
  let ma = max(abs(uv.x), max(abs(uv.y), abs(uv.z)));
  if (ma > 1.0) { return center + v / ma; }
  return q;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let cur = textureSampleLevel(cur_tex, samp, in.uv, 0.0).rgb;
  if (p.cfg.y < 0.5 || p.cfg.z < 0.5) { return vec4<f32>(cur, 1.0); }
  let g = textureSampleLevel(pos_tex, samp, in.uv, 0.0);
  if (g.w < 0.5) { return vec4<f32>(cur, 1.0); } // background — no reprojection
  let world = (p.inv_view * vec4<f32>(g.xyz, 1.0)).xyz;
  let pc = p.prev_view_proj * vec4<f32>(world, 1.0);
  let cc = p.cur_view_proj * vec4<f32>(world, 1.0);
  if (pc.w <= 0.0 || cc.w <= 0.0) { return vec4<f32>(cur, 1.0); }
  let prev_uv = vec2<f32>((pc.x / pc.w) * 0.5 + 0.5, -(pc.y / pc.w) * 0.5 + 0.5);
  let cur_uv  = vec2<f32>((cc.x / cc.w) * 0.5 + 0.5, -(cc.y / cc.w) * 0.5 + 0.5);
  // Sample history by the screen-space MOTION (prev-cur unjittered), added to this pixel's
  // uv. A static camera → zero motion → exact pixel fetch (no jitter-resampling blur).
  let motion = prev_uv - cur_uv;
  let puv = in.uv + motion;
  if (puv.x < 0.0 || puv.x > 1.0 || puv.y < 0.0 || puv.y > 1.0) { return vec4<f32>(cur, 1.0); }
  let tex_size = vec2<f32>(textureDimensions(cur_tex, 0));
  var hist = sampleHistoryCR(puv, tex_size);
  let texel = 1.0 / tex_size;
  // Variance neighbourhood clip (Salvi/INSIDE) in YCoCg: estimate the local colour distribution's
  // mean+stddev over the 3x3, form an AABB at mean ± γ·σ, and clip the reprojected history into it.
  // Rejects stale/disoccluded history like the old min/max box but is tighter (less ghosting on
  // subtle gradients) and clips in luma/chroma so it doesn't introduce colour fringes.
  var m1 = vec3<f32>(0.0);
  var m2 = vec3<f32>(0.0);
  // Cross-tap RGB stashed during the 3x3 loop so the unsharp blur below reuses them (no re-fetch).
  var tap_px = vec3<f32>(0.0);
  var tap_nx = vec3<f32>(0.0);
  var tap_py = vec3<f32>(0.0);
  var tap_ny = vec3<f32>(0.0);
  for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
      let rgb = textureSampleLevel(cur_tex, samp, in.uv + vec2<f32>(f32(dx), f32(dy)) * texel, 0.0).rgb;
      if (dx == 1 && dy == 0) { tap_px = rgb; }
      else if (dx == -1 && dy == 0) { tap_nx = rgb; }
      else if (dx == 0 && dy == 1) { tap_py = rgb; }
      else if (dx == 0 && dy == -1) { tap_ny = rgb; }
      let c = rgb2ycocg(rgb);
      m1 += c;
      m2 += c * c;
    }
  }
  let mean = m1 / 9.0;
  let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3<f32>(0.0)));
  let gamma = 1.0;
  hist = ycocg2rgb(clipAABB(mean - gamma * sigma, mean + gamma * sigma, rgb2ycocg(hist)));
  // Velocity-aware blend: keep a high history weight when the pixel is (near-)static so jittered
  // samples accumulate into crisp anti-aliasing, but drop the history weight as screen-space motion
  // grows so moving content stays sharp instead of smearing. motion is in UV; scale to pixels.
  let motion_px = length(motion * tex_size);
  let blend = mix(p.cfg.x, 0.5, clamp(motion_px * 0.5, 0.0, 1.0));
  var resolved = mix(cur, hist, blend);
  // Mild unsharp against the 4-neighbour average to counteract the slight softening the temporal
  // resolve + Catmull-Rom history introduce — keeps the accumulated image crisp.
  let blur = (tap_px + tap_nx + tap_py + tap_ny) * 0.25;
  resolved = max(resolved + (cur - blur) * 0.20, vec3<f32>(0.0));
  return vec4<f32>(resolved, 1.0);
}
