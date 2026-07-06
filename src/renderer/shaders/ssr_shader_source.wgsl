struct SsrParams {
  proj: mat4x4<f32>,  // view → clip
  cfg: vec4<f32>,     // x=intensity, y=max_distance, z=thickness, w=step_count
}
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var pos_tex: texture_2d<f32>;
@group(0) @binding(2) var normal_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> p: SsrParams;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  var o: VOut;
  o.uv = vec2<f32>(x, y);
  o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return o;
}

fn to_uv(view_pos: vec3<f32>) -> vec3<f32> {
  let clip = p.proj * vec4<f32>(view_pos, 1.0);
  let ndc = clip.xyz / clip.w;
  return vec3<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5, clip.w);
}

// Sample the reflected scene as a roughness-scaled glossy bokeh. A per-pixel-rotated golden-angle
// (Vogel) disk gives a round, filled kernel (vs the old 4-tap axis cross, which smeared highlights
// into a '+'), aspect-corrected so the disk is round in PIXELS on a non-square frame, and each HDR
// sample is firefly-clamped: SSR is composited additively in linear HDR before tonemap, so a single
// hot reflected texel (sun glint / emissive) would otherwise sparkle and bloom the frame.
fn sample_reflection(uv: vec2<f32>, radius: f32) -> vec3<f32> {
  let hdr_max = vec3<f32>(8.0); // firefly ceiling (~3 stops over diffuse white); min() also tames NaN/Inf
  if (radius < 0.0008) { return min(textureSampleLevel(scene_tex, samp, uv, 0.0).rgb, hdr_max); }
  let dim = vec2<f32>(textureDimensions(scene_tex, 0));
  let rscale = vec2<f32>(1.0, dim.x / dim.y); // round disk in pixels (no-op on a square viewport)
  // Per-pixel rotation (seed from pixel-scaled uv so neighbours decorrelate) breaks axis-aligned ringing.
  let rot = 6.2831853 * fract(52.9829189 * fract(dot(uv * 2048.0, vec2<f32>(0.06711056, 0.00583715))));
  let ga = 2.39996323; // golden angle (radians)
  var c = min(textureSampleLevel(scene_tex, samp, uv, 0.0).rgb, hdr_max);
  var wsum = 1.0;
  for (var k: i32 = 0; k < 8; k = k + 1) {
    let fk = f32(k);
    let r = radius * sqrt((fk + 0.5) / 8.0); // sqrt spacing FILLS the disk (not bunched at the rim)
    let a = ga * fk + rot;
    let o = vec2<f32>(cos(a), sin(a)) * r * rscale;
    let w = 1.0 - 0.6 * (fk / 8.0); // soft radial falloff (feathered edge)
    c += min(textureSampleLevel(scene_tex, samp, uv + o, 0.0).rgb, hdr_max) * w;
    wsum += w;
  }
  return c / wsum;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let g = textureSampleLevel(pos_tex, samp, in.uv, 0.0);
  if (g.w < 0.5) { return vec4<f32>(0.0); } // background
  let nr = textureSampleLevel(normal_tex, samp, in.uv, 0.0);
  let roughness = nr.w;
  // Roughness-aware: smooth → glossy surfaces get SSR (mirror chrome through glossy paint). Beyond
  // ~0.45 the glossy disk under-samples into low-frequency haze, so hand off to the prefiltered env
  // cubemap (already applied by the forward pass), which covers that range better.
  let smoothness = 1.0 - smoothstep(0.05, 0.45, roughness);
  if (smoothness <= 0.001) { return vec4<f32>(0.0); }
  let pos = g.xyz; // view space (camera at origin, looking -z)
  var n = normalize(nr.xyz);
  if (dot(n, -pos) < 0.0) { n = -n; }
  let view_dir = normalize(pos);
  let refl = normalize(reflect(view_dir, n));
  // Reflections pointing back at the camera barely move on screen and self-intersect. Hard-cull only
  // the steep cases; the near-grazing band is faded smoothly in `conf` below to avoid a hard seam.
  if (refl.z > 0.25) { return vec4<f32>(0.0); }

  let steps = i32(clamp(p.cfg.w, 1.0, 64.0));
  // Geometric step schedule: dense near the reflector (fine contact reflections, where 1 view-unit
  // covers many screen pixels), coarse far away. Scaled so total reach still equals max_distance
  // (growth→1 degenerates to the old uniform stepping). Spends the step budget where it's perceptible.
  let growth = 1.08;
  let geo_sum = (pow(growth, f32(steps)) - 1.0) / (growth - 1.0);
  let step_len = p.cfg.y / geo_sum;
  var cur_len = step_len;
  // Interleaved-gradient-noise jitter on the START offset breaks the depth-quantisation banding on
  // curved reflectors; the binary refine below lands the exact crossing.
  let ign = fract(52.9829189 * fract(dot(in.pos.xy, vec2<f32>(0.06711056, 0.00583715))));
  var prev = pos + refl * step_len * ign; // last point KNOWN to be in front of geometry
  var march = prev;
  var result = vec3<f32>(0.0);
  var conf = 0.0;
  for (var i: i32 = 0; i < steps; i = i + 1) {
    march = march + refl * cur_len;
    cur_len = cur_len * growth; // grow BEFORE the continues below, or far-field reach collapses
    let uvw = to_uv(march);
    if (uvw.z <= 0.0) { break; }
    if (uvw.x < 0.0 || uvw.x > 1.0 || uvw.y < 0.0 || uvw.y > 1.0) { break; }
    let scene_g = textureSampleLevel(pos_tex, samp, uvw.xy, 0.0);
    if (scene_g.w < 0.5) { prev = march; continue; } // sky: ray still in front → advance bracket
    let diff = scene_g.z - march.z; // >0: geometry nearer camera than the ray → ray is behind it
    if (diff <= 0.0) { prev = march; continue; } // ray in front of geometry → advance the bracket
    // Depth-proportional acceptance window: a constant view-space thickness is too thick near the
    // camera (false hits / smear) and too thin far away (missed hits). Scaling by distance-from-camera
    // makes it perceptually constant in screen space; sqrt-damped so the far window can't straddle a
    // depth discontinuity (which would smear/firefly under the additive-HDR composite).
    let thick = p.cfg.z * sqrt(max(1.0, -march.z));
    if (diff < thick) {
      // Binary-search refine between the last front point (prev) and this behind point (march):
      // 6 halvings converge on the exact intersection so the reflection is smooth, not stepped.
      var lo = prev;
      var hi = march;
      for (var b: i32 = 0; b < 6; b = b + 1) {
        let mid = (lo + hi) * 0.5;
        let mg = textureSampleLevel(pos_tex, samp, to_uv(mid).xy, 0.0);
        if (mg.w > 0.5 && (mg.z - mid.z) > 0.0) { hi = mid; } else { lo = mid; }
      }
      let huv = to_uv(hi);
      // Reject refined hits that straddled a silhouette: the converged uv can project onto a different
      // (foreground/background) surface than the one bracketed, leaving the depth gap > thickness —
      // accepting it smears edge-coloured streaks along silhouettes.
      let hg = textureSampleLevel(pos_tex, samp, huv.xy, 0.0);
      if (hg.w < 0.5) { break; }
      let hdiff = hg.z - hi.z;
      if (hdiff < 0.0 || hdiff >= thick) { break; }
      // Per-edge screen fade (corners fade fastest; a wider band on the top edge, huv.y→0, where
      // world-up reflections exit) so reflections don't pop at screen borders.
      let edge = smoothstep(0.0, 0.10, huv.x)
               * smoothstep(0.0, 0.10, 1.0 - huv.x)
               * smoothstep(0.0, 0.22, huv.y)
               * smoothstep(0.0, 0.10, 1.0 - huv.y);
      // Physical Schlick Fresnel (F0=0.04, exponent 5 — same as the forward pass). No 0.25 head-on
      // floor, so SSR does NOT re-add the broad reflection the env IBL already gave the surface.
      let f0 = 0.04;
      let fres = f0 + (1.0 - f0) * pow(1.0 - max(dot(n, -view_dir), 0.0), 5.0);
      // Glossy blur: wider for rougher surfaces and for longer reflection rays (cone widening).
      let blur = roughness * 0.05 * (0.3 + 0.7 * (f32(i) / f32(steps)));
      result = sample_reflection(huv.xy, blur);
      // Ray-distance fade (distance-based, so it is correct under the geometric step schedule): long
      // marches are the least trustworthy (depth error + thickness false-positives) and pop at the
      // range limit — fade them so the env cube cleanly owns distant reflections.
      let dist_t = clamp(length(hi - pos) / max(p.cfg.y, 1e-3), 0.0, 1.0);
      let dist_fade = 1.0 - dist_t * dist_t;
      // Smooth toward-camera fade (meets the refl.z>0.25 hard cull at its zero crossing → no seam).
      let cam_fade = 1.0 - smoothstep(-0.05, 0.25, refl.z);
      conf = edge * fres * smoothness * dist_fade * cam_fade * p.cfg.x;
      break;
    } else {
      // diff >= thickness: ray passed BEHIND a thick occluder. Re-anchor the front bracket here so a
      // later hit refines against a recent sample instead of bisecting across the discontinuity
      // (which smeared the wrong surface onto silhouettes).
      prev = march;
    }
  }
  return vec4<f32>(result * conf, conf);
}
