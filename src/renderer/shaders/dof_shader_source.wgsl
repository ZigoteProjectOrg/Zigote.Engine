struct DofParams {
  texel: vec2<f32>,   // 1/width, 1/height
  cfg: vec2<f32>,     // x=focus_distance (view units), y=max_coc (pixels)
  cfg2: vec4<f32>,    // x=coc_scale, y=unused, z=near_blur_boost, w=enabled (0/1)
  bokeh: vec4<f32>,   // x=blades (0/<3 = circular), y=anamorphic (1 = round), zw unused
}
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var pos_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> p: DofParams;

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

// View-space linear depth (camera looks -z, so depth = -z). Background (w<0.5) reads as far.
fn linear_depth(g: vec4<f32>) -> f32 {
  if (g.w < 0.5) { return 1.0e6; }
  return max(-g.z, 1.0e-4);
}

// Centre-spot autofocus: lock the focal plane onto whatever geometry sits at screen centre
// (the subject), averaged over a tiny central cross for stability. As the camera orbits, the
// focus tracks the subject so it stays sharp — no fixed focus distance. Falls back to the
// uniform focus (cfg.x) only when the centre is background (sky).
fn autofocus() -> f32 {
  var sum = 0.0;
  var n = 0.0;
  var pts = array<vec2<f32>, 5>(
    vec2<f32>(0.5, 0.5),
    vec2<f32>(0.47, 0.5), vec2<f32>(0.53, 0.5),
    vec2<f32>(0.5, 0.47), vec2<f32>(0.5, 0.53));
  for (var i: i32 = 0; i < 5; i = i + 1) {
    let g = textureSampleLevel(pos_tex, samp, pts[i], 0.0);
    if (g.w > 0.5) { sum = sum + max(-g.z, 1.0e-4); n = n + 1.0; }
  }
  if (n > 0.0) { return sum / n; }
  return p.cfg.x;
}

// Signed CoC in pixels (negative = near / in front of focus). A focus dead-band keeps a depth
// slab around the subject tack-sharp (the whole car, not just the focal plane) before the blur
// ramps up toward the background. Magnitude clamped to max_coc.
fn coc_px(depth: f32, focus: f32) -> f32 {
  let maxc = p.cfg.y;
  let signed_raw = (depth - focus) / max(depth, 1.0e-4);
  let dead = 0.22; // ±22% relative depth around focus stays sharp (keeps the whole subject crisp)
  let signed = sign(signed_raw) * max(abs(signed_raw) - dead, 0.0);
  var c = p.cfg2.x * signed;
  if (c < 0.0) { c = c * p.cfg2.z; }
  return clamp(c, -maxc, maxc);
}

const TAPS: i32 = 32;
const GOLDEN: f32 = 2.39996323;

// Regular N-gon radius at an angle, normalised so vertices touch the unit circle (edges pulled in to the
// apothem cos(pi/N)). Shaping the gather disc by this turns round bokeh into a polygonal aperture.
fn ngon_scale(angle: f32, blades: f32) -> f32 {
  if (blades < 3.0) { return 1.0; } // circular aperture
  let seg = 6.28318531 / blades;
  let half = seg * 0.5;
  let folded = angle - floor(angle / seg + 0.5) * seg; // into [-half, half]
  return cos(half) / cos(folded);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let center = textureSampleLevel(scene_tex, samp, in.uv, 0.0).rgb;
  if (p.cfg2.w < 0.5) { return vec4<f32>(center, 1.0); } // DoF disabled — passthrough

  let focus = autofocus(); // adaptive focal plane locked on the central subject
  let gc = textureSampleLevel(pos_tex, samp, in.uv, 0.0);
  let center_depth = linear_depth(gc);
  let center_coc = coc_px(center_depth, focus);
  let center_coc_abs = abs(center_coc);
  // In-focus pixels return the SHARP centre unaltered (no gather → no 1px softening). Safe to
  // early-out because every sample uses explicit-LOD textureSampleLevel (no implicit-LOD UB).
  if (center_coc_abs < 1.5) { return vec4<f32>(center, 1.0); }
  let radius_px = center_coc_abs; // gather disk radius in pixels = this pixel's CoC
  // Adaptive taps: scale the sample count with the blur radius so a tight CoC spends few taps
  // while a wide CoC keeps full quality. Divide the spiral by the ACTUAL count (not TAPS) so the
  // disc always fills radius_px — using TAPS here would shrink the silhouette as taps drop.
  let taps = clamp(i32(ceil(radius_px * 1.8)), 8, TAPS);

  var sum = center;
  var weight = 1.0;
  for (var i: i32 = 0; i < taps; i = i + 1) {
    let fi = f32(i);
    let r = sqrt((fi + 0.5) / f32(taps)) * radius_px; // golden-angle spiral, even areal coverage
    let a = fi * GOLDEN;
    // Aperture shape: fold the disc into an N-gon (polygonal blades) and stretch vertically (anamorphic).
    let shaped_r = r * ngon_scale(a, p.bokeh.x);
    let dir = vec2<f32>(cos(a), sin(a) * max(p.bokeh.y, 1.0));
    let offs = dir * shaped_r * p.texel;
    let suv = in.uv + offs;
    let sc = textureSampleLevel(scene_tex, samp, suv, 0.0).rgb;
    let sg = textureSampleLevel(pos_tex, samp, suv, 0.0);
    let s_depth = linear_depth(sg);
    let s_coc = abs(coc_px(s_depth, focus));
    // Scatter-as-gather: a sample contributes only if its bokeh disc reaches this offset, so a
    // sharp foreground (small disc) can't smear onto a blurred background, while a blurred
    // foreground (disc >= r) spills onto the in-focus background behind it.
    let sample_reach = max(s_coc, 1.0);
    var w = clamp(sample_reach - r + 1.0, 0.0, 1.0);
    // Depth tie-break: a sample that is BOTH behind the center AND itself in focus (a sharp
    // background, small CoC) must not bleed onto the nearer subject's silhouette. (Was gated on
    // center_coc_abs < 1.5, which line 75 already early-returns on → the guard never fired.)
    if (s_depth > center_depth + 0.05 && s_coc < 1.5) { w = 0.0; }
    let safe = select(sc, vec3<f32>(0.0), sc != sc); // NaN guard
    let finite = min(safe, vec3<f32>(65504.0));       // +inf guard (half-float max)
    sum = sum + finite * w;
    weight = weight + w;
  }
  return vec4<f32>(sum / max(weight, 1.0e-4), 1.0);
}
