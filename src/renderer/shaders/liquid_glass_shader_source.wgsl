@group(0) @binding(0) var backdrop: texture_2d<f32>;
@group(0) @binding(1) var backdrop_sampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local_pos: vec2<f32>,
  @location(2) rect_size: vec2<f32>,
  @location(3) radius: f32,
  @location(4) thickness: f32,
  @location(5) glow_pos: vec2<f32>,
  @location(6) pinch_strength: f32,
  @location(7) clear_tint: f32,
  @location(8) adapt: f32,
};

@vertex
fn vs_main(
  @location(0) position: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) local_pos: vec2<f32>,
  @location(3) rect_size: vec2<f32>,
  @location(4) radius: f32,
  @location(5) thickness: f32,
  @location(6) glow_pos: vec2<f32>,
  @location(7) pinch_strength: f32,
  @location(8) clear_tint: f32,
  @location(9) adapt: f32,
) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.color = color;
  out.local_pos = local_pos;
  out.rect_size = rect_size;
  out.radius = radius;
  out.thickness = thickness;
  out.glow_pos = glow_pos;
  out.pinch_strength = pinch_strength;
  out.clear_tint = clear_tint;
  out.adapt = adapt;
  return out;
}

// Apple-style "Liquid Glass": a clear, vibrant lens with a light-catching rim, not a grey frosted
// slab. Three things carry the look, in this order of importance:
//   1. a squircle-profile bevel — flat and undistorted through the middle, turning over fast at the
//      rim, where it squeezes the surrounding content inward like the edge of a real lens;
//   2. a hairline specular on the very edge, bright where the rim faces the light AND (dimmer) on
//      the opposite side where light exits the slab, nearly absent in between;
//   3. vibrancy — the backdrop behind the pane is softened and its saturation lifted, so colour
//      survives the blur instead of averaging to mud.
// Everything is faked from a single `backdrop` (the scene captured beneath the glass): one bind
// group, no extra passes.
//
// The maximum sampling reach (lens + frost) is mirrored by backdropCopyRegion in wgpu.zig — change
// BEND / FROST_* / SLOPE_MAX and that bound must follow.

// ── Tunables: the knobs for matching the reference look ─────────────────────────
const IOR         = 1.5;    // index of refraction (1 = none, glass ≈ 1.5)
const PROFILE     = 2.4;    // bevel superellipse exponent: 2 = round cap, higher = flatter top, tighter rim
const SLOPE_MAX   = 4.0;    // steepest bevel slope (≈76°) — bounds the lens at the outermost pixels
const BEND        = 2.8;    // lens reach in slab thicknesses: how far the rim pulls content inward
const FROST_PX    = 3.0;    // frosted-blur radius, in pixels
const FROST_THICK = 0.9;    // added radius per unit of slab thickness — the material's weight. Most
                            // of the frost rides on this rather than on FROST_PX because thickness
                            // arrives scaled by display density and a bare pixel count does not.
const FROST_TAPS  = 20;     // spiral taps around the refracted point (+1 for the centre)
const VIBRANCY    = 1.35;   // backdrop saturation behind the glass (1 = untouched)
const FRINGE      = 0.05;   // always-on prismatic dispersion at the rim (pinch adds to this)
const HAIRLINE    = 0.60;   // edge specular: opacity of the white hairline where it faces the light
const BACKLIGHT   = 0.35;   // the hairline's strength on the side opposite the light
const SHEEN       = 0.16;   // soft highlight spread across the bevel behind the hairline
const AMBIENT     = 0.06;   // rim light that ignores direction, so no edge ever vanishes
const CONTACT     = 0.10;   // inner-edge darkening that seats the glass on its background

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);

// Gaussian blur of the backdrop gathered in one pass around `uv`. The taps are a golden-angle
// spiral with area-uniform radii (r = √u) and gaussian weights in r² — 21 samples reconstruct a
// σ ≈ radius/2 gaussian closely enough that no ghosting of hard edges survives. The spiral is
// rotated per pixel by interleaved gradient noise: with one shared pattern, sparse taps alias into
// streaks that track the backdrop's edges; the rotation decorrelates neighbours and depends on
// nothing that moves, so there is no shimmer. textureSampleLevel because implicit-derivative
// sampling is not allowed inside a loop (and the backdrop has one mip anyway).
fn frosted_backdrop(uv: vec2<f32>, radius: vec2<f32>, frag: vec2<f32>) -> vec3<f32> {
  let spin = 6.2831853 * fract(52.9829189 * fract(dot(frag, vec2<f32>(0.06711056, 0.00583715))));
  var acc = textureSampleLevel(backdrop, backdrop_sampler, uv, 0.0).rgb;
  var w_sum = 1.0;
  for (var i = 0; i < FROST_TAPS; i++) {
    let u = (f32(i) + 0.5) / f32(FROST_TAPS);
    let angle = f32(i) * 2.39996323 + spin;      // golden angle — no spokes at any tap count
    let w = exp(-2.0 * u);                       // gaussian in r²: σ = radius / 2
    acc += textureSampleLevel(backdrop, backdrop_sampler, uv + vec2<f32>(cos(angle), sin(angle)) * sqrt(u) * radius, 0.0).rgb * w;
    w_sum += w;
  }
  return acc / w_sum;
}

// Adaptive luminance: pull the backdrop toward a legibility anchor. `adapt` < 0 is dark glass
// (anchored low so LIGHT content always reads), > 0 is light glass (anchored high for DARK
// content); magnitude is strength. Per pixel, so a chip straddling a bright sky and a black coat
// gets a strong scrim only where it needs one. The move is a contrast compression about the anchor
// in perceptual space — a tone curve, not a flat wash: backdrops already near the anchor pass
// through untouched, highlights fold hard. Dimming is multiplicative (keeps hue, calmed slightly
// toward grey); lifting screens toward white, which is what makes light glass go milky over a dark
// photo instead of grey.
fn adapt_luminance(rgb: vec3<f32>, adapt: f32) -> vec3<f32> {
  let s = abs(adapt);
  if (s < 0.001) { return rgb; }
  let lum = dot(max(rgb, vec3<f32>(0.0)), LUMA);
  let anchor_p = select(0.22, 0.85, adapt > 0.0);
  let aim = pow(anchor_p + (pow(lum, 1.0 / 2.2) - anchor_p) * (1.0 - s * 0.8), 2.2);
  if (aim < lum) {
    return mix(rgb * (aim / max(lum, 1e-4)), vec3<f32>(aim), s * 0.15);
  }
  return mix(rgb, vec3<f32>(1.0), (aim - lum) / max(1.0 - lum, 1e-4));
}

// Tint. `amount` (0 = clear glass, 1 = strongest) is independent of edge coverage: dark tints
// multiply (deepen), light tints screen (brighten). The tint arrives sRGB-encoded like every UI
// colour while the backdrop sample is linear — decode so the blend happens in one space.
fn apply_tint(rgb: vec3<f32>, tint_srgb: vec3<f32>, amount: f32) -> vec3<f32> {
  let gc = pow(clamp(tint_srgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
  if (dot(gc, LUMA) < 0.5) {
    return mix(rgb, rgb * (gc * 2.0), amount);
  }
  return mix(rgb, vec3<f32>(1.0) - (vec3<f32>(1.0) - rgb) * (vec3<f32>(1.0) - gc), amount);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  // ── Shape: squircle-cornered rect SDF (common_corner.wgsl) + analytic edge coverage ───────────
  let sd = sd_squircle_rect(in.local_pos, in.rect_size * 0.5, in.radius);
  let aa = max(fwidth(sd), 0.6);
  let coverage = 1.0 - smoothstep(-aa, aa, sd);
  if (coverage < 0.002) { discard; }

  // ── Bevel: a superellipse cross-section h(x) = (1 - u^n)^(1/n), u = 1 at the edge → 0 one
  //    `thickness` in. `slope` is dh/dx; `edge` (0 flat middle → ~0.76 at the rim) drives every
  //    rim-localised term below. `out_dir` is the screen-space outward direction, taken from
  //    derivatives so it stays right under a paint transform. ─────────────────────────────────
  let thickness = max(in.thickness, 1.0);
  let u = clamp(1.0 + sd / thickness, 0.0, 1.0);
  let slope = min(pow(u, PROFILE - 1.0) / pow(max(1.0 - pow(u, PROFILE), 1e-4), (PROFILE - 1.0) / PROFILE), SLOPE_MAX);
  let grad = vec2<f32>(dpdx(sd), dpdy(sd));
  let out_dir = grad / max(length(grad), 1e-4);
  let normal = normalize(vec3<f32>(out_dir * slope, 1.0));
  let edge = 1.0 - normal.z;

  // ── Refraction: bend the view ray through the bevel and follow it down through the slab. The
  //    refracted ray can never pass the critical angle, so |disp| ≤ 0.86 · BEND · thickness. ────
  let inv_dims = 1.0 / vec2<f32>(textureDimensions(backdrop));
  let refr = refract(vec3<f32>(0.0, 0.0, -1.0), normal, 1.0 / IOR);
  let disp = refr.xy / -refr.z * (thickness * BEND) * inv_dims;
  let uv = in.position.xy * inv_dims + disp;

  var rgb = frosted_backdrop(uv, (FROST_PX + thickness * FROST_THICK) * inv_dims, in.position.xy);

  // ── Chromatic dispersion: pull R/B apart along the refraction vector, ramped by `edge` so the
  //    prismatic fringe lives only on the rim. `pinch` lets a caller exaggerate it. ───────────────
  if (edge > 0.02) {
    let ca = (in.pinch_strength * 0.5 + FRINGE) * edge;
    let cr = textureSampleLevel(backdrop, backdrop_sampler, uv + disp * ca, 0.0).r;
    let cb = textureSampleLevel(backdrop, backdrop_sampler, uv - disp * ca, 0.0).b;
    rgb = vec3<f32>(mix(rgb.r, cr, edge), rgb.g, mix(rgb.b, cb, edge));
  }

  // ── Material: vibrancy, then the legibility tone curve, then the caller's tint. ────────────
  rgb = max(mix(vec3<f32>(dot(rgb, LUMA)), rgb, VIBRANCY), vec3<f32>(0.0));
  rgb = adapt_luminance(rgb, clamp(in.adapt, -1.0, 1.0));
  let tint_amount = clamp(in.clear_tint, 0.0, 1.0);
  if (tint_amount > 0.0) { rgb = apply_tint(rgb, in.color.rgb, tint_amount); }

  // ── Lighting. `toward` is how squarely this stretch of rim faces the light: the lit side takes
  //    the full hairline, the far side a dimmer one (light leaving the slab), the flanks only
  //    AMBIENT. The hairline is a white overlay about a pixel wide hugging the edge, with a much
  //    softer sheen across the bevel behind it. Gated by thickness from one pixel so a hairline
  //    bevel still catches light while a zero-thickness (faded-out) pane goes unlit. ────────────
  var light_dir = vec2<f32>(-0.6, -0.8);
  if (length(in.glow_pos) > 0.001) { light_dir = normalize(in.glow_pos); }
  let toward = dot(out_dir, light_dir);
  let lit = AMBIENT + (1.0 - AMBIENT) * toward * toward * select(BACKLIGHT, 1.0, toward > 0.0);
  let hair_w = clamp(thickness * 0.12, 1.0, 2.5);
  let hairline = 1.0 - smoothstep(hair_w * 0.5, hair_w * 1.5, -sd);
  let tg = clamp((thickness - 1.0) * 0.5, 0.0, 1.0);
  rgb *= 1.0 - CONTACT * edge * tg;
  rgb = mix(rgb, vec3<f32>(1.0), clamp((hairline * HAIRLINE + edge * SHEEN) * lit * tg, 0.0, 1.0));

  // Straight (non-premultiplied) colour; coverage drives the alpha-over blend so the soft edge melts
  // into the real framebuffer background — no second unrefracted backdrop tap needed.
  return vec4<f32>(rgb, coverage);
}
