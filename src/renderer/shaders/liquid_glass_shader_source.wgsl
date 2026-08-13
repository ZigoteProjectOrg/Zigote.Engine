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

// Apple-style "Liquid Glass": a frosted, light-bending lens rather than a clear refractor.
// The optical model is a rounded-rect sphere-cap bevel — flat and clear through the middle, curving
// hard at the rim where it lenses, frosts, fringes and catches a specular highlight. The whole effect
// is faked from a single `backdrop` (the scene captured beneath the glass): we displace and blur its
// UVs, so the shape stays cheap (one bind group, no extra passes) and identical on wgpu and Metal.
//
// Tunables are grouped at the top of fs_main — they are the knobs for matching Apple's look.

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  // ── Tunables ────────────────────────────────────────────────────────────────
  let IOR          = 1.5;    // index of refraction (1 = none, glass ≈ 1.5)
  let BEND         = 6.0;    // refraction reach: higher = content pulled further through the lens
  let FROST        = 0.45;   // frosted-blur radius as a fraction of thickness (0 = clear lens)
  let FRINGE       = 0.06;   // always-on prismatic dispersion at the rim (pinch adds to this)
  let SPEC_POWER   = 26.0;   // specular tightness (higher = smaller, sharper highlight)
  let SPEC_GAIN    = 0.85;   // specular brightness
  let RIM_GAIN     = 0.14;   // soft fresnel rim brightness
  let CONTACT      = 0.10;   // inner-edge darkening that seats the glass on its background

  // ── Rounded-rect SDF + analytic (DPI-accurate) edge coverage ──────────────────
  let half_size = in.rect_size * 0.5;
  let q = abs(in.local_pos) - half_size + vec2<f32>(in.radius);
  let sd = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - in.radius;
  let aa = max(fwidth(sd), 0.6);
  let coverage = 1.0 - smoothstep(-aa, aa, sd);
  if (coverage < 0.002) { discard; }

  // ── Sphere-cap bevel: surface height + normal. `facing` is normal.z (1 flat-centre → 0 grazing-
  //    rim); `edge` is its complement and drives every rim-localised term below. ─────────────────
  let thickness = max(in.thickness, 1.0);
  let n_cos = clamp((thickness + sd) / thickness, 0.0, 1.0);
  let facing = sqrt(max(0.0, 1.0 - n_cos * n_cos));
  let grad = vec2<f32>(dpdx(sd), dpdy(sd));
  let normal = normalize(vec3<f32>(grad * n_cos, facing));
  let edge = 1.0 - facing;
  let height = facing * thickness;

  // ── Refraction: bend the view ray through the bevel and offset the backdrop UV. The z-guard caps
  //    the lens at grazing angles so the rim samples a sane neighbourhood instead of blowing up. ──
  let dims = vec2<f32>(textureDimensions(backdrop));
  let inv_dims = 1.0 / dims;
  let screen_uv = in.position.xy * inv_dims;
  let refr = refract(vec3<f32>(0.0, 0.0, -1.0), normal, 1.0 / IOR);
  let refr_len = (height + thickness * BEND) / max(0.25, abs(refr.z));
  let disp = refr.xy * refr_len * inv_dims;
  let uvR = screen_uv + disp;

  // ── Frosted refraction: a tiny Poisson disk around the refracted point. The radius opens up at
  //    the rim, so the centre reads clear and the curved edge reads as ground glass (Apple's look).
  //    Centre is double-weighted; 5 taps total. ──────────────────────────────────────────────────
  let frost = (thickness * FROST) * (0.5 + edge) * inv_dims;
  let c0 = textureSample(backdrop, backdrop_sampler, uvR).rgb;
  let f1 = textureSample(backdrop, backdrop_sampler, uvR + vec2<f32>( 0.36,  0.85) * frost).rgb;
  let f2 = textureSample(backdrop, backdrop_sampler, uvR + vec2<f32>(-0.80,  0.39) * frost).rgb;
  let f3 = textureSample(backdrop, backdrop_sampler, uvR + vec2<f32>(-0.41, -0.86) * frost).rgb;
  let f4 = textureSample(backdrop, backdrop_sampler, uvR + vec2<f32>( 0.86, -0.34) * frost).rgb;
  let frosted = (c0 * 2.0 + f1 + f2 + f3 + f4) * (1.0 / 6.0);

  // ── Chromatic dispersion: pull R/B apart along the refraction vector, ramped by `edge` so the
  //    prismatic fringe lives only on the rim. `pinch` lets a caller exaggerate it. ───────────────
  let ca = (in.pinch_strength * 0.5 + FRINGE) * edge;
  var cr = frosted.r;
  var cb = frosted.b;
  if (edge > 0.02) {
    cr = textureSampleLevel(backdrop, backdrop_sampler, uvR + disp * ca, 0.0).r;
    cb = textureSampleLevel(backdrop, backdrop_sampler, uvR - disp * ca, 0.0).b;
  }
  var rgb = vec3<f32>(mix(frosted.r, cr, edge), frosted.g, mix(frosted.b, cb, edge));

  // ── Adaptive luminance (Apple's adaptive glass): pull the backdrop toward a legibility anchor.
  //    `adapt` < 0 is dark glass (anchored low so LIGHT content always reads), > 0 is light glass
  //    (anchored high for DARK content); magnitude is strength. Per pixel, so a chip straddling a
  //    bright sky and a black coat gets a strong scrim only where it needs one. The move is a
  //    contrast compression about the anchor in perceptual space — a tone curve, not a flat wash:
  //    backdrops already near the anchor pass through untouched, highlights fold hard. Dimming is
  //    multiplicative (keeps hue, calmed toward grey); lifting screens toward white, which is what
  //    makes light glass go milky over a dark photo instead of grey. ──────────────────────────────
  let adapt = clamp(in.adapt, -1.0, 1.0);
  if (abs(adapt) > 0.001) {
    let s = abs(adapt);
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let L = dot(max(rgb, vec3<f32>(0.0)), lw);
    let Lp = pow(L, 1.0 / 2.2);
    let anchor_p = select(0.22, 0.85, adapt > 0.0);
    let keep = 1.0 - s * 0.8;
    let aim = pow(anchor_p + (Lp - anchor_p) * keep, 2.2);
    if (aim < L) {
      let dimmed = rgb * (aim / max(L, 1e-4));
      rgb = mix(dimmed, vec3<f32>(aim), s * 0.30);
    } else {
      rgb = mix(rgb, vec3<f32>(1.0), (aim - L) / max(1.0 - L, 1e-4));
    }
  }

  // ── Tint. `clear_tint` (0 = clear glass, 1 = strongest) is independent of edge coverage: dark
  //    tints multiply (deepen), light tints screen (brighten), matching the source material. ──────
  let luma_w = vec3<f32>(0.299, 0.587, 0.114);
  // Tint arrives sRGB-encoded like every UI color; the backdrop sample is linear — decode so the
  // multiply/screen happens in one space (see shape shader).
  let gc = pow(clamp(in.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2));
  let tint_amount = clamp(in.clear_tint, 0.0, 1.0);
  if (tint_amount > 0.0) {
    if (dot(gc, luma_w) < 0.5) {
      rgb = mix(rgb, rgb * (gc * 2.0), tint_amount);
    } else {
      let screened = vec3<f32>(1.0) - (vec3<f32>(1.0) - rgb) * (vec3<f32>(1.0) - gc);
      rgb = mix(rgb, screened, tint_amount);
    }
  }

  // ── Lighting: a crisp specular catch on the bevel + a soft fresnel rim, gated by thickness so a
  //    thin sliver of glass barely lights and a thick slab catches a bright edge. ─────────────────
  var light_dir = vec2<f32>(0.707, 0.707);
  if (length(in.glow_pos) > 0.001) { light_dir = normalize(in.glow_pos); }
  let L = normalize(vec3<f32>(light_dir, 0.65));
  let H = normalize(L + vec3<f32>(0.0, 0.0, 1.0));
  let spec = pow(max(dot(normal, H), 0.0), SPEC_POWER);
  let fres = edge * edge;
  let tg = clamp((thickness - 3.0) * 0.25, 0.0, 1.0);
  rgb += vec3<f32>(spec * SPEC_GAIN + fres * RIM_GAIN) * tg;
  rgb *= 1.0 - CONTACT * fres * tg;

  // Straight (non-premultiplied) colour; coverage drives the alpha-over blend so the soft edge melts
  // into the real framebuffer background — no second unrefracted backdrop tap needed.
  return vec4<f32>(rgb, coverage);
}
