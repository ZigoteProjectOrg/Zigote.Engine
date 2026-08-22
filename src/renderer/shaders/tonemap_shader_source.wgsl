struct TonemapParams {
  cfg: vec4<f32>, // x=exposure, y=contrast, z=saturation, w=bloom_intensity
  debug: vec4<u32>, // x = debug view code (0 = off)
  look: vec4<f32>, // x=look(0=Default,1=Punchy,2=Golden), y=vignette_strength, z=vignette_softness, w=grain_amount
  wb: vec4<f32>, // x=temperature, y=tint, z=chromatic_aberration, w=frame_seed
  lens: vec4<f32>, // x=lens_distortion_k1, y=lens_distortion_k2, zw unused
}
@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var bloom_tex: texture_2d<f32>;
@group(0) @binding(3) var ao_tex: texture_2d<f32>;
@group(0) @binding(4) var ssr_tex: texture_2d<f32>;
@group(0) @binding(5) var<uniform> p: TonemapParams;
// (bindings 6–8 were the Metal-RT AO/shadow/reflection inputs — removed with the Metal backend;
//  9/10 keep their slot numbers so the bind group needs no renumber.)
@group(0) @binding(9) var lighting_meta_tex: texture_2d<f32>;
@group(0) @binding(10) var scene_meta_tex: texture_2d<f32>;
@group(0) @binding(11) var albedo_tex: texture_2d<f32>; // receiver base colour, tints the SSGI bounce
@group(0) @binding(12) var exposure_tex: texture_2d<f32>; // 1×1 auto-exposure multiplier (.r); used when debug.y==1

// Inverse of the hardware sRGB encode (approx). The swapchain target is sRGB, so the GPU
// applies linear→sRGB on write; pre-applying this to an already-display-ready value makes the
// final pixel reproduce that value faithfully (used for debug-view passthrough).
// AgX view transform — Blender/EEVEE's default display transform. A filmic, hue-preserving
// curve that rolls highlights off gracefully and desaturates them (the "AgX look"). Constants
// from the standard minimal AgX (Sobotka / Wrensch). Input: linear scene radiance (post-exposure).
// Output: linearised display value, so the sRGB swapchain re-encodes it to the intended pixel.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
  let x2 = x * x;
  let x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
fn agx_tonemap(val_in: vec3<f32>) -> vec3<f32> {
  let agx_mat = mat3x3<f32>(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  let agx_mat_inv = mat3x3<f32>(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  let min_ev = -12.47393;
  let max_ev = 4.026069;
  var val = agx_mat * max(val_in, vec3<f32>(0.0));
  val = clamp(log2(max(val, vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
  val = (val - vec3<f32>(min_ev)) / (max_ev - min_ev);
  val = agx_contrast(val);
  val = agx_mat_inv * val;
  return pow(max(val, vec3<f32>(0.0)), vec3<f32>(2.2));
}

// AgX "looks": an ASC-CDL grade (slope/offset/power + saturation) on the display-linear AgX
// output. Default = identity. Punchy = steeper power + more sat. Golden = warm slope + slight lift.
fn apply_look(col_in: vec3<f32>, look: f32) -> vec3<f32> {
  var slope = vec3<f32>(1.0);
  var offset = vec3<f32>(0.0);
  var power = vec3<f32>(1.0);
  var sat = 1.0;
  if (look > 1.5) {
    slope = vec3<f32>(1.0, 0.98, 0.90);
    offset = vec3<f32>(0.02, 0.01, -0.01);
    sat = 1.05;
  } else if (look > 0.5) {
    power = vec3<f32>(1.35);
    sat = 1.18;
  }
  var c = max(col_in, vec3<f32>(0.0));
  c = pow(c * slope + offset, power);
  let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  c = mix(vec3<f32>(l), c, sat);
  return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// White balance in LINEAR space (pre-AgX). temperature>0 warms (boost R, cut B); tint>0 -> magenta.
// Renormalised by the mean gain to keep overall energy roughly constant.
fn white_balance(c: vec3<f32>, temperature: f32, tint: f32) -> vec3<f32> {
  let t = clamp(temperature, -1.0, 1.0);
  let g = clamp(tint, -1.0, 1.0);
  let gain = vec3<f32>(1.0 + 0.15 * t, 1.0 + 0.10 * g, 1.0 - 0.15 * t);
  let mean_g = (gain.r + gain.g + gain.b) / 3.0;
  return (c * gain) / max(mean_g, 1e-4);
}

// hash grain, animated by frame seed. Returns [-0.5, 0.5]. Integer-style hash (Dave Hoskins
// hash13) instead of fract(sin(...)) — the sin-fract hash is precision-divergent across GPUs and
// bands diagonally in flat areas (exactly what the grain is meant to mask).
fn grain_hash(uv: vec2<f32>, seed: f32) -> f32 {
  var p3 = fract(vec3<f32>(uv.x, uv.y, uv.x) * 0.1031 + seed * 0.0017);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z) - 0.5;
}

// Radial lens distortion (Brown–Conrady): r' = r*(1 + k1*r^2 + k2*r^4) about the frame centre.
// k1<0 = barrel (edges pulled in), k1>0 = pincushion. Returns the source UV to sample from.
fn lens_distort(uv: vec2<f32>, k1: f32, k2: f32) -> vec2<f32> {
  if (k1 == 0.0 && k2 == 0.0) { return uv; }
  let c = uv - vec2<f32>(0.5);
  let rr = dot(c, c);
  let f = 1.0 + k1 * rr + k2 * rr * rr;
  return vec2<f32>(0.5) + c * f;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  // Lens distortion remaps the scene-sample UV (physical camera); the overlay buffers below keep the
  // undistorted UV (additive/small — negligible misalignment except at extreme distortion).
  let duv = lens_distort(in.uv, p.lens.x, p.lens.y);
  // Chromatic aberration: per-channel scene fetch split by a radius^2-scaled UV offset. Explicit
  // LOD keeps it uniform-safe despite the per-channel UVs (no implicit-derivative LOD UB).
  let ca = p.wb.z;
  let center = in.uv - vec2<f32>(0.5);
  let r2 = dot(center, center);
  let ca_off = center * (r2 * ca * 4.0);
  let scene_r = textureSampleLevel(scene_tex, samp, duv + ca_off, 0.0).r;
  let scene_g = textureSampleLevel(scene_tex, samp, duv, 0.0).g;
  let scene_b = textureSampleLevel(scene_tex, samp, duv - ca_off, 0.0).b;
  let scene = vec3<f32>(scene_r, scene_g, scene_b);
  var color = scene;
  let bloom = textureSampleLevel(bloom_tex, samp, in.uv, 0.0).rgb;
  // 5x5 box blur of the (noisy) SSAO/SSGI buffer to denoise before applying. rgb = SSGI indirect
  // bounce, a = AO×contact. Skipped on non-PBR pixels (sky/background/emissive/glass) where the
  // result is masked out by pbr_surface below anyway — the 25 taps are dead work there. The AO/SSGI
  // debug views (11/16) still force the full blur so they stay byte-identical to pre-change.
  let texel = 1.0 / vec2<f32>(textureDimensions(ao_tex, 0));
  let lighting_meta = textureSampleLevel(lighting_meta_tex, samp, in.uv, 0.0).w;
  let pbr_surface = select(0.0, 1.0, lighting_meta >= 1.5);
  let dv = p.debug.x;
  var aogi = vec4<f32>(0.0);
  if (pbr_surface > 0.5 || dv == 11u || dv == 16u) {
    for (var sy: i32 = -2; sy <= 2; sy = sy + 1) {
      for (var sx: i32 = -2; sx <= 2; sx = sx + 1) {
        aogi += textureSampleLevel(ao_tex, samp, in.uv + vec2<f32>(f32(sx), f32(sy)) * texel, 0.0);
      }
    }
    aogi = aogi / 25.0;
  }
  let ao = aogi.a;
  let indirect = aogi.rgb;
  let ssr = textureSampleLevel(ssr_tex, samp, in.uv, 0.0);
  let ambient_ratio = clamp(textureSampleLevel(scene_meta_tex, samp, in.uv, 0.0).a, 0.0, 1.0) * pbr_surface;

  // ── Debug views: unchanged, early-return BEFORE any look/wb/vignette/grain. Re-fetch the
  // un-split scene so codes 1..10 / 15 are byte-identical to pre-change.
  if (dv != 0u) {
    if (dv == 11u)      { return vec4<f32>(srgb_decode(vec3<f32>(ao)), 1.0); }      // AO
    else if (dv == 12u) { return vec4<f32>(srgb_decode(ssr.rgb), 1.0); }            // SSR contribution
    else if (dv == 13u) { return vec4<f32>(srgb_decode(vec3<f32>(ssr.a)), 1.0); }   // SSR hit/miss
    else if (dv == 14u) { return vec4<f32>(srgb_decode(bloom), 1.0); }              // bloom
    else if (dv == 16u) { return vec4<f32>(srgb_decode(clamp(indirect, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0); } // SSGI indirect
    else if (dv == 15u) {                                                           // HDR luminance
      let scene0 = textureSampleLevel(scene_tex, samp, in.uv, 0.0).rgb;
      let lum = dot(scene0, vec3<f32>(0.2126, 0.7152, 0.0722));
      return vec4<f32>(vec3<f32>(lum), 1.0);
    }
    let scene0 = textureSampleLevel(scene_tex, samp, in.uv, 0.0).rgb;
    return vec4<f32>(srgb_decode(clamp(scene0, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
  }

  // ── Composite in linear HDR. The forward pass stores a compact ambient-occlusion ratio so the
  // AO term modifies only indirect/environment energy; emissive, glass, and SSR stay intact.
  let ambient_visibility = clamp(ao, 0.0, 1.0);
  color *= 1.0 - ambient_ratio * (1.0 - ambient_visibility);
  // SSGI indirect bounce — the incoming bounce light reflected by the RECEIVER's own albedo (a green
  // floor reflects green, not white). Added on PBR surfaces (emissive/glass/background excluded).
  let recv_albedo = textureSampleLevel(albedo_tex, samp, in.uv, 0.0).rgb;
  color = color + indirect * recv_albedo * pbr_surface;
  color = color + ssr.rgb;
  color = color + bloom * p.cfg.w;

  // Exposure, then white balance (linear, pre-AgX), then the AgX view transform.
  // Auto-exposure (debug.y==1): multiply by the metered/adapted exposure so the frame's average
  // luminance maps to middle grey; p.cfg.x stays a manual EV bias on top. Off → fixed exposure only.
  var auto_mult = 1.0;
  if (p.debug.y == 1u) { auto_mult = textureSampleLevel(exposure_tex, samp, vec2<f32>(0.5), 0.0).r; }
  color = color * p.cfg.x * auto_mult;
  color = white_balance(color, p.wb.x, p.wb.y);
  color = max(color, vec3<f32>(0.0)); // guard before AgX log2
  color = agx_tonemap(color);

  // AgX look (post-AgX slope/offset/power + per-look saturation).
  color = apply_look(color, p.look.x);

  // Retained user grade: saturation then a gentle contrast push.
  let sat_luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  color = clamp(mix(vec3<f32>(sat_luma), color, p.cfg.z), vec3<f32>(0.0), vec3<f32>(1.0));
  color = clamp(mix(color, color * color * (3.0 - 2.0 * color), p.cfg.y), vec3<f32>(0.0), vec3<f32>(1.0));

  // ── Vignette: smooth radial darkening from centre.
  let v_strength = p.look.y;
  let v_soft = max(p.look.z, 1e-3);
  let dist = length(center) * 1.41421356; // 0 centre, ~1 corner
  let vig = 1.0 - v_strength * smoothstep(1.0 - v_soft, 1.0, dist);
  color = color * vig;

  // ── Film grain: animated, luma-modulated (peaks at mid-grey, fades at 0/1).
  let grain = grain_hash(in.uv, p.wb.w) * p.look.w;
  let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
  let grain_mod = grain * (1.0 - abs(luma * 2.0 - 1.0));
  color = clamp(color + vec3<f32>(grain_mod), vec3<f32>(0.0), vec3<f32>(1.0));

  // Always-on triangular-PDF dither (~1 LSB) to break 8-bit banding in smooth AgX gradients
  // (skies, vignettes). The film grain above is luma-gated to ~0 at black/white where banding is
  // worst, so it can't do this; a TPDF dither floor can. NOTE: this is a STATIC spatial pattern
  // (fixed seed, NOT the per-frame p.wb.w) — an animated dither floor "crawls" frame-to-frame on
  // flat low-contrast areas (the near-white floor/horizon shimmer), which the grain avoids only
  // because it's luma-gated off there.
  let d1 = grain_hash(in.uv + vec2<f32>(0.5, 0.0), 0.0);
  let d2 = grain_hash(in.uv + vec2<f32>(0.0, 0.5), 1.0);
  color = clamp(color + vec3<f32>((d1 + d2) * (1.0 / 255.0)), vec3<f32>(0.0), vec3<f32>(1.0));

  // Final NaN/inf guard.
  color = clamp(select(color, vec3<f32>(0.0), color != color), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(color, 1.0);
}
