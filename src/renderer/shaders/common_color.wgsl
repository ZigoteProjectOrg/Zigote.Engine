// The engine's sRGB transfer approximation. Deliberately the cheap gamma-2.2 curve rather than
// the piecewise IEC curve: it is what the UI and tonemap paths have always used, and switching
// would shift every existing colour.
fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
  return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

// Alpha remap for flat UI fills. The *_srgb target blends in LINEAR space, but the theme's
// translucent tokens are designed as gamma-space blends (the way every design tool shows them):
// white at 10% over a dark panel is meant to be a faint lift, and blended linearly it lands at
// ~#595959; black at 10% over a light panel all but disappears. So remap alpha by the colour's own
// luminance: dark colours get 1-(1-a)^2.2 — exactly the gamma-space result on any backdrop —
// and bright colours are solved for a typical dark surface (sRGB 0.11, about #1c1c1c) rather
// than pure black, which would leave every hairline and fill on a real panel dimmer than its
// token. Mid-tones mix the two. 0 and 1 are fixed points, so opaque fills are untouched.
// `srgb` is the still-encoded colour.
fn perceptual_alpha(srgb: vec3<f32>, a: f32) -> f32 {
  let x = clamp(a, 0.0, 1.0);
  let lum = dot(srgb, vec3<f32>(0.299, 0.587, 0.114));
  let dark = 1.0 - pow(1.0 - x, 2.2);
  // 0.00778 = 0.11^2.2, the assumed backdrop in linear.
  let bright = (pow(x + (1.0 - x) * 0.11, 2.2) - 0.00778) / (1.0 - 0.00778);
  return mix(dark, bright, clamp(lum, 0.0, 1.0));
}
