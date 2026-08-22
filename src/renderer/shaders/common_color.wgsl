// The engine's sRGB transfer approximation. Deliberately the cheap gamma-2.2 curve rather than
// the piecewise IEC curve: it is what the UI and tonemap paths have always used, and switching
// would shift every existing colour.
fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
  return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}
