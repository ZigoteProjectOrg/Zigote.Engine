// Adaptive-glass probe: how bright is the content under a pane?
//
// Each probed pane is one POINT drawn into its slot of a 64×1 r32float target. The fragment samples
// the captured backdrop on a 12×6 grid across the pane and writes the mean perceptual luminance
// (0 black … 1 white). The host reads the target back asynchronously and flips the pane between
// light and dark glass with hysteresis — the way Apple's small glass controls follow what scrolls
// under them. Sampled with the bilinear backdrop sampler, so a 72-tap grid already averages ~4×
// that many texels; text under a bar registers as the mix of ink and paper it is.

@group(0) @binding(0) var backdrop: texture_2d<f32>;
@group(0) @binding(1) var backdrop_sampler: sampler;

const LUMA = vec3<f32>(0.2126, 0.7152, 0.0722);
const COLS = 12;
const ROWS = 6;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) rect: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec2<f32>, @location(1) rect: vec4<f32>) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.rect = rect;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let inv_dims = 1.0 / vec2<f32>(textureDimensions(backdrop));
  var acc = 0.0;
  for (var y = 0; y < ROWS; y++) {
    for (var x = 0; x < COLS; x++) {
      let f = vec2<f32>((f32(x) + 0.5) / f32(COLS), (f32(y) + 0.5) / f32(ROWS));
      let rgb = textureSampleLevel(backdrop, backdrop_sampler, (in.rect.xy + in.rect.zw * f) * inv_dims, 0.0).rgb;
      acc += pow(max(dot(rgb, LUMA), 0.0), 1.0 / 2.2);
    }
  }
  return vec4<f32>(acc / f32(COLS * ROWS), 0.0, 0.0, 1.0);
}
