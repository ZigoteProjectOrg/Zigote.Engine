// Per-batch rounded-clip uniform (dynamic offset into a small slot ring). radius.x <= 0 disables
// the mask entirely — slot 0 is all zeros, so unclipped batches multiply coverage by exactly 1.0
// and stay byte-identical to the pre-rounded-clip renderer. rect = (center.xy, half_size.xy) and
// radius.x are in physical (framebuffer) pixels, matching @builtin(position).
struct RoundedClip {
  rect: vec4<f32>,
  radius: vec4<f32>,
};

@group(0) @binding(0) var<uniform> rounded_clip: RoundedClip;

fn rounded_clip_coverage(frag_pos: vec2<f32>) -> f32 {
  if (rounded_clip.radius.x <= 0.0) {
    return 1.0;
  }
  let q = abs(frag_pos - rounded_clip.rect.xy) - rounded_clip.rect.zw + vec2<f32>(rounded_clip.radius.x);
  let dist = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - rounded_clip.radius.x;
  return clamp(0.5 - dist, 0.0, 1.0);
}

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) color: vec4<f32>,
  @location(1) local_pos: vec2<f32>,
  @location(2) rect_size: vec2<f32>,
  @location(3) radius: f32,
  @location(4) border_width: f32,
  @location(5) blur_radius: f32,
};

@vertex
fn vs_main(
  @location(0) position: vec2<f32>,
  @location(1) color: vec4<f32>,
  @location(2) local_pos: vec2<f32>,
  @location(3) rect_size: vec2<f32>,
  @location(4) radius: f32,
  @location(5) border_width: f32,
  @location(6) blur_radius: f32,
) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.color = color;
  out.local_pos = local_pos;
  out.rect_size = rect_size;
  out.radius = radius;
  out.border_width = border_width;
  out.blur_radius = blur_radius;
  return out;
}

// UI colors arrive sRGB-encoded (theme values). The swapchain is an *_srgb format that encodes
// on write, so emit linear here or every flat fill gets encoded twice and washes out toward
// white (#222226 -> #676769, accent #3584e4 -> pastel). Same 2.2 approximation as the
// tonemapper's srgb_decode; blending then happens in linear space, which pairs correctly with
// the sRGB target.
fn srgb_decode(c: vec3<f32>) -> vec3<f32> {
  return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let half_size = in.rect_size / 2.0;
  var q = abs(in.local_pos) - half_size + vec2<f32>(in.radius);
  if (in.blur_radius > 0.0) {
    q = abs(in.local_pos) - (half_size - vec2<f32>(2.0 * in.blur_radius)) + vec2<f32>(in.radius);
  }
  let dist = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - in.radius;

  // Derivative-based AA half-window. At a 1:1 logical→physical mapping fwidth(dist)≈1, so this is
  // 0.5 and matches the previous fixed smoothstep(0.5,-0.5,dist); under a DPI/content scale != 1
  // it tracks the real screen gradient instead of softening/aliasing the edge.
  let aa = max(fwidth(dist) * 0.5, 0.0001);

  var alpha: f32 = 0.0;
  if (in.blur_radius > 0.0) {
    let blur_start = -in.blur_radius;
    let blur_end = in.blur_radius;
    alpha = 1.0 - smoothstep(blur_start, blur_end, dist);
  } else if (in.border_width > 0.0) {
    let inside_outer = smoothstep(aa, -aa, dist);
    let inside_inner = smoothstep(aa, -aa, dist + in.border_width);
    alpha = inside_outer - inside_inner;
  } else {
    alpha = smoothstep(aa, -aa, dist);
  }

  alpha = alpha * rounded_clip_coverage(in.position.xy);

  if (alpha <= 0.0) {
    discard;
  }

  return vec4<f32>(srgb_decode(in.color.rgb), in.color.a * alpha);
}
