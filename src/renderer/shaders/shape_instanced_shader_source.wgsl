// Instanced rounded-rect / border / shadow.
//
// The non-instanced shape path emits six fully-expanded 40-byte vertices per shape — 240 bytes to
// describe a rectangle whose four corners are derivable from two of them. Here one 40-byte INSTANCE
// carries the rect and its SDF parameters, and the four corners come from the vertex index, drawn
// through the shared quad index buffer ({0,1,2, 1,3,2}) the text and image paths already use.
//
// Corner order matches the vertex path exactly: 0 = TL, 1 = TR, 2 = BL, 3 = BR.
//
// Polygon fills and bezier strokes keep the per-vertex pipeline: they are arbitrary triangles, not
// rectangles, and no quad instancing can express them.

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
  @builtin(vertex_index) vid: u32,
  // The four corners in clip space, vertex-index order TL, TR, BL, BR. Corners rather than an
  // axis-aligned rect because the transform stack can rotate or skew a shape, and a rotated
  // rectangle is not describable by two opposite corners.
  @location(0) corners01: vec4<f32>,  // TL.xy, TR.xy
  @location(1) corners23: vec4<f32>,  // BL.xy, BR.xy
  @location(2) color: vec4<f32>,
  @location(3) rect_size: vec2<f32>,
  @location(4) radius: f32,
  @location(5) border_width: f32,
  @location(6) blur_radius: f32,
) -> VertexOut {
  let right = (vid & 1u) != 0u;
  let bottom = (vid & 2u) != 0u;

  let top_pt = select(corners01.xy, corners01.zw, right);
  let bot_pt = select(corners23.xy, corners23.zw, right);
  let pt = select(top_pt, bot_pt, bottom);
  let x = pt.x;
  let y = pt.y;

  let half = rect_size / 2.0;
  let lx = select(-half.x, half.x, right);
  let ly = select(-half.y, half.y, bottom);

  var out: VertexOut;
  out.position = vec4<f32>(x, y, 0.0, 1.0);
  out.color = color;
  out.local_pos = vec2<f32>(lx, ly);
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
