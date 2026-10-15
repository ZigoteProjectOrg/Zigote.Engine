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
