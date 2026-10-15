// Per-vertex shapes: polygon fans and bezier stroke ribbons (fragment stage in common_shape.wgsl).

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
