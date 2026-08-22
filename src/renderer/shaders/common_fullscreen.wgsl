// Fullscreen-triangle vertex stage shared by every post pass (bloom down/up, ssao, ssr, dof,
// taa, exposure, tonemap). One oversized triangle, no vertex/index buffer: vertex_index 0..2
// maps to (0,0) (2,0) (0,2) in UV and the matching clip-space corners, so the visible area is
// covered by a single primitive with no diagonal seam.
struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  var o: VOut;
  o.uv = vec2<f32>(x, y);
  o.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return o;
}
