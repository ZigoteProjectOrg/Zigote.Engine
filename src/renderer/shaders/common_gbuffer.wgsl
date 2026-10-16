// View-normal G-buffer encoding: octahedral, two channels (rg16float) instead of xyz in an
// rgba16float. MSAA resolve averages the encoded values; over the front hemisphere (every surface
// the camera can see) the mapping is continuous, so an averaged edge sample still decodes to an
// in-between normal, as averaging xyz did.
fn gbuf_oct_wrap(v: vec2<f32>) -> vec2<f32> {
  return (1.0 - abs(v.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
}

fn gbuf_encode_normal(n: vec3<f32>) -> vec2<f32> {
  let p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
  return select(gbuf_oct_wrap(p), p, n.z >= 0.0);
}

fn gbuf_decode_normal(e: vec2<f32>) -> vec3<f32> {
  var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
  let t = max(-n.z, 0.0);
  n.x += select(t, -t, n.x >= 0.0);
  n.y += select(t, -t, n.y >= 0.0);
  return normalize(n);
}

// View-position G-buffer: rg16float, x = view-space z (negative in front of the camera), y = lighting
// metadata (0 background, 1 geometry, 2 + direct-sun fraction for lit PBR). x and y are rebuilt from
// the pixel and the projection the frame was rasterized with (`recon_proj`, TAA jitter included),
// which is exact at pixel centres. Handles perspective and orthographic projections alike.
fn gbuf_view_pos(uv: vec2<f32>, z: f32, proj: mat4x4<f32>) -> vec3<f32> {
  let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  let w = proj[2][3] * z + proj[3][3];
  return vec3<f32>(
    (ndc.x * w - proj[2][0] * z - proj[3][0]) / proj[0][0],
    (ndc.y * w - proj[2][1] * z - proj[3][1]) / proj[1][1],
    z,
  );
}
