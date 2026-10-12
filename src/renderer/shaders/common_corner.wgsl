// Rounded-rect distance shared by shapes, shadows, rounded clips and Liquid Glass, so a pane, its
// shadow and the clip on its content all trace the same outline.

// Where a squircle corner starts to curve, in radii from the corner (1 would be a circular arc).
const CORNER_REACH = 1.528;

// Rounded rect with continuous-curvature ("squircle") corners. The corner is a superellipse that
// starts curving CORNER_REACH radii out — well before a circular arc would — with its exponent
// chosen so the curve still passes through the same diagonal point as a circle of `radius`: the
// shape keeps its apparent roundness and loses the visible join where straight edge meets arc.
// Where the reach will not fit (a capsule) the exponent falls back to 2, a true circle.
fn sd_squircle_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
  let reach = max(min(radius * CORNER_REACH, min(half_size.x, half_size.y)), 1e-3);
  let n = 0.6931472 / max(-log(1.0 - 0.2928932 * min(radius / reach, 1.0)), 0.0866);  // ≤ 8
  let q = abs(p) - half_size + vec2<f32>(reach);
  let c = max(q, vec2<f32>(0.0));
  let len = pow(pow(c.x, n) + pow(c.y, n), 1.0 / n);
  // A p-norm is not a distance: on the corner diagonal it reads up to ~12% short, which would
  // fatten anything measured from the edge (a bevel, a hairline, a blur). Dividing by its gradient
  // length is the first-order fix; it is eased in toward the outline so the interior, where the
  // correction is not valid, stays continuous.
  let grad = pow(c / max(len, 1e-6), vec2<f32>(n - 1.0));
  let scale = mix(1.0, max(length(grad), 0.5), clamp(len / reach, 0.0, 1.0));
  return (len - reach) / scale + min(max(q.x, q.y), 0.0);
}

// Distance to a rounded rect: `radius` >= 0 is the usual circular corner, `radius` < 0 a squircle
// corner of |radius|. The sign is how the squircle flag rides the existing vertex/uniform layout.
fn sd_corner_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
  if (radius < 0.0) {
    return sd_squircle_rect(p, half_size, -radius);
  }
  let q = abs(p) - half_size + vec2<f32>(radius);
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}
