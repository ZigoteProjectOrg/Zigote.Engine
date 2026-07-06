@group(0) @binding(0) var image_tex: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

// Per-batch rounded clip — see shape_shader_source.wgsl. Slot 0 (all zeros) = disabled, so the
// blit pass and unclipped batches return exactly 1.0 here.
struct RoundedClip {
  rect: vec4<f32>,
  radius: vec4<f32>,
};

@group(1) @binding(0) var<uniform> rounded_clip: RoundedClip;

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
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(
  @location(0) position: vec2<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) color: vec4<f32>,
) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  out.uv = uv;
  out.color = color;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let c = textureSample(image_tex, image_sampler, in.uv);
  var out = c * in.color;
  out.a = out.a * rounded_clip_coverage(in.position.xy);
  return out;
}
