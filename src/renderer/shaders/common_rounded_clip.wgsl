// Per-batch rounded-clip uniform (dynamic offset into a small slot ring). radius.x <= 0 disables
// the mask entirely — slot 0 is all zeros, so unclipped batches multiply coverage by exactly 1.0
// and stay byte-identical to the pre-rounded-clip renderer. rect = (center.xy, half_size.xy) and
// radius.x are in physical (framebuffer) pixels, matching @builtin(position). radius.y > 0 selects
// squircle corners (see common_corner.wgsl, which roundedClip() prepends).
//
// The bind group differs per pipeline (shapes use 0, images and text use 1), so it is written as
// the token $CLIP_GROUP and substituted at comptime by shader_prelude.roundedClip().
struct RoundedClip {
  rect: vec4<f32>,
  radius: vec4<f32>,
};

@group($CLIP_GROUP) @binding(0) var<uniform> rounded_clip: RoundedClip;

fn rounded_clip_coverage(frag_pos: vec2<f32>) -> f32 {
  if (rounded_clip.radius.x <= 0.0) {
    return 1.0;
  }
  let r = rounded_clip.radius.x;
  let dist = sd_corner_rect(frag_pos - rounded_clip.rect.xy, rounded_clip.rect.zw, select(r, -r, rounded_clip.radius.y > 0.5));
  return clamp(0.5 - dist, 0.0, 1.0);
}
