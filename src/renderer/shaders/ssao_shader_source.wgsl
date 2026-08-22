struct SsaoParams {
  proj: mat4x4<f32>,   // view → clip
  cfg: vec4<f32>,      // x=radius, y=bias(sin), z=strength, w=power
  contact: vec4<f32>,  // xyz = view-space dir TO the sun, w = contact-shadow strength
  contact2: vec4<f32>, // x=length(view units), y=steps, z=thickness, w=ssgi_strength
  prev_view_proj: mat4x4<f32>, // temporal SSGI reprojection
  cur_view_proj: mat4x4<f32>,
  inv_view: mat4x4<f32>,
  gi: vec4<f32>,       // x=history_valid, y=history feedback weight
}
@group(0) @binding(0) var pos_tex: texture_2d<f32>;
@group(0) @binding(1) var nrm_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> p: SsaoParams;
@group(0) @binding(4) var color_tex: texture_2d<f32>; // lit scene colour, for SSGI bounce
@group(0) @binding(5) var gi_history: texture_2d<f32>; // previous accumulated GI/AO, for temporal blend

fn hash23(p2: vec2<f32>) -> vec3<f32> {
  var p3 = fract(vec3<f32>(p2.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yzz) * p3.zyx);
}

// Project a view-space position to screen UV.
fn to_uv(view_pos: vec3<f32>) -> vec3<f32> {
  let clip = p.proj * vec4<f32>(view_pos, 1.0);
  let ndc = clip.xyz / clip.w;
  return vec3<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5, clip.w);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let g = textureSampleLevel(pos_tex, samp, in.uv, 0.0);
  if (g.w < 0.5) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); } // background — no occlusion (a), no bounce (rgb)
  let P = g.xyz; // view space (camera at origin, looking -z)
  // View-space shading normal straight from the G-buffer (.xyz; .w is roughness). It is already
  // unit-length, camera-facing, and carries normal-mapped detail — far cleaner than the previous
  // cross(dpdx(P), dpdy(P)) face normal, which faceted on curved/low-poly geometry and haloed
  // where the derivatives straddled a depth discontinuity. (Same target SSR reads.)
  let N = normalize(textureSampleLevel(nrm_tex, samp, in.uv, 0.0).xyz);

  let radius = p.cfg.x;
  let bias = p.cfg.y;
  let rnd = hash23(in.pos.xy);
  // Perspective screen-space radius for this depth (proj[0].x is the x focal scale).
  let radius_uv = 0.5 * p.proj[0].x * radius / max(-P.z, 0.1);

  // ── Horizon-Based AO (GTAO-style): per-direction max-horizon accumulation. Far cleaner than
  // random hemisphere sampling at the same tap count.
  let DIRS = 6;
  let STEPS = 4;
  var occ = 0.0;
  for (var d: i32 = 0; d < DIRS; d = d + 1) {
    let ang = (f32(d) + rnd.x) * (6.2831853 / f32(DIRS));
    let dir = vec2<f32>(cos(ang), sin(ang));
    var max_sin = bias; // tangent bias
    for (var s: i32 = 1; s <= STEPS; s = s + 1) {
      let t = (f32(s) - rnd.y) / f32(STEPS);
      let suv = in.uv + dir * t * radius_uv;
      if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { break; }
      let sg = textureSampleLevel(pos_tex, samp, suv, 0.0);
      if (sg.w < 0.5) { continue; }
      let diff = sg.xyz - P;
      let l = length(diff);
      if (l < 0.001 || l > radius) { continue; }
      let sin_h = dot(N, diff / l); // elevation of the sample above the tangent plane
      if (sin_h > max_sin) {
        let atten = 1.0 - (l / radius) * (l / radius);
        occ += (sin_h - max_sin) * atten;
        max_sin = sin_h;
      }
    }
  }
  var ao = 1.0 - clamp(occ / f32(DIRS) * p.cfg.z, 0.0, 1.0);
  if (p.cfg.w != 1.0) { ao = pow(ao, p.cfg.w); }

  // ── SSGI: one bounce of indirect diffuse. A dedicated larger-radius disk gather (independent of the
  // AO radius so it reaches nearby coloured surfaces): for each visible sample that this pixel faces
  // and that faces back, add its lit colour, cosine + facing + distance weighted. Denoised by the
  // tonemap's 5×5 blur; strength in p.contact2.w. Untinted by receiver albedo (no albedo G-buffer).
  var indirect = vec3<f32>(0.0);
  let gi_strength = p.contact2.w;
  if (gi_strength > 0.0) {
    let gi_radius = radius * 4.0;
    let gi_radius_sq = gi_radius * gi_radius;
    let gi_radius_uv = 0.5 * p.proj[0].x * gi_radius / max(-P.z, 0.1);
    var giw = 0.0;
    for (var gi_i: i32 = 0; gi_i < 12; gi_i = gi_i + 1) {
      let ga = (f32(gi_i) + rnd.x) * (6.2831853 / 12.0);
      let gr = sqrt((f32(gi_i) + rnd.y) / 12.0);
      let guv = in.uv + vec2<f32>(cos(ga), sin(ga)) * gr * gi_radius_uv;
      if (guv.x < 0.0 || guv.x > 1.0 || guv.y < 0.0 || guv.y > 1.0) { continue; }
      let gg = textureSampleLevel(pos_tex, samp, guv, 0.0);
      if (gg.w < 0.5) { continue; }
      let gd = gg.xyz - P;
      let gl2 = dot(gd, gd);
      if (gl2 < 0.0004 || gl2 > gi_radius_sq) { continue; }
      let gdir = gd * inverseSqrt(gl2);
      let ndl = max(dot(N, gdir), 0.0); // receiver faces the sample
      if (ndl <= 0.0) { continue; }
      let gsn = normalize(textureSampleLevel(nrm_tex, samp, guv, 0.0).xyz);
      let gfacing = max(-dot(gsn, gdir), 0.0); // sample faces the receiver
      let gatten = 1.0 / (1.0 + gl2 * 4.0);
      indirect += textureSampleLevel(color_tex, samp, guv, 0.0).rgb * (ndl * gfacing * gatten);
      giw += 1.0;
    }
    indirect = indirect / max(giw, 1.0) * gi_strength;
  }

  // ── Contact shadows: short screen-space ray-march toward the sun. Catches the small contact
  // gaps the shadow map misses (trim, badges, wheels meeting the floor). Folded into the AO term.
  var contact = 1.0;
  // Fade contact shadows out as the surface turns away from the sun: past the terminator the
  // direct light is already zero, and since this term is folded into AO (which also darkens the
  // ambient fill), marching there only painted a dirty dark crescent along curved silhouettes.
  let sun_face = smoothstep(0.0, 0.25, dot(N, p.contact.xyz));
  let cstr = p.contact.w * sun_face;
  let sun = p.contact.xyz;
  if (cstr > 0.0 && dot(sun, sun) > 0.01) {
    let csteps = i32(p.contact2.y);
    let stepv = sun * (p.contact2.x / f32(csteps));
    // Bias off the surface to avoid self-occlusion, and jitter the march start per pixel (rnd.z)
    // — the fixed-step march otherwise prints a hard jagged banding edge at silhouettes (a dirty
    // dark crescent on curved surfaces); dithered, it dissolves to noise that the temporal
    // accumulation below smooths out.
    var march = P + N * 0.02 + stepv * rnd.z;
    for (var s: i32 = 0; s < csteps; s = s + 1) {
      march = march + stepv;
      let cuv = to_uv(march);
      if (cuv.z <= 0.0 || cuv.x < 0.0 || cuv.x > 1.0 || cuv.y < 0.0 || cuv.y > 1.0) { break; }
      let cg = textureSampleLevel(pos_tex, samp, cuv.xy, 0.0);
      if (cg.w < 0.5) { continue; }
      let dz = cg.z - march.z; // geometry nearer the camera than the ray → occluder
      if (dz > 0.003 && dz < p.contact2.z) { contact = 1.0 - cstr; break; }
    }
  }

  let outv = ao * contact;
  // rgb = indirect bounce (SSGI), a = AO×contact. The tonemap denoises (5×5) then adds rgb / applies a.
  var result = vec4<f32>(indirect, outv);

  // ── Temporal accumulation ────────────────────────────────────────────────────────────────
  // Blend with the previous accumulated GI/AO, reprojected by camera motion (same reprojection as TAA).
  // GI/AO are low-frequency, so a heavy history weight is stable; it's dropped under fast motion to avoid
  // ghosting. This accumulates the sparse gather into a clean result far better than spatial blur alone.
  if (p.gi.x > 0.5) {
    let world = (p.inv_view * vec4<f32>(P, 1.0)).xyz;
    let pc = p.prev_view_proj * vec4<f32>(world, 1.0);
    let cc = p.cur_view_proj * vec4<f32>(world, 1.0);
    if (pc.w > 0.0 && cc.w > 0.0) {
      let prev_uv = vec2<f32>((pc.x / pc.w) * 0.5 + 0.5, -(pc.y / pc.w) * 0.5 + 0.5);
      let cur_uv = vec2<f32>((cc.x / cc.w) * 0.5 + 0.5, -(cc.y / cc.w) * 0.5 + 0.5);
      let motion = prev_uv - cur_uv;
      let puv = in.uv + motion;
      if (puv.x >= 0.0 && puv.x <= 1.0 && puv.y >= 0.0 && puv.y <= 1.0) {
        let hist = textureSampleLevel(gi_history, samp, puv, 0.0);
        let tex_size = vec2<f32>(textureDimensions(gi_history, 0));
        let motion_px = length(motion * tex_size);
        let blend = mix(p.gi.y, 0.4, clamp(motion_px * 0.5, 0.0, 1.0));
        result = mix(result, hist, blend);
      }
    }
  }
  return result;
}
