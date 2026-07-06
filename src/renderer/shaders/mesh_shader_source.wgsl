struct CameraUniforms {
  view_proj: mat4x4<f32>,
  view: mat4x4<f32>,
  camera_pos: vec3<f32>,
  _pad: f32,
}

struct ModelUniforms {
  model: mat4x4<f32>,
  normal_mat: mat4x4<f32>, // transpose(inverse(model)); only upper-left 3x3 used
  base_color: vec4<f32>,
  metallic_roughness: vec4<f32>,
  emissive: vec4<f32>,
  effect: vec4<u32>,
  // x = IOR (1.5 default), y = transmission factor (0..1), z = occlusion strength
  // (>0 → MR map's R channel is glTF ORM occlusion), w = reserved.
  surface: vec4<f32>,
}

struct LightData {
  position_or_dir: vec4<f32>,
  color_range: vec4<f32>,
}

struct LightExt {
  spot_dir: vec4<f32>,  // xyz = spot forward dir, w = cos(outer)
  spot_cone: vec4<f32>, // x = cos(inner), y = shadow layer (<0 = none)
}

struct LightUniforms {
  view_proj: mat4x4<f32>,
  ambient_color: vec4<f32>,
  light_count: vec4<u32>,
  sky_horizon: vec4<f32>,
  sky_zenith: vec4<f32>,
  sky_ground: vec4<f32>,
  env_avg: vec4<f32>,
  sun_dir: vec4<f32>,
  studio: vec4<f32>,
  post: vec4<f32>,
  shadow: vec4<f32>,
  lights: array<LightData, 16>,
  debug: vec4<u32>, // x = debug view code, y = diagnostic mode flag
  probe_center: vec4<f32>, // xyz = box centre, w = enabled (0/1)
  probe_extents: vec4<f32>, // xyz = box half-extents
  csm_view_proj: array<mat4x4<f32>, 4>, // directional cascade light view-proj matrices
  csm_info: vec4<f32>, // x = cascade count, y = shadow-map resolution
  light_ext: array<LightExt, 16>, // per-light spot cone + shadow index
  spot_view_proj: array<mat4x4<f32>, 4>, // per shadow-casting spot light
  fog_color: vec4<f32>,  // rgb fog/in-scatter colour, .w = density (0 = off)
  fog_params: vec4<f32>, // x = height falloff, y = height base, z = sun in-scatter, w = anisotropy g
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;
@group(1) @binding(0) var<uniform> model_data: ModelUniforms;
@group(2) @binding(0) var base_color_tex: texture_2d<f32>;
@group(2) @binding(1) var base_color_sampler: sampler;
@group(2) @binding(2) var normal_tex: texture_2d<f32>;
@group(2) @binding(3) var normal_sampler: sampler;
@group(2) @binding(4) var mr_tex: texture_2d<f32>;
@group(2) @binding(5) var mr_sampler: sampler;
@group(2) @binding(6) var emissive_tex: texture_2d<f32>;
@group(2) @binding(7) var emissive_sampler: sampler;

@group(3) @binding(0) var shadow_map: texture_depth_2d_array;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var env_cube: texture_cube<f32>;
@group(3) @binding(3) var env_samp: sampler;
@group(3) @binding(4) var point_shadow_map: texture_depth_cube_array;
// Refraction source (copy of the opaque scene taken before the glass pass) for screen-space
// refraction. Sampled with env_samp (binding 3). Group 3 is the global per-pass group; WebGPU caps
// pipelines at 4 bind groups, so this rides in group 3 rather than a 5th group.
@group(3) @binding(5) var refraction_tex: texture_2d<f32>;
// View-space position G-buffer (xyz, w=1 on geometry / 0 on sky). Glass reads the depth of the surface
// behind it (textureLoad, no filtering) to make refraction thickness-aware.
@group(3) @binding(6) var gbuf_pos_tex: texture_2d<f32>;

struct VertexOut {
  @builtin(position) clip_pos: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) light_space_pos: vec4<f32>,
  @location(4) tangent: vec4<f32>,
}

// Karis 2013 analytical split-sum DFG term — the (scale, bias) pair for the specular IBL
// integral, avoiding a precomputed LUT texture. Returned separately so the caller can apply
// multi-scatter energy compensation.
fn env_dfg(roughness: f32, ndotv: f32) -> vec2<f32> {
  let c0 = vec4<f32>(-1.0, -0.0275, -0.572,  0.022);
  let c1 = vec4<f32>( 1.0,  0.0425,  1.04,  -0.04 );
  let r  = roughness * c0 + c1;
  let a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
  return vec2<f32>(-1.04, 1.04) * a004 + r.zw;
}
// Single-scatter env BRDF (kept for compatibility): F0 * scale + bias.
fn brdf_approx(f0: vec3<f32>, roughness: f32, ndotv: f32) -> vec3<f32> {
  let ab = env_dfg(roughness, ndotv);
  return clamp(f0 * ab.x + ab.y, vec3<f32>(0.0), vec3<f32>(1.0));
}
// Schlick Fresnel exponent-5 term, ALU-cheaper than pow(x, 5.0).
fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }

// Rotated Poisson disk shared by the cascade PCSS, cascade cross-fade, spot PCF and point PCF.
const POISSON16 = array<vec2<f32>, 16>(
  vec2<f32>(-0.942016, -0.399062), vec2<f32>( 0.945586, -0.768907),
  vec2<f32>(-0.094184, -0.929389), vec2<f32>( 0.344959,  0.293878),
  vec2<f32>(-0.915886,  0.457714), vec2<f32>(-0.815442, -0.879125),
  vec2<f32>(-0.382775,  0.276768), vec2<f32>( 0.974844,  0.756484),
  vec2<f32>( 0.443233, -0.975116), vec2<f32>( 0.537430, -0.473734),
  vec2<f32>(-0.264969, -0.418930), vec2<f32>( 0.791975,  0.190902),
  vec2<f32>(-0.241888,  0.997065), vec2<f32>(-0.814100,  0.914376),
  vec2<f32>( 0.199841,  0.786414), vec2<f32>( 0.143832, -0.141008));

// PCSS blocker-search + variable-radius PCF for one directional-cascade layer → lit fraction [0,1].
// Extracted so the same code samples both the chosen cascade and the next one for cross-fade at the
// cascade boundary. Uses textureSampleCompareLevel / textureLoad (both allowed in non-uniform control
// flow — no implicit derivatives), so it is safe to call from the per-pixel shadow branch.
fn cascadeShadowFactor(cascade: i32, uv: vec2<f32>, z: f32, bias: f32, cs: f32, sn: f32, shadow_size: f32, softness: f32) -> f32 {
  let rot = mat2x2<f32>(cs, sn, -sn, cs);
  let texel = 1.0 / shadow_size;
  // 1) Blocker search — average occluder depth in a small kernel (raw depth reads).
  let search_r = softness * 3.0 * texel;
  let smax = i32(shadow_size) - 1;
  var blk_sum = 0.0;
  var blk_cnt = 0.0;
  for (var bi: i32 = 0; bi < 16; bi = bi + 1) {
    let bsuv = uv + (rot * POISSON16[bi]) * search_r;
    let bc = clamp(vec2<i32>(bsuv * shadow_size), vec2<i32>(0), vec2<i32>(smax));
    let bd = textureLoad(shadow_map, bc, cascade, 0);
    if (bd < z - bias) { blk_sum += bd; blk_cnt += 1.0; }
  }
  if (blk_cnt < 0.5) { return 1.0; } // no occluder → fully lit
  // 2) Penumbra ∝ receiver-to-blocker depth gap → sharp at contact, soft far from the occluder.
  let avg_blk = blk_sum / blk_cnt;
  let penumbra = (z - avg_blk) * softness * 0.6;
  let radius = clamp(penumbra, 1.5 * texel, 9.0 * texel);
  // 3) Variable-radius 16-tap PCF.
  var shadow_sum = 0.0;
  for (var pi: i32 = 0; pi < 16; pi = pi + 1) {
    let soff = (rot * POISSON16[pi]) * radius;
    shadow_sum += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + soff, cascade, z - bias);
  }
  return shadow_sum / 16.0;
}

@vertex
fn vs_main(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) tangent: vec4<f32>,
) -> VertexOut {
  let world_pos = (model_data.model * vec4<f32>(position, 1.0)).xyz;
  var out: VertexOut;
  out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
  out.world_pos = world_pos;
  // Normals transform by the normal matrix (inverse-transpose) so non-uniform scale doesn't
  // skew them. Tangents are surface directions, so they transform by the model matrix.
  let nmat = mat3x3<f32>(model_data.normal_mat[0].xyz, model_data.normal_mat[1].xyz, model_data.normal_mat[2].xyz);
  out.normal = normalize(nmat * normal);
  out.uv = uv;
  out.light_space_pos = light.view_proj * vec4<f32>(world_pos, 1.0);
  // Meshes without tangents (procedural cube/sphere) arrive with tangent.xyz = 0.
  // normalize(0) is NaN, which poisons the fragment and renders the mesh black, so
  // emit a zero tangent in that case (the fragment treats zero-length as "no tangent").
  let t_world = (model_data.model * vec4<f32>(tangent.xyz, 0.0)).xyz;
  let t = select(vec3<f32>(0.0, 0.0, 0.0), normalize(t_world), dot(t_world, t_world) > 0.0000001);
  out.tangent = vec4<f32>(t, tangent.w);
  return out;
}

// Instanced variant. A second vertex buffer (step_mode=instance) supplies a per-instance
// model matrix as four vec4 columns (locations 4..7); the renderer issues one
// drawIndexed(idx, N) for the whole batch instead of N single draws. Material factors still
// come from the shared ModelUniforms (group 1); only the transform is per-instance. Normals
// use the model's upper-left 3x3 (instances use uniform scale, so no inverse-transpose needed).
@vertex
fn vs_instanced(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
  @location(3) tangent: vec4<f32>,
  @location(4) m0: vec4<f32>,
  @location(5) m1: vec4<f32>,
  @location(6) m2: vec4<f32>,
  @location(7) m3: vec4<f32>,
) -> VertexOut {
  let model = mat4x4<f32>(m0, m1, m2, m3);
  let world_pos = (model * vec4<f32>(position, 1.0)).xyz;
  var out: VertexOut;
  out.clip_pos = camera.view_proj * vec4<f32>(world_pos, 1.0);
  out.world_pos = world_pos;
  let nmat = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
  out.normal = normalize(nmat * normal);
  out.uv = uv;
  out.light_space_pos = light.view_proj * vec4<f32>(world_pos, 1.0);
  let t_world = (model * vec4<f32>(tangent.xyz, 0.0)).xyz;
  let t = select(vec3<f32>(0.0, 0.0, 0.0), normalize(t_world), dot(t_world, t_world) > 0.0000001);
  out.tangent = vec4<f32>(t, tangent.w);
  return out;
}

// MRT: location 0 = linear HDR colour, location 1 = view-space position (xyz) + 1.0 in w.
// The position target feeds screen-space ambient occlusion; w=1 marks "geometry here" so
// the SSAO pass can tell lit surfaces from the cleared background (sky).
struct FragOut {
  @location(0) color: vec4<f32>,
  @location(1) view_pos: vec4<f32>,
  @location(2) view_normal: vec4<f32>, // xyz = view-space normal, w = roughness (for SSR)
  @location(3) albedo: vec4<f32>,      // base colour, for receiver-albedo tinting of SSGI
}

// Height-based exponential fog with analytic sun in-scatter (aerial perspective + god-ray glow).
// Optical depth is integrated analytically along the view ray for an exponential height profile
// (Wronski). The sun in-scatter uses the Henyey-Greenstein phase toward the studio key direction so
// fog brightens (and blooms) when looking toward the sun. density (fog_color.w) 0 → no-op.
fn applyFog(col: vec3<f32>, world_pos: vec3<f32>, cam_pos: vec3<f32>) -> vec3<f32> {
  let density = light.fog_color.w;
  if (density <= 0.0) { return col; }
  let to_frag = world_pos - cam_pos;
  let dist = length(to_frag);
  let dir = to_frag / max(dist, 1e-4);
  let falloff = max(light.fog_params.x, 0.0);
  let base = light.fog_params.y;
  let c0 = density * exp(-falloff * (cam_pos.y - base));
  var optical: f32;
  if (abs(falloff * dir.y) > 1e-4) {
    optical = c0 * (1.0 - exp(-falloff * dir.y * dist)) / (falloff * dir.y);
  } else {
    optical = c0 * dist;
  }
  let fog_amt = clamp(1.0 - exp(-max(optical, 0.0)), 0.0, 1.0);
  // Sun in-scatter: HG phase toward the studio key (light.sun_dir points TOWARD the sun).
  let sun_dir = normalize(light.sun_dir.xyz + vec3<f32>(1e-5));
  let cosT = dot(dir, sun_dir);
  let g = clamp(light.fog_params.w, -0.95, 0.95);
  let g2 = g * g;
  let hg = (1.0 - g2) / (4.0 * 3.14159265 * pow(max(1.0 + g2 - 2.0 * g * cosT, 1e-4), 1.5));
  let inscatter = light.fog_color.rgb * (1.0 + hg * light.fog_params.z);
  return mix(col, inscatter, fog_amt);
}

@fragment
fn fs_main(in: VertexOut) -> FragOut {
  let vpos = vec4<f32>((camera.view * vec4<f32>(in.world_pos, 1.0)).xyz, 1.0);
  // Geometric view-space normal fallback for the unlit/CRT early-outs (roughness 1 = no SSR).
  let vn_geo = vec4<f32>(normalize((camera.view * vec4<f32>(normalize(in.normal), 0.0)).xyz), 1.0);
  // Wireframe debug mode (light.debug.z): this primitive is drawn with line-list topology by the
  // dedicated wireframe pipeline; emit a flat unlit colour so the edges read clearly over the
  // scene regardless of material. Skips all lighting/post so it stays a clean overlay.
  if (light.debug.z == 1u) {
    return FragOut(vec4<f32>(0.55, 0.95, 0.60, 1.0), vpos, vn_geo, vec4<f32>(0.0));
  }
  if (model_data.effect.x == 1u) {
    let centered = in.uv * 2.0 - vec2<f32>(1.0);
    let r2 = dot(centered, centered);
    let warped = centered * (1.0 + r2 * 0.10);
    let uv = warped * 0.5 + vec2<f32>(0.5);
    let rgb_shift = vec2<f32>(0.0025, 0.0);
    let sample_r = textureSample(base_color_tex, base_color_sampler, uv + rgb_shift);
    let sample_g = textureSample(base_color_tex, base_color_sampler, uv);
    let sample_b = textureSample(base_color_tex, base_color_sampler, uv - rgb_shift);
    let scanline = 0.82 + 0.18 * sin(in.uv.y * 900.0);
    let vignette = clamp(1.0 - r2 * 0.45, 0.0, 1.0);
    let color = vec3<f32>(sample_r.r, sample_g.g, sample_b.b) * scanline * vignette;
    return FragOut(vec4<f32>(color, sample_g.a), vpos, vn_geo, vec4<f32>(0.0));
  }
  var base_color = model_data.base_color * textureSample(base_color_tex, base_color_sampler, in.uv);
  // Alpha-mask (effect.z == 1): hard cutout. effect.w carries the cutoff as raw f32 bits.
  if (model_data.effect.z == 1u && base_color.a < bitcast<f32>(model_data.effect.w)) {
    discard;
  }
  if (model_data.effect.x == 2u) {
    return FragOut(base_color, vpos, vn_geo, base_color);
  }
  
  var n = normalize(in.normal);
  let v = normalize(camera.camera_pos - in.world_pos);

  // Sample the normal map in UNIFORM control flow. Implicit-LOD textureSample inside a
  // non-uniform `if` is undefined behaviour in WGSL — it returns correct texels at LOD 0
  // but garbage once the mip LOD matters (i.e. at distance), which rendered meshes black
  // when zoomed out. Sample unconditionally, then only apply it when a tangent exists.
  let map_normal = textureSample(normal_tex, normal_sampler, in.uv).xyz * 2.0 - vec3<f32>(1.0);
  // Emissive map (default 1×1 white → factor passes through untextured materials unchanged).
  // Sampled here, in uniform control flow, for the same implicit-LOD reason as the normal map.
  let emissive_rgb = model_data.emissive.rgb * textureSample(emissive_tex, emissive_sampler, in.uv).rgb;
  if (length(in.tangent.xyz) > 0.1) {
    let t = normalize(in.tangent.xyz);
    let b = cross(n, t) * in.tangent.w;
    let tbn = mat3x3<f32>(t, b, n);
    n = normalize(tbn * map_normal);
  }
  // Per-pixel rotation angle shared by every shadow PCF dither (cascade, spot, point). It depends
  // only on in.clip_pos.xy, so compute the sin/fract hash and its cos/sin ONCE here and reuse across
  // all three branches instead of re-deriving the identical value up to three times per fragment.
  let dither_ang = fract(sin(dot(in.clip_pos.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.2831853;
  let dither_cs = cos(dither_ang);
  let dither_sn = sin(dither_ang);
  // ── Cascaded shadow maps ─────────────────────────────────────────────────────────────────
  // Walk the cascades (tightest first) and pick the first whose light-space projection contains
  // this surface. The cascades are concentric, increasing boxes centred ahead of the camera, so
  // cascade 0 gives the crispest near shadow and higher indices extend the range.
  var shadow_factor = 1.0;
  let cascade_count = u32(max(light.csm_info.x, 1.0));
  var chosen: i32 = -1;
  var sel_uv = vec2<f32>(0.0);
  var sel_z = 0.0;
  var sel_ndc = vec2<f32>(0.0);
  for (var c: u32 = 0u; c < cascade_count; c = c + 1u) {
    let lp = light.csm_view_proj[c] * vec4<f32>(in.world_pos, 1.0);
    let pc = lp.xyz / lp.w;
    let uv = pc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
    if (chosen < 0 && pc.z >= 0.0 && pc.z <= 1.0 &&
        uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
      chosen = i32(c);
      sel_uv = uv;
      sel_z = pc.z;
      sel_ndc = pc.xy;
    }
  }
  if (chosen >= 0) {
    // Slope-scaled depth bias: grazing surfaces (low sun NdotL) need more bias to avoid acne,
    // while face-on surfaces keep the base bias to avoid peter-panning. lights[0] is the
    // shadow-casting sun (see the i==0 shadow_factor multiply in the light loop). Use the GEOMETRIC
    // vertex normal, NOT the normal-mapped `n` — slope bias must track the real surface orientation,
    // or the bump detail makes the bias jitter per-pixel → shadow-acne speckle on normal-mapped panels.
    let sun_l0 = normalize(light.lights[0].position_or_dir.xyz + vec3<f32>(0.00001));
    let bias = light.shadow.y * (1.0 + 2.0 * (1.0 - max(dot(normalize(in.normal), sun_l0), 0.0)));
    // light.shadow.z scales the PCF radius (in texels, normalised by the cascade map resolution);
    // the per-pixel rotation dithers the pattern (it converges to smooth under TAA).
    let shadow_size = max(light.csm_info.y, 1.0);
    let softness = max(light.shadow.z, 0.5);
    shadow_factor = cascadeShadowFactor(chosen, sel_uv, sel_z, bias, dither_cs, dither_sn, shadow_size, softness);
    // Cascade cross-fade: near the outer edge of the chosen cascade's footprint, blend toward the
    // next (looser) cascade so the resolution/penumbra change doesn't read as a hard ring on the
    // ground. `edge` is the surface's max |NDC.xy| in the chosen cascade (1 = at the frustum wall).
    let edge = max(abs(sel_ndc.x), abs(sel_ndc.y));
    let fade = smoothstep(0.85, 1.0, edge);
    if (fade > 0.0 && chosen + 1 < i32(cascade_count)) {
      let nc = chosen + 1;
      let lp2 = light.csm_view_proj[nc] * vec4<f32>(in.world_pos, 1.0);
      let pc2 = lp2.xyz / lp2.w;
      let uv2 = pc2.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
      if (pc2.z >= 0.0 && pc2.z <= 1.0 && uv2.x >= 0.0 && uv2.x <= 1.0 && uv2.y >= 0.0 && uv2.y <= 1.0) {
        let f2 = cascadeShadowFactor(nc, uv2, pc2.z, bias, dither_cs, dither_sn, shadow_size, softness);
        shadow_factor = mix(shadow_factor, f2, fade);
      }
    }
  }

  // glTF metallic-roughness map: roughness in G, metallic in B. Factors multiply the map
  // (default map is white, so untextured materials use their factor unchanged).
  let mr_sample = textureSample(mr_tex, mr_sampler, in.uv);
  let metallic = clamp(model_data.metallic_roughness.x * mr_sample.b, 0.0, 1.0);
  var roughness = max(model_data.metallic_roughness.y * mr_sample.g, 0.05); // Prevent 0 roughness
  // Specular anti-aliasing / prefiltering: as a surface minifies, its normal varies fast on
  // screen, so a pixel covers many reflection directions at once. Widen roughness with that
  // screen-space normal variance so distant metal CONVERGES to the prefiltered average
  // (env_avg) below instead of flickering between the dim sharp sky and the bright studio
  // bands. env_avg must be a bright silver (see below) or this convergence reads as a dark
  // grey silhouette — that was the "black when far" bug.
  // Derive the variance from the FINAL shading normal `n` (post normal-map), not the raw vertex
  // normal — otherwise normal-mapped sub-pixel detail (the dominant aliasing source) never widens
  // roughness. The multiplier is tuned down accordingly so smooth normal-mapped paint stays glossy.
  let dndx = dpdx(n);
  let dndy = dpdy(n);
  let geo_var = dot(dndx, dndx) + dot(dndy, dndy);
  // Specular AA: widen roughness only for genuine sub-pixel normal aliasing (distant
  // minified surfaces). The previous multiplier/cap (1.0 / 0.4) were so aggressive that
  // the mere CURVATURE of a car body inflated glossy paint to ~0.66 roughness — the
  // "clay" look. Use a small amount so curved glossy surfaces stay reflective.
  roughness = clamp(sqrt(roughness * roughness + min(geo_var * 0.10, 0.05)), roughness, 1.0);
  // (Removed the global `if (metallic > 0.5) roughness = min(roughness, 0.4)` clamp — it was
  // non-physical and prevented genuinely rough metals from looking rough. Roughness now comes
  // purely from the material data.)

  let NdotV = max(dot(n, v), 0.0001); // Prevent exactly 0
  // KHR_materials_ior + _specular: the dielectric base reflectance comes from the material's real
  // IOR — F0 = ((n-1)/(n+1))², so 1.5 (crown glass / the default) gives the classic 0.04 —
  // scaled by the specular factor (emissive.w). surface.x < 1 means "unset" → 0.04 base.
  let mat_ior = model_data.surface.x;
  let f0_ior = select(0.04, pow((mat_ior - 1.0) / (mat_ior + 1.0), 2.0), mat_ior >= 1.0);
  var F0 = vec3<f32>(f0_ior * model_data.emissive.w);
  F0 = mix(F0, base_color.rgb, metallic);
  // Multi-scatter energy compensation for DIRECT specular (Kulla-Conty). The single-scatter Cook-
  // Torrance BRDF loses energy at higher roughness (light that would bounce multiple times across
  // microfacets is dropped), so rough metals read dull/dark. `Ess` is the single-scatter directional
  // albedo (the white-furnace integral, F0=1); 1/Ess scales the lobe back to energy-conserving. This
  // is the direct-light analogue of the Fdez-Agüera term applied to the ambient IBL below (same
  // (roughness, NdotV) → the one env_dfg evaluation serves both).
  let dfg = env_dfg(roughness, NdotV);
  let Ess_direct = max(dfg.x + dfg.y, 0.001);
  let energy_comp = vec3<f32>(1.0) + F0 * (1.0 / Ess_direct - 1.0);
  // Per-material clearcoat (KHR_materials_clearcoat): metallic_roughness.z = factor (0 = no coat,
  // the default), .w = clearcoat roughness. The global editor Clearcoat slider (studio.w) scales
  // it. This replaces the old blanket "clearcoat on every smooth surface" — a material gets the
  // glossy lacquer lobe only when its glTF asks for it, so chrome stays metal and paint stays
  // lacquered regardless of their metallic value.
  let cc_strength = clamp(model_data.metallic_roughness.z, 0.0, 1.0) * light.studio.w;
  let cc_rough = clamp(model_data.metallic_roughness.w, 0.02, 0.6);
  let cc_alpha2 = (cc_rough * cc_rough) * (cc_rough * cc_rough);

  var Lo = vec3<f32>(0.0);
  var Lo_cc = vec3<f32>(0.0);
  var sun_direct = vec3<f32>(0.0);
  for (var i: u32 = 0u; i < light.light_count.x; i++) {
    let l_data = light.lights[i];
    var l: vec3<f32>;
    var attenuation: f32 = 1.0;
    var light_color = l_data.color_range.rgb;

    if (l_data.position_or_dir.w == 0.0) {
      // Directional
      l = normalize(l_data.position_or_dir.xyz + vec3<f32>(0.00001));
      if (i == 0u) { light_color = light_color * shadow_factor; }
    } else {
      // Point or spot: inverse-square-ish range attenuation.
      let dir = l_data.position_or_dir.xyz - in.world_pos;
      let dist = length(dir);
      let range = max(l_data.color_range.w, 0.001);
      attenuation = clamp(1.0 - (dist * dist) / (range * range), 0.0, 1.0);
      attenuation = attenuation * attenuation;
      l = normalize(dir + vec3<f32>(0.00001));
      if (l_data.position_or_dir.w >= 1.5) {
        // Spot: smooth cone falloff between the inner and outer half-angles…
        let ext = light.light_ext[i];
        let cos_ang = dot(-l, ext.spot_dir.xyz);
        attenuation = attenuation * smoothstep(ext.spot_dir.w, ext.spot_cone.x, cos_ang);
        // …and an optional perspective shadow sampled from this spot's array layer.
        let slayer = i32(ext.spot_cone.y);
        if (slayer >= 0 && attenuation > 0.0) {
          let spot_idx = slayer - i32(light.csm_info.x);
          let sp = light.spot_view_proj[spot_idx] * vec4<f32>(in.world_pos, 1.0);
          let spc = sp.xyz / sp.w;
          let suv = spc.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5);
          if (spc.z >= 0.0 && spc.z <= 1.0 && suv.x >= 0.0 && suv.x <= 1.0 && suv.y >= 0.0 && suv.y <= 1.0) {
            let sbias = light.shadow.y * 2.0;
            // Rotated 8-tap Poisson PCF (was a single hardware compare → hard aliased edge).
            let ssize = max(light.csm_info.y, 1.0);
            let sradius = max(light.shadow.z, 0.5) * 1.5 / ssize;
            let srot = mat2x2<f32>(dither_cs, dither_sn, -dither_sn, dither_cs);
            var ssum = 0.0;
            for (var sk: i32 = 0; sk < 8; sk = sk + 1) {
              let so = (srot * POISSON16[sk * 2]) * sradius;
              ssum += textureSampleCompareLevel(shadow_map, shadow_sampler, suv + so, slayer, spc.z - sbias);
            }
            attenuation = attenuation * (ssum / 8.0);
          }
        }
      } else {
        // Point: omnidirectional cube shadow, sampled by direction against linear distance-to-light.
        // 8-tap PCF in the tangent plane of the sample direction softens the cube edge.
        let pidx = i32(light.light_ext[i].spot_cone.z);
        if (pidx >= 0 && attenuation > 0.0) {
          let to_frag = in.world_pos - l_data.position_or_dir.xyz;
          let pref = length(to_frag) / range;
          let pbias = light.shadow.y * 3.0;
          let nd = normalize(to_frag);
          let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(nd.y) > 0.99);
          let pt = normalize(cross(up_ref, nd));
          let pb = cross(nd, pt);
          let pradius = 0.018 * max(light.shadow.z, 0.5);
          var pdisk = array<vec2<f32>, 8>(
            vec2<f32>( 1.0,  0.0), vec2<f32>(-1.0,  0.0), vec2<f32>( 0.0,  1.0), vec2<f32>( 0.0, -1.0),
            vec2<f32>( 0.707, 0.707), vec2<f32>(-0.707, 0.707), vec2<f32>(0.707, -0.707), vec2<f32>(-0.707, -0.707));
          // Per-pixel rotation of the tangent-plane taps → dithered soft edge (converges under TAA)
          // instead of the 8 fixed axis/diagonal taps producing a faint star-shaped penumbra.
          let pcs = dither_cs;
          let psn = dither_sn;
          var psum = 0.0;
          for (var pk: i32 = 0; pk < 8; pk = pk + 1) {
            let rd = vec2<f32>(pdisk[pk].x * pcs - pdisk[pk].y * psn, pdisk[pk].x * psn + pdisk[pk].y * pcs);
            let o = (pt * rd.x + pb * rd.y) * pradius;
            psum += textureSampleCompareLevel(point_shadow_map, shadow_sampler, normalize(nd + o), pidx, pref - pbias);
          }
          attenuation = attenuation * (psum / 8.0);
        }
      }
    }

    let h = normalize(v + l + vec3<f32>(0.00001));
    let NdotL = max(dot(n, l), 0.0001);
    let NdotH = max(dot(n, h), 0.0001);
    let HdotV = max(dot(h, v), 0.0001);

    let alpha = roughness * roughness;
    let alpha2 = alpha * alpha;
    let NdotH2 = NdotH * NdotH;

    let denom = (NdotH2 * (alpha2 - 1.0) + 1.0);
    let D = alpha2 / (3.14159265 * denom * denom + 0.0001);

    // Height-correlated Smith visibility (Heitz). The Vis term already folds in the
    // 1/(4·NdotV·NdotL) denominator, so it replaces both the separable G and that divide — more
    // physically accurate than the UE4 k-remap and matches EEVEE's direct-specular response.
    let ggxV = NdotL * sqrt(NdotV * NdotV * (1.0 - alpha2) + alpha2);
    let ggxL = NdotV * sqrt(NdotL * NdotL * (1.0 - alpha2) + alpha2);
    let Vis = 0.5 / max(ggxV + ggxL, 1e-4);

    let F = F0 + (1.0 - F0) * pow5(clamp(1.0 - HdotV, 0.0, 1.0));

    let specular = D * Vis * F * energy_comp;
    let kS = F;
    var kD = vec3<f32>(1.0) - kS;
    kD = kD * (1.0 - metallic);

    // Clearcoat specular: sharp GGX (clearcoat roughness from cc_alpha2 above) + Kelemen visibility.
    let cc_denom = (NdotH2 * (cc_alpha2 - 1.0) + 1.0);
    let cc_D = cc_alpha2 / (3.14159265 * cc_denom * cc_denom + 0.0001);
    let cc_Fr = 0.04 + 0.96 * pow5(clamp(1.0 - HdotV, 0.0, 1.0));
    let cc_vis = 0.25 / (HdotV * HdotV + 0.0001);
    Lo_cc += vec3<f32>(cc_D * cc_Fr * cc_vis) * light_color * attenuation * NdotL;

    let direct_lobe = (kD * base_color.rgb / 3.14159265 + specular) * light_color * attenuation * NdotL;
    Lo += direct_lobe;
    if (i == 0u && l_data.position_or_dir.w == 0.0) {
      sun_direct += direct_lobe;
    }
  }

  // Cubemap IBL. The environment (procedural studio or a prefiltered HDRI) is baked into
  // env_cube with a roughness mip chain; we sample diffuse irradiance from the blurriest
  // mip along N and glossy reflection along R at a roughness-selected LOD. light.post.w
  // carries the max mip index (env LOD range).
  let env_max_lod = light.post.w;
  var R           = reflect(-v, n);
  // Reflection-probe box projection (EEVEE-style): when a finite probe box is active, re-aim the
  // reflection vector at the point where it hits the box wall, then re-base it at the box centre.
  // This anchors reflections to the room instead of an infinitely-distant sky.
  if (light.probe_center.w > 0.5) {
    let bmin = light.probe_center.xyz - light.probe_extents.xyz;
    let bmax = light.probe_center.xyz + light.probe_extents.xyz;
    let inv_r = 1.0 / R;
    let t_to_max = (bmax - in.world_pos) * inv_r;
    let t_to_min = (bmin - in.world_pos) * inv_r;
    let t_far = max(t_to_max, t_to_min); // far slab hit per axis
    let dist = min(min(t_far.x, t_far.y), t_far.z);
    if (dist > 0.0) {
      let hit = in.world_pos + R * dist;
      R = normalize(hit - light.probe_center.xyz);
    }
  }
  let irradiance  = textureSampleLevel(env_cube, env_samp, n, env_max_lod).rgb;
  let F_ibl       = F0 + (vec3<f32>(1.0) - F0) * pow5(clamp(1.0 - NdotV, 0.0, 1.0));
  var kD_ibl      = (vec3<f32>(1.0) - F_ibl) * (1.0 - metallic);
  let diffuse_ibl = irradiance * base_color.rgb * kD_ibl;
  // Specular: perceptual roughness -> mip LOD (sqrt curve widens low-roughness detail).
  let env_color   = textureSampleLevel(env_cube, env_samp, R, sqrt(roughness) * env_max_lod).rgb; // roughness already in [0.05,1]
  // Multi-scatter energy compensation (Fdez-Agüera) — without it, rough metals lose energy and
  // read dull/dark (EEVEE compensates for this). FssEss is the single-scatter term; Fms*Ems adds
  // back the light lost to multiple microfacet bounces. Reuses `dfg` from the direct-light
  // energy compensation above (identical (roughness, NdotV) arguments).
  let FssEss  = F0 * dfg.x + vec3<f32>(dfg.y);
  let Ess     = dfg.x + dfg.y;
  let Ems     = 1.0 - Ess;
  let Favg    = F0 + (vec3<f32>(1.0) - F0) / 21.0;
  let Fms     = FssEss * Favg / (vec3<f32>(1.0) - Ems * Favg);
  // Physically-weighted environment specular (no global boost). Glossiness on painted surfaces
  // comes from the clearcoat lobe (below), not a blanket multiplier.
  // Specular occlusion / horizon fade (Lagarde): when normal-mapping (or probe box-projection)
  // aims the reflection below the true geometric surface, fade the env reflection so concave
  // grazing areas don't glow. On flat un-mapped surfaces dot(R, geo_n) = NdotV >= 0 → no-op.
  let spec_horizon = clamp(1.0 + dot(R, normalize(in.normal)), 0.0, 1.0);
  let spec_occ = spec_horizon * spec_horizon;
  let specular_ibl = env_color * (FssEss + Fms * Ems) * spec_occ;
  // ambient_color.a is the IBL intensity (currently 1.0; exposes a tuning knob via C# later).
  // The sun shadow darkens the ambient FILL (diffuse IBL) to ground objects, but must NOT kill the
  // environment REFLECTION (specular IBL): a shadow blocks the sun, not the sky, so a shadowed
  // glass/metal still mirrors the world. Floor the fill well above black too — the sky always fills
  // shadows, and SSAO/GTAO handles genuine ambient occlusion. (Previously this multiplied BOTH terms
  // and could reach 0, crushing shadowed reflective surfaces — and anything refracting them — to pure
  // black.)
  let shadow_ao = max(mix(1.0 - light.shadow.x, 1.0, shadow_factor), 0.3);
  // Baked occlusion (glTF ORM: occlusion in the MR map's R channel, gated by surface.z so plain
  // metallic-roughness maps aren't misread as AO). Full strength on diffuse ambient; half-weighted
  // on the specular reflection (Filament-style: baked AO over-darkens sharp reflections).
  let occl = mix(1.0, mr_sample.r, clamp(model_data.surface.z, 0.0, 1.0));
  // Diffuse IBL is the fill light → scaled by the ambient-intensity knob and grounded by shadow_ao.
  // Specular IBL is the ENVIRONMENT REFLECTION → stays at full strength so metals/glass keep glossy.
  let ambient = diffuse_ibl * light.ambient_color.a * shadow_ao * occl + specular_ibl * mix(1.0, occl, 0.5);
  // Clearcoat layer: a sharp Fresnel-weighted environment reflection over the paint. The
  // base layer is dimmed by the coat's reflectance so energy is roughly conserved.
  // The coat reflects the environment at ITS OWN roughness (cc_rough), not a forced mip-0 mirror —
  // so a rough clearcoat blurs its reflection the way the base layer does, making cc_rough actually
  // affect reflections instead of only the direct-light lobe. Guarded on cc_strength: clearcoat is
  // off in the common case, so skip the Fresnel + env cube fetch entirely then (ambient_lobe →
  // ambient * 1.0 + 0). textureSampleLevel with explicit LOD is legal in non-uniform control flow.
  var cc_F_ibl = 0.0;
  var cc_env = vec3<f32>(0.0);
  if (cc_strength > 0.0) {
    cc_F_ibl = (0.04 + 0.96 * pow5(clamp(1.0 - NdotV, 0.0, 1.0))) * cc_strength;
    let cc_env_color = textureSampleLevel(env_cube, env_samp, R, sqrt(clamp(cc_rough, 0.0, 1.0)) * env_max_lod).rgb;
    cc_env = cc_env_color * cc_F_ibl;
  }
  let ambient_lobe = ambient * (1.0 - cc_F_ibl) + cc_env;
  var color = ambient_lobe + Lo * (1.0 - cc_F_ibl) + Lo_cc * cc_strength + emissive_rgb;
  // Compact lighting metadata for the RT composite. Opaque scene alpha carries the fraction
  // attributable to environment/indirect light, while view_pos.w stores 2 + the direct-sun
  // fraction. This keeps the forward renderer while allowing RT AO and visibility to affect
  // the physically correct lighting lobes instead of darkening emissive and reflections too.
  let lit_luma = max(dot(color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0001);
  let ambient_ratio = clamp(dot(ambient_lobe, vec3<f32>(0.2126, 0.7152, 0.0722)) / lit_luma, 0.0, 1.0);
  let sun_ratio = clamp(dot(sun_direct * (1.0 - cc_F_ibl), vec3<f32>(0.2126, 0.7152, 0.0722)) / lit_luma, 0.0, 1.0);

  // Selection rim glow (effect.y == 1 when selected in editor). Suppressed in diagnostic mode
  // and while a debug view is active so it can't leak into the inspected image.
  if (model_data.effect.y == 1u && light.debug.x == 0u && light.debug.y == 0u) {
    let rim = 1.0 - clamp(dot(n, v), 0.0, 1.0);
    color = color + vec3<f32>(0.1, 0.5, 1.0) * pow(rim, 2.0) * 1.8;
  }

  // NOTE: exposure / ACES tonemap / gamma / saturation / contrast now live in the
  // dedicated post-processing tonemap pass (renderPostProcess). This shader writes
  // LINEAR HDR radiance into the rgba16float scene buffer so bloom and tonemapping
  // can operate on the full dynamic range.

  // Editor world-space grid overlay (anti-aliased derivative-based). Applied in linear
  // space; it darkens toward a small linear value so it survives tonemapping downstream.
  // Suppressed in diagnostic mode so it can't darken inspected materials near y=0.
  if (light.debug.y == 0u) {
  let gx = in.world_pos.x;
  let gz = in.world_pos.z;
  let fw_gx = fwidth(gx);
  let fw_gz = fwidth(gz);
  let ln_x = 1.0 - smoothstep(fw_gx * 0.5, fw_gx * 1.5, abs(fract(gx + 0.5) - 0.5));
  let ln_z = 1.0 - smoothstep(fw_gz * 0.5, fw_gz * 1.5, abs(fract(gz + 0.5) - 0.5));
  let mj_x = 1.0 - smoothstep(fw_gx * 0.5, fw_gx * 1.5, abs(fract(gx / 5.0 + 0.5) - 0.5));
  let mj_z = 1.0 - smoothstep(fw_gz * 0.5, fw_gz * 1.5, abs(fract(gz / 5.0 + 0.5) - 0.5));
  let grid  = max(max(ln_x, ln_z) * 0.06, max(mj_x, mj_z) * 0.16);
  let gfade = clamp(1.0 - abs(in.world_pos.y) * 3.0, 0.0, 1.0);
  // Only upward-facing (floor-like) surfaces receive the grid. Without this gate the world-space
  // x/z lines painted a dark vertical stripe across ANY mesh crossing y≈0 — e.g. a sphere at the
  // origin got a dark seam down its front where the x=0 grid plane sliced it.
  let up_face = smoothstep(0.55, 0.85, normalize(in.normal).y);
  // Faint editor grid only — the floor reads as a clean studio surface (EEVEE "final render"
  // look) instead of graph paper; reflections + contact shadow now ground the car.
  color = mix(color, vec3<f32>(0.006, 0.006, 0.008), clamp(grid * gfade * up_face, 0.0, 0.22));
  }

  // Transparency. Opaque geometry stays fully opaque (out_alpha = 1). The alpha mode rides in
  // effect.z: 2 = generic alpha blend (decals, stickers, hair, cloth — the lit surface with the
  // texture's own alpha), 3 = dedicated glass (windows, lenses). Both render on the transparent
  // pipeline (depth-write off, src-alpha blend on); opaque uses a non-blending pipeline.
  var out_alpha = 1.0;
  if (model_data.effect.z == 2u) {
    // Generic alpha blend: keep the lit PBR colour, take opacity straight from the material /
    // texture alpha. This is what most glTF BLEND materials want — NOT the glass response.
    out_alpha = clamp(base_color.a, 0.0, 1.0);
  } else if (model_data.effect.z == 3u) {
    // Dedicated glass with SCREEN-SPACE REFRACTION. We sample the opaque scene (refraction_tex — a
    // copy taken before this glass pass) at a UV bent by the physically refracted view ray, so the
    // background visibly distorts through the glass. A Fresnel-weighted environment mirror sits on
    // top; transmitted light is tinted by the base colour (Beer-Lambert-ish absorption). The fragment
    // composites the background itself, so it is OPAQUE (out_alpha = 1) — no alpha haze.
    // Real-IOR Fresnel (KHR_materials_ior): F0 = ((n-1)/(n+1))² — 1.5 (crown glass) → the classic
    // 0.04; denser glass (diamond 2.4 → 0.17) gets the visibly stronger mirror rim it should.
    let g_ior = clamp(select(1.5, model_data.surface.x, model_data.surface.x >= 1.0), 1.0, 3.0);
    let F0_glass = pow((g_ior - 1.0) / (g_ior + 1.0), 2.0);
    let glass_fresnel = pow5(clamp(1.0 - NdotV, 0.0, 1.0));
    let F_glass = F0_glass + (1.0 - F0_glass) * glass_fresnel;
    let screen_dims = vec2<f32>(textureDimensions(refraction_tex, 0));
    let screen_uv = in.clip_pos.xy / screen_dims;       // framebuffer pixel → [0,1], top-left origin
    // Physical refraction direction (Snell): bend the view ray through the surface and offset the
    // screen lookup by the bent ray's DEVIATION from the straight-through ray, in view space. At
    // normal incidence the deviation is zero (no fake wobble); at the silhouette of a sphere it is
    // large (the magnify/invert tell-tale); ior → 1 degenerates to a perfectly clear pass-through.
    // eta = 1/ior < 1 entering the denser medium, so refract() can never TIR here.
    let refr_world = refract(-v, n, 1.0 / g_ior);
    let refr_view = (camera.view * vec4<f32>(refr_world, 0.0)).xyz;
    let fwd_view = (camera.view * vec4<f32>(-v, 0.0)).xyz;
    let deviation = refr_view.xy - fwd_view.xy;
    // Refraction only makes sense for CLOSE content behind the glass (a bottle's ship/liquid). Refracting
    // DISTANT scenery seen through the glass (a car window looking out at the track) just shifts far
    // objects into a doubled/ghosted copy — the reported artifact. So scale refraction by the depth gap
    // between the glass and the surface behind it, as a BAND: rises for near content, falls back to zero
    // for far backgrounds (and sky, which has no geometry → treated as far). This also keeps thin flat
    // glass clean, since a window's interior/exterior is either far or absent.
    let bg_pos = textureLoad(gbuf_pos_tex, vec2<i32>(in.clip_pos.xy), 0);
    let glass_vz = (camera.view * vec4<f32>(in.world_pos, 1.0)).z;
    let thickness = select(100.0, max(glass_vz - bg_pos.z, 0.0), bg_pos.w > 0.5);
    let thick_scale = smoothstep(0.03, 0.15, thickness) * (1.0 - smoothstep(0.7, 1.6, thickness));
    // Damp the bend slightly at grazing angles — reflection dominates there anyway and the lookup
    // would otherwise walk off-screen into the edge fade.
    let refr_strength = 0.5 * (1.0 - glass_fresnel) * thick_scale;
    let refr_uv_raw = screen_uv + vec2<f32>(deviation.x, -deviation.y) * refr_strength;
    // Fade refraction out as the sample nears the screen edge — clamping there smears the edge pixels
    // into long streaks (the classic screen-space-refraction artifact). Off-screen → environment mirror.
    let edge2 = min(refr_uv_raw, vec2<f32>(1.0) - refr_uv_raw);
    let valid = smoothstep(0.0, 0.04, min(edge2.x, edge2.y));
    let refr_uv = clamp(refr_uv_raw, vec2<f32>(0.0), vec2<f32>(1.0));
    // Frosted glass: rough glass scatters both the transmitted and reflected light. Blur the
    // refraction lookup (hexagonal taps, rotated per-pixel so the fixed pattern dissolves to smooth
    // scatter under TAA instead of printing a visible hex ghost) and pull the reflection to a rougher
    // env mip, both scaled by roughness. Smooth glass (frost≈0) keeps the single-sample fast path.
    let frost = clamp(roughness, 0.0, 1.0);
    var behind = textureSampleLevel(refraction_tex, env_samp, refr_uv, 0.0).rgb;
    if (frost > 0.04) {
      let fb = frost * 0.05; // transmission blur radius (uv)
      let frot = mat2x2<f32>(dither_cs, dither_sn, -dither_sn, dither_cs);
      var fd = array<vec2<f32>, 6>(
        vec2<f32>( 1.0, 0.0), vec2<f32>(-0.5, 0.866), vec2<f32>(-0.5, -0.866),
        vec2<f32>(-1.0, 0.0), vec2<f32>( 0.5, 0.866), vec2<f32>( 0.5, -0.866));
      var facc = behind;
      for (var fi: i32 = 0; fi < 6; fi = fi + 1) {
        let fuv = clamp(refr_uv + (frot * fd[fi]) * fb, vec2<f32>(0.0), vec2<f32>(1.0));
        facc += textureSampleLevel(refraction_tex, env_samp, fuv, 0.0).rgb;
      }
      behind = facc / 7.0;
    }
    let absorption = mix(vec3<f32>(1.0), base_color.rgb, clamp(1.0 - base_color.a, 0.0, 0.9));
    let env_refl = textureSampleLevel(env_cube, env_samp, R, sqrt(frost) * light.post.w).rgb; // rough → blurred mirror
    let transmitted = mix(env_refl, behind * absorption, valid); // off-screen sample → reflection
    // Fresnel: clear transmission head-on, environment mirror at grazing. Direct-light glints ride
    // the sharp clearcoat-shaped lobe, scaled to the material's real F0 so dense glass sparkles
    // harder. Kept subtle so the glass reads crystal-clear, not milky.
    let glass_col = mix(transmitted, env_refl, F_glass) + Lo_cc * (0.5 * F0_glass / 0.04) + emissive_rgb;
    // KHR_materials_transmission: partial transmission blends the standard lit surface toward the
    // glass response (native side maps legacy glass-with-no-transmission to 1.0 = full glass).
    color = mix(color, glass_col, clamp(model_data.surface.y, 0.0, 1.0));
    out_alpha = 1.0; // background already composited into `color`
  }
  // Atmospheric fog / aerial perspective (applied to the final lit + glass colour, in linear HDR so
  // bloom picks up the sun-facing in-scatter glow). No-op when density is 0 (default / diagnostic).
  color = applyFog(color, in.world_pos, camera.camera_pos);

  let view_n = vec4<f32>(normalize((camera.view * vec4<f32>(n, 0.0)).xyz), roughness);

  // ── Debug views (material / geometry channels) ──
  // When a debug view is active, replace the lit colour with a display-ready visualisation of
  // the requested channel. The tonemap pass passes these through un-tonemapped (codes 1..10),
  // while codes 11..15 are produced there from the post-process buffers. Drawn opaque.
  let dv = light.debug.x;
  if (dv != 0u) {
    out_alpha = 1.0;
    if (dv == 1u)       { color = base_color.rgb; }
    else if (dv == 2u)  { color = n * 0.5 + vec3<f32>(0.5); }
    else if (dv == 3u)  { color = view_n.xyz * 0.5 + vec3<f32>(0.5); }
    else if (dv == 4u)  { color = vec3<f32>(roughness); }
    else if (dv == 5u)  { color = vec3<f32>(metallic); }
    else if (dv == 6u)  { color = vec3<f32>(base_color.a); }
    else if (dv == 7u)  { color = emissive_rgb; }
    else if (dv == 8u)  { color = vec3<f32>(clamp(-vpos.z / 50.0, 0.0, 1.0)); }
    else if (dv == 9u)  { color = clamp(vpos.xyz * 0.1 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0)); }
    else if (dv == 10u) { color = vec3<f32>(shadow_factor); }
  }
  var meta_pos = vpos;
  meta_pos.w = 2.0 + sun_ratio;
  let scene_alpha = select(ambient_ratio, out_alpha, model_data.effect.z == 2u || model_data.effect.z == 3u);
  return FragOut(vec4<f32>(color, scene_alpha), meta_pos, view_n, vec4<f32>(base_color.rgb, 1.0));
}
