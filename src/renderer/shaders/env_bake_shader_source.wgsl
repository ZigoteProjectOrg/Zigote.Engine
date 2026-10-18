struct EnvBakeParams {
  face_mode_rough: vec4<f32>, // x=face index, y=mode, z=roughness, w=sky mode behind a probe (0/1)
  sky_horizon: vec4<f32>,
  sky_zenith: vec4<f32>,
  sky_ground: vec4<f32>,
  sun_dir: vec4<f32>,         // xyz key dir, w intensity
  studio: vec4<f32>,          // x=overhead, y=horizon glow, z=key sharpness, w=unused
}
@group(0) @binding(0) var<uniform> p: EnvBakeParams;
@group(0) @binding(1) var equirect: texture_2d<f32>;
@group(0) @binding(2) var equirect_samp: sampler;
// Mode 2 (captured probe): the scene rendered around the viewer + its G-buffer metadata, whose y > 0.5
// marks geometry. Where nothing was drawn (open sky, through windows) the regular sky source shows.
@group(0) @binding(3) var probe_color: texture_cube<f32>;
@group(0) @binding(4) var probe_meta: texture_cube<f32>;

const PI: f32 = 3.14159265;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
  let x = f32((vid << 1u) & 2u);
  let y = f32(vid & 2u);
  var out: VOut;
  out.uv = vec2<f32>(x, y);
  out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
  return out;
}

// Cube face basis matching wgpu's layer order: +X,-X,+Y,-Y,+Z,-Z.
fn faceDir(face: u32, uv: vec2<f32>) -> vec3<f32> {
  let a = uv * 2.0 - vec2<f32>(1.0);
  var d: vec3<f32>;
  if (face == 0u)      { d = vec3<f32>( 1.0, -a.y, -a.x); }
  else if (face == 1u) { d = vec3<f32>(-1.0, -a.y,  a.x); }
  else if (face == 2u) { d = vec3<f32>( a.x,  1.0,  a.y); }
  else if (face == 3u) { d = vec3<f32>( a.x, -1.0, -a.y); }
  else if (face == 4u) { d = vec3<f32>( a.x, -a.y,  1.0); }
  else                 { d = vec3<f32>(-a.x, -a.y, -1.0); }
  return normalize(d);
}

// Rectangular softbox seen from the origin: `d` projected onto the panel's plane (gnomonic), then a
// smooth box mask. half = tangent-space half extents along (u, v); soft widens the edge falloff.
fn softbox(d: vec3<f32>, c: vec3<f32>, half: vec2<f32>, soft: f32) -> f32 {
  let z = dot(d, c);
  if (z <= 0.0) { return 0.0; }
  let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(c.y) > 0.95);
  let u = normalize(cross(up, c));
  let v = cross(c, u);
  let q = abs(vec2<f32>(dot(d, u), dot(d, v)) / z) - half;
  let e = vec2<f32>(0.004) + soft * (half + vec2<f32>(0.05));
  let m = vec2<f32>(1.0) - smoothstep(-e, e, q);
  return m.x * m.y;
}

// The procedural studio at a given edge softness (0 = crisp; the GGX prefilter below does the real
// roughness blur and only uses `soft` to keep its finite sample count from banding on hard edges).
fn proceduralEnv(d: vec3<f32>, soft: f32) -> vec3<f32> {
  let sky_up = mix(p.sky_horizon.rgb, p.sky_zenith.rgb, pow(clamp(d.y, 0.0, 1.0), 0.55));
  // A readable horizon line: glossy surfaces need a light/dark split to show their curvature.
  let ground = smoothstep(0.0, 0.02 + soft * 0.3, -d.y);
  let base   = mix(sky_up, p.sky_ground.rgb * mix(1.0, 0.7, clamp(-d.y * 2.0, 0.0, 1.0)), ground);
  let sun_spot = pow(clamp(dot(d, p.sun_dir.xyz), 0.0, 1.0), max(p.studio.z * (1.0 - soft * 0.8), 1.0)) * p.sun_dir.w;
  let overhead = smoothstep(0.25, 0.85, d.y) * p.studio.x;
  let horizon  = exp(-(d.y * d.y) * 14.0) * p.studio.y;
  // Car-studio softboxes: a long overhead strip, a tall key strip, a broad dim fill and a narrow back
  // rim. Crisp rectangles reflect as the streaks that read glossy paint, chrome and glass. Scaled by
  // the "Overhead Softbox" knob so they stay editor-tunable.
  let boxes =
      softbox(d, normalize(vec3<f32>( 0.0, 1.0,  0.12)), vec2<f32>(0.22, 1.10), soft) * 1.6
    + softbox(d, normalize(vec3<f32>( 0.65, 0.45, 0.62)), vec2<f32>(0.16, 0.55), soft) * 2.0
    + softbox(d, normalize(vec3<f32>(-0.85, 0.35, 0.25)), vec2<f32>(0.55, 0.30), soft) * 0.7
    + softbox(d, normalize(vec3<f32>( 0.15, 0.30, -0.95)), vec2<f32>(0.70, 0.08), soft) * 1.8;
  let studio = vec3<f32>(1.0, 0.98, 0.94) * ((sun_spot + overhead + horizon) * 0.4 + boxes * p.studio.x * 0.45);
  return base + studio;
}

// GGX prefilter of the procedural studio — the same lobe the HDRI path integrates, so a mip baked at
// roughness r is what the mesh shader expects when it samples at r.
fn prefilterProcedural(d: vec3<f32>, rough: f32) -> vec3<f32> {
  if (rough < 0.01) { return proceduralEnv(d, 0.0); }
  var col = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = 0u; i < 128u; i = i + 1u) {
    let xi = vec2<f32>(f32(i) / 128.0, radicalInverse(i));
    let h = importanceGGX(xi, d, rough);
    let l = normalize(2.0 * dot(d, h) * h - d);
    let ndl = dot(d, l);
    if (ndl > 0.0) {
      col = col + min(proceduralEnv(l, rough * 0.5), vec3<f32>(20.0)) * ndl;
      wsum = wsum + ndl;
    }
  }
  return select(proceduralEnv(d, rough), col / wsum, wsum > 0.0);
}

fn dirToEquirectUv(d: vec3<f32>) -> vec2<f32> {
  return vec2<f32>(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}

fn radicalInverse(bitsIn: u32) -> f32 {
  var bits = bitsIn;
  bits = (bits << 16u) | (bits >> 16u);
  bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
  bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
  bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
  bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
  return f32(bits) * 2.3283064365386963e-10;
}

fn importanceGGX(xi: vec2<f32>, n: vec3<f32>, rough: f32) -> vec3<f32> {
  let a = rough * rough;
  let phi = 2.0 * PI * xi.x;
  let cosT = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinT = sqrt(1.0 - cosT * cosT);
  let h = vec3<f32>(cos(phi) * sinT, sin(phi) * sinT, cosT);
  var up = vec3<f32>(0.0, 0.0, 1.0);
  if (abs(n.z) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  return normalize(tx * h.x + ty * h.y + n * h.z);
}

fn prefilterHdri(d: vec3<f32>, rough: f32) -> vec3<f32> {
  if (rough < 0.01) {
    return textureSampleLevel(equirect, equirect_samp, dirToEquirectUv(d), 0.0).rgb;
  }
  // Per-sample mip selection (Karis 2014): a broad GGX lobe spreads 128 samples thin, so reading
  // mip 0 aliases the bright HDRI sun into sparkles. Selecting a mip from the sample's solid angle
  // (saSample/saTexel) low-pass-filters before the average — far smoother glossy reflections.
  let dims = vec2<f32>(textureDimensions(equirect, 0));
  let sa_texel = (2.0 * PI / dims.x) * (PI / dims.y); // equirect texel solid angle (~equator)
  let a = rough * rough;
  var col = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = 0u; i < 128u; i = i + 1u) {
    let xi = vec2<f32>(f32(i) / 128.0, radicalInverse(i));
    let h = importanceGGX(xi, d, rough);
    let l = normalize(2.0 * dot(d, h) * h - d);
    let ndl = max(dot(d, l), 0.0);
    if (ndl > 0.0) {
      let ndh = max(dot(d, h), 1e-4);
      let dterm = (ndh * ndh * (a * a - 1.0) + 1.0);
      let dgg = (a * a) / (PI * dterm * dterm + 1e-6);
      let pdf = dgg * 0.25 + 1e-4;                    // pdf = D·NdotH/(4·VdotH); N=V=R → D/4
      let sa_sample = 1.0 / (128.0 * pdf);
      let mip = max(0.5 * log2(sa_sample / sa_texel), 0.0);
      // Firefly clamp on top of the mip blur: the sharp mip-0 mirror path (rough<0.01) stays
      // unclamped, so chrome still shows the real bright sun.
      let s = min(textureSampleLevel(equirect, equirect_samp, dirToEquirectUv(l), mip).rgb, vec3<f32>(20.0));
      col = col + s * ndl;
      wsum = wsum + ndl;
    }
  }
  if (wsum > 0.0) { return col / wsum; }
  return textureSampleLevel(equirect, equirect_samp, dirToEquirectUv(d), 0.0).rgb;
}

// Cosine-weighted diffuse irradiance convolution (the proper integral for the blurriest/diffuse
// mip). Reusing the GGX-prefiltered specular mip as diffuse irradiance left matte surfaces dark
// and firefly-noisy on a real HDRI; this samples the full hemisphere with a hard firefly clamp.
fn irradianceHdri(n: vec3<f32>) -> vec3<f32> {
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  var col = vec3<f32>(0.0);
  let samples = 512u;
  for (var i = 0u; i < samples; i = i + 1u) {
    let xi = vec2<f32>(f32(i) / f32(samples), radicalInverse(i));
    let phi = 2.0 * PI * xi.x;
    let ct = sqrt(1.0 - xi.y);   // cosine-weighted: pdf = cos/PI, estimator = mean(radiance)
    let st = sqrt(xi.y);
    let l = tx * (cos(phi) * st) + ty * (sin(phi) * st) + n * ct;
    // Hard firefly clamp — irradiance is low-frequency, so capping the sun keeps it smooth
    // without visibly changing the diffuse fill level.
    col = col + min(textureSampleLevel(equirect, equirect_samp, dirToEquirectUv(l), 0.0).rgb, vec3<f32>(12.0));
  }
  return col / f32(samples);
}

// Cosine-weighted diffuse irradiance convolution of the PROCEDURAL studio (the analytic analogue
// of irradianceHdri). Without this, the blurriest procedural mip was just `proceduralEnv(d, 1)` —
// still directional, so the lower hemisphere of every matte object went near-black and its albedo
// never read (EEVEE integrates the world over the hemisphere, lighting all orientations evenly).
fn irradianceProcedural(n: vec3<f32>) -> vec3<f32> {
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  var col = vec3<f32>(0.0);
  let samples = 256u;
  for (var i = 0u; i < samples; i = i + 1u) {
    let xi = vec2<f32>(f32(i) / f32(samples), radicalInverse(i));
    let phi = 2.0 * PI * xi.x;
    let ct = sqrt(1.0 - xi.y);   // cosine-weighted: estimator = mean(radiance)
    let st = sqrt(xi.y);
    let l = tx * (cos(phi) * st) + ty * (sin(phi) * st) + n * ct;
    // Sample the near-sharp studio so bright panels contribute to the fill; firefly-clamp the sun.
    col = col + min(proceduralEnv(l, 0.25), vec3<f32>(8.0));
  }
  return col / f32(samples);
}

fn skyEnv(d: vec3<f32>) -> vec3<f32> {
  if (p.face_mode_rough.w > 0.5) { return textureSampleLevel(equirect, equirect_samp, dirToEquirectUv(d), 0.0).rgb; }
  return proceduralEnv(d, 0.0);
}

fn probeEnv(d: vec3<f32>) -> vec3<f32> {
  if (textureSampleLevel(probe_meta, equirect_samp, d, 0.0).y > 0.5) {
    return textureSampleLevel(probe_color, equirect_samp, d, 0.0).rgb;
  }
  return skyEnv(d);
}

// GGX prefilter / cosine irradiance of the captured probe — the same integrals as the sky paths, so
// the mesh shader reads both cubes with one roughness → mip mapping.
fn prefilterProbe(d: vec3<f32>, rough: f32) -> vec3<f32> {
  if (rough < 0.01) { return probeEnv(d); }
  var col = vec3<f32>(0.0);
  var wsum = 0.0;
  for (var i = 0u; i < 128u; i = i + 1u) {
    let xi = vec2<f32>(f32(i) / 128.0, radicalInverse(i));
    let h = importanceGGX(xi, d, rough);
    let l = normalize(2.0 * dot(d, h) * h - d);
    let ndl = dot(d, l);
    if (ndl > 0.0) {
      col = col + min(probeEnv(l), vec3<f32>(20.0)) * ndl;
      wsum = wsum + ndl;
    }
  }
  return select(probeEnv(d), col / wsum, wsum > 0.0);
}

fn irradianceProbe(n: vec3<f32>) -> vec3<f32> {
  var up = vec3<f32>(0.0, 1.0, 0.0);
  if (abs(n.y) > 0.999) { up = vec3<f32>(1.0, 0.0, 0.0); }
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  var col = vec3<f32>(0.0);
  let samples = 512u;
  for (var i = 0u; i < samples; i = i + 1u) {
    let xi = vec2<f32>(f32(i) / f32(samples), radicalInverse(i));
    let phi = 2.0 * PI * xi.x;
    let ct = sqrt(1.0 - xi.y);
    let st = sqrt(xi.y);
    let l = tx * (cos(phi) * st) + ty * (sin(phi) * st) + n * ct;
    col = col + min(probeEnv(l), vec3<f32>(12.0));
  }
  return col / f32(samples);
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let face  = u32(p.face_mode_rough.x + 0.5);
  let mode  = u32(p.face_mode_rough.y + 0.5);
  let rough = p.face_mode_rough.z;
  let d = faceDir(face, in.uv);
  var c: vec3<f32>;
  // The blurriest mip is sampled as diffuse irradiance by the mesh shader → use a proper cosine
  // hemisphere convolution there; the sharper mips are the GGX specular prefilter.
  if (mode == 2u) {
    if (rough > 0.95) { c = irradianceProbe(d); }
    else { c = prefilterProbe(d, rough); }
  } else if (mode == 1u) {
    if (rough > 0.95) { c = irradianceHdri(d); }
    else { c = prefilterHdri(d, rough); }
  } else {
    if (rough > 0.95) { c = irradianceProcedural(d); }
    else { c = prefilterProcedural(d, rough); }
  }
  return vec4<f32>(c, 1.0);
}
