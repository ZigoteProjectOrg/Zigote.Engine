struct EnvBakeParams {
  face_mode_rough: vec4<f32>, // x=face index, y=mode, z=roughness, w=unused
  sky_horizon: vec4<f32>,
  sky_zenith: vec4<f32>,
  sky_ground: vec4<f32>,
  sun_dir: vec4<f32>,         // xyz key dir, w intensity
  studio: vec4<f32>,          // x=overhead, y=horizon glow, z=key sharpness, w=unused
}
@group(0) @binding(0) var<uniform> p: EnvBakeParams;
@group(0) @binding(1) var equirect: texture_2d<f32>;
@group(0) @binding(2) var equirect_samp: sampler;

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

fn proceduralEnv(d: vec3<f32>, rough: f32) -> vec3<f32> {
  let sky_up = mix(p.sky_horizon.rgb, p.sky_zenith.rgb, pow(clamp(d.y, 0.0, 1.0), 0.55));
  let base   = mix(sky_up, p.sky_ground.rgb, clamp(-d.y, 0.0, 1.0));
  // Studio key/softbox reflections, broadened as roughness rises (cheap analytic prefilter).
  let sharp    = mix(1.0, 0.2, rough);
  let sun_spot = pow(clamp(dot(d, p.sun_dir.xyz), 0.0, 1.0), max(p.studio.z * sharp, 1.0)) * p.sun_dir.w;
  let overhead = smoothstep(0.25, 0.85, d.y) * p.studio.x;
  let horizon  = exp(-(d.y * d.y) * 14.0) * p.studio.y;
  // Defined studio softboxes — bright panels (key / side fill / back rim) that reflect as
  // crisp streaks across glossy paint and glass (the car-studio look). Their angular size
  // narrows for smooth surfaces and broadens as roughness rises. Scaled by the "Overhead
  // Softbox" studio knob so they stay editor-tunable.
  let sb_sharp = mix(50.0, 6.0, rough);
  let key  = pow(clamp(dot(d, normalize(vec3<f32>( 0.4, 0.9, 0.5))), 0.0, 1.0), sb_sharp) * 2.2;
  let fill = pow(clamp(dot(d, normalize(vec3<f32>(-0.8, 0.6, 0.2))), 0.0, 1.0), sb_sharp * 0.7) * 0.9;
  let rim  = pow(clamp(dot(d, normalize(vec3<f32>( 0.3, 0.5, -0.9))), 0.0, 1.0), sb_sharp * 0.8) * 1.4;
  let softbox = (key + fill + rim) * p.studio.x * (1.0 - rough * 0.4);
  // Studio reflections are kept subtle so they read as soft highlights on glossy surfaces rather
  // than a high-contrast chrome studio that buries every material's albedo. (Was ~2.5x hotter.)
  let studio   = vec3<f32>(1.0, 0.98, 0.94) * (sun_spot + overhead + horizon + softbox) * (1.0 - rough * 0.5);
  return base + studio * 0.4;
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

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
  let face  = u32(p.face_mode_rough.x + 0.5);
  let mode  = u32(p.face_mode_rough.y + 0.5);
  let rough = p.face_mode_rough.z;
  let d = faceDir(face, in.uv);
  var c: vec3<f32>;
  // The blurriest mip is sampled as diffuse irradiance by the mesh shader → use a proper cosine
  // hemisphere convolution there; the sharper mips are the GGX specular prefilter.
  if (mode == 1u) {
    if (rough > 0.95) { c = irradianceHdri(d); }
    else { c = prefilterHdri(d, rough); }
  } else {
    if (rough > 0.95) { c = irradianceProcedural(d); }
    else { c = proceduralEnv(d, rough); }
  }
  return vec4<f32>(c, 1.0);
}
