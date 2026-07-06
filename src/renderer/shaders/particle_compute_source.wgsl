// GPU particle simulation kernel (Zigote VFX). One thread per particle slot. Persistent GPU-owned state
// buffer; the host computes a per-frame spawn budget and uploads the lowered emitter params. Each frame:
// dead slots claim a spawn budget atomically and respawn; live slots run the update modules + integrate;
// every slot writes its render data (pos/size/rot/color) into the instance buffer the billboard pass draws
// (dead = size 0 → degenerate quad → no fragments). Mirrors the CPU CpuParticleSimulator module set so the
// GPU path looks the same. The instance buffer is bound storage(here) + vertex(render) on the same buffer.

struct P {
    pa: vec4<f32>,  // xyz position, w age
    vl: vec4<f32>,  // xyz velocity, w lifetime
    sc: vec4<f32>,  // start colour rgba
    sr: vec4<f32>,  // x start_size, y rotation, z angular_vel, w unused
    sd: vec4<u32>,  // x per-particle seed
};

// All-f32 uniform layout (112 floats / 28 vec4); the host fills it 1:1. u32 fields are stored as exact
// small floats and cast in-kernel — avoids bit-reinterpret across the FFI.
struct Params {
    counts: vec4<f32>,  // x=capacity, y=spawn_count, z=frame_seed, w=shape
    flags: vec4<f32>,   // x=module bitmask, y=space (0=world), zw unused
    timing: vec4<f32>,  // x=dt, y=time, zw unused
    epos: vec4<f32>,    // emitter world position xyz
    erot: vec4<f32>,    // emitter world rotation quaternion xyzw
    edir: vec4<f32>,    // local emission axis xyz
    shape0: vec4<f32>,  // x=radius, y=cone_cos (cos of cone half-angle)
    boxext: vec4<f32>,  // box half-extents xyz
    life: vec4<f32>,    // x=life_min, y=life_max, z=speed_min, w=speed_max
    size: vec4<f32>,    // x=size_min, y=size_max, z=rot_min, w=rot_max
    spin: vec4<f32>,    // x=spin_min, y=spin_max
    col0: vec4<f32>,    // start colour
    col1: vec4<f32>,    // start colour variation (random lerp col0..col1)
    grav: vec4<f32>,    // gravity xyz, w=drag
    turb: vec4<f32>,    // x=strength, y=frequency
    vort: vec4<f32>,    // vortex axis xyz, w=strength
    ramp: array<vec4<f32>, 8>,  // 8 colour-over-life stops (uniform 0..1)
    szc: array<vec4<f32>, 2>,   // 8 size-over-life samples
    alc: array<vec4<f32>, 2>,   // 8 alpha-over-life samples
};

@group(0) @binding(0) var<storage, read_write> state: array<P>;
@group(0) @binding(1) var<storage, read_write> inst: array<f32>;
@group(0) @binding(2) var<uniform> prm: Params;
@group(0) @binding(3) var<storage, read_write> spawn_ctr: atomic<u32>;

const TAU: f32 = 6.2831853;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rndf(s: ptr<function, u32>) -> f32 {
    *s = pcg(*s);
    return f32(*s) * (1.0 / 4294967296.0);
}

fn rng_range(s: ptr<function, u32>, a: f32, b: f32) -> f32 {
    return a + (b - a) * rndf(s);
}

fn rand_sphere(s: ptr<function, u32>) -> vec3<f32> {
    let z = rndf(s) * 2.0 - 1.0;
    let a = rndf(s) * TAU;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

fn sample_ramp(t: f32) -> vec4<f32> {
    let x = clamp(t, 0.0, 1.0) * 7.0;
    let i = u32(floor(x));
    if (i >= 7u) { return prm.ramp[7]; }
    return mix(prm.ramp[i], prm.ramp[i + 1u], x - floor(x));
}

fn sample_curve(c0: vec4<f32>, c1: vec4<f32>, t: f32) -> f32 {
    var s = array<f32, 8>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w);
    let x = clamp(t, 0.0, 1.0) * 7.0;
    let i = u32(floor(x));
    if (i >= 7u) { return s[7]; }
    return mix(s[i], s[i + 1u], x - floor(x));
}

fn write_inst(idx: u32, pos: vec3<f32>, sz: f32, rot: f32, col: vec4<f32>) {
    let o = idx * 9u;
    inst[o + 0u] = pos.x;
    inst[o + 1u] = pos.y;
    inst[o + 2u] = pos.z;
    inst[o + 3u] = sz;
    inst[o + 4u] = rot;
    inst[o + 5u] = col.x;
    inst[o + 6u] = col.y;
    inst[o + 7u] = col.z;
    inst[o + 8u] = col.w;
}

fn spawn(idx: u32, claim: u32) -> P {
    var seed = pcg(u32(prm.counts.z) ^ (idx * 2654435761u) ^ (claim * 40503u) ^ 0x9E3779B9u);

    var axis = vec3<f32>(0.0, 1.0, 0.0);
    if (length(prm.edir.xyz) > 1e-5) { axis = normalize(prm.edir.xyz); }

    let radius = prm.shape0.x;
    let shape = u32(prm.counts.w);
    var lpos = vec3<f32>(0.0);
    var ldir = axis;

    if (shape == 1u) { // Sphere
        let d = rand_sphere(&seed) * radius * pow(rndf(&seed), 1.0 / 3.0);
        lpos = d;
        if (length(d) > 0.0) { ldir = normalize(d); }
    } else if (shape == 2u) { // Hemisphere
        var d = rand_sphere(&seed) * radius * pow(rndf(&seed), 1.0 / 3.0);
        if (dot(d, axis) < 0.0) { d = d - axis * (2.0 * dot(d, axis)); }
        lpos = d;
        if (length(d) > 0.0) { ldir = normalize(d); }
    } else if (shape == 3u) { // Box
        lpos = vec3<f32>(
            (rndf(&seed) * 2.0 - 1.0) * prm.boxext.x,
            (rndf(&seed) * 2.0 - 1.0) * prm.boxext.y,
            (rndf(&seed) * 2.0 - 1.0) * prm.boxext.z);
        ldir = axis;
    } else if (shape == 5u) { // Circle
        var refv = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(axis.y) > 0.99) { refv = vec3<f32>(1.0, 0.0, 0.0); }
        let tang = normalize(cross(refv, axis));
        let bitang = cross(axis, tang);
        let phi = rndf(&seed) * TAU;
        let radial = tang * cos(phi) + bitang * sin(phi);
        lpos = radial * radius;
        ldir = radial;
    } else if (shape == 4u) { // Cone (default)
        var refv = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(axis.y) > 0.99) { refv = vec3<f32>(1.0, 0.0, 0.0); }
        let tang = normalize(cross(refv, axis));
        let bitang = cross(axis, tang);
        let phiPos = rndf(&seed) * TAU;
        let rPos = sqrt(rndf(&seed)) * radius;
        lpos = (tang * cos(phiPos) + bitang * sin(phiPos)) * rPos;
        let z = rng_range(&seed, prm.shape0.y, 1.0);
        let sn = sqrt(max(0.0, 1.0 - z * z));
        let phiDir = rndf(&seed) * TAU;
        ldir = axis * z + (tang * cos(phiDir) + bitang * sin(phiDir)) * sn;
    }

    let life = max(0.0001, rng_range(&seed, prm.life.x, prm.life.y));
    let speed = rng_range(&seed, prm.life.z, prm.life.w);
    let ssize = rng_range(&seed, prm.size.x, prm.size.y);
    let rot = rng_range(&seed, prm.size.z, prm.size.w);
    let ang = rng_range(&seed, prm.spin.x, prm.spin.y);
    let col = mix(prm.col0, prm.col1, rndf(&seed));

    var p: P;
    p.pa = vec4<f32>(prm.epos.xyz + qrot(prm.erot, lpos), 0.0);
    p.vl = vec4<f32>(qrot(prm.erot, ldir) * speed, life);
    p.sc = col;
    p.sr = vec4<f32>(ssize, rot, ang, 0.0);
    p.sd = vec4<u32>(seed, 0u, 0u, 0u);
    return p;
}

@compute @workgroup_size(64)
fn cs_particle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u32(prm.counts.x)) { return; }

    var p = state[idx];
    let dt = prm.timing.x;
    let mask = u32(prm.flags.x);
    let alive = p.vl.w > 0.0 && p.pa.w < p.vl.w;

    if (!alive) {
        let claim = atomicAdd(&spawn_ctr, 1u);
        if (claim < u32(prm.counts.y)) {
            p = spawn(idx, claim);
        } else {
            write_inst(idx, vec3<f32>(0.0), 0.0, 0.0, vec4<f32>(0.0)); // stay dead
            state[idx] = p;
            return;
        }
    } else {
        var vel = p.vl.xyz;
        if ((mask & 1u) != 0u) { vel += prm.grav.xyz * dt; }                       // Gravity
        if ((mask & 2u) != 0u) { vel *= max(0.0, 1.0 - prm.grav.w * dt); }          // Drag
        if ((mask & 4u) != 0u) {                                                    // Turbulence
            let ph = f32(p.sd.x & 0xFFFFu) * (TAU / 65536.0);
            let tm = prm.timing.y;
            vel += vec3<f32>(
                sin(p.pa.y * prm.turb.y + tm + ph),
                sin(p.pa.z * prm.turb.y + tm * 1.3 + ph),
                sin(p.pa.x * prm.turb.y + tm * 0.7 + ph)) * (prm.turb.x * dt);
        }
        if ((mask & 8u) != 0u) {                                                    // Vortex
            vel += cross(prm.vort.xyz, p.pa.xyz) * (prm.vort.w * dt);
        }
        p.vl = vec4<f32>(vel, p.vl.w);
        p.pa = vec4<f32>(p.pa.xyz + vel * dt, p.pa.w + dt);
        p.sr.y += p.sr.z * dt;
    }

    let nage = clamp(p.pa.w / max(p.vl.w, 0.0001), 0.0, 1.0);
    var col = p.sc;
    if ((mask & 16u) != 0u) { col = sample_ramp(nage); }                           // ColorOverLife
    var sz = p.sr.x;
    if ((mask & 32u) != 0u) { sz = p.sr.x * sample_curve(prm.szc[0], prm.szc[1], nage); } // SizeOverLife
    if ((mask & 64u) != 0u) { col.w = sample_curve(prm.alc[0], prm.alc[1], nage); }       // AlphaOverLife

    write_inst(idx, p.pa.xyz, sz, p.sr.y, col);
    state[idx] = p;
}
