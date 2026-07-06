// Particle billboard shader (Zigote VFX). Each particle is one instance; the quad is generated from
// @builtin(vertex_index) (vertex-pulling — no quad/index buffer), expanded into a camera-facing
// billboard using the camera's world-space right/up from the view matrix rows. Renders inside the main
// geometry pass after the transparent meshes, so it depth-tests against the opaque scene (no depth write).
// Two pipelines share this shader: additive and (premultiplied) alpha blend. The G-buffer MRT targets
// (1 = view-position, 2 = view-normal) are masked off in the pipeline, so particles never disturb SSAO/SSR.

struct Camera {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    camera_pos: vec4<f32>,
    inv_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) size: f32,
    @location(2) rot: f32,
    @location(3) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) gpos: vec4<f32>,
    @location(2) gnorm: vec4<f32>,
};

@vertex
fn vs_particle(@builtin(vertex_index) vid: u32, in: VsIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.5),
    );
    let corner = corners[vid];

    // The view matrix transforms world -> view; its rows are the camera's world-space basis vectors.
    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);

    let c = cos(in.rot);
    let s = sin(in.rot);
    let rc = vec2<f32>(corner.x * c - corner.y * s, corner.x * s + corner.y * c);

    let world = in.pos + (right * rc.x + up * rc.y) * in.size;

    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.color = in.color;
    out.uv = corner + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_particle(in: VsOut) -> FsOut {
    // Soft round sprite: radial falloff so the quad reads as a disc, not a square.
    let d = length(in.uv - vec2<f32>(0.5, 0.5)) * 2.0;
    let falloff = clamp(1.0 - d, 0.0, 1.0);
    let a = in.color.a * falloff * falloff;

    // Premultiplied output: the additive pipeline (one, one) adds color*a; the alpha pipeline
    // (one, one_minus_src_alpha) composites premultiplied color correctly.
    var o: FsOut;
    o.color = vec4<f32>(in.color.rgb * a, a);
    o.gpos = vec4<f32>(0.0);
    o.gnorm = vec4<f32>(0.0);
    return o;
}
