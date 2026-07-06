// 2D sprite shader (default, shader handle 0). Each sprite is one instance; the quad is generated
// from @builtin(vertex_index) (vertex-pulling — no vertex/index buffer), rotated and placed in
// world space, then projected by the sprite camera (world ortho in play mode, the editor's
// perspective view-proj in edit mode — sprites are true world-space XY quads).
//
// This module is also the CONTRACT for custom sprite shaders (zigote_sprites_shader_create):
//   @group(0) camera UBO {view_proj, viewport(w, h, 0, 0)}
//   @group(1) primary texture + sampler
//   @group(2) material params UBO (16 floats, dynamic offset)
//   @group(3) secondary texture + sampler (white 1×1 when unset — LUT/mask/dissolve inputs)
// A custom module must define vs_main/fs_main over the same VsIn instance layout and output
// PREMULTIPLIED alpha (the alpha pipeline blends one / one-minus-src-alpha; additive one / one).

struct Camera {
    view_proj: mat4x4<f32>,
    viewport: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct Params {
    data: array<vec4<f32>, 4>,
};

@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var tex2: texture_2d<f32>;
@group(3) @binding(1) var samp2: sampler;

struct VsIn {
    @location(0) pos: vec3<f32>,
    @location(1) rot: f32,
    @location(2) size: vec2<f32>,
    @location(3) uv0: vec2<f32>,
    @location(4) uv1: vec2<f32>,
    @location(5) color: vec4<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32, in: VsIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5),
        vec2<f32>(-0.5, -0.5), vec2<f32>(0.5, 0.5), vec2<f32>(-0.5, 0.5),
    );
    let corner = corners[vid];

    let local = corner * in.size;
    let c = cos(in.rot);
    let s = sin(in.rot);
    let world = in.pos.xy + vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);

    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world, in.pos.z, 1.0);
    // corner.y = +0.5 is the sprite's top; v0 is the TOP of the sub-rect in image space
    // (textures upload row 0 = top), so the v axis mixes with 1 - t.y.
    let t = corner + vec2<f32>(0.5, 0.5);
    out.uv = vec2<f32>(mix(in.uv0.x, in.uv1.x, t.x), mix(in.uv0.y, in.uv1.y, 1.0 - t.y));
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let col = textureSample(tex, samp, in.uv) * in.color;
    return vec4<f32>(col.rgb * col.a, col.a);
}
