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
    // shape.x = corner radius, shape.y = border width — both in the sprite's local units (the same
    // space as `size`). radius 0 + border 0 is a plain quad, which is the common case.
    @location(6) shape: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Position within the quad in local units, for the fragment-side shape SDF.
    @location(2) local: vec2<f32>,
    @location(3) half_size: vec2<f32>,
    @location(4) shape: vec2<f32>,
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
    out.local = local;
    out.half_size = abs(in.size) * 0.5;
    out.shape = in.shape;
    return out;
}

/// Signed distance to a rounded box centred at the origin; negative inside.
fn sd_round_box(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(radius, radius);
    return length(max(q, vec2<f32>(0.0, 0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    var col = textureSample(tex, samp, in.uv) * in.color;

    let radius = in.shape.x;
    let border = in.shape.y;
    if (radius > 0.0 || border > 0.0) {
        // A radius past half the shorter side would invert the SDF; clamping degrades it to a
        // capsule/circle instead, which is how a circle is expressed (radius >= half the side).
        let r = clamp(radius, 0.0, min(in.half_size.x, in.half_size.y));
        let d = sd_round_box(in.local, in.half_size, r);

        // Antialias across one pixel, using the screen-space derivative of the distance so the
        // edge stays one pixel wide at any zoom.
        let aa = max(fwidth(d), 1e-6);
        var mask = 1.0 - smoothstep(-aa, aa, d);

        // A border keeps only the ring: inside the outer edge AND outside the inner edge.
        if (border > 0.0) {
            mask = mask * smoothstep(-aa, aa, d + border);
        }

        col = vec4<f32>(col.rgb, col.a * mask);
    }

    return vec4<f32>(col.rgb * col.a, col.a);
}
