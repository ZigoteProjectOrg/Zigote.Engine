const std = @import("std");
const zm = @import("zmath");
const vec = @import("vec.zig");
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;

// ── zmath bridge ────────────────────────────────────────────────────────────────
// Zigote's Mat4 is COLUMN-major (cols[i] = i-th column); zmath's Mat is ROW-major
// ([4]F32x4, row i). Loading Zigote columns as zmath rows therefore yields the
// TRANSPOSE in zmath's representation, and storing zmath rows back into Zigote
// columns transposes again — so `store(load(m)) == m`. The single-operand ops
// (inverse/transpose) compose cleanly through this bridge; mul swaps its operands
// to compensate for the transpose (see `mul`). All correctness is pinned by the
// "matches scalar reference" test below.
inline fn loadCols(m: Mat4) zm.Mat {
    return .{
        zm.F32x4{ m.cols[0].x, m.cols[0].y, m.cols[0].z, m.cols[0].w },
        zm.F32x4{ m.cols[1].x, m.cols[1].y, m.cols[1].z, m.cols[1].w },
        zm.F32x4{ m.cols[2].x, m.cols[2].y, m.cols[2].z, m.cols[2].w },
        zm.F32x4{ m.cols[3].x, m.cols[3].y, m.cols[3].z, m.cols[3].w },
    };
}

inline fn storeCols(zmat: zm.Mat) Mat4 {
    return .{ .cols = .{
        .{ .x = zmat[0][0], .y = zmat[0][1], .z = zmat[0][2], .w = zmat[0][3] },
        .{ .x = zmat[1][0], .y = zmat[1][1], .z = zmat[1][2], .w = zmat[1][3] },
        .{ .x = zmat[2][0], .y = zmat[2][1], .z = zmat[2][2], .w = zmat[2][3] },
        .{ .x = zmat[3][0], .y = zmat[3][1], .z = zmat[3][2], .w = zmat[3][3] },
    } };
}

/// Column-major 4x4 matrix. cols[i] is the i-th column.
/// Matches wgpu/Metal/Vulkan shader layout (column_major storage).
pub const Mat4 = struct {
    cols: [4]Vec4,

    pub const identity = Mat4{ .cols = .{
        .{ .x = 1, .y = 0, .z = 0, .w = 0 },
        .{ .x = 0, .y = 1, .z = 0, .w = 0 },
        .{ .x = 0, .y = 0, .z = 1, .w = 0 },
        .{ .x = 0, .y = 0, .z = 0, .w = 1 },
    } };

    /// SIMD column-major multiply (result = A·B). With both operands loaded as their
    /// transposes (loadCols), `zm.mul(B^T, A^T) = (A·B)^T`, which storeCols transposes
    /// back to A·B in Zigote's column layout. 4 vector FMAs vs the old scalar triple loop.
    pub fn mul(a: Mat4, b: Mat4) Mat4 {
        return storeCols(zm.mul(loadCols(b), loadCols(a)));
    }

    /// M·v (column vector). `zm.mul(v, M^T)` computes the row-vector product v·Mᵀ,
    /// whose j-th component is Σᵢ M(j,i)·v(i) = (M·v)(j).
    pub fn mulVec4(m: Mat4, v: Vec4) Vec4 {
        const r = zm.mul(zm.F32x4{ v.x, v.y, v.z, v.w }, loadCols(m));
        return .{ .x = r[0], .y = r[1], .z = r[2], .w = r[3] };
    }

    pub fn mulPoint(m: Mat4, v: Vec3) Vec3 {
        const r = m.mulVec4(.{ .x = v.x, .y = v.y, .z = v.z, .w = 1.0 });
        return .{ .x = r.x, .y = r.y, .z = r.z };
    }

    pub fn mulDirection(m: Mat4, v: Vec3) Vec3 {
        const r = m.mulVec4(.{ .x = v.x, .y = v.y, .z = v.z, .w = 0.0 });
        return .{ .x = r.x, .y = r.y, .z = r.z };
    }

    /// Row r, column c (0-indexed).
    pub fn getElement(m: Mat4, c: usize, r: usize) f32 {
        const col: [4]f32 = .{ m.cols[c].x, m.cols[c].y, m.cols[c].z, m.cols[c].w };
        return col[r];
    }

    pub fn setElement(m: *Mat4, c: usize, r: usize, v: f32) void {
        switch (r) {
            0 => switch (c) {
                0 => m.cols[0].x = v,
                1 => m.cols[1].x = v,
                2 => m.cols[2].x = v,
                3 => m.cols[3].x = v,
                else => {},
            },
            1 => switch (c) {
                0 => m.cols[0].y = v,
                1 => m.cols[1].y = v,
                2 => m.cols[2].y = v,
                3 => m.cols[3].y = v,
                else => {},
            },
            2 => switch (c) {
                0 => m.cols[0].z = v,
                1 => m.cols[1].z = v,
                2 => m.cols[2].z = v,
                3 => m.cols[3].z = v,
                else => {},
            },
            3 => switch (c) {
                0 => m.cols[0].w = v,
                1 => m.cols[1].w = v,
                2 => m.cols[2].w = v,
                3 => m.cols[3].w = v,
                else => {},
            },
            else => {},
        }
    }

    /// Flat array in column-major order, suitable for uploading to GPU uniforms.
    pub fn toArray(m: Mat4) [16]f32 {
        return .{
            m.cols[0].x, m.cols[0].y, m.cols[0].z, m.cols[0].w,
            m.cols[1].x, m.cols[1].y, m.cols[1].z, m.cols[1].w,
            m.cols[2].x, m.cols[2].y, m.cols[2].z, m.cols[2].w,
            m.cols[3].x, m.cols[3].y, m.cols[3].z, m.cols[3].w,
        };
    }

    pub fn fromArray(a: [16]f32) Mat4 {
        return .{ .cols = .{
            .{ .x = a[0], .y = a[1], .z = a[2], .w = a[3] },
            .{ .x = a[4], .y = a[5], .z = a[6], .w = a[7] },
            .{ .x = a[8], .y = a[9], .z = a[10], .w = a[11] },
            .{ .x = a[12], .y = a[13], .z = a[14], .w = a[15] },
        } };
    }

    pub fn transpose(m: Mat4) Mat4 {
        return storeCols(zm.transpose(loadCols(m)));
    }

    pub fn translation(v: Vec3) Mat4 {
        var m = Mat4.identity;
        m.cols[3] = .{ .x = v.x, .y = v.y, .z = v.z, .w = 1.0 };
        return m;
    }

    pub fn scaling(v: Vec3) Mat4 {
        var m = Mat4.identity;
        m.cols[0].x = v.x;
        m.cols[1].y = v.y;
        m.cols[2].z = v.z;
        return m;
    }

    pub fn rotationX(angle: f32) Mat4 {
        const c = @cos(angle);
        const s = @sin(angle);
        var m = Mat4.identity;
        m.cols[1].y = c;
        m.cols[1].z = s;
        m.cols[2].y = -s;
        m.cols[2].z = c;
        return m;
    }

    pub fn rotationY(angle: f32) Mat4 {
        const c = @cos(angle);
        const s = @sin(angle);
        var m = Mat4.identity;
        m.cols[0].x = c;
        m.cols[0].z = -s;
        m.cols[2].x = s;
        m.cols[2].z = c;
        return m;
    }

    pub fn rotationZ(angle: f32) Mat4 {
        const c = @cos(angle);
        const s = @sin(angle);
        var m = Mat4.identity;
        m.cols[0].x = c;
        m.cols[0].y = s;
        m.cols[1].x = -s;
        m.cols[1].y = c;
        return m;
    }

    /// Right-handed perspective projection mapping near→z=0, far→z=1 (wgpu clip space).
    pub fn perspectiveRhZo(fovy_radians: f32, aspect: f32, near: f32, far: f32) Mat4 {
        const f = 1.0 / @tan(fovy_radians * 0.5);
        var m = Mat4{ .cols = .{ Vec4.zero, Vec4.zero, Vec4.zero, Vec4.zero } };
        m.cols[0].x = f / aspect;
        m.cols[1].y = f;
        m.cols[2].z = far / (near - far);
        m.cols[2].w = -1.0;
        m.cols[3].z = (near * far) / (near - far);
        return m;
    }

    /// Orthographic projection for UI/debug overlays, mapping near→0, far→1.
    pub fn orthographicRhZo(left: f32, right_val: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4 {
        var m = Mat4.identity;
        m.cols[0].x = 2.0 / (right_val - left);
        m.cols[1].y = 2.0 / (top - bottom);
        m.cols[2].z = 1.0 / (near - far);
        m.cols[3].x = -(right_val + left) / (right_val - left);
        m.cols[3].y = -(top + bottom) / (top - bottom);
        m.cols[3].z = near / (near - far);
        return m;
    }

    /// Look-at view matrix, right-handed.
    pub fn lookAt(eye: Vec3, center: Vec3, world_up: Vec3) Mat4 {
        const f = center.sub(eye).normalize(); // forward (-z in view space)
        const r = f.cross(world_up).normalize(); // right
        const u = r.cross(f); // up

        var m = Mat4.identity;
        m.cols[0].x = r.x;
        m.cols[1].x = r.y;
        m.cols[2].x = r.z;
        m.cols[0].y = u.x;
        m.cols[1].y = u.y;
        m.cols[2].y = u.z;
        m.cols[0].z = -f.x;
        m.cols[1].z = -f.y;
        m.cols[2].z = -f.z;
        m.cols[3].x = -r.dot(eye);
        m.cols[3].y = -u.dot(eye);
        m.cols[3].z = f.dot(eye);
        return m;
    }

    /// SIMD 4×4 inverse (zmath cofactor method). det(Mᵀ)=det(M) and (Mᵀ)⁻¹=(M⁻¹)ᵀ,
    /// so the loadCols/storeCols bridge round-trips correctly. Preserves the old
    /// contract of returning identity for a (near-)singular matrix.
    pub fn inverse(m: Mat4) Mat4 {
        var det: zm.F32x4 = undefined;
        const inv = zm.inverseDet(loadCols(m), &det);
        if (@abs(det[0]) < std.math.floatEps(f32)) return Mat4.identity;
        return storeCols(inv);
    }
};

test "Mat4 identity mul" {
    const a = Mat4.identity;
    const b = Mat4.identity;
    const c = a.mul(b);
    try std.testing.expectApproxEqAbs(c.cols[0].x, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(c.cols[1].y, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(c.cols[0].y, 0.0, 1e-6);
}

test "Mat4 toArray roundtrip" {
    const m = Mat4.translation(.{ .x = 1, .y = 2, .z = 3 });
    const arr = m.toArray();
    const m2 = Mat4.fromArray(arr);
    try std.testing.expectApproxEqAbs(m2.cols[3].x, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(m2.cols[3].y, 2.0, 1e-6);
    try std.testing.expectApproxEqAbs(m2.cols[3].z, 3.0, 1e-6);
}

// Independent scalar reference, mirroring the pre-zmath implementations. Pins the
// SIMD/zmath column-major bridge against accidental transpose/convention errors.
fn refMul(a: Mat4, b: Mat4) Mat4 {
    var result: Mat4 = undefined;
    for (0..4) |c| {
        for (0..4) |r| {
            var sum: f32 = 0;
            for (0..4) |k| sum += a.getElement(k, r) * b.getElement(c, k);
            result.setElement(c, r, sum);
        }
    }
    return result;
}

fn refMulVec4(m: Mat4, v: Vec4) Vec4 {
    return .{
        .x = m.cols[0].x * v.x + m.cols[1].x * v.y + m.cols[2].x * v.z + m.cols[3].x * v.w,
        .y = m.cols[0].y * v.x + m.cols[1].y * v.y + m.cols[2].y * v.z + m.cols[3].y * v.w,
        .z = m.cols[0].z * v.x + m.cols[1].z * v.y + m.cols[2].z * v.z + m.cols[3].z * v.w,
        .w = m.cols[0].w * v.x + m.cols[1].w * v.y + m.cols[2].w * v.z + m.cols[3].w * v.w,
    };
}

test "Mat4 zmath mul/mulVec4 match scalar reference" {
    var rng = std.Random.DefaultPrng.init(0xC0FFEE);
    const r = rng.random();
    for (0..64) |_| {
        var a: Mat4 = undefined;
        var b: Mat4 = undefined;
        for (0..4) |c| {
            a.cols[c] = .{ .x = r.float(f32) * 4 - 2, .y = r.float(f32) * 4 - 2, .z = r.float(f32) * 4 - 2, .w = r.float(f32) * 4 - 2 };
            b.cols[c] = .{ .x = r.float(f32) * 4 - 2, .y = r.float(f32) * 4 - 2, .z = r.float(f32) * 4 - 2, .w = r.float(f32) * 4 - 2 };
        }
        const got = a.mul(b);
        const want = refMul(a, b);
        for (0..4) |c| {
            try std.testing.expectApproxEqAbs(want.cols[c].x, got.cols[c].x, 1e-4);
            try std.testing.expectApproxEqAbs(want.cols[c].y, got.cols[c].y, 1e-4);
            try std.testing.expectApproxEqAbs(want.cols[c].z, got.cols[c].z, 1e-4);
            try std.testing.expectApproxEqAbs(want.cols[c].w, got.cols[c].w, 1e-4);
        }
        const v = Vec4{ .x = r.float(f32), .y = r.float(f32), .z = r.float(f32), .w = r.float(f32) };
        const gv = a.mulVec4(v);
        const wv = refMulVec4(a, v);
        try std.testing.expectApproxEqAbs(wv.x, gv.x, 1e-4);
        try std.testing.expectApproxEqAbs(wv.y, gv.y, 1e-4);
        try std.testing.expectApproxEqAbs(wv.z, gv.z, 1e-4);
        try std.testing.expectApproxEqAbs(wv.w, gv.w, 1e-4);
    }
}

test "Mat4 zmath inverse round-trips to identity" {
    const m = Mat4.translation(.{ .x = 3, .y = -2, .z = 5 })
        .mul(Mat4.rotationY(0.7))
        .mul(Mat4.rotationX(-0.3))
        .mul(Mat4.scaling(.{ .x = 2, .y = 0.5, .z = 1.5 }));
    const prod = m.mul(m.inverse());
    for (0..4) |c| {
        for (0..4) |rr| {
            const expected: f32 = if (c == rr) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, prod.getElement(c, rr), 1e-4);
        }
    }
}
