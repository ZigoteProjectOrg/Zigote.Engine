const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;
const Mat4 = mat.Mat4;

/// A view frustum as six world-space planes, extracted from a view-projection matrix
/// (Gribb–Hartmann) for zero-to-one clip depth (wgpu/Metal/D3D) — matching Mat4.perspectiveRhZo.
/// Plane normals point INTO the frustum: a point is inside when n·p + d >= 0 for all six.
/// Mirror of Zigote.Core.Math3D.Frustum (C#); keep the two in sync.
pub const Frustum = struct {
    // Planes stored SoA (nx/ny/nz/d as 6-lane vectors, one lane per plane) so intersectsSphere is a
    // single vectorized dot-and-compare instead of a 6-iteration scalar loop. Plane i is
    // (nx[i], ny[i], nz[i], d[i]), normalized (nx²+ny²+nz²=1), normal pointing INTO the frustum.
    nx: @Vector(6, f32),
    ny: @Vector(6, f32),
    nz: @Vector(6, f32),
    d: @Vector(6, f32),

    /// Extract the six planes from a view-projection (proj·view), column-major.
    pub fn fromViewProj(m: Mat4) Frustum {
        // Rows of the matrix (clip component r = row r · worldPos). cols[c].{x,y,z,w} = element (row, c).
        const r0 = Vec4{ .x = m.cols[0].x, .y = m.cols[1].x, .z = m.cols[2].x, .w = m.cols[3].x };
        const r1 = Vec4{ .x = m.cols[0].y, .y = m.cols[1].y, .z = m.cols[2].y, .w = m.cols[3].y };
        const r2 = Vec4{ .x = m.cols[0].z, .y = m.cols[1].z, .z = m.cols[2].z, .w = m.cols[3].z };
        const r3 = Vec4{ .x = m.cols[0].w, .y = m.cols[1].w, .z = m.cols[2].w, .w = m.cols[3].w };
        const p = [6]Vec4{
            normalizePlane(r3.add(r0)), // left
            normalizePlane(r3.sub(r0)), // right
            normalizePlane(r3.add(r1)), // bottom
            normalizePlane(r3.sub(r1)), // top
            normalizePlane(r2), // near
            normalizePlane(r3.sub(r2)), // far
        };
        return .{
            .nx = .{ p[0].x, p[1].x, p[2].x, p[3].x, p[4].x, p[5].x },
            .ny = .{ p[0].y, p[1].y, p[2].y, p[3].y, p[4].y, p[5].y },
            .nz = .{ p[0].z, p[1].z, p[2].z, p[3].z, p[4].z, p[5].z },
            .d = .{ p[0].w, p[1].w, p[2].w, p[3].w, p[4].w, p[5].w },
        };
    }

    /// True if the sphere is at least partially inside. Conservative: never hides a visible sphere.
    /// Per lane the signed distance is nx·cx + ny·cy + nz·cz + d (same arithmetic/order as the former
    /// scalar loop); the sphere is culled iff it lies fully outside ANY plane (dist < -radius there).
    pub fn intersectsSphere(self: Frustum, center: Vec3, radius: f32) bool {
        const cx: @Vector(6, f32) = @splat(center.x);
        const cy: @Vector(6, f32) = @splat(center.y);
        const cz: @Vector(6, f32) = @splat(center.z);
        const dist = self.nx * cx + self.ny * cy + self.nz * cz + self.d;
        const neg_r: @Vector(6, f32) = @splat(-radius);
        return !@reduce(.Or, dist < neg_r);
    }
};

fn normalizePlane(p: Vec4) Vec4 {
    const len = @sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    if (len < 1e-20) return p;
    const inv = 1.0 / len;
    return .{ .x = p.x * inv, .y = p.y * inv, .z = p.z * inv, .w = p.w * inv };
}

test "frustum keeps what's in front and culls what's behind / off-screen" {
    const fovy: f32 = 60.0 * (std.math.pi / 180.0);
    const proj = Mat4.perspectiveRhZo(fovy, 1.0, 0.1, 100.0);
    const view = Mat4.lookAt(
        .{ .x = 0, .y = 0, .z = 0 },
        .{ .x = 0, .y = 0, .z = -1 }, // look down -Z
        .{ .x = 0, .y = 1, .z = 0 },
    );
    const f = Frustum.fromViewProj(proj.mul(view));

    try std.testing.expect(f.intersectsSphere(.{ .x = 0, .y = 0, .z = -5 }, 1.0)); // in front
    try std.testing.expect(!f.intersectsSphere(.{ .x = 0, .y = 0, .z = 5 }, 1.0)); // behind camera
    try std.testing.expect(!f.intersectsSphere(.{ .x = 0, .y = 80, .z = -5 }, 1.0)); // far above view
    try std.testing.expect(!f.intersectsSphere(.{ .x = 0, .y = 0, .z = -500 }, 1.0)); // beyond far plane
    // A huge sphere straddling the camera is still considered visible (radius covers the frustum).
    try std.testing.expect(f.intersectsSphere(.{ .x = 0, .y = 0, .z = 5 }, 50.0));
}
