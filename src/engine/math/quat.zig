const std = @import("std");
const vec = @import("vec.zig");
const mat = @import("mat.zig");
const Vec3 = vec.Vec3;
const Mat4 = mat.Mat4;

/// Unit quaternion representing a rotation. Stored as (x, y, z, w) where w is the scalar part.
pub const Quat = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 1,

    pub const identity = Quat{};

    pub fn fromAxisAngle(axis: Vec3, angle_radians: f32) Quat {
        const half = angle_radians * 0.5;
        const s = @sin(half);
        const n = axis.normalize();
        return .{ .x = n.x * s, .y = n.y * s, .z = n.z * s, .w = @cos(half) };
    }

    pub fn fromEuler(pitch: f32, yaw: f32, roll: f32) Quat {
        const qp = Quat.fromAxisAngle(.{ .x = 1, .y = 0, .z = 0 }, pitch);
        const qy = Quat.fromAxisAngle(.{ .x = 0, .y = 1, .z = 0 }, yaw);
        const qr = Quat.fromAxisAngle(.{ .x = 0, .y = 0, .z = 1 }, roll);
        return qy.mul(qp).mul(qr);
    }

    pub fn mul(a: Quat, b: Quat) Quat {
        return .{
            .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        };
    }

    pub fn normalize(q: Quat) Quat {
        const len = @sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        if (len < std.math.floatEps(f32)) return Quat.identity;
        const inv = 1.0 / len;
        return .{ .x = q.x * inv, .y = q.y * inv, .z = q.z * inv, .w = q.w * inv };
    }

    pub fn conjugate(q: Quat) Quat {
        return .{ .x = -q.x, .y = -q.y, .z = -q.z, .w = q.w };
    }

    pub fn inverse(q: Quat) Quat {
        return q.conjugate().normalize();
    }

    pub fn rotateVec(q: Quat, v: Vec3) Vec3 {
        const qv = Vec3{ .x = q.x, .y = q.y, .z = q.z };
        const uv = qv.cross(v);
        const uuv = qv.cross(uv);
        return v.add(uv.scale(2.0 * q.w)).add(uuv.scale(2.0));
    }

    pub fn slerp(a: Quat, b_in: Quat, t: f32) Quat {
        var b = b_in;
        var dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;

        if (dot < 0.0) {
            b = .{ .x = -b.x, .y = -b.y, .z = -b.z, .w = -b.w };
            dot = -dot;
        }

        if (dot > 0.9995) {
            const lerped = Quat{
                .x = a.x + (b.x - a.x) * t,
                .y = a.y + (b.y - a.y) * t,
                .z = a.z + (b.z - a.z) * t,
                .w = a.w + (b.w - a.w) * t,
            };
            return lerped.normalize();
        }

        const theta0 = std.math.acos(dot);
        const theta = theta0 * t;
        const sin_theta = @sin(theta);
        const sin_theta0 = @sin(theta0);
        const s0 = @cos(theta) - dot * sin_theta / sin_theta0;
        const s1 = sin_theta / sin_theta0;

        return .{
            .x = s0 * a.x + s1 * b.x,
            .y = s0 * a.y + s1 * b.y,
            .z = s0 * a.z + s1 * b.z,
            .w = s0 * a.w + s1 * b.w,
        };
    }

    pub fn toMat4(q: Quat) Mat4 {
        const n = q.normalize();
        const xx = n.x * n.x;
        const yy = n.y * n.y;
        const zz = n.z * n.z;
        const xy = n.x * n.y;
        const xz = n.x * n.z;
        const yz = n.y * n.z;
        const wx = n.w * n.x;
        const wy = n.w * n.y;
        const wz = n.w * n.z;

        return Mat4{ .cols = .{
            .{ .x = 1 - 2 * (yy + zz), .y = 2 * (xy + wz), .z = 2 * (xz - wy), .w = 0 },
            .{ .x = 2 * (xy - wz), .y = 1 - 2 * (xx + zz), .z = 2 * (yz + wx), .w = 0 },
            .{ .x = 2 * (xz + wy), .y = 2 * (yz - wx), .z = 1 - 2 * (xx + yy), .w = 0 },
            .{ .x = 0, .y = 0, .z = 0, .w = 1 },
        } };
    }

    /// Build quaternion from a rotation matrix (upper-left 3x3 of a Mat4).
    pub fn fromMat4(m: Mat4) Quat {
        const trace = m.cols[0].x + m.cols[1].y + m.cols[2].z;
        if (trace > 0) {
            const s = 0.5 / @sqrt(trace + 1.0);
            return .{
                .w = 0.25 / s,
                .x = (m.cols[1].z - m.cols[2].y) * s,
                .y = (m.cols[2].x - m.cols[0].z) * s,
                .z = (m.cols[0].y - m.cols[1].x) * s,
            };
        } else if (m.cols[0].x > m.cols[1].y and m.cols[0].x > m.cols[2].z) {
            const s = 2.0 * @sqrt(1.0 + m.cols[0].x - m.cols[1].y - m.cols[2].z);
            return .{
                .w = (m.cols[1].z - m.cols[2].y) / s,
                .x = 0.25 * s,
                .y = (m.cols[1].x + m.cols[0].y) / s,
                .z = (m.cols[2].x + m.cols[0].z) / s,
            };
        } else if (m.cols[1].y > m.cols[2].z) {
            const s = 2.0 * @sqrt(1.0 + m.cols[1].y - m.cols[0].x - m.cols[2].z);
            return .{
                .w = (m.cols[2].x - m.cols[0].z) / s,
                .x = (m.cols[1].x + m.cols[0].y) / s,
                .y = 0.25 * s,
                .z = (m.cols[2].y + m.cols[1].z) / s,
            };
        } else {
            const s = 2.0 * @sqrt(1.0 + m.cols[2].z - m.cols[0].x - m.cols[1].y);
            return .{
                .w = (m.cols[0].y - m.cols[1].x) / s,
                .x = (m.cols[2].x + m.cols[0].z) / s,
                .y = (m.cols[2].y + m.cols[1].z) / s,
                .z = 0.25 * s,
            };
        }
    }

    pub fn toArray(q: Quat) [4]f32 {
        return .{ q.x, q.y, q.z, q.w };
    }
};

test "Quat identity rotation" {
    const v = Vec3{ .x = 1, .y = 0, .z = 0 };
    const r = Quat.identity.rotateVec(v);
    try std.testing.expectApproxEqAbs(r.x, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(r.y, 0.0, 1e-6);
}

test "Quat 90 degree rotation around Y" {
    const q = Quat.fromAxisAngle(.{ .x = 0, .y = 1, .z = 0 }, std.math.pi / 2.0);
    const v = Vec3{ .x = 1, .y = 0, .z = 0 };
    const r = q.rotateVec(v);
    try std.testing.expectApproxEqAbs(r.x, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(r.z, -1.0, 1e-5);
}
