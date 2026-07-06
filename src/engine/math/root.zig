pub const vec = @import("vec.zig");
pub const mat = @import("mat.zig");
pub const quat = @import("quat.zig");
pub const ray = @import("ray.zig");
pub const frustum = @import("frustum.zig");

pub const Vec2 = vec.Vec2;
pub const Vec3 = vec.Vec3;
pub const Vec4 = vec.Vec4;
pub const Mat4 = mat.Mat4;
pub const Quat = quat.Quat;
pub const Ray = ray.Ray;
pub const Frustum = frustum.Frustum;

pub const pi = @import("std").math.pi;
pub const tau = @import("std").math.tau;

pub fn toRadians(degrees: f32) f32 {
    return degrees * (pi / 180.0);
}

pub fn toDegrees(radians: f32) f32 {
    return radians * (180.0 / pi);
}

pub fn clamp(v: f32, lo: f32, hi: f32) f32 {
    return @max(lo, @min(hi, v));
}

pub fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) f32 {
    const t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

test {
    _ = vec;
    _ = mat;
    _ = quat;
    _ = frustum;
}
