const std = @import("std");

pub const Vec2 = struct {
    x: f32 = 0,
    y: f32 = 0,

    pub const zero = Vec2{};
    pub const one = Vec2{ .x = 1, .y = 1 };
    pub const up = Vec2{ .y = 1 };
    pub const right = Vec2{ .x = 1 };

    pub fn splat(v: f32) Vec2 {
        return .{ .x = v, .y = v };
    }

    pub fn add(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }

    pub fn sub(a: Vec2, b: Vec2) Vec2 {
        return .{ .x = a.x - b.x, .y = a.y - b.y };
    }

    pub fn scale(v: Vec2, s: f32) Vec2 {
        return .{ .x = v.x * s, .y = v.y * s };
    }

    pub fn neg(v: Vec2) Vec2 {
        return .{ .x = -v.x, .y = -v.y };
    }

    pub fn dot(a: Vec2, b: Vec2) f32 {
        return a.x * b.x + a.y * b.y;
    }

    pub fn lengthSq(v: Vec2) f32 {
        return v.x * v.x + v.y * v.y;
    }

    pub fn length(v: Vec2) f32 {
        return @sqrt(v.lengthSq());
    }

    pub fn normalize(v: Vec2) Vec2 {
        const len = v.length();
        if (len < std.math.floatEps(f32)) return Vec2.zero;
        return v.scale(1.0 / len);
    }

    pub fn lerp(a: Vec2, b: Vec2, t: f32) Vec2 {
        return .{ .x = a.x + (b.x - a.x) * t, .y = a.y + (b.y - a.y) * t };
    }

    pub fn toArray(v: Vec2) [2]f32 {
        return .{ v.x, v.y };
    }
};

pub const Vec3 = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,

    pub const zero = Vec3{};
    pub const one = Vec3{ .x = 1, .y = 1, .z = 1 };
    pub const up = Vec3{ .y = 1 };
    pub const down = Vec3{ .y = -1 };
    pub const right = Vec3{ .x = 1 };
    pub const left = Vec3{ .x = -1 };
    pub const forward = Vec3{ .z = -1 };
    pub const back = Vec3{ .z = 1 };

    pub fn splat(v: f32) Vec3 {
        return .{ .x = v, .y = v, .z = v };
    }

    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    pub fn scale(v: Vec3, s: f32) Vec3 {
        return .{ .x = v.x * s, .y = v.y * s, .z = v.z * s };
    }

    pub fn mul(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x * b.x, .y = a.y * b.y, .z = a.z * b.z };
    }

    pub fn neg(v: Vec3) Vec3 {
        return .{ .x = -v.x, .y = -v.y, .z = -v.z };
    }

    pub fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub fn lengthSq(v: Vec3) f32 {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    pub fn length(v: Vec3) f32 {
        return @sqrt(v.lengthSq());
    }

    pub fn normalize(v: Vec3) Vec3 {
        const len = v.length();
        if (len < std.math.floatEps(f32)) return Vec3.zero;
        return v.scale(1.0 / len);
    }

    pub fn lerp(a: Vec3, b: Vec3, t: f32) Vec3 {
        return .{
            .x = a.x + (b.x - a.x) * t,
            .y = a.y + (b.y - a.y) * t,
            .z = a.z + (b.z - a.z) * t,
        };
    }

    pub fn toVec4(v: Vec3, w: f32) Vec4 {
        return .{ .x = v.x, .y = v.y, .z = v.z, .w = w };
    }

    pub fn toArray(v: Vec3) [3]f32 {
        return .{ v.x, v.y, v.z };
    }

    pub fn distance(a: Vec3, b: Vec3) f32 {
        return b.sub(a).length();
    }

    pub fn reflect(v: Vec3, n: Vec3) Vec3 {
        return v.sub(n.scale(2.0 * dot(v, n)));
    }
};

pub const Vec4 = struct {
    x: f32 = 0,
    y: f32 = 0,
    z: f32 = 0,
    w: f32 = 0,

    pub const zero = Vec4{};
    pub const one = Vec4{ .x = 1, .y = 1, .z = 1, .w = 1 };

    pub fn splat(v: f32) Vec4 {
        return .{ .x = v, .y = v, .z = v, .w = v };
    }

    pub fn add(a: Vec4, b: Vec4) Vec4 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z, .w = a.w + b.w };
    }

    pub fn sub(a: Vec4, b: Vec4) Vec4 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z, .w = a.w - b.w };
    }

    pub fn scale(v: Vec4, s: f32) Vec4 {
        return .{ .x = v.x * s, .y = v.y * s, .z = v.z * s, .w = v.w * s };
    }

    pub fn dot(a: Vec4, b: Vec4) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    pub fn xyz(v: Vec4) Vec3 {
        return .{ .x = v.x, .y = v.y, .z = v.z };
    }

    pub fn toArray(v: Vec4) [4]f32 {
        return .{ v.x, v.y, v.z, v.w };
    }
};
