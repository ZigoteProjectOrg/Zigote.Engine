const vec = @import("vec.zig");
const Vec3 = vec.Vec3;

pub const Ray = struct {
    origin: Vec3,
    direction: Vec3, // should be normalized

    pub fn at(ray: Ray, t: f32) Vec3 {
        return ray.origin.add(ray.direction.scale(t));
    }

    /// Möller–Trumbore intersection test. Returns t > 0 if hit, else null.
    pub fn intersectTriangle(ray: Ray, v0: Vec3, v1: Vec3, v2: Vec3) ?f32 {
        const eps = 1e-7;
        const edge1 = v1.sub(v0);
        const edge2 = v2.sub(v0);
        const h = ray.direction.cross(edge2);
        const a = Vec3.dot(edge1, h);
        if (a > -eps and a < eps) return null;
        const f = 1.0 / a;
        const s = ray.origin.sub(v0);
        const u = f * Vec3.dot(s, h);
        if (u < 0.0 or u > 1.0) return null;
        const q = s.cross(edge1);
        const v = f * Vec3.dot(ray.direction, q);
        if (v < 0.0 or u + v > 1.0) return null;
        const t = f * Vec3.dot(edge2, q);
        if (t < eps) return null;
        return t;
    }

    /// AABB slab intersection. Returns (tmin, tmax) or null if miss.
    pub fn intersectAabb(ray: Ray, aabb_min: Vec3, aabb_max: Vec3) ?[2]f32 {
        const inv_dir = Vec3{
            .x = 1.0 / ray.direction.x,
            .y = 1.0 / ray.direction.y,
            .z = 1.0 / ray.direction.z,
        };
        const t1 = (aabb_min.x - ray.origin.x) * inv_dir.x;
        const t2 = (aabb_max.x - ray.origin.x) * inv_dir.x;
        const t3 = (aabb_min.y - ray.origin.y) * inv_dir.y;
        const t4 = (aabb_max.y - ray.origin.y) * inv_dir.y;
        const t5 = (aabb_min.z - ray.origin.z) * inv_dir.z;
        const t6 = (aabb_max.z - ray.origin.z) * inv_dir.z;
        const tmin = @max(@max(@min(t1, t2), @min(t3, t4)), @min(t5, t6));
        const tmax = @min(@min(@max(t1, t2), @max(t3, t4)), @max(t5, t6));
        if (tmax < 0 or tmin > tmax) return null;
        return .{ tmin, tmax };
    }
};
