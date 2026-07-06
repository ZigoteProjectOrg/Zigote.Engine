const math = @import("../math/root.zig");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mat4 = math.Mat4;

pub const Transform = struct {
    position: Vec3 = Vec3.zero,
    rotation: Quat = Quat.identity,
    scale: Vec3 = Vec3.one,

    pub fn toMat4(t: Transform) Mat4 {
        const s = Mat4.scaling(t.scale);
        const r = t.rotation.toMat4();
        const p = Mat4.translation(t.position);
        return p.mul(r).mul(s);
    }

    pub fn lerp(a: Transform, b: Transform, t: f32) Transform {
        return .{
            .position = Vec3.lerp(a.position, b.position, t),
            .rotation = Quat.slerp(a.rotation, b.rotation, t),
            .scale = Vec3.lerp(a.scale, b.scale, t),
        };
    }

    pub fn combine(parent: Transform, child: Transform) Transform {
        return .{
            .position = parent.position.add(parent.rotation.rotateVec(child.position.mul(parent.scale))),
            .rotation = parent.rotation.mul(child.rotation),
            .scale = parent.scale.mul(child.scale),
        };
    }
};
