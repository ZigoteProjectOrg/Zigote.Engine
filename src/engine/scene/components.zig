const math = @import("../math/root.zig");
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;
const Vec2 = math.Vec2;

/// Handle into World.meshes. u32 max = unset.
pub const MeshHandle = u32;
pub const null_mesh: MeshHandle = std.math.maxInt(MeshHandle);

/// Handle into World.materials.
pub const MaterialHandle = u32;
pub const null_material: MaterialHandle = std.math.maxInt(MaterialHandle);

pub const RenderLayer = enum { world_3d, scene2d };

const std = @import("std");

pub const ProjectionKind = enum { perspective, orthographic };

pub const Camera = struct {
    fovy_degrees: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 1000.0,
    kind: ProjectionKind = .perspective,
    orthographic_size: Vec2 = .{ .x = 2.0, .y = 2.0 },

    pub fn projMatrix(cam: Camera, aspect: f32) math.Mat4 {
        return switch (cam.kind) {
            .perspective => math.Mat4.perspectiveRhZo(
                math.toRadians(cam.fovy_degrees),
                aspect,
                cam.near,
                cam.far,
            ),
            .orthographic => math.Mat4.orthographicRhZo(
                -cam.orthographic_size.x * 0.5,
                cam.orthographic_size.x * 0.5,
                -cam.orthographic_size.y * 0.5,
                cam.orthographic_size.y * 0.5,
                cam.near,
                cam.far,
            ),
        };
    }
};

pub const MeshRenderer = struct {
    mesh: MeshHandle = null_mesh,
    material: MaterialHandle = null_material,
    visible: bool = true,
    layer: RenderLayer = .world_3d,
};

pub const LightKind = enum { directional, point, spot };

pub const Light = struct {
    kind: LightKind = .directional,
    color: Vec3 = .{ .x = 1, .y = 1, .z = 1 },
    intensity: f32 = 1.0,
    range: f32 = 10.0, // for point/spot
    inner_angle: f32 = 0.3, // for spot (radians)
    outer_angle: f32 = 0.5, // for spot (radians)
    cast_shadows: bool = false,
};

pub const RigidBody = struct {
    velocity: Vec3 = Vec3.zero,
    angular_velocity: Vec3 = Vec3.zero,
    mass: f32 = 1.0,
    is_static: bool = false,
    use_gravity: bool = true,
};

pub const Component = union(enum) {
    camera: Camera,
    mesh_renderer: MeshRenderer,
    light: Light,
    rigid_body: RigidBody,
};
