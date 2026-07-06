pub const entity = @import("entity.zig");
pub const transform = @import("transform.zig");
pub const components = @import("components.zig");
pub const node = @import("node.zig");
pub const world = @import("world.zig");

pub const Entity = entity.Entity;
pub const null_entity = entity.null_entity;
pub const Transform = transform.Transform;
pub const Component = components.Component;
pub const Camera = components.Camera;
pub const MeshRenderer = components.MeshRenderer;
pub const RenderLayer = components.RenderLayer;
pub const Light = components.Light;
pub const RigidBody = components.RigidBody;
pub const MeshHandle = components.MeshHandle;
pub const MaterialHandle = components.MaterialHandle;
pub const null_mesh = components.null_mesh;
pub const null_material = components.null_material;
pub const SceneNode = node.SceneNode;
pub const World = world.World;
pub const Renderable = world.Renderable;
pub const ActiveLight = world.ActiveLight;
pub const InputState = world.InputState;

// Pull the submodules' `test` blocks into the engine test binary. Without this, referencing the
// `scene` namespace from engine/root.zig only includes THIS file's tests — the world/node/etc. tests
// (e.g. "World retains state") were silently never compiled or run under `zig build test`.
test {
    _ = entity;
    _ = transform;
    _ = components;
    _ = node;
    _ = world;
}
