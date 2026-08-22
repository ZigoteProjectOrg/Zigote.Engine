const std = @import("std");
const entity_mod = @import("entity.zig");
const transform_mod = @import("transform.zig");
const components_mod = @import("components.zig");
const math = @import("../math/root.zig");

pub const Entity = entity_mod.Entity;
pub const Transform = transform_mod.Transform;
pub const Component = components_mod.Component;

/// A node in the scene graph. Owns its children list.
/// Application code creates nodes via World.createNode / World.createChild.
pub const SceneNode = struct {
    entity: Entity,
    name: []const u8,
    local_transform: Transform = .{},
    world_transform: Transform = .{}, // recomputed when dirty
    // Derived GPU matrices, cached alongside world_transform and refreshed only when the node
    // (or an ancestor) is dirty — so static nodes pay the toMat4 + inverse-transpose cost once,
    // not 2-3× per frame across the shadow/geometry/light passes that read them.
    world_matrix: math.Mat4 = math.Mat4.identity,
    normal_matrix: math.Mat4 = math.Mat4.identity,
    dirty_transform: bool = true,
    parent: ?*SceneNode = null,
    children: std.ArrayListUnmanaged(*SceneNode) = .{ .items = &.{}, .capacity = 0 },
    components: std.ArrayListUnmanaged(Component) = .{ .items = &.{}, .capacity = 0 },
    active: bool = true,
    /// The FFI handle this node was issued, or 0 if it was never handed across the C ABI.
    ///
    /// The handle used to BE `@intFromPtr(node)`, so anything needing one could derive it. It is a
    /// generational table index now (see EngineState.nodes), which is not derivable from the
    /// pointer — so the node carries it. Two places need the reverse direction: subtree removal,
    /// which must untrack every descendant, and the renderer's selection highlight, which compares
    /// the host's selected handle against the node it is drawing.
    ffi_handle: u64 = 0,

    pub fn deinit(self: *SceneNode, allocator: std.mem.Allocator) void {
        for (self.children.items) |child| {
            child.deinit(allocator);
            allocator.destroy(child);
        }
        self.children.deinit(allocator);
        self.components.deinit(allocator);
        allocator.free(self.name);
    }

    pub fn addComponent(self: *SceneNode, allocator: std.mem.Allocator, component: Component) !void {
        // One component per tag: getComponent returns the FIRST match, so a duplicate tag would be
        // silently shadowed (and its assets leaked). Callers must update the existing entry instead.
        if (std.debug.runtime_safety) {
            for (self.components.items) |c| {
                std.debug.assert(std.meta.activeTag(c) != std.meta.activeTag(component));
            }
        }
        try self.components.append(allocator, component);
    }

    /// Returns a pointer to the Component union entry whose active tag matches.
    /// Caller accesses the payload via e.g. `&comp.camera` or `&comp.mesh_renderer`.
    pub fn getComponent(self: *SceneNode, comptime tag: std.meta.Tag(Component)) ?*Component {
        for (self.components.items) |*c| {
            if (c.* == tag) return c;
        }
        return null;
    }

    pub fn removeComponent(self: *SceneNode, comptime tag: std.meta.Tag(Component)) bool {
        for (self.components.items, 0..) |c, i| {
            if (c == tag) {
                _ = self.components.swapRemove(i);
                return true;
            }
        }
        return false;
    }

    pub fn updateWorldTransform(self: *SceneNode, parent_dirty: bool) void {
        const is_dirty = self.dirty_transform or parent_dirty;
        if (is_dirty) {
            if (self.parent) |p| {
                self.world_transform = transform_mod.Transform.combine(p.world_transform, self.local_transform);
            } else {
                self.world_transform = self.local_transform;
            }
            // Refresh the cached GPU matrices once, here, instead of recomputing them per draw.
            // (inverse() returns identity for a singular matrix — e.g. a zero-scale hidden node —
            // matching the previous per-draw behaviour exactly, so output stays bit-for-bit equal.)
            self.world_matrix = self.world_transform.toMat4();
            self.normal_matrix = self.world_matrix.inverse().transpose();
            self.dirty_transform = false;
        }
        for (self.children.items) |child| {
            child.updateWorldTransform(is_dirty);
        }
    }

    pub fn worldMatrix(self: *const SceneNode) math.Mat4 {
        return self.world_matrix;
    }

    pub fn normalMatrix(self: *const SceneNode) math.Mat4 {
        return self.normal_matrix;
    }
};
