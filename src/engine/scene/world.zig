const std = @import("std");
const node_mod = @import("node.zig");
const entity_mod = @import("entity.zig");
const components_mod = @import("components.zig");
const transform_mod = @import("transform.zig");
const resources_mod = @import("../resources/root.zig");
const math = @import("../math/root.zig");

pub const InputState = struct {
    mouse_pos: math.Vec2 = .{},
    /// Delta since last frame; zeroed at the start of each frame.
    mouse_delta: math.Vec2 = .{},
    /// Scroll wheel delta; zeroed at the start of each frame.
    scroll_y: f32 = 0,
    left_down: bool = false,
    right_down: bool = false,
    middle_down: bool = false,
};

pub const SceneNode = node_mod.SceneNode;
pub const Entity = entity_mod.Entity;
pub const Transform = transform_mod.Transform;
pub const Component = components_mod.Component;

/// The game world. Owns all scene nodes and all loaded assets.
pub const World = struct {
    allocator: std.mem.Allocator,
    roots: std.ArrayListUnmanaged(*SceneNode) = .{ .items = &.{}, .capacity = 0 },
    entity_gen: entity_mod.EntityGen = .{},
    meshes: std.ArrayListUnmanaged(resources_mod.Mesh) = .{ .items = &.{}, .capacity = 0 },
    materials: std.ArrayListUnmanaged(resources_mod.Material) = .{ .items = &.{}, .capacity = 0 },
    // Recyclable mesh/material slots. Handles are append-only array indices (see addMesh/addMaterial),
    // so a removed asset must NOT orderedRemove — that shifts every later handle and corrupts live
    // references. Instead its slot is tombstoned (contents freed, replaced with an empty sentinel) and
    // its index parked here for the next addMesh/addMaterial to reuse. Keeps memory bounded across
    // repeated add/remove without ever invalidating an outstanding handle.
    free_meshes: std.ArrayListUnmanaged(components_mod.MeshHandle) = .{ .items = &.{}, .capacity = 0 },
    free_materials: std.ArrayListUnmanaged(components_mod.MaterialHandle) = .{ .items = &.{}, .capacity = 0 },
    active_camera: ?*SceneNode = null,
    active_camera_2d: ?*SceneNode = null,
    viewport_size: math.Vec2 = .{ .x = 1, .y = 1 },
    elapsed_seconds: f64 = 0,
    input: InputState = .{},

    pub fn init(allocator: std.mem.Allocator) World {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *World) void {
        for (self.roots.items) |root| {
            root.deinit(self.allocator);
            self.allocator.destroy(root);
        }
        self.roots.deinit(self.allocator);

        // Tombstoned slots (freed by freeMeshSlot/freeMaterialSlot) hold an empty sentinel, so their
        // deinit is a harmless no-op — no double free.
        for (self.meshes.items) |*mesh| {
            mesh.deinit(self.allocator);
        }
        self.meshes.deinit(self.allocator);
        self.free_meshes.deinit(self.allocator);

        for (self.materials.items) |*mat| {
            mat.deinit(self.allocator);
        }
        self.materials.deinit(self.allocator);
        self.free_materials.deinit(self.allocator);
    }

    /// Create a root-level scene node.
    pub fn createNode(self: *World, name: []const u8) !*SceneNode {
        const n = try self.allocator.create(SceneNode);
        n.* = .{
            .entity = self.entity_gen.create(),
            // Own the name: the caller's slice (e.g. a marshaled C# string) may not outlive this call.
            .name = try self.allocator.dupe(u8, name),
        };
        try self.roots.append(self.allocator, n);
        return n;
    }

    /// Remove and destroy a scene node (root or child) and its subtree.
    pub fn removeNode(self: *World, node: *SceneNode) void {
        // Detach from whichever list owns it: a child lives in parent.children, a root in self.roots.
        // Without the parent case, removing a nested node was a silent no-op — it leaked and left the
        // C# side holding a handle the renderer believed was gone.
        if (node.parent) |parent| {
            for (parent.children.items, 0..) |child, i| {
                if (child == node) {
                    _ = parent.children.orderedRemove(i);
                    break;
                }
            }
        } else {
            for (self.roots.items, 0..) |root, i| {
                if (root == node) {
                    _ = self.roots.orderedRemove(i);
                    break;
                }
            }
        }

        // A destroyed node must not remain the active camera — the render path dereferences these
        // every frame, so a stale pointer here is a use-after-free. Clear them if the removed subtree
        // owns the active camera.
        if (self.active_camera) |cam| {
            if (isSelfOrAncestor(node, cam)) self.active_camera = null;
        }
        if (self.active_camera_2d) |cam| {
            if (isSelfOrAncestor(node, cam)) self.active_camera_2d = null;
        }

        // deinit recursively frees the subtree; destroy frees the node itself.
        node.deinit(self.allocator);
        self.allocator.destroy(node);
    }

    /// True if `node` is `descendant` itself or an ancestor of it. Walks parent pointers up from
    /// `descendant` (O(depth)), so it needs no recursion over the subtree.
    fn isSelfOrAncestor(node: *SceneNode, descendant: *SceneNode) bool {
        var cur: ?*SceneNode = descendant;
        while (cur) |n| : (cur = n.parent) {
            if (n == node) return true;
        }
        return false;
    }

    /// Free a mesh's CPU contents and park its slot for reuse. Handles are array indices, so the slot
    /// is NOT removed (that would shift every later handle) — it is tombstoned with an empty mesh and
    /// its index pushed to the free list. The caller must also drop the renderer's matching GPU cache
    /// entry (MeshCache.invalidate): World owns only the CPU side.
    pub fn freeMeshSlot(self: *World, handle: components_mod.MeshHandle) void {
        if (handle >= self.meshes.items.len) return;
        self.meshes.items[handle].deinit(self.allocator);
        self.meshes.items[handle] = .{ .primitives = &[_]resources_mod.Primitive{} };
        self.free_meshes.append(self.allocator, handle) catch {};
    }

    /// Free a material's CPU pixel buffers and park its slot for reuse (see freeMeshSlot). The caller
    /// must also drop the renderer's cached MaterialGpu (textures/views/bind group) for this handle.
    pub fn freeMaterialSlot(self: *World, handle: components_mod.MaterialHandle) void {
        if (handle >= self.materials.items.len) return;
        self.materials.items[handle].deinit(self.allocator);
        self.materials.items[handle] = .{};
        self.free_materials.append(self.allocator, handle) catch {};
    }

    /// True if any live node still renders with `handle`. Checked before freeing a mesh on node
    /// removal so a mesh shared by a surviving node is kept (only the last reference frees it). Today
    /// each node owns a unique mesh, but this keeps removal correct if asset sharing is introduced.
    pub fn isMeshReferenced(self: *const World, handle: components_mod.MeshHandle) bool {
        for (self.roots.items) |root| {
            if (subtreeUsesMesh(root, handle)) return true;
        }
        return false;
    }

    fn subtreeUsesMesh(node: *SceneNode, handle: components_mod.MeshHandle) bool {
        if (node.getComponent(.mesh_renderer)) |c| {
            if (c.mesh_renderer.mesh == handle) return true;
        }
        for (node.children.items) |child| {
            if (subtreeUsesMesh(child, handle)) return true;
        }
        return false;
    }

    /// True if any live node still uses `handle` as its material (see isMeshReferenced).
    pub fn isMaterialReferenced(self: *const World, handle: components_mod.MaterialHandle) bool {
        for (self.roots.items) |root| {
            if (subtreeUsesMaterial(root, handle)) return true;
        }
        return false;
    }

    fn subtreeUsesMaterial(node: *SceneNode, handle: components_mod.MaterialHandle) bool {
        if (node.getComponent(.mesh_renderer)) |c| {
            if (c.mesh_renderer.material == handle) return true;
        }
        for (node.children.items) |child| {
            if (subtreeUsesMaterial(child, handle)) return true;
        }
        return false;
    }

    /// Create a child node under `parent`.
    pub fn createChild(self: *World, parent: *SceneNode, name: []const u8) !*SceneNode {
        const n = try self.allocator.create(SceneNode);
        n.* = .{
            .entity = self.entity_gen.create(),
            // Own the name: the caller's slice (e.g. a marshaled C# string) may not outlive this call.
            .name = try self.allocator.dupe(u8, name),
            .parent = parent,
        };
        try parent.children.append(self.allocator, n);
        return n;
    }

    /// Add a mesh asset and return its handle. Computes the mesh's local bounding sphere here so every
    /// source (primitives, .zmesh blobs, imported models) gets bounds for frustum culling.
    pub fn addMesh(self: *World, mesh: resources_mod.Mesh) !components_mod.MeshHandle {
        var m = mesh;
        m.computeBounds();
        // Reuse a tombstoned slot before growing the array (bounds slot growth across add/remove).
        // The slot was freed only when unreferenced, so overwriting it invalidates no live handle.
        if (self.free_meshes.pop()) |handle| {
            self.meshes.items[handle] = m;
            return handle;
        }
        const handle: components_mod.MeshHandle = @intCast(self.meshes.items.len);
        try self.meshes.append(self.allocator, m);
        return handle;
    }

    /// Add a material asset and return its handle.
    pub fn addMaterial(self: *World, mat: resources_mod.Material) !components_mod.MaterialHandle {
        if (self.free_materials.pop()) |handle| {
            self.materials.items[handle] = mat;
            return handle;
        }
        const handle: components_mod.MaterialHandle = @intCast(self.materials.items.len);
        try self.materials.append(self.allocator, mat);
        return handle;
    }

    pub fn getMesh(self: *const World, handle: components_mod.MeshHandle) ?*const resources_mod.Mesh {
        if (handle >= self.meshes.items.len) return null;
        return &self.meshes.items[handle];
    }

    pub fn getMaterial(self: *const World, handle: components_mod.MaterialHandle) ?*const resources_mod.Material {
        if (handle >= self.materials.items.len) return null;
        return &self.materials.items[handle];
    }

    /// Recompute all world transforms from root down. Call once per frame before rendering.
    pub fn updateTransforms(self: *World) void {
        for (self.roots.items) |root| {
            root.updateWorldTransform(false);
        }
    }

    pub fn update(self: *World, delta: f64) void {
        self.elapsed_seconds += delta;
        // Clear per-frame input deltas
        self.input.mouse_delta = .{};
        self.input.scroll_y = 0;
        self.updateTransforms();
    }

    /// Find the first node (depth-first) with the given name.
    pub fn findByName(self: *World, name: []const u8) ?*SceneNode {
        for (self.roots.items) |root| {
            if (searchByName(root, name)) |found| return found;
        }
        return null;
    }

    fn searchByName(node: *SceneNode, name: []const u8) ?*SceneNode {
        if (std.mem.eql(u8, node.name, name)) return node;
        for (node.children.items) |child| {
            if (searchByName(child, name)) |found| return found;
        }
        return null;
    }

    /// Collect all nodes that carry a MeshRenderer for a given render layer.
    pub fn collectRenderables(
        self: *const World,
        allocator: std.mem.Allocator,
        layer: components_mod.RenderLayer,
        list: *std.ArrayListUnmanaged(Renderable),
    ) !void {
        for (self.roots.items) |root| {
            try gatherRenderables(root, allocator, layer, list);
        }
    }

    fn gatherRenderables(
        node: *SceneNode,
        allocator: std.mem.Allocator,
        layer: components_mod.RenderLayer,
        list: *std.ArrayListUnmanaged(Renderable),
    ) !void {
        if (!node.active) return;
        if (node.getComponent(.mesh_renderer)) |comp| {
            const mr = &comp.mesh_renderer;
            if (mr.visible and mr.mesh != components_mod.null_mesh and mr.layer == layer) {
                try list.append(allocator, .{
                    .node = node,
                    .mesh_renderer = mr,
                });
            }
        }
        for (node.children.items) |child| {
            try gatherRenderables(child, allocator, layer, list);
        }
    }

    pub fn collectLights(
        self: *const World,
        allocator: std.mem.Allocator,
        list: *std.ArrayListUnmanaged(ActiveLight),
    ) !void {
        for (self.roots.items) |root| {
            try gatherLights(root, allocator, list);
        }
    }

    fn gatherLights(
        node: *SceneNode,
        allocator: std.mem.Allocator,
        list: *std.ArrayListUnmanaged(ActiveLight),
    ) !void {
        if (!node.active) return;
        if (node.getComponent(.light)) |comp| {
            try list.append(allocator, .{
                .node = node,
                .light = &comp.light,
            });
        }
        for (node.children.items) |child| {
            try gatherLights(child, allocator, list);
        }
    }
};

pub const ActiveLight = struct {
    node: *SceneNode,
    light: *components_mod.Light,
};

pub const Renderable = struct {
    node: *SceneNode,
    mesh_renderer: *components_mod.MeshRenderer,
};

test "World retains state" {
    const testing = std.testing;
    var world = World.init(testing.allocator);
    defer world.deinit();

    // Insert retained state
    const mesh_handle = try world.addMesh(.{ .primitives = &[_]resources_mod.Primitive{}, .name = "test" });
    const mat_handle = try world.addMaterial(resources_mod.Material{});
    const root = try world.createNode("root");
    try root.addComponent(testing.allocator, .{ .mesh_renderer = .{ .mesh = mesh_handle, .material = mat_handle } });

    // Verify it is retained
    try testing.expectEqual(@as(usize, 1), world.roots.items.len);
    try testing.expectEqualStrings("root", world.roots.items[0].name);

    const found = world.findByName("root");
    try testing.expect(found != null);
    try testing.expectEqual(root, found.?);

    var renderables: std.ArrayListUnmanaged(Renderable) = .empty;
    defer renderables.deinit(testing.allocator);
    try world.collectRenderables(testing.allocator, .world_3d, &renderables);
    try testing.expectEqual(@as(usize, 1), renderables.items.len);
}

test "World frees and recycles mesh/material slots on removal" {
    const testing = std.testing;
    var world = World.init(testing.allocator);
    defer world.deinit();

    // Two nodes, each owning a unique mesh + material (mirrors the FFI create path).
    const mesh_a = try world.addMesh(.{ .primitives = &[_]resources_mod.Primitive{}, .name = "a" });
    const mat_a = try world.addMaterial(resources_mod.Material{});
    const mesh_b = try world.addMesh(.{ .primitives = &[_]resources_mod.Primitive{}, .name = "b" });
    const mat_b = try world.addMaterial(resources_mod.Material{});

    const node_a = try world.createNode("a");
    try node_a.addComponent(testing.allocator, .{ .mesh_renderer = .{ .mesh = mesh_a, .material = mat_a } });
    const node_b = try world.createNode("b");
    try node_b.addComponent(testing.allocator, .{ .mesh_renderer = .{ .mesh = mesh_b, .material = mat_b } });

    try testing.expect(world.isMeshReferenced(mesh_a));
    try testing.expect(world.isMaterialReferenced(mat_a));

    // Destroy node_a, then free its now-unreferenced assets — the two halves of node removal.
    world.removeNode(node_a);
    try testing.expect(!world.isMeshReferenced(mesh_a));
    try testing.expect(!world.isMaterialReferenced(mat_a));
    world.freeMeshSlot(mesh_a);
    world.freeMaterialSlot(mat_a);

    // node_b's assets are untouched — freeing must never disturb a surviving reference.
    try testing.expect(world.isMeshReferenced(mesh_b));
    try testing.expect(world.isMaterialReferenced(mat_b));

    // The freed slots are recycled by the next add — no array growth, and the reused handle equals
    // the freed one (append-only indices are preserved, never shifted).
    const mesh_c = try world.addMesh(.{ .primitives = &[_]resources_mod.Primitive{}, .name = "c" });
    const mat_c = try world.addMaterial(resources_mod.Material{});
    try testing.expectEqual(mesh_a, mesh_c);
    try testing.expectEqual(mat_a, mat_c);
    try testing.expectEqual(@as(usize, 2), world.meshes.items.len);
    try testing.expectEqual(@as(usize, 2), world.materials.items.len);
    try testing.expectEqual(@as(usize, 0), world.free_meshes.items.len);
    try testing.expectEqual(@as(usize, 0), world.free_materials.items.len);
}

test "World clears active camera when its node is removed" {
    const testing = std.testing;
    var world = World.init(testing.allocator);
    defer world.deinit();

    const cam_root = try world.createNode("cam_root");
    const cam = try world.createChild(cam_root, "cam");
    world.active_camera = cam;

    // Removing an ancestor of the active camera must drop the dangling pointer, not leave it live.
    world.removeNode(cam_root);
    try testing.expect(world.active_camera == null);
}
