/// JoltPhysics C-ABI wrapper for the Zigote FFI layer.
/// Exposes a simple rigid-body physics world to C#.
///
/// Layer scheme:
///   ObjectLayer 0 = NON_MOVING  (static/immovable bodies)
///   ObjectLayer 1 = MOVING      (dynamic/kinematic bodies)
///
/// Broad-phase mirrors object layers 1-to-1:
///   BPLayer 0 = NON_MOVING
///   BPLayer 1 = MOVING
const std = @import("std");

const joltc = @cImport({
    @cInclude("JoltPhysicsC.h");
});

pub const INVALID_BODY_ID: u32 = 0xFFFF_FFFF;

const LAYER_NON_MOVING: u16 = 0;
const LAYER_MOVING: u16 = 1;
const BP_LAYER_NON_MOVING: u8 = 0;
const BP_LAYER_MOVING: u8 = 1;

// ── Vtable callback implementations ───────────────────────────────────────────

fn bpGetNumLayers(_: ?*const anyopaque) callconv(.c) u32 {
    return 2;
}

fn bpGetLayer(_: ?*const anyopaque, layer: joltc.JPC_ObjectLayer) callconv(.c) joltc.JPC_BroadPhaseLayer {
    return @intCast(layer); // 0 → 0, 1 → 1
}

fn bpFilterShouldCollide(
    _: ?*const anyopaque,
    layer1: joltc.JPC_ObjectLayer,
    layer2: joltc.JPC_BroadPhaseLayer,
) callconv(.c) bool {
    if (layer1 == LAYER_NON_MOVING) return layer2 == BP_LAYER_MOVING;
    return true;
}

fn objFilterShouldCollide(
    _: ?*const anyopaque,
    layer1: joltc.JPC_ObjectLayer,
    layer2: joltc.JPC_ObjectLayer,
) callconv(.c) bool {
    // static vs static never collide
    if (layer1 == LAYER_NON_MOVING and layer2 == LAYER_NON_MOVING) return false;
    return true;
}

// ── Interface impls (vtable pointer is the first field) ───────────────────────

const BPLayerInterfaceImpl = extern struct {
    vtable: ?*const joltc.JPC_BroadPhaseLayerInterfaceVTable,
};

const BPFilterImpl = extern struct {
    vtable: ?*const joltc.JPC_ObjectVsBroadPhaseLayerFilterVTable,
};

const ObjFilterImpl = extern struct {
    vtable: ?*const joltc.JPC_ObjectLayerPairFilterVTable,
};

// ── Module-level vtables (stable addresses in shared lib) ─────────────────────

var g_bp_vtable: joltc.JPC_BroadPhaseLayerInterfaceVTable = undefined;
var g_bp_filter_vtable: joltc.JPC_ObjectVsBroadPhaseLayerFilterVTable = undefined;
var g_obj_filter_vtable: joltc.JPC_ObjectLayerPairFilterVTable = undefined;
var g_vtables_ready: bool = false;
var g_jolt_initialized: bool = false;

fn ensureVtables() void {
    if (g_vtables_ready) return;

    g_bp_vtable = std.mem.zeroes(joltc.JPC_BroadPhaseLayerInterfaceVTable);
    g_bp_vtable.GetNumBroadPhaseLayers = &bpGetNumLayers;
    g_bp_vtable.GetBroadPhaseLayer = &bpGetLayer;

    g_bp_filter_vtable = std.mem.zeroes(joltc.JPC_ObjectVsBroadPhaseLayerFilterVTable);
    g_bp_filter_vtable.ShouldCollide = &bpFilterShouldCollide;

    g_obj_filter_vtable = std.mem.zeroes(joltc.JPC_ObjectLayerPairFilterVTable);
    g_obj_filter_vtable.ShouldCollide = &objFilterShouldCollide;

    g_vtables_ready = true;
}

// ── Physics world state ───────────────────────────────────────────────────────

pub const PhysicsState = struct {
    allocator: std.mem.Allocator,
    temp_allocator: *joltc.JPC_TempAllocator,
    job_system: *joltc.JPC_JobSystem,
    bp_interface: BPLayerInterfaceImpl,
    bp_filter: BPFilterImpl,
    obj_filter: ObjFilterImpl,
    physics_system: *joltc.JPC_PhysicsSystem,
};

pub fn init(allocator: std.mem.Allocator, max_bodies: u32, num_threads: i32) !*PhysicsState {
    ensureVtables();

    if (!g_jolt_initialized) {
        joltc.JPC_RegisterDefaultAllocator();
        joltc.JPC_CreateFactory();
        joltc.JPC_RegisterTypes();
        g_jolt_initialized = true;
    }

    const temp_alloc = joltc.JPC_TempAllocator_Create(16 * 1024 * 1024) orelse
        return error.PhysicsTempAllocFailed;
    errdefer joltc.JPC_TempAllocator_Destroy(temp_alloc);

    const job_sys = joltc.JPC_JobSystem_Create(
        2048, // JPC_MAX_PHYSICS_JOBS
        8, // JPC_MAX_PHYSICS_BARRIERS
        num_threads,
    ) orelse return error.PhysicsJobSystemFailed;
    errdefer joltc.JPC_JobSystem_Destroy(job_sys);

    const state = try allocator.create(PhysicsState);
    errdefer allocator.destroy(state);

    state.* = .{
        .allocator = allocator,
        .temp_allocator = temp_alloc,
        .job_system = job_sys,
        .bp_interface = .{ .vtable = &g_bp_vtable },
        .bp_filter = .{ .vtable = &g_bp_filter_vtable },
        .obj_filter = .{ .vtable = &g_obj_filter_vtable },
        .physics_system = undefined,
    };

    const max_pairs = max_bodies * 2;
    const max_contacts = max_bodies * 2;

    const phys_sys = joltc.JPC_PhysicsSystem_Create(
        max_bodies,
        0,
        max_pairs,
        max_contacts,
        &state.bp_interface,
        &state.bp_filter,
        &state.obj_filter,
    ) orelse return error.PhysicsSystemCreateFailed;

    state.physics_system = phys_sys;
    return state;
}

pub fn deinit(state: *PhysicsState) void {
    joltc.JPC_PhysicsSystem_Destroy(state.physics_system);
    joltc.JPC_JobSystem_Destroy(state.job_system);
    joltc.JPC_TempAllocator_Destroy(state.temp_allocator);
    state.allocator.destroy(state);
}

pub fn step(state: *PhysicsState, delta_time: f32, collision_steps: i32) void {
    _ = joltc.JPC_PhysicsSystem_Update(
        state.physics_system,
        delta_time,
        collision_steps,
        state.temp_allocator,
        state.job_system,
    );
}

pub fn setGravity(state: *PhysicsState, x: f32, y: f32, z: f32) void {
    const g = [3]f32{ x, y, z };
    joltc.JPC_PhysicsSystem_SetGravity(state.physics_system, &g);
}

pub fn optimizeBroadPhase(state: *PhysicsState) void {
    joltc.JPC_PhysicsSystem_OptimizeBroadPhase(state.physics_system);
}

// ── Euler ↔ Quaternion helpers ────────────────────────────────────────────────

fn eulerToQuat(rx: f32, ry: f32, rz: f32) [4]f32 {
    const cx = @cos(rx * 0.5);
    const sx = @sin(rx * 0.5);
    const cy = @cos(ry * 0.5);
    const sy = @sin(ry * 0.5);
    const cz = @cos(rz * 0.5);
    const sz = @sin(rz * 0.5);
    return .{
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
        cx * cy * cz + sx * sy * sz,
    };
}

fn quatToEuler(q: [4]f32) [3]f32 {
    const qx = q[0];
    const qy = q[1];
    const qz = q[2];
    const qw = q[3];

    const sinr = 2.0 * (qw * qx + qy * qz);
    const cosr = 1.0 - 2.0 * (qx * qx + qy * qy);
    const roll = std.math.atan2(sinr, cosr);

    const sinp = 2.0 * (qw * qy - qz * qx);
    const pitch = if (@abs(sinp) >= 1.0)
        std.math.copysign(@as(f32, std.math.pi / 2.0), sinp)
    else
        std.math.asin(sinp);

    const siny = 2.0 * (qw * qz + qx * qy);
    const cosy = 1.0 - 2.0 * (qy * qy + qz * qz);
    const yaw = std.math.atan2(siny, cosy);

    return .{ roll, pitch, yaw };
}

// ── Body management ───────────────────────────────────────────────────────────

fn createShape(shape_type: u8, hx: f32, hy: f32, hz: f32) ?*joltc.JPC_Shape {
    return switch (shape_type) {
        0 => blk: { // Box
            const he = [3]f32{ hx, hy, hz };
            const s = joltc.JPC_BoxShapeSettings_Create(&he) orelse break :blk null;
            defer joltc.JPC_ShapeSettings_Release(@ptrCast(s));
            break :blk joltc.JPC_ShapeSettings_CreateShape(@ptrCast(s));
        },
        1 => blk: { // Sphere (radius = hx)
            const s = joltc.JPC_SphereShapeSettings_Create(hx) orelse break :blk null;
            defer joltc.JPC_ShapeSettings_Release(@ptrCast(s));
            break :blk joltc.JPC_ShapeSettings_CreateShape(@ptrCast(s));
        },
        2 => blk: { // Capsule (radius = hx, half-height = hy)
            const s = joltc.JPC_CapsuleShapeSettings_Create(hy, hx) orelse break :blk null;
            defer joltc.JPC_ShapeSettings_Release(@ptrCast(s));
            break :blk joltc.JPC_ShapeSettings_CreateShape(@ptrCast(s));
        },
        3 => blk: { // Cylinder (radius = hx, half-height = hy)
            const s = joltc.JPC_CylinderShapeSettings_Create(hy, hx) orelse break :blk null;
            defer joltc.JPC_ShapeSettings_Release(@ptrCast(s));
            break :blk joltc.JPC_ShapeSettings_CreateShape(@ptrCast(s));
        },
        else => null,
    };
}

pub fn createBody(
    state: *PhysicsState,
    shape_type: u8,
    hx: f32,
    hy: f32,
    hz: f32,
    px: f32,
    py: f32,
    pz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    motion_type: u8,
    friction: f32,
    restitution: f32,
    gravity_factor: f32,
    mass: f32,
) u32 {
    const shape = createShape(shape_type, hx, hy, hz) orelse return INVALID_BODY_ID;
    defer joltc.JPC_Shape_Release(shape);

    // 0=Static, 1=Kinematic, 2=Dynamic  (matches JPC_MOTION_TYPE_* values)
    const mt: u8 = switch (motion_type) {
        0 => 0,
        1 => 1,
        else => 2,
    };
    const layer: u16 = if (mt == 0) LAYER_NON_MOVING else LAYER_MOVING;

    const pos = [3]f32{ px, py, pz };
    const quat = eulerToQuat(rx, ry, rz);

    var cs: joltc.JPC_BodyCreationSettings = undefined;
    joltc.JPC_BodyCreationSettings_Set(&cs, shape, &pos, &quat, mt, layer);
    cs.friction = friction;
    cs.restitution = restitution;
    cs.gravity_factor = gravity_factor;
    // mass <= 0 means "let Jolt derive mass + inertia from the shape" (the default). A positive value
    // overrides the mass and recomputes inertia from the shape scaled to it. Only meaningful for
    // dynamic bodies; Jolt ignores it for static/kinematic.
    if (mass > 0.0) {
        cs.override_mass_properties = @intCast(joltc.JPC_OVERRIDE_MASS_PROPS_CALC_INERTIA);
        cs.mass_properties_override.mass = mass;
    }

    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse
        return INVALID_BODY_ID;
    const body = joltc.JPC_BodyInterface_CreateBody(iface, &cs) orelse return INVALID_BODY_ID;
    return joltc.JPC_Body_GetID(body).id;
}

pub fn destroyBody(state: *PhysicsState, body_id: u32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    // Jolt asserts if a still-added body is destroyed; remove it from the
    // simulation first when necessary.
    if (joltc.JPC_BodyInterface_IsAdded(iface, jid)) {
        joltc.JPC_BodyInterface_RemoveBody(iface, jid);
    }
    joltc.JPC_BodyInterface_DestroyBody(iface, jid);
}

pub fn addBody(state: *PhysicsState, body_id: u32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    joltc.JPC_BodyInterface_AddBody(iface, jid, 0); // JPC_ACTIVATION_ACTIVATE
}

pub fn removeBody(state: *PhysicsState, body_id: u32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    joltc.JPC_BodyInterface_RemoveBody(iface, jid);
}

pub fn getBodyPosition(state: *PhysicsState, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    var pos: [3]f32 = .{ 0, 0, 0 };
    joltc.JPC_BodyInterface_GetPosition(iface, jid, &pos);
    out_x.* = pos[0];
    out_y.* = pos[1];
    out_z.* = pos[2];
}

pub fn getBodyRotation(state: *PhysicsState, body_id: u32, out_rx: *f32, out_ry: *f32, out_rz: *f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    var quat: [4]f32 = .{ 0, 0, 0, 1 };
    joltc.JPC_BodyInterface_GetRotation(iface, jid, &quat);
    const euler = quatToEuler(quat);
    out_rx.* = euler[0];
    out_ry.* = euler[1];
    out_rz.* = euler[2];
}

pub fn setBodyPosition(state: *PhysicsState, body_id: u32, x: f32, y: f32, z: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    const pos = [3]f32{ x, y, z };
    joltc.JPC_BodyInterface_SetPosition(iface, jid, &pos, 0); // JPC_ACTIVATION_ACTIVATE
}

pub fn setLinearVelocity(state: *PhysicsState, body_id: u32, x: f32, y: f32, z: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    const vel = [3]f32{ x, y, z };
    joltc.JPC_BodyInterface_SetLinearVelocity(iface, jid, &vel);
}

pub fn setAngularVelocity(state: *PhysicsState, body_id: u32, x: f32, y: f32, z: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    const vel = [3]f32{ x, y, z };
    joltc.JPC_BodyInterface_SetAngularVelocity(iface, jid, &vel);
}

// Jolt drops a force/torque/impulse on a sleeping body (BodyInterface guards on IsActive()), so a body
// that has gone to sleep silently ignores everything applied to it — a force-driven body (e.g. a vehicle
// chassis held still by its own suspension) would never wake and stop responding to drive/steer/brake.
// Activate first so applying a force always wakes the body, which is the intuitive engine-level contract.
pub fn addForce(state: *PhysicsState, body_id: u32, x: f32, y: f32, z: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    joltc.JPC_BodyInterface_ActivateBody(iface, jid);
    const force = [3]f32{ x, y, z };
    joltc.JPC_BodyInterface_AddForce(iface, jid, &force);
}

pub fn addImpulse(state: *PhysicsState, body_id: u32, x: f32, y: f32, z: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    joltc.JPC_BodyInterface_ActivateBody(iface, jid);
    const impulse = [3]f32{ x, y, z };
    joltc.JPC_BodyInterface_AddImpulse(iface, jid, &impulse);
}

pub fn addTorque(state: *PhysicsState, body_id: u32, x: f32, y: f32, z: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    joltc.JPC_BodyInterface_ActivateBody(iface, jid);
    const t = [3]f32{ x, y, z };
    joltc.JPC_BodyInterface_AddTorque(iface, jid, &t);
}

pub fn addForceAtPoint(state: *PhysicsState, body_id: u32, fx: f32, fy: f32, fz: f32, px: f32, py: f32, pz: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    joltc.JPC_BodyInterface_ActivateBody(iface, jid);
    const f = [3]f32{ fx, fy, fz };
    const p = [3]f32{ px, py, pz }; // JPC_Real == f32 in this build
    joltc.JPC_BodyInterface_AddForceAtPosition(iface, jid, &f, &p);
}

pub fn getLinearVelocity(state: *PhysicsState, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    var v: [3]f32 = .{ 0, 0, 0 };
    joltc.JPC_BodyInterface_GetLinearVelocity(iface, jid, &v);
    out_x.* = v[0];
    out_y.* = v[1];
    out_z.* = v[2];
}

pub fn getAngularVelocity(state: *PhysicsState, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    var v: [3]f32 = .{ 0, 0, 0 };
    joltc.JPC_BodyInterface_GetAngularVelocity(iface, jid, &v);
    out_x.* = v[0];
    out_y.* = v[1];
    out_z.* = v[2];
}

pub fn getBodyRotationQuat(state: *PhysicsState, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32, out_w: *f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    var q: [4]f32 = .{ 0, 0, 0, 1 };
    joltc.JPC_BodyInterface_GetRotation(iface, jid, &q);
    out_x.* = q[0];
    out_y.* = q[1];
    out_z.* = q[2];
    out_w.* = q[3];
}

pub fn setBodyRotationQuat(state: *PhysicsState, body_id: u32, qx: f32, qy: f32, qz: f32, qw: f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    const jid = joltc.JPC_BodyID{ .id = body_id };
    const q = [4]f32{ qx, qy, qz, qw };
    joltc.JPC_BodyInterface_SetRotation(iface, jid, &q, 0); // JPC_ACTIVATION_ACTIVATE
}

/// Batched transform read for the per-tick body→node sync: writes 7 f32 per body
/// (pos.xyz + quat.xyzw) into `out_xforms`, which must hold `count * 7` floats.
pub fn getBodyTransforms(state: *PhysicsState, ids: [*]const u32, count: u32, out_xforms: [*]f32) void {
    const iface = joltc.JPC_PhysicsSystem_GetBodyInterface(state.physics_system) orelse return;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const jid = joltc.JPC_BodyID{ .id = ids[i] };
        var pos: [3]f32 = .{ 0, 0, 0 };
        var quat: [4]f32 = .{ 0, 0, 0, 1 };
        joltc.JPC_BodyInterface_GetPosition(iface, jid, &pos);
        joltc.JPC_BodyInterface_GetRotation(iface, jid, &quat);
        const base = i * 7;
        out_xforms[base + 0] = pos[0];
        out_xforms[base + 1] = pos[1];
        out_xforms[base + 2] = pos[2];
        out_xforms[base + 3] = quat[0];
        out_xforms[base + 4] = quat[1];
        out_xforms[base + 5] = quat[2];
        out_xforms[base + 6] = quat[3];
    }
}

// A body filter that rejects a single body id (so e.g. a vehicle's wheel ray skips its own chassis).
const BodyFilterImpl = extern struct {
    vtable: ?*const joltc.JPC_BodyFilterVTable,
    ignore_id: u32,
};

fn bodyFilterShouldCollide(self: ?*const anyopaque, body_id: [*c]const joltc.JPC_BodyID) callconv(.c) bool {
    const f: *const BodyFilterImpl = @ptrCast(@alignCast(self));
    if (body_id != null) return body_id.*.id != f.ignore_id;
    return true;
}

fn bodyFilterShouldCollideLocked(_: ?*const anyopaque, _: [*c]const joltc.JPC_Body) callconv(.c) bool {
    return true;
}

var g_body_filter_vtable: joltc.JPC_BodyFilterVTable = undefined;
var g_body_filter_ready: bool = false;

fn ensureBodyFilterVtable() void {
    if (g_body_filter_ready) return;
    g_body_filter_vtable = std.mem.zeroes(joltc.JPC_BodyFilterVTable);
    g_body_filter_vtable.ShouldCollide = &bodyFilterShouldCollide;
    g_body_filter_vtable.ShouldCollideLocked = &bodyFilterShouldCollideLocked;
    g_body_filter_ready = true;
}

/// Closest-hit ray cast against the world. Returns true on hit and fills the out params. When
/// `ignore_body` is a valid id, that body is skipped (e.g. a vehicle's own chassis). The contact normal
/// is reported as world-up (correct for flat terrain); a true surface normal would need a body lock +
/// JPC_Body_GetWorldSpaceSurfaceNormal (deferred).
pub fn raycastClosest(
    state: *PhysicsState,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    max_dist: f32,
    ignore_body: u32,
    out_body: *u32,
    out_fraction: *f32,
    out_px: *f32,
    out_py: *f32,
    out_pz: *f32,
    out_nx: *f32,
    out_ny: *f32,
    out_nz: *f32,
) bool {
    const query = joltc.JPC_PhysicsSystem_GetNarrowPhaseQuery(state.physics_system) orelse return false;

    var ray: joltc.JPC_RRayCast = undefined;
    ray.origin = .{ ox, oy, oz, 0 };
    ray.direction = .{ dx * max_dist, dy * max_dist, dz * max_dist, 0 };

    var hit: joltc.JPC_RayCastResult = undefined;
    hit.body_id = .{ .id = INVALID_BODY_ID };
    hit.fraction = 1.0 + std.math.floatEps(f32);
    hit.sub_shape_id = std.mem.zeroes(@TypeOf(hit.sub_shape_id));

    var filter: BodyFilterImpl = undefined;
    var filter_ptr: ?*const anyopaque = null;
    if (ignore_body != INVALID_BODY_ID) {
        ensureBodyFilterVtable();
        filter = .{ .vtable = &g_body_filter_vtable, .ignore_id = ignore_body };
        filter_ptr = @ptrCast(&filter);
    }

    const did = joltc.JPC_NarrowPhaseQuery_CastRay(query, &ray, &hit, null, null, filter_ptr);
    if (!did) return false;

    out_body.* = hit.body_id.id;
    out_fraction.* = hit.fraction;
    out_px.* = ox + dx * max_dist * hit.fraction;
    out_py.* = oy + dy * max_dist * hit.fraction;
    out_pz.* = oz + dz * max_dist * hit.fraction;
    out_nx.* = 0;
    out_ny.* = 1;
    out_nz.* = 0;
    return true;
}
