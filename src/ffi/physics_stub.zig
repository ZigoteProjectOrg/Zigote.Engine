/// No-op stand-in for `physics.zig`, selected by `root.zig` when the build sets
/// `-Dphysics3d=false`. It provides the exact public surface the `zigote_physics_*` FFI wrappers
/// reference so root.zig still compiles and links WITHOUT the Jolt (JoltC) static library — a
/// pure-2D/UI build (2D physics is pure-C# `Zigote.Physics2D`) can drop ~2-4 MB of native 3D physics.
///
/// `init` fails, so `EngineState.physics` stays null and every wrapper short-circuits on its
/// `state.physics orelse return` guard — these bodies never actually run. Mirrors the Assimp stub
/// pattern in root.zig. All parameters are named `_` since none are used.
const std = @import("std");

pub const INVALID_BODY_ID: u32 = 0xFFFF_FFFF;

/// Never instantiated (init always errors); exists only so `?*PhysicsState` type-checks.
pub const PhysicsState = opaque {};

pub fn init(_: std.mem.Allocator, _: u32, _: i32) !*PhysicsState {
    return error.Physics3DDisabled;
}

pub fn deinit(_: *PhysicsState) void {}
pub fn step(_: *PhysicsState, _: f32, _: i32) void {}
pub fn setGravity(_: *PhysicsState, _: f32, _: f32, _: f32) void {}
pub fn optimizeBroadPhase(_: *PhysicsState) void {}

pub fn createBody(
    _: *PhysicsState,
    _: u8,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: u8,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
) u32 {
    return INVALID_BODY_ID;
}

pub fn destroyBody(_: *PhysicsState, _: u32) void {}
pub fn addBody(_: *PhysicsState, _: u32) void {}
pub fn removeBody(_: *PhysicsState, _: u32) void {}
pub fn getBodyPosition(_: *PhysicsState, _: u32, _: *f32, _: *f32, _: *f32) void {}
pub fn getBodyRotation(_: *PhysicsState, _: u32, _: *f32, _: *f32, _: *f32) void {}
pub fn setBodyPosition(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32) void {}
pub fn setLinearVelocity(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32) void {}
pub fn setAngularVelocity(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32) void {}
pub fn addForce(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32) void {}
pub fn addImpulse(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32) void {}
pub fn addTorque(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32) void {}
pub fn addForceAtPoint(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32, _: f32, _: f32, _: f32) void {}
pub fn getLinearVelocity(_: *PhysicsState, _: u32, _: *f32, _: *f32, _: *f32) void {}
pub fn getAngularVelocity(_: *PhysicsState, _: u32, _: *f32, _: *f32, _: *f32) void {}
pub fn getBodyRotationQuat(_: *PhysicsState, _: u32, _: *f32, _: *f32, _: *f32, _: *f32) void {}
pub fn setBodyRotationQuat(_: *PhysicsState, _: u32, _: f32, _: f32, _: f32, _: f32) void {}
pub fn getBodyTransforms(_: *PhysicsState, _: [*]const u32, _: u32, _: [*]f32) void {}

pub fn raycastClosest(
    _: *PhysicsState,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: u32,
    _: *u32,
    _: *f32,
    _: *f32,
    _: *f32,
    _: *f32,
    _: *f32,
    _: *f32,
    _: *f32,
) bool {
    return false;
}
