//! No-op stand-in for `ecs.zig`, compiled when the build sets `-Decs=false`.
//!
//! Dropping flecs must not drop the *symbols*. The C# P/Invoke bindings are generated from the Zig
//! export list and are therefore the same on every configuration, so a build without these
//! definitions leaves ~50 dangling `zigote_ecs_*` imports. On desktop that is survivable (lazy
//! binding, and nothing calls them); on iOS the engine is linked statically into the app binary
//! and an undeclared symbol is a link error whether or not it is reachable. That is precisely why
//! `-Decs=false` was documented as unusable on iOS in build/Zigote.Native.targets — a build-system
//! constraint that was really this missing file. 3D and Jolt already had stubs; ECS did not.
//!
//! Every handle-taking export returns its normal failure value, which is what a host that ignores
//! the "ECS is unavailable" signal would see anyway: world creation yields 0, and every operation
//! on handle 0 is a no-op. Mirrors physics_stub.zig. See docs/v2-design.md §5.2.

const std = @import("std");
const ZgStatus = @import("zigote_abi").ZgStatus;

export fn zigote_ecs_world_create() u64 {
    return 0;
}

export fn zigote_ecs_world_destroy(_: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_progress(_: u64, _: f32) u8 {
    return 0;
}

export fn zigote_ecs_set_threads(_: u64, _: i32) ZgStatus {
    return .ok;
}

export fn zigote_ecs_set_target_fps(_: u64, _: f32) ZgStatus {
    return .ok;
}

export fn zigote_ecs_defer_begin(_: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_defer_end(_: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_component_register(_: u64, _: [*c]const u8, _: usize, _: usize) u64 {
    return 0;
}

export fn zigote_ecs_entity_create(_: u64) u64 {
    return 0;
}

export fn zigote_ecs_entity_create_named(_: u64, _: [*c]const u8) u64 {
    return 0;
}

export fn zigote_ecs_entity_destroy(_: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_entity_is_alive(_: u64, _: u64) u8 {
    return 0;
}

export fn zigote_ecs_add(_: u64, _: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_remove(_: u64, _: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_has(_: u64, _: u64, _: u64) u8 {
    return 0;
}

export fn zigote_ecs_owns(_: u64, _: u64, _: u64) u8 {
    return 0;
}

export fn zigote_ecs_set(_: u64, _: u64, _: u64, _: [*c]const u8, _: usize) ZgStatus {
    return .ok;
}

export fn zigote_ecs_get(_: u64, _: u64, _: u64) [*c]const u8 {
    return null;
}

export fn zigote_ecs_get_mut(_: u64, _: u64, _: u64) [*c]u8 {
    return null;
}

export fn zigote_ecs_modified(_: u64, _: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_query_create(_: u64, _: [*c]const u64, _: u32) u64 {
    return 0;
}

export fn zigote_ecs_query_destroy(_: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_query_iter(_: u64, _: u64) u64 {
    return 0;
}

export fn zigote_ecs_query_next(_: u64) u8 {
    return 0;
}

export fn zigote_ecs_iter_count(_: u64) i32 {
    return 0;
}

export fn zigote_ecs_iter_entities(_: u64) [*c]const u64 {
    return null;
}

export fn zigote_ecs_iter_field(_: u64, _: i32, _: usize) [*c]u8 {
    return null;
}

export fn zigote_ecs_iter_fini(_: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_iter_free(_: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_system_create(_: u64, _: [*c]const u8, _: u64, _: [*c]const u64, _: u32, _: usize, _: u64) u64 {
    return 0;
}

export fn zigote_ecs_observer_create(_: u64, _: [*c]const u8, _: u64, _: [*c]const u64, _: u32, _: usize, _: u64) u64 {
    return 0;
}

export fn zigote_ecs_iter_field_from_ptr(_: usize, _: i32, _: usize) [*c]u8 {
    return null;
}

export fn zigote_ecs_iter_count_from_ptr(_: usize) i32 {
    return 0;
}

export fn zigote_ecs_iter_entities_from_ptr(_: usize) [*c]const u64 {
    return null;
}

export fn zigote_ecs_iter_ctx(_: usize) u64 {
    return 0;
}

export fn zigote_ecs_add_pair(_: u64, _: u64, _: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_set_parent(_: u64, _: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_get_parent(_: u64, _: u64) u64 {
    return 0;
}

export fn zigote_ecs_is_a(_: u64, _: u64, _: u64) ZgStatus {
    return .ok;
}

export fn zigote_ecs_new_prefab(_: u64, _: [*c]const u8) u64 {
    return 0;
}

export fn zigote_ecs_instantiate(_: u64, _: u64) u64 {
    return 0;
}

export fn zigote_ecs_builtin_childof() u64 {
    return 0;
}

export fn zigote_ecs_builtin_isa() u64 {
    return 0;
}

export fn zigote_ecs_builtin_oninstantiate() u64 {
    return 0;
}

export fn zigote_ecs_builtin_inherit() u64 {
    return 0;
}

export fn zigote_ecs_builtin_phase(_: u8) u64 {
    return 0;
}

export fn zigote_ecs_event_onset() u64 {
    return 0;
}

export fn zigote_ecs_event_onremove() u64 {
    return 0;
}

export fn zigote_ecs_event_onadd() u64 {
    return 0;
}
