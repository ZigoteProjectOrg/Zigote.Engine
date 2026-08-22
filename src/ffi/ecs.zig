/// flecs ECS C ABI for the Zigote FFI layer — the zigote_ecs_* exports.
///
/// Every operation used to exist twice: a `pub fn` here and a one-line `export fn` wrapper that
/// called it. The header said the wrappers lived in root.zig (the audio.zig/physics.zig pattern),
/// but they had been moved into the bottom of this same file, so the indirection had no remaining
/// purpose — 55 pairs of function and forwarder. The exports now carry the bodies directly.
/// See docs/v2-design.md §2.4.
///
/// All handles are u64. World handles come from zigote_ecs_world_create() and are
/// INDEPENDENT of the engine EngineState (no zigote_init / SDL window required).
/// Entity ids are ecs_entity_t (u64), which packs index+generation; 0 = null entity.
///
/// Naming note: a Zig parameter named for a C# keyword breaks the binding generator, so the
/// exported names differ from the obvious ones in two places (`evt`, `baseEntity`).
const std = @import("std");

pub const flecs = @cImport({
    @cInclude("flecs.h");
});

// ── Handle helpers ─────────────────────────────────────────────────────────────

pub fn worldFromHandle(h: u64) *flecs.ecs_world_t {
    return @ptrFromInt(h);
}

pub fn queryFromHandle(h: u64) *flecs.ecs_query_t {
    return @ptrFromInt(h);
}

/// Heap-allocated iterator wrapper so we never marshal ecs_iter_t (a ~300-byte struct)
/// across the C ABI. C# only ever sees a u64 handle to the box.
pub const IterBox = struct {
    it: flecs.ecs_iter_t,
};

pub fn iterFromHandle(h: u64) *IterBox {
    return @ptrFromInt(h);
}

// ── World lifecycle ────────────────────────────────────────────────────────────

export fn zigote_ecs_world_create() u64 {
    const world = flecs.ecs_init();
    return @intFromPtr(world);
}

export fn zigote_ecs_world_destroy(world: u64) void {
    _ = flecs.ecs_fini(worldFromHandle(world));
}

export fn zigote_ecs_progress(world: u64, dt: f32) u8 {
    return if (flecs.ecs_progress(worldFromHandle(world), dt)) 1 else 0;
}

export fn zigote_ecs_set_threads(world: u64, threads: i32) void {
    flecs.ecs_set_threads(worldFromHandle(world), threads);
}

export fn zigote_ecs_set_target_fps(world: u64, fps: f32) void {
    flecs.ecs_set_target_fps(worldFromHandle(world), fps);
}

export fn zigote_ecs_defer_begin(world: u64) void {
    _ = flecs.ecs_defer_begin(worldFromHandle(world));
}

export fn zigote_ecs_defer_end(world: u64) void {
    _ = flecs.ecs_defer_end(worldFromHandle(world));
}

// ── Component registration ─────────────────────────────────────────────────────

export fn zigote_ecs_component_register(world: u64, name: [*c]const u8, size: usize, alignment: usize) u64 {
    const w = worldFromHandle(world);

    // Register by name so re-registration is idempotent (safe under C# hot reload).
    var entity_desc = std.mem.zeroes(flecs.ecs_entity_desc_t);
    entity_desc.name = name;
    entity_desc.symbol = name;
    const entity = flecs.ecs_entity_init(w, &entity_desc);

    var comp_desc = std.mem.zeroes(flecs.ecs_component_desc_t);
    comp_desc.entity = entity;
    comp_desc.type.size = @intCast(size);
    comp_desc.type.alignment = @intCast(alignment);

    return flecs.ecs_component_init(w, &comp_desc);
}

// ── Entity operations ──────────────────────────────────────────────────────────

export fn zigote_ecs_entity_create(world: u64) u64 {
    return flecs.ecs_new(worldFromHandle(world));
}

export fn zigote_ecs_entity_create_named(world: u64, name: [*c]const u8) u64 {
    var desc = std.mem.zeroes(flecs.ecs_entity_desc_t);
    desc.name = name;
    return flecs.ecs_entity_init(worldFromHandle(world), &desc);
}

export fn zigote_ecs_entity_destroy(world: u64, entity: u64) void {
    flecs.ecs_delete(worldFromHandle(world), entity);
}

export fn zigote_ecs_entity_is_alive(world: u64, entity: u64) u8 {
    return if (flecs.ecs_is_alive(worldFromHandle(world), entity)) 1 else 0;
}

export fn zigote_ecs_add(world: u64, entity: u64, component: u64) void {
    flecs.ecs_add_id(worldFromHandle(world), entity, component);
}

export fn zigote_ecs_remove(world: u64, entity: u64, component: u64) void {
    flecs.ecs_remove_id(worldFromHandle(world), entity, component);
}

export fn zigote_ecs_has(world: u64, entity: u64, component: u64) u8 {
    return if (flecs.ecs_has_id(worldFromHandle(world), entity, component)) 1 else 0;
}

// Owns = the entity has the component on ITSELF, not merely inherited via (IsA, prefab). Distinguishes
// a prefab-instance override (owned) from an inherited value — the editor needs this to show overrides.
export fn zigote_ecs_owns(world: u64, entity: u64, component: u64) u8 {
    return if (flecs.ecs_owns_id(worldFromHandle(world), entity, component)) 1 else 0;
}

export fn zigote_ecs_set(world: u64, entity: u64, component: u64, data: [*c]const u8, size: usize) void {
    _ = flecs.ecs_set_id(worldFromHandle(world), entity, component, size, data);
}

export fn zigote_ecs_get(world: u64, entity: u64, component: u64) [*c]const u8 {
    return @ptrCast(flecs.ecs_get_id(worldFromHandle(world), entity, component));
}

export fn zigote_ecs_get_mut(world: u64, entity: u64, component: u64) [*c]u8 {
    return @ptrCast(flecs.ecs_get_mut_id(worldFromHandle(world), entity, component));
}

export fn zigote_ecs_ensure(world: u64, entity: u64, component: u64, size: usize) [*c]u8 {
    return @ptrCast(flecs.ecs_ensure_id(worldFromHandle(world), entity, component, size));
}

export fn zigote_ecs_modified(world: u64, entity: u64, component: u64) void {
    flecs.ecs_modified_id(worldFromHandle(world), entity, component);
}

// ── Queries — pull model ───────────────────────────────────────────────────────

export fn zigote_ecs_query_create(world: u64, components: [*c]const u64, count: u32) u64 {
    var desc = std.mem.zeroes(flecs.ecs_query_desc_t);
    var i: u32 = 0;
    while (i < count and i < flecs.FLECS_TERM_COUNT_MAX) : (i += 1) {
        desc.terms[i].id = components[i];
    }
    const q = flecs.ecs_query_init(worldFromHandle(world), &desc);
    return @intFromPtr(q);
}

export fn zigote_ecs_query_destroy(query: u64) void {
    flecs.ecs_query_fini(queryFromHandle(query));
}

export fn zigote_ecs_query_iter(world: u64, query: u64) u64 {
    const box = std.heap.c_allocator.create(IterBox) catch return 0;
    box.it = flecs.ecs_query_iter(worldFromHandle(world), queryFromHandle(query));
    return @intFromPtr(box);
}

export fn zigote_ecs_query_next(iter: u64) u8 {
    const box = iterFromHandle(iter);
    return if (flecs.ecs_query_next(&box.it)) 1 else 0;
}

export fn zigote_ecs_iter_count(iter: u64) i32 {
    return iterFromHandle(iter).it.count;
}

export fn zigote_ecs_iter_entities(iter: u64) [*c]const u64 {
    return @ptrCast(iterFromHandle(iter).it.entities);
}

export fn zigote_ecs_iter_field(iter: u64, term_index: i32, size: usize) [*c]u8 {
    const box = iterFromHandle(iter);
    return @ptrCast(flecs.ecs_field_w_size(&box.it, size, @intCast(term_index)));
}

export fn zigote_ecs_iter_fini(iter: u64) void {
    const box = iterFromHandle(iter);
    flecs.ecs_iter_fini(&box.it);
    std.heap.c_allocator.destroy(box);
}

// Free only our IterBox wrapper WITHOUT calling ecs_iter_fini. Use this after a pull-query loop
// has run ecs_query_next to completion (false return): flecs finalizes the iterator internally at
// that point, so a second ecs_iter_fini is a double-free (it->query is NULL → crash). Only an
// early-terminated iteration still owns a live flecs iterator and must use iterFini instead.
export fn zigote_ecs_iter_free(iter: u64) void {
    const box = iterFromHandle(iter);
    std.heap.c_allocator.destroy(box);
}

// ── Systems & observers ────────────────────────────────────────────────────────

export fn zigote_ecs_system_create(world: u64, name: [*c]const u8, phase: u64, components: [*c]const u64, count: u32, callback: usize, ctx: u64) u64 {
    const cb: *const fn ([*c]flecs.ecs_iter_t) callconv(.c) void = @ptrFromInt(callback);
    const w = worldFromHandle(world);

    // add is a pointer to a 0-terminated array of ids — use a stack array.
    // flecs v4: a system joins a pipeline phase via a (DependsOn, phase) pair, NOT a bare phase tag.
    // With only the bare tag the default pipeline never schedules it, so it never runs on ecs_progress.
    const phase_ids: [2]flecs.ecs_id_t = .{ flecs.ecs_make_pair(flecs.EcsDependsOn, phase), 0 };
    var entity_desc = std.mem.zeroes(flecs.ecs_entity_desc_t);
    entity_desc.name = name;
    entity_desc.add = &phase_ids;
    const sys_entity = flecs.ecs_entity_init(w, &entity_desc);

    var desc = std.mem.zeroes(flecs.ecs_system_desc_t);
    desc.entity = sys_entity;
    var i: u32 = 0;
    while (i < count and i < flecs.FLECS_TERM_COUNT_MAX) : (i += 1) {
        desc.query.terms[i].id = components[i];
    }
    desc.callback = cb;
    desc.ctx = @ptrFromInt(ctx);

    return flecs.ecs_system_init(w, &desc);
}

export fn zigote_ecs_observer_create(world: u64, name: [*c]const u8, evt: u64, components: [*c]const u64, count: u32, callback: usize, ctx: u64) u64 {
    const cb: *const fn ([*c]flecs.ecs_iter_t) callconv(.c) void = @ptrFromInt(callback);
    const w = worldFromHandle(world);

    var entity_desc = std.mem.zeroes(flecs.ecs_entity_desc_t);
    entity_desc.name = name;
    const obs_entity = flecs.ecs_entity_init(w, &entity_desc);

    var desc = std.mem.zeroes(flecs.ecs_observer_desc_t);
    desc.entity = obs_entity;
    desc.events[0] = evt;
    var i: u32 = 0;
    while (i < count and i < flecs.FLECS_TERM_COUNT_MAX) : (i += 1) {
        desc.query.terms[i].id = components[i];
    }
    desc.callback = cb;
    desc.ctx = @ptrFromInt(ctx);

    return flecs.ecs_observer_init(w, &desc);
}

// Iter accessors keyed on raw *ecs_iter_t passed to C# callbacks.
export fn zigote_ecs_iter_field_from_ptr(iter_ptr: usize, term_index: i32, size: usize) [*c]u8 {
    const it: *flecs.ecs_iter_t = @ptrFromInt(iter_ptr);
    return @ptrCast(flecs.ecs_field_w_size(it, size, @intCast(term_index)));
}

export fn zigote_ecs_iter_count_from_ptr(iter_ptr: usize) i32 {
    const it: *flecs.ecs_iter_t = @ptrFromInt(iter_ptr);
    return it.count;
}

export fn zigote_ecs_iter_entities_from_ptr(iter_ptr: usize) [*c]const u64 {
    const it: *flecs.ecs_iter_t = @ptrFromInt(iter_ptr);
    return @ptrCast(it.entities);
}

export fn zigote_ecs_iter_ctx(iter_ptr: usize) u64 {
    const it: *flecs.ecs_iter_t = @ptrFromInt(iter_ptr);
    return @intFromPtr(it.ctx);
}

// ── Relationships / pairs / prefabs / hierarchy ────────────────────────────────

export fn zigote_ecs_make_pair(relation: u64, target: u64) u64 {
    return flecs.ecs_make_pair(relation, target);
}

export fn zigote_ecs_add_pair(world: u64, e: u64, relation: u64, target: u64) void {
    flecs.ecs_add_pair(worldFromHandle(world), e, relation, target);
}

export fn zigote_ecs_remove_pair(world: u64, e: u64, relation: u64, target: u64) void {
    flecs.ecs_remove_pair(worldFromHandle(world), e, relation, target);
}

export fn zigote_ecs_has_pair(world: u64, e: u64, relation: u64, target: u64) u8 {
    return if (flecs.ecs_has_pair(worldFromHandle(world), e, relation, target)) 1 else 0;
}

export fn zigote_ecs_new_w_pair(world: u64, relation: u64, target: u64) u64 {
    return flecs.ecs_new_w_pair(worldFromHandle(world), relation, target);
}

export fn zigote_ecs_set_parent(world: u64, child: u64, parent: u64) void {
    flecs.ecs_add_pair(worldFromHandle(world), child, flecs.EcsChildOf, parent);
}

export fn zigote_ecs_get_parent(world: u64, child: u64) u64 {
    return flecs.ecs_get_target(worldFromHandle(world), child, flecs.EcsChildOf, 0);
}

export fn zigote_ecs_is_a(world: u64, e: u64, baseEntity: u64) void {
    flecs.ecs_add_pair(worldFromHandle(world), e, flecs.EcsIsA, baseEntity);
}

export fn zigote_ecs_new_prefab(world: u64, name: [*c]const u8) u64 {
    const add_ids: [2]flecs.ecs_id_t = .{ flecs.EcsPrefab, 0 };
    var desc = std.mem.zeroes(flecs.ecs_entity_desc_t);
    desc.name = name;
    desc.add = &add_ids;
    return flecs.ecs_entity_init(worldFromHandle(world), &desc);
}

export fn zigote_ecs_instantiate(world: u64, prefab: u64) u64 {
    return flecs.ecs_new_w_pair(worldFromHandle(world), flecs.EcsIsA, prefab);
}

// ── Built-in entity ids ────────────────────────────────────────────────────────

export fn zigote_ecs_builtin_childof() u64 {
    return flecs.EcsChildOf;
}
export fn zigote_ecs_builtin_isa() u64 {
    return flecs.EcsIsA;
}
export fn zigote_ecs_builtin_prefab() u64 {
    return flecs.EcsPrefab;
}

// Component-instantiation traits. flecs v4 defaults a component to (OnInstantiate, Override) — copied to
// each instance (owned). Adding (OnInstantiate, Inherit) makes it SHARED via (IsA, prefab) instead, so a
// prefab edit propagates to non-overriding instances and ecs_owns distinguishes an explicit override.
export fn zigote_ecs_builtin_oninstantiate() u64 {
    return flecs.EcsOnInstantiate;
}
export fn zigote_ecs_builtin_inherit() u64 {
    return flecs.EcsInherit;
}

/// 0=OnStart 1=PreFrame 2=OnLoad 3=PostLoad 4=PreUpdate 5=OnUpdate 6=OnValidate
/// 7=PostUpdate 8=PreStore 9=OnStore 10=PostFrame
export fn zigote_ecs_builtin_phase(which: u8) u64 {
    return switch (which) {
        0 => flecs.EcsOnStart,
        1 => flecs.EcsPreFrame,
        2 => flecs.EcsOnLoad,
        3 => flecs.EcsPostLoad,
        4 => flecs.EcsPreUpdate,
        5 => flecs.EcsOnUpdate,
        6 => flecs.EcsOnValidate,
        7 => flecs.EcsPostUpdate,
        8 => flecs.EcsPreStore,
        9 => flecs.EcsOnStore,
        10 => flecs.EcsPostFrame,
        else => flecs.EcsOnUpdate,
    };
}

export fn zigote_ecs_event_onset() u64 {
    return flecs.EcsOnSet;
}
export fn zigote_ecs_event_onremove() u64 {
    return flecs.EcsOnRemove;
}
export fn zigote_ecs_event_onadd() u64 {
    return flecs.EcsOnAdd;
}

// ── ECS (flecs) ───────────────────────────────────────────────────────────────
// Thin C-ABI shims delegating to  All handles are u64; entity ids are
// ecs_entity_t (u64 packed index+generation); 0 = null entity.
// ECS worlds are INDEPENDENT of the engine EngineState — no zigote_init required.
