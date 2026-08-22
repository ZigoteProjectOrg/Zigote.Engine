//! Generational handle table.
//!
//! The engine hands `u64` handles across the C ABI and gets them back later. The dangerous case is
//! not a handle that is wrong, it is a handle that *used to be right*: C# holds an id, native frees
//! the object, C# passes the id again. If the handle is a raw pointer that is a use-after-free; if
//! it is a bare index it silently addresses whatever now occupies the slot.
//!
//! A handle here packs `(generation, index)`. Freeing a slot bumps its generation, so every handle
//! issued before the free stops resolving — `get` returns null instead of a dangling pointer. The
//! slot itself is reused, so a long-lived table does not grow without bound.
//!
//! This replaces five different schemes that coexisted in the FFI layer: pointer-as-handle plus a
//! parallel `AutoHashMap(u64, void)` of live pointers (nodes), three separate `AutoHashMap`s keyed
//! by pointer (images, render textures, windows), and — in `ffi/ecs.zig` — a bare `@ptrFromInt`
//! with **no validation at all**, where a stale handle from C# was an immediate segfault. It also
//! replaces the vendored `zpool`, which the engine used for exactly one `Pool()` instantiation.
//! See docs/v2-design.md §2.1.

const std = @import("std");

/// An opaque handle. 0 is never valid, so a zeroed struct or a forgotten field reads as "none"
/// rather than as slot 0 — the mistake this representation is most likely to meet in practice.
pub const Handle = u64;
pub const none: Handle = 0;

/// Generation 0 is reserved so that a handle can never encode to 0 (see `none`). Live slots start
/// at generation 1 and only ever increase.
const first_generation: u32 = 1;

inline fn pack(generation: u32, index: u32) Handle {
    return (@as(u64, generation) << 32) | @as(u64, index);
}
inline fn generationOf(h: Handle) u32 {
    return @intCast(h >> 32);
}
inline fn indexOf(h: Handle) u32 {
    return @truncate(h);
}

pub fn HandleTable(comptime T: type) type {
    return struct {
        const Self = @This();

        const Slot = struct {
            /// Even/odd is not used as the liveness bit: `live` is explicit so a slot can be
            /// reused many times without the generation's meaning depending on parity.
            generation: u32 = first_generation,
            live: bool = false,
            value: T = undefined,
        };

        allocator: std.mem.Allocator,
        slots: std.ArrayListUnmanaged(Slot) = .empty,
        /// Indices of dead slots, newest first. Reuse keeps the table compact.
        free: std.ArrayListUnmanaged(u32) = .empty,
        live_count: u32 = 0,

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{ .allocator = allocator };
        }

        /// Releases the table's own memory. Values are POD to this type — anything owning
        /// resources must be drained first (see `iterator`).
        pub fn deinit(self: *Self) void {
            self.slots.deinit(self.allocator);
            self.free.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn count(self: *const Self) u32 {
            return self.live_count;
        }

        /// Store `value` and return a handle to it. Never returns `none`.
        pub fn add(self: *Self, value: T) !Handle {
            self.live_count += 1;
            if (self.free.pop()) |idx| {
                const slot = &self.slots.items[idx];
                slot.live = true;
                slot.value = value;
                return pack(slot.generation, idx);
            }
            const idx: u32 = @intCast(self.slots.items.len);
            // A u32 index is the ABI; refuse rather than wrap into an alias of slot 0.
            if (self.slots.items.len == std.math.maxInt(u32)) {
                self.live_count -= 1;
                return error.HandleTableFull;
            }
            try self.slots.append(self.allocator, .{
                .generation = first_generation,
                .live = true,
                .value = value,
            });
            return pack(first_generation, idx);
        }

        /// The stored value, or null if the handle is `none`, out of range, already freed, or
        /// from a previous occupant of the slot.
        pub fn get(self: *Self, h: Handle) ?*T {
            const slot = self.slotFor(h) orelse return null;
            return &slot.value;
        }

        pub fn getConst(self: *const Self, h: Handle) ?*const T {
            if (h == none) return null;
            const idx = indexOf(h);
            if (idx >= self.slots.items.len) return null;
            const slot = &self.slots.items[idx];
            if (!slot.live or slot.generation != generationOf(h)) return null;
            return &slot.value;
        }

        pub fn contains(self: *const Self, h: Handle) bool {
            return self.getConst(h) != null;
        }

        /// Free the slot and return the value that was in it, or null if the handle was already
        /// invalid. Returning the value is what lets a caller release whatever it owns.
        pub fn remove(self: *Self, h: Handle) ?T {
            const slot = self.slotFor(h) orelse return null;
            const value = slot.value;
            slot.live = false;
            slot.value = undefined;
            // Wrap rather than saturate: after 2^32 reuses of one slot a very old handle could
            // alias again, which is not a practical failure mode, whereas a saturating generation
            // would permanently retire slots in a long-running process.
            slot.generation +%= 1;
            if (slot.generation == 0) slot.generation = first_generation;
            self.free.append(self.allocator, indexOf(h)) catch {
                // The slot is already dead and unreachable; losing it from the free list costs one
                // slot's memory, which is strictly better than failing a destroy.
            };
            self.live_count -= 1;
            return value;
        }

        /// Invalidate every handle. Slots are kept for reuse; generations bump so nothing issued
        /// before the clear resolves afterwards.
        pub fn clear(self: *Self) void {
            self.free.clearRetainingCapacity();
            for (self.slots.items, 0..) |*slot, i| {
                if (slot.live) {
                    slot.live = false;
                    slot.value = undefined;
                    slot.generation +%= 1;
                    if (slot.generation == 0) slot.generation = first_generation;
                }
                self.free.append(self.allocator, @intCast(i)) catch {};
            }
            self.live_count = 0;
        }

        fn slotFor(self: *Self, h: Handle) ?*Slot {
            if (h == none) return null;
            const idx = indexOf(h);
            if (idx >= self.slots.items.len) return null;
            const slot = &self.slots.items[idx];
            if (!slot.live or slot.generation != generationOf(h)) return null;
            return slot;
        }

        /// Walks the live values, so an owner can release them before `deinit`.
        pub const Iterator = struct {
            table: *Self,
            i: usize = 0,

            pub fn next(self: *Iterator) ?*T {
                while (self.i < self.table.slots.items.len) {
                    const slot = &self.table.slots.items[self.i];
                    self.i += 1;
                    if (slot.live) return &slot.value;
                }
                return null;
            }
        };

        pub fn iterator(self: *Self) Iterator {
            return .{ .table = self };
        }
    };
}

// ── tests ─────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "add and get round-trip" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    const a = try t.add(11);
    const b = try t.add(22);
    try testing.expect(a != none and b != none and a != b);
    try testing.expectEqual(@as(u32, 11), t.get(a).?.*);
    try testing.expectEqual(@as(u32, 22), t.get(b).?.*);
    try testing.expectEqual(@as(u32, 2), t.count());
}

test "a freed handle stops resolving" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    const h = try t.add(7);
    try testing.expectEqual(@as(u32, 7), t.remove(h).?);
    try testing.expect(t.get(h) == null);
    try testing.expect(!t.contains(h));
    try testing.expect(t.remove(h) == null); // double free is a no-op, not a corruption
    try testing.expectEqual(@as(u32, 0), t.count());
}

test "a reused slot does not honour the old handle" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    const old = try t.add(1);
    _ = t.remove(old);
    const new = try t.add(2); // same slot index, next generation

    try testing.expectEqual(indexOf(old), indexOf(new));
    try testing.expect(old != new);
    try testing.expect(t.get(old) == null); // the whole point
    try testing.expectEqual(@as(u32, 2), t.get(new).?.*);
}

test "junk handles are rejected rather than dereferenced" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    const h = try t.add(5);
    try testing.expect(t.get(none) == null);
    try testing.expect(t.get(0xFFFF_FFFF_FFFF_FFFF) == null); // index past the end
    try testing.expect(t.get(pack(999, indexOf(h))) == null); // right slot, wrong generation
    try testing.expect(t.get(h) != null); // ...and the real one still works
}

test "clear invalidates everything and keeps slots reusable" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    const a = try t.add(1);
    const b = try t.add(2);
    t.clear();
    try testing.expectEqual(@as(u32, 0), t.count());
    try testing.expect(t.get(a) == null);
    try testing.expect(t.get(b) == null);

    const c = try t.add(3);
    try testing.expect(c != a and c != b);
    try testing.expectEqual(@as(u32, 3), t.get(c).?.*);
}

test "slots are reused instead of growing" {
    var t = HandleTable(u64).init(testing.allocator);
    defer t.deinit();

    var live: [8]Handle = undefined;
    for (&live, 0..) |*h, i| h.* = try t.add(@intCast(i));
    for (live) |h| _ = t.remove(h);
    for (0..8) |i| _ = try t.add(@intCast(i));

    try testing.expectEqual(@as(usize, 8), t.slots.items.len);
    try testing.expectEqual(@as(u32, 8), t.count());
}

test "iterator visits exactly the live values" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    const a = try t.add(1);
    _ = try t.add(2);
    const c = try t.add(3);
    _ = t.remove(a);
    _ = t.remove(c);

    var seen: u32 = 0;
    var n: u32 = 0;
    var it = t.iterator();
    while (it.next()) |v| {
        seen += v.*;
        n += 1;
    }
    try testing.expectEqual(@as(u32, 1), n);
    try testing.expectEqual(@as(u32, 2), seen);
}

test "handles survive a churn workload without aliasing" {
    var t = HandleTable(u32).init(testing.allocator);
    defer t.deinit();

    // Deterministic churn: keep a rolling set, free the oldest, and check that every handle ever
    // freed stays dead while every live one still resolves to its own value.
    var dead: std.ArrayListUnmanaged(Handle) = .empty;
    defer dead.deinit(testing.allocator);
    var live: std.ArrayListUnmanaged(struct { h: Handle, v: u32 }) = .empty;
    defer live.deinit(testing.allocator);

    for (0..500) |i| {
        const v: u32 = @intCast(i);
        try live.append(testing.allocator, .{ .h = try t.add(v), .v = v });
        if (live.items.len > 16) {
            const oldest = live.orderedRemove(0);
            try testing.expectEqual(oldest.v, t.remove(oldest.h).?);
            try dead.append(testing.allocator, oldest.h);
        }
    }
    for (dead.items) |h| try testing.expect(t.get(h) == null);
    for (live.items) |e| try testing.expectEqual(e.v, t.get(e.h).?.*);
    try testing.expectEqual(@as(u32, 16), t.count());
}
