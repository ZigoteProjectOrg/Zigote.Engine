//! Small synchronisation primitives shared across the engine.
//!
//! This Zig version dropped `std.Thread.Mutex`, so three call sites each grew their own
//! near-identical spinlock (`ffi/root.zig`, `ffi/netstream.zig`, `ffi/channel.zig`) with three
//! copies of the same comment explaining why. One implementation now. See docs/v2-design.md §2.4.

const std = @import("std");

/// A spin-then-yield lock.
///
/// Contention is the exception: the image registry sections are a hashmap op and never get past
/// the spin phase. But some holders are long — the audio lock is held across a container-header
/// parse on a loader thread — and a pure spin would burn a core of the frame loop waiting. So
/// once it is clear this is not a short wait, hand the CPU back on *every* subsequent attempt
/// rather than re-spinning between yields: the long holds are the case that matters, and for
/// them repeated yielding is what keeps the waiter off the CPU.
pub const SpinLock = struct {
    locked: std.atomic.Value(bool) = .init(false),

    /// Long enough to cover a short memcpy, short enough not to matter next to a scheduler round.
    const spins_before_yield: u32 = 64;

    pub fn lock(self: *SpinLock) void {
        var spins: u32 = 0;
        while (self.locked.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            spins += 1;
            if (spins < spins_before_yield) {
                std.atomic.spinLoopHint();
            } else {
                std.Thread.yield() catch std.atomic.spinLoopHint();
            }
        }
    }

    pub fn unlock(self: *SpinLock) void {
        self.locked.store(false, .release);
    }

    /// Acquire only if uncontended. Never blocks.
    pub fn tryLock(self: *SpinLock) bool {
        return self.locked.cmpxchgStrong(false, true, .acquire, .monotonic) == null;
    }
};

test "uncontended lock/unlock round-trips and excludes" {
    var l = SpinLock{};
    l.lock();
    try std.testing.expect(!l.tryLock()); // held
    l.unlock();
    try std.testing.expect(l.tryLock()); // free again
    l.unlock();
}

test "mutual exclusion under real contention" {
    var l = SpinLock{};
    var counter: u64 = 0;
    const Worker = struct {
        fn run(lock: *SpinLock, c: *u64) void {
            for (0..20_000) |_| {
                lock.lock();
                defer lock.unlock();
                // Non-atomic read-modify-write: races here show up as a short final count.
                c.* += 1;
            }
        }
    };
    var threads: [4]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Worker.run, .{ &l, &counter });
    for (threads) |t| t.join();
    try std.testing.expectEqual(@as(u64, 4 * 20_000), counter);
}
