//! Named message channels between the managed host and native platform code.
//!
//! The engine gives C# a window and a GPU; it does not give it a media session, a notification,
//! a share sheet or a widget. Those live in code the managed side cannot always reach — a Kotlin
//! `Service` still running after the activity died, an Objective-C delegate, a C++ SDK — and the
//! two halves need a way to talk that does not mean growing a new FFI export per feature. This is
//! that way: a name, a UTF-8 payload, and two directions.
//!
//! Deliberately not a serialization format. The payload is bytes — carried with an explicit
//! length, so binary data (an image, an audio clip) crosses as-is with no base64 detour; what the
//! bytes mean is between the two ends of a channel, and in practice is a short string or a small
//! JSON object. A codec here would have to be implemented once per language on the native side,
//! which is exactly the cost this is meant to avoid. Payloads are NOT guaranteed NUL-terminated:
//! a handler honors `payload_len`, never `strlen`.
//!
//! **Replies never allocate across the boundary.** A handler writes into a buffer the caller owns
//! and returns the length, C-style. The alternative — returning a pointer — obliges every caller
//! in every language to know which allocator freed it, and gets that wrong eventually.
//!
//! Thread-safety: registration and lookup are mutex-guarded, so a service thread may send while
//! the UI thread invokes. Handlers themselves run on the calling thread; the managed receiver is
//! expected to hand work to its own loop rather than touch UI state where it lands.

const std = @import("std");
const SpinLock = @import("zigote").core.sync.SpinLock;

/// A native channel implementation. Reads `payload[0..payload_len]`, writes at most `reply_cap`
/// bytes of reply into `reply`, and returns the reply length — or a negative value for "failed".
/// Returning a length greater than `reply_cap` means the reply was truncated; the caller can retry
/// with a bigger buffer.
pub const Handler = *const fn (
    payload: [*c]const u8,
    payload_len: usize,
    reply: [*c]u8,
    reply_cap: usize,
) callconv(.c) i32;

/// Called on the managed side when native code sends on a channel. One receiver for all channels:
/// the host already has a name-keyed dispatch table, and a second one here would only duplicate it.
pub const Receiver = *const fn (
    name: [*c]const u8,
    payload: [*c]const u8,
    payload_len: usize,
) callconv(.c) void;

/// No allocator, so the registry is fixed. Thirty-two channels is far past what an app uses (a
/// media session, notifications, permissions, share, lifecycle — five or six), and the cost of
/// being wrong is a `false` from register rather than a crash.
const max_channels = 32;
const max_name_len = 63;

const Entry = struct {
    name: [max_name_len + 1]u8 = @splat(0),
    len: usize = 0,
    handler: Handler = undefined,
};

/// A spin lock rather than a mutex: `std.Thread.Mutex` is gone in this zig version and
/// `std.Io.Mutex` wants an `Io` handle the FFI layer has no reason to hold — the same trade
/// root.zig and netstream.zig already make. It fits even better here: every critical section is a
/// scan of at most 32 entries, and handlers are always called with the lock released.

var entries: [max_channels]Entry = @splat(.{});
var entry_count: usize = 0;
var receiver: ?Receiver = null;
var lock: SpinLock = .{};

fn span(s: [*c]const u8) []const u8 {
    if (s == null) return &.{};
    return std.mem.span(@as([*:0]const u8, @ptrCast(s)));
}

fn find(name: []const u8) ?*Entry {
    for (entries[0..entry_count]) |*e| {
        if (std.mem.eql(u8, e.name[0..e.len], name)) return e;
    }
    return null;
}

/// Register (or replace) the native implementation of a channel. Returns false when the name is
/// empty, longer than the fixed limit, or the registry is full.
///
/// Replacing rather than rejecting a duplicate is deliberate: on Android the process outlives the
/// activity, so a relaunch re-registers every channel, and refusing the second registration would
/// leave the channel pointing at a handler belonging to a dead instance.
export fn zigote_channel_register(name: [*c]const u8, handler: *const fn (payload: [*c]const u8, payload_len: usize, reply: [*c]u8, reply_cap: usize) callconv(.c) i32) bool {
    const key = span(name);
    if (key.len == 0 or key.len > max_name_len) return false;

    lock.lock();
    defer lock.unlock();

    if (find(key)) |existing| {
        existing.handler = handler;
        return true;
    }
    if (entry_count == max_channels) return false;

    const slot = &entries[entry_count];
    @memcpy(slot.name[0..key.len], key);
    slot.len = key.len;
    slot.handler = handler;
    entry_count += 1;
    return true;
}

/// Drop a channel's native implementation. Safe to call for a name that was never registered.
export fn zigote_channel_unregister(name: [*c]const u8) void {
    const key = span(name);
    lock.lock();
    defer lock.unlock();
    for (entries[0..entry_count], 0..) |*e, i| {
        if (!std.mem.eql(u8, e.name[0..e.len], key)) continue;
        // Order carries no meaning, so the last entry fills the hole — no shifting.
        entries[i] = entries[entry_count - 1];
        entry_count -= 1;
        return;
    }
}

/// Call a channel's native implementation. Returns the reply length written into `reply`, or -1 when
/// no handler is registered for the name (which the host reports as "this platform does not
/// implement it" rather than an error — an app runs on desktops that have no media session).
///
/// The handler pointer is copied out under the lock and called outside it: a handler that calls
/// back into the bus — a media-session channel asking the host what is playing — would otherwise
/// deadlock on a non-recursive mutex.
export fn zigote_channel_invoke(
    name: [*c]const u8,
    payload: [*c]const u8,
    payload_len: usize,
    reply: [*c]u8,
    reply_cap: usize,
) i32 {
    const handler = blk: {
        lock.lock();
        defer lock.unlock();
        const entry = find(span(name)) orelse return -1;
        break :blk entry.handler;
    };
    return handler(payload, payload_len, reply, reply_cap);
}

/// Whether a channel has a native implementation. Lets the host answer "can I do this here?"
/// without inventing a payload for a call whose only purpose is the probe.
export fn zigote_channel_has(name: [*c]const u8) bool {
    lock.lock();
    defer lock.unlock();
    return find(span(name)) != null;
}

/// Install the managed receiver. Passing null detaches it, which is what shutdown does so a late
/// send from a platform thread cannot reach a torn-down runtime.
export fn zigote_channel_set_receiver(cb: ?*const fn (name: [*c]const u8, payload: [*c]const u8, payload_len: usize) callconv(.c) void) void {
    lock.lock();
    defer lock.unlock();
    receiver = cb;
}

/// Send a message from native code to the managed host. Returns false when no receiver is
/// attached — the app has not started yet, or has already shut down — so a platform callback can
/// tell "nobody listening" from "delivered" and drop the work instead of queueing it forever.
///
/// Callable from any thread. This is the entry point behind the Java/Kotlin, Swift and C++ sides.
export fn zigote_channel_send(name: [*c]const u8, payload: [*c]const u8, payload_len: usize) bool {
    const cb = blk: {
        lock.lock();
        defer lock.unlock();
        break :blk receiver orelse return false;
    };
    cb(name, payload orelse "", if (payload == null) 0 else payload_len);
    return true;
}

test "register, invoke, replace and unregister" {
    const H = struct {
        fn echo(payload: [*c]const u8, payload_len: usize, reply: [*c]u8, reply_cap: usize) callconv(.c) i32 {
            if (payload_len > reply_cap) return @intCast(payload_len);
            @memcpy(reply[0..payload_len], payload[0..payload_len]);
            return @intCast(payload_len);
        }
        fn constant(_: [*c]const u8, _: usize, reply: [*c]u8, reply_cap: usize) callconv(.c) i32 {
            if (reply_cap < 2) return 2;
            @memcpy(reply[0..2], "ok");
            return 2;
        }
    };

    entry_count = 0;
    var buf: [32]u8 = undefined;

    // Unknown channel is distinguishable from a handler that replied with nothing.
    try std.testing.expectEqual(@as(i32, -1), zigote_channel_invoke("nope", "x", 1, &buf, buf.len));
    try std.testing.expect(!zigote_channel_has("media"));

    try std.testing.expect(zigote_channel_register("media", H.echo));
    try std.testing.expect(zigote_channel_has("media"));
    try std.testing.expectEqual(@as(i32, 5), zigote_channel_invoke("media", "hello", 5, &buf, buf.len));
    try std.testing.expectEqualStrings("hello", buf[0..5]);

    // The length is authoritative, not any NUL — embedded zero bytes pass through intact.
    const binary = [_]u8{ 1, 0, 2, 0, 3 };
    try std.testing.expectEqual(@as(i32, 5), zigote_channel_invoke("media", &binary, binary.len, &buf, buf.len));
    try std.testing.expectEqualSlices(u8, &binary, buf[0..5]);

    // Re-registering replaces, so an app relaunch cannot leave a dead handler installed.
    try std.testing.expect(zigote_channel_register("media", H.constant));
    try std.testing.expectEqual(@as(i32, 2), zigote_channel_invoke("media", "hello", 5, &buf, buf.len));
    try std.testing.expectEqualStrings("ok", buf[0..2]);

    zigote_channel_unregister("media");
    try std.testing.expect(!zigote_channel_has("media"));
}

test "a reply larger than the buffer reports the length it needed" {
    const H = struct {
        fn big(_: [*c]const u8, _: usize, reply: [*c]u8, reply_cap: usize) callconv(.c) i32 {
            const text = "0123456789";
            if (reply_cap < text.len) return @intCast(text.len);
            @memcpy(reply[0..text.len], text);
            return @intCast(text.len);
        }
    };
    entry_count = 0;
    try std.testing.expect(zigote_channel_register("big", H.big));
    var small: [4]u8 = undefined;
    try std.testing.expectEqual(@as(i32, 10), zigote_channel_invoke("big", "", 0, &small, small.len));
}

test "send reaches the receiver and reports when there is none" {
    const R = struct {
        var last_name: [32]u8 = @splat(0);
        var last_len: usize = 0;
        var hits: usize = 0;
        fn recv(name: [*c]const u8, _: [*c]const u8, payload_len: usize) callconv(.c) void {
            const n = span(name);
            @memcpy(last_name[0..n.len], n);
            last_len = payload_len;
            hits += 1;
        }
    };
    zigote_channel_set_receiver(null);
    try std.testing.expect(!zigote_channel_send("transport", "next", 4));

    zigote_channel_set_receiver(R.recv);
    try std.testing.expect(zigote_channel_send("transport", "next", 4));
    try std.testing.expectEqual(@as(usize, 1), R.hits);
    try std.testing.expectEqual(@as(usize, 4), R.last_len);
    try std.testing.expectEqualStrings("transport", R.last_name[0..9]);
    zigote_channel_set_receiver(null);
}
