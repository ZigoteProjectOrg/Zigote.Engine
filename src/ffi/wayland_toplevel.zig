//! wlr-foreign-toplevel-management-unstable-v1, client side: the list of every toplevel the
//! compositor manages (app id, title, state) and the requests a dock needs — activate, close,
//! minimise. Same hand-rolled proxy approach as wayland_layer.zig, whose libwayland handle and
//! registry scan this reuses.
//!
//! The compositor only hands this global to clients it trusts to see other windows; a shell is
//! one. Events arrive on the main thread inside SDL's Wayland dispatch; the C ABI is polled from
//! the same thread, so there is no locking.
const std = @import("std");
const wl = @import("wayland_layer.zig");

const Proxy = wl.Proxy;
const WlMessage = wl.WlMessage;
const WlInterface = wl.WlInterface;
const WlArgument = wl.WlArgument;

const WlArray = extern struct { size: usize, alloc: usize, data: ?*anyopaque };

// ── interface tables ─────────────────────────────────────────────────────────

var handle_new_types = [_]?*const WlInterface{&handle_interface};
// The parent event names the handle's own interface — a comptime cycle, so it is filled at start.
var handle_parent_types = [_]?*const WlInterface{null};

const manager_requests = [_]WlMessage{
    .{ .name = "stop", .signature = "", .types = &wl.null_types },
};
const manager_events = [_]WlMessage{
    .{ .name = "toplevel", .signature = "n", .types = &handle_new_types },
    .{ .name = "finished", .signature = "", .types = &wl.null_types },
};
pub var manager_interface = WlInterface{
    .name = "zwlr_foreign_toplevel_manager_v1",
    .version = 3,
    .method_count = manager_requests.len,
    .methods = &manager_requests,
    .event_count = manager_events.len,
    .events = &manager_events,
};

const H_SET_MAXIMIZED: u32 = 0;
const H_UNSET_MAXIMIZED: u32 = 1;
const H_SET_MINIMIZED: u32 = 2;
const H_UNSET_MINIMIZED: u32 = 3;
const H_ACTIVATE: u32 = 4;
const H_CLOSE: u32 = 5;
const H_DESTROY: u32 = 7;

const handle_requests = [_]WlMessage{
    .{ .name = "set_maximized", .signature = "", .types = &wl.null_types },
    .{ .name = "unset_maximized", .signature = "", .types = &wl.null_types },
    .{ .name = "set_minimized", .signature = "", .types = &wl.null_types },
    .{ .name = "unset_minimized", .signature = "", .types = &wl.null_types },
    .{ .name = "activate", .signature = "o", .types = &wl.null_types }, // wl_seat, untyped is legal
    .{ .name = "close", .signature = "", .types = &wl.null_types },
    .{ .name = "set_rectangle", .signature = "oiiii", .types = &wl.null_types },
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
    .{ .name = "set_fullscreen", .signature = "2?o", .types = &wl.null_types },
    .{ .name = "unset_fullscreen", .signature = "2", .types = &wl.null_types },
};
const handle_events = [_]WlMessage{
    .{ .name = "title", .signature = "s", .types = &wl.null_types },
    .{ .name = "app_id", .signature = "s", .types = &wl.null_types },
    .{ .name = "output_enter", .signature = "o", .types = &wl.null_types },
    .{ .name = "output_leave", .signature = "o", .types = &wl.null_types },
    .{ .name = "state", .signature = "a", .types = &wl.null_types },
    .{ .name = "done", .signature = "", .types = &wl.null_types },
    .{ .name = "closed", .signature = "", .types = &wl.null_types },
    .{ .name = "parent", .signature = "3?o", .types = &handle_parent_types },
};
pub var handle_interface = WlInterface{
    .name = "zwlr_foreign_toplevel_handle_v1",
    .version = 3,
    .method_count = handle_requests.len,
    .methods = &handle_requests,
    .event_count = handle_events.len,
    .events = &handle_events,
};

// ── model ────────────────────────────────────────────────────────────────────

/// State bits as the C ABI reports them (protocol enum value → bit).
pub const STATE_MAXIMIZED: u32 = 1;
pub const STATE_MINIMIZED: u32 = 2;
pub const STATE_ACTIVATED: u32 = 4;
pub const STATE_FULLSCREEN: u32 = 8;

pub const Toplevel = struct {
    id: u64,
    handle: *Proxy,
    manager: *Manager,
    title: [256]u8 = undefined,
    title_len: u32 = 0,
    app_id: [256]u8 = undefined,
    app_id_len: u32 = 0,
    state: u32 = 0,
    // Pending until `done`, as the protocol batches them.
    p_title: [256]u8 = undefined,
    p_title_len: u32 = 0,
    p_app_id: [256]u8 = undefined,
    p_app_id_len: u32 = 0,
    p_state: u32 = 0,
};

pub const Manager = struct {
    alloc: std.mem.Allocator,
    lib: *wl.Lib,
    registry: *Proxy,
    manager: *Proxy,
    seat: ?*Proxy,
    list: std.ArrayListUnmanaged(*Toplevel) = .empty,
    /// Bumps on every `done` and `closed`; a poller compares it to skip unchanged frames.
    generation: u32 = 0,
    next_id: u64 = 1,

    pub fn start(alloc: std.mem.Allocator, wl_display: *anyopaque) !*Manager {
        const lib = try wl.Lib.get();
        handle_parent_types[0] = &handle_interface;
        const display: *Proxy = @ptrCast(wl_display);
        const g = try wl.Globals.scanWith(lib, display, true);
        if (g.compositor) |c| lib.destroy(c);
        if (g.layer_shell) |ls| lib.destroy(ls);
        if (g.vkbd_manager) |v| lib.destroy(v);
        const manager = g.toplevel_manager orelse {
            if (g.seat) |s| lib.destroy(s);
            lib.destroy(g.registry);
            return error.ForeignToplevelUnavailable;
        };
        const self = try alloc.create(Manager);
        self.* = .{ .alloc = alloc, .lib = lib, .registry = g.registry, .manager = manager, .seat = g.seat };
        _ = lib.add_listener(manager, &manager_listener, self);
        // The initial burst of toplevel events arrives on the next roundtrip.
        _ = lib.roundtrip(display);
        return self;
    }

    pub fn find(self: *Manager, id: u64) ?*Toplevel {
        for (self.list.items) |t| if (t.id == id) return t;
        return null;
    }

    /// action: 0 activate, 1 close, 2 minimize, 3 unminimize, 4 maximize, 5 unmaximize.
    pub fn act(self: *Manager, id: u64, action: u32) bool {
        const t = self.find(id) orelse return false;
        const v = self.lib.get_version(t.handle);
        var no_args = [_]WlArgument{};
        switch (action) {
            0 => {
                const seat = self.seat orelse return false;
                var args = [_]WlArgument{.{ .o = seat }};
                _ = self.lib.marshal(t.handle, H_ACTIVATE, null, v, 0, &args);
            },
            1 => _ = self.lib.marshal(t.handle, H_CLOSE, null, v, 0, &no_args),
            2 => _ = self.lib.marshal(t.handle, H_SET_MINIMIZED, null, v, 0, &no_args),
            3 => _ = self.lib.marshal(t.handle, H_UNSET_MINIMIZED, null, v, 0, &no_args),
            4 => _ = self.lib.marshal(t.handle, H_SET_MAXIMIZED, null, v, 0, &no_args),
            5 => _ = self.lib.marshal(t.handle, H_UNSET_MAXIMIZED, null, v, 0, &no_args),
            else => return false,
        }
        return true;
    }

    fn onToplevel(data: ?*anyopaque, _: *Proxy, handle: *Proxy) callconv(.c) void {
        const self: *Manager = @ptrCast(@alignCast(data.?));
        const t = self.alloc.create(Toplevel) catch return;
        t.* = .{ .id = self.next_id, .handle = handle, .manager = self };
        self.next_id += 1;
        self.list.append(self.alloc, t) catch {
            self.alloc.destroy(t);
            return;
        };
        _ = self.lib.add_listener(handle, &handle_listener, t);
    }

    fn onFinished(_: ?*anyopaque, _: *Proxy) callconv(.c) void {}

    const manager_listener = [_]?*const anyopaque{ @ptrCast(&onToplevel), @ptrCast(&onFinished) };

    fn copy(dst: *[256]u8, len: *u32, src: [*:0]const u8) void {
        const s = std.mem.span(src);
        const n: usize = @min(s.len, dst.len);
        @memcpy(dst[0..n], s[0..n]);
        len.* = @intCast(n);
    }

    fn onTitle(data: ?*anyopaque, _: *Proxy, title: [*:0]const u8) callconv(.c) void {
        const t: *Toplevel = @ptrCast(@alignCast(data.?));
        copy(&t.p_title, &t.p_title_len, title);
    }

    fn onAppId(data: ?*anyopaque, _: *Proxy, app_id: [*:0]const u8) callconv(.c) void {
        const t: *Toplevel = @ptrCast(@alignCast(data.?));
        copy(&t.p_app_id, &t.p_app_id_len, app_id);
    }

    fn onOutput(_: ?*anyopaque, _: *Proxy, _: *Proxy) callconv(.c) void {}

    fn onState(data: ?*anyopaque, _: *Proxy, array: *const WlArray) callconv(.c) void {
        const t: *Toplevel = @ptrCast(@alignCast(data.?));
        var bits: u32 = 0;
        if (array.data) |d| {
            const n = array.size / @sizeOf(u32);
            const values: [*]const u32 = @ptrCast(@alignCast(d));
            for (values[0..n]) |v| {
                if (v < 4) bits |= @as(u32, 1) << @intCast(v);
            }
        }
        t.p_state = bits;
    }

    fn onDone(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const t: *Toplevel = @ptrCast(@alignCast(data.?));
        t.title = t.p_title;
        t.title_len = t.p_title_len;
        t.app_id = t.p_app_id;
        t.app_id_len = t.p_app_id_len;
        t.state = t.p_state;
        t.manager.generation +%= 1;
    }

    fn onClosed(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const t: *Toplevel = @ptrCast(@alignCast(data.?));
        const m = t.manager;
        var no_args = [_]WlArgument{};
        _ = m.lib.marshal(t.handle, H_DESTROY, null, m.lib.get_version(t.handle), wl.WL_MARSHAL_FLAG_DESTROY, &no_args);
        for (m.list.items, 0..) |x, i| if (x == t) {
            _ = m.list.swapRemove(i);
            break;
        };
        m.alloc.destroy(t);
        m.generation +%= 1;
    }

    fn onParent(_: ?*anyopaque, _: *Proxy, _: ?*Proxy) callconv(.c) void {}

    const handle_listener = [_]?*const anyopaque{
        @ptrCast(&onTitle),  @ptrCast(&onAppId), @ptrCast(&onOutput), @ptrCast(&onOutput),
        @ptrCast(&onState),  @ptrCast(&onDone),  @ptrCast(&onClosed), @ptrCast(&onParent),
    };
};

test "handle table matches the protocol's opcodes" {
    try std.testing.expectEqualStrings("activate", std.mem.span(handle_requests[H_ACTIVATE].name));
    try std.testing.expectEqualStrings("destroy", std.mem.span(handle_requests[H_DESTROY].name));
    try std.testing.expectEqualStrings("closed", std.mem.span(handle_events[6].name));
    try std.testing.expectEqual(@as(usize, 8), Manager.handle_listener.len);
}
