//! ext-session-lock-v1, client side: the compositor-held lock. Once `lock` is sent the
//! compositor blanks every output and shows only our lock surfaces until `unlock_and_destroy`;
//! if this client dies while locked, the compositor keeps the session locked (a solid colour) —
//! which is exactly why the lock is this protocol and not a full-screen window. A lock surface
//! is an SDL window with a custom role, like a layer surface, with the same configure/ack dance.
const std = @import("std");
const wl = @import("wayland_layer.zig");

const Proxy = wl.Proxy;
const WlMessage = wl.WlMessage;
const WlInterface = wl.WlInterface;
const WlArgument = wl.WlArgument;

var lock_new_types = [_]?*const WlInterface{&lock_interface};
var lock_surface_types = [_]?*const WlInterface{ &lock_surface_interface, null, null }; // wl_surface, wl_output filled at start

const manager_requests = [_]WlMessage{
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
    .{ .name = "lock", .signature = "n", .types = &lock_new_types },
};
pub var manager_interface = WlInterface{
    .name = "ext_session_lock_manager_v1",
    .version = 1,
    .method_count = manager_requests.len,
    .methods = &manager_requests,
    .event_count = 0,
    .events = &[_]WlMessage{},
};

const LOCK_DESTROY: u32 = 0;
const LOCK_GET_SURFACE: u32 = 1;
const LOCK_UNLOCK_AND_DESTROY: u32 = 2;
const lock_requests = [_]WlMessage{
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
    .{ .name = "get_lock_surface", .signature = "noo", .types = &lock_surface_types },
    .{ .name = "unlock_and_destroy", .signature = "", .types = &wl.null_types },
};
const lock_events = [_]WlMessage{
    .{ .name = "locked", .signature = "", .types = &wl.null_types },
    .{ .name = "finished", .signature = "", .types = &wl.null_types },
};
pub var lock_interface = WlInterface{
    .name = "ext_session_lock_v1",
    .version = 1,
    .method_count = lock_requests.len,
    .methods = &lock_requests,
    .event_count = lock_events.len,
    .events = &lock_events,
};

const LS_DESTROY: u32 = 0;
const LS_ACK_CONFIGURE: u32 = 1;
const lock_surface_requests = [_]WlMessage{
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
    .{ .name = "ack_configure", .signature = "u", .types = &wl.null_types },
};
const lock_surface_events = [_]WlMessage{
    .{ .name = "configure", .signature = "uuu", .types = &wl.null_types },
};
pub var lock_surface_interface = WlInterface{
    .name = "ext_session_lock_surface_v1",
    .version = 1,
    .method_count = lock_surface_requests.len,
    .methods = &lock_surface_requests,
    .event_count = lock_surface_events.len,
    .events = &lock_surface_events,
};

const DISPLAY_GET_REGISTRY: u32 = 1;
const REGISTRY_BIND: u32 = 0;

pub const State = enum(u32) { none = 0, pending = 1, locked = 2, finished = 3 };

/// One session lock: from `lock` until `unlock_and_destroy` (or the compositor's `finished`).
pub const Lock = struct {
    alloc: std.mem.Allocator,
    lib: *wl.Lib,
    display: *Proxy,
    registry: *Proxy,
    manager: *Proxy,
    lock: *Proxy,
    state: State = .pending,
    manager_set: bool = false,

    fn onGlobal(data: ?*anyopaque, registry: *Proxy, name: u32, interface: [*:0]const u8, version: u32) callconv(.c) void {
        const self: *Lock = @ptrCast(@alignCast(data.?));
        if (!std.mem.eql(u8, std.mem.span(interface), "ext_session_lock_manager_v1")) return;
        _ = version;
        var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = 1 }, .{ .n = 0 } };
        if (self.lib.marshal(registry, REGISTRY_BIND, &manager_interface, 1, 0, &args)) |m| {
            self.manager = m;
            self.manager_set = true;
        }
    }
    fn onGlobalRemove(_: ?*anyopaque, _: *Proxy, _: u32) callconv(.c) void {}
    const registry_listener = [_]?*const anyopaque{ @ptrCast(&onGlobal), @ptrCast(&onGlobalRemove) };

    fn onLocked(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const self: *Lock = @ptrCast(@alignCast(data.?));
        self.state = .locked;
    }
    fn onFinished(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const self: *Lock = @ptrCast(@alignCast(data.?));
        self.state = .finished;
    }
    const lock_listener = [_]?*const anyopaque{ @ptrCast(&onLocked), @ptrCast(&onFinished) };

    pub fn start(alloc: std.mem.Allocator, wl_display: *anyopaque) !*Lock {
        const lib = try wl.Lib.get();
        lock_surface_types[1] = lib.surface_interface;
        lock_surface_types[2] = lib.output_interface;
        const display: *Proxy = @ptrCast(wl_display);
        var args = [_]WlArgument{.{ .n = 0 }};
        const registry = lib.marshal(display, DISPLAY_GET_REGISTRY, lib.registry_interface, lib.get_version(display), 0, &args) orelse return error.WaylandRegistryUnavailable;
        const self = try alloc.create(Lock);
        errdefer alloc.destroy(self);
        self.* = .{ .alloc = alloc, .lib = lib, .display = display, .registry = registry, .manager = undefined, .lock = undefined };
        _ = lib.add_listener(registry, &registry_listener, self);
        _ = lib.roundtrip(display);
        if (!self.manager_set) {
            lib.destroy(registry);
            return error.SessionLockUnavailable;
        }
        var lock_args = [_]WlArgument{.{ .n = 0 }};
        self.lock = lib.marshal(self.manager, 1, &lock_interface, 1, 0, &lock_args) orelse return error.SessionLockUnavailable;
        _ = lib.add_listener(self.lock, &lock_listener, self);
        _ = lib.roundtrip(display); // locked (or finished, if another client holds a lock)
        return self;
    }

    /// Give an SDL window's wl_surface the lock-surface role on `output`; the returned struct
    /// carries the compositor's size after the first configure.
    pub fn createSurface(self: *Lock, wl_surface: *anyopaque, output: *Proxy) !*LockSurface {
        const s = try self.alloc.create(LockSurface);
        errdefer self.alloc.destroy(s);
        const surface: *Proxy = @ptrCast(wl_surface);
        var args = [_]WlArgument{ .{ .n = 0 }, .{ .o = surface }, .{ .o = output } };
        const proxy = self.lib.marshal(self.lock, LOCK_GET_SURFACE, &lock_surface_interface, 1, 0, &args) orelse return error.LockSurfaceUnavailable;
        s.* = .{ .lib = self.lib, .surface = surface, .proxy = proxy };
        _ = self.lib.add_listener(proxy, &LockSurface.listener, s);
        // Unlike layer-shell there is no bufferless commit here: a lock surface committed with
        // no buffer is a protocol error, and the compositor sends the first configure on its own.
        var tries: u32 = 0;
        while (!s.configured and tries < 8) : (tries += 1) {
            if (self.lib.roundtrip(self.display) < 0) break;
        }
        if (!s.configured) return error.LockSurfaceNotConfigured;
        return s;
    }

    /// `unlock_and_destroy`: the compositor shows the desktop again. Lock surfaces should be
    /// destroyed by the caller afterwards.
    pub fn unlock(self: *Lock) void {
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(self.lock, LOCK_UNLOCK_AND_DESTROY, null, 1, wl.WL_MARSHAL_FLAG_DESTROY, &no_args);
        _ = self.lib.roundtrip(self.display);
        self.lib.destroy(self.manager);
        self.lib.destroy(self.registry);
        self.state = .none;
        self.alloc.destroy(self);
    }
};

pub const LockSurface = struct {
    lib: *wl.Lib,
    surface: *Proxy,
    proxy: *Proxy,
    width: u32 = 0,
    height: u32 = 0,
    configured: bool = false,
    on_configure: ?*const fn (*LockSurface) void = null,
    user: ?*anyopaque = null,

    fn onConfigure(data: ?*anyopaque, proxy: *Proxy, serial: u32, width: u32, height: u32) callconv(.c) void {
        const self: *LockSurface = @ptrCast(@alignCast(data.?));
        var args = [_]WlArgument{.{ .u = serial }};
        _ = self.lib.marshal(proxy, LS_ACK_CONFIGURE, null, 1, 0, &args);
        self.width = width;
        self.height = height;
        const first = !self.configured;
        self.configured = true;
        if (!first) if (self.on_configure) |cb| cb(self);
    }
    const listener = [_]?*const anyopaque{@ptrCast(&onConfigure)};

    pub fn destroy(self: *LockSurface, alloc: std.mem.Allocator) void {
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(self.proxy, LS_DESTROY, null, 1, wl.WL_MARSHAL_FLAG_DESTROY, &no_args);
        alloc.destroy(self);
    }
};

test "tables match the protocol" {
    try std.testing.expectEqualStrings("get_lock_surface", std.mem.span(lock_requests[LOCK_GET_SURFACE].name));
    try std.testing.expectEqualStrings("unlock_and_destroy", std.mem.span(lock_requests[LOCK_UNLOCK_AND_DESTROY].name));
    try std.testing.expectEqualStrings("ack_configure", std.mem.span(lock_surface_requests[LS_ACK_CONFIGURE].name));
}
