//! zwp_virtual_keyboard_v1, client side: an on-screen keyboard types by injecting evdev key
//! events into the seat. The keymap the compositor needs comes from xkbcommon (dlopen'd, like
//! libwayland) for the default layout, written into a memfd and passed by file descriptor.
const std = @import("std");
const wl = @import("wayland_layer.zig");

const Proxy = wl.Proxy;
const WlMessage = wl.WlMessage;
const WlInterface = wl.WlInterface;
const WlArgument = wl.WlArgument;

var keyboard_new_types = [_]?*const WlInterface{ null, &keyboard_interface }; // wl_seat filled at start

const manager_requests = [_]WlMessage{
    .{ .name = "create_virtual_keyboard", .signature = "on", .types = &keyboard_new_types },
};
pub var manager_interface = WlInterface{
    .name = "zwp_virtual_keyboard_manager_v1",
    .version = 1,
    .method_count = manager_requests.len,
    .methods = &manager_requests,
    .event_count = 0,
    .events = &[_]WlMessage{},
};

const K_KEYMAP: u32 = 0;
const K_KEY: u32 = 1;
const K_MODIFIERS: u32 = 2;
const K_DESTROY: u32 = 3;
const keyboard_requests = [_]WlMessage{
    .{ .name = "keymap", .signature = "uhu", .types = &wl.null_types },
    .{ .name = "key", .signature = "uuu", .types = &wl.null_types },
    .{ .name = "modifiers", .signature = "uuuu", .types = &wl.null_types },
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
};
pub var keyboard_interface = WlInterface{
    .name = "zwp_virtual_keyboard_v1",
    .version = 1,
    .method_count = keyboard_requests.len,
    .methods = &keyboard_requests,
    .event_count = 0,
    .events = &[_]WlMessage{},
};

const XkbContextNew = *const fn (c_int) callconv(.c) ?*anyopaque;
const XkbKeymapNewFromNames = *const fn (*anyopaque, ?*anyopaque, c_int) callconv(.c) ?*anyopaque;
const XkbKeymapGetAsString = *const fn (*anyopaque, c_int) callconv(.c) ?[*:0]u8;
const XkbKeymapUnref = *const fn (*anyopaque) callconv(.c) void;
const XkbContextUnref = *const fn (*anyopaque) callconv(.c) void;

pub const VirtualKeyboard = struct {
    lib: *wl.Lib,
    registry: *Proxy,
    manager: *Proxy,
    seat: *Proxy,
    keyboard: *Proxy,

    pub fn start(alloc: std.mem.Allocator, wl_display: *anyopaque) !*VirtualKeyboard {
        const lib = try wl.Lib.get();
        keyboard_new_types[0] = lib.seat_interface;
        const display: *Proxy = @ptrCast(wl_display);
        const g = try wl.Globals.scanWith(lib, display, true);
        // scanWith binds the seat and the toplevel manager; the manager is not wanted here.
        if (g.toplevel_manager) |m| lib.destroy(m);
        if (g.compositor) |c| lib.destroy(c);
        if (g.layer_shell) |ls| lib.destroy(ls);
        const seat = g.seat orelse {
            lib.destroy(g.registry);
            return error.SeatUnavailable;
        };
        const manager = g.vkbd_manager orelse {
            lib.destroy(seat);
            lib.destroy(g.registry);
            return error.VirtualKeyboardUnavailable;
        };
        var args = [_]WlArgument{ .{ .o = seat }, .{ .n = 0 } };
        const keyboard = lib.marshal(manager, 0, &keyboard_interface, 1, 0, &args) orelse return error.VirtualKeyboardUnavailable;
        const self = try alloc.create(VirtualKeyboard);
        self.* = .{ .lib = lib, .registry = g.registry, .manager = manager, .seat = seat, .keyboard = keyboard };
        try self.sendKeymap();
        _ = lib.roundtrip(display);
        return self;
    }

    fn sendKeymap(self: *VirtualKeyboard) !void {
        var xkb = std.DynLib.open("libxkbcommon.so.0") catch return error.XkbUnavailable;
        defer xkb.close();
        const context_new = xkb.lookup(XkbContextNew, "xkb_context_new") orelse return error.XkbUnavailable;
        const keymap_new = xkb.lookup(XkbKeymapNewFromNames, "xkb_keymap_new_from_names") orelse return error.XkbUnavailable;
        const get_string = xkb.lookup(XkbKeymapGetAsString, "xkb_keymap_get_as_string") orelse return error.XkbUnavailable;
        const keymap_unref = xkb.lookup(XkbKeymapUnref, "xkb_keymap_unref") orelse return error.XkbUnavailable;
        const context_unref = xkb.lookup(XkbContextUnref, "xkb_context_unref") orelse return error.XkbUnavailable;
        const ctx = context_new(0) orelse return error.XkbUnavailable;
        defer context_unref(ctx);
        const keymap = keymap_new(ctx, null, 0) orelse return error.XkbUnavailable;
        defer keymap_unref(keymap);
        const text = get_string(keymap, 1) orelse return error.XkbUnavailable; // XKB_KEYMAP_FORMAT_TEXT_V1
        defer std.c.free(text);
        const slice = std.mem.span(text);
        const size: usize = slice.len + 1;
        const fd = std.c.memfd_create("zigote-keymap", 0);
        if (fd < 0) return error.MemfdFailed;
        defer _ = std.c.close(fd);
        var written: usize = 0;
        while (written < size) {
            const n = std.c.write(fd, @as([*]const u8, @ptrCast(text)) + written, size - written);
            if (n <= 0) return error.MemfdFailed;
            written += @intCast(n);
        }
        var args = [_]WlArgument{ .{ .u = 1 }, .{ .h = fd }, .{ .u = @intCast(size) } }; // XKB_V1
        _ = self.lib.marshal(self.keyboard, K_KEYMAP, null, 1, 0, &args);
    }

    const Timespec = extern struct { sec: isize, nsec: isize };
    extern "c" fn clock_gettime(clk: c_int, ts: *Timespec) c_int;

    /// Milliseconds on the monotonic clock, as wl_keyboard timestamps are.
    fn now(self: *VirtualKeyboard) u32 {
        _ = self;
        var ts: Timespec = undefined;
        _ = clock_gettime(1, &ts); // CLOCK_MONOTONIC
        return @intCast((@as(u64, @intCast(ts.sec)) * 1000 + @as(u64, @intCast(ts.nsec)) / 1_000_000) & 0x7FFFFFFF);
    }

    /// evdev keycode (KEY_A = 30), pressed or released.
    pub fn key(self: *VirtualKeyboard, keycode: u32, pressed: bool) void {
        var args = [_]WlArgument{ .{ .u = self.now() }, .{ .u = keycode }, .{ .u = if (pressed) 1 else 0 } };
        _ = self.lib.marshal(self.keyboard, K_KEY, null, 1, 0, &args);
    }

    pub fn modifiers(self: *VirtualKeyboard, depressed: u32, latched: u32, locked: u32, group: u32) void {
        var args = [_]WlArgument{ .{ .u = depressed }, .{ .u = latched }, .{ .u = locked }, .{ .u = group } };
        _ = self.lib.marshal(self.keyboard, K_MODIFIERS, null, 1, 0, &args);
    }

    pub fn destroy(self: *VirtualKeyboard, alloc: std.mem.Allocator) void {
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(self.keyboard, K_DESTROY, null, 1, wl.WL_MARSHAL_FLAG_DESTROY, &no_args);
        self.lib.destroy(self.manager);
        self.lib.destroy(self.seat);
        self.lib.destroy(self.registry);
        alloc.destroy(self);
    }
};

test "request table order" {
    try std.testing.expectEqualStrings("keymap", std.mem.span(keyboard_requests[K_KEYMAP].name));
    try std.testing.expectEqualStrings("modifiers", std.mem.span(keyboard_requests[K_MODIFIERS].name));
}
