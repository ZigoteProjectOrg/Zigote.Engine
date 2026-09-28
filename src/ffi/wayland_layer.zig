//! wlr-layer-shell for shell surfaces — panels, docks, launchers. A Zigote window created through
//! `zigote_window_create_layer` is an ordinary SDL window whose `wl_surface` SDL leaves role-less
//! (`SDL_PROP_WINDOW_CREATE_WAYLAND_SURFACE_ROLE_CUSTOM_BOOLEAN`); this file gives it the
//! `zwlr_layer_surface_v1` role instead of the xdg_toplevel one, and keeps the compositor's
//! configure/ack handshake going for as long as the window lives.
//!
//! No wayland-scanner and no link-time dependency: libwayland-client is dlopen'd (SDL loads it the
//! same way), and the two zwlr interfaces are written out by hand as the `wl_interface` tables the
//! scanner would have generated. Every request is `wl_proxy_marshal_array_flags`, no varargs.
//!
//! Protocol: wlr-layer-shell-unstable-v1 (we speak version 1 — enough for anchor, size, exclusive
//! zone and keyboard interactivity). The compositor must advertise `zwlr_layer_shell_v1`; GNOME's
//! Mutter does not, so `available()` is checked before the SDL window is created and the caller
//! falls back to a plain toplevel there.
const std = @import("std");
const wayland_toplevel = @import("wayland_toplevel.zig");
const wayland_vkbd = @import("wayland_vkbd.zig");

pub const Layer = enum(u32) { background = 0, bottom = 1, top = 2, overlay = 3 };
/// Bitmask, matching zwlr_layer_surface_v1.anchor: top=1, bottom=2, left=4, right=8.
pub const Anchor = u32;
pub const Keyboard = enum(u32) { none = 0, exclusive = 1, on_demand = 2 };

pub const Options = struct {
    /// The wl_output to place the surface on; null lets the compositor choose (the focused one).
    output: ?*Proxy = null,
    layer: Layer,
    anchor: Anchor,
    exclusive_zone: i32,
    keyboard: Keyboard,
    /// Logical size; 0 on an axis means "stretch between the anchors on that axis".
    width: u32,
    height: u32,
};

// ── libwayland-client ABI ─────────────────────────────────────────────────────

pub const Proxy = opaque {};

pub const WlMessage = extern struct {
    name: [*:0]const u8,
    signature: [*:0]const u8,
    types: [*]const ?*const WlInterface,
};

pub const WlInterface = extern struct {
    name: [*:0]const u8,
    version: c_int,
    method_count: c_int,
    methods: [*]const WlMessage,
    event_count: c_int,
    events: [*]const WlMessage,
};

pub const WlArgument = extern union {
    i: i32,
    u: u32,
    f: i32,
    s: ?[*:0]const u8,
    o: ?*Proxy,
    n: u32,
    a: ?*anyopaque,
    h: i32,
};

pub const WL_MARSHAL_FLAG_DESTROY: u32 = 1;

const MarshalFn = *const fn (*Proxy, u32, ?*const WlInterface, u32, u32, [*]WlArgument) callconv(.c) ?*Proxy;
const AddListenerFn = *const fn (*Proxy, [*]const ?*const anyopaque, ?*anyopaque) callconv(.c) c_int;
const GetVersionFn = *const fn (*Proxy) callconv(.c) u32;
const DestroyFn = *const fn (*Proxy) callconv(.c) void;
const RoundtripFn = *const fn (*Proxy) callconv(.c) c_int;

pub const Lib = struct {
    handle: std.DynLib,
    marshal: MarshalFn,
    add_listener: AddListenerFn,
    get_version: GetVersionFn,
    destroy: DestroyFn,
    roundtrip: RoundtripFn,
    registry_interface: *const WlInterface,
    surface_interface: *const WlInterface,
    output_interface: *const WlInterface,
    compositor_interface: *const WlInterface,
    region_interface: *const WlInterface,
    seat_interface: *const WlInterface,

    var cached: ?Lib = null;

    pub fn get() !*Lib {
        if (cached) |*l| return l;
        var handle = std.DynLib.open("libwayland-client.so.0") catch return error.WaylandClientUnavailable;
        errdefer handle.close();
        const l = Lib{
            .handle = handle,
            .marshal = handle.lookup(MarshalFn, "wl_proxy_marshal_array_flags") orelse return error.WaylandSymbolMissing,
            .add_listener = handle.lookup(AddListenerFn, "wl_proxy_add_listener") orelse return error.WaylandSymbolMissing,
            .get_version = handle.lookup(GetVersionFn, "wl_proxy_get_version") orelse return error.WaylandSymbolMissing,
            .destroy = handle.lookup(DestroyFn, "wl_proxy_destroy") orelse return error.WaylandSymbolMissing,
            .roundtrip = handle.lookup(RoundtripFn, "wl_display_roundtrip") orelse return error.WaylandSymbolMissing,
            .registry_interface = handle.lookup(*const WlInterface, "wl_registry_interface") orelse return error.WaylandSymbolMissing,
            .surface_interface = handle.lookup(*const WlInterface, "wl_surface_interface") orelse return error.WaylandSymbolMissing,
            .output_interface = handle.lookup(*const WlInterface, "wl_output_interface") orelse return error.WaylandSymbolMissing,
            .compositor_interface = handle.lookup(*const WlInterface, "wl_compositor_interface") orelse return error.WaylandSymbolMissing,
            .region_interface = handle.lookup(*const WlInterface, "wl_region_interface") orelse return error.WaylandSymbolMissing,
            .seat_interface = handle.lookup(*const WlInterface, "wl_seat_interface") orelse return error.WaylandSymbolMissing,
        };
        // The zwlr tables reference wl_surface/wl_output by address, which only exist at runtime.
        layer_shell_types[1] = l.surface_interface;
        layer_shell_types[2] = l.output_interface;
        cached = l;
        return &cached.?;
    }
};

// ── zwlr_layer_shell_v1 / zwlr_layer_surface_v1 interface tables ──────────────

pub var null_types = [_]?*const WlInterface{ null, null, null, null, null };
var layer_shell_types = [_]?*const WlInterface{ &layer_surface_interface, null, null, null, null };

const layer_shell_requests = [_]WlMessage{
    .{ .name = "get_layer_surface", .signature = "no?ous", .types = &layer_shell_types },
    .{ .name = "destroy", .signature = "3", .types = &null_types },
};

pub var layer_shell_interface = WlInterface{
    .name = "zwlr_layer_shell_v1",
    .version = 4,
    .method_count = layer_shell_requests.len,
    .methods = &layer_shell_requests,
    .event_count = 0,
    .events = &[_]WlMessage{},
};

// Request opcodes, in protocol order.
const LS_SET_SIZE: u32 = 0;
const LS_SET_ANCHOR: u32 = 1;
const LS_SET_EXCLUSIVE_ZONE: u32 = 2;
const LS_SET_MARGIN: u32 = 3;
const LS_SET_KEYBOARD_INTERACTIVITY: u32 = 4;
const LS_GET_POPUP: u32 = 5;
const LS_ACK_CONFIGURE: u32 = 6;
const LS_DESTROY: u32 = 7;

const layer_surface_requests = [_]WlMessage{
    .{ .name = "set_size", .signature = "uu", .types = &null_types },
    .{ .name = "set_anchor", .signature = "u", .types = &null_types },
    .{ .name = "set_exclusive_zone", .signature = "i", .types = &null_types },
    .{ .name = "set_margin", .signature = "iiii", .types = &null_types },
    .{ .name = "set_keyboard_interactivity", .signature = "u", .types = &null_types },
    // xdg_popup's interface lives inside SDL, unexported; untyped is legal and we never send it.
    .{ .name = "get_popup", .signature = "o", .types = &null_types },
    .{ .name = "ack_configure", .signature = "u", .types = &null_types },
    .{ .name = "destroy", .signature = "", .types = &null_types },
    .{ .name = "set_layer", .signature = "2u", .types = &null_types },
    .{ .name = "set_exclusive_edge", .signature = "5u", .types = &null_types },
};

const layer_surface_events = [_]WlMessage{
    .{ .name = "configure", .signature = "uuu", .types = &null_types },
    .{ .name = "closed", .signature = "", .types = &null_types },
};

pub var layer_surface_interface = WlInterface{
    .name = "zwlr_layer_surface_v1",
    .version = 4,
    .method_count = layer_surface_requests.len,
    .methods = &layer_surface_requests,
    .event_count = layer_surface_events.len,
    .events = &layer_surface_events,
};

// wl_registry / wl_surface opcodes we use.
const DISPLAY_GET_REGISTRY: u32 = 1;
const REGISTRY_BIND: u32 = 0;
const SURFACE_SET_INPUT_REGION: u32 = 5;
const SURFACE_COMMIT: u32 = 6;
const COMPOSITOR_CREATE_REGION: u32 = 1;
const REGION_DESTROY: u32 = 0;
const REGION_ADD: u32 = 1;

// ── Registry scan ─────────────────────────────────────────────────────────────

pub const Globals = struct {
    lib: *Lib,
    registry: *Proxy,
    layer_shell: ?*Proxy = null,
    compositor: ?*Proxy = null,
    /// Only bound when `want_toplevels` — the foreign-toplevel client asks for these.
    want_toplevels: bool = false,
    toplevel_manager: ?*Proxy = null,
    seat: ?*Proxy = null,
    vkbd_manager: ?*Proxy = null,

    fn onGlobal(data: ?*anyopaque, registry: *Proxy, name: u32, interface: [*:0]const u8, version: u32) callconv(.c) void {
        const self: *Globals = @ptrCast(@alignCast(data.?));
        const name_slice = std.mem.span(interface);
        if (std.mem.eql(u8, name_slice, "zwlr_layer_shell_v1")) {
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = @min(version, 1) }, .{ .n = 0 } };
            self.layer_shell = self.lib.marshal(registry, REGISTRY_BIND, &layer_shell_interface, @min(version, 1), 0, &args);
        } else if (std.mem.eql(u8, name_slice, "wl_compositor")) {
            // Only for create_region (input regions); version 1 has it.
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = 1 }, .{ .n = 0 } };
            self.compositor = self.lib.marshal(registry, REGISTRY_BIND, self.lib.compositor_interface, 1, 0, &args);
        } else if (self.want_toplevels and std.mem.eql(u8, name_slice, "zwlr_foreign_toplevel_manager_v1")) {
            const v = @min(version, 3);
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = v }, .{ .n = 0 } };
            self.toplevel_manager = self.lib.marshal(registry, REGISTRY_BIND, &wayland_toplevel.manager_interface, v, 0, &args);
        } else if (self.want_toplevels and std.mem.eql(u8, name_slice, "zwp_virtual_keyboard_manager_v1")) {
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = 1 }, .{ .n = 0 } };
            self.vkbd_manager = self.lib.marshal(registry, REGISTRY_BIND, &wayland_vkbd.manager_interface, 1, 0, &args);
        } else if (self.want_toplevels and self.seat == null and std.mem.eql(u8, name_slice, "wl_seat")) {
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = 1 }, .{ .n = 0 } };
            self.seat = self.lib.marshal(registry, REGISTRY_BIND, self.lib.seat_interface, 1, 0, &args);
        }
    }

    fn onGlobalRemove(_: ?*anyopaque, _: *Proxy, _: u32) callconv(.c) void {}

    const listener = [_]?*const anyopaque{ @ptrCast(&onGlobal), @ptrCast(&onGlobalRemove) };

    /// Bind the layer-shell global, if the compositor has one. The registry stays alive with it.
    pub fn scan(lib: *Lib, display: *Proxy) !Globals {
        return scanWith(lib, display, false);
    }

    pub fn scanWith(lib: *Lib, display: *Proxy, want_toplevels: bool) !Globals {
        var args = [_]WlArgument{.{ .n = 0 }};
        const registry = lib.marshal(display, DISPLAY_GET_REGISTRY, lib.registry_interface, lib.get_version(display), 0, &args) orelse return error.WaylandRegistryUnavailable;
        var g = Globals{ .lib = lib, .registry = registry, .want_toplevels = want_toplevels };
        _ = lib.add_listener(registry, &listener, &g);
        _ = lib.roundtrip(display);
        return g;
    }
};

/// Whether the compositor behind `wl_display` speaks layer-shell. Cheap: one registry roundtrip.
pub fn available(wl_display: *anyopaque) bool {
    const lib = Lib.get() catch return false;
    const g = Globals.scan(lib, @ptrCast(wl_display)) catch return false;
    defer lib.destroy(g.registry);
    if (g.compositor) |c| lib.destroy(c);
    if (g.layer_shell) |ls| {
        lib.destroy(ls);
        return true;
    }
    return false;
}

// ── The layer surface ─────────────────────────────────────────────────────────

pub const LayerSurface = struct {
    lib: *Lib,
    registry: *Proxy,
    layer_shell: *Proxy,
    compositor: ?*Proxy,
    surface: *Proxy,
    layer_surface: *Proxy,
    /// Last size the compositor configured, in logical pixels (0 = it left the axis to us).
    width: u32 = 0,
    height: u32 = 0,
    configured: bool = false,
    /// Set by the compositor's `closed` event (the output went away, or the shell was told to
    /// stop). The window owner should destroy the window.
    closed: bool = false,
    /// Called on every configure after the first, on the thread SDL dispatches Wayland events on
    /// (the main thread). The engine resizes the SDL window here so the swapchain follows.
    on_configure: ?*const fn (*LayerSurface) void = null,
    user: ?*anyopaque = null,

    fn onConfigure(data: ?*anyopaque, proxy: *Proxy, serial: u32, width: u32, height: u32) callconv(.c) void {
        const self: *LayerSurface = @ptrCast(@alignCast(data.?));
        var args = [_]WlArgument{.{ .u = serial }};
        _ = self.lib.marshal(proxy, LS_ACK_CONFIGURE, null, self.lib.get_version(proxy), 0, &args);
        self.width = width;
        self.height = height;
        const first = !self.configured;
        self.configured = true;
        if (!first) if (self.on_configure) |cb| cb(self);
    }

    fn onClosed(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const self: *LayerSurface = @ptrCast(@alignCast(data.?));
        self.closed = true;
    }

    const listener = [_]?*const anyopaque{ @ptrCast(&onConfigure), @ptrCast(&onClosed) };

    /// Give `wl_surface` the layer role and run the first configure handshake. On return the
    /// surface is acked and `width`/`height` hold what the compositor granted; the caller sizes
    /// the SDL window to that before attaching any buffer. The returned struct must not move
    /// (the listener holds its address) — it is heap-allocated for that reason.
    pub fn create(alloc: std.mem.Allocator, wl_display: *anyopaque, wl_surface: *anyopaque, opts: Options) !*LayerSurface {
        const lib = try Lib.get();
        const display: *Proxy = @ptrCast(wl_display);
        const surface: *Proxy = @ptrCast(wl_surface);

        const g = try Globals.scan(lib, display);
        errdefer lib.destroy(g.registry);
        errdefer if (g.compositor) |c| lib.destroy(c);
        const layer_shell = g.layer_shell orelse return error.LayerShellUnavailable;
        errdefer lib.destroy(layer_shell);

        const self = try alloc.create(LayerSurface);
        errdefer alloc.destroy(self);

        var get_args = [_]WlArgument{ .{ .n = 0 }, .{ .o = surface }, .{ .o = opts.output }, .{ .u = @intFromEnum(opts.layer) }, .{ .s = "zigote-shell" } };
        const ls = lib.marshal(layer_shell, 0, &layer_surface_interface, lib.get_version(layer_shell), 0, &get_args) orelse return error.LayerSurfaceUnavailable;

        self.* = .{ .lib = lib, .registry = g.registry, .layer_shell = layer_shell, .compositor = g.compositor, .surface = surface, .layer_surface = ls };
        _ = lib.add_listener(ls, &listener, self);

        const v = lib.get_version(ls);
        var size_args = [_]WlArgument{ .{ .u = opts.width }, .{ .u = opts.height } };
        _ = lib.marshal(ls, LS_SET_SIZE, null, v, 0, &size_args);
        var anchor_args = [_]WlArgument{.{ .u = opts.anchor }};
        _ = lib.marshal(ls, LS_SET_ANCHOR, null, v, 0, &anchor_args);
        var zone_args = [_]WlArgument{.{ .i = opts.exclusive_zone }};
        _ = lib.marshal(ls, LS_SET_EXCLUSIVE_ZONE, null, v, 0, &zone_args);
        var kb_args = [_]WlArgument{.{ .u = @intFromEnum(opts.keyboard) }};
        _ = lib.marshal(ls, LS_SET_KEYBOARD_INTERACTIVITY, null, v, 0, &kb_args);

        // A commit with no buffer asks for the first configure; nothing may be attached before
        // its ack, or the compositor closes the connection with a protocol error.
        var no_args = [_]WlArgument{};
        _ = lib.marshal(surface, SURFACE_COMMIT, null, lib.get_version(surface), 0, &no_args);
        var tries: u32 = 0;
        while (!self.configured and tries < 8) : (tries += 1) {
            if (lib.roundtrip(display) < 0) break;
        }
        if (!self.configured) {
            // Only the role object goes here; the errdefers above release the registry and the
            // layer-shell global (releasing them twice is a use-after-free in libwayland).
            var destroy_args = [_]WlArgument{};
            _ = lib.marshal(ls, LS_DESTROY, null, v, WL_MARSHAL_FLAG_DESTROY, &destroy_args);
            return error.LayerSurfaceNotConfigured;
        }
        // A 0 on an axis means we asked for a fixed size there and the compositor kept it.
        if (self.width == 0) self.width = opts.width;
        if (self.height == 0) self.height = opts.height;
        return self;
    }

    /// Ask the compositor for a new size (0 on an axis keeps stretching between anchors). The
    /// answer arrives as a configure, which `on_configure` turns into an SDL resize.
    pub fn setSize(self: *LayerSurface, width: u32, height: u32) void {
        var size_args = [_]WlArgument{ .{ .u = width }, .{ .u = height } };
        _ = self.lib.marshal(self.layer_surface, LS_SET_SIZE, null, self.lib.get_version(self.layer_surface), 0, &size_args);
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(self.surface, SURFACE_COMMIT, null, self.lib.get_version(self.surface), 0, &no_args);
    }

    /// Restrict input to the given surface-local rectangles (x, y, w, h quadruples); everything
    /// else passes through to what is below — a desktop-widget layer stays clickable only where
    /// a widget is. No rectangles: the whole surface takes input again.
    pub fn setInputRects(self: *LayerSurface, rects: []const i32) void {
        const compositor = self.compositor orelse return;
        const sv = self.lib.get_version(self.surface);
        var no_args = [_]WlArgument{};
        if (rects.len < 4) {
            var clear = [_]WlArgument{.{ .o = null }};
            _ = self.lib.marshal(self.surface, SURFACE_SET_INPUT_REGION, null, sv, 0, &clear);
            _ = self.lib.marshal(self.surface, SURFACE_COMMIT, null, sv, 0, &no_args);
            return;
        }
        var new_args = [_]WlArgument{.{ .n = 0 }};
        const region = self.lib.marshal(compositor, COMPOSITOR_CREATE_REGION, self.lib.region_interface, self.lib.get_version(compositor), 0, &new_args) orelse return;
        var i: usize = 0;
        while (i + 3 < rects.len) : (i += 4) {
            var add = [_]WlArgument{ .{ .i = rects[i] }, .{ .i = rects[i + 1] }, .{ .i = rects[i + 2] }, .{ .i = rects[i + 3] } };
            _ = self.lib.marshal(region, REGION_ADD, null, 1, 0, &add);
        }
        var set = [_]WlArgument{.{ .o = region }};
        _ = self.lib.marshal(self.surface, SURFACE_SET_INPUT_REGION, null, sv, 0, &set);
        _ = self.lib.marshal(region, REGION_DESTROY, null, 1, WL_MARSHAL_FLAG_DESTROY, &no_args);
        _ = self.lib.marshal(self.surface, SURFACE_COMMIT, null, sv, 0, &no_args);
    }

    /// Offset from the anchored edges, logical pixels — how a menu lands under its bar item.
    pub fn setMargin(self: *LayerSurface, top: i32, right: i32, bottom: i32, left: i32) void {
        const v = self.lib.get_version(self.layer_surface);
        var args = [_]WlArgument{ .{ .i = top }, .{ .i = right }, .{ .i = bottom }, .{ .i = left } };
        _ = self.lib.marshal(self.layer_surface, LS_SET_MARGIN, null, v, 0, &args);
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(self.surface, SURFACE_COMMIT, null, self.lib.get_version(self.surface), 0, &no_args);
    }

    fn destroyProxies(self: *LayerSurface) void {
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(self.layer_surface, LS_DESTROY, null, self.lib.get_version(self.layer_surface), WL_MARSHAL_FLAG_DESTROY, &no_args);
        self.lib.destroy(self.layer_shell);
        if (self.compositor) |c| self.lib.destroy(c);
        self.lib.destroy(self.registry);
    }

    /// Drop the role objects. Call before SDL destroys the wl_surface.
    pub fn destroy(self: *LayerSurface, alloc: std.mem.Allocator) void {
        self.destroyProxies();
        alloc.destroy(self);
    }
};

test "interface tables line up with the protocol" {
    // Opcodes are positions in the request table; a reorder here would send the wrong request.
    try std.testing.expectEqualStrings("ack_configure", std.mem.span(layer_surface_requests[LS_ACK_CONFIGURE].name));
    try std.testing.expectEqualStrings("destroy", std.mem.span(layer_surface_requests[LS_DESTROY].name));
    try std.testing.expectEqualStrings("set_keyboard_interactivity", std.mem.span(layer_surface_requests[LS_SET_KEYBOARD_INTERACTIVITY].name));
    try std.testing.expectEqual(@as(usize, 6), std.mem.span(layer_shell_requests[0].signature).len);
    try std.testing.expectEqual(@as(c_int, 2), layer_surface_interface.event_count);
}
