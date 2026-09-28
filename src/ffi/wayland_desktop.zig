//! Outputs and workspaces for a desktop shell: every `wl_output` the compositor advertises (name,
//! logical size, scale — kept live through global add/remove) and the `ext-workspace-v1` list
//! with activate. One persistent registry with its own listener, unlike the one-shot scans in
//! wayland_layer.zig. Same proxy machinery, same main-thread-only rule.
const std = @import("std");
const wl = @import("wayland_layer.zig");

const Proxy = wl.Proxy;
const WlMessage = wl.WlMessage;
const WlInterface = wl.WlInterface;
const WlArgument = wl.WlArgument;
const WlArray = extern struct { size: usize, alloc: usize, data: ?*anyopaque };

const REGISTRY_BIND: u32 = 0;
const DISPLAY_GET_REGISTRY: u32 = 1;

// ── ext_workspace_manager_v1 tables ──────────────────────────────────────────

var group_new_types = [_]?*const WlInterface{&group_interface};
var workspace_new_types = [_]?*const WlInterface{&workspace_interface};
var workspace_obj_types = [_]?*const WlInterface{null}; // filled at start (self-referential)

const manager_requests = [_]WlMessage{
    .{ .name = "commit", .signature = "", .types = &wl.null_types },
    .{ .name = "stop", .signature = "", .types = &wl.null_types },
};
const manager_events = [_]WlMessage{
    .{ .name = "workspace_group", .signature = "n", .types = &group_new_types },
    .{ .name = "workspace", .signature = "n", .types = &workspace_new_types },
    .{ .name = "done", .signature = "", .types = &wl.null_types },
    .{ .name = "finished", .signature = "", .types = &wl.null_types },
};
pub var manager_interface = WlInterface{
    .name = "ext_workspace_manager_v1",
    .version = 1,
    .method_count = manager_requests.len,
    .methods = &manager_requests,
    .event_count = manager_events.len,
    .events = &manager_events,
};

const group_requests = [_]WlMessage{
    .{ .name = "create_workspace", .signature = "s", .types = &wl.null_types },
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
};
const group_events = [_]WlMessage{
    .{ .name = "capabilities", .signature = "u", .types = &wl.null_types },
    .{ .name = "output_enter", .signature = "o", .types = &wl.null_types },
    .{ .name = "output_leave", .signature = "o", .types = &wl.null_types },
    .{ .name = "workspace_enter", .signature = "o", .types = &workspace_obj_types },
    .{ .name = "workspace_leave", .signature = "o", .types = &workspace_obj_types },
    .{ .name = "removed", .signature = "", .types = &wl.null_types },
};
pub var group_interface = WlInterface{
    .name = "ext_workspace_group_handle_v1",
    .version = 1,
    .method_count = group_requests.len,
    .methods = &group_requests,
    .event_count = group_events.len,
    .events = &group_events,
};

const WS_DESTROY: u32 = 0;
const WS_ACTIVATE: u32 = 1;
const workspace_requests = [_]WlMessage{
    .{ .name = "destroy", .signature = "", .types = &wl.null_types },
    .{ .name = "activate", .signature = "", .types = &wl.null_types },
    .{ .name = "deactivate", .signature = "", .types = &wl.null_types },
    .{ .name = "assign", .signature = "o", .types = &group_new_types },
    .{ .name = "remove", .signature = "", .types = &wl.null_types },
};
const workspace_events = [_]WlMessage{
    .{ .name = "id", .signature = "s", .types = &wl.null_types },
    .{ .name = "name", .signature = "s", .types = &wl.null_types },
    .{ .name = "coordinates", .signature = "a", .types = &wl.null_types },
    .{ .name = "state", .signature = "u", .types = &wl.null_types },
    .{ .name = "capabilities", .signature = "u", .types = &wl.null_types },
    .{ .name = "removed", .signature = "", .types = &wl.null_types },
};
pub var workspace_interface = WlInterface{
    .name = "ext_workspace_handle_v1",
    .version = 1,
    .method_count = workspace_requests.len,
    .methods = &workspace_requests,
    .event_count = workspace_events.len,
    .events = &workspace_events,
};

// ── model ────────────────────────────────────────────────────────────────────

pub const Output = struct {
    id: u64,
    global: u32,
    proxy: *Proxy,
    desktop: *Desktop,
    name: [64]u8 = undefined,
    name_len: u32 = 0,
    // Pending until `done`.
    p_w: i32 = 0,
    p_h: i32 = 0,
    p_scale: i32 = 1,
    p_transform: i32 = 0,
    w: u32 = 0,
    h: u32 = 0,
    scale: f32 = 1,
};

pub const Workspace = struct {
    id: u64,
    proxy: *Proxy,
    desktop: *Desktop,
    name: [64]u8 = undefined,
    name_len: u32 = 0,
    state: u32 = 0, // 1 active, 2 urgent, 4 hidden
    caps: u32 = 0, // 1 activate, 2 deactivate, 4 remove, 8 assign
    p_name: [64]u8 = undefined,
    p_name_len: u32 = 0,
    p_state: u32 = 0,
    p_caps: u32 = 0,
};

pub const Desktop = struct {
    alloc: std.mem.Allocator,
    lib: *wl.Lib,
    display: *Proxy,
    registry: *Proxy,
    outputs: std.ArrayListUnmanaged(*Output) = .empty,
    outputs_generation: u32 = 1,
    workspace_manager: ?*Proxy = null,
    workspaces: std.ArrayListUnmanaged(*Workspace) = .empty,
    workspaces_generation: u32 = 1,
    next_id: u64 = 1,

    pub fn start(alloc: std.mem.Allocator, wl_display: *anyopaque) !*Desktop {
        const lib = try wl.Lib.get();
        workspace_obj_types[0] = &workspace_interface;
        const display: *Proxy = @ptrCast(wl_display);
        var args = [_]WlArgument{.{ .n = 0 }};
        const registry = lib.marshal(display, DISPLAY_GET_REGISTRY, lib.registry_interface, lib.get_version(display), 0, &args) orelse return error.WaylandRegistryUnavailable;
        const self = try alloc.create(Desktop);
        self.* = .{ .alloc = alloc, .lib = lib, .display = display, .registry = registry };
        _ = lib.add_listener(registry, &registry_listener, self);
        _ = lib.roundtrip(display); // globals
        _ = lib.roundtrip(display); // their first events (output modes, workspace list)
        return self;
    }

    pub fn findOutput(self: *Desktop, id: u64) ?*Output {
        for (self.outputs.items) |o| if (o.id == id) return o;
        return null;
    }

    pub fn activateWorkspace(self: *Desktop, id: u64) bool {
        const m = self.workspace_manager orelse return false;
        for (self.workspaces.items) |w| if (w.id == id) {
            var no_args = [_]WlArgument{};
            _ = self.lib.marshal(w.proxy, WS_ACTIVATE, null, 1, 0, &no_args);
            _ = self.lib.marshal(m, 0, null, 1, 0, &no_args); // commit
            return true;
        };
        return false;
    }

    // ── registry ──

    fn onGlobal(data: ?*anyopaque, registry: *Proxy, name: u32, interface: [*:0]const u8, version: u32) callconv(.c) void {
        const self: *Desktop = @ptrCast(@alignCast(data.?));
        const iface = std.mem.span(interface);
        if (std.mem.eql(u8, iface, "wl_output")) {
            const v = @min(version, 4);
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = v }, .{ .n = 0 } };
            const proxy = self.lib.marshal(registry, REGISTRY_BIND, self.lib.output_interface, v, 0, &args) orelse return;
            const o = self.alloc.create(Output) catch return;
            o.* = .{ .id = self.next_id, .global = name, .proxy = proxy, .desktop = self };
            self.next_id += 1;
            self.outputs.append(self.alloc, o) catch {
                self.alloc.destroy(o);
                return;
            };
            _ = self.lib.add_listener(proxy, &output_listener, o);
        } else if (self.workspace_manager == null and std.mem.eql(u8, iface, "ext_workspace_manager_v1")) {
            var args = [_]WlArgument{ .{ .u = name }, .{ .s = interface }, .{ .u = 1 }, .{ .n = 0 } };
            self.workspace_manager = self.lib.marshal(registry, REGISTRY_BIND, &manager_interface, 1, 0, &args);
            if (self.workspace_manager) |m| _ = self.lib.add_listener(m, &manager_listener, self);
        }
    }

    fn onGlobalRemove(data: ?*anyopaque, _: *Proxy, name: u32) callconv(.c) void {
        const self: *Desktop = @ptrCast(@alignCast(data.?));
        for (self.outputs.items, 0..) |o, i| if (o.global == name) {
            self.lib.destroy(o.proxy);
            _ = self.outputs.orderedRemove(i);
            self.alloc.destroy(o);
            self.outputs_generation +%= 1;
            return;
        };
    }

    const registry_listener = [_]?*const anyopaque{ @ptrCast(&onGlobal), @ptrCast(&onGlobalRemove) };

    // ── wl_output ──

    fn onGeometry(data: ?*anyopaque, _: *Proxy, _: i32, _: i32, _: i32, _: i32, _: i32, _: [*:0]const u8, _: [*:0]const u8, transform: i32) callconv(.c) void {
        const o: *Output = @ptrCast(@alignCast(data.?));
        o.p_transform = transform;
    }

    fn onMode(data: ?*anyopaque, _: *Proxy, flags: u32, w: i32, h: i32, _: i32) callconv(.c) void {
        const o: *Output = @ptrCast(@alignCast(data.?));
        if (flags & 1 == 0) return; // not the current mode
        o.p_w = w;
        o.p_h = h;
    }

    fn onOutputDone(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const o: *Output = @ptrCast(@alignCast(data.?));
        const scale: f32 = @floatFromInt(@max(o.p_scale, 1));
        // Transforms 90/270 (and their flipped forms, odd values) swap the axes.
        const rotated = @rem(o.p_transform, 2) == 1;
        const w: f32 = @floatFromInt(if (rotated) o.p_h else o.p_w);
        const h: f32 = @floatFromInt(if (rotated) o.p_w else o.p_h);
        o.w = @intFromFloat(@round(w / scale));
        o.h = @intFromFloat(@round(h / scale));
        o.scale = scale;
        o.desktop.outputs_generation +%= 1;
    }

    fn onScale(data: ?*anyopaque, _: *Proxy, factor: i32) callconv(.c) void {
        const o: *Output = @ptrCast(@alignCast(data.?));
        o.p_scale = factor;
    }

    fn onOutputName(data: ?*anyopaque, _: *Proxy, name: [*:0]const u8) callconv(.c) void {
        const o: *Output = @ptrCast(@alignCast(data.?));
        copy(&o.name, &o.name_len, name);
    }

    fn onDescription(_: ?*anyopaque, _: *Proxy, _: [*:0]const u8) callconv(.c) void {}

    const output_listener = [_]?*const anyopaque{
        @ptrCast(&onGeometry), @ptrCast(&onMode), @ptrCast(&onOutputDone), @ptrCast(&onScale), @ptrCast(&onOutputName), @ptrCast(&onDescription),
    };

    // ── workspaces ──

    fn onGroup(_: ?*anyopaque, _: *Proxy, group: *Proxy) callconv(.c) void {
        // ponytail: one group per output is labwc's model; groups are not tracked — a
        // per-output switcher needs group→output mapping, add when a second-output UI exists.
        _ = group;
    }

    fn onWorkspace(data: ?*anyopaque, _: *Proxy, handle: *Proxy) callconv(.c) void {
        const self: *Desktop = @ptrCast(@alignCast(data.?));
        const w = self.alloc.create(Workspace) catch return;
        w.* = .{ .id = self.next_id, .proxy = handle, .desktop = self };
        self.next_id += 1;
        self.workspaces.append(self.alloc, w) catch {
            self.alloc.destroy(w);
            return;
        };
        _ = self.lib.add_listener(handle, &workspace_listener, w);
    }

    fn onManagerDone(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const self: *Desktop = @ptrCast(@alignCast(data.?));
        for (self.workspaces.items) |w| {
            w.name = w.p_name;
            w.name_len = w.p_name_len;
            w.state = w.p_state;
            w.caps = w.p_caps;
        }
        self.workspaces_generation +%= 1;
    }

    fn onFinished(_: ?*anyopaque, _: *Proxy) callconv(.c) void {}

    const manager_listener = [_]?*const anyopaque{ @ptrCast(&onGroup), @ptrCast(&onWorkspace), @ptrCast(&onManagerDone), @ptrCast(&onFinished) };

    fn onWsId(_: ?*anyopaque, _: *Proxy, _: [*:0]const u8) callconv(.c) void {}

    fn onWsName(data: ?*anyopaque, _: *Proxy, name: [*:0]const u8) callconv(.c) void {
        const w: *Workspace = @ptrCast(@alignCast(data.?));
        copy(&w.p_name, &w.p_name_len, name);
    }

    fn onWsCoordinates(_: ?*anyopaque, _: *Proxy, _: *const WlArray) callconv(.c) void {}

    fn onWsState(data: ?*anyopaque, _: *Proxy, state: u32) callconv(.c) void {
        const w: *Workspace = @ptrCast(@alignCast(data.?));
        w.p_state = state;
    }

    fn onWsCaps(data: ?*anyopaque, _: *Proxy, caps: u32) callconv(.c) void {
        const w: *Workspace = @ptrCast(@alignCast(data.?));
        w.p_caps = caps;
    }

    fn onWsRemoved(data: ?*anyopaque, _: *Proxy) callconv(.c) void {
        const w: *Workspace = @ptrCast(@alignCast(data.?));
        const self = w.desktop;
        var no_args = [_]WlArgument{};
        _ = self.lib.marshal(w.proxy, WS_DESTROY, null, 1, wl.WL_MARSHAL_FLAG_DESTROY, &no_args);
        for (self.workspaces.items, 0..) |x, i| if (x == w) {
            _ = self.workspaces.orderedRemove(i);
            break;
        };
        self.alloc.destroy(w);
        self.workspaces_generation +%= 1;
    }

    const workspace_listener = [_]?*const anyopaque{
        @ptrCast(&onWsId), @ptrCast(&onWsName), @ptrCast(&onWsCoordinates), @ptrCast(&onWsState), @ptrCast(&onWsCaps), @ptrCast(&onWsRemoved),
    };

    fn copy(dst: *[64]u8, len: *u32, src: [*:0]const u8) void {
        const s = std.mem.span(src);
        const n: usize = @min(s.len, dst.len);
        @memcpy(dst[0..n], s[0..n]);
        len.* = @intCast(n);
    }
};

test "listener tables match the protocol event counts" {
    try std.testing.expectEqual(@as(usize, 6), Desktop.output_listener.len);
    try std.testing.expectEqual(@as(usize, 6), Desktop.workspace_listener.len);
    try std.testing.expectEqual(@as(usize, 4), Desktop.manager_listener.len);
    try std.testing.expectEqualStrings("activate", std.mem.span(workspace_requests[WS_ACTIVATE].name));
}
