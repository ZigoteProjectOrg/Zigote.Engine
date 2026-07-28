//! Native OS file/folder dialogs — `zigote_file_dialog_*` C-ABI exports.
//!
//! Poll-based, one request outstanding at a time. Two backends behind the same exports
//! (design record: docs/file-dialogs.md):
//! - macOS: our own NSOpenPanel/NSSavePanel implementation (src/platform/macos_file_dialog.m)
//!   — visible titles, save-name prefill, prompt labels, Format popup, hidden/create-dir flags.
//! - Windows/Linux: SDL3's dialog subsystem (IFileDialog on its own thread / xdg-desktop-portal
//!   → zenity), which covers title/accept/multi-select but ignores the hidden/create-dir flags.
//!
//! Threading contract: begin/status/result/consume are MAIN-THREAD only (SDL requires begin
//! there anyway; the C# pump polls from the UI thread). SDL may invoke the completion callback
//! on a foreign thread (e.g. the Windows dialog thread), so completion is a classic
//! single-producer publish: the callback writes `result`, then release-stores `status`; the
//! main thread acquire-loads `status` before touching `result`. No lock needed — the callback
//! never reads the request storage, and the main thread never frees it while a dialog is pending.

const std = @import("std");
const builtin = @import("builtin");
const sdl3 = @import("sdl3");

const is_macos = builtin.os.tag == .macos;

// Must match ZIG_DLG_* in macos_file_dialog.m and the C# FileDialog flag bits.
const FLAG_MANY: u32 = 1;
const FLAG_SHOW_HIDDEN: u32 = 2;
const FLAG_NO_CREATE_DIRS: u32 = 4;

const MacDlgDone = *const fn (paths_nl: [*c]const u8, outcome: i32) callconv(.c) void;
extern fn zigote_macdlg_begin(
    kind: i32,
    title: [*c]const u8,
    dir: [*c]const u8,
    name: [*c]const u8,
    filters: [*c]const u8,
    accept: [*c]const u8,
    flags: u32,
    nswindow: ?*anyopaque,
    done: MacDlgDone,
) i32;

/// libc malloc — thread-safe, because the completion callback can run off the main thread.
const alloc = std.heap.c_allocator;

const Status = enum(i32) {
    idle = 0,
    pending = 1,
    selected = 2,
    cancelled = 3,
    err = 4,
};

/// Option storage the native dialog reads until its callback has run (SDL requires the filter
/// array to stay alive that long), plus the SDL properties group backing the request.
const Request = struct {
    arena: std.heap.ArenaAllocator,
    props: ?sdl3.properties.Group,

    fn deinit(self: *Request) void {
        if (self.props) |p| p.deinit();
        self.arena.deinit();
    }
};

var status: std.atomic.Value(Status) = .init(.idle);
/// Newline-joined selected locations; written by the completion callback (then published via
/// `status`), read + freed by the main thread. Valid until consume / next begin.
var result: ?[:0]u8 = null;
/// Main-thread only.
var request: ?Request = null;

extern fn zigote_mac_trash_item(path: [*c]const u8) i32;

/// Move a file/folder to the OS trash (recoverable, unlike deletion). Native only on macOS
/// (NSFileManager); returns false elsewhere — the managed layer covers Windows (shell) and
/// Linux (XDG Trash) itself.
export fn zigote_file_trash(path: [*c]const u8) bool {
    if (is_macos) {
        if (path == null or path[0] == 0) return false;
        return zigote_mac_trash_item(path) != 0;
    }
    return false;
}

/// True when this build has a native dialog backend. Compile-time only: a Linux desktop without
/// a portal or zenity can't be detected up front — it surfaces as a done-error completion.
export fn zigote_file_dialog_supported() bool {
    return switch (builtin.os.tag) {
        .macos, .windows, .linux => true,
        else => false,
    };
}

/// Show a native dialog. kind: 0 = open file, 1 = pick folder, 2 = save file. title, directory,
/// file_name (save prefill) and accept_label (the OK button) are optional (null/empty → platform
/// default). filters is newline-separated "Name|pattern" entries in SDL pattern form ("ext1;ext2"
/// or "*"); ignored for folders. flags: 1 = multi-select, 2 = show hidden files, 4 = disallow
/// creating directories (macOS backend; the SDL path ignores 2 and 4). parent_window_id is the
/// SDL window id to parent/sheet to (0 = key window on macOS, unparented elsewhere). Returns
/// false when a request is already outstanding or the request is malformed. Main thread only.
export fn zigote_file_dialog_begin(
    kind: u32,
    title: [*c]const u8,
    directory: [*c]const u8,
    file_name: [*c]const u8,
    filters: [*c]const u8,
    accept_label: [*c]const u8,
    flags: u32,
    parent_window_id: u32,
) bool {
    if (!zigote_file_dialog_supported()) return false;
    if (kind > 2) return false;
    if (status.load(.acquire) == .pending) return false;
    clear();
    status.store(.idle, .release);

    // Pending must be visible before the dialog can complete — the callback may fire
    // synchronously inside show (error paths on some platforms).
    status.store(.pending, .release);

    if (is_macos) {
        const rc = zigote_macdlg_begin(
            @intCast(kind),
            title,
            directory,
            file_name,
            filters,
            accept_label,
            flags,
            resolveNsWindow(parent_window_id),
            &macDialogDone,
        );
        if (rc != 0) {
            status.store(.idle, .release);
            std.log.warn("zigote_file_dialog_begin: macOS backend refused the request", .{});
            return false;
        }
        return true;
    }

    return beginSdl(kind, title, directory, file_name, filters, accept_label, flags, parent_window_id);
}

/// SDL3 dialog path (Windows/Linux). Owns the option storage in `request` until consume.
fn beginSdl(
    kind: u32,
    title: [*c]const u8,
    directory: [*c]const u8,
    file_name: [*c]const u8,
    filters: [*c]const u8,
    accept_label: [*c]const u8,
    flags: u32,
    parent_window_id: u32,
) bool {
    const dialog_type: sdl3.dialog.Type = switch (kind) {
        0 => .open_file,
        1 => .open_folder,
        2 => .save_file,
        else => unreachable, // begin validated kind
    };

    var req = Request{ .arena = std.heap.ArenaAllocator.init(alloc), .props = null };
    const props = buildProps(
        req.arena.allocator(),
        dialog_type,
        title,
        directory,
        file_name,
        filters,
        accept_label,
        flags,
        parent_window_id,
    ) catch |e| {
        req.deinit();
        status.store(.idle, .release);
        std.log.warn("zigote_file_dialog_begin: bad request: {}", .{e});
        return false;
    };
    request = req;

    const group = sdl3.dialog.showWithProperties(dialog_type, void, onDone, null, props) catch |e| {
        // Properties-group creation failed before SDL_Show ran, so the callback cannot have fired.
        status.store(.idle, .release);
        clear();
        std.log.warn("zigote_file_dialog_begin: {}", .{e});
        return false;
    };
    request.?.props = group;
    return true;
}

/// NSWindow* for an SDL window id, for sheet parenting (macOS only; null → key window).
fn resolveNsWindow(window_id: u32) ?*anyopaque {
    if (!is_macos or window_id == 0) return null;
    const win = sdl3.c.SDL_GetWindowFromID(window_id) orelse return null;
    const props = sdl3.c.SDL_GetWindowProperties(win);
    return sdl3.c.SDL_GetPointerProperty(
        props,
        sdl3.c.SDL_PROP_WINDOW_COCOA_WINDOW_POINTER,
        null,
    );
}

/// Completion from macos_file_dialog.m — main thread, but publish like the SDL path anyway.
fn macDialogDone(paths_nl: [*c]const u8, outcome: i32) callconv(.c) void {
    switch (outcome) {
        0 => {
            const src: [:0]const u8 = std.mem.span(paths_nl orelse {
                status.store(.err, .release);
                return;
            });
            const buf = alloc.dupeZ(u8, src) catch {
                status.store(.err, .release);
                return;
            };
            result = buf;
            status.store(.selected, .release);
        },
        1 => status.store(.cancelled, .release),
        else => status.store(.err, .release),
    }
}

/// 0 idle · 1 pending · 2 done-selected · 3 done-cancelled · 4 done-error. Main thread only.
export fn zigote_file_dialog_status() i32 {
    return @intFromEnum(status.load(.acquire));
}

/// Newline-joined selected locations (UTF-8). Non-null only in status 2; valid until
/// zigote_file_dialog_consume or the next zigote_file_dialog_begin. Main thread only.
export fn zigote_file_dialog_result() [*c]const u8 {
    if (status.load(.acquire) != .selected) return null;
    return if (result) |r| r.ptr else null;
}

/// Release a completed request (result buffer + option storage) and return to idle. No-op while
/// a dialog is still pending — the native dialog owns the storage until then. Main thread only.
export fn zigote_file_dialog_consume() void {
    if (status.load(.acquire) == .pending) return;
    clear();
    status.store(.idle, .release);
}

fn buildProps(
    a: std.mem.Allocator,
    dialog_type: sdl3.dialog.Type,
    title: [*c]const u8,
    directory: [*c]const u8,
    file_name: [*c]const u8,
    filters: [*c]const u8,
    accept_label: [*c]const u8,
    flags: u32,
    parent_window_id: u32,
) !sdl3.dialog.Properties {
    var props = sdl3.dialog.Properties{};
    if (title != null and title[0] != 0)
        props.title = try a.dupeZ(u8, std.mem.span(title));
    if (accept_label != null and accept_label[0] != 0)
        props.accept = try a.dupeZ(u8, std.mem.span(accept_label));

    // SDL has a single "location" (folder, or folder + suggested name for save) — join the
    // split fields back together for it. The macOS backend takes them separately.
    const has_dir = directory != null and directory[0] != 0;
    const has_name = dialog_type == .save_file and file_name != null and file_name[0] != 0;
    if (has_dir and has_name) {
        props.location = try std.fs.path.joinZ(
            a,
            &.{ std.mem.span(directory), std.mem.span(file_name) },
        );
    } else if (has_dir) {
        props.location = try a.dupeZ(u8, std.mem.span(directory));
    } else if (has_name) {
        props.location = try a.dupeZ(u8, std.mem.span(file_name));
    }

    if (filters != null and filters[0] != 0 and dialog_type != .open_folder)
        props.filters = try parseFilters(a, std.mem.span(filters));
    if (flags & FLAG_MANY != 0) props.many = true;
    if (parent_window_id != 0)
        props.window = sdl3.video.Window.fromId(parent_window_id) catch null;
    return props;
}

/// Parse a newline-separated "Name|pattern" spec into SDL's filter array. Patterns pass through
/// verbatim (SDL validates the "ext1;ext2" / "*" grammar itself).
fn parseFilters(a: std.mem.Allocator, spec: []const u8) ![]sdl3.dialog.FileFilter {
    var count: usize = 0;
    var it = std.mem.splitScalar(u8, spec, '\n');
    while (it.next()) |line| {
        if (line.len != 0) count += 1;
    }
    if (count == 0) return error.EmptyFilterSpec;

    const list = try a.alloc(sdl3.dialog.FileFilter, count);
    var i: usize = 0;
    it = std.mem.splitScalar(u8, spec, '\n');
    while (it.next()) |line| {
        if (line.len == 0) continue;
        const bar = std.mem.indexOfScalar(u8, line, '|') orelse return error.BadFilterSpec;
        if (bar == 0 or bar == line.len - 1) return error.BadFilterSpec;
        list[i] = .{
            .name = try a.dupeZ(u8, line[0..bar]),
            .pattern = try a.dupeZ(u8, line[bar + 1 ..]),
        };
        i += 1;
    }
    return list;
}

/// May run on a foreign thread. Writes `result` then publishes through `status` — nothing else.
fn onDone(_: ?*void, file_list: ?[]const [*:0]const u8, _: ?usize, err: bool) void {
    if (err) {
        std.log.warn("file dialog failed: {s}", .{sdl3.c.SDL_GetError()});
        status.store(.err, .release);
        return;
    }
    const list = file_list orelse {
        status.store(.cancelled, .release);
        return;
    };
    if (list.len == 0) {
        status.store(.cancelled, .release);
        return;
    }

    var total: usize = 0;
    for (list) |p| total += std.mem.span(p).len + 1; // +1 = separator (last becomes the sentinel)
    const buf = alloc.allocSentinel(u8, total - 1, 0) catch {
        status.store(.err, .release);
        return;
    };
    var off: usize = 0;
    for (list, 0..) |p, i| {
        if (i != 0) {
            buf[off] = '\n';
            off += 1;
        }
        const s = std.mem.span(p);
        @memcpy(buf[off .. off + s.len], s);
        off += s.len;
    }
    result = buf;
    status.store(.selected, .release);
}

/// Free the result buffer + request storage. Main thread, non-pending states only.
fn clear() void {
    if (result) |r| alloc.free(r);
    result = null;
    if (request) |*req| req.deinit();
    request = null;
}

test "parseFilters splits names and patterns" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const list = try parseFilters(arena.allocator(), "Zigote Project|zigoteproj\nAll Files|*");
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expectEqualStrings("Zigote Project", std.mem.span(list[0].name));
    try std.testing.expectEqualStrings("zigoteproj", std.mem.span(list[0].pattern));
    try std.testing.expectEqualStrings("*", std.mem.span(list[1].pattern));
}

test "parseFilters rejects malformed specs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try std.testing.expectError(error.BadFilterSpec, parseFilters(arena.allocator(), "no-bar"));
    try std.testing.expectError(error.BadFilterSpec, parseFilters(arena.allocator(), "name|"));
    try std.testing.expectError(error.EmptyFilterSpec, parseFilters(arena.allocator(), "\n"));
}
