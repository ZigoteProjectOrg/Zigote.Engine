//! Window chrome — `zigote_window_chrome_*` C-ABI exports.
//!
//! Per-window titlebar/decoration styling for the in-app titlebar looks: macOS "unified"
//! (native traffic lights over a full-size content view — src/platform/macos_window_chrome.m)
//! and client-side decorations (borderless window; the app draws Adwaita-style buttons). The
//! app declares its draggable titlebar rects here and an SDL hit-test callback turns them into
//! OS window drags; CSD windows also get resize edges from the same callback. All exports take
//! the SDL window id (the `ZgEvent.window_id` domain, same as the file-dialog parenting) and
//! are main-thread only. Design record: docs/file-dialogs.md (windowed dialogs section).

const std = @import("std");
const builtin = @import("builtin");
const sdl3 = @import("sdl3");

const is_macos = builtin.os.tag == .macos;

extern fn zigote_macwin_set_unified(nswindow: ?*anyopaque, enabled: i32) void;
extern fn zigote_macwin_get_unified(nswindow: ?*anyopaque) i32;

/// Diagnostic: report the actually-applied chrome for a window. -3 = not macOS, -2 = unknown
/// SDL window id, -1 = no NSWindow behind it, 0 = system chrome, 1 = unified titlebar live.
export fn zigote_window_chrome_probe(window_id: u32) i32 {
    if (is_macos) {
        const win = sdlWindow(window_id) orelse return -2;
        const ns = nsWindowOf(win) orelse return -1;
        return zigote_macwin_get_unified(ns);
    }
    return -3;
}

const STYLE_SYSTEM: u32 = 0;
const STYLE_MAC_UNIFIED: u32 = 1;
const STYLE_BORDERLESS_CSD: u32 = 2;

const max_windows = 8;
const max_rects = 4;

const Entry = struct {
    window_id: u32 = 0, // 0 = free slot
    style: u32 = STYLE_SYSTEM,
    rect_count: u32 = 0,
    rects: [max_rects * 4]f32 = @splat(0),
};

var entries: [max_windows]Entry = @splat(.{});

fn entryFor(window_id: u32, create: bool) ?*Entry {
    if (window_id == 0) return null;
    for (&entries) |*e| {
        if (e.window_id == window_id) return e;
    }
    if (!create) return null;
    // Reclaim slots whose window is gone (closed dialog windows) before allocating.
    for (&entries) |*e| {
        if (e.window_id != 0 and sdl3.c.SDL_GetWindowFromID(e.window_id) == null) e.* = .{};
    }
    for (&entries) |*e| {
        if (e.window_id == 0) {
            e.* = .{ .window_id = window_id };
            return e;
        }
    }
    return null;
}

fn sdlWindow(window_id: u32) ?*sdl3.c.SDL_Window {
    const win = sdl3.c.SDL_GetWindowFromID(window_id);
    return if (win == null) null else win;
}

fn nsWindowOf(win: *sdl3.c.SDL_Window) ?*anyopaque {
    const props = sdl3.c.SDL_GetWindowProperties(win);
    return sdl3.c.SDL_GetPointerProperty(
        props,
        sdl3.c.SDL_PROP_WINDOW_COCOA_WINDOW_POINTER,
        null,
    );
}

/// Apply a chrome style. 0 = system decorations (restore), 1 = macOS unified titlebar (native
/// traffic lights over full-size content; macOS only — returns false elsewhere so callers fall
/// back), 2 = borderless for client-side decorations (any desktop OS; the app draws the buttons
/// and calls the minimize/maximize exports). Styles 1/2 install an SDL hit-test so the drag
/// rects (below) move the window; CSD also gets 8px resize edges. Main thread only.
export fn zigote_window_chrome_set(window_id: u32, style: u32) bool {
    const win = sdlWindow(window_id) orelse return false;
    const entry = entryFor(window_id, true) orelse return false;

    switch (style) {
        STYLE_SYSTEM => {
            if (is_macos) zigote_macwin_set_unified(nsWindowOf(win), 0);
            _ = sdl3.c.SDL_SetWindowBordered(win, true);
            _ = sdl3.c.SDL_SetWindowHitTest(win, null, null);
            entry.* = .{}; // free the slot
            return true;
        },
        STYLE_MAC_UNIFIED => {
            // Comptime-pruned so the ObjC extern is never referenced off-macOS.
            if (is_macos) zigote_macwin_set_unified(nsWindowOf(win), 1) else return false;
            // The styleMask change moves the content rect (it now spans the full frame incl.
            // the titlebar band); SDL doesn't notice on its own — re-assert the size so its
            // view/drawable re-layout against the new geometry.
            var cw: c_int = 0;
            var ch: c_int = 0;
            _ = sdl3.c.SDL_GetWindowSize(win, &cw, &ch);
            _ = sdl3.c.SDL_SetWindowSize(win, cw, ch);
        },
        STYLE_BORDERLESS_CSD => {
            _ = sdl3.c.SDL_SetWindowBordered(win, false);
        },
        else => return false,
    }

    entry.style = style;
    _ = sdl3.c.SDL_SetWindowHitTest(win, hitTest, null);
    return true;
}

/// Declare the window-relative draggable rects (logical coordinates; x,y,w,h quads, up to 4 —
/// the app's titlebar strip minus its interactive controls). Count 0 clears. The titlebar
/// widget refreshes these on every layout, so resizes stay correct.
export fn zigote_window_chrome_drag_rects(
    window_id: u32,
    rects: [*c]const f32,
    count: u32,
) void {
    const entry = entryFor(window_id, true) orelse return;
    const n = @min(count, max_rects);
    entry.rect_count = n;
    if (n > 0 and rects != null) {
        for (0..n * 4) |i| entry.rects[i] = rects[i];
    }
}

/// Minimize (CSD button action).
export fn zigote_window_chrome_minimize(window_id: u32) void {
    const win = sdlWindow(window_id) orelse return;
    _ = sdl3.c.SDL_MinimizeWindow(win);
}

/// Maximize, or restore when already maximized (CSD button action).
export fn zigote_window_chrome_toggle_maximize(window_id: u32) void {
    const win = sdlWindow(window_id) orelse return;
    const flags = sdl3.c.SDL_GetWindowFlags(win);
    if (flags & sdl3.c.SDL_WINDOW_MAXIMIZED != 0)
        _ = sdl3.c.SDL_RestoreWindow(win)
    else
        _ = sdl3.c.SDL_MaximizeWindow(win);
}

const edge = 8; // CSD resize border, logical px
const corner = 16;

fn hitTest(
    win: ?*sdl3.c.SDL_Window,
    area: [*c]const sdl3.c.SDL_Point,
    _: ?*anyopaque,
) callconv(.c) sdl3.c.SDL_HitTestResult {
    const w = win orelse return sdl3.c.SDL_HITTEST_NORMAL;
    if (area == null) return sdl3.c.SDL_HITTEST_NORMAL;
    const id = sdl3.c.SDL_GetWindowID(w);
    const entry = entryFor(id, false) orelse return sdl3.c.SDL_HITTEST_NORMAL;

    const x: f32 = @floatFromInt(area.*.x);
    const y: f32 = @floatFromInt(area.*.y);

    // Borderless windows have no OS resize borders — synthesize them at the edges first, so a
    // drag rect that touches an edge doesn't swallow the resize zone.
    if (entry.style == STYLE_BORDERLESS_CSD) {
        var ww: c_int = 0;
        var wh: c_int = 0;
        _ = sdl3.c.SDL_GetWindowSize(w, &ww, &wh);
        const width: f32 = @floatFromInt(ww);
        const height: f32 = @floatFromInt(wh);
        const l = x < edge;
        const r = x >= width - edge;
        const t = y < edge;
        const b = y >= height - edge;
        const lc = x < corner;
        const rc = x >= width - corner;
        const tc = y < corner;
        const bc = y >= height - corner;
        if (t and lc or l and tc) return sdl3.c.SDL_HITTEST_RESIZE_TOPLEFT;
        if (t and rc or r and tc) return sdl3.c.SDL_HITTEST_RESIZE_TOPRIGHT;
        if (b and lc or l and bc) return sdl3.c.SDL_HITTEST_RESIZE_BOTTOMLEFT;
        if (b and rc or r and bc) return sdl3.c.SDL_HITTEST_RESIZE_BOTTOMRIGHT;
        if (t) return sdl3.c.SDL_HITTEST_RESIZE_TOP;
        if (b) return sdl3.c.SDL_HITTEST_RESIZE_BOTTOM;
        if (l) return sdl3.c.SDL_HITTEST_RESIZE_LEFT;
        if (r) return sdl3.c.SDL_HITTEST_RESIZE_RIGHT;
    }

    for (0..entry.rect_count) |i| {
        const rx = entry.rects[i * 4];
        const ry = entry.rects[i * 4 + 1];
        const rw = entry.rects[i * 4 + 2];
        const rh = entry.rects[i * 4 + 3];
        if (x >= rx and x < rx + rw and y >= ry and y < ry + rh)
            return sdl3.c.SDL_HITTEST_DRAGGABLE;
    }

    return sdl3.c.SDL_HITTEST_NORMAL;
}
