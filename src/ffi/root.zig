/// C ABI export layer for C# (ZigoteCS) integration.
///
/// C# owns the frame loop. Zig owns the window, GPU, and renderer.
/// Flow each frame (v2 render graph):
///   1. zigote_poll_events()            — SDL3 poll → ZgEvent buffer
///   2. [C# builds widget tree, layout, paint list]
///   3. zigote_begin_frame()            — store frame parameters
///   4. zigote_submit_paint_commands()  — store paint commands
///   5. zigote_submit_overlay_commands()— store overlay commands (optional)
///   6. zigote_render_frame_v2()        — execute render graph + present
///   7. zigote_end_frame()              — reset per-frame state
///
/// Init / shutdown:
///   zigote_init()   — SDL3 window + wgpu device + GpuUi
///   zigote_shutdown()
///
/// Text measurement (headless, no GPU):
///   zigote_measure_text()
const std = @import("std");
const sdl3 = @import("sdl3");
const wgpu = @import("wgpu");
const zg = @import("zigote");
// Real Jolt wrapper when `-Dphysics3d=true` (default); a no-op stub with the same public surface
// when off, so the zigote_physics_* wrappers below compile + link WITHOUT the JoltC static lib.
// Gated independently of enable_3d: game exports pass -Denable3d=false yet still need Jolt.
const physics_ffi = if (@import("build_options").enable_physics3d)
    @import("physics.zig")
else
    @import("physics_stub.zig");
const audio_ffi = @import("audio.zig");
// The `zigote_ecs_*` C-ABI exports live in ecs.zig (moved out of root.zig). Force-reference the
// module so its `export fn`s are analyzed and linked — an unreferenced `@import` can be lazily skipped.
// Gated by -Decs (default on): a pure-UI app that never touches World/ECS drops flecs entirely
// (ecs.zig isn't referenced → not compiled → flecs not linked; see build.zig).
comptime {
    if (@import("build_options").enable_ecs) _ = @import("ecs.zig");
}
// The `zigote_file_dialog_*` C-ABI exports (native OS open/save/folder dialogs over SDL3's
// dialog subsystem) live in dialogs.zig — force-reference for the same reason as ecs.zig above.
comptime {
    _ = @import("dialogs.zig");
}
// The `zigote_window_chrome_*` exports (unified/CSD titlebars, drag regions, minimize/maximize)
// live in chrome.zig.
comptime {
    _ = @import("chrome.zig");
}
// The `zigote_channel_*` exports: named message channels between the managed host and native
// platform code (a Kotlin service, an Objective-C delegate, a C++ SDK). Nothing in the engine
// calls them — they exist for the host and its platform head — so the reference is what keeps
// them in the binary. On every target, not just mobile: a static link (iOS) rejects an undeclared
// symbol even when unreachable, and desktop heads use the same bus for their own platform code.
comptime {
    _ = @import("channel.zig");
}
// The native menu bar and OS drag-out are implemented in Objective-C for macOS
// (src/platform/macos_{menu,drag}.m, compiled only there). Everywhere else this stub supplies
// the same exports as no-ops, so the FFI surface — and therefore the generated, platform-
// independent C# P/Invoke set — resolves on every target. That matters most on iOS, where the
// engine is linked statically into the app binary and an undeclared symbol is a link error even
// if unreachable.
comptime {
    if (@import("builtin").os.tag != .macos) _ = @import("desktop_shims_stub.zig");
}
const build_options = @import("build_options");
/// Gates 3D-game-only native subsystems (Assimp model import). Lean 2D-only builds
/// (`-Denable3d=false`) compile these out and don't link Assimp — see build.zig.
const enable_3d = build_options.enable_3d;
// Real importer when 3D is on; a no-op stub (no Assimp dependency) when off, so the
// model-import FFI below compiles and links either way without per-call guards.
// `.zmesh` parsing is NOT gated — it lives in engine/resources/zmesh_format.zig because
// game exports build with `-Denable3d=false` yet still load pre-baked `.zmesh` assets.
const zmesh_format = zg.resources.zmesh_format;
const assimp_loader = if (enable_3d) @import("assimp_loader.zig") else struct {
    pub fn importModelJson(path_z: [*:0]const u8, cache_dir_z: [*:0]const u8) ?[*:0]u8 {
        _ = path_z;
        _ = cache_dir_z;
        return null;
    }
};
const wgpu_renderer = zg.renderer;
const text_mod = zg.text_style;
const zigimg = @import("zigimg");
const GpuUi = wgpu_renderer.wgpu.GpuUi;
const backend_mod = wgpu_renderer.backend;
const gpu_select = wgpu_renderer.gpu_select;
const WgpuBackend = wgpu_renderer.wgpu_backend.WgpuBackend;

const render_mod = zg.render;
const TransientPool = render_mod.TransientPool;
const RenderGraph = render_mod.RenderGraph;
const RenderSettings = render_mod.RenderSettings;
const wgpu_blur = wgpu_renderer.wgpu_blur;

const webp = @cImport({
    @cInclude("webp/decode.h");
    @cInclude("webp/types.h");
});
const builtin = @import("builtin");

// ── Native crash diagnostics ──────────────────────────────────────────────────
//
// On a fatal signal, print the signal name + native backtrace to stderr, then hand
// the signal back to whoever owned it before us. Chaining is not optional: the .NET
// host runtime owns SIGSEGV/SIGABRT for its own machinery (null-reference translation,
// stack-overflow probes) and replacing its handler outright would turn every managed
// NullReferenceException into a process kill. Desktop glibc/macOS only — that's where
// engine crashes get debugged and where execinfo.h exists (bionic pre-33 and musl lack it).
const crash_diagnostics = builtin.os.tag == .macos or
    (builtin.os.tag == .linux and builtin.abi.isGnu());

const execinfo = if (crash_diagnostics) @cImport({
    @cInclude("execinfo.h");
}) else struct {};

const fatal_sigs = [_]std.posix.SIG{ .ILL, .TRAP, .ABRT, .BUS, .FPE, .SEGV };
var prev_sigactions: [fatal_sigs.len]std.posix.Sigaction = undefined;
var crash_handler_installed = false;

fn nativeCrashSignalHandler(sig: std.posix.SIG, info: *const std.posix.siginfo_t, uctx: ?*anyopaque) callconv(.c) void {
    const stderr_fd = std.posix.STDERR_FILENO;

    const sig_name: []const u8 = switch (sig) {
        .ILL => "SIGILL (illegal instruction, exit code 132)",
        .TRAP => "SIGTRAP (trace/breakpoint trap, exit code 133)",
        .ABRT => "SIGABRT (abort, exit code 134)",
        .BUS => "SIGBUS (bus error, exit code 135)",
        .FPE => "SIGFPE (arithmetic exception, exit code 136)",
        .SEGV => "SIGSEGV (segmentation fault, exit code 139)",
        else => "unknown crash signal",
    };
    const addr: usize = switch (builtin.os.tag) {
        .linux => @intFromPtr(info.fields.sigfault.addr),
        .macos => @intFromPtr(info.addr),
        else => 0,
    };

    var buf: [256]u8 = undefined;
    const msg = std.fmt.bufPrint(
        &buf,
        "\n[Zigote::FATAL] native crash: signal {d} {s}, fault address 0x{x}\n--- native callstack ---\n",
        .{ @intFromEnum(sig), sig_name, addr },
    ) catch "\n[Zigote::FATAL] native crash\n";
    _ = std.c.write(stderr_fd, msg.ptr, msg.len);

    var frames: [64]?*anyopaque = undefined;
    const count = execinfo.backtrace(@ptrCast(&frames), frames.len);
    if (count > 0) execinfo.backtrace_symbols_fd(@ptrCast(&frames), count, stderr_fd);

    // Hand the signal back: permanently restore the previous action, then either invoke
    // it with the original fault context (the .NET runtime needs siginfo/ucontext intact
    // to resume or translate) or re-raise so the default action sets the real exit code.
    // ponytail: one-shot — after the first chained signal our reporter stays detached;
    // re-arming safely needs Breakpad-style managed-fault filtering.
    const idx = std.mem.indexOfScalar(std.posix.SIG, &fatal_sigs, sig) orelse return;
    const prev = prev_sigactions[idx];
    std.posix.sigaction(sig, &prev, null);
    if (prev.flags & std.posix.SA.SIGINFO != 0) {
        if (prev.handler.sigaction) |h| h(sig, info, uctx);
    } else if (prev.handler.handler) |h| {
        if (h != std.posix.SIG.IGN.?) h(sig);
    } else {
        // SIG_DFL: the signal is blocked while we run, so this delivers on return.
        std.posix.raise(sig) catch {};
    }
}

fn installCrashHandler() void {
    if (!crash_diagnostics) return;
    if (crash_handler_installed) return;
    crash_handler_installed = true;

    // Warm up the unwinder outside signal context: backtrace() lazily loads libgcc
    // (malloc, dlopen) on first call, which is not signal-safe.
    var warmup: [1]?*anyopaque = undefined;
    _ = execinfo.backtrace(@ptrCast(&warmup), 1);

    const sa = std.posix.Sigaction{
        .handler = .{ .sigaction = nativeCrashSignalHandler },
        .mask = std.posix.sigemptyset(),
        .flags = std.posix.SA.SIGINFO,
    };
    for (fatal_sigs, 0..) |sig, i| {
        std.posix.sigaction(sig, &sa, &prev_sigactions[i]);
    }
}

var log_callback: ?*const fn (i32, [*c]const u8) callconv(.c) void = null;

/// Android throws away stdout/stderr, so anything the engine (or a Rust panic inside
/// wgpu-native) writes there is invisible — including the message that explains a fatal error.
/// Everything therefore also goes to logcat under the "zigote" tag.
const android_log = if (@import("builtin").abi.isAndroid()) struct {
    extern fn __android_log_write(prio: c_int, tag: [*:0]const u8, text: [*:0]const u8) c_int;

    fn write(level: std.log.Level, msg: [*:0]const u8) void {
        // Android priorities: 3 DEBUG, 4 INFO, 5 WARN, 6 ERROR.
        const prio: c_int = switch (level) {
            .debug => 3,
            .info => 4,
            .warn => 5,
            .err => 6,
        };
        _ = __android_log_write(prio, "zigote", msg);
    }
} else struct {
    fn write(level: std.log.Level, msg: [*:0]const u8) void {
        _ = level;
        _ = msg;
    }
};

pub fn myLogFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.EnumLiteral),
    comptime format: []const u8,
    args: anytype,
) void {
    _ = scope;
    var buf: [4096]u8 = undefined;
    const msg = std.fmt.bufPrintZ(&buf, format, args) catch return;
    android_log.write(level, msg.ptr);
    if (log_callback) |cb| cb(@intFromEnum(level), msg.ptr);
}

/// Forward wgpu-native's own diagnostics into the same sink. Without this the only trace of a
/// surface/adapter failure on Android is an abort with no message.
fn wgpuLogToEngine(level: wgpu.LogLevel, message: wgpu.StringView, userdata: ?*anyopaque) callconv(.c) void {
    _ = userdata;
    const text = message.toSlice() orelse return;
    var buf: [4096]u8 = undefined;
    const msg = std.fmt.bufPrintZ(&buf, "wgpu: {s}", .{text}) catch return;
    const lvl: std.log.Level = switch (level) {
        .@"error" => .err,
        .warn => .warn,
        .info => .info,
        else => .debug,
    };
    android_log.write(lvl, msg.ptr);
    if (log_callback) |cb| cb(@intFromEnum(lvl), msg.ptr);
}

pub const std_options: std.Options = .{
    .logFn = myLogFn,
};

// ── Wire types ────────────────────────────────────────────────────────────────

/// Discriminant values for ZgPaintCommand.kind.
pub const CMD_RECT: u8 = 0;
pub const CMD_BORDER: u8 = 1;
pub const CMD_TEXT: u8 = 2;
pub const CMD_IMAGE: u8 = 3;
pub const CMD_CLIP_START: u8 = 4;
pub const CMD_CLIP_END: u8 = 5;
pub const CMD_PUSH_OPACITY: u8 = 6;
pub const CMD_POP_OPACITY: u8 = 7;
pub const CMD_SHADOW: u8 = 8;
pub const CMD_LIQUID_GLASS: u8 = 9;
pub const CMD_SHADER_EFFECT: u8 = 10;
pub const CMD_TEXT_LAYOUT: u8 = 11;
pub const CMD_GLYPH_RUN: u8 = 12;
pub const CMD_RENDER_TEXTURE_BEGIN: u8 = 13;
pub const CMD_RENDER_TEXTURE_END: u8 = 14;
pub const CMD_BLUR: u8 = 15;
pub const CMD_BEZIER: u8 = 16;
pub const CMD_POLYGON: u8 = 17;
// 2-D transform stack: push a 2×3 affine (x' = a·x + c·y + tx; y' = b·x + d·y + ty) that
// applies to every subsequent command until the matching pop. Coefficients ride existing
// float slots (a=rect_x b=rect_y c=rect_w d=rect_h tx=radius ty=border_width) — no new
// ZgPaintCommand fields, so the ABI stays at version 9 (same pattern as CMD_POLYGON).
pub const CMD_TRANSFORM_PUSH: u8 = 18;
pub const CMD_TRANSFORM_POP: u8 = 19;

/// Glyph quad for CMD_GLYPH_RUN — matches C# ZgGlyphQuad (32 bytes).
pub const ZgGlyphRunQuad = extern struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    u0: f32,
    v0: f32,
    u1: f32,
    v1: f32,
};

/// Flat C-ABI paint command. Layout must match ZgStructs.cs ZgPaintCommand.
/// Total size: 112 bytes on 64-bit.
/// Fields are ordered large→small (8-byte pointers first, then f32/u32, then the small ints) so the
/// struct packs with a single 3-byte hole instead of the ~11 padding bytes the natural declaration
/// order used to force — 120→112 B, ~8 B saved on every one of the hundreds–thousands of commands a
/// painted frame streams to fillPaintList. All fields keep their meaning; Zig reads them by name, so
/// the reorder is transparent to the renderer. The comptime block below pins every offset — a drift
/// on either side of the ABI now fails the Zig build (C# side is pinned by AbiLayoutTests).
pub const ZgPaintCommand = extern struct {
    kind: u8, // offset   0
    font_style: u8, // offset   1  (0=normal, 1=italic)
    font_weight: u16, // offset   2  (100..900)
    has_cache_key: u8, // offset   4
    pad0: [3]u8 = .{0} ** 3,
    text_ptr: [*c]const u8, // offset  8  (also the GlyphRunQuad array pointer for CMD_GLYPH_RUN)
    pixels_ptr: [*c]const u8, // offset 16  (image pixels / font-family bytes / polygon points)
    rect_x: f32, // offset  24
    rect_y: f32, // offset  28
    rect_w: f32, // offset  32
    rect_h: f32, // offset  36
    color_r: f32, // offset  40
    color_g: f32, // offset  44
    color_b: f32, // offset  48
    color_a: f32, // offset  52
    radius: f32, // offset  56  (aliased: image u0 / shader id via @bitCast)
    border_width: f32, // offset  60  (image v0)
    baseline_x: f32, // offset  64  (image u1)
    baseline_y: f32, // offset  68  (image v1)
    font_size: f32, // offset  72
    line_height: f32, // offset  76
    letter_spacing: f32, // offset  80
    word_spacing: f32, // offset  84
    img_pixel_w: u32, // offset  88
    img_pixel_h: u32, // offset  92
    cache_key_lo: u32, // offset  96
    cache_key_hi: u32, // offset 100
    text_len: u32, // offset 104
    pixels_len: u32, // offset 108
    // total: 112 bytes
};

comptime {
    std.debug.assert(@sizeOf(ZgPaintCommand) == 112);
    std.debug.assert(@offsetOf(ZgPaintCommand, "kind") == 0);
    std.debug.assert(@offsetOf(ZgPaintCommand, "font_style") == 1);
    std.debug.assert(@offsetOf(ZgPaintCommand, "font_weight") == 2);
    std.debug.assert(@offsetOf(ZgPaintCommand, "has_cache_key") == 4);
    std.debug.assert(@offsetOf(ZgPaintCommand, "text_ptr") == 8);
    std.debug.assert(@offsetOf(ZgPaintCommand, "pixels_ptr") == 16);
    std.debug.assert(@offsetOf(ZgPaintCommand, "rect_x") == 24);
    std.debug.assert(@offsetOf(ZgPaintCommand, "color_r") == 40);
    std.debug.assert(@offsetOf(ZgPaintCommand, "radius") == 56);
    std.debug.assert(@offsetOf(ZgPaintCommand, "border_width") == 60);
    std.debug.assert(@offsetOf(ZgPaintCommand, "baseline_x") == 64);
    std.debug.assert(@offsetOf(ZgPaintCommand, "baseline_y") == 68);
    std.debug.assert(@offsetOf(ZgPaintCommand, "font_size") == 72);
    std.debug.assert(@offsetOf(ZgPaintCommand, "line_height") == 76);
    std.debug.assert(@offsetOf(ZgPaintCommand, "letter_spacing") == 80);
    std.debug.assert(@offsetOf(ZgPaintCommand, "word_spacing") == 84);
    std.debug.assert(@offsetOf(ZgPaintCommand, "img_pixel_w") == 88);
    std.debug.assert(@offsetOf(ZgPaintCommand, "img_pixel_h") == 92);
    std.debug.assert(@offsetOf(ZgPaintCommand, "cache_key_lo") == 96);
    std.debug.assert(@offsetOf(ZgPaintCommand, "cache_key_hi") == 100);
    std.debug.assert(@offsetOf(ZgPaintCommand, "text_len") == 104);
    std.debug.assert(@offsetOf(ZgPaintCommand, "pixels_len") == 108);
}

/// Event kind values for ZgEvent.kind.
pub const EVT_MOUSE_MOVE: u8 = 0;
pub const EVT_MOUSE_DOWN: u8 = 1;
pub const EVT_MOUSE_UP: u8 = 2;
pub const EVT_SCROLL: u8 = 3;
pub const EVT_KEY_DOWN: u8 = 4;
pub const EVT_KEY_UP: u8 = 5;
pub const EVT_QUIT: u8 = 6;
pub const EVT_RESIZE: u8 = 7;
pub const EVT_TEXT_INPUT: u8 = 8;
pub const EVT_TEXT_EDITING: u8 = 9;
pub const EVT_WINDOW_FOCUS: u8 = 10; // button carries gained (1) / lost (0)
pub const EVT_WINDOW_CLOSE: u8 = 11; // close requested (titlebar ✕); window_id says which window
pub const EVT_SYSTEM_THEME: u8 = 12; // OS appearance changed; button carries 0 unknown / 1 light / 2 dark
// OS → app drag-and-drop. A drop of N files arrives as DROP_BEGIN, then N DROP_FILE (or DROP_TEXT),
// then DROP_COMPLETE. DROP_POSITION fires while the pointer hovers with a drag payload (x/y carry the
// window-relative position for drag-over feedback). DROP_FILE/DROP_TEXT carry their UTF-8 payload
// out-of-band in poll_text (text_off/text_len), exactly like TEXT_INPUT, plus x/y and window_id.
pub const EVT_DROP_BEGIN: u8 = 13;
pub const EVT_DROP_FILE: u8 = 14;
pub const EVT_DROP_TEXT: u8 = 15;
pub const EVT_DROP_POSITION: u8 = 16;
pub const EVT_DROP_COMPLETE: u8 = 17;
// Touchscreen fingers. Only DIRECT touch devices (screens) arrive here — trackpads are
// indirect and stay cursor/wheel input, and SDL's mouse/pen simulated-touch ids are filtered
// (pen contact is treated as a direct touch: a stylus drives the UI like a finger). x/y carry
// the window-local position in window coordinates (the same space as mouse events; SDL reports
// fingers normalized 0..1, de-normalized here against the event's window). key_scancode
// carries the finger slot — a compact id (0..MAX_TOUCH_FINGERS-1) stable while that finger
// stays down, so C# can key per-pointer state on it. scroll_x carries pressure (0..1).
// TOUCH_CANCEL ends a sequence with no logical "up": OS gesture takeover, palm rejection, or
// the app being backgrounded — the UI must abandon (not commit) whatever the finger was doing.
pub const EVT_TOUCH_DOWN: u8 = 18;
pub const EVT_TOUCH_MOVE: u8 = 19;
pub const EVT_TOUCH_UP: u8 = 20;
pub const EVT_TOUCH_CANCEL: u8 = 21;
// Mobile app lifecycle. BACKGROUND is SDL's will_enter_background — it arrives BEFORE the OS
// suspends the app, and on iOS the app must stop presenting to the drawable before returning
// to the pump (rendering while backgrounded is a watchdog kill). FOREGROUND is
// did_enter_foreground: safe to render again. LOW_MEMORY asks the app to drop caches.
// SDL's `terminating` already maps to EVT_QUIT. Note for the eventual iOS port: SDL delivers
// these synchronously via event watches at the transition moment (the poll loop may never run
// again before suspension), so the resize-style event watch must flush/stop GPU work directly;
// the polled copies here remain correct for Android and desktop.
pub const EVT_APP_BACKGROUND: u8 = 22;
pub const EVT_APP_FOREGROUND: u8 = 23;
pub const EVT_LOW_MEMORY: u8 = 24;
// The mobile on-screen keyboard appeared/disappeared. Occlusion itself needs no app work —
// the platform backends pan the view so the SDL_SetTextInputArea rect (already fed from the
// text widgets) stays visible — but the app layer wants the state for layout/scroll polish.
pub const EVT_SCREEN_KEYBOARD_SHOWN: u8 = 25;
pub const EVT_SCREEN_KEYBOARD_HIDDEN: u8 = 26;
/// The window moved to another display, or its display's scale/mode changed. The host re-queries
/// zigote_get_refresh_hz + zigote_get_scale and re-paces its frame loop (a 60 Hz and a 144 Hz panel
/// want different caps). window_id says which window; carries no other payload.
pub const EVT_DISPLAY_CHANGED: u8 = 27;

/// Simultaneous touch fingers tracked; fingers beyond this are ignored at down and never
/// surface. 10 matches the practical ceiling of phone/tablet digitizers.
pub const MAX_TOUCH_FINGERS = 10;

/// Modifier bits in ZgEvent.modifiers.
pub const MOD_SHIFT: u8 = 1;
pub const MOD_CTRL: u8 = 2;
pub const MOD_ALT: u8 = 4;
pub const MOD_GUI: u8 = 8; // ⌘ on macOS, Super/Win elsewhere — the platform "command" modifier

/// Mouse button values in ZgEvent.button.
pub const BTN_LEFT: u8 = 0;
pub const BTN_RIGHT: u8 = 1;
pub const BTN_MIDDLE: u8 = 2;

/// Flat C-ABI input event. Layout must match ZgStructs.cs ZgEvent.
/// Total size: 44 bytes. The text_input / text_editing UTF-8 payload is stored OUT OF BAND: it is
/// appended to the engine's per-poll `poll_text` buffer and the event carries only (text_off, text_len)
/// into it — so the common flood of mouse/key events (which have no text) costs 44 B, not 288 B. The
/// out-of-band buffer is unbounded, so IME pre-edit is never truncated. C# reads it via
/// zigote_poll_text_ptr right after polling (valid until the next poll; single-threaded drain-decode).
pub const ZgEvent = extern struct {
    kind: u8, // offset  0
    button: u8, // offset  1  (mouse button; for key events: 1 = OS auto-repeat)
    modifiers: u8, // offset  2
    key_char: u8, // offset  3  (ASCII, 0 if not a printable key)
    key_scancode: u32, // offset  4  (raw SDL scancode)
    x: f32, // offset  8
    y: f32, // offset 12
    scroll_x: f32, // offset 16
    scroll_y: f32, // offset 20
    resize_w: u32, // offset 24  (text_editing: IME composition start)
    resize_h: u32, // offset 28  (text_editing: IME composition length)
    text_off: u32, // offset 32  (text_input / text_editing: byte offset into poll_text)
    text_len: u32, // offset 36  (text_input / text_editing: byte length in poll_text)
    window_id: u32, // offset 40  (SDL window id; 0 = unknown → treated as the main window)
    // total: 44 bytes
};

/// Result of zigote_measure_text.
pub const ZgSize = extern struct {
    width: f32,
    height: f32,
};

/// ABI compatibility info returned by zigote_get_renderer_abi_info().
/// C# must call this at startup and verify sizes match its compile-time @sizeOf values.
pub const ZgAbiInfo = extern struct {
    abi_version: u32, // offset  0  — bump when breaking ABI changes occur
    paint_command_size: u32, // offset  4  — must equal sizeof(ZgPaintCommand) on C# side
    event_size: u32, // offset  8  — must equal sizeof(ZgEvent) on C# side
    handle_size: u32, // offset 12  — size of an opaque resource handle (usize)
    render_settings_3d_size: u32, // offset 16  — must equal sizeof(ZgRenderSettings3D) on C# side
    // total: 20 bytes
};

/// Runtime renderer capabilities returned by zigote_get_renderer_caps() AFTER init.
/// Reports the backend actually selected plus optional native features (vendor upscalers /
/// hardware ray tracing). Kept separate from ZgAbiInfo (whose 16-byte size is a fixed
/// compile-time ABI guard). Mirrors backend.Caps and C# ZgRendererCaps. Total: 12 bytes.
pub const ZgRendererCaps = extern struct {
    active_backend: u32, // offset 0 — BackendId actually in use (auto may fall back)
    upscalers: u32, // offset 4 — bitset of backend.UpscalerKind (0 = none)
    raytracing: u8, // offset 8 — hardware ray tracing available
    raytracing_from_render: u8, // offset 9 — RT usable from fragment shaders (not only compute)
    pad: [2]u8 = .{0} ** 2, // offset 10 — pad to 12 bytes
    // total: 12 bytes
};

/// Typed result code returned by fallible FFI functions.
/// Replaces raw i32 0/-1 returns so the C# side has a typed enum to check.
pub const ZgResult = enum(i32) {
    ok = 0,
    err = -1,
};

// ── Render texture ───────────────────────────────────────────────────────────

/// A GPU-side render texture. Paint commands can be routed into it via
/// CMD_RENDER_TEXTURE_BEGIN / END. The result can then be referenced as
/// an image (CMD_IMAGE with cache_key = rt.cache_key).
const RenderTextureEntry = struct {
    width: u32,
    height: u32,
    /// surface_format texture: render_attachment | texture_binding
    texture: *wgpu.Texture,
    view: *wgpu.TextureView,
    /// rgba8unorm texture for blur output: texture_binding | storage_binding (lazy)
    blur_texture: ?*wgpu.Texture = null,
    blur_view: ?*wgpu.TextureView = null,
    /// Key used in gpu_ui.image_cache (equals the RT handle)
    cache_key: u64,
    /// Paint list accumulated during fillPaintList for this RT; rendered before UIPass
    pending_paint: zg.PaintList = .{},

    fn deinit(self: *RenderTextureEntry, alloc: std.mem.Allocator) void {
        self.pending_paint.deinit(alloc);
        if (self.blur_view) |v| v.release();
        if (self.blur_texture) |t| t.release();
        self.view.release();
        self.texture.release();
    }
};

const BlurRequest = struct {
    src_handle: u64,
    sigma: f32,
};

// ── Engine state ──────────────────────────────────────────────────────────────

/// Guards the image registry against a worker thread decoding while the render thread paints.
/// A spin lock rather than a mutex: `std.Thread.Mutex` is gone in this zig version and
/// `std.Io.Mutex` wants an `Io` handle the FFI layer has no reason to hold, while every critical
/// section here is one hashmap lookup or insert. Decoding always happens outside the lock.
const SpinLock = struct {
    locked: std.atomic.Value(bool) = .init(false),

    fn lock(self: *SpinLock) void {
        var spins: u32 = 0;
        while (self.locked.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            // Image sections are a hashmap op and never get past the spins. The audio lock is also
            // held across a container-header parse on a loader thread, which is long enough that a
            // pure spin would burn a core of the frame loop waiting for it — so hand the CPU back
            // once it is clear this is not a short wait.
            spins += 1;
            if (spins < 64) {
                std.atomic.spinLoopHint();
            } else {
                std.Thread.yield() catch std.atomic.spinLoopHint();
            }
        }
    }

    fn unlock(self: *SpinLock) void {
        self.locked.store(false, .release);
    }
};

pub const LoadedImage = struct {
    width: u32,
    height: u32,
    /// The decoded RGBA copy, freed once the GPU texture exists (see drainImageUploads) — an
    /// empty slice therefore means "resident on the GPU", not "broken". Width/height stay valid
    /// either way, which is all the paint path needs after the first upload.
    pixels: []const u8,
};

/// Upper bound on sub-rectangle partial-repaint damage regions per frame. Must match
/// RepaintTracker.MaxDamageRects on the C# side; excess rects there collapse to a full repaint.
const MAX_UI_DAMAGE_RECTS = 16;

/// A font registered on the engine (the boot font + every zigote_load_font call), kept so a
/// secondary window's fresh GpuUi — whose FreeType face table starts empty — can replay them.
const LoadedFont = struct {
    name: [:0]u8,
    path: [:0]u8,
};

/// A secondary OS window. UI-only: it runs the 2D paint path (no 3D scene / render graph), sharing
/// the engine's wgpu device + queue but owning its SDL window, surface, and GpuUi instance — GpuUi
/// holds per-target state (vertex buffers, scene/backdrop textures sized to one target, glyph
/// atlas), so it cannot be shared across surfaces. The handle C# holds is this struct's address.
const SecondaryWindow = struct {
    id: u32, // SDL window id — routes events on the C# side
    window: sdl3.video.Window,
    metal_view: ?sdl3.MetalView, // macOS-only, like EngineState.metal_view
    surface: *wgpu.Surface,
    config: wgpu.SurfaceConfiguration,
    gpu_ui: GpuUi,
    paint_list: zg.PaintList,
    overlay_paint_list: zg.PaintList,
    frame_index: u32,

    fn deinit(self: *SecondaryWindow, alloc: std.mem.Allocator) void {
        self.overlay_paint_list.deinit(alloc);
        self.paint_list.deinit(alloc);
        self.gpu_ui.deinit();
        self.surface.unconfigure();
        self.surface.release();
        if (self.metal_view) |mv| mv.deinit();
        self.window.deinit();
    }
};

const EngineState = struct {
    allocator: std.mem.Allocator,
    window: sdl3.video.Window,
    metal_view: ?sdl3.MetalView, // macOS-only (CAMetalLayer backing); null on Windows/Linux
    instance: *wgpu.Instance,
    surface: *wgpu.Surface,
    adapter: *wgpu.Adapter,
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: GpuUi,
    wgpu_config: wgpu.SurfaceConfiguration,
    frame_index: u32,
    paint_list: zg.PaintList,
    overlay_paint_list: zg.PaintList,
    font_name_buf: [256]u8,
    font_name_len: usize,
    image_registry: std.AutoHashMap(u64, LoadedImage),
    /// One counter for every handle that can name a GPU texture — decoded images *and* render
    /// textures, because both end up as keys in the one image cache. See `nextGpuHandle`.
    next_gpu_handle: u64,
    /// Guards image_registry + next_gpu_handle so zigote_load_texture* can be called from a
    /// worker thread while the render thread paints. Decoding a page-sized JPEG takes tens of
    /// milliseconds — far too long to sit on the frame loop.
    image_lock: SpinLock = .{},
    /// Handles released by the app, freed at end-of-frame. Deferred because a release can arrive
    /// mid-frame (a widget disposed during layout) while the texture is still recorded in the
    /// open command encoder.
    pending_image_releases: std.ArrayListUnmanaged(u64) = .empty,
    /// Handles whose pixels were rewritten by zigote_update_texture_rgba since the last frame.
    /// Drained at the top of the next render, where the GPU texture can be written safely.
    pending_image_updates: std.ArrayListUnmanaged(u64) = .empty,
    // Out-of-band UTF-8 payload for text_input / text_editing events, appended during a poll and
    // referenced by (text_off, text_len) on each such event. Cleared (retaining capacity) at the
    // start of every poll; read by C# via zigote_poll_text_ptr before the next poll.
    poll_text: std.ArrayListUnmanaged(u8) = .empty,
    // Live-node tracking: prevents C# from dereferencing freed/stale handles.
    node_handles: std.AutoHashMap(u64, void),
    selected_node_ptr: u64,
    world: zg.World,
    // The 3D renderer, created lazily by ensure3d() on first real 3D use (render, scene resource
    // upload, sprites/particles, environment). A UI-only app never pays its 22-pipeline /
    // shadow-array / env-cubemap cost — or the cold-start Metal shader-compile spike. Never reset
    // to null before shutdown (frame passes rely on it staying alive once created).
    gpu_3d: ?*wgpu_renderer.wgpu_3d.Gpu3d,
    // Latches a failed lazy init so it is attempted (and logged) once, mirroring ensureGamepad/
    // ensureAudio. Note this shifts boot semantics: a 3D-init failure no longer aborts app launch —
    // the app runs UI-only and 3D surfaces as a stale/black viewport plus the one log line.
    gpu_3d_failed: bool = false,
    // Render settings / frustum toggle written before the renderer exists; applied at creation.
    // Reads pre-creation are served from these (they equal Gpu3d's own defaults).
    pending_settings_3d: wgpu_renderer.wgpu_3d.Settings3D = .{},
    pending_frustum_cull: bool = true,
    offscreen_3d_texture: ?*wgpu.Texture,
    offscreen_3d_view: ?*wgpu.TextureView,
    physics: ?*physics_ffi.PhysicsState,

    // ── GPU backend abstraction (RHI seam) ─────────────────────────────────────
    // The backend actually in use — always `.wgpu` today (the only implemented backend). The seam
    // is kept so a native Vulkan/D3D12 backend can be added later. `wgpu_backend` owns the
    // per-frame swapchain bookkeeping; `gpu_backend` is its vtable view (Level-1 device/frame
    // lifecycle). Both hold stable self-pointers into this (heap-allocated) state.
    backend_id: backend_mod.BackendId,
    wgpu_backend: WgpuBackend,
    gpu_backend: backend_mod.GpuBackend,

    // ── Render graph / pass model ──────────────────────────────────────────────
    render_graph: RenderGraph,
    transient_pool: TransientPool,
    render_settings: RenderSettings,
    // Transient scene-3D command encoder shared across the scene_3d graph passes
    // (begin → shadow → sky → geometry → submit); null outside that span.
    scene_enc: ?*wgpu.CommandEncoder = null,
    scene_rendered: bool = false,
    // Set by the composite pass when the swapchain was unavailable, so the frame tail
    // skips advancing frame_index / clearing the overlay (matches pre-graph behaviour).
    frame_dropped: bool = false,
    // Pending frame data accumulated between begin_frame / render_frame_v2
    pending_scene_w: u32,
    pending_scene_h: u32,
    pending_scale: f32,
    pending_dt: f32,

    // Sub-rectangle partial-repaint damage regions for the next render_frame_v2 (absolute logical px).
    // Populated by zigote_submit_frame_damage; count 0 = repaint the whole frame (full clear). Reset
    // each begin_frame so stale damage never leaks into a frame C# did not annotate.
    pending_damage: [MAX_UI_DAMAGE_RECTS]zg.Rect = undefined,
    pending_damage_count: u32 = 0,

    // ── Render textures ────────────────────────────────────────────────────────
    render_textures: std.AutoHashMap(u64, RenderTextureEntry),
    blur_requests: std.ArrayListUnmanaged(BlurRequest),
    gaussian_blur: ?wgpu_blur.GaussianBlur,

    // Opened game controllers by player slot, packed from slot 0. First query brings the
    // subsystem up and scans; SDL gamepad_added/removed events request a rescan (hotplug).
    gamepads: [max_gamepads]?sdl3.gamepad.Gamepad = @splat(null),
    gamepad_scanned: bool = false,
    gamepad_sub_ready: bool = false,
    gamepad_rescan: bool = false,

    // Host scroll orientation, learned from the last mouse-wheel event's SDL direction flag
    // (0 unknown until the first scroll, 1 normal, 2 flipped/natural). SDL exposes the OS
    // "natural scroll" setting only per wheel event, so there is no static query — we latch it
    // here and serve zigote_get_scroll_orientation from it.
    scroll_orientation: u8 = 0,

    // Active touch fingers: slot index (reported to C# in key_scancode) → the SDL
    // (touch device id, finger id) pair it stands for. SDL finger ids are u64s that may be
    // pointers or indices depending on the platform; C# gets the compact slot instead so its
    // per-pointer maps stay small and the 44-byte ZgEvent needs no u64 field. A slot lives
    // from finger_down to finger_up/finger_canceled; fingers arriving with all slots taken
    // are dropped entirely (their motion/up never surfaces either, since lookups miss).
    touch_fingers: [MAX_TOUCH_FINGERS]?TouchFingerSlot = @splat(null),

    // miniaudio software synth for UI/game sound. Opened lazily on first use (see ensureAudio); a
    // machine with no audio device leaves this null and the engine runs silently.
    audio: ?*audio_ffi.AudioState = null,
    audio_scanned: bool = false,
    // Guards `audio` and everything reachable from it. The host opens files off the UI thread (a
    // container header parse is plainly visible as a hitch at every track change), so the handle
    // table is written from a worker while the frame loop reads it — and `zigote_audio_reopen`
    // frees the whole AudioState out from under both. miniaudio's own graph is thread-safe; our
    // bookkeeping around it was not.
    // ponytail: one coarse lock per audio call, so a slow file open blocks the frame's audio calls
    // for its duration. Split into a table lock + an unlocked create if that ever shows up.
    audio_lock: SpinLock = .{},

    // ── Secondary OS windows (UI-only; see SecondaryWindow) ───────────────────
    windows: std.AutoHashMap(u64, *SecondaryWindow),
    main_window_id: u32,
    // Fonts registered so far, replayed onto every new window's GpuUi (entry 0 = the boot font).
    loaded_fonts: std.ArrayListUnmanaged(LoadedFont) = .empty,
    emoji_family: ?[:0]u8 = null,
    /// Script-fallback families, in priority order, so a window created after they were registered
    /// still gets them — same reason emoji_family is kept.
    fallback_families: std.ArrayList([:0]u8) = .empty,

    // ── Live-resize render callback ────────────────────────────────────────────
    // Invoked from the SDL event-watch (see resizeEventWatch) during a modal window-resize drag —
    // on macOS/Windows the OS runs a nested event loop while the user drags a window edge, so the C#
    // frame loop is blocked inside SDL and cannot relayout/repaint until the drag ends. The watch is
    // the one place we get control mid-drag; it reconfigures the surface and calls this back into C#
    // to relayout + paint + present a live frame. Null = disabled (window catches up only on release).
    resize_render_cb: ?*const fn (window_id: u32, width: u32, height: u32) callconv(.c) void = null,
    // Reentrancy guard: a render triggered from the watch must never re-enter the watch.
    in_resize_cb: bool = false,
    // Earliest time (SDL ns-since-init) the watch may render the next live-resize frame — see
    // resizeEventWatch. 0 = render immediately.
    next_live_resize_ns: u64 = 0,

    // ── GPU selection (see gpu_select.zig) ────────────────────────────────────
    // The adapters enumerated at init and which one the device was created on. Snapshotted rather
    // than re-enumerated on demand: the adapters we didn't pick are released right after selection,
    // and the host only needs this to show "which GPUs exist / which is in use" in settings.
    gpus: [gpu_select.max_gpus]gpu_select.GpuInfo = undefined,
    gpu_count: u32 = 0,
    active_gpu: i32 = -1,

    fn fontName(self: *const EngineState) []const u8 {
        return self.font_name_buf[0..self.font_name_len];
    }

    fn isValidNode(self: *const EngineState, handle: u64) bool {
        return handle != 0 and self.node_handles.contains(handle);
    }

    fn trackNode(self: *EngineState, handle: u64) void {
        self.node_handles.put(handle, {}) catch {
            std.log.warn("zigote: OOM tracking node handle 0x{x}", .{handle});
        };
    }

    fn untrackNode(self: *EngineState, handle: u64) void {
        _ = self.node_handles.remove(handle);
    }

    fn untrackAllNodes(self: *EngineState) void {
        self.node_handles.clearRetainingCapacity();
    }
};

/// The single live engine, for validating opaque handles at the FFI boundary. Cleared FIRST in
/// zigote_shutdown so calls still in flight on .NET worker threads (async image decodes, texture
/// releases from disposers) are rejected instead of racing the teardown — an app quit while covers
/// were still decoding used to panic inside image_registry.put on a deinited map.
var live_engine: std.atomic.Value(u64) = .init(0);

// ZIGOTE_RESIZE_TORTURE state — see zigote_poll_events. null = env not read yet.
var resize_torture: ?bool = null;
var resize_torture_tick: u32 = 0;

/// Cast an opaque C# handle back to an EngineState pointer.
/// Returns null if the handle is 0, stale, or the engine is shutting down.
inline fn stateFromHandle(handle: u64) ?*EngineState {
    if (handle == 0 or live_engine.load(.acquire) != handle) return null;
    return @ptrFromInt(handle);
}

/// Resolve a secondary-window handle, validating it against the live-window table so C# can never
/// dereference a destroyed window (same pattern as node handles).
inline fn windowFromHandle(state: *EngineState, window_handle: u64) ?*SecondaryWindow {
    if (window_handle == 0) return null;
    if (!state.windows.contains(window_handle)) return null;
    return @ptrFromInt(window_handle);
}

fn windowFromSdlId(state: *EngineState, sdl_id: u32) ?*SecondaryWindow {
    var it = state.windows.valueIterator();
    while (it.next()) |win| {
        if (win.*.id == sdl_id) return win.*;
    }
    return null;
}

fn recordLoadedFont(state: *EngineState, name: []const u8, path: []const u8) !void {
    const owned_path = try state.allocator.dupeZ(u8, path);
    errdefer state.allocator.free(owned_path);
    // Re-registering an existing family swaps its face (the runtime font-switch path) — replace
    // the recorded path so window replay loads the current face, not the whole history.
    for (state.loaded_fonts.items) |*lf| {
        if (std.mem.eql(u8, lf.name, name)) {
            state.allocator.free(lf.path);
            lf.path = owned_path;
            return;
        }
    }
    const owned_name = try state.allocator.dupeZ(u8, name);
    errdefer state.allocator.free(owned_name);
    try state.loaded_fonts.append(state.allocator, .{ .name = owned_name, .path = owned_path });
}

/// Current swapchain pixel size — reads the live wgpu surface config.
fn currentPixelSize(state: *EngineState) [2]u32 {
    return .{ state.wgpu_config.width, state.wgpu_config.height };
}

// ── Exported functions ────────────────────────────────────────────────────────

/// Toggle vsync on the swapchain (for FPS testing). enabled != 0 → fifo (vsync, capped to the
/// display refresh). enabled == 0 → uncapped: immediate if the surface supports it, else mailbox,
/// else fifo. Reconfigures the surface in place (the same path the resize handler uses).
export fn zigote_set_vsync(handle: u64, enabled: u8) void {
    const state = stateFromHandle(handle) orelse return;

    var desired: wgpu.PresentMode = .fifo; // always supported, vsync on
    if (enabled == 0) {
        var caps: wgpu.SurfaceCapabilities = undefined;
        if (state.surface.getCapabilities(state.adapter, &caps) == .success) {
            defer caps.freeMembers();
            var has_immediate = false;
            var has_mailbox = false;
            for (caps.present_modes[0..caps.present_mode_count]) |pm| {
                if (pm == .immediate) has_immediate = true;
                if (pm == .mailbox) has_mailbox = true;
            }
            desired = if (has_immediate) .immediate else if (has_mailbox) .mailbox else .fifo;
        }
    }

    if (state.wgpu_config.present_mode == desired) return;
    state.wgpu_config.present_mode = desired;
    state.surface.configure(&state.wgpu_config);
}

export fn zigote_set_log_callback(cb: *const fn (i32, [*c]const u8) callconv(.c) void) void {
    log_callback = cb;
}

/// Initialize the engine: open a native window, set up wgpu, load fonts.
///
/// Parameters:
///   out_handle     — receives the opaque engine handle; pass to all subsequent calls
///   width, height  — window size in logical pixels
///   title          — window title (UTF-8, null-terminated)
///   font_path      — path to a .ttf/.ttc font file (null → macOS default)
///   font_name      — font family name matching font_path (null → "Inter")
///   gpu_power      — which GPU to prefer on a multi-GPU machine (gpu_select.Power): 0 auto,
///                    1 performance (3D apps → discrete), 2 efficiency (2D/UI apps → integrated)
///   gpu_index      — pin a specific GPU by its index in zigote_enumerate_gpus; -1 = use gpu_power.
///                    Overridden by ZIGOTE_GPU / ZIGOTE_GPU_POWER when those are set.
///
/// Returns .ok on success, .err on failure (check stderr for details).
/// Pre-init switch for a transparent (alpha-composited) main window. Main-thread, call before
/// `zigote_init`; a window cannot change transparency after creation.
export fn zigote_set_window_transparent(enabled: bool) void {
    pending_transparent_window = enabled;
}

// glibc malloc tunables (malloc.h). Negative by design — they share a namespace with the
// M_* mallopt parameters.
const M_TRIM_THRESHOLD: c_int = -1;
const M_MMAP_THRESHOLD: c_int = -3;
const M_ARENA_MAX: c_int = -8;

extern "c" fn mallopt(param: c_int, value: c_int) c_int;

/// Teach glibc that this process frees large buffers on purpose.
///
/// Image decoding allocates multi-megabyte buffers, uses them briefly and frees them. glibc's
/// default mmap threshold is 128 KB, but it *raises* that threshold dynamically — up to 32 MB —
/// whenever it sees a large mmapped block freed, on the theory that the program will want another
/// one soon and a heap block is cheaper than a syscall. For a decode path that assumption is
/// exactly backwards: the buffers come back at wildly different sizes, so they land in the sbrk
/// heap, fragment it, and the freed space cannot be returned to the OS. Scrolling a music library
/// measured a heap that grew to 67 MB and stayed there while only 20 MB was ever in use.
///
/// Pinning the threshold disables the dynamic adjustment, so anything this size or larger is
/// mmapped and handed straight back on free. 1 MB rather than the 128 KB default so ordinary
/// per-frame allocations keep using the fast heap path and do not start paying for syscalls.
fn tuneAllocatorForLargeTransients() void {
    if (builtin.os.tag != .linux or !builtin.abi.isGnu()) return;
    _ = mallopt(M_MMAP_THRESHOLD, 1024 * 1024);
    // Also return the top of the heap more eagerly than the 128 KB default.
    _ = mallopt(M_TRIM_THRESHOLD, 4 * 1024 * 1024);
    // And cap the arena count. glibc gives each thread that contends for the heap its own arena,
    // up to eight per core — so a host that decodes images on a thread pool ends up with dozens,
    // each keeping its own high-water mark of fragmented, unreturnable space. A handful is ample
    // for a UI process, whose allocation is nothing like a server's.
    _ = mallopt(M_ARENA_MAX, 4);
}

export fn zigote_init(
    out_handle: *u64,
    width: u32,
    height: u32,
    title: [*c]const u8,
    font_path: [*c]const u8,
    font_name: [*c]const u8,
    backend: u32,
    gpu_power: u32,
    gpu_index: i32,
) ZgResult {
    installCrashHandler();
    const state = zigote_init_impl(width, height, title, font_path, font_name, backend, gpu_power, gpu_index) catch |err| {
        std.log.err("zigote_init failed: {}", .{err});
        out_handle.* = 0;
        return .err;
    };
    live_engine.store(@intFromPtr(state), .release);
    out_handle.* = @intFromPtr(state);
    return .ok;
}

/// Create a wgpu surface from the SDL window's native handle, per platform: a CAMetalLayer on macOS,
/// the Win32 HWND on Windows, and the Wayland surface (preferred) or X11 window on Linux. `metal_layer`
/// is only used on macOS (and is null elsewhere). Replaces the old macOS-only `SurfaceSourceMetalLayer`
/// path that made the engine fail with `MetalViewUnavailable` on Windows/Linux.
fn createNativeSurface(instance: *wgpu.Instance, window: sdl3.video.Window, metal_layer: ?*anyopaque) !*wgpu.Surface {
    const label = wgpu.StringView.fromSlice("zigote-ffi surface");
    const props = sdl3.c.SDL_GetWindowProperties(window.value);

    switch (@import("builtin").os.tag) {
        // iOS shares the macOS path: SDL's MetalView wraps a CAMetalLayer on both (the comment
        // at the MetalView creation site notes it exists exactly for macOS/iOS).
        .macos, .ios => {
            var src = wgpu.SurfaceSourceMetalLayer{ .layer = metal_layer orelse return error.MetalLayerUnavailable };
            return instance.createSurface(&.{ .next_in_chain = @ptrCast(&src.chain), .label = label }) orelse error.WgpuSurfaceUnavailable;
        },
        .windows => {
            const hwnd = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WIN32_HWND_POINTER, null) orelse return error.NoNativeWindowHandle;
            const hinstance = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WIN32_INSTANCE_POINTER, null) orelse return error.NoNativeWindowHandle;
            var src = wgpu.SurfaceSourceWindowsHWND{ .hinstance = hinstance, .hwnd = hwnd };
            return instance.createSurface(&.{ .next_in_chain = @ptrCast(&src.chain), .label = label }) orelse error.WgpuSurfaceUnavailable;
        },
        .linux => {
            // Android rides the linux OS tag (aarch64-linux-android): the surface wraps SDL's
            // ANativeWindow. NOTE: on Android that window handle dies on every backgrounding —
            // surface recreation on EVT_APP_FOREGROUND is part of the mobile bring-up.
            if (comptime @import("builtin").abi.isAndroid()) {
                const a_window = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, null) orelse return error.NoNativeWindowHandle;
                var src = wgpu.SurfaceSourceAndroidNativeWindow{ .window = a_window };
                return instance.createSurface(&.{ .next_in_chain = @ptrCast(&src.chain), .label = label }) orelse error.WgpuSurfaceUnavailable;
            }

            // Wayland first, then X11 — SDL exposes whichever driver is active.
            if (sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, null)) |wl_display| {
                const wl_surface = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, null) orelse return error.NoNativeWindowHandle;
                var src = wgpu.SurfaceSourceWaylandSurface{ .display = wl_display, .surface = wl_surface };
                return instance.createSurface(&.{ .next_in_chain = @ptrCast(&src.chain), .label = label }) orelse error.WgpuSurfaceUnavailable;
            }
            const x_display = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_X11_DISPLAY_POINTER, null) orelse return error.NoNativeWindowHandle;
            const x_window = sdl3.c.SDL_GetNumberProperty(props, sdl3.c.SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
            var src = wgpu.SurfaceSourceXlibWindow{ .display = x_display, .window = @intCast(x_window) };
            return instance.createSurface(&.{ .next_in_chain = @ptrCast(&src.chain), .label = label }) orelse error.WgpuSurfaceUnavailable;
        },
        else => return error.UnsupportedPlatform,
    }
}

fn zigote_init_impl(
    width: u32,
    height: u32,
    title_c: [*c]const u8,
    font_path_c: [*c]const u8,
    font_name_c: [*c]const u8,
    backend_raw: u32,
    gpu_power: u32,
    gpu_index: i32,
) !*EngineState {
    // Resolve the requested backend. Only wgpu is implemented today; native backends
    // (Vulkan/D3D12) fall back to wgpu — log when a request is downgraded so the host can surface it.
    const requested_backend = backend_mod.BackendId.fromU32(backend_raw);
    const resolved_backend = backend_mod.resolve(requested_backend);
    if (resolved_backend != requested_backend and requested_backend != .auto) {
        std.log.warn("zigote: requested backend {s} unavailable; using {s}", .{
            @tagName(requested_backend), @tagName(resolved_backend),
        });
    }
    const allocator = std.heap.c_allocator;
    tuneAllocatorForLargeTransients();

    // wgpu is the sole GPU stack: SDL is windowing/events only. Pin SDL's own GPU API off so no
    // code path (present or future) can quietly stand up a second GPU device through SDL —
    // SDL_CreateGPUDevice reads this hint and now fails fast instead of selecting Metal/Vulkan.
    // Inert today (nothing calls SDL_GPU), so failure to set it is harmless.
    sdl3.hints.set(.gpu_driver, "none") catch {};

    // Touch is first-class input: fingers surface as EVT_TOUCH_* and the C# side routes them
    // as pointers. Without pinning these off, SDL would ALSO synthesize mouse events from
    // touches (its default), so every tap would fire twice — once as touch, once as a fake
    // click. The reverse simulation (mouse → fake touch) stays off too: real mice must keep
    // hover/right-click semantics instead of masquerading as fingers.
    sdl3.hints.set(.touch_mouse_events, "0") catch {};
    // Android's back gesture/button must reach the app as a key event (AC_BACK) instead of
    // finishing the activity behind our back — the UI decides whether there is somewhere to go
    // back TO, and only closes the app when there is not.
    sdl3.hints.set(.android_trap_back_button, "1") catch {};
    sdl3.hints.set(.mouse_touch_events, "0") catch {};

    const title_slice: [:0]const u8 = if (title_c != null)
        std.mem.span(title_c)
    else
        "Zigote";

    // App identity, before SDL_Init (SDL only reads it during subsystem startup). This is what
    // labels the process in audio mixers (pipewire/pavucontrol shows the stream as "Zigote Editor"
    // instead of the bare process name) and in the macOS About box.
    // ponytail: no app identifier — that is the Wayland `app_id`, and it only buys an icon and
    // taskbar grouping if it matches an installed `.desktop` file. Pass one here when we ship one;
    // until then SDL's process-name fallback is no worse and at least matches the real binary.
    sdl3.setAppMetadata(title_slice, "0.1.0", null) catch {};

    // Only the subsystems window creation needs. The gamepad subsystem is initialized lazily on
    // first query (see ensureGamepad) — NOT here — because a connected controller can make SDL's
    // gamepad init fail (macOS GameController/HID path), and a hard `try` at the top of init would
    // take the whole window down with it. Controllers are opt-in; a missing one must never block boot.
    try sdl3.init(.{ .video = true, .events = true });

    var window = try sdl3.video.Window.init(
        title_slice,
        width,
        height,
        .{
            .resizable = true,
            .high_pixel_density = true,
            // Alpha-composited window for CSD rounded corners (see zigote_set_window_transparent).
            // Without it SDL declares the whole surface opaque and the compositor ignores alpha.
            .transparent = pending_transparent_window,
        },
    );
    errdefer window.deinit();

    // The SDL Metal view / CAMetalLayer exists only on macOS/iOS — it backs the wgpu surface on
    // both (wgpu renders through Metal under the hood). On Windows/Linux the wgpu surface comes
    // from the HWND/X11/Wayland handle instead (see createNativeSurface), so we don't create one there.
    const metal_view: ?sdl3.MetalView = if (@import("builtin").os.tag == .macos or @import("builtin").os.tag == .ios)
        (sdl3.MetalView.init(window) orelse return error.MetalViewUnavailable)
    else
        null;
    errdefer if (metal_view) |mv| mv.deinit();

    const metal_layer: ?*anyopaque = if (metal_view) |mv|
        @ptrCast(mv.getLayer() orelse return error.MetalLayerUnavailable)
    else
        null;

    // Resolve font (shared by both backends). The host normally passes an absolute path to the bundled
    // Inter (resolved from the executable dir), so this fallback is only a last resort if it doesn't —
    // a cross-platform relative path to the bundled font, NOT a macOS system font (which would fail on
    // Windows/Linux).
    const resolved_font_path: []const u8 = if (font_path_c != null)
        std.mem.span(font_path_c)
    else
        "Fonts/Inter-Regular.ttf";
    const resolved_font_name: []const u8 = if (font_name_c != null)
        std.mem.span(font_name_c)
    else
        "Inter";

    // ── wgpu device + surface ──────────────────────────────────────────────────
    // Restrict the instance to the platform's native backend(s) instead of the all-backends
    // default, so instance/adapter creation never probes or loads a GPU stack we don't use —
    // e.g. on a Mac with the Vulkan SDK installed the default would dlopen MoltenVK and stand up
    // a whole second adapter chain. Same policy as the SDL gpu_driver=none hint above.
    var instance_extras = wgpu.InstanceExtras{
        .backends = if (@import("builtin").abi.isAndroid())
            // Vulkan ONLY on Android. An ANativeWindow can be connected to exactly one graphics
            // API at a time, so leaving GL in the mask makes wgpu's GL backend call
            // eglCreateWindowSurface on the window Vulkan already owns — the connect fails
            // ("already connected to another API") and surface configuration then aborts.
            wgpu.InstanceBackends.vulkan
        else switch (@import("builtin").os.tag) {
            // iOS is Metal-only — falling into the generic arm would ask for Vulkan/GL and
            // find neither.
            .macos, .ios => wgpu.InstanceBackends.metal,
            .windows => wgpu.InstanceBackends.dx12 | wgpu.InstanceBackends.vulkan,
            // Desktop Linux (and anything else): Vulkan with the GL fallback wgpu would
            // resolve anyway.
            else => wgpu.InstanceBackends.vulkan | wgpu.InstanceBackends.gl,
        },
    };
    // Route wgpu's diagnostics into the engine log before anything can fail: on Android a fatal
    // surface/adapter error otherwise aborts with no message at all.
    wgpu.setLogCallback(wgpuLogToEngine, null);
    wgpu.setLogLevel(if (@import("builtin").abi.isAndroid()) .debug else if (@import("builtin").mode == .Debug) .info else .warn);

    std.log.info("wgpu backends mask = 0x{x} (android={})", .{ instance_extras.backends, @import("builtin").abi.isAndroid() });
    const instance_desc = (wgpu.InstanceDescriptor{}).withNativeExtras(&instance_extras);
    const instance = wgpu.Instance.create(&instance_desc) orelse return error.WgpuInstanceUnavailable;
    errdefer instance.release();

    var surface = try createNativeSurface(instance, window, metal_layer);
    errdefer surface.release();

    // Pick a GPU. On a multi-GPU machine this is where a 3D app gets the discrete card and a 2D/UI
    // app gets the integrated one (see gpu_select.zig) — and because an adapter carries its own
    // backend, choosing the GPU also chooses the graphics API. Falls back to wgpu's own pick.
    // On Android the enumeration mask (instance_extras.backends) is already Vulkan-only, and
    // gpu_select's fallback pins backend_type to match — see the note there for why GL/EGL
    // winning the adapter request aborts the process.
    const selection = try gpu_select.select(
        instance,
        surface,
        instance_extras.backends,
        gpu_select.Power.fromU32(gpu_power),
        gpu_index,
    );
    const adapter = selection.adapter;
    errdefer adapter.release();

    // wgpu-native 29.0.1 renamed the native push-constants feature to IMMEDIATES (same value
    // 0x00030001). A device must enable it before any pipeline layout declares a non-zero immediate
    // size. Our pipelines currently use immediate_size = 0, so this is not strictly required today,
    // but we enable it when the adapter offers it so a future immediates-using pipeline just works.
    // Build the required-feature list dynamically. `immediates` (see note below) is enabled when
    // offered; `texture_adapter_specific_format_features` unlocks adapter-specific texture caps —
    // notably 2× MSAA for rgba16float (only [1,4] is spec-guaranteed), which lets the 3D renderer
    // halve its multisampled targets. It only ADDS capabilities, so requiring it never restricts.
    var wanted_features: [2]wgpu.FeatureName = undefined;
    var feature_count: usize = 0;
    if (adapter.hasFeature(.immediates)) {
        wanted_features[feature_count] = .immediates;
        feature_count += 1;
    } else {
        std.log.warn("zigote: adapter lacks the immediates feature; UI pipeline may fail", .{});
    }
    const has_format_features = adapter.hasFeature(.texture_adapter_specific_format_features);
    if (has_format_features) {
        wanted_features[feature_count] = .texture_adapter_specific_format_features;
        feature_count += 1;
    }

    // Whether 2× rgba16float MSAA is actually legal here — a stricter question than whether the
    // feature above was granted; see gpu_select.allowsMsaa2 for why. One query serves two
    // decisions: the backend picks MSAA, and `adapter_type` flags a software rasterizer for the
    // environment-bake shrink further down.
    var adapter_info: wgpu.AdapterInfo = undefined;
    var adapter_backend: wgpu.BackendType = .undefined;
    var adapter_is_cpu = false;
    if (adapter.getInfo(&adapter_info) == .success) {
        defer adapter_info.freeMembers();
        adapter_backend = adapter_info.backend_type;
        adapter_is_cpu = adapter_info.adapter_type == .cpu;
    }
    // The feature only unlocks ADAPTER-SPECIFIC sample counts — it does not promise 2×. Both the
    // iOS simulator's paravirtual GPU and the Android emulator's Vulkan-over-host-Metal grant it
    // yet support only [1,4] for rgba16float (pipeline creation then hard-fails), and there is no
    // per-format count query in the C API — so mobile targets stay at the spec-guaranteed 4×.
    // iOS needs the explicit exclusion: allowsMsaa2 admits every Metal backend, and iOS is Metal.
    const mobile_target = @import("builtin").os.tag == .ios or
        @import("builtin").abi.isAndroid();
    const msaa2_supported = gpu_select.allowsMsaa2(adapter_backend, has_format_features) and
        !mobile_target;
    var device_desc = wgpu.DeviceDescriptor{
        .label = wgpu.StringView.fromSlice("zigote-ffi device"),
        .required_limits = null,
    };
    if (feature_count > 0) {
        device_desc.required_feature_count = feature_count;
        device_desc.required_features = &wanted_features;
    }
    const device_resp = adapter.requestDeviceSync(instance, &device_desc, 1_000_000);
    const device = device_resp.device orelse return error.WgpuDeviceUnavailable;
    errdefer device.release();

    // 2× rgba16float MSAA is legal here (see msaa2_supported) — lower the 3D renderer's sample
    // count before any Gpu3d is created, since its pipelines and targets both read this.
    if (msaa2_supported) wgpu_renderer.wgpu_3d.MSAA_SAMPLES = 2;

    // Software rasterizer (the Android emulator's Vulkan is SwiftShader on every host): the
    // full-size environment-IBL bake runs long enough on a CPU that the emulator's fence
    // watchdog kills the device ("Parent device is lost" on the next acquire). Shrink it to
    // something a CPU finishes comfortably; real GPUs keep full quality.
    if (adapter_is_cpu) {
        wgpu_renderer.wgpu_3d.ENV_SIZE = 64;
        std.log.warn("zigote: software GPU adapter detected — reducing environment bake to 64px", .{});
    }

    const queue = device.getQueue() orelse return error.WgpuQueueUnavailable;
    errdefer queue.release();

    var capabilities: wgpu.SurfaceCapabilities = undefined;
    if (surface.getCapabilities(adapter, &capabilities) != .success) {
        return error.WgpuSurfaceCapabilitiesUnavailable;
    }
    defer capabilities.freeMembers();

    if (capabilities.format_count == 0) return error.WgpuNoSurfaceFormats;
    const preferred_format = pickSurfaceFormat(capabilities.formats[0..capabilities.format_count]);

    const pixel_size = try window.getSizeInPixels();
    var wgpu_config = wgpu.SurfaceConfiguration{
        .device = device,
        .format = preferred_format,
        .width = @intCast(pixel_size[0]),
        .height = @intCast(pixel_size[1]),
        .present_mode = .fifo,
        .alpha_mode = pickAlphaMode(&capabilities, pending_transparent_window),
    };
    surface.configure(&wgpu_config);

    const font_asset = zg.FontAsset.fromPlatform(resolved_font_name, resolved_font_path);
    const fonts: []const zg.FontAsset = &.{font_asset};

    var gpu_ui = try GpuUi.init(
        allocator,
        device,
        preferred_format,
        fonts,
        resolved_font_name,
    );
    errdefer gpu_ui.deinit();

    // Transparent frames clear to alpha 0 so uncovered pixels (rounded-corner cutouts) show the
    // desktop through. Effective only when SDL actually granted the transparent flag AND the
    // surface composites premultiplied alpha.
    gpu_ui.transparent_clear = pending_transparent_window and
        wgpu_config.alpha_mode == .premultiplied and
        sdl3.c.SDL_GetWindowFlags(window.value) & sdl3.c.SDL_WINDOW_TRANSPARENT != 0;

    var state = try allocator.create(EngineState);
    state.* = .{
        .allocator = allocator,
        .window = window,
        .metal_view = metal_view,
        .instance = instance,
        .surface = surface,
        .adapter = adapter,
        .device = device,
        .queue = queue,
        .gpu_ui = gpu_ui,
        .wgpu_config = wgpu_config,
        .frame_index = 0,
        .paint_list = .{},
        .overlay_paint_list = .{},
        .font_name_buf = undefined,
        .font_name_len = 0,
        .image_registry = std.AutoHashMap(u64, LoadedImage).init(allocator),
        .next_gpu_handle = 1,
        .node_handles = std.AutoHashMap(u64, void).init(allocator),
        .selected_node_ptr = 0,
        .world = zg.World.init(allocator),
        .gpu_3d = null,
        .offscreen_3d_texture = null,
        .offscreen_3d_view = null,
        .physics = null,
        .backend_id = resolved_backend,
        .wgpu_backend = undefined, // set below (needs a stable &state pointer)
        .gpu_backend = undefined, // set below (vtable view of state.wgpu_backend)
        .render_graph = RenderGraph.init(allocator),
        .transient_pool = TransientPool.init(allocator, device),
        .render_settings = .{},
        .pending_scene_w = 0,
        .pending_scene_h = 0,
        .pending_scale = 1.0,
        .pending_dt = 0.0,
        .render_textures = std.AutoHashMap(u64, RenderTextureEntry).init(allocator),
        .blur_requests = .empty,
        .gaussian_blur = null,
        .windows = std.AutoHashMap(u64, *SecondaryWindow).init(allocator),
        .main_window_id = window.getId() catch 0,
        .gpus = selection.gpus,
        .gpu_count = selection.count,
        .active_gpu = selection.active,
    };

    // Record the boot font so new secondary windows replay it into their own GpuUi (whose face
    // table starts empty). Failure is non-fatal: a window would just fall back to its init font.
    recordLoadedFont(state, resolved_font_name, resolved_font_path) catch {};

    // Wire up the GPU backend abstraction now that `state` has a stable address. The wgpu
    // backend borrows the shared handles and points at the live surface config; `gpu_backend`
    // is its Level-1 vtable view. (When a native backend lands, this is where it's selected.)
    state.wgpu_backend = WgpuBackend.init(state.surface, state.device, state.queue, &state.wgpu_config);
    state.gpu_backend = state.wgpu_backend.asGpuBackend();

    // Copy font name into state buffer
    const copy_len = @min(resolved_font_name.len, state.font_name_buf.len);
    @memcpy(state.font_name_buf[0..copy_len], resolved_font_name[0..copy_len]);
    state.font_name_len = copy_len;

    // Register the frame pipeline as render-graph passes (executed in this order).
    try buildRenderGraph(state);

    // Install the live-resize event watch: SDL calls it from inside the modal window-resize loop, so
    // the UI keeps laying out + rendering while the user drags a window edge instead of freezing until
    // release (see resizeEventWatch). Failure is non-fatal — resize just catches up on release.
    // Windows/macOS only: those are the platforms whose OS runs a nested modal loop that blocks the
    // app's frame loop during the drag. On Linux (X11/Wayland) the normal poll loop keeps running and
    // already renders each size step — the watch would render a SECOND full frame per step, each
    // ending in a vsync-blocking present, halving the resize frame rate.
    if (comptime builtin.os.tag == .windows or builtin.os.tag == .macos) {
        if (sdl3.events.addWatch(EngineState, resizeEventWatch, state)) |_| {} else |err| {
            std.log.warn("zigote: resize event watch unavailable: {}", .{err});
        }
    }

    return state;
}

/// Rebuild the main window's wgpu surface against the CURRENT ANativeWindow.
///
/// Android destroys the window's native surface when the app is backgrounded and hands back a
/// NEW one on resume, so the surface wgpu holds is dead from that moment: presenting to it draws
/// nothing (the symptom is a black screen with only freshly-damaged regions visible). No SDL
/// event reports the loss on this path — RENDER_DEVICE_RESET is GL-only — so the
/// background/foreground transition IS the protocol. SDL only resumes the app thread once the
/// new surface is ready, so the window property already points at it by the time this runs.
fn recreateAndroidSurface(state: *EngineState) void {
    const new_surface = createNativeSurface(state.instance, state.window, null) catch |err| {
        std.log.err("android: surface recreation failed: {}", .{err});
        return;
    };

    state.surface.unconfigure();
    state.surface.release();
    state.surface = new_surface;

    // Rotation while backgrounded changes the drawable size, so re-read it rather than reusing
    // the stale extent; format/present/alpha stay as chosen at boot.
    if (state.window.getSizeInPixels()) |size| {
        state.wgpu_config.width = @max(1, @as(u32, @intCast(size[0])));
        state.wgpu_config.height = @max(1, @as(u32, @intCast(size[1])));
    } else |_| {}
    state.surface.configure(&state.wgpu_config);

    // The backend holds the surface BY VALUE; without this it keeps presenting to the dead one.
    state.wgpu_backend.surface = state.surface;
    std.log.info("android: surface recreated ({d}x{d})", .{ state.wgpu_config.width, state.wgpu_config.height });
}

/// Reconfigure the wgpu surface for whichever window changed size. The main window owns the engine's
/// wgpu_config; a secondary window owns its own. Shared by the poll path and the live-resize watch.
/// Reconfiguring is a full swapchain rebuild, and SDL delivers each size change TWICE — once to the
/// live-resize watch, then again through the poll queue (watches don't consume events). Skipping an
/// unchanged size halves the rebuilds during a drag.
fn reconfigureSurfaceForWindow(state: *EngineState, win_id: u32, new_w: u32, new_h: u32) void {
    if (win_id == state.main_window_id) {
        if (state.wgpu_config.width == new_w and state.wgpu_config.height == new_h) return;
        state.wgpu_config.width = new_w;
        state.wgpu_config.height = new_h;
        state.surface.configure(&state.wgpu_config);
    } else if (windowFromSdlId(state, win_id)) |win| {
        if (win.config.width == new_w and win.config.height == new_h) return;
        win.config.width = new_w;
        win.config.height = new_h;
        win.surface.configure(&win.config);
    }
}

/// Refresh interval of the display the given window is on, in nanoseconds. Falls back to 60 Hz when
/// SDL reports no rate (headless, or a driver that doesn't publish one).
fn refreshIntervalNs(state: *EngineState, win_id: u32) u64 {
    const hz = refreshHzForWindow(state, win_id);
    if (hz <= 0) return std.time.ns_per_s / 60;
    return @intFromFloat(@as(f64, std.time.ns_per_s) / hz);
}

/// SDL event-watch installed at init. SDL invokes this synchronously as events are pumped — crucially
/// including from *inside* the OS modal resize loop, when the app's own frame loop is blocked. On a live
/// size change we reconfigure the surface and call the registered C# callback to relayout+paint+present a
/// frame, so the UI tracks the window continuously during the drag instead of freezing until release.
/// The return value is ignored for watches (we never filter). Window events run on the main thread.
fn resizeEventWatch(userdata: ?*EngineState, event: *sdl3.events.Event) bool {
    const state = userdata orelse return true;
    switch (event.*) {
        .window_pixel_size_changed => |resized| {
            if (state.in_resize_cb) return true; // rendering must not re-enter the watch
            const cb = state.resize_render_cb orelse return true;
            // Throttle live frames to the window's display refresh. Each one ends in a .fifo
            // present, which blocks until vblank, and the OS emits size steps faster than that —
            // so rendering every step makes the window trail the cursor (worst when a 60 Hz and a
            // 144 Hz monitor are mixed and the swapchain is synced to the slower one). Skipped
            // steps cost nothing: the poll path reconfigures and repaints at the final size as
            // soon as the modal drag loop exits.
            const now = sdl3.timer.getNanosecondsSinceInit();
            if (now < state.next_live_resize_ns) return true;
            state.next_live_resize_ns = now + refreshIntervalNs(state, resized.id);
            const new_w: u32 = @max(1, @as(u32, @intCast(resized.width)));
            const new_h: u32 = @max(1, @as(u32, @intCast(resized.height)));
            reconfigureSurfaceForWindow(state, resized.id, new_w, new_h);
            state.in_resize_cb = true;
            cb(resized.id, new_w, new_h);
            state.in_resize_cb = false;
        },
        else => {},
    }
    return true;
}

/// Register (null clears) the C# callback the resize watch invokes to render live frames during a modal
/// window-resize drag. See EngineState.resize_render_cb and resizeEventWatch.
export fn zigote_set_resize_render_callback(
    handle: u64,
    cb: ?*const fn (window_id: u32, width: u32, height: u32) callconv(.c) void,
) void {
    const state = stateFromHandle(handle) orelse return;
    state.resize_render_cb = cb;
}

// System-cursor cache: SDL cursors are process-global (one active cursor, not per-window), created
// lazily on first request and kept for the process lifetime. Index = the ZgCursor id from C#.
var cursor_cache: [12]?sdl3.mouse.Cursor = [_]?sdl3.mouse.Cursor{null} ** 12;

fn systemCursorFromId(id: u32) sdl3.mouse.SystemCursor {
    return switch (id) {
        1 => .text,
        2 => .wait,
        3 => .crosshair,
        4 => .progress,
        5 => .northwest_southeast_resize,
        6 => .northeast_southwest_resize,
        7 => .east_west_resize,
        8 => .north_south_resize,
        9 => .move,
        10 => .not_allowed,
        11 => .pointer,
        else => .default, // 0 and anything unknown
    };
}

/// Set the active OS mouse cursor to a system cursor by id (mirrors the C# MouseCursor enum). Cursors
/// are created once and cached; out-of-range ids fall back to the default arrow.
export fn zigote_set_cursor(cursor_id: u32) void {
    const id: u32 = if (cursor_id < cursor_cache.len) cursor_id else 0;
    if (cursor_cache[id] == null) {
        cursor_cache[id] = sdl3.mouse.Cursor.initSystem(systemCursorFromId(id)) catch return;
    }
    if (cursor_cache[id]) |cur| sdl3.mouse.set(cur) catch {};
}

/// Capture the pointer for mouselook: hide the cursor, hold it inside the window, and report motion
/// as deltas on `scroll_x`/`scroll_y` of the move event rather than as a position.
///
/// This is what a first-person camera needs and what no amount of application-side work can fake —
/// without it the cursor reaches a window edge and the view stops turning. Enabling it flushes any
/// pending motion, so the first event after the switch is not a jump.
///
/// Returns true on success. Fails only if the platform refuses the mode.
export fn zigote_set_relative_mouse_mode(handle: u64, enabled: bool) bool {
    const state = stateFromHandle(handle) orelse return false;
    sdl3.mouse.setWindowRelativeMode(state.window, enabled) catch return false;
    return true;
}

/// Whether the pointer is currently captured for this window.
export fn zigote_get_relative_mouse_mode(handle: u64) bool {
    const state = stateFromHandle(handle) orelse return false;
    return sdl3.mouse.getWindowRelativeMode(state.window);
}

/// Move the cursor to a window-relative position. Used to park it somewhere sensible before
/// releasing capture, so a menu opens with the pointer where the player expects it.
export fn zigote_warp_mouse_in_window(handle: u64, x: f32, y: f32) void {
    const state = stateFromHandle(handle) orelse return;
    sdl3.mouse.warpInWindow(state.window, x, y);
}

/// Block the calling thread until an SDL event arrives or timeout_ms elapses.
/// After returning, call zigote_poll_events to drain the queue.
/// Used by the C# frame loop to sleep instead of spinning when the UI is idle.
export fn zigote_wait_events(timeout_ms: u32) void {
    _ = sdl3.events.waitTimeout(@intCast(timeout_ms));
}

/// The host callback zigote_run_app runs as the app's real main (single-shot; the process
/// lives inside it on iOS, so a global is fine).
var run_app_main: ?*const fn () callconv(.c) void = null;

fn runAppTrampoline(argc: c_int, argv: [*c][*c]u8) callconv(.c) c_int {
    _ = argc;
    _ = argv;
    if (run_app_main) |cb| cb();
    return 0;
}

/// Hand the process to SDL's platform main wrapper and run `main_fn` as the app's real main.
/// On iOS this calls UIApplicationMain and invokes the callback after didFinishLaunching, on
/// the main thread, with the UIKit runloop serviced from inside SDL's event pump — so the
/// managed host's classic `while (!quit) Frame()` loop keeps working. On desktop platforms
/// SDL's generic wrapper just calls the function directly, so hosts may use this entry
/// unconditionally. The callback must contain the WHOLE app lifetime (init → loop →
/// shutdown); on iOS this function never returns (SDL exits the process when main_fn does).
/// Takes a zero-arg callback (argc/argv are meaningless to a managed host) so the generated
/// C# binding stays a plain `delegate* unmanaged[Cdecl]<void>`.
export fn zigote_run_app(main_fn: ?*const fn () callconv(.c) void) i32 {
    run_app_main = main_fn;
    // Synthesized argv: UIApplicationMain and some SDL internals expect a non-null argv0.
    const argv0: [*c]u8 = @ptrCast(@constCast("zigote"));
    var argv = [_][*c]u8{ argv0, null };
    return @intCast(sdl3.c.SDL_RunApp(1, @ptrCast(&argv), &runAppTrampoline, null));
}

/// Android inverts the other way round from iOS: Java owns the entry point, and SDL's
/// `nativeRunMain` dlsyms a named C function out of the app's .so and runs it on the SDL thread
/// (already wrapped in SDL_RunApp). So the managed host does NOT call zigote_run_app there — it
/// registers its app-main here during process startup, and Java calls zigote_android_main later.
var android_main: ?*const fn () callconv(.c) void = null;

/// Register the managed app-main. Must be called before the SDL activity starts the app thread
/// (the managed Application object's startup runs first, which is what makes this ordering hold).
export fn zigote_set_android_main(main_fn: ?*const fn () callconv(.c) void) void {
    android_main = main_fn;
}

/// SDL's `nativeRunMain` entry point. Returns non-zero when no app-main was registered, which
/// surfaces as a non-zero exit code rather than a silent blank window.
export fn zigote_android_main(argc: i32, argv: usize) i32 {
    // Signature is ABI-identical to SDL_main_func (int, char**); spelled with types the C#
    // binding generator can map (it does not know c_int or [*c][*c]u8).
    _ = argc;
    _ = argv;
    installAndroidStderrForwarder();
    const cb = android_main orelse return 1;
    cb();
    return 0;
}

/// Main-window safe-area insets in window coordinates, written as [left, top, right, bottom]
/// into `insets` (4 floats). The safe area is the region free of OS obstructions — notches,
/// rounded corners, home indicators, TV overscan. All-zero on desktop and on any query
/// failure, so callers can apply the insets unconditionally.
export fn zigote_get_safe_area(handle: u64, insets: [*c]f32) void {
    if (insets == null) return;
    insets[0] = 0;
    insets[1] = 0;
    insets[2] = 0;
    insets[3] = 0;
    const state = stateFromHandle(handle) orelse return;
    const safe = state.window.getSafeArea() catch return;
    const w, const h = state.window.getSize() catch return;
    insets[0] = @floatFromInt(@max(0, safe.x));
    insets[1] = @floatFromInt(@max(0, safe.y));
    insets[2] = @floatFromInt(@max(0, @as(i64, @intCast(w)) - safe.x - safe.w));
    insets[3] = @floatFromInt(@max(0, @as(i64, @intCast(h)) - safe.y - safe.h));
}

/// Rust panics inside wgpu-native print their explanation to raw STDERR, which Android discards —
/// leaving a SIGABRT with no message as the only trace of a fatal GPU error. Redirect fds 1/2
/// into a pipe drained onto logcat so that text survives. Android-only, installed once before
/// the app body runs.
fn installAndroidStderrForwarder() void {
    if (comptime !@import("builtin").abi.isAndroid()) return;
    const linux = std.os.linux;
    var fds: [2]i32 = undefined;
    if (linux.pipe2(&fds, .{}) != 0) return;
    // dup3, not dup2. Android's seccomp filter allows the legacy dup2 syscall on arm64 but NOT on
    // x86_64, where the process is killed outright with SIGSYS ("seccomp prevented call to
    // disallowed x86_64 system call 33") — inside this function, before a single line of app code
    // runs. dup3 is the modern spelling of the same call and is permitted on both, so the emulator
    // and a real device take the same path. The zero flag makes it behave exactly like dup2.
    _ = linux.dup3(fds[1], 1, 0);
    _ = linux.dup3(fds[1], 2, 0);
    _ = linux.close(fds[1]);
    const t = std.Thread.spawn(.{}, drainStderrPipe, .{fds[0]}) catch return;
    t.detach();
}

fn drainStderrPipe(fd: i32) void {
    const linux = std.os.linux;
    var buf: [4096]u8 = undefined;
    var line: [4096]u8 = undefined;
    var len: usize = 0;
    while (true) {
        const n = linux.read(fd, &buf, buf.len);
        if (n == 0 or n > buf.len) return; // 0 = EOF; > len = negative errno bit-cast
        for (buf[0..n]) |ch| {
            if (ch == '\n' or len == line.len - 1) {
                if (len > 0) {
                    line[len] = 0;
                    android_log.write(.err, line[0..len :0].ptr);
                }
                len = 0;
                if (ch != '\n') {
                    line[len] = ch;
                    len += 1;
                }
            } else {
                line[len] = ch;
                len += 1;
            }
        }
    }
}

/// Base pointer of the out-of-band UTF-8 text buffer filled by the most recent zigote_poll_events.
/// text_input / text_editing events carry (text_off, text_len) into this buffer. Valid only until the
/// next poll; the caller must read all text payloads from the just-polled batch before polling again.
export fn zigote_poll_text_ptr(handle: u64) [*c]const u8 {
    const state = stateFromHandle(handle) orelse return null;
    return state.poll_text.items.ptr;
}

/// Poll SDL3 events into caller-provided buffer. Returns event count written.
/// C# should call this once per frame before building the widget tree.
export fn zigote_poll_events(handle: u64, buf: [*]ZgEvent, capacity: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    var count: u32 = 0;

    // ZIGOTE_RESIZE_TORTURE=1: drive a continuous sawtooth of window resizes from inside the poll,
    // so a real app exercises its full resize path (relayout, reactive rebuilds, swapchain churn)
    // unattended. Debug/repro hook only — costs one getenv on the first poll when unset.
    if (resize_torture == null)
        resize_torture = std.c.getenv("ZIGOTE_RESIZE_TORTURE") != null;
    if (resize_torture == true) {
        resize_torture_tick +%= 1;
        if (resize_torture_tick % 2 == 0) {
            // 520..919 wide: sweeps across BOTH adaptive breakpoints (600 and 840 logical px), so
            // every pass exercises the size-class swap path, not just intra-class relayout.
            const step: i32 = @intCast((resize_torture_tick / 2) % 400);
            _ = sdl3.c.SDL_SetWindowSize(state.window.value, 520 + step, 500 + @mod(step * 3, 300));
        }
    }

    // Reset the out-of-band text buffer for this poll batch (retains capacity across polls).
    state.poll_text.clearRetainingCapacity();

    // Check capacity before popping: an event popped with the buffer already full would be
    // silently dropped instead of staying queued for the next poll.
    while (count < capacity) {
        const event = sdl3.events.poll() orelse break;

        var zge = std.mem.zeroes(ZgEvent);

        switch (event) {
            .quit, .terminating => {
                zge.kind = EVT_QUIT;
                buf[count] = zge;
                count += 1;
            },
            .mouse_motion => |m| {
                zge.kind = EVT_MOUSE_MOVE;
                zge.x = m.x;
                zge.y = m.y;
                // A move event carries no scroll, so those two slots carry the frame's relative
                // motion instead — the same trick the key events use for the auto-repeat flag, and it
                // keeps ZgEvent's size (and therefore the ABI guard) unchanged.
                //
                // This is the only motion a captured pointer produces: with relative mode on, SDL
                // stops the cursor moving and `x`/`y` stop being meaningful, so mouselook has to read
                // the deltas rather than differencing positions.
                zge.scroll_x = m.x_rel;
                zge.scroll_y = m.y_rel;
                zge.window_id = m.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .mouse_button_down => |btn| {
                zge.kind = EVT_MOUSE_DOWN;
                zge.x = btn.x;
                zge.y = btn.y;
                zge.button = switch (btn.button) {
                    .right => BTN_RIGHT,
                    .middle => BTN_MIDDLE,
                    else => BTN_LEFT,
                };
                zge.window_id = btn.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .mouse_button_up => |btn| {
                zge.kind = EVT_MOUSE_UP;
                zge.x = btn.x;
                zge.y = btn.y;
                zge.button = switch (btn.button) {
                    .right => BTN_RIGHT,
                    .middle => BTN_MIDDLE,
                    else => BTN_LEFT,
                };
                zge.window_id = btn.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .mouse_wheel => |wheel| {
                zge.kind = EVT_SCROLL;
                var sx = wheel.scroll_x;
                var sy = wheel.scroll_y;
                state.scroll_orientation = if (wheel.direction == .flipped) 2 else 1;
                if (wheel.direction == .flipped) {
                    sx = -sx;
                    sy = -sy;
                }
                zge.scroll_x = sx;
                zge.scroll_y = sy;
                zge.window_id = wheel.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .key_down, .key_up => |ke| {
                zge.kind = if (ke.down) EVT_KEY_DOWN else EVT_KEY_UP;
                // Key events don't use the mouse `button` byte, so reuse it for the OS auto-repeat flag.
                zge.button = if (ke.repeat) 1 else 0;
                const mod = ke.mod;
                if (mod.left_shift or mod.right_shift) zge.modifiers |= MOD_SHIFT;
                if (mod.left_control or mod.right_control) zge.modifiers |= MOD_CTRL;
                if (mod.left_alt or mod.right_alt) zge.modifiers |= MOD_ALT;
                if (mod.left_gui or mod.right_gui) zge.modifiers |= MOD_GUI;
                if (ke.key) |k| {
                    const int_val = @intFromEnum(k);
                    if (int_val >= 32 and int_val <= 126) {
                        zge.key_char = @intCast(int_val);
                    }
                }
                zge.key_scancode = if (ke.scancode) |sc| @intFromEnum(sc) else 0;
                zge.window_id = ke.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .text_input => |ti| {
                zge.kind = EVT_TEXT_INPUT;
                zge.text_off = @intCast(state.poll_text.items.len);
                state.poll_text.appendSlice(state.allocator, ti.text) catch {};
                zge.text_len = @as(u32, @intCast(state.poll_text.items.len)) - zge.text_off;
                zge.window_id = ti.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .text_editing => |editing| {
                zge.kind = EVT_TEXT_EDITING;
                zge.resize_w = @intCast(editing.start orelse 0);
                zge.resize_h = @intCast(editing.length orelse 0);
                zge.text_off = @intCast(state.poll_text.items.len);
                state.poll_text.appendSlice(state.allocator, editing.text) catch {};
                zge.text_len = @as(u32, @intCast(state.poll_text.items.len)) - zge.text_off;
                zge.window_id = editing.window_id orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .window_pixel_size_changed => |resized| {
                const new_w: u32 = @max(1, @as(u32, @intCast(resized.width)));
                const new_h: u32 = @max(1, @as(u32, @intCast(resized.height)));
                // Route the reconfigure to whichever window resized: the main window owns the
                // engine's wgpu_config; a secondary window owns its own surface config.
                reconfigureSurfaceForWindow(state, resized.id, new_w, new_h);

                zge.kind = EVT_RESIZE;
                zge.resize_w = new_w;
                zge.resize_h = new_h;
                zge.window_id = resized.id;
                buf[count] = zge;
                count += 1;
            },
            .window_exposed => |we| {
                if (emitWindowRefresh(state, we.id, &zge)) {
                    buf[count] = zge;
                    count += 1;
                }
            },
            .window_shown => |w| {
                if (emitWindowRefresh(state, w.id, &zge)) {
                    buf[count] = zge;
                    count += 1;
                }
            },
            .window_restored => |w| {
                // Android does NOT send will_enter_background/did_enter_foreground: the app
                // lifecycle arrives as window minimize/restore. This is also where the surface
                // must be rebuilt — the ANativeWindow the app was rendering into died when it
                // was backgrounded, and presenting to it silently draws nothing (the symptom is
                // a black screen showing only whatever repainted since).
                if (comptime @import("builtin").abi.isAndroid()) {
                    recreateAndroidSurface(state);
                    zge.kind = EVT_APP_FOREGROUND;
                    buf[count] = zge;
                    count += 1;
                    if (count >= capacity) break;
                    zge = std.mem.zeroes(ZgEvent);
                }
                if (emitWindowRefresh(state, w.id, &zge)) {
                    buf[count] = zge;
                    count += 1;
                }
            },
            .window_minimized => {
                if (comptime @import("builtin").abi.isAndroid()) {
                    zge.kind = EVT_APP_BACKGROUND;
                    buf[count] = zge;
                    count += 1;
                }
            },
            .window_focus_gained => |w| {
                zge.kind = EVT_WINDOW_FOCUS;
                zge.button = 1;
                zge.window_id = w.id;
                buf[count] = zge;
                count += 1;
            },
            .window_focus_lost => |w| {
                zge.kind = EVT_WINDOW_FOCUS;
                zge.button = 0;
                zge.window_id = w.id;
                buf[count] = zge;
                count += 1;
            },
            .window_close_requested => |w| {
                zge.kind = EVT_WINDOW_CLOSE;
                zge.window_id = w.id;
                buf[count] = zge;
                count += 1;
            },
            // Dragged onto another monitor, or that monitor's mode/scale changed. Both mean the
            // refresh rate and content scale the host is pacing + laying out against may now be
            // stale, so surface one event kind for either.
            .window_display_changed => |w| {
                zge.kind = EVT_DISPLAY_CHANGED;
                zge.window_id = w.id;
                buf[count] = zge;
                count += 1;
            },
            .window_display_scale_changed => |w| {
                zge.kind = EVT_DISPLAY_CHANGED;
                zge.window_id = w.id;
                buf[count] = zge;
                count += 1;
            },
            .system_theme_changed => {
                zge.kind = EVT_SYSTEM_THEME;
                zge.button = systemThemeValue();
                buf[count] = zge;
                count += 1;
            },
            .drop_begin => |d| {
                zge.kind = EVT_DROP_BEGIN;
                zge.window_id = d.window orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .drop_position => |d| {
                zge.kind = EVT_DROP_POSITION;
                zge.x = d.x;
                zge.y = d.y;
                zge.window_id = d.window orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .drop_file => |d| {
                zge.kind = EVT_DROP_FILE;
                zge.x = d.x;
                zge.y = d.y;
                zge.text_off = @intCast(state.poll_text.items.len);
                state.poll_text.appendSlice(state.allocator, d.file_name) catch {};
                zge.text_len = @as(u32, @intCast(state.poll_text.items.len)) - zge.text_off;
                zge.window_id = d.window orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .drop_text => |d| {
                zge.kind = EVT_DROP_TEXT;
                zge.x = d.x;
                zge.y = d.y;
                zge.text_off = @intCast(state.poll_text.items.len);
                state.poll_text.appendSlice(state.allocator, d.text) catch {};
                zge.text_len = @as(u32, @intCast(state.poll_text.items.len)) - zge.text_off;
                zge.window_id = d.window orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .drop_complete => |d| {
                zge.kind = EVT_DROP_COMPLETE;
                zge.x = d.x;
                zge.y = d.y;
                zge.window_id = d.window orelse 0;
                buf[count] = zge;
                count += 1;
            },
            .finger_down => |tf| {
                if (touchIsDirect(tf.id)) {
                    if (touchSlotAcquire(state, tf.id.value, tf.finger_id.value)) |slot| {
                        fillTouchEvent(state, &zge, tf, EVT_TOUCH_DOWN, slot);
                        buf[count] = zge;
                        count += 1;
                    }
                }
            },
            .finger_motion => |tf| {
                // No touchIsDirect re-check: only direct fingers ever get a slot, so the
                // lookup itself is the filter (and skips SDL's per-event device-type query).
                if (touchSlotFind(state, tf.id.value, tf.finger_id.value)) |slot| {
                    fillTouchEvent(state, &zge, tf, EVT_TOUCH_MOVE, slot);
                    buf[count] = zge;
                    count += 1;
                }
            },
            .finger_up => |tf| {
                if (touchSlotFind(state, tf.id.value, tf.finger_id.value)) |slot| {
                    fillTouchEvent(state, &zge, tf, EVT_TOUCH_UP, slot);
                    state.touch_fingers[slot] = null;
                    buf[count] = zge;
                    count += 1;
                }
            },
            .finger_canceled => |tf| {
                if (touchSlotFind(state, tf.id.value, tf.finger_id.value)) |slot| {
                    fillTouchEvent(state, &zge, tf, EVT_TOUCH_CANCEL, slot);
                    state.touch_fingers[slot] = null;
                    buf[count] = zge;
                    count += 1;
                }
            },
            .will_enter_background => {
                zge.kind = EVT_APP_BACKGROUND;
                buf[count] = zge;
                count += 1;
            },
            .did_enter_foreground => {
                // Must happen before the host resumes rendering: it is still holding the surface
                // that died when the app was backgrounded.
                if (comptime @import("builtin").abi.isAndroid()) recreateAndroidSurface(state);
                zge.kind = EVT_APP_FOREGROUND;
                buf[count] = zge;
                count += 1;
            },
            .low_memory => {
                zge.kind = EVT_LOW_MEMORY;
                buf[count] = zge;
                count += 1;
            },
            // Bare notifications (no window id in the SDL payload); window_id 0 = main window,
            // which is the only window that exists on the mobile platforms that send these.
            .screen_keyboard_shown => {
                zge.kind = EVT_SCREEN_KEYBOARD_SHOWN;
                buf[count] = zge;
                count += 1;
            },
            .screen_keyboard_hidden => {
                zge.kind = EVT_SCREEN_KEYBOARD_HIDDEN;
                buf[count] = zge;
                count += 1;
            },
            // Controller hotplug: rescan lazily on the next gamepad query. Only meaningful once
            // the subsystem is up (before that, the first query scans anyway).
            .gamepad_added, .gamepad_removed => {
                state.gamepad_rescan = true;
            },
            else => {},
        }
    }

    return count;
}

/// Fill a refresh (EVT_RESIZE re-emit at the current size) for the given SDL window id.
/// Returns false when the id belongs to no live window (e.g. events for a just-destroyed one).
fn emitWindowRefresh(state: *EngineState, sdl_id: u32, zge: *ZgEvent) bool {
    zge.kind = EVT_RESIZE;
    zge.window_id = sdl_id;
    if (sdl_id == state.main_window_id) {
        const sz = currentPixelSize(state);
        zge.resize_w = sz[0];
        zge.resize_h = sz[1];
        return true;
    }
    if (windowFromSdlId(state, sdl_id)) |win| {
        zge.resize_w = win.config.width;
        zge.resize_h = win.config.height;
        return true;
    }
    return false;
}

/// Current OS appearance as the FFI wire value: 0 unknown, 1 light, 2 dark.
fn systemThemeValue() u8 {
    const theme = sdl3.video.getSystemTheme() orelse return 0;
    return switch (theme) {
        .light => 1,
        .dark => 2,
    };
}

/// One entry of EngineState.touch_fingers: the SDL identity of the finger occupying a slot.
const TouchFingerSlot = struct { touch_id: u64, finger_id: u64 };

/// Whether finger events from this SDL touch device should become EVT_TOUCH_* pointer input.
/// Direct devices (touchscreens) qualify; trackpads (indirect) must not — their resting
/// fingers would ghost-tap the UI, and they already speak through cursor + wheel events.
/// SDL's simulated-touch ids: mouse-simulated touches are skipped (the real mouse events
/// cover them; letting both through would double-fire), pen-simulated ones pass (a stylus
/// on a tablet screen is direct input with no other event channel here).
fn touchIsDirect(id: sdl3.touch.Id) bool {
    if (id.value == sdl3.touch.Id.mouse.value) return false;
    if (id.value == sdl3.touch.Id.pen.value) return true;
    return (id.getType() orelse return false) == .direct;
}

/// Slot already assigned to this finger, if any.
fn touchSlotFind(state: *EngineState, touch_id: u64, finger_id: u64) ?u32 {
    for (state.touch_fingers, 0..) |maybe, i| {
        const slot = maybe orelse continue;
        if (slot.touch_id == touch_id and slot.finger_id == finger_id) return @intCast(i);
    }
    return null;
}

/// Assign the lowest free slot to a new finger; null when all MAX_TOUCH_FINGERS are down.
fn touchSlotAcquire(state: *EngineState, touch_id: u64, finger_id: u64) ?u32 {
    for (&state.touch_fingers, 0..) |*maybe, i| {
        if (maybe.* == null) {
            maybe.* = .{ .touch_id = touch_id, .finger_id = finger_id };
            return @intCast(i);
        }
    }
    return null;
}

/// LOGICAL size of the window a finger event targets — the factor that turns SDL's normalized
/// 0..1 finger position into the coordinate space the UI lays out in.
///
/// Derived as pixels / display scale rather than from SDL's window size, because those two agree
/// only on desktop. On Android SDL reports the window size in PIXELS (1080 wide) while the
/// display scale is ~2.75, so using the window size directly put every touch ~2.75x too far out
/// and nothing was hittable. macOS/iOS are unaffected: there the window size already equals
/// pixels / scale.
fn touchWindowLogicalSize(state: *EngineState, sdl_id: u32) [2]f32 {
    const win = if (sdl_id != 0)
        sdl3.video.Window.fromId(sdl_id) catch state.window
    else
        state.window;
    const w, const h = win.getSizeInPixels() catch return .{ 0, 0 };
    const scale = win.getDisplayScale() catch 1.0;
    const safe_scale = if (scale > 0.0) scale else 1.0;
    return .{ @as(f32, @floatFromInt(w)) / safe_scale, @as(f32, @floatFromInt(h)) / safe_scale };
}

/// Fill the shared fields of an EVT_TOUCH_* event from an SDL finger event. `slot` was
/// resolved by the caller (acquire on down, find on move/up/cancel) so this stays a pure
/// formatter.
fn fillTouchEvent(state: *EngineState, zge: *ZgEvent, tf: anytype, kind: u8, slot: u32) void {
    const size = touchWindowLogicalSize(state, tf.window_id orelse 0);
    zge.kind = kind;
    zge.x = tf.x * size[0];
    zge.y = tf.y * size[1];
    zge.key_scancode = slot;
    zge.scroll_x = tf.pressure;
    zge.window_id = tf.window_id orelse 0;
}

// ── Lazy 3D renderer ──────────────────────────────────────────────────────────

/// Create the 3D renderer on first real 3D use (same latch pattern as ensureGamepad/ensureAudio).
/// A UI-only app never calls this, skipping Gpu3d's 22 pipelines / 14 shader modules / shadow-array
/// / env-cubemap allocations AND the cold-start Metal shader-compile transient. Returns null after
/// a latched failure (attempted + logged exactly once; the app keeps running UI-only).
/// The returned pointer is stable (heap-allocated, never freed before shutdown) — but treat it as
/// frame-local; never store it across FFI calls.
fn ensure3d(state: *EngineState) ?*wgpu_renderer.wgpu_3d.Gpu3d {
    if (state.gpu_3d) |g| return g;
    if (state.gpu_3d_failed) return null;
    std.log.info("zigote: initializing 3D renderer (first 3D use)", .{});
    const g = state.allocator.create(wgpu_renderer.wgpu_3d.Gpu3d) catch {
        state.gpu_3d_failed = true;
        return null;
    };
    // wgpu_config.format is the boot-time preferred surface format (resizes never change it), so
    // init inputs are identical to the former eager call in zigote_init_impl.
    g.* = wgpu_renderer.wgpu_3d.Gpu3d.init(state.allocator, state.device, state.queue, state.wgpu_config.format) catch |err| {
        std.log.err("zigote: 3D renderer init failed: {}", .{err});
        state.allocator.destroy(g);
        state.gpu_3d_failed = true;
        return null;
    };
    // Apply settings written before creation. env_dirty defaults to true, so pre-creation
    // sky/environment settings bake on the first render without an explicit diff.
    g.settings = state.pending_settings_3d;
    g.frustum_cull = state.pending_frustum_cull;
    state.gpu_3d = g;
    return g;
}

// ── Game controllers (SDL gamepad, up to 8 player slots) ──────────────────────

const max_gamepads = 8;

fn rescanGamepads(state: *EngineState) void {
    for (&state.gamepads) |*slot| {
        if (slot.*) |g| g.deinit();
        slot.* = null;
    }
    const pads = sdl3.gamepad.getGamepads() catch {
        std.log.warn("zigote: gamepad enumeration failed", .{});
        return;
    };
    var slot: usize = 0;
    for (pads) |id| {
        if (slot >= state.gamepads.len) break;
        state.gamepads[slot] = sdl3.gamepad.Gamepad.init(id) catch continue;
        slot += 1;
    }
    std.log.info("zigote: {d} game controller(s) connected", .{slot});
}

fn ensureGamepads(state: *EngineState) void {
    if (state.gamepad_scanned) {
        if (state.gamepad_rescan and state.gamepad_sub_ready) {
            state.gamepad_rescan = false;
            rescanGamepads(state);
        }
        return;
    }
    // Latch the attempt BEFORE touching SDL so a failure (or a one-time stall) never repeats every
    // frame: gamepad support is best-effort, and the engine must keep running without it.
    state.gamepad_scanned = true;

    // macOS controller bring-up can hang in the raw-HID joystick drivers: IOKit / HIDAPI block on a
    // synchronous device open or the Input-Monitoring permission path. Prefer the GameController
    // framework (MFi) — sanctioned, permission-free, and covers modern Xbox/PlayStation/Switch pads on
    // current macOS — and disable the raw-HID drivers. These are NORMAL priority, so the matching
    // environment variable still overrides them (e.g. SDL_JOYSTICK_HIDAPI=1) to experiment without a
    // rebuild. The log lines below pinpoint where bring-up stops if a controller is still unhappy.
    sdl3.hints.set(.joystick_mfi, "1") catch {};
    sdl3.hints.set(.joystick_iokit, "0") catch {};
    sdl3.hints.set(.joystick_hidapi, "0") catch {};

    // Bring up the gamepad subsystem here rather than at boot. It's ref-counted on top of the already
    // initialized video/events, and isolating it means a controller that SDL can't init disables
    // gamepad input instead of killing the window.
    std.log.info("zigote: initializing gamepad subsystem…", .{});
    sdl3.init(.{ .gamepad = true }) catch {
        std.log.warn("zigote: gamepad subsystem init failed; controller input disabled", .{});
        return;
    };
    state.gamepad_sub_ready = true;
    rescanGamepads(state);
}

fn gamepadAt(state: *EngineState, pad: u8) ?sdl3.gamepad.Gamepad {
    if (pad >= max_gamepads) return null;
    ensureGamepads(state);
    return state.gamepads[pad];
}

/// Number of connected (opened) game controllers, 0-8. Player slots are packed from 0.
export fn zigote_input_gamepad_count(handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    ensureGamepads(state);
    var n: u32 = 0;
    for (state.gamepads) |g| n += @intFromBool(g != null);
    return n;
}

/// 1 if the game controller in slot pad is connected (and opened), else 0.
export fn zigote_input_gamepad_connected(handle: u64, pad: u8) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    return @intFromBool(gamepadAt(state, pad) != null);
}

/// Read a controller axis for slot pad, normalised to [-1, 1] (triggers report [0, 1]). SDL axis order:
/// 0 left-X, 1 left-Y, 2 right-X, 3 right-Y, 4 left-trigger, 5 right-trigger.
export fn zigote_input_gamepad_axis(handle: u64, pad: u8, axis: u8) f32 {
    if (axis >= 6) return 0;
    const state = stateFromHandle(handle) orelse return 0;
    const g = gamepadAt(state, pad) orelse return 0;
    const ax = sdl3.gamepad.Axis.fromSdl(@intCast(axis)) orelse return 0;
    const v = @as(f32, @floatFromInt(g.getAxis(ax))) / 32767.0;
    return std.math.clamp(v, -1.0, 1.0);
}

/// 1 while a controller button is held for slot pad. SDL button order: 0 south(A), 1 east(B), 2 west(X),
/// 3 north(Y), 4 back, 5 guide, 6 start, 7 L-stick, 8 R-stick, 9 LB, 10 RB, 11-14 d-pad.
export fn zigote_input_gamepad_button(handle: u64, pad: u8, button: u8) u32 {
    if (button >= 21) return 0;
    const state = stateFromHandle(handle) orelse return 0;
    const g = gamepadAt(state, pad) orelse return 0;
    const btn = sdl3.gamepad.Button.fromSdl(@intCast(button)) orelse return 0;
    return @intFromBool(g.getButton(btn));
}

// ── Audio (miniaudio software synth) ───────────────────────────────────────────

fn ensureAudio(state: *EngineState) ?*audio_ffi.AudioState {
    if (state.audio) |a| return a;
    if (state.audio_scanned) return null;
    // The iOS SIMULATOR's AudioToolbox abort()s the process on an XPC timeout inside
    // AURemoteIO::Initialize (audio-daemon wedge; deterministic on current runtimes, and an
    // abort cannot be caught) — a silent simulator beats a dead app. Devices keep audio.
    if (comptime @import("builtin").os.tag == .ios and @import("builtin").abi == .simulator) {
        state.audio_scanned = true;
        std.log.warn("zigote: audio disabled on the iOS simulator (AudioToolbox XPC abort)", .{});
        return null;
    }
    // Latch before touching SDL so a one-time failure never retries every call.
    state.audio_scanned = true;
    std.log.info("zigote: initializing audio subsystem…", .{});
    state.audio = audio_ffi.init(state.allocator);
    return state.audio;
}

/// Fire a one-shot tone (UI click / blip / beep). waveform: 0 sine, 1 square, 2 triangle, 3 saw, 4 noise.
export fn zigote_audio_beep(handle: u64, freq: f32, duration: f32, volume: f32, waveform: u8) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return;
    audio_ffi.beep(a, freq, duration, volume, audio_ffi.Waveform.fromU8(waveform));
}

/// Set a sustained tone on a channel (held until changed). volume<=0 or freq<=0 silences the channel.
export fn zigote_audio_voice(handle: u64, channel: u32, freq: f32, volume: f32, waveform: u8) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return;
    audio_ffi.setVoice(a, @intCast(channel), freq, volume, audio_ffi.Waveform.fromU8(waveform));
}

/// Silence every voice (one-shots, sustained channels, and all handle sources).
export fn zigote_audio_stop_all(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.stopAll(a);
}

/// Age + reap fire-and-forget one-shots. Call once per frame from the host loop.
export fn zigote_audio_update(handle: u64, dt: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.update(a, dt);
}

/// Set the spatial listener pose (position + forward + world-up). All sounds spatialise against it.
export fn zigote_audio_set_listener(handle: u64, px: f32, py: f32, pz: f32, fx: f32, fy: f32, fz: f32, ux: f32, uy: f32, uz: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return;
    audio_ffi.setListener(a, px, py, pz, fx, fy, fz, ux, uy, uz);
}

/// Master output volume [0,4]; 1 = unity.
export fn zigote_audio_set_master_volume(handle: u64, volume: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return;
    audio_ffi.setMasterVolume(a, volume);
}

/// Positioned procedural one-shot (spatialised + attenuated). waveform: 0 sine,1 square,2 tri,3 saw,4 noise.
export fn zigote_audio_beep_3d(handle: u64, px: f32, py: f32, pz: f32, freq: f32, duration: f32, volume: f32, waveform: u8, min_dist: f32, max_dist: f32, rolloff: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return;
    audio_ffi.beep3d(a, px, py, pz, freq, duration, volume, audio_ffi.Waveform.fromU8(waveform), min_dist, max_dist, rolloff);
}

/// Create a sustained procedural-tone source (not started). Returns a handle id (0 = failure).
export fn zigote_audio_sound_create_tone(handle: u64, freq: f32, waveform: u8) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return 0;
    return audio_ffi.createTone(a, freq, audio_ffi.Waveform.fromU8(waveform)) orelse 0;
}

/// Create a source from a decoded/streamed audio file (not started). Returns a handle id (0 = failure).
export fn zigote_audio_sound_create_file(handle: u64, path_c: [*c]const u8, streaming: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (path_c == null) return 0;
    const a = ensureAudio(state) orelse return 0;
    const path = std.mem.span(@as([*:0]const u8, @ptrCast(path_c)));
    return audio_ffi.createFile(a, path, streaming != 0) orelse 0;
}

/// Create a sound fed by `zigote_audio_stream_push` instead of by a file — network radio, or any
/// source the host holds bytes for (not started). Returns a handle id (0 = failure).
export fn zigote_audio_stream_create(handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return 0;
    return audio_ffi.createStream(a) orelse 0;
}

/// Hand encoded bytes to a stream source. Returns how many were accepted; a short count means its
/// queue is full and the caller should stop reading until it drains.
export fn zigote_audio_stream_push(handle: u64, id: u32, data: [*c]const u8, len: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (data == null or len == 0) return 0;
    const a = state.audio orelse return 0;
    return @intCast(audio_ffi.streamPush(a, id, data[0..len]));
}

/// No more bytes are coming. What is queued still plays out, then the sound reports end-of-stream.
export fn zigote_audio_stream_finish(handle: u64, id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.streamFinish(a, id);
}

/// 0 connecting, 1 playing, 2 undecodable, 3 ended.
export fn zigote_audio_stream_state(handle: u64, id: u32) u32 {
    const state = stateFromHandle(handle) orelse return 2;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = state.audio orelse return 2;
    return audio_ffi.streamState(a, id);
}

/// Decoded audio held ahead of the mixer, in seconds — what a "Buffering…" indicator shows.
export fn zigote_audio_stream_buffered(handle: u64, id: u32) f32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = state.audio orelse return 0;
    return audio_ffi.streamBuffered(a, id);
}

export fn zigote_audio_sound_play(handle: u64, id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.play(a, id);
}

export fn zigote_audio_sound_stop(handle: u64, id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.stop(a, id);
}

export fn zigote_audio_sound_destroy(handle: u64, id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.destroyHandle(a, id);
}

export fn zigote_audio_sound_set_volume(handle: u64, id: u32, volume: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setVolume(a, id, volume);
}

export fn zigote_audio_sound_set_pitch(handle: u64, id: u32, pitch: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setPitch(a, id, pitch);
}

export fn zigote_audio_sound_set_looping(handle: u64, id: u32, looping: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setLooping(a, id, looping != 0);
}

export fn zigote_audio_sound_set_spatial(handle: u64, id: u32, enabled: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setSpatial(a, id, enabled != 0);
}

export fn zigote_audio_sound_set_position(handle: u64, id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setPosition(a, id, x, y, z);
}

export fn zigote_audio_sound_set_velocity(handle: u64, id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setVelocity(a, id, x, y, z);
}

export fn zigote_audio_sound_set_attenuation(handle: u64, id: u32, min_dist: f32, max_dist: f32, rolloff: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.setAttenuation(a, id, min_dist, max_dist, rolloff);
}

/// Returns 1 while the source is playing, else 0.
export fn zigote_audio_sound_is_playing(handle: u64, id: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| return if (audio_ffi.isPlaying(a, id)) 1 else 0;
    return 0;
}

/// Create a mixer bus (miniaudio sound group). Returns a bus id (0 = failure). Buses live until the
/// engine's audio state is torn down — there is deliberately no per-bus destroy (see audio.zig).
export fn zigote_audio_group_create(handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return 0;
    return audio_ffi.groupCreate(a) orelse 0;
}

export fn zigote_audio_group_set_volume(handle: u64, group_id: u32, volume: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.groupSetVolume(a, group_id, volume);
}

export fn zigote_audio_group_set_pitch(handle: u64, group_id: u32, pitch: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.groupSetPitch(a, group_id, pitch);
}

/// Route a sound through a bus (group_id 0 = back to the master output).
export fn zigote_audio_sound_set_group(handle: u64, id: u32, group_id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.soundSetGroup(a, id, group_id);
}

// ── Device rate (high-resolution playback) ─────────────────────────────────────

/// The output device's current sample rate in Hz (0 = no audio device).
export fn zigote_audio_output_rate(handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return 0;
    return audio_ffi.outputRate(a);
}

/// Reopen the audio device at `sample_rate` Hz (0 = the device's preferred rate) and return the rate
/// actually achieved (0 = failure, sound now disabled). A host that refuses the requested rate keeps
/// the rate it already had, so the return value can differ from `sample_rate` without being an error —
/// compare it against what you asked for if the distinction matters to the caller.
///
/// The rate is fixed when a device is created, so this tears the engine down and builds a new one:
/// **every sound handle, mixer bus and equalizer chain is destroyed and their ids are invalid
/// afterwards.** The caller must recreate them — which is not as harsh as it sounds, since the
/// reason to call this is that a track with a different rate is about to be loaded anyway.
export fn zigote_audio_reopen(handle: u64, sample_rate: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    var previous_rate: u32 = 0;
    if (state.audio) |a| {
        previous_rate = audio_ffi.outputRate(a);
        audio_ffi.deinit(a);
        state.audio = null;
    }

    // Clear the "already tried and failed" latch: a different rate may well succeed where the last
    // attempt did not.
    state.audio_scanned = true;
    state.audio = audio_ffi.initWithRate(state.allocator, sample_rate);

    // The device is destroyed before the new rate is attempted, because the rate is fixed at creation
    // and a device cannot be reconfigured in place. So a refused rate used to leave NO device at all,
    // permanently and silently: nothing here retried, and the caller only learns that the rate was not
    // achieved. Any host that grants exactly one rate — WASAPI shared mode against a fixed mixer
    // format, some PipeWire and ALSA dmix configurations, Wine — went mute on the first track whose
    // rate differed, for the rest of the session. Reclaiming the rate that was already working keeps
    // playback alive; the caller sees a rate it did not ask for and resamples, which is the same
    // outcome as never having asked.
    if (state.audio == null and previous_rate != 0 and previous_rate != sample_rate)
        state.audio = audio_ffi.initWithRate(state.allocator, previous_rate);

    const a = state.audio orelse return 0;
    return audio_ffi.outputRate(a);
}

// ── Transport (media playback: seek + position) ────────────────────────────────

/// Seek a sound to an absolute position in seconds.
export fn zigote_audio_sound_seek(handle: u64, id: u32, seconds: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.seekSeconds(a, id, seconds);
}

/// Playback cursor in seconds; -1 when the source cannot report one.
export fn zigote_audio_sound_cursor(handle: u64, id: u32) f32 {
    const state = stateFromHandle(handle) orelse return -1;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| return audio_ffi.cursorSeconds(a, id);
    return -1;
}

/// Total length in seconds; -1 when unknown.
export fn zigote_audio_sound_duration(handle: u64, id: u32) f32 {
    const state = stateFromHandle(handle) orelse return -1;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| return audio_ffi.durationSeconds(a, id);
    return -1;
}

/// The source decoded past its last frame (playlist auto-advance signal). Unlike `is_playing` this
/// stays false for a sound that was merely paused.
export fn zigote_audio_sound_at_end(handle: u64, id: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| return if (audio_ffi.atEnd(a, id)) 1 else 0;
    return 0;
}

/// Start a sound at an exact point on the audio clock, `seconds_from_now` ahead of now — the
/// primitive gapless playback is built on.
export fn zigote_audio_sound_schedule_start(handle: u64, id: u32, seconds_from_now: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.scheduleStart(a, id, seconds_from_now);
}

// ── Equalizer chains ───────────────────────────────────────────────────────────

/// Create a chain of `band_count` filters (max 16), flat until configured. 0 on failure.
export fn zigote_audio_eq_create(handle: u64, band_count: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    const a = ensureAudio(state) orelse return 0;
    return audio_ffi.eqCreate(a, band_count) orelse 0;
}

/// Configure one band. kind: 0 peak, 1 low shelf, 2 high shelf. Shelves take Q (converted to the
/// RBJ slope internally), matching how AutoEq and every parametric EQ UI specify them.
export fn zigote_audio_eq_set_band(handle: u64, eq_id: u32, index: u32, kind: u8, freq_hz: f32, gain_db: f32, q: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a|
        audio_ffi.eqSetBand(a, eq_id, index, audio_ffi.BandKind.fromU8(kind), freq_hz, gain_db, q);
}

/// Bypass (0) or engage (1) the chain without losing its band settings.
export fn zigote_audio_eq_set_enabled(handle: u64, eq_id: u32, enabled: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.eqSetEnabled(a, eq_id, enabled != 0);
}

export fn zigote_audio_eq_destroy(handle: u64, eq_id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.eqDestroy(a, eq_id);
}

/// Route a sound through an equalizer chain (eq_id 0 = dry).
export fn zigote_audio_sound_set_eq(handle: u64, id: u32, eq_id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    state.audio_lock.lock();
    defer state.audio_lock.unlock();
    if (state.audio) |a| audio_ffi.soundSetEq(a, id, eq_id);
}

// ── Offline decoding ───────────────────────────────────────────────────────────

/// Decode a whole file to interleaved f32 at its native rate/channels — for callers that need the
/// samples rather than playback (waveform overviews, loudness analysis, sampler/IR loading).
/// Returns the buffer pointer as an integer (0 = failure); free it with `zigote_audio_decode_free`.
/// Needs no audio device, so it works on machines with sound disabled. Loader threads only.
export fn zigote_audio_decode_file(handle: u64, path_c: [*c]const u8, out_channels: *u32, out_sample_rate: *u32, out_frame_count: *u64) usize {
    _ = handle;
    out_channels.* = 0;
    out_sample_rate.* = 0;
    out_frame_count.* = 0;
    const path = std.mem.span(path_c);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (path.len >= buf.len) return 0;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const decoded = audio_ffi.decodeFile(buf[0..path.len :0], out_channels, out_sample_rate, out_frame_count) orelse return 0;
    return @intFromPtr(decoded);
}

export fn zigote_audio_decode_free(handle: u64, frames: usize) void {
    _ = handle;
    if (frames == 0) return;
    audio_ffi.decodeFree(@ptrFromInt(frames));
}

/// Translate raw ZgPaintCommand array into a native PaintList.
/// CMD_RENDER_TEXTURE_BEGIN/END route commands into RT sub-lists.
/// CMD_BLUR records blur requests for processing in renderFrameV2Impl.
fn fillPaintList(
    state: *EngineState,
    list: *zg.PaintList,
    commands: [*]const ZgPaintCommand,
    count: u32,
) !void {
    const alloc = state.allocator;
    list.clearRetainingCapacity(alloc);

    // Route commands into the appropriate sub-list (main or RT)
    var list_stack: [8]*zg.PaintList = undefined;
    var stack_depth: usize = 0;
    var current: *zg.PaintList = list;

    for (commands[0..count]) |*cmd| {
        switch (cmd.kind) {
            CMD_RECT => {
                try current.append(alloc, .{ .rect = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .radius = cmd.radius,
                } });
            },
            CMD_BORDER => {
                try current.append(alloc, .{ .border = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .radius = cmd.radius,
                    .width = cmd.border_width,
                } });
            },
            CMD_SHADOW => {
                try current.append(alloc, .{ .shadow = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .radius = cmd.radius,
                    .blur_radius = cmd.border_width,
                    .spread = cmd.baseline_x,
                } });
            },
            CMD_LIQUID_GLASS => {
                try current.append(alloc, .{ .liquid_glass = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .radius = cmd.radius,
                    .thickness = cmd.border_width,
                    .glow_x = cmd.baseline_x,
                    .glow_y = cmd.baseline_y,
                    .pinch = cmd.font_size,
                    // Glass never carries text metrics, so line_height is free to be the
                    // adaptive-luminance knob (see PaintList.AddLiquidGlass).
                    .adapt = cmd.line_height,
                } });
            },
            CMD_TEXT => {
                if (cmd.text_ptr == null or cmd.text_len == 0) continue;
                const text_slice = cmd.text_ptr[0..cmd.text_len];
                const fw: text_mod.FontWeight = @enumFromInt(cmd.font_weight);
                const fs: text_mod.FontStyle = if (cmd.font_style == 1) .italic else .normal;
                // Text commands never carry image pixels, so the pixels side-channel doubles as the
                // optional font-family name (UTF-8). null → renderer uses the default UI face.
                const font_family: ?[]const u8 =
                    if (cmd.pixels_ptr != null and cmd.pixels_len > 0)
                        cmd.pixels_ptr[0..cmd.pixels_len]
                    else
                        null;
                // Text shadow rides in slots CMD_TEXT never used (rect = color, radius /
                // border_width = offset, img_pixel_w = blur bitcast). Present iff alpha > 0;
                // drawn as a second blurred text run underneath the real one.
                if (cmd.rect_h > 0) {
                    try current.appendOwnedText(alloc, .{
                        .baseline_x = cmd.baseline_x + cmd.radius,
                        .baseline_y = cmd.baseline_y + cmd.border_width,
                        .text = text_slice,
                        .color = zg.geometry.Color.rgba(
                            @intFromFloat(std.math.clamp(cmd.rect_x * 255, 0, 255)),
                            @intFromFloat(std.math.clamp(cmd.rect_y * 255, 0, 255)),
                            @intFromFloat(std.math.clamp(cmd.rect_w * 255, 0, 255)),
                            @intFromFloat(std.math.clamp(cmd.rect_h * 255, 0, 255)),
                        ),
                        .size = cmd.font_size,
                        .line_height = cmd.line_height,
                        .font_family = font_family,
                        .font_weight = fw,
                        .font_style = fs,
                        .letter_spacing = cmd.letter_spacing,
                        .word_spacing = cmd.word_spacing,
                        .blur = @bitCast(cmd.img_pixel_w),
                    });
                }
                try current.appendOwnedText(alloc, .{
                    .baseline_x = cmd.baseline_x,
                    .baseline_y = cmd.baseline_y,
                    .text = text_slice,
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .size = cmd.font_size,
                    .line_height = cmd.line_height,
                    .font_family = font_family,
                    .font_weight = fw,
                    .font_style = fs,
                    .letter_spacing = cmd.letter_spacing,
                    .word_spacing = cmd.word_spacing,
                });
            },
            CMD_IMAGE => {
                const cache_key: ?u64 = if (cmd.has_cache_key != 0)
                    (@as(u64, cmd.cache_key_hi) << 32) | @as(u64, cmd.cache_key_lo)
                else
                    null;

                var pixels: []const u8 = &.{};
                var w = cmd.img_pixel_w;
                var h = cmd.img_pixel_h;

                const render3d_magic: u64 = 0x3D3D3D3D3D3D3D3D;

                if (cmd.pixels_ptr != null and cmd.pixels_len > 0) {
                    pixels = cmd.pixels_ptr[0..cmd.pixels_len];
                } else if (cache_key) |key| {
                    // Locked: a worker thread may be inserting a freshly decoded image right now,
                    // and a rehash under a reader would hand back a dangling slice. The pixels a
                    // hit yields stay valid for the frame — releases are deferred to end-of-frame,
                    // which runs on this thread.
                    state.image_lock.lock();
                    const found = state.image_registry.get(key);
                    state.image_lock.unlock();

                    if (found) |cached_img| {
                        // An empty slice means the texture is already on the GPU and the CPU copy
                        // was dropped: pass through, the renderer resolves it by cache_key.
                        pixels = cached_img.pixels;
                        w = cached_img.width;
                        h = cached_img.height;
                    } else if (key == render3d_magic) {
                        // 3D offscreen texture — lives in gpu_ui.image_cache, not
                        // image_registry. Pass through with empty pixels; the wgpu
                        // renderer will find the GPU texture by its magic cache key.
                    } else if (state.render_textures.contains(key)) {
                        // RT texture — will be registered in image_cache by renderFrameV2Impl.
                        // Pass through with empty pixels; renderer finds it via cache_key.
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }

                try current.append(alloc, .{ .image = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .width = w,
                    .height = h,
                    .pixels = pixels,
                    .cache_key = cache_key,
                    .u0 = cmd.radius,
                    .v0 = cmd.border_width,
                    .u1 = cmd.baseline_x,
                    .v1 = cmd.baseline_y,
                } });
            },
            CMD_CLIP_START => {
                // radius rides the shared Radius field (offset 56) — always 0 before rounded
                // clips existed, so old paint streams parse identically. No ABI change.
                try current.append(alloc, .{ .clip_start = .{
                    .rect = .{
                        .x = cmd.rect_x,
                        .y = cmd.rect_y,
                        .width = cmd.rect_w,
                        .height = cmd.rect_h,
                    },
                    .radius = @max(0, cmd.radius),
                } });
            },
            CMD_CLIP_END => {
                try current.append(alloc, .clip_end);
            },
            CMD_PUSH_OPACITY => {
                try current.append(alloc, .{ .push_opacity = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .alpha = cmd.color_a,
                } });
            },
            CMD_POP_OPACITY => {
                try current.append(alloc, .pop_opacity);
            },
            CMD_SHADER_EFFECT => {
                const shader_id: u32 = @bitCast(cmd.radius);
                // Optional @group(1) texture: same registry resolve as CMD_IMAGE, so the
                // renderer can materialize the GPU entry even for a texture never drawn as an
                // image (a LUT, a mask). Same locking rationale as there.
                var image_key: ?u64 = null;
                var image_pixels: []const u8 = &.{};
                var image_w: u32 = 0;
                var image_h: u32 = 0;
                if (cmd.has_cache_key != 0) {
                    const key = (@as(u64, cmd.cache_key_hi) << 32) | @as(u64, cmd.cache_key_lo);
                    image_key = key;
                    state.image_lock.lock();
                    const found = state.image_registry.get(key);
                    state.image_lock.unlock();
                    if (found) |img| {
                        image_pixels = img.pixels;
                        image_w = img.width;
                        image_h = img.height;
                    }
                }
                try current.append(alloc, .{ .shader_effect = .{
                    .bounds = .{ .x = cmd.rect_x, .y = cmd.rect_y, .width = cmd.rect_w, .height = cmd.rect_h },
                    .shader_id = shader_id,
                    .params = .{
                        cmd.color_r,      cmd.color_g,    cmd.color_b,    cmd.color_a,
                        cmd.border_width, cmd.baseline_x, cmd.baseline_y, cmd.font_size,
                    },
                    .image_key = image_key,
                    .image_width = image_w,
                    .image_height = image_h,
                    .image_pixels = image_pixels,
                } });
            },
            CMD_TEXT_LAYOUT => {
                const handle = (@as(u64, cmd.cache_key_hi) << 32) | @as(u64, cmd.cache_key_lo);
                if (handle == 0) continue;
                try current.append(alloc, .{ .text_layout = .{
                    .handle = handle,
                    .draw_x = cmd.baseline_x,
                    .draw_y = cmd.baseline_y,
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                } });
            },
            CMD_GLYPH_RUN => {
                const handle = (@as(u64, cmd.cache_key_hi) << 32) | @as(u64, cmd.cache_key_lo);
                if (cmd.text_ptr == null or cmd.text_len == 0) continue;
                const quad_count = @as(usize, cmd.text_len);
                const quads_src: [*]const zg.paint.GlyphRunQuad = @ptrCast(@alignCast(cmd.text_ptr));
                try current.appendOwnedGlyphRun(alloc, .{
                    .atlas_handle = handle,
                    .tint = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .quads = quads_src[0..quad_count],
                });
            },
            CMD_RENDER_TEXTURE_BEGIN => {
                if (cmd.has_cache_key == 0) continue;
                const rt_handle = (@as(u64, cmd.cache_key_hi) << 32) | @as(u64, cmd.cache_key_lo);
                if (state.render_textures.getPtr(rt_handle)) |rt| {
                    rt.pending_paint.clearRetainingCapacity(alloc);
                    if (stack_depth < list_stack.len) {
                        list_stack[stack_depth] = current;
                        stack_depth += 1;
                        current = &rt.pending_paint;
                    } else {
                        std.log.warn("zigote: RT nesting too deep, clamping at {d}", .{list_stack.len});
                    }
                } else {
                    std.log.warn("zigote: CMD_RENDER_TEXTURE_BEGIN: unknown RT handle 0x{x}", .{rt_handle});
                }
            },
            CMD_RENDER_TEXTURE_END => {
                if (stack_depth > 0) {
                    stack_depth -= 1;
                    current = list_stack[stack_depth];
                }
            },
            CMD_BLUR => {
                if (cmd.has_cache_key == 0) continue;
                const src_handle = (@as(u64, cmd.cache_key_hi) << 32) | @as(u64, cmd.cache_key_lo);
                try state.blur_requests.append(alloc, .{
                    .src_handle = src_handle,
                    .sigma = cmd.radius,
                });
            },
            CMD_BEZIER => {
                // Four control points packed into rect / radius / baseline slots; width in font_size.
                try current.append(alloc, .{ .bezier = .{
                    .x0 = cmd.rect_x,
                    .y0 = cmd.rect_y,
                    .x1 = cmd.rect_w,
                    .y1 = cmd.rect_h,
                    .x2 = cmd.radius,
                    .y2 = cmd.border_width,
                    .x3 = cmd.baseline_x,
                    .y3 = cmd.baseline_y,
                    .color = zg.geometry.Color.rgba(
                        @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                        @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                    ),
                    .width = cmd.font_size,
                } });
            },
            CMD_POLYGON => {
                // Points (x,y f32 pairs) ride the pixels side-channel; need at least 3 (24 bytes).
                if (cmd.pixels_ptr == null or cmd.pixels_len < 24) continue;
                const bytes = cmd.pixels_ptr[0..cmd.pixels_len];
                try current.appendOwnedPolygon(alloc, bytes, zg.geometry.Color.rgba(
                    @intFromFloat(std.math.clamp(cmd.color_r * 255, 0, 255)),
                    @intFromFloat(std.math.clamp(cmd.color_g * 255, 0, 255)),
                    @intFromFloat(std.math.clamp(cmd.color_b * 255, 0, 255)),
                    @intFromFloat(std.math.clamp(cmd.color_a * 255, 0, 255)),
                ));
            },
            CMD_TRANSFORM_PUSH => {
                try current.append(alloc, .{ .transform_push = .{
                    .a = cmd.rect_x,
                    .b = cmd.rect_y,
                    .c = cmd.rect_w,
                    .d = cmd.rect_h,
                    .tx = cmd.radius,
                    .ty = cmd.border_width,
                } });
            },
            CMD_TRANSFORM_POP => {
                try current.append(alloc, .transform_pop);
            },
            else => {},
        }
    }
}

export fn zigote_register_shader(handle: u64, id: u32, wgsl_ptr: [*]const u8, wgsl_len: usize) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    const wgsl = wgsl_ptr[0..wgsl_len];
    wgpu_renderer.wgpu.GpuUi.registerShader(&state.gpu_ui, state.device, id, wgsl) catch |err| {
        std.log.err("zigote_register_shader failed: {}", .{err});
        return .err;
    };
    return .ok;
}

/// Measure text using the heuristic text model (no GPU required).
///
/// Returns (0,0) if the engine is not initialized.
export fn zigote_measure_text(
    handle: u64,
    text: [*c]const u8,
    text_len: u32,
    font_size: f32,
    max_width: f32,
    font_weight: u16,
    font_style: u8,
    letter_spacing: f32,
    word_spacing: f32,
    font_family: [*c]const u8,
    font_family_len: u32,
) ZgSize {
    if (text == null or text_len == 0) return .{ .width = 0, .height = 0 };

    const text_slice = text[0..text_len];
    const fw: text_mod.FontWeight = @enumFromInt(font_weight);
    const fs: text_mod.FontStyle = if (font_style == 1) .italic else .normal;
    const family: ?[]const u8 =
        if (font_family != null and font_family_len > 0) font_family[0..font_family_len] else null;

    // Unbounded (single-line) requests are shaped accurately via HarfBuzz in the requested face, so
    // font-family'd labels (Material Icons, Iosevka) size correctly. Wrapped requests (max_width > 0)
    // and the headless path fall through to the coarse per-character estimate below, which
    // ignores the family (the accurate path has no word-wrap yet).
    if (max_width <= 0) {
        if (stateFromHandle(handle)) |state| {
            const tr: ?*@TypeOf(state.gpu_ui.text) = &state.gpu_ui.text;
            if (tr) |t| {
                var out_w: f32 = 0;
                var out_h: f32 = 0;
                if (t.measureText(state.allocator, .{
                    .baseline_x = 0,
                    .baseline_y = 0,
                    .text = text_slice,
                    .color = .{ .r = 255, .g = 255, .b = 255, .a = 255 },
                    .size = font_size,
                    .line_height = 0,
                    .font_family = family,
                    .font_weight = fw,
                    .font_style = fs,
                    .letter_spacing = letter_spacing,
                    .word_spacing = word_spacing,
                }, &out_w, &out_h)) |_| {
                    return .{ .width = out_w, .height = out_h };
                } else |_| {
                    // Shaping failed (no face loaded) — fall through to the estimate.
                }
            }
        }
    }

    const style = text_mod.TextStyle{
        .size = font_size,
        .font_family = family,
        .font_weight = fw,
        .font_style = fs,
    };
    const metrics = text_mod.measure(text_slice, style, max_width);
    return .{ .width = metrics.size.width, .height = metrics.size.height };
}

/// Retrieve the current window scale factor (e.g. 2.0 on Retina displays).
/// Returns 1.0 if the handle is invalid.
export fn zigote_get_scale(handle: u64) f32 {
    const state = stateFromHandle(handle) orelse return 1.0;
    return state.window.getDisplayScale() catch 1.0;
}

/// Refresh rate (Hz) of the display the given window currently sits on. 0 when unknown — the caller
/// picks its own fallback. Re-query on EVT_DISPLAY_CHANGED: dragging a window from a 60 Hz panel to
/// a 144 Hz one changes the answer.
fn refreshHzForWindow(state: *EngineState, win_id: u32) f32 {
    const window = if (win_id == 0 or win_id == state.main_window_id)
        state.window
    else if (windowFromSdlId(state, win_id)) |win| win.window else return 0;
    const display = window.getDisplayForWindow() catch return 0;
    const mode = display.getCurrentMode() catch return 0;
    return mode.refresh_rate orelse 0;
}

/// Refresh rate (Hz) of the display showing `window_id` (0 = main window); 0 if unknown.
export fn zigote_get_refresh_hz(handle: u64, window_id: u32) f32 {
    const state = stateFromHandle(handle) orelse return 0;
    return refreshHzForWindow(state, window_id);
}

/// One enumerated GPU as handed to the host. Alias so the FFI surface names it the way the rest of
/// the C ABI does (and so the C# binding generator maps it to the `ZgGpuInfo` struct).
pub const ZgGpuInfo = gpu_select.GpuInfo;

/// Copy up to `max` enumerated GPUs into `out`, returning how many were written. This is the list
/// the engine chose from at init (see gpu_select.zig) — it does not re-scan, so it is cheap and
/// stable for the lifetime of the handle. Pass a null `out_gpus` to just query the count.
/// (Named `out_gpus`, not `out`: the C# binding generator uses the parameter name verbatim and
/// `out` is a keyword there.)
export fn zigote_enumerate_gpus(handle: u64, out_gpus: [*c]ZgGpuInfo, max: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    const n = @min(state.gpu_count, max);
    if (out_gpus != null) @memcpy(out_gpus[0..n], state.gpus[0..n]);
    return n;
}

/// Index (into zigote_enumerate_gpus) of the GPU the device was created on, or -1 when the adapter
/// came from wgpu's own fallback pick and is not one of the enumerated entries.
export fn zigote_get_active_gpu(handle: u64) i32 {
    const state = stateFromHandle(handle) orelse return -1;
    return state.active_gpu;
}

// ── 3D Scene FFI ──────────────────────────────────────────────────────────────

export fn zigote_scene_clear(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    state.untrackAllNodes();
    // The selection points at a node that is about to be freed.
    state.selected_node_ptr = 0;

    // Release ALL of the previous scene's GPU resources — geometry, material textures, per-entity
    // model UBOs and instancing buffers — not just the per-entity/instance maps. Without freeing
    // prim_buffers + material caches here, every scene load / project switch leaked the prior scene's
    // vertex/index buffers and textures until app shutdown.
    if (state.gpu_3d) |g| g.clearScene();

    state.world.deinit();
    state.world = zg.World.init(state.allocator);
}

export fn zigote_scene_add_child_node(handle: u64, parent_handle: u64, name_ptr: [*c]const u8, kind: u8) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    const name = std.mem.span(name_ptr);
    var node: *zg.SceneNode = undefined;
    if (parent_handle == 0) {
        node = state.world.createNode(name) catch return 0;
    } else {
        if (!state.isValidNode(parent_handle)) {
            std.log.warn("zigote_scene_add_child_node: invalid parent handle 0x{x}", .{parent_handle});
            return 0;
        }
        const parent: *zg.SceneNode = @ptrFromInt(parent_handle);
        node = state.world.createChild(parent, name) catch return 0;
    }

    if (kind == 3) {
        node.addComponent(state.allocator, .{
            .camera = .{
                // Initial defaults only. The host authors per-camera fov/near/far on the SceneNode and pushes
                // them via zigote_scene_set_camera_params; PlaySession.PublishRenderView builds its culling
                // frustum from the SAME per-camera values, so the two stay in lockstep without a shared constant.
                .fovy_degrees = 45.0,
                .near = 0.1,
                // Far plane sized for large scenes (e.g. the asteroid-belt benchmark, viewed from
                // ~950 units out with a 500-radius belt → far edge ~1500 away). 1000 clipped them.
                // The 40000:1 far:near ratio is an accepted depth-precision tradeoff (no reverse-Z yet);
                // if z-fighting appears at mid-range, raise near or adopt reverse-Z.
                .far = 4000.0,
            },
        }) catch return 0;
        if (state.world.active_camera == null) {
            state.world.active_camera = node;
        }
    } else if (kind == 2) {
        node.addComponent(state.allocator, .{ .light = .{
            .kind = .point,
            .color = .{ .x = 1, .y = 1, .z = 1 },
            .intensity = 1.0,
            .range = 50.0,
        } }) catch return 0;
    }

    const node_handle = @intFromPtr(node);
    state.trackNode(node_handle);
    return node_handle;
}

/// Set the projection parameters of a camera node: perspective vertical FOV (degrees) + clip planes.
/// The renderer rebuilds the projection from these fields every frame (see wgpu_3d.zig projMatrix), so
/// the change takes effect next frame. No-op if the node is invalid or carries no camera component.
/// NOTE: the host's published culling frustum (PlaySession.PublishRenderView) must be kept in sync with
/// whatever FOV/near/far is set here — see the FOV lockstep note there.
export fn zigote_scene_set_camera_params(handle: u64, node_handle: u64, fovy_degrees: f32, near: f32, far: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const cam_comp = node.getComponent(.camera) orelse return;
    cam_comp.camera.fovy_degrees = fovy_degrees;
    cam_comp.camera.near = near;
    cam_comp.camera.far = far;
}

/// Import any model file Assimp understands. Writes per-mesh `.zmesh` caches and extracted
/// textures under `cache_dir`, and returns a JSON manifest (node tree, materials, lights,
/// animations) as a malloc'd C string. Caller releases it with `zigote_model_free`.
/// Returns null on failure. Does not touch the engine/world — pure import + cache emit.
export fn zigote_model_import(path_c: [*c]const u8, cache_dir_c: [*c]const u8) [*c]u8 {
    if (path_c == null or cache_dir_c == null) return null;
    const json = assimp_loader.importModelJson(@ptrCast(path_c), @ptrCast(cache_dir_c)) orelse return null;
    return @ptrCast(json);
}

/// Free a manifest string returned by `zigote_model_import`.
export fn zigote_model_free(ptr: [*c]u8) void {
    if (ptr != null) std.c.free(ptr);
}

/// Point `node` at freshly created mesh/material slots. If the node already carries a
/// mesh_renderer, its handles are updated in place and the previous mesh/material are released
/// when no surviving node still references them (same rule as zigote_scene_remove_node) —
/// consumers read the FIRST matching component, so stacking a second one would leave the old
/// mesh rendering and leak both slots. Fresh nodes get the component added.
fn setNodeMeshRenderer(state: *EngineState, node: *zg.SceneNode, mesh_handle: u32, mat_handle: u32) void {
    if (node.getComponent(.mesh_renderer)) |c| {
        const old_mesh = c.mesh_renderer.mesh;
        const old_mat = c.mesh_renderer.material;
        c.mesh_renderer.mesh = mesh_handle;
        c.mesh_renderer.material = mat_handle;
        if (old_mesh != std.math.maxInt(u32) and old_mesh != mesh_handle and !state.world.isMeshReferenced(old_mesh)) {
            state.world.freeMeshSlot(old_mesh);
            if (state.gpu_3d) |g| g.mesh_cache.invalidate(old_mesh);
        }
        if (old_mat != std.math.maxInt(u32) and old_mat != mat_handle and !state.world.isMaterialReferenced(old_mat)) {
            state.world.freeMaterialSlot(old_mat);
            invalidateMaterialGpu(state, old_mat);
        }
    } else {
        node.addComponent(state.allocator, .{ .mesh_renderer = .{
            .mesh = mesh_handle,
            .material = mat_handle,
        } }) catch return;
    }
}

/// Upload a `.zmesh` binary blob (engine Vertex layout, see zmesh_format.zig) to a node's
/// mesh renderer. Replaces the former GLB-based path; the merged geometry is produced by the
/// Assimp importer and read back here as a flat vertex/index buffer.
export fn zigote_scene_set_mesh_blob(handle: u64, node_handle: u64, data_ptr: [*c]const u8, data_len: usize) void {
    const state = stateFromHandle(handle) orelse return;
    if (data_ptr == null or data_len == 0) return;
    if (!state.isValidNode(node_handle)) {
        std.log.warn("zigote_scene_set_mesh_blob: invalid handle 0x{x}", .{node_handle});
        return;
    }
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const data = data_ptr[0..data_len];

    const mesh = zmesh_format.parseZmesh(state.allocator, data) catch |err| {
        std.log.warn("zigote_scene_set_mesh_blob: parse failed: {}", .{err});
        return;
    };

    const mesh_handle = state.world.addMesh(mesh) catch return;
    const mat_handle = state.world.addMaterial(.{}) catch return;

    setNodeMeshRenderer(state, node, mesh_handle, mat_handle);
}

export fn zigote_scene_set_mesh_primitive(handle: u64, node_handle: u64, prim_type: u8) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) {
        std.log.warn("zigote_scene_set_mesh_primitive: invalid handle 0x{x}", .{node_handle});
        return;
    }
    const node: *zg.SceneNode = @ptrFromInt(node_handle);

    var mesh: zg.resources.Mesh = undefined;
    switch (prim_type) {
        0 => mesh = zg.resources.Mesh.createCube(state.allocator) catch return,
        1 => mesh = zg.resources.Mesh.createQuad(state.allocator) catch return,
        2 => mesh = zg.resources.Mesh.createSphere(state.allocator, 16, 16) catch return,
        3 => mesh = zg.resources.Mesh.createCylinder(state.allocator, 24) catch return,
        else => return,
    }

    const mesh_handle = state.world.addMesh(mesh) catch return;
    const mat_handle = state.world.addMaterial(.{}) catch return;

    setNodeMeshRenderer(state, node, mesh_handle, mat_handle);
}

export fn zigote_scene_set_light_properties(
    handle: u64,
    node_handle: u64,
    kind: u8,
    r: f32,
    g: f32,
    b: f32,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    cast_shadows: u32,
) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);

    var light_comp = node.getComponent(.light) orelse return;
    light_comp.light.kind = switch (kind) {
        0 => .directional,
        1 => .point,
        2 => .spot,
        else => .point,
    };
    light_comp.light.color = .{ .x = r, .y = g, .z = b };
    light_comp.light.intensity = intensity;
    light_comp.light.range = range;
    light_comp.light.inner_angle = inner_angle;
    light_comp.light.outer_angle = outer_angle;
    light_comp.light.cast_shadows = cast_shadows != 0;
}

export fn zigote_scene_set_mesh_color(handle: u64, node_handle: u64, r: f32, g: f32, b: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);

    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].base_color_factor = .{ .x = r, .y = g, .z = b, .w = 1.0 };
}

/// Submit per-instance model matrices for a node's mesh (GPU instancing). `matrices` points at
/// count * 16 column-major f32 values (one 4x4 model matrix per instance). The node then draws
/// as `count` instances of its shared mesh+material in a single instanced draw, ignoring its own
/// transform. Pass count == 0 to draw nothing for this node (an instanced node is never rendered as
/// a single fallback draw — used to empty a LOD bucket / stop drawing without leaving a stray mesh).
export fn zigote_scene_set_mesh_instances(handle: u64, node_handle: u64, matrices: [*c]const f32, count: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const g = ensure3d(state) orelse return;
    g.setInstances(node.entity, matrices, count);
}

// ── VFX particles ───────────────────────────────────────────────────────────────
// Upload one emitter's CPU-simulated particles for the native billboard pass (drawn in the geometry
// pass). `data` is `count` particles × 9 f32 (position.xyz, size, rotation, color.rgba); `blend` is
// 0 = additive, 1 = alpha. `node_handle` is only a stable batch key here (not dereferenced), so a stale
// handle is harmless. The renderer's particle system is lazy + failure-isolated — these are no-ops until
// it initialises, and a shader failure disables it rather than crashing.
export fn zigote_particles_upload(handle: u64, node_handle: u64, data: [*c]const f32, count: u32, blend: u32) void {
    const state = stateFromHandle(handle) orelse return;
    const g = ensure3d(state) orelse return;
    g.particles.upload(state.device, state.queue, g.allocator, node_handle, data, count, blend);
}

export fn zigote_particles_clear(handle: u64, node_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (state.gpu_3d) |g| g.particles.clearNode(node_handle);
}

export fn zigote_particles_clear_all(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (state.gpu_3d) |g| g.particles.clearAll();
}

// GPU-compute particle simulation. Register/update one emitter's lowered params (the host computes the
// per-frame spawn budget); a compute kernel then spawns + updates them on the GPU and writes the billboard
// instance buffer. `params` is 112 f32 (see particle_compute_source.wgsl Params); `capacity` is the slot
// count (the persistent GPU state buffer is sized to it); `blend` 0=additive, 1=alpha.
export fn zigote_particles_compute_emit(handle: u64, node_handle: u64, param_values: [*c]const f32, param_count: u32, capacity: u32, blend: u32) void {
    const state = stateFromHandle(handle) orelse return;
    const g = ensure3d(state) orelse return;
    g.particles.computeEmit(state.device, state.queue, g.allocator, node_handle, param_values, param_count, capacity, blend);
}

export fn zigote_particles_compute_clear(handle: u64, node_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (state.gpu_3d) |g| g.particles.clearNode(node_handle);
}

// ── 2D sprite renderer (wgpu_sprites.zig) ───────────────────────────────────────
// Immediate-mode frame model: the host calls _begin once per frame (scene + overlay cameras),
// then _draw per PRE-SORTED batch (C# owns sorting layers/order; native draws in submission
// order). Batches render in two stages hooked inside Gpu3d: stage 0 after scene geometry
// (HDR, bloom/tonemap apply), stage 1 after post (LDR, exact colors). Textures and custom
// WGSL shaders are u32 handles owned by the sprite system (0 = invalid / default).

/// Create a sprite texture from tightly-packed RGBA8 pixels. filter 0=nearest/1=linear,
/// srgb 0/1 (use 0 for data textures fed to custom shaders), wrap 0=clamp/1=repeat.
/// Returns the texture handle, or 0 on failure (dimensions must be 1..8192).
export fn zigote_sprites_texture_create(handle: u64, pixels: [*c]const u8, width: u32, height: u32, filter: u32, srgb: u32, wrap: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    if (pixels == null) return 0;
    const g = ensure3d(state) orelse return 0;
    const len = @as(usize, width) * @as(usize, height) * 4;
    return g.sprites.createTexture(state.device, state.queue, state.allocator, pixels[0..len], width, height, filter, srgb, wrap);
}

/// Create a sprite texture by decoding an image file (PNG/JPG/WebP/GIF — same decoders as
/// scene textures). Returns the handle (0 on failure) and writes the pixel size.
export fn zigote_sprites_texture_create_file(handle: u64, path_c: [*c]const u8, filter: u32, srgb: u32, wrap: u32, out_w: *u32, out_h: *u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    if (path_c == null) return 0;
    // Ensure BEFORE the file read/decode so a failed 3D init doesn't pay a full image decode per call.
    const g = ensure3d(state) orelse return 0;
    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 64)) catch return 0;
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse return 0;
    defer state.allocator.free(bytes);

    const id = g.sprites.createTexture(state.device, state.queue, state.allocator, bytes, w, h, filter, srgb, wrap);
    if (id != 0) {
        out_w.* = w;
        out_h.* = h;
    }
    return id;
}

export fn zigote_sprites_texture_destroy(handle: u64, texture: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (state.gpu_3d) |g| g.sprites.destroyTexture(texture);
}

/// Compile a custom sprite shader (contract in sprite_shader_source.wgsl: vs_main/fs_main over
/// the sprite instance layout; group 0 camera, 1 texture, 2 params, 3 secondary texture;
/// premultiplied-alpha output). Returns the shader handle, or 0 if the WGSL is rejected.
export fn zigote_sprites_shader_create(handle: u64, wgsl_ptr: [*c]const u8, wgsl_len: u32) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    if (wgsl_ptr == null or wgsl_len == 0) return 0;
    const g = ensure3d(state) orelse return 0;
    return g.sprites.createShader(state.device, state.queue, state.allocator, wgsl_ptr[0..wgsl_len], .rgba16_float);
}

/// Start the sprite frame: 16 column-major floats per camera (scene stage = world camera,
/// overlay stage = usually a pixel-space ortho), plus the viewport size in pixels.
export fn zigote_sprites_begin(handle: u64, scene_vp: [*c]const f32, overlay_vp: [*c]const f32, viewport_w: f32, viewport_h: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (scene_vp == null or overlay_vp == null) return;
    const g = ensure3d(state) orelse return;
    g.sprites.begin(state.allocator, scene_vp[0..16], overlay_vp[0..16], viewport_w, viewport_h);
}

/// Append one pre-sorted sprite batch: `count` instances × 14 f32 (pos.xyz, rot, size.xy,
/// uv0.xy, uv1.xy, rgba). texture2 = secondary texture for custom shaders (0 = white 1×1);
/// shader 0 = default; blend 0 alpha / 1 additive / 2 opaque; stage 0 scene / 1 overlay;
/// params = up to 16 floats of material data (dynamic-offset UBO slot).
export fn zigote_sprites_draw(handle: u64, texture: u32, texture2: u32, shader: u32, blend: u32, stage: u32, param_values: [*c]const f32, param_count: u32, data: [*c]const f32, count: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (data == null or count == 0) return;
    const params: []const f32 = if (param_values == null) &.{} else param_values[0..@min(param_count, 16)];
    const floats = @as(usize, count) * wgpu_renderer.wgpu_sprites.SpriteSystem.INSTANCE_FLOATS;
    const g = ensure3d(state) orelse return;
    g.sprites.draw(texture, texture2, shader, blend, stage, params, data[0..floats], count);
}

/// Toggle a mesh node's visibility. The renderer skips invisible mesh renderers entirely — a real
/// draw-call cull, unlike scaling to zero (which still issues the draw). Used by the C# LOD/cull
/// system to hide distant detail / select LOD levels without re-uploading geometry.
export fn zigote_scene_set_node_visible(handle: u64, node_handle: u64, visible: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    mr_comp.mesh_renderer.visible = visible != 0;
}

/// Enable/disable world-3D frustum culling (on by default). Honored by both backends: the wgpu
/// reference renderer and the Metal backend each filter their forward geometry pass.
export fn zigote_render_set_frustum_cull(handle: u64, enabled: u32) void {
    const state = stateFromHandle(handle) orelse return;
    // Must NOT create the 3D renderer — remember the toggle and apply it at creation.
    state.pending_frustum_cull = enabled != 0;
    if (state.gpu_3d) |g| g.frustum_cull = enabled != 0;
}

export fn zigote_scene_set_mesh_roughness(handle: u64, node_handle: u64, metallic: f32, roughness: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].metallic_factor = metallic;
    state.world.materials.items[mat_handle].roughness_factor = roughness;
}

/// Set extended-PBR surface parameters (KHR_materials_clearcoat / _specular). Read straight from
/// the material each frame into the per-draw uniforms, so no GPU cache invalidation is needed.
export fn zigote_scene_set_mesh_surface(handle: u64, node_handle: u64, clearcoat: f32, clearcoat_roughness: f32, specular: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].clearcoat_factor = clearcoat;
    state.world.materials.items[mat_handle].clearcoat_roughness = clearcoat_roughness;
    state.world.materials.items[mat_handle].specular_factor = specular;
}

/// Set the emissive colour (already pre-multiplied by KHR_materials_emissive_strength on the C#
/// side). Read into the per-draw uniforms each frame; no GPU cache invalidation needed.
export fn zigote_scene_set_mesh_emissive(handle: u64, node_handle: u64, r: f32, g: f32, b: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].emissive_factor = .{ .x = r, .y = g, .z = b };
}

export fn zigote_scene_set_mesh_effect(handle: u64, node_handle: u64, effect: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].effect = @enumFromInt(effect);
}

/// Set the material's alpha mode (0=opaque, 1=mask, 2=blend, 3=glass) and the mask cutoff
/// threshold (used only by mode 1; pass 0.5 when unknown). Blend routes the surface to the
/// depth-sorted transparent pipeline and glass to the refraction pass. For untextured glass
/// we also seed a see-through base-colour tint so it reads as a window, not a solid panel.
export fn zigote_scene_set_mesh_alpha_mode(handle: u64, node_handle: u64, mode: u32, cutoff: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    const mat = &state.world.materials.items[mat_handle];
    mat.alpha_mode = @enumFromInt(mode);
    mat.alpha_cutoff = std.math.clamp(cutoff, 0.0, 1.0);
    if (mode == 2 and mat.base_color_factor.w >= 0.999) {
        mat.base_color_factor.w = 0.35; // default glass tint when the glTF gave no alpha
    }
}

/// Set whether the material renders double-sided (no back-face cull). Honoured by the opaque,
/// transparent and glass passes via their per-draw pipeline selection.
export fn zigote_scene_set_mesh_double_sided(handle: u64, node_handle: u64, double_sided: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].double_sided = double_sided != 0;
}

/// Set KHR_materials_ior / _transmission: `ior` drives the dielectric F0 (((n-1)/(n+1))²) and the
/// glass refraction bend; `transmission` (0..1) blends the lit surface toward the transmissive
/// glass response. Read into the per-draw uniforms each frame; no GPU cache invalidation needed.
export fn zigote_scene_set_mesh_volume(handle: u64, node_handle: u64, ior: f32, transmission: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].ior = ior;
    state.world.materials.items[mat_handle].transmission = std.math.clamp(transmission, 0.0, 1.0);
}

/// Set the glTF ORM occlusion strength: > 0 tells the shader the metallic-roughness map's R
/// channel is baked ambient occlusion (the glTF ORM packing) and scales its effect on ambient.
export fn zigote_scene_set_mesh_occlusion_strength(handle: u64, node_handle: u64, strength: f32) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;
    state.world.materials.items[mat_handle].occlusion_strength = std.math.clamp(strength, 0.0, 1.0);
}

/// Drop any cached GPU material maps for `mat_handle` so the renderer rebuilds them from the
/// (just-changed) CPU pixels next frame.
fn invalidateMaterialGpu(state: *EngineState, mat_handle: u32) void {
    // Nothing to invalidate before the 3D renderer exists — its caches populate at render time.
    const g = state.gpu_3d orelse return;
    const cache = &g.material_gpu_cache;
    if (mat_handle < cache.items.len) {
        if (cache.items[mat_handle]) |*val| {
            val.deinit();
            cache.items[mat_handle] = null;
        }
    }
}

export fn zigote_scene_set_mesh_texture_file(handle: u64, node_handle: u64, path_c: [*c]const u8) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;

    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;

    const mat = &state.world.materials.items[mat_handle];

    if (path_c == null or path_c[0] == 0) {
        if (mat.base_color_pixels) |p| {
            state.allocator.free(p);
        }
        mat.base_color_pixels = null;
        mat.base_color_width = 0;
        mat.base_color_height = 0;

        invalidateMaterialGpu(state, mat_handle);
        return;
    }

    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 256)) catch |err| {
        std.log.err("zigote: base-color texture read failed for '{s}': {}", .{ path, err });
        return;
    };
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse {
        std.log.err("zigote: base-color texture decode failed for '{s}' ({d} bytes)", .{ path, file_data.len });
        return;
    };
    std.log.info("zigote: base-color texture loaded '{s}' {d}x{d}", .{ path, w, h });

    // Free old pixels if any
    if (mat.base_color_pixels) |p| {
        state.allocator.free(p);
    }

    mat.base_color_pixels = bytes;
    mat.base_color_width = w;
    mat.base_color_height = h;

    // Invalidate the cached GPU maps so the active renderer recreates them on the next frame.
    invalidateMaterialGpu(state, mat_handle);
}

/// Set (or clear, with null/empty path) the metallic-roughness map for a mesh node.
/// glTF convention: roughness in the green channel, metallic in the blue channel.
export fn zigote_scene_set_mesh_mr_texture_file(handle: u64, node_handle: u64, path_c: [*c]const u8) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;

    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;

    const mat = &state.world.materials.items[mat_handle];

    if (path_c == null or path_c[0] == 0) {
        if (mat.metallic_roughness_pixels) |p| state.allocator.free(p);
        mat.metallic_roughness_pixels = null;
        mat.metallic_roughness_width = 0;
        mat.metallic_roughness_height = 0;
        invalidateMaterialGpu(state, mat_handle);
        return;
    }

    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 64)) catch return;
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse return;

    if (mat.metallic_roughness_pixels) |p| state.allocator.free(p);
    mat.metallic_roughness_pixels = bytes;
    mat.metallic_roughness_width = w;
    mat.metallic_roughness_height = h;

    invalidateMaterialGpu(state, mat_handle);
}

/// Set (or clear, with null/empty path) the tangent-space normal map for a mesh node. The image is
/// linear (NOT sRGB); the mesh shader applies it via the TBN basis (tangents auto-generated if the
/// mesh shipped none).
export fn zigote_scene_set_mesh_normal_texture_file(handle: u64, node_handle: u64, path_c: [*c]const u8) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;

    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;

    const mat = &state.world.materials.items[mat_handle];

    if (path_c == null or path_c[0] == 0) {
        if (mat.normal_pixels) |p| state.allocator.free(p);
        mat.normal_pixels = null;
        mat.normal_width = 0;
        mat.normal_height = 0;
        invalidateMaterialGpu(state, mat_handle);
        return;
    }

    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 64)) catch return;
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse return;

    if (mat.normal_pixels) |p| state.allocator.free(p);
    mat.normal_pixels = bytes;
    mat.normal_width = w;
    mat.normal_height = h;

    invalidateMaterialGpu(state, mat_handle);
}

/// Set (or clear, with null/empty path) the emissive map for a mesh node. The image is sRGB
/// colour; the shader multiplies it by the material's emissive factor.
export fn zigote_scene_set_mesh_emissive_texture_file(handle: u64, node_handle: u64, path_c: [*c]const u8) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;

    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return;
    if (mat_handle >= state.world.materials.items.len) return;

    const mat = &state.world.materials.items[mat_handle];

    if (path_c == null or path_c[0] == 0) {
        if (mat.emissive_pixels) |p| state.allocator.free(p);
        mat.emissive_pixels = null;
        mat.emissive_width = 0;
        mat.emissive_height = 0;
        invalidateMaterialGpu(state, mat_handle);
        return;
    }

    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 64)) catch return;
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse return;

    if (mat.emissive_pixels) |p| state.allocator.free(p);
    mat.emissive_pixels = bytes;
    mat.emissive_width = w;
    mat.emissive_height = h;

    invalidateMaterialGpu(state, mat_handle);
}

/// One entry for the parallel batch texture loader. `base_color_path` / `mr_path` / `normal_path`
/// / `emissive_path` are optional (null = skip that map). Layout must match ZgTextureLoadItem in C#.
pub const ZgTextureLoadItem = extern struct {
    node_handle: u64,
    base_color_path: [*c]const u8,
    mr_path: [*c]const u8,
    normal_path: [*c]const u8,
    emissive_path: [*c]const u8,
};

/// Which material slot a decoded image targets.
const TexKind = enum(u8) { base, mr, normal, emissive };

/// One queued image decode for the parallel batch loader. The file is read on the calling thread,
/// decoded on a worker thread, then stored into the material on the calling thread.
const TextureDecodeJob = struct {
    item_index: u32,
    kind: TexKind,
    file_data: []u8, // owned; freed by the worker once decoded
    pixels: ?[]u8 = null, // decoded RGBA8 (c_allocator), null on failure; ownership moves to material
    width: u32 = 0,
    height: u32 = 0,
};

/// Worker thread body: pull jobs off the shared atomic cursor and decode each. Uses
/// std.heap.c_allocator (thread-safe) and touches no shared engine state, so any number of these
/// run concurrently. Image decode (zigimg/webp/gif) is CPU-bound and pure — the win this buys.
fn textureDecodeWorker(jobs: []TextureDecodeJob, cursor: *std.atomic.Value(usize)) void {
    while (true) {
        const i = cursor.fetchAdd(1, .monotonic);
        if (i >= jobs.len) break;
        const job = &jobs[i];
        var w: u32 = 0;
        var h: u32 = 0;
        job.pixels = loadTextureBytes(std.heap.c_allocator, job.file_data, &w, &h);
        job.width = w;
        job.height = h;
        std.heap.c_allocator.free(job.file_data);
        job.file_data = &.{};
    }
}

/// Read one optional texture file (calling thread) and append a decode job. No-op for a null path.
fn queueDecode(
    jobs: *std.ArrayListUnmanaged(TextureDecodeJob),
    alloc: std.mem.Allocator,
    io: std.Io,
    idx: u32,
    kind: TexKind,
    path_c: [*c]const u8,
) void {
    if (path_c == null or path_c[0] == 0) return;
    const path = std.mem.span(path_c);
    if (std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1024 * 1024 * 256))) |data| {
        jobs.append(alloc, .{ .item_index = idx, .kind = kind, .file_data = data }) catch alloc.free(data);
    } else |err| {
        std.log.err("zigote: batch {s} texture read failed for '{s}': {}", .{ @tagName(kind), path, err });
    }
}

/// Resolve a node handle to its material index, or null if the node has no valid material.
fn materialHandleForNode(state: *EngineState, node_handle: u64) ?u32 {
    if (!state.isValidNode(node_handle)) return null;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);
    const mr_comp = node.getComponent(.mesh_renderer) orelse return null;
    const mat_handle = mr_comp.mesh_renderer.material;
    if (mat_handle == std.math.maxInt(u32)) return null;
    if (mat_handle >= state.world.materials.items.len) return null;
    return mat_handle;
}

/// Load many mesh textures at once, decoding them in PARALLEL across a thread pool. Each item names
/// a node plus an optional base-colour and/or metallic-roughness image. Files are read on the
/// calling thread, decoded concurrently, then stored serially (the material list + GPU cache are
/// not thread-safe). Far faster than calling the per-texture FFI in a loop for a model with many
/// materials — the dominant cost (image decode) now scales with core count.
export fn zigote_scene_load_textures_batch(handle: u64, items_ptr: [*]const ZgTextureLoadItem, count: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (count == 0) return;
    const items = items_ptr[0..count];
    const alloc = state.allocator; // std.heap.c_allocator — thread-safe

    var jobs: std.ArrayListUnmanaged(TextureDecodeJob) = .empty;
    defer {
        for (jobs.items) |*j| {
            if (j.file_data.len != 0) alloc.free(j.file_data);
            if (j.pixels) |p| alloc.free(p); // any pixels not transferred to a material
        }
        jobs.deinit(alloc);
    }

    // ── Phase 1: read files (calling thread) and queue decode jobs ──
    var io_state = std.Io.Threaded.init(alloc, .{});
    defer io_state.deinit();
    const io = io_state.io();

    for (items, 0..) |it, idx| {
        queueDecode(&jobs, alloc, io, @intCast(idx), .base, it.base_color_path);
        queueDecode(&jobs, alloc, io, @intCast(idx), .mr, it.mr_path);
        queueDecode(&jobs, alloc, io, @intCast(idx), .normal, it.normal_path);
        queueDecode(&jobs, alloc, io, @intCast(idx), .emissive, it.emissive_path);
    }
    if (jobs.items.len == 0) return;

    // ── Phase 2: decode in parallel ──
    var cursor = std.atomic.Value(usize).init(0);
    const want: usize = @min(std.Thread.getCpuCount() catch 4, jobs.items.len);
    const threads_opt: ?[]std.Thread = alloc.alloc(std.Thread, want) catch null;
    defer if (threads_opt) |t| alloc.free(t);
    var spawned: usize = 0;
    if (threads_opt) |threads| {
        for (0..threads.len) |_| {
            threads[spawned] = std.Thread.spawn(.{}, textureDecodeWorker, .{ jobs.items, &cursor }) catch break;
            spawned += 1;
        }
    }
    // The calling thread also drains the queue, so all jobs complete even if no thread spawned.
    textureDecodeWorker(jobs.items, &cursor);
    if (threads_opt) |threads| for (threads[0..spawned]) |t| t.join();

    // ── Phase 3: store decoded pixels into materials (serial — mutates shared state) ──
    for (jobs.items) |*job| {
        const pixels = job.pixels orelse continue; // decode failed
        const mat_handle = materialHandleForNode(state, items[job.item_index].node_handle) orelse continue;
        const mat = &state.world.materials.items[mat_handle];
        switch (job.kind) {
            .mr => {
                if (mat.metallic_roughness_pixels) |old| alloc.free(old);
                mat.metallic_roughness_pixels = pixels;
                mat.metallic_roughness_width = job.width;
                mat.metallic_roughness_height = job.height;
            },
            .base => {
                if (mat.base_color_pixels) |old| alloc.free(old);
                mat.base_color_pixels = pixels;
                mat.base_color_width = job.width;
                mat.base_color_height = job.height;
            },
            .normal => {
                if (mat.normal_pixels) |old| alloc.free(old);
                mat.normal_pixels = pixels;
                mat.normal_width = job.width;
                mat.normal_height = job.height;
            },
            .emissive => {
                if (mat.emissive_pixels) |old| alloc.free(old);
                mat.emissive_pixels = pixels;
                mat.emissive_width = job.width;
                mat.emissive_height = job.height;
            },
        }
        job.pixels = null; // ownership transferred to the material
        // Invalidate the cached GPU maps so the active renderer rebuilds them next frame.
        invalidateMaterialGpu(state, mat_handle);
    }
}

/// Mesh/material handles used by a subtree, collected before it is destroyed. May contain
/// duplicates if two nodes happen to share a handle; the free pass in zigote_scene_remove_node
/// marks each handle as it is freed, so a slot is released at most once.
const SubtreeAssets = struct {
    meshes: std.ArrayListUnmanaged(u32) = .empty,
    materials: std.ArrayListUnmanaged(u32) = .empty,

    fn deinit(self: *SubtreeAssets, alloc: std.mem.Allocator) void {
        self.meshes.deinit(alloc);
        self.materials.deinit(alloc);
    }
};

/// Walk `node` + descendants, shedding the per-node state the world's CPU tree does NOT own: drop
/// each node's per-entity renderer state (instancing / model UBO), untrack its C# handle, clear it
/// if selected, and record the mesh/material it references. Must run BEFORE world.removeNode frees
/// the subtree — the node pointers and handles are read here. Every node in the subtree is handled,
/// not just the root: children were tracked at creation (see zigote_scene_add_child_node), so
/// untracking only the root left their freed pointers reported as valid (a use-after-free waiting to
/// happen), and their instancing state leaked.
fn gatherSubtreeForRemoval(state: *EngineState, node: *zg.SceneNode, assets: *SubtreeAssets) void {
    if (state.gpu_3d) |g| g.invalidateEntity(node.entity);

    const nh = @intFromPtr(node);
    state.untrackNode(nh);
    if (state.selected_node_ptr == nh) state.selected_node_ptr = 0;

    if (node.getComponent(.mesh_renderer)) |c| {
        const mr = c.mesh_renderer;
        if (mr.mesh != std.math.maxInt(u32)) assets.meshes.append(state.allocator, mr.mesh) catch {};
        if (mr.material != std.math.maxInt(u32)) assets.materials.append(state.allocator, mr.material) catch {};
    }

    for (node.children.items) |child| gatherSubtreeForRemoval(state, child, assets);
}

/// Mark every mesh/material handle still referenced by `node` + descendants in the given bitsets
/// (indexed by handle — handles are dense array indices, see World.addMesh/addMaterial).
fn markSubtreeAssets(node: *zg.SceneNode, used_meshes: []bool, used_materials: []bool) void {
    if (node.getComponent(.mesh_renderer)) |c| {
        const mr = c.mesh_renderer;
        if (mr.mesh < used_meshes.len) used_meshes[mr.mesh] = true;
        if (mr.material < used_materials.len) used_materials[mr.material] = true;
    }
    for (node.children.items) |child| markSubtreeAssets(child, used_meshes, used_materials);
}

export fn zigote_scene_remove_node(handle: u64, node_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) return;
    const node: *zg.SceneNode = @ptrFromInt(node_handle);

    // Collect the subtree's assets + shed per-node renderer/handle state while the tree is still live.
    var assets = SubtreeAssets{};
    defer assets.deinit(state.allocator);
    gatherSubtreeForRemoval(state, node, &assets);

    // Frees the CPU node subtree. After this the collected handles are referenced only by nodes that
    // survive, so the reference scan below decides correctly what is safe to release.
    state.world.removeNode(node);

    // Release every mesh/material the removed subtree held that no surviving node still uses — this is
    // the actual leak fix: previously the CPU Mesh/Material and their GPU buffers/textures were never
    // freed on node removal. A shared handle (kept by another node) is left intact; only the last
    // reference frees it. CPU slot: tombstoned + recycled by addMesh/addMaterial. GPU: the cached
    // vertex/index/line buffers (MeshCache) and textures/views/bind group (MaterialGpu) are released.
    // One walk over the surviving tree marks the handles still in use — O(nodes + assets) instead of
    // a full-scene reference scan per collected handle. On marking-set OOM the free pass is skipped
    // (a leak, never a wrong free). A freed handle is re-marked so a duplicate frees the slot once.
    const used_meshes = state.allocator.alloc(bool, state.world.meshes.items.len) catch return;
    defer state.allocator.free(used_meshes);
    const used_materials = state.allocator.alloc(bool, state.world.materials.items.len) catch return;
    defer state.allocator.free(used_materials);
    @memset(used_meshes, false);
    @memset(used_materials, false);
    for (state.world.roots.items) |root| markSubtreeAssets(root, used_meshes, used_materials);

    for (assets.meshes.items) |h| {
        if (h < used_meshes.len and !used_meshes[h]) {
            state.world.freeMeshSlot(h);
            if (state.gpu_3d) |g| g.mesh_cache.invalidate(h);
            used_meshes[h] = true;
        }
    }
    for (assets.materials.items) |h| {
        if (h < used_materials.len and !used_materials[h]) {
            state.world.freeMaterialSlot(h);
            invalidateMaterialGpu(state, h);
            used_materials[h] = true;
        }
    }
}

export fn zigote_scene_update_node(
    handle: u64,
    node_handle: u64,
    x: f32,
    y: f32,
    z: f32,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    sx: f32,
    sy: f32,
    sz: f32,
) void {
    const state = stateFromHandle(handle) orelse return;
    if (!state.isValidNode(node_handle)) {
        std.log.warn("zigote_scene_update_node: invalid handle 0x{x}", .{node_handle});
        return;
    }
    const node: *zg.SceneNode = @ptrFromInt(node_handle);

    node.local_transform.position = .{ .x = x, .y = y, .z = z };

    // Rotation is passed as a quaternion directly — avoids a lossy Euler round-trip
    // for compound rotations (e.g. imported glTF node matrices).
    node.local_transform.rotation = (zg.math3d.Quat{ .x = qx, .y = qy, .z = qz, .w = qw }).normalize();
    node.local_transform.scale = .{ .x = sx, .y = sy, .z = sz };
    node.dirty_transform = true;
}

/// Set the currently selected scene node so the renderer can apply a selection highlight.
/// Pass 0 to clear the selection.
export fn zigote_scene_set_selected_node(handle: u64, node_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    state.selected_node_ptr = node_handle;
}

export fn zigote_render_3d(handle: u64, width: u32, height: u32) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (width == 0 or height == 0) return 0;

    // We recreate the texture if its size changed or it doesn't exist
    var recreate = false;
    if (state.offscreen_3d_texture) |tex| {
        if (tex.getWidth() != width or tex.getHeight() != height) {
            recreate = true;
            if (state.offscreen_3d_view) |v| v.release();
            tex.release();
        }
    } else {
        recreate = true;
    }

    if (recreate) {
        state.offscreen_3d_texture = state.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("zigote_3d_offscreen"),
            // copy_src lets the dev-tooling framebuffer capture (ZIGOTE_SHOT) read this target back.
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.render_attachment |
                wgpu.TextureUsages.copy_src | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = state.wgpu_config.format,
            .mip_level_count = 1,
            .sample_count = 1,
        });
        if (state.offscreen_3d_texture == null) return 0;
        state.offscreen_3d_view = state.offscreen_3d_texture.?.createView(null);
        if (state.offscreen_3d_view == null) return 0;
    }

    // The editor uses this immediate offscreen path rather than renderFrameV2.
    const internal_w: u32 = width;
    const internal_h: u32 = height;

    const g3d = ensure3d(state) orelse return 0;

    state.world.updateTransforms();
    g3d.taa_path_supported = true;

    var geometry_encoder = state.device.createCommandEncoder(&.{}) orelse return 0;
    g3d.beginScene(
        &state.world,
        state.queue,
        geometry_encoder,
        internal_w,
        internal_h,
        state.selected_node_ptr,
    ) catch {
        geometry_encoder.release();
        return 0;
    };
    g3d.renderShadowPass(&state.world, state.queue, geometry_encoder) catch {
        geometry_encoder.release();
        return 0;
    };
    g3d.renderSkyPass(geometry_encoder, state.world.active_camera != null) catch {
        geometry_encoder.release();
        return 0;
    };
    _ = g3d.renderSceneGeometry(
        &state.world,
        state.queue,
        geometry_encoder,
        internal_w,
        internal_h,
    ) catch {
        geometry_encoder.release();
        return 0;
    };
    var geometry_cmd = geometry_encoder.finish(&.{}) orelse {
        geometry_encoder.release();
        return 0;
    };
    geometry_encoder.release();
    state.queue.submit(&.{geometry_cmd});
    geometry_cmd.release();

    var post_encoder = state.device.createCommandEncoder(&.{}) orelse return 0;
    g3d.renderPostProcess(
        state.queue,
        post_encoder,
        state.offscreen_3d_view.?,
        state.offscreen_3d_texture,
    );
    var post_cmd = post_encoder.finish(&.{}) orelse {
        post_encoder.release();
        return 0;
    };
    post_encoder.release();
    state.queue.submit(&.{post_cmd});
    post_cmd.release();

    // Cache key for 3D is a magical huge number
    const magic_key: u64 = 0x3D3D3D3D3D3D3D3D;

    // We must manually add it to `gpu_ui.image_cache` if not present
    if (!state.gpu_ui.image_cache.contains(magic_key)) {
        // Create bind group
        if (state.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("zigote_3d_bind_group"),
            .layout = state.gpu_ui.text.bindGroupLayout(),
            .entries = &.{
                .{ .binding = 0, .texture_view = state.offscreen_3d_view.? },
                .{ .binding = 1, .sampler = state.gpu_ui.text.getSampler() },
            },
            .entry_count = 2,
        })) |bg| {
            state.gpu_ui.image_cache.put(magic_key, .{
                .texture = state.offscreen_3d_texture.?,
                .texture_view = state.offscreen_3d_view.?,
                .bind_group = bg,
                // state.offscreen_3d_* owns these — the cache is only how a paint command reaches
                // them by key.
                .pinned = true,
                .borrowed = true,
            }) catch {
                bg.release();
            };
        }
    } else {
        // If size changed, we need to update the cache entry!
        // Actually, if we just recreate it, `gpu_ui.image_cache` holds the old pointer!
        // We need to update it.
        if (recreate) {
            const old = state.gpu_ui.image_cache.get(magic_key).?;
            old.bind_group.release();
            if (state.device.createBindGroup(&.{
                .label = wgpu.StringView.fromSlice("zigote_3d_bind_group"),
                .layout = state.gpu_ui.text.bindGroupLayout(),
                .entries = &.{
                    .{ .binding = 0, .texture_view = state.offscreen_3d_view.? },
                    .{ .binding = 1, .sampler = state.gpu_ui.text.getSampler() },
                },
                .entry_count = 2,
            })) |bg| {
                state.gpu_ui.image_cache.put(magic_key, .{
                    .texture = state.offscreen_3d_texture.?,
                    .texture_view = state.offscreen_3d_view.?,
                    .bind_group = bg,
                    .pinned = true,
                    .borrowed = true,
                }) catch {
                    bg.release();
                };
            }
        }
    }

    return magic_key;
}

// ── Offscreen render capture (dev tooling) ─────────────────────────────────────
// Set ZIGOTE_SHOT=/path/out.bmp to dump the 3D viewport's tonemapped output to a 24-bit
// BMP exactly once, at frame ZIGOTE_SHOT_FRAME (default 120 — late enough for the glTF
// async load + TAA accumulation to settle). Self-contained: triggered from the per-frame
// render path, no C# plumbing, no screen-recording permission. Used to compare the renderer
// against reference images while tuning. The offscreen target already carries copy_src.
var g_shot_done: bool = false;

fn shotU16LE(b: []u8, v: u16) void {
    b[0] = @truncate(v);
    b[1] = @truncate(v >> 8);
}
fn shotU32LE(b: []u8, v: u32) void {
    b[0] = @truncate(v);
    b[1] = @truncate(v >> 8);
    b[2] = @truncate(v >> 16);
    b[3] = @truncate(v >> 24);
}

fn captureTextureBmp(state: *EngineState, tex: *wgpu.Texture, path: []const u8) bool {
    const w = tex.getWidth();
    const h = tex.getHeight();
    if (w == 0 or h == 0) return false;

    const fmt = state.wgpu_config.format;
    const is_bgra = (fmt == .bgra8_unorm or fmt == .bgra8_unorm_srgb);

    // wgpu requires the readback row stride aligned to 256 bytes.
    const unpadded: u32 = w * 4;
    const padded: u32 = (unpadded + 255) & ~@as(u32, 255);
    const buf_size: u64 = @as(u64, padded) * @as(u64, h);

    const rb = state.device.createBuffer(&.{
        .label = wgpu.StringView.fromSlice("capture readback"),
        .usage = wgpu.BufferUsages.map_read | wgpu.BufferUsages.copy_dst,
        .size = buf_size,
    }) orelse return false;
    defer rb.release();

    const encoder = state.device.createCommandEncoder(&.{}) orelse return false;
    encoder.copyTextureToBuffer(
        &.{ .texture = tex, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &.{ .buffer = rb, .layout = .{ .offset = 0, .bytes_per_row = padded, .rows_per_image = h } },
        &.{ .width = w, .height = h, .depth_or_array_layers = 1 },
    );
    const cmd = encoder.finish(&.{}) orelse {
        encoder.release();
        return false;
    };
    encoder.release();
    state.queue.submit(&.{cmd});
    cmd.release();

    const MapCtx = struct { done: bool = false, ok: bool = false };
    var mc = MapCtx{};
    const cb = struct {
        fn f(status: wgpu.MapAsyncStatus, _: wgpu.StringView, ud1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
            const c: *MapCtx = @ptrCast(@alignCast(ud1.?));
            c.done = true;
            c.ok = (status == .success);
        }
    }.f;
    _ = rb.mapAsync(wgpu.MapModes.read, 0, @intCast(buf_size), .{ .callback = cb, .userdata1 = &mc });
    var guard: u32 = 0;
    while (!mc.done and guard < 500000) : (guard += 1) {
        _ = state.device.poll(true, null);
    }
    if (!mc.ok) return false;
    const mapped = rb.getConstMappedRange(0, @intCast(buf_size)) orelse return false;
    const src: [*]const u8 = @ptrCast(mapped);

    // 24-bit bottom-up BMP (the most universally readable; converted to PNG with `sips`).
    const stride: u32 = ((w * 3 + 3) / 4) * 4;
    const img_size: u32 = stride * h;
    const file_size: u32 = 54 + img_size;
    const out = state.allocator.alloc(u8, file_size) catch {
        rb.unmap();
        return false;
    };
    defer state.allocator.free(out);
    @memset(out, 0);
    out[0] = 'B';
    out[1] = 'M';
    shotU32LE(out[2..6], file_size);
    shotU32LE(out[10..14], 54); // pixel data offset
    shotU32LE(out[14..18], 40); // info header size
    shotU32LE(out[18..22], w);
    shotU32LE(out[22..26], h); // positive height → bottom-up
    shotU16LE(out[26..28], 1); // planes
    shotU16LE(out[28..30], 24); // bpp
    shotU32LE(out[34..38], img_size);
    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const src_row = src + @as(usize, y) * padded;
        const dst_row = h - 1 - y; // flip for bottom-up storage
        var dst_off: usize = 54 + @as(usize, dst_row) * stride;
        var x: u32 = 0;
        while (x < w) : (x += 1) {
            const s = src_row + @as(usize, x) * 4;
            const bb: u8 = if (is_bgra) s[0] else s[2];
            const gg: u8 = s[1];
            const rr: u8 = if (is_bgra) s[2] else s[0];
            out[dst_off] = bb;
            out[dst_off + 1] = gg;
            out[dst_off + 2] = rr;
            dst_off += 3;
        }
    }
    rb.unmap();

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out }) catch return false;
    return true;
}

// ── GPU memory diagnostics (macOS / Metal) ──────────────────────────────────────────────────
// `currentAllocatedSize` is Metal's own device-wide GPU-allocation counter — the exact figure the
// Metal HUD and Instruments "Game Memory" report. We read it through the objc runtime (Metal /
// Foundation / libobjc are already linked for wgpu's Metal backend). On single-GPU Apple Silicon the
// system-default device is the one wgpu allocates on, so this reflects the whole process's GPU usage.
const MetalMem = if (@import("builtin").os.tag == .macos) struct {
    extern fn MTLCreateSystemDefaultDevice() ?*anyopaque;
    extern fn sel_registerName(name: [*:0]const u8) ?*anyopaque;
    extern fn objc_msgSend() void;
    var device: ?*anyopaque = null;
    fn dev() ?*anyopaque {
        if (device == null) device = MTLCreateSystemDefaultDevice();
        return device;
    }
    fn u64Getter(sel_name: [*:0]const u8) u64 {
        const d = dev() orelse return 0;
        const sel = sel_registerName(sel_name) orelse return 0;
        const send: *const fn (?*anyopaque, ?*anyopaque) callconv(.c) u64 = @ptrCast(&objc_msgSend);
        return send(d, sel);
    }
    fn allocatedBytes() u64 {
        return u64Getter("currentAllocatedSize");
    }
    fn maxWorkingSet() u64 {
        return u64Getter("recommendedMaxWorkingSetSize");
    }
} else struct {
    fn allocatedBytes() u64 {
        return 0;
    }
    fn maxWorkingSet() u64 {
        return 0;
    }
};

/// Device-wide GPU allocated bytes (Metal `currentAllocatedSize`). 0 off macOS / before device init.
export fn zigote_debug_gpu_allocated_bytes(handle: u64) u64 {
    _ = handle;
    return MetalMem.allocatedBytes();
}

/// Set ZIGOTE_GPU_MEM=1 to log the Metal device allocation + the renderer's own target breakdown
/// every 120 frames. Pure diagnostics; no effect when the env var is unset.
fn logGpuMem(state: *EngineState) void {
    if (std.c.getenv("ZIGOTE_GPU_MEM") == null) return;
    if (state.frame_index % 120 != 0) return;
    const mb: f64 = 1024.0 * 1024.0;
    const total = MetalMem.allocatedBytes();
    const maxws = MetalMem.maxWorkingSet();
    if (state.gpu_3d) |g3d| {
        const m = g3d.targetMemoryBytes();
        std.log.info("[gpu-mem] f{d}: METAL device={d:.1}MB (max {d:.0}MB) | 3D targets={d:.1}MB @ {d}x{d} [shadow={d:.1} point={d:.1} env={d:.1} depth+msaa={d:.1} hdr={d:.1} gbuffer={d:.1} post={d:.1} taa={d:.1}]", .{
            state.frame_index,
            @as(f64, @floatFromInt(total)) / mb,
            @as(f64, @floatFromInt(maxws)) / mb,
            @as(f64, @floatFromInt(m.total)) / mb,
            g3d.depth_width,
            g3d.depth_height,
            @as(f64, @floatFromInt(m.shadow)) / mb,
            @as(f64, @floatFromInt(m.point_shadow)) / mb,
            @as(f64, @floatFromInt(m.env)) / mb,
            @as(f64, @floatFromInt(m.depth_msaa)) / mb,
            @as(f64, @floatFromInt(m.hdr)) / mb,
            @as(f64, @floatFromInt(m.gbuffer)) / mb,
            @as(f64, @floatFromInt(m.post)) / mb,
            @as(f64, @floatFromInt(m.taa)) / mb,
        });
    } else {
        const um = state.gpu_ui.memoryBytes();
        // Estimated swapchain footprint: surface size × 4 B × ~3 buffers (owned by wgpu, not by us).
        const sw: u64 = state.wgpu_config.width;
        const sh: u64 = state.wgpu_config.height;
        const swapchain_est: u64 = sw * sh * 4 * 3;
        std.log.info("[gpu-mem] f{d}: METAL device={d:.1}MB (max {d:.0}MB) | UI targets={d:.2}MB [coverage-atlas={d:.2} emoji-atlas={d:.2} scene={d:.2} backdrop={d:.2} vtx-bufs={d:.2} images={d}] | surface {d}x{d} (~swapchain {d:.1}MB)", .{
            state.frame_index,
            @as(f64, @floatFromInt(total)) / mb,
            @as(f64, @floatFromInt(maxws)) / mb,
            @as(f64, @floatFromInt(um.total)) / mb,
            @as(f64, @floatFromInt(um.coverage_atlas)) / mb,
            @as(f64, @floatFromInt(um.emoji_atlas)) / mb,
            @as(f64, @floatFromInt(um.scene)) / mb,
            @as(f64, @floatFromInt(um.backdrop)) / mb,
            @as(f64, @floatFromInt(um.vertex_buffers)) / mb,
            um.image_count,
            sw,
            sh,
            @as(f64, @floatFromInt(swapchain_est)) / mb,
        });
    }
}

fn tryAutoCapture(state: *EngineState) void {
    if (g_shot_done) return;
    const path_c = std.c.getenv("ZIGOTE_SHOT") orelse return;
    const path = std.mem.span(path_c);
    var target: u32 = 120;
    if (std.c.getenv("ZIGOTE_SHOT_FRAME")) |fs| {
        target = std.fmt.parseInt(u32, std.mem.span(fs), 10) catch 120;
    }
    if (state.frame_index < target) return;
    g_shot_done = true;
    const tex = state.offscreen_3d_texture orelse {
        std.log.info("[zigote] auto-capture skipped: no 3D offscreen target", .{});
        return;
    };
    const ok = captureTextureBmp(state, tex, path);
    std.log.info("[zigote] auto-capture {s} -> {s}", .{ if (ok) "ok" else "FAILED", path });
}

/// Dev tooling: render the currently-submitted main UI paint list (state.paint_list) into a fresh
/// offscreen target and dump it to a 24-bit BMP — the 2D counterpart of the ZIGOTE_SHOT 3D capture,
/// giving the 2D paint path a headless golden-image regression seam it otherwise lacks. Additive: it
/// reuses the existing renderToTexture + BMP readback and never touches the live present path. Submit
/// a frame's paint commands first (zigote_submit_paint_commands), then call this.
export fn zigote_capture_ui_bmp(
    handle: u64,
    path_ptr: [*c]const u8,
    path_len: usize,
    width: u32,
    height: u32,
    scale: f32,
) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    if (width == 0 or height == 0 or path_ptr == null) return .err;
    const path = path_ptr[0..path_len];

    // Offscreen target in the swapchain format (so the UI pipeline matches) + copy_src for readback.
    const tex = state.device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote ui capture"),
        .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.copy_src | wgpu.TextureUsages.texture_binding,
        .dimension = .@"2d",
        .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
        .format = state.wgpu_config.format,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return .err;
    defer tex.release();

    const view = tex.createView(null) orelse return .err;
    defer view.release();

    wgpu_renderer.wgpu.renderToTexture(
        state.device,
        state.queue,
        &state.gpu_ui,
        state.paint_list,
        view,
        width,
        height,
        scale,
    ) catch |err| {
        std.log.err("zigote_capture_ui_bmp render failed: {}", .{err});
        return .err;
    };

    return if (captureTextureBmp(state, tex, path)) .ok else .err;
}

/// Get the current surface pixel dimensions.
export fn zigote_get_size(handle: u64, out_w: *u32, out_h: *u32) void {
    const state = stateFromHandle(handle) orelse {
        out_w.* = 0;
        out_h.* = 0;
        return;
    };
    const sz = currentPixelSize(state);
    out_w.* = sz[0];
    out_h.* = sz[1];
}

/// Enable SDL3 text-input mode for the current window.
/// Call when a text field gains focus so SDL_EVENT_TEXT_INPUT events are generated.
export fn zigote_start_text_input(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    sdl3.keyboard.startTextInput(state.window) catch {};
}

/// Disable SDL3 text-input mode for the current window.
/// Call when no text field is focused.
export fn zigote_stop_text_input(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    sdl3.keyboard.stopTextInput(state.window) catch {};
}

/// Tell the native IME where the active caret is so its candidate window does not cover the text.
export fn zigote_set_text_input_area(handle: u64, x: i32, y: i32, w: i32, h: i32, cursor: i32) void {
    const state = stateFromHandle(handle) orelse return;
    sdl3.keyboard.setTextInputArea(state.window, .{ .x = x, .y = y, .w = w, .h = h }, cursor) catch {};
}

// ── Secondary OS windows (UI-only) ────────────────────────────────────────────
//
// A secondary window runs only the 2D paint path: C# submits a per-window paint list and calls
// zigote_window_render, which draws it through the window's own GpuUi and presents its surface.
// The 3D scene / render graph stay bound to the main window. Events carry ZgEvent.window_id so
// the C# side routes input to the right window's widget tree.

fn createSecondaryWindowImpl(
    state: *EngineState,
    width: u32,
    height: u32,
    title_c: [*c]const u8,
) !*SecondaryWindow {
    const title_slice: [:0]const u8 = if (title_c != null) std.mem.span(title_c) else "Zigote";

    var window = try sdl3.video.Window.init(
        title_slice,
        width,
        height,
        // Same alpha channel the main window asked for, or a secondary window (devtools, Settings)
        // would sit next to it with square corners: the per-frame CSD rounding clips only on a
        // window the compositor actually composites (zigote_window_is_transparent).
        .{
            .resizable = true,
            .high_pixel_density = true,
            .transparent = pending_transparent_window,
        },
    );
    errdefer window.deinit();

    const metal_view: ?sdl3.MetalView = if (@import("builtin").os.tag == .macos or @import("builtin").os.tag == .ios)
        (sdl3.MetalView.init(window) orelse return error.MetalViewUnavailable)
    else
        null;
    errdefer if (metal_view) |mv| mv.deinit();

    const metal_layer: ?*anyopaque = if (metal_view) |mv|
        @ptrCast(mv.getLayer() orelse return error.MetalLayerUnavailable)
    else
        null;

    var surface = try createNativeSurface(state.instance, window, metal_layer);
    errdefer surface.release();

    var capabilities: wgpu.SurfaceCapabilities = undefined;
    if (surface.getCapabilities(state.adapter, &capabilities) != .success) {
        return error.WgpuSurfaceCapabilitiesUnavailable;
    }
    defer capabilities.freeMembers();
    if (capabilities.format_count == 0) return error.WgpuNoSurfaceFormats;
    const format = pickSurfaceFormat(capabilities.formats[0..capabilities.format_count]);

    const pixel_size = try window.getSizeInPixels();
    var config = wgpu.SurfaceConfiguration{
        .device = state.device,
        .format = format,
        .width = @intCast(pixel_size[0]),
        .height = @intCast(pixel_size[1]),
        .present_mode = .fifo,
        .alpha_mode = if (pending_transparent_window)
            pickAlphaMode(&capabilities, true)
        else if (capabilities.alpha_mode_count > 0)
            capabilities.alpha_modes[0]
        else
            .auto,
    };
    surface.configure(&config);

    // Fresh GpuUi seeded with the boot font, then replay every font registered since (loadFont
    // calls, emoji family) — the window's FreeType face table starts empty while C# paint
    // commands reference faces by family name, expecting the same set as the main window.
    const boot = state.loaded_fonts.items[0];
    const font_asset = zg.FontAsset.fromPlatform(boot.name, boot.path);
    const fonts: []const zg.FontAsset = &.{font_asset};
    var gpu_ui = try GpuUi.init(state.allocator, state.device, format, fonts, state.fontName());
    errdefer gpu_ui.deinit();

    if (state.loaded_fonts.items.len > 1) {
        for (state.loaded_fonts.items[1..]) |lf| {
            gpu_ui.text.loadFontFromCPath(lf.name, lf.path.ptr) catch |err| {
                std.log.warn("zigote: window font replay '{s}' failed: {}", .{ lf.name, err });
            };
        }
    }
    if (state.emoji_family) |fam| gpu_ui.text.addEmojiFontFamily(fam);
    for (state.fallback_families.items) |fam| gpu_ui.text.addFallbackFontFamily(fam);

    // Clear to alpha 0 so the rounded-corner cutouts show the desktop (see the same line in
    // zigote_init) — only where SDL granted transparency and the surface premultiplies.
    gpu_ui.transparent_clear = pending_transparent_window and
        config.alpha_mode == .premultiplied and
        sdl3.c.SDL_GetWindowFlags(window.value) & sdl3.c.SDL_WINDOW_TRANSPARENT != 0;

    const win = try state.allocator.create(SecondaryWindow);
    errdefer state.allocator.destroy(win);
    win.* = .{
        .id = window.getId() catch 0,
        .window = window,
        .metal_view = metal_view,
        .surface = surface,
        .config = config,
        .gpu_ui = gpu_ui,
        .paint_list = .{},
        .overlay_paint_list = .{},
        .frame_index = 0,
    };
    try state.windows.put(@intFromPtr(win), win);
    return win;
}

/// Create a secondary UI-only OS window. out_window receives the opaque window handle to pass to
/// the other zigote_window_* calls. width/height are logical pixels (like zigote_init).
export fn zigote_window_create(
    handle: u64,
    width: u32,
    height: u32,
    title: [*c]const u8,
    out_window: *u64,
) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    const win = createSecondaryWindowImpl(state, width, height, title) catch |err| {
        std.log.err("zigote_window_create failed: {}", .{err});
        out_window.* = 0;
        return .err;
    };
    out_window.* = @intFromPtr(win);
    return .ok;
}

/// Destroy a secondary window and free all its GPU/window resources. The handle is dead after this.
export fn zigote_window_destroy(handle: u64, window_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    _ = state.windows.remove(window_handle);
    win.deinit(state.allocator);
    state.allocator.destroy(win);
}

/// SDL window id of a secondary window (matches ZgEvent.window_id).
export fn zigote_window_id(handle: u64, window_handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    const win = windowFromHandle(state, window_handle) orelse return 0;
    return win.id;
}

/// SDL window id of the main engine window (matches ZgEvent.window_id).
export fn zigote_main_window_id(handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    return state.main_window_id;
}

/// Current swapchain pixel size of a secondary window.
export fn zigote_window_pixel_size(handle: u64, window_handle: u64, out_w: *u32, out_h: *u32) void {
    out_w.* = 0;
    out_h.* = 0;
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    out_w.* = win.config.width;
    out_h.* = win.config.height;
}

/// Display scale factor of a secondary window (e.g. 2.0 on Retina).
export fn zigote_window_scale(handle: u64, window_handle: u64) f32 {
    const state = stateFromHandle(handle) orelse return 1.0;
    const win = windowFromHandle(state, window_handle) orelse return 1.0;
    return win.window.getDisplayScale() catch 1.0;
}

/// Raise a secondary window above other windows and give it input focus.
export fn zigote_window_raise(handle: u64, window_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    win.window.raise() catch {};
}

/// Screen position (top-left, logical desktop coordinates) of a secondary window. With the
/// window-relative pointer coordinates from input events this gives the global cursor position —
/// the basis for cross-window panel drag-and-drop.
export fn zigote_window_position(handle: u64, window_handle: u64, out_x: *i32, out_y: *i32) void {
    out_x.* = 0;
    out_y.* = 0;
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    const pos = win.window.getPosition() catch return;
    out_x.* = @intCast(pos[0]);
    out_y.* = @intCast(pos[1]);
}

/// Move a secondary window to an absolute screen position (logical desktop coordinates).
export fn zigote_window_set_position(handle: u64, window_handle: u64, x: i32, y: i32) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    win.window.setPosition(.{ .absolute = x }, .{ .absolute = y }) catch {};
}

/// Hide or show the MAIN window without destroying it.
///
/// What a media player needs to keep playing after its window is closed: destroying the window
/// would take the render surface and the event loop's reason to exist with it, while hiding leaves
/// everything running with nothing on screen. Showing raises as well, because the only reason to
/// show a hidden window is that the user asked for it back.
export fn zigote_main_window_set_visible(handle: u64, visible: u32) void {
    const state = stateFromHandle(handle) orelse return;
    if (visible == 0) {
        state.window.hide() catch {};
        return;
    }
    state.window.show() catch {};
    state.window.raise() catch {};
}

/// Screen position (top-left, logical desktop coordinates) of the MAIN engine window.
export fn zigote_main_window_position(handle: u64, out_x: *i32, out_y: *i32) void {
    out_x.* = 0;
    out_y.* = 0;
    const state = stateFromHandle(handle) orelse return;
    const pos = state.window.getPosition() catch return;
    out_x.* = @intCast(pos[0]);
    out_y.* = @intCast(pos[1]);
}

/// Retitle a secondary window. title must be null-terminated UTF-8.
export fn zigote_window_set_title(handle: u64, window_handle: u64, title: [*c]const u8) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    if (title == null) return;
    const title_slice: [:0]const u8 = std.mem.span(title);
    win.window.setTitle(title_slice) catch {};
}

/// Native parent handle of a window, for embedding platform child views (webviews, native
/// controls) OVER the engine's surface — the overlay path: the host positions the child against
/// widget layout each frame, and the child always draws above engine content.
///
/// out_kind: 0 = unavailable, 1 = Win32 (ptr1 = HWND), 2 = macOS (ptr1 = NSWindow*),
/// 3 = X11 (ptr1 = Display*, ptr2 = Window), 4 = Wayland (ptr1 = wl_display*,
/// ptr2 = wl_surface*), 5 = Android (ptr1 = ANativeWindow*), 6 = iOS (ptr1 = UIWindow*).
///
/// Wayland is reported for completeness only: a foreign toolkit's view cannot be parented into
/// another client's wl_surface, so embedding on Linux needs the X11 driver (SDL_VIDEO_DRIVER=x11,
/// which is XWayland on a Wayland desktop). Android attachment likewise goes through
/// SDLActivity.getLayout() on the Java side rather than the ANativeWindow.
/// window_handle 0 = the main window.
export fn zigote_window_native_parent(handle: u64, window_handle: u64, out_kind: *u32, out_ptr1: *u64, out_ptr2: *u64) void {
    out_kind.* = 0;
    out_ptr1.* = 0;
    out_ptr2.* = 0;
    const state = stateFromHandle(handle) orelse return;
    const window = if (window_handle == 0)
        state.window
    else if (windowFromHandle(state, window_handle)) |win|
        win.window
    else
        return;
    const props = sdl3.c.SDL_GetWindowProperties(window.value);

    switch (@import("builtin").os.tag) {
        .windows => {
            const hwnd = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WIN32_HWND_POINTER, null) orelse return;
            out_kind.* = 1;
            out_ptr1.* = @intFromPtr(hwnd);
        },
        .macos => {
            const nswindow = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, null) orelse return;
            out_kind.* = 2;
            out_ptr1.* = @intFromPtr(nswindow);
        },
        .ios => {
            const uiwindow = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_UIKIT_WINDOW_POINTER, null) orelse return;
            out_kind.* = 6;
            out_ptr1.* = @intFromPtr(uiwindow);
        },
        .linux => {
            if (comptime @import("builtin").abi.isAndroid()) {
                const a_window = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, null) orelse return;
                out_kind.* = 5;
                out_ptr1.* = @intFromPtr(a_window);
                return;
            }
            // Same driver detection order as createNativeSurface: whichever SDL is actually on.
            if (sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WAYLAND_DISPLAY_POINTER, null)) |wl_display| {
                const wl_surface = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_WAYLAND_SURFACE_POINTER, null) orelse return;
                out_kind.* = 4;
                out_ptr1.* = @intFromPtr(wl_display);
                out_ptr2.* = @intFromPtr(wl_surface);
                return;
            }
            const x_display = sdl3.c.SDL_GetPointerProperty(props, sdl3.c.SDL_PROP_WINDOW_X11_DISPLAY_POINTER, null) orelse return;
            const x_window = sdl3.c.SDL_GetNumberProperty(props, sdl3.c.SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0);
            out_kind.* = 3;
            out_ptr1.* = @intFromPtr(x_display);
            out_ptr2.* = @intCast(x_window);
        },
        else => {},
    }
}

/// Store a secondary window's paint commands for its next zigote_window_render.
export fn zigote_window_submit_paint(
    handle: u64,
    window_handle: u64,
    commands: [*]const ZgPaintCommand,
    count: u32,
) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    fillPaintList(state, &win.paint_list, commands, count) catch |err| {
        std.log.err("zigote_window_submit_paint failed: {}", .{err});
    };
}

/// Like zigote_window_submit_paint but for the window's overlay layer (popups, tooltips) —
/// rendered after the main list, mirroring the main window's root/overlay split.
export fn zigote_window_submit_overlay(
    handle: u64,
    window_handle: u64,
    commands: [*]const ZgPaintCommand,
    count: u32,
) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    fillPaintList(state, &win.overlay_paint_list, commands, count) catch |err| {
        std.log.err("zigote_window_submit_overlay failed: {}", .{err});
    };
}

/// Render the window's submitted paint lists through its own GpuUi and present its surface.
/// scale converts the paint lists' logical coordinates to this window's pixels.
export fn zigote_window_render(handle: u64, window_handle: u64, scale: f32) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    const win = windowFromHandle(state, window_handle) orelse return .err;
    wgpu_renderer.wgpu.renderFrame(
        win.surface,
        state.device,
        state.queue,
        &win.gpu_ui,
        win.paint_list,
        win.overlay_paint_list,
        win.config.width,
        win.config.height,
        scale,
        win.frame_index,
        &[_]zg.Rect{},
        false, // secondary window: no partial repaint, no glass → render straight to its swapchain
    ) catch |err| {
        std.log.err("zigote_window_render failed: {}", .{err});
        return .err;
    };
    win.frame_index +%= 1;
    return .ok;
}

/// Enable SDL3 text-input mode for a secondary window (IME/keyboard routing follows OS focus).
export fn zigote_window_start_text_input(handle: u64, window_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    sdl3.keyboard.startTextInput(win.window) catch {};
}

/// Disable SDL3 text-input mode for a secondary window.
export fn zigote_window_stop_text_input(handle: u64, window_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    sdl3.keyboard.stopTextInput(win.window) catch {};
}

/// IME caret anchor for a secondary window (see zigote_set_text_input_area).
export fn zigote_window_set_text_input_area(
    handle: u64,
    window_handle: u64,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    cursor: i32,
) void {
    const state = stateFromHandle(handle) orelse return;
    const win = windowFromHandle(state, window_handle) orelse return;
    sdl3.keyboard.setTextInputArea(win.window, .{ .x = x, .y = y, .w = w, .h = h }, cursor) catch {};
}

/// Current OS appearance: 0 unknown, 1 light, 2 dark. Live changes also arrive as
/// EVT_SYSTEM_THEME events from zigote_poll_events.
export fn zigote_get_system_theme(handle: u64) u32 {
    _ = stateFromHandle(handle) orelse return 0;
    return systemThemeValue();
}

/// Host scroll orientation ("natural scroll" setting): 0 unknown, 1 normal, 2 flipped/natural.
/// SDL surfaces this only per wheel event, so the value is latched from the last mouse-wheel event
/// and is 0 until the user first scrolls.
export fn zigote_get_scroll_orientation(handle: u64) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    return state.scroll_orientation;
}

/// Drop all text caches (shaped runs, glyph atlases) on the main window AND every secondary
/// window, forcing a clean re-shape on the next frame. Call after a wholesale text sizing change
/// (live UI font-scale switch) — the same invalidation a font face swap performs.
export fn zigote_text_reset_caches(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    state.gpu_ui.text.resetAllTextCaches();
    var it = state.windows.valueIterator();
    while (it.next()) |win| win.*.gpu_ui.text.resetAllTextCaches();
}

/// Get UTF-8 clipboard text. Copies up to capacity-1 bytes into the caller buffer (NUL-terminated) and
/// returns the FULL byte length of the clipboard text (excluding NUL) — which may exceed what was
/// written. The C# side detects a short buffer (return >= capacity) and re-queries with a larger one,
/// so arbitrarily long clipboard content is never silently truncated. Returns 0 if empty / on error.
export fn zigote_get_clipboard(buf: [*]u8, capacity: u32) u32 {
    const text = sdl3.clipboard.getText() catch return 0;
    defer sdl3.free(text);
    if (capacity > 0) {
        const n = @min(text.len, @as(usize, capacity) -| 1);
        @memcpy(buf[0..n], text[0..n]);
        buf[n] = 0;
    }
    return @intCast(text.len);
}

/// Set UTF-8 clipboard text. text_ptr must be null-terminated.
export fn zigote_set_clipboard(text_ptr: [*c]const u8) void {
    const text: [:0]const u8 = std.mem.span(text_ptr);
    sdl3.clipboard.setText(text) catch {};
}

/// Shut down the engine and free all resources.
export fn zigote_shutdown(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;

    // Close the FFI gate FIRST: async image decodes on .NET worker threads can still be in flight
    // (an app quit while covers were decoding), and every entry point re-validates the handle via
    // stateFromHandle. Anyone already past the gate is drained by taking image_lock below.
    live_engine.store(0, .release);

    // Secondary windows first — they borrow the shared device/queue released below.
    {
        var win_it = state.windows.valueIterator();
        while (win_it.next()) |win| {
            win.*.deinit(state.allocator);
            state.allocator.destroy(win.*);
        }
        state.windows.deinit();
    }
    for (state.loaded_fonts.items) |lf| {
        state.allocator.free(lf.name);
        state.allocator.free(lf.path);
    }
    state.loaded_fonts.deinit(state.allocator);
    if (state.emoji_family) |fam| state.allocator.free(fam);
    for (state.fallback_families.items) |fam| state.allocator.free(fam);
    state.fallback_families.deinit(state.allocator);

    // Under image_lock: waits out any worker already inside registerImage/release_texture; workers
    // arriving after this are rejected by the gate (or the re-check those paths do under the lock).
    state.image_lock.lock();
    var it = state.image_registry.iterator();
    while (it.next()) |entry| {
        // Empty = the CPU copy was already dropped after upload (drainImageUploads).
        if (entry.value_ptr.*.pixels.len > 0) state.allocator.free(entry.value_ptr.*.pixels);
    }
    state.image_registry.deinit();
    state.pending_image_releases.deinit(state.allocator);
    state.pending_image_updates.deinit(state.allocator);
    state.image_lock.unlock();
    state.node_handles.deinit();
    state.poll_text.deinit(state.allocator);

    // Remove the 3D offscreen entry from the image cache BEFORE gpu_ui.deinit() releases
    // all cache entries — otherwise the texture and view get double-released (crash/freeze).
    {
        const magic_key: u64 = 0x3D3D3D3D3D3D3D3D;
        if (state.gpu_ui.image_cache.fetchRemove(magic_key)) |kv| {
            kv.value.bind_group.release();
            // texture_view and texture released below via offscreen_3d_*
        }
    }
    if (state.offscreen_3d_view) |v| v.release();
    if (state.offscreen_3d_texture) |t| t.release();

    if (state.physics) |phys| {
        physics_ffi.deinit(phys);
        state.physics = null;
    }

    // Under the lock: a loader thread may still be inside an audio call when the host shuts down.
    {
        state.audio_lock.lock();
        defer state.audio_lock.unlock();
        if (state.audio) |a| {
            audio_ffi.deinit(a);
            state.audio = null;
        }
        state.audio_scanned = true; // nothing may lazily reopen a device on a shutting-down engine
    }

    // Clean up render textures (remove their image_cache entries first to avoid double-release)
    var rt_it = state.render_textures.iterator();
    while (rt_it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (state.gpu_ui.image_cache.fetchRemove(key)) |kv| kv.value.bind_group.release();
        entry.value_ptr.deinit(state.allocator);
    }
    state.render_textures.deinit();
    state.blur_requests.deinit(state.allocator);
    if (state.gaussian_blur) |*gb| gb.deinit();

    state.transient_pool.deinit();
    state.render_graph.deinit();
    state.world.deinit();
    state.paint_list.deinit(state.allocator);
    state.overlay_paint_list.deinit(state.allocator);
    if (state.gpu_3d) |g| {
        g.deinit();
        state.allocator.destroy(g);
        state.gpu_3d = null;
    }
    state.gpu_ui.deinit();
    state.gpu_backend.deinit(); // drops any per-frame swapchain view it may hold
    state.surface.unconfigure();
    state.queue.release();
    state.device.release();
    state.adapter.release();
    state.surface.release();
    state.instance.release();
    if (state.metal_view) |mv| mv.deinit();
    state.window.deinit();
    sdl3.quit(.{ .video = true, .events = true });
    sdl3.shutdown();

    // Deliberately NOT destroyed: a worker that passed the gate before it closed may still spin on
    // image_lock (it then re-checks the gate and bails). Freeing the state under it would turn that
    // benign late call into use-after-free. One EngineState leaks per init/shutdown cycle — a few KB,
    // once, at process exit in practice.
    // ponytail: intentional leak; refcount the handle if init/shutdown ever cycles in-process.
}

/// The next handle for anything that can end up as a key in `gpu_ui.image_cache`.
///
/// There is one counter because there is one cache. Images and render textures used to have a
/// counter each, both starting at 1, and a paint command carries only the number — so image 1 and
/// render texture 1 were the same slot: whichever registered last replaced the other's texture, and
/// releasing the image released the render texture's GPU memory. It surfaced as an intermittent
/// wgpu panic in the blur pass ("Texture[Id(15,1)] does not exist") one frame after a resize, which
/// is where an app is most likely to drop an image and keep a render texture.
fn nextGpuHandle(state: *EngineState) u64 {
    state.image_lock.lock();
    defer state.image_lock.unlock();
    const handle = state.next_gpu_handle;
    state.next_gpu_handle += 1;
    return handle;
}

/// Take ownership of a decoded RGBA buffer and hand back the handle the paint stream refers to.
/// Frees `bytes` and returns 0 if the registry cannot grow, so callers never leak on failure.
fn registerImage(state: *EngineState, w: u32, h: u32, bytes: []u8) u64 {
    state.image_lock.lock();
    defer state.image_lock.unlock();

    // Re-check under the lock: the engine may have shut down while this worker was decoding
    // (zigote_shutdown deinits the registry holding this same lock).
    if (live_engine.load(.acquire) != @intFromPtr(state)) {
        state.allocator.free(bytes);
        return 0;
    }

    // The counter itself, not nextGpuHandle: the lock is already held here and it is not reentrant.
    const img_handle = state.next_gpu_handle;
    state.next_gpu_handle += 1;
    state.image_registry.put(img_handle, .{
        .width = w,
        .height = h,
        .pixels = bytes,
    }) catch {
        state.allocator.free(bytes);
        return 0;
    };
    return img_handle;
}

/// Release a texture handle returned by any zigote_load_texture* call: the CPU pixel copy and the
/// GPU texture both go. Deferred to the end of the current frame (see drainImageReleases), so it
/// is safe to call at any point, including from a widget being disposed mid-layout. Releasing an
/// unknown or already-released handle is a no-op.
export fn zigote_release_texture(handle: u64, image_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (image_handle == 0) return;
    state.image_lock.lock();
    defer state.image_lock.unlock();
    // Re-check under the lock — see registerImage.
    if (live_engine.load(.acquire) != handle) return;
    state.pending_image_releases.append(state.allocator, image_handle) catch {};
}

/// Live texture accounting: how many handles exist, how much decoded RGBA is still held on the CPU
/// (only images not yet painted), and how much is resident on the GPU. An image-heavy app drives its
/// own cache budget off this — and it is what makes a leak visible instead of merely fatal.
export fn zigote_image_stats(handle: u64, out_count: *u32, out_cpu_bytes: *u64, out_gpu_bytes: *u64) void {
    out_count.* = 0;
    out_cpu_bytes.* = 0;
    out_gpu_bytes.* = 0;
    const state = stateFromHandle(handle) orelse return;

    state.image_lock.lock();
    defer state.image_lock.unlock();

    var cpu: u64 = 0;
    var gpu: u64 = 0;
    var it = state.image_registry.iterator();
    while (it.next()) |entry| {
        cpu += entry.value_ptr.pixels.len;
        if (state.gpu_ui.image_cache.contains(entry.key_ptr.*))
            gpu += @as(u64, entry.value_ptr.width) * entry.value_ptr.height * 4;
    }

    out_count.* = state.image_registry.count();
    out_cpu_bytes.* = cpu;
    out_gpu_bytes.* = gpu;
}

/// End-of-frame: drop the CPU copy of every image that now has a GPU texture. Holding both costs
/// a second full RGBA buffer per image — 24 MB for a single 2000×3000 page.
fn drainImageUploads(state: *EngineState) void {
    if (state.gpu_ui.uploaded_keys.items.len == 0) return;

    for (state.gpu_ui.uploaded_keys.items) |key| {
        // Detach the buffer under the lock, free it outside: freeing a 24 MB page can walk the
        // allocator for a while and a decoding worker should not spin on that.
        state.image_lock.lock();
        const pixels = blk: {
            const entry = state.image_registry.getPtr(key) orelse break :blk &.{};
            const p = entry.pixels;
            entry.pixels = &.{};
            break :blk p;
        };
        state.image_lock.unlock();
        if (pixels.len > 0) state.allocator.free(pixels);
    }
    state.gpu_ui.uploaded_keys.clearRetainingCapacity();
}

/// End-of-frame: free everything the app released during the frame.
fn drainImageReleases(state: *EngineState) void {
    // Detach the pending list under the lock before iterating: zigote_release_texture appends from
    // worker threads (widget disposers), and an append mid-iteration reallocates under our feet.
    state.image_lock.lock();
    const keys = state.pending_image_releases.toOwnedSlice(state.allocator) catch {
        state.image_lock.unlock();
        return; // OOM: keep the list intact, retry next frame
    };
    state.image_lock.unlock();
    defer state.allocator.free(keys);

    for (keys) |key| {
        state.image_lock.lock();
        const removed = state.image_registry.fetchRemove(key);
        state.image_lock.unlock();

        if (removed) |kv| {
            if (kv.value.pixels.len > 0) state.allocator.free(kv.value.pixels);
            // gpu_ui is render-thread-only, so this needs no image lock — and must not hold one:
            // releasing wgpu resources is not a short operation.
            _ = state.gpu_ui.releaseCachedImage(key);
        }
    }
}

export fn zigote_load_texture(handle: u64, path_c: [*c]const u8, out_w: *u32, out_h: *u32) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (path_c == null) return 0;
    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 64)) catch return 0;
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse return 0;

    const img_handle = registerImage(state, w, h, bytes);
    if (img_handle == 0) return 0;

    out_w.* = w;
    out_h.* = h;
    return img_handle;
}

export fn zigote_load_texture_mask(handle: u64, path_c: [*c]const u8, out_w: *u32, out_h: *u32) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (path_c == null) return 0;
    const path = std.mem.span(path_c);

    var io_state = std.Io.Threaded.init(state.allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const file_data = std.Io.Dir.cwd().readFileAlloc(io, path, state.allocator, .limited(1024 * 1024 * 64)) catch return 0;
    defer state.allocator.free(file_data);

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, file_data, &w, &h) orelse return 0;

    // Convert grayscale white-on-black image to transparent mask
    var i: usize = 0;
    while (i < bytes.len) : (i += 4) {
        const r = bytes[i];
        const g = bytes[i + 1];
        const b = bytes[i + 2];
        const val = @max(r, @max(g, b));
        bytes[i] = 255;
        bytes[i + 1] = 255;
        bytes[i + 2] = 255;
        bytes[i + 3] = val;
    }

    const img_handle = registerImage(state, w, h, bytes);
    if (img_handle == 0) return 0;

    out_w.* = w;
    out_h.* = h;
    return img_handle;
}

export fn zigote_load_texture_from_memory(handle: u64, data_ptr: [*c]const u8, data_len: usize, out_w: *u32, out_h: *u32) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (data_ptr == null or data_len == 0) return 0;
    const data = data_ptr[0..data_len];

    var w: u32 = 0;
    var h: u32 = 0;
    const bytes = loadTextureBytes(state.allocator, data, &w, &h) orelse return 0;

    const img_handle = registerImage(state, w, h, bytes);
    if (img_handle == 0) return 0;

    out_w.* = w;
    out_h.* = h;
    return img_handle;
}

const ScaledImage = struct { pixels: []u8, width: u32, height: u32 };

/// Box-downsample an RGBA buffer so neither axis exceeds max_dim. Returns null if no scaling is needed.
/// Averages each destination pixel over the source block it covers (cheap, alias-free enough for thumbnails).
fn downsampleRgba(allocator: std.mem.Allocator, src: []const u8, w: u32, h: u32, max_dim: u32) ?ScaledImage {
    const fw: f32 = @floatFromInt(w);
    const fh: f32 = @floatFromInt(h);
    const fmax: f32 = @floatFromInt(max_dim);
    const scale = @min(fmax / fw, fmax / fh);
    if (scale >= 1.0) return null;

    const dw: u32 = @max(1, @as(u32, @intFromFloat(fw * scale)));
    const dh: u32 = @max(1, @as(u32, @intFromFloat(fh * scale)));
    const dst = allocator.alloc(u8, @as(usize, dw) * @as(usize, dh) * 4) catch return null;

    const bx = fw / @as(f32, @floatFromInt(dw));
    const by = fh / @as(f32, @floatFromInt(dh));

    var dy: u32 = 0;
    while (dy < dh) : (dy += 1) {
        const sy0: u32 = @intFromFloat(@as(f32, @floatFromInt(dy)) * by);
        var sy1: u32 = @intFromFloat(@as(f32, @floatFromInt(dy + 1)) * by);
        if (sy1 <= sy0) sy1 = sy0 + 1;
        if (sy1 > h) sy1 = h;
        var dx: u32 = 0;
        while (dx < dw) : (dx += 1) {
            const sx0: u32 = @intFromFloat(@as(f32, @floatFromInt(dx)) * bx);
            var sx1: u32 = @intFromFloat(@as(f32, @floatFromInt(dx + 1)) * bx);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sx1 > w) sx1 = w;

            var r: u32 = 0;
            var g: u32 = 0;
            var b: u32 = 0;
            var a: u32 = 0;
            var n: u32 = 0;
            var yy: u32 = sy0;
            while (yy < sy1) : (yy += 1) {
                const row = @as(usize, yy) * @as(usize, w) * 4;
                var xx: u32 = sx0;
                while (xx < sx1) : (xx += 1) {
                    const i = row + @as(usize, xx) * 4;
                    r += src[i];
                    g += src[i + 1];
                    b += src[i + 2];
                    a += src[i + 3];
                    n += 1;
                }
            }
            if (n == 0) n = 1;
            const di = (@as(usize, dy) * @as(usize, dw) + @as(usize, dx)) * 4;
            dst[di] = @intCast(r / n);
            dst[di + 1] = @intCast(g / n);
            dst[di + 2] = @intCast(b / n);
            dst[di + 3] = @intCast(a / n);
        }
    }

    return ScaledImage{ .pixels = dst, .width = dw, .height = dh };
}

/// One source pixel as RGBA8, straight out of whatever the decoder produced. Covers the formats
/// photographic content actually arrives in; anything exotic returns null and the caller falls back
/// to converting the whole image first.
inline fn samplePixel(pixels: *const zigimg.color.PixelStorage, i: usize) ?[4]u32 {
    return switch (pixels.*) {
        .rgba32 => |p| .{ p[i].r, p[i].g, p[i].b, p[i].a },
        .rgb24 => |p| .{ p[i].r, p[i].g, p[i].b, 255 },
        .bgra32 => |p| .{ p[i].r, p[i].g, p[i].b, p[i].a },
        .bgr24 => |p| .{ p[i].r, p[i].g, p[i].b, 255 },
        .grayscale8 => |p| .{ p[i].value, p[i].value, p[i].value, 255 },
        .grayscale8Alpha => |p| .{ p[i].value, p[i].value, p[i].value, p[i].alpha },
        else => null,
    };
}

/// Box-downsample straight from decoded pixels into the destination, converting per sample.
///
/// This exists so a thumbnail never costs a full-size RGBA buffer. Going through an intermediate
/// meant a 1800×1800 JPEG allocated ~10 MB of rgb24, then ~13 MB of rgba32, then a ~13 MB copy of
/// that, to produce 147 KB — and with a general-purpose allocator those peaks are what the process
/// keeps. Here the only allocation is the destination.
///
/// Returns null when no scaling is needed, or when the pixel format is one samplePixel does not
/// know; the caller handles both.
fn downsampleFromPixels(allocator: std.mem.Allocator, pixels: *const zigimg.color.PixelStorage, w: u32, h: u32, max_dim: u32) ?ScaledImage {
    // A malformed file can decode with a zero axis, and the format probe below indexes pixel 0.
    if (w == 0 or h == 0) return null;
    const fw: f32 = @floatFromInt(w);
    const fh: f32 = @floatFromInt(h);
    const fmax: f32 = @floatFromInt(max_dim);
    const scale = @min(fmax / fw, fmax / fh);
    if (scale >= 1.0) return null;
    if (samplePixel(pixels, 0) == null) return null; // format we cannot read directly

    const dw: u32 = @max(1, @as(u32, @intFromFloat(fw * scale)));
    const dh: u32 = @max(1, @as(u32, @intFromFloat(fh * scale)));
    const dst = allocator.alloc(u8, @as(usize, dw) * @as(usize, dh) * 4) catch return null;

    const bx = fw / @as(f32, @floatFromInt(dw));
    const by = fh / @as(f32, @floatFromInt(dh));

    var dy: u32 = 0;
    while (dy < dh) : (dy += 1) {
        const sy0: u32 = @intFromFloat(@as(f32, @floatFromInt(dy)) * by);
        var sy1: u32 = @intFromFloat(@as(f32, @floatFromInt(dy + 1)) * by);
        if (sy1 <= sy0) sy1 = sy0 + 1;
        if (sy1 > h) sy1 = h;
        var dx: u32 = 0;
        while (dx < dw) : (dx += 1) {
            const sx0: u32 = @intFromFloat(@as(f32, @floatFromInt(dx)) * bx);
            var sx1: u32 = @intFromFloat(@as(f32, @floatFromInt(dx + 1)) * bx);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sx1 > w) sx1 = w;

            var r: u32 = 0;
            var g: u32 = 0;
            var b: u32 = 0;
            var a: u32 = 0;
            var n: u32 = 0;
            var yy: u32 = sy0;
            while (yy < sy1) : (yy += 1) {
                const row = @as(usize, yy) * @as(usize, w);
                var xx: u32 = sx0;
                while (xx < sx1) : (xx += 1) {
                    const px = samplePixel(pixels, row + @as(usize, xx)) orelse continue;
                    r += px[0];
                    g += px[1];
                    b += px[2];
                    a += px[3];
                    n += 1;
                }
            }
            if (n == 0) n = 1;
            const di = (@as(usize, dy) * @as(usize, dw) + @as(usize, dx)) * 4;
            dst[di] = @intCast(r / n);
            dst[di + 1] = @intCast(g / n);
            dst[di + 2] = @intCast(b / n);
            dst[di + 3] = @intCast(a / n);
        }
    }

    return ScaledImage{ .pixels = dst, .width = dw, .height = dh };
}

/// Decode an image from memory, downsampling to fit within max_dim (0 = no scaling). Bounds the CPU +
/// GPU memory of image-heavy UIs whose source images are far larger than they are displayed.
export fn zigote_load_texture_from_memory_scaled(handle: u64, data_ptr: [*c]const u8, data_len: usize, max_dim: u32, out_w: *u32, out_h: *u32) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (data_ptr == null or data_len == 0) return 0;
    const data = data_ptr[0..data_len];

    var w: u32 = 0;
    var h: u32 = 0;

    // Everything zigimg handles decodes exactly once here, and the result is either scaled straight
    // out of the decoder's buffer or moved out of it whole. WebP has its own decoder and falls
    // through to the general path below.
    if (!isWebP(data)) {
        var img = zigimg.Image.fromMemory(state.allocator, data) catch |err| {
            std.log.err("zigote: image decode failed: {}", .{err});
            return 0;
        };
        defer img.deinit(state.allocator);

        const iw: u32 = @intCast(img.width);
        const ih: u32 = @intCast(img.height);

        // Scaling reads the decoded pixels directly, so a thumbnail never costs a full-size RGBA
        // buffer. Returns null when there is nothing to scale or the format is one it cannot read,
        // and then the whole image is moved out instead.
        const result: ScaledImage = if (max_dim > 0)
            downsampleFromPixels(state.allocator, &img.pixels, iw, ih, max_dim) orelse
                ScaledImage{ .pixels = toRgbaOwned(state.allocator, &img) orelse return 0, .width = iw, .height = ih }
        else
            ScaledImage{ .pixels = toRgbaOwned(state.allocator, &img) orelse return 0, .width = iw, .height = ih };

        const decoded_handle = registerImage(state, result.width, result.height, result.pixels);
        if (decoded_handle == 0) return 0;
        out_w.* = result.width;
        out_h.* = result.height;
        return decoded_handle;
    }

    var bytes = loadTextureBytes(state.allocator, data, &w, &h) orelse return 0;

    if (max_dim > 0 and (w > max_dim or h > max_dim)) {
        if (downsampleRgba(state.allocator, bytes, w, h, max_dim)) |scaled| {
            state.allocator.free(bytes);
            bytes = scaled.pixels;
            w = scaled.width;
            h = scaled.height;
        }
    }

    const img_handle = registerImage(state, w, h, bytes);
    if (img_handle == 0) return 0;

    out_w.* = w;
    out_h.* = h;
    return img_handle;
}

// ── Text layout handle FFI ────────────────────────────────────────────────────

/// Pre-compute a text layout, cache it on the Zig side, and return an opaque handle.
/// The handle can be passed to zigote_draw_text_layout each frame to skip HarfBuzz shaping.
/// Call zigote_text_layout_release() when the layout is no longer needed.
/// Returns 0 on error.
export fn zigote_text_layout_create(
    handle: u64,
    text_ptr: [*c]const u8,
    text_len: usize,
    font_family_ptr: [*c]const u8,
    font_family_len: usize,
    font_size: f32,
    font_weight: u16,
    font_style: u8,
    line_height: f32,
    letter_spacing: f32,
    word_spacing: f32,
    max_width: f32,
) u64 {
    _ = max_width; // TODO: word-wrap support
    const state = stateFromHandle(handle) orelse return 0;
    if (text_ptr == null or text_len == 0) return 0;

    const text_slice = text_ptr[0..text_len];
    // Optional family (e.g. "code" for the monospace editor face); null falls back to the default.
    const family_slice: ?[]const u8 =
        if (font_family_ptr != null and font_family_len > 0) font_family_ptr[0..font_family_len] else null;
    const fw: text_mod.FontWeight = @enumFromInt(font_weight);
    const fs: text_mod.FontStyle = if (font_style == 1) .italic else .normal;

    const layout_h = state.gpu_ui.text.appendTextLayout(state.allocator, .{
        .baseline_x = 0,
        .baseline_y = 0,
        .text = text_slice,
        .color = .{ .r = 255, .g = 255, .b = 255, .a = 255 },
        .size = font_size,
        .line_height = line_height,
        .font_family = family_slice,
        .font_weight = fw,
        .font_style = fs,
        .letter_spacing = letter_spacing,
        .word_spacing = word_spacing,
    }) catch return 0;
    return layout_h;
}

/// Release a previously created text layout handle.
export fn zigote_text_layout_release(eng: u64, layout_handle: u64) void {
    const state = stateFromHandle(eng) orelse return;
    state.gpu_ui.text.releaseTextLayout(layout_handle);
}

/// Query the bounding box of a text layout (in logical pixels).
export fn zigote_text_layout_measure(eng: u64, layout_handle: u64, out_w: *f32, out_h: *f32) void {
    const state = stateFromHandle(eng) orelse {
        out_w.* = 0;
        out_h.* = 0;
        return;
    };
    state.gpu_ui.text.measureTextLayout(layout_handle, out_w, out_h);
}

/// Resolve the nearest visual HarfBuzz cluster boundary. Returns a UTF-8 byte offset.
export fn zigote_text_layout_hit_test(eng: u64, layout_handle: u64, x: f32, y: f32) u32 {
    const state = stateFromHandle(eng) orelse return 0;
    return state.gpu_ui.text.hitTestTextLayout(layout_handle, x, y);
}

/// Resolve a UTF-8 text offset to its engine-derived visual caret geometry.
export fn zigote_text_layout_caret_position(
    eng: u64,
    layout_handle: u64,
    text_offset: u32,
    out_x: *f32,
    out_y: *f32,
    out_h: *f32,
) bool {
    const state = stateFromHandle(eng) orelse return false;
    return state.gpu_ui.text.caretPositionTextLayout(layout_handle, text_offset, out_x, out_y, out_h);
}

/// Move one visual HarfBuzz cluster stop left (-1) or right (+1). Returns a UTF-8 byte offset.
export fn zigote_text_layout_move_caret(eng: u64, layout_handle: u64, text_offset: u32, direction: i32) u32 {
    const state = stateFromHandle(eng) orelse return text_offset;
    return state.gpu_ui.text.moveCaretTextLayout(layout_handle, text_offset, direction);
}

/// Register already-decoded RGBA8 pixels as a texture handle. For callers that produced their
/// pixels rather than loading a file — a procedural page, a video frame, an extension handing back
/// raw bytes — which otherwise have to encode to PNG purely to get past the decoder.
/// Release it with zigote_release_texture like any other handle. Returns 0 on error.
export fn zigote_load_texture_from_rgba(
    handle: u64,
    pixels_ptr: [*c]const u8,
    pixels_len: usize,
    width: u32,
    height: u32,
) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (pixels_ptr == null or width == 0 or height == 0) return 0;

    const expected_len = @as(usize, width) * @as(usize, height) * 4;
    if (pixels_len < expected_len) return 0;

    const copy = state.allocator.dupe(u8, pixels_ptr[0..expected_len]) catch return 0;
    return registerImage(state, width, height, copy);
}

/// Rewrite an existing texture's RGBA8 pixels, keeping the handle. For a source of frames — a video,
/// a camera, a procedural surface — where zigote_load_texture_from_rgba + zigote_release_texture
/// would allocate and destroy a GPU texture and its bind group sixty times a second to show the same
/// rectangle.
///
/// The dimensions must match the ones the handle was created with: this is an overwrite, not a
/// resize. Returns false for an unknown handle, a size mismatch, or a short buffer — a caller that
/// changed resolution should create a new handle.
///
/// Safe from any thread. The texel upload itself is deferred to the top of the next frame, on the
/// render thread, where writing a texture is legal (see drainImageUpdates).
export fn zigote_update_texture_rgba(
    handle: u64,
    image_handle: u64,
    pixels_ptr: [*c]const u8,
    pixels_len: usize,
    width: u32,
    height: u32,
) bool {
    const state = stateFromHandle(handle) orelse return false;
    if (image_handle == 0 or pixels_ptr == null or width == 0 or height == 0) return false;

    const expected_len = @as(usize, width) * @as(usize, height) * 4;
    if (pixels_len < expected_len) return false;

    state.image_lock.lock();
    defer state.image_lock.unlock();
    // Re-check under the lock — see registerImage.
    if (live_engine.load(.acquire) != handle) return false;

    const entry = state.image_registry.getPtr(image_handle) orelse return false;
    if (entry.width != width or entry.height != height) return false;

    // The CPU copy is dropped once an image reaches the GPU (drainImageUploads), so the steady
    // state for a video is: allocate one buffer on the first update, then memcpy into it forever.
    if (entry.pixels.len != expected_len) {
        const fresh = state.allocator.alloc(u8, expected_len) catch return false;
        if (entry.pixels.len > 0) state.allocator.free(entry.pixels);
        entry.pixels = fresh;
    }

    @memcpy(@constCast(entry.pixels[0..expected_len]), pixels_ptr[0..expected_len]);

    // One entry per handle per frame: a producer faster than the display would otherwise grow the
    // list without bound, and only the last write is visible anyway.
    for (state.pending_image_updates.items) |key| {
        if (key == image_handle) return true;
    }
    state.pending_image_updates.append(state.allocator, image_handle) catch return false;
    return true;
}

/// Push every pending zigote_update_texture_rgba into its GPU texture. Render thread, top of frame:
/// `queue.writeTexture` is ordered ahead of the submits that follow, so an update handed over during
/// the previous frame is visible in this one.
fn drainImageUpdates(state: *EngineState) void {
    state.image_lock.lock();
    const keys = state.pending_image_updates.toOwnedSlice(state.allocator) catch {
        state.image_lock.unlock();
        return; // OOM: keep the list intact, retry next frame
    };
    state.image_lock.unlock();
    defer state.allocator.free(keys);

    for (keys) |key| {
        state.image_lock.lock();
        const entry = state.image_registry.getPtr(key);
        const pixels = if (entry) |e| e.pixels else &.{};
        const width = if (entry) |e| e.width else 0;
        const height = if (entry) |e| e.height else 0;
        state.image_lock.unlock();

        if (pixels.len == 0) continue;

        // A miss means the image has never been painted, so no texture exists yet; leaving the
        // pixels attached lets the ordinary first-upload path pick them up.
        if (state.gpu_ui.updateCachedImage(state.queue, key, pixels, width, height)) {
            // Uploaded: drop the CPU copy exactly like drainImageUploads does, so a paused video
            // does not hold a second full-resolution buffer. The next update reallocates it.
            state.image_lock.lock();
            const stale = blk: {
                const e = state.image_registry.getPtr(key) orelse break :blk &.{};
                const p = e.pixels;
                e.pixels = &.{};
                break :blk p;
            };
            state.image_lock.unlock();
            if (stale.len > 0) state.allocator.free(stale);
        }
    }
}

/// Upload a custom glyph atlas (R8 grayscale) and return a texture handle.
/// The handle can be used in CMD_GLYPH_RUN commands via AddGlyphRun().
/// Returns 0 on error.
export fn zigote_upload_glyph_atlas(
    handle: u64,
    pixels_ptr: [*c]const u8,
    pixels_len: usize,
    width: u32,
    height: u32,
) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (pixels_ptr == null or pixels_len == 0 or width == 0 or height == 0) return 0;

    const expected_len = @as(usize, width) * @as(usize, height);
    if (pixels_len < expected_len) return 0;

    // Convert R8 → RGBA (all 4 channels = gray so tint multiplication works correctly)
    const rgba = state.allocator.alloc(u8, expected_len * 4) catch return 0;
    for (0..expected_len) |i| {
        const g = pixels_ptr[i];
        rgba[i * 4 + 0] = g;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = g;
        rgba[i * 4 + 3] = g;
    }

    // Assign a handle and store pixels in image_registry for lifecycle tracking
    return registerImage(state, width, height, rgba);
}

/// Load a font face from a file path and register it under `name`.
/// Both `name_ptr` and `path_ptr` are null-terminated C strings.
/// Returns .ok on success, .err on failure.
export fn zigote_load_font(handle: u64, name_ptr: [*c]const u8, path_ptr: [*c]const u8) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    if (name_ptr == null or path_ptr == null) return .err;
    const name = std.mem.span(name_ptr);

    state.gpu_ui.text.loadFontFromCPath(name, path_ptr) catch |err| {
        std.log.err("zigote_load_font failed: {}", .{err});
        return .err;
    };

    // Keep every window's face table in sync: broadcast to live secondary windows and record the
    // font so windows created later replay it (see createSecondaryWindowImpl).
    var it = state.windows.valueIterator();
    while (it.next()) |win| {
        win.*.gpu_ui.text.loadFontFromCPath(name, path_ptr) catch |err| {
            std.log.warn("zigote: window font load '{s}' failed: {}", .{ name, err });
        };
    }
    recordLoadedFont(state, name, std.mem.span(path_ptr)) catch {};
    return .ok;
}

/// Register `name` as an emoji font family for color glyph rendering.
/// The font must have been loaded first via zigote_load_font or the initial font list.
/// Returns .ok on success, .err on failure.
export fn zigote_add_emoji_font(handle: u64, name_ptr: [*c]const u8) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    if (name_ptr == null) return .err;
    const name = std.mem.span(name_ptr);
    // Refuse fonts the color path cannot draw (e.g. COLR-outline-only faces): registration
    // would capture every emoji codepoint and render nothing. The host sees .err and keeps
    // its monochrome fallback instead.
    if (!state.gpu_ui.text.emojiFamilyRendersColor(name)) return .err;
    state.gpu_ui.text.addEmojiFontFamily(name);

    var it = state.windows.valueIterator();
    while (it.next()) |win| win.*.gpu_ui.text.addEmojiFontFamily(name);
    if (state.emoji_family) |old| state.allocator.free(old);
    state.emoji_family = state.allocator.dupeZ(u8, name) catch null;
    return .ok;
}

/// Register `name` as a script-fallback family: text the requested face cannot render falls
/// through these, in registration order, before giving up and drawing .notdef. The font must have
/// been loaded first via zigote_load_font or the initial font list.
///
/// This is what lets an app whose bundled face covers only Latin still display Japanese, Korean,
/// Chinese, Arabic or Thai — the host registers whatever the platform ships.
/// Returns .ok on success, .err on failure.
export fn zigote_add_fallback_font(handle: u64, name_ptr: [*c]const u8) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    if (name_ptr == null) return .err;
    const name = std.mem.span(name_ptr);

    state.gpu_ui.text.addFallbackFontFamily(name);
    var it = state.windows.valueIterator();
    while (it.next()) |win| win.*.gpu_ui.text.addFallbackFontFamily(name);

    const owned = state.allocator.dupeZ(u8, name) catch return .ok;
    state.fallback_families.append(state.allocator, owned) catch state.allocator.free(owned);
    return .ok;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Set by `zigote_set_window_transparent` BEFORE `zigote_init`: create the main window with an
/// alpha channel the compositor composites (SDL_WINDOW_TRANSPARENT + a premultiplied wgpu
/// surface). CSD hosts use it to draw rounded window corners — the frame clears to alpha 0 and
/// the app clips its paint to a rounded rect. Ignored where the compositor/surface can't do
/// alpha (the query export below reports what actually took).
var pending_transparent_window: bool = false;

fn pickAlphaMode(caps: *const wgpu.SurfaceCapabilities, want_transparent: bool) wgpu.CompositeAlphaMode {
    if (want_transparent) {
        for (caps.alpha_modes[0..caps.alpha_mode_count]) |m|
            if (m == .premultiplied) return m;
    }
    return if (caps.alpha_mode_count > 0) caps.alpha_modes[0] else .auto;
}

fn pickSurfaceFormat(formats: []const wgpu.TextureFormat) wgpu.TextureFormat {
    for (formats) |f| {
        switch (f) {
            .bgra8_unorm_srgb, .rgba8_unorm_srgb => return f,
            else => {},
        }
    }
    return formats[0];
}

fn isWebP(data: []const u8) bool {
    if (data.len < 12) return false;
    return std.mem.eql(u8, data[0..4], "RIFF") and std.mem.eql(u8, data[8..12], "WEBP");
}

fn decodeWebP(allocator: std.mem.Allocator, data: []const u8, width: *u32, height: *u32) ?[]u8 {
    var w: c_int = 0;
    var h: c_int = 0;
    if (webp.WebPGetInfo(data.ptr, data.len, &w, &h) == 0) return null;

    width.* = @intCast(w);
    height.* = @intCast(h);

    const decoded_ptr = webp.WebPDecodeRGBA(data.ptr, data.len, &w, &h) orelse return null;
    defer webp.WebPFree(decoded_ptr);

    const pixel_count = @as(usize, @intCast(w)) * @as(usize, @intCast(h));
    const output = allocator.alloc(u8, pixel_count * 4) catch return null;
    @memcpy(output, decoded_ptr[0 .. pixel_count * 4]);
    return output;
}

// ── Physics FFI ───────────────────────────────────────────────────────────────

/// Initialize the JoltPhysics world.
/// max_bodies: maximum simultaneous bodies.
/// num_threads: worker threads (-1 = auto-detect).
/// Returns .ok on success, .err on failure.
export fn zigote_physics_init(handle: u64, max_bodies: u32, num_threads: i32) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    if (state.physics != null) return .ok; // already initialized

    state.physics = physics_ffi.init(state.allocator, max_bodies, num_threads) catch |err| {
        std.log.err("zigote_physics_init failed: {}", .{err});
        return .err;
    };
    return .ok;
}

/// Shut down the physics world and free all bodies.
export fn zigote_physics_shutdown(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (state.physics) |phys| {
        physics_ffi.deinit(phys);
        state.physics = null;
    }
}

/// Advance the simulation by delta_time seconds.
/// collision_steps: sub-steps for collision detection (1 is fine for 60 Hz).
export fn zigote_physics_step(handle: u64, delta_time: f32, collision_steps: i32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.step(phys, delta_time, collision_steps);
}

/// Set the gravity vector (default: 0, -9.81, 0).
export fn zigote_physics_set_gravity(handle: u64, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.setGravity(phys, x, y, z);
}

/// Rebuild the broad-phase acceleration structure.
/// Call once after adding all static bodies, before the first step.
export fn zigote_physics_optimize_broadphase(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.optimizeBroadPhase(phys);
}

/// Create a rigid body. Returns body_id (0xFFFFFFFF on error).
///
/// shape_type: 0=Box, 1=Sphere, 2=Capsule, 3=Cylinder
///   Box:     hx/hy/hz = half-extents
///   Sphere:  hx = radius
///   Capsule: hx = radius, hy = half-height of cylinder part
///   Cylinder:hx = radius, hy = half-height
/// px/py/pz: initial position
/// rx/ry/rz: initial rotation as Euler angles (radians)
/// motion_type: 0=Static, 1=Kinematic, 2=Dynamic
/// friction: surface friction coefficient (0–1, default 0.2)
/// restitution: bounce coefficient (0–1, default 0.0)
export fn zigote_physics_create_body(
    handle: u64,
    shape_type: u8,
    hx: f32,
    hy: f32,
    hz: f32,
    px: f32,
    py: f32,
    pz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    motion_type: u8,
    friction: f32,
    restitution: f32,
    gravity_factor: f32,
    mass: f32,
) u32 {
    const state = stateFromHandle(handle) orelse return physics_ffi.INVALID_BODY_ID;
    const phys = state.physics orelse return physics_ffi.INVALID_BODY_ID;
    return physics_ffi.createBody(phys, shape_type, hx, hy, hz, px, py, pz, rx, ry, rz, motion_type, friction, restitution, gravity_factor, mass);
}

/// Destroy a body (removes from simulation and frees it).
export fn zigote_physics_destroy_body(handle: u64, body_id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.destroyBody(phys, body_id);
}

/// Add a body to the simulation (activates it).
export fn zigote_physics_add_body(handle: u64, body_id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.addBody(phys, body_id);
}

/// Remove a body from the simulation without destroying it.
export fn zigote_physics_remove_body(handle: u64, body_id: u32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.removeBody(phys, body_id);
}

/// Read the current world-space position of a body.
export fn zigote_physics_get_body_position(handle: u64, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.getBodyPosition(phys, body_id, out_x, out_y, out_z);
}

/// Read the current world-space rotation as Euler angles (radians).
export fn zigote_physics_get_body_rotation(handle: u64, body_id: u32, out_rx: *f32, out_ry: *f32, out_rz: *f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.getBodyRotation(phys, body_id, out_rx, out_ry, out_rz);
}

/// Teleport a body to a new position (activates it).
export fn zigote_physics_set_body_position(handle: u64, body_id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.setBodyPosition(phys, body_id, x, y, z);
}

/// Set the linear velocity of a body.
export fn zigote_physics_set_linear_velocity(handle: u64, body_id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.setLinearVelocity(phys, body_id, x, y, z);
}

/// Set the angular velocity of a body (radians/s).
export fn zigote_physics_set_angular_velocity(handle: u64, body_id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.setAngularVelocity(phys, body_id, x, y, z);
}

/// Apply a continuous force to a body (N). Resets each step.
export fn zigote_physics_add_force(handle: u64, body_id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.addForce(phys, body_id, x, y, z);
}

/// Apply an instantaneous impulse to a body (kg·m/s).
export fn zigote_physics_add_impulse(handle: u64, body_id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.addImpulse(phys, body_id, x, y, z);
}

/// Apply a continuous torque to a body (N·m). Resets each step.
export fn zigote_physics_add_torque(handle: u64, body_id: u32, x: f32, y: f32, z: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.addTorque(phys, body_id, x, y, z);
}

/// Apply a continuous force (N) at a world-space point — produces both linear force and torque.
export fn zigote_physics_add_force_at_point(handle: u64, body_id: u32, fx: f32, fy: f32, fz: f32, px: f32, py: f32, pz: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.addForceAtPoint(phys, body_id, fx, fy, fz, px, py, pz);
}

/// Read the linear velocity of a body (m/s).
export fn zigote_physics_get_linear_velocity(handle: u64, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.getLinearVelocity(phys, body_id, out_x, out_y, out_z);
}

/// Read the angular velocity of a body (rad/s).
export fn zigote_physics_get_angular_velocity(handle: u64, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.getAngularVelocity(phys, body_id, out_x, out_y, out_z);
}

/// Read the rotation of a body as a quaternion (x, y, z, w).
export fn zigote_physics_get_body_rotation_quat(handle: u64, body_id: u32, out_x: *f32, out_y: *f32, out_z: *f32, out_w: *f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.getBodyRotationQuat(phys, body_id, out_x, out_y, out_z, out_w);
}

/// Set the rotation of a body from a quaternion (x, y, z, w); activates it.
export fn zigote_physics_set_body_rotation_quat(handle: u64, body_id: u32, qx: f32, qy: f32, qz: f32, qw: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.setBodyRotationQuat(phys, body_id, qx, qy, qz, qw);
}

/// Batched transform read: for each of the `count` body ids, writes 7 f32 (pos.xyz + quat.xyzw)
/// into `out_xforms` (must hold `count * 7` floats). One call replaces a position + rotation
/// FFI pair per body on the per-tick sync path.
export fn zigote_physics_get_body_transforms(handle: u64, ids: [*c]const u32, count: u32, out_xforms: [*]f32) void {
    const state = stateFromHandle(handle) orelse return;
    const phys = state.physics orelse return;
    physics_ffi.getBodyTransforms(phys, ids, count, out_xforms);
}

/// Closest-hit world ray cast, skipping `ignore_body` (0xFFFFFFFF = none). Returns 1 on hit, 0 on miss.
export fn zigote_physics_raycast_closest(
    handle: u64,
    ox: f32,
    oy: f32,
    oz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    max_dist: f32,
    ignore_body: u32,
    out_body: *u32,
    out_fraction: *f32,
    out_px: *f32,
    out_py: *f32,
    out_pz: *f32,
    out_nx: *f32,
    out_ny: *f32,
    out_nz: *f32,
) u32 {
    const state = stateFromHandle(handle) orelse return 0;
    const phys = state.physics orelse return 0;
    const hit = physics_ffi.raycastClosest(phys, ox, oy, oz, dx, dy, dz, max_dist, ignore_body, out_body, out_fraction, out_px, out_py, out_pz, out_nx, out_ny, out_nz);
    return if (hit) 1 else 0;
}

// ── Render graph / pass model FFI ────────────────────────────────────────────

/// Return ABI compatibility information.
/// C# must call this after zigote_init() and verify that all size fields match
/// its compile-time sizeof() values before rendering any frames.
export fn zigote_get_renderer_abi_info(out_info: *ZgAbiInfo) void {
    out_info.* = .{
        .abi_version = 9,
        .paint_command_size = @sizeOf(ZgPaintCommand),
        .event_size = @sizeOf(ZgEvent),
        .handle_size = @sizeOf(usize),
        .render_settings_3d_size = @sizeOf(ZgRenderSettings3D),
    };
}

/// Report the active backend's runtime capabilities (vendor upscalers / hardware ray tracing).
/// Call AFTER zigote_init(): the GPU device exists by then, so capabilities reflect the real
/// hardware/backend. The wgpu backend reports none today (no host-side RT or vendor upscaler).
/// Lets the host gray-out unavailable features. Distinct from the fixed-size ZgAbiInfo guard.
export fn zigote_get_renderer_caps(handle: u64, out_caps: *ZgRendererCaps) void {
    const state = stateFromHandle(handle) orelse {
        out_caps.* = .{ .active_backend = 0, .upscalers = 0, .raytracing = 0, .raytracing_from_render = 0 };
        return;
    };

    const caps = state.gpu_backend.caps();
    out_caps.* = .{
        .active_backend = @intFromEnum(caps.active_backend),
        .upscalers = caps.upscalers,
        .raytracing = @intFromBool(caps.raytracing),
        .raytracing_from_render = @intFromBool(caps.raytracing_from_render),
    };
}

/// Begin a new logical frame.  Stores per-frame parameters used by
/// zigote_render_frame_v2().  The caller must also supply commands via
/// zigote_submit_paint_commands() before calling render.
///
/// scene_w / scene_h — pixel dimensions of the 3D viewport (0 = no 3D this frame).
/// scale             — display scale factor (e.g. 2.0 on Retina).
/// delta_time        — seconds elapsed since the previous frame.
export fn zigote_begin_frame(handle: u64, scene_w: u32, scene_h: u32, scale: f32, delta_time: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.pending_scene_w = scene_w;
    state.pending_scene_h = scene_h;
    state.pending_scale = if (scale > 0) scale else 1.0;
    state.pending_dt = delta_time;
    // Default to a full-frame repaint; C# opts into partial repaint by calling
    // zigote_submit_frame_damage after submitting paint commands this frame.
    state.pending_damage_count = 0;
    state.transient_pool.resetFrame();
}

/// Declare the damaged sub-rectangles for the next zigote_render_frame_v2 (absolute logical px, 4 floats
/// per rect: x, y, width, height). The pure-UI render path repaints only these regions into the
/// persistent scene texture (loadOp = load + scissor) instead of a full clear. count 0 (or no call) =
/// repaint the whole frame. C# keeps the regions pairwise non-overlapping; excess collapses to full.
export fn zigote_submit_frame_damage(handle: u64, rects: [*c]const f32, count: u32) void {
    const state = stateFromHandle(handle) orelse return;

    // More regions than we can hold means C# already gave up on tracking them individually — fall back
    // to a full-frame repaint rather than dropping regions and leaving stale pixels.
    if (count == 0 or count > MAX_UI_DAMAGE_RECTS or rects == null) {
        state.pending_damage_count = 0;
        return;
    }

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        state.pending_damage[i] = .{
            .x = rects[i * 4 + 0],
            .y = rects[i * 4 + 1],
            .width = rects[i * 4 + 2],
            .height = rects[i * 4 + 3],
        };
    }
    state.pending_damage_count = count;
}

/// Store paint commands for the next zigote_render_frame_v2() call.
export fn zigote_submit_paint_commands(handle: u64, commands: [*]const ZgPaintCommand, count: u32) void {
    const state = stateFromHandle(handle) orelse return;
    fillPaintList(state, &state.paint_list, commands, count) catch |err| {
        std.log.err("zigote_submit_paint_commands failed: {}", .{err});
    };
}

/// Like zigote_submit_paint_commands but for the overlay pass.
export fn zigote_submit_overlay_commands(handle: u64, commands: [*]const ZgPaintCommand, count: u32) void {
    const state = stateFromHandle(handle) orelse return;
    fillPaintList(state, &state.overlay_paint_list, commands, count) catch |err| {
        std.log.err("zigote_submit_overlay_commands failed: {}", .{err});
    };
}

/// Execute the render graph and present the frame.
///
/// Flow:
///   Scene3DPass       — renders 3D scene into offscreen texture (if scene dims set)
///   BackdropCapture   — copies 3D output into scene_texture so glass can sample it
///   UIPass            — renders stored paint commands on top
///   PresentPass       — blits to swapchain
///
/// Returns .ok on success, .err on error.
export fn zigote_render_frame_v2(handle: u64) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;

    renderFrameV2Impl(state) catch |err| {
        std.log.err("zigote_render_frame_v2 failed: {}", .{err});
        return .err;
    };
    return .ok;
}

const PassContext = render_mod.graph.PassContext;

inline fn rgState(ctx: *PassContext) *EngineState {
    return ctx.userAs(EngineState);
}

/// RTPrePass — render each render-texture's pending paint list into its own texture and
/// (re)register the result in the image cache so CMD_IMAGE can sample it. Self-submitting.
fn passRtPrepass(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    const scale = state.pending_scale;
    var rt_it = state.render_textures.valueIterator();
    while (rt_it.next()) |rt| {
        if (rt.pending_paint.commands.items.len == 0) continue;
        wgpu_renderer.wgpu.renderToTexture(
            state.device,
            state.queue,
            &state.gpu_ui,
            rt.pending_paint,
            rt.view,
            rt.width,
            rt.height,
            scale,
        ) catch |err| {
            std.log.err("zigote: RT render failed: {}", .{err});
            continue;
        };
        // Register / update the RT texture in image_cache so CMD_IMAGE finds it
        const bgl = state.gpu_ui.text.bindGroupLayout();
        if (state.gpu_ui.image_cache.fetchRemove(rt.cache_key)) |old| old.value.bind_group.release();
        if (state.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("zigote rt image bg"),
            .layout = bgl,
            .entries = &.{
                .{ .binding = 0, .texture_view = rt.view },
                .{ .binding = 1, .sampler = state.gpu_ui.text.getSampler() },
            },
            .entry_count = 2,
        })) |bg| {
            state.gpu_ui.image_cache.put(rt.cache_key, .{
                .texture = rt.texture,
                .texture_view = rt.view,
                .bind_group = bg,
                // The RenderTextureEntry owns both; the cache only points at them.
                .pinned = true,
                .borrowed = true,
            }) catch bg.release();
        }
    }
}

/// BlurPass — gaussian-blur compute over render-textures with pending blur requests.
/// Lazy-inits the compute pipelines and uses a transient intermediate. Self-submitting.
fn passBlur(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    if (state.blur_requests.items.len == 0) return;
    // Lazy-init compute pipelines on first use
    if (state.gaussian_blur == null) {
        state.gaussian_blur = wgpu_blur.GaussianBlur.init(state.device, state.queue) catch |err| blk: {
            std.log.err("zigote: GaussianBlur init failed: {}", .{err});
            break :blk null;
        };
    }

    if (state.gaussian_blur) |*gb| {
        for (state.blur_requests.items) |blur| {
            const rt = state.render_textures.getPtr(blur.src_handle) orelse continue;

            // Ensure blur_texture exists (rgba8unorm, storage_binding)
            if (rt.blur_texture == null) {
                rt.blur_texture = state.device.createTexture(&.{
                    .label = wgpu.StringView.fromSlice("zigote rt blur dst"),
                    .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.storage_binding,
                    .dimension = .@"2d",
                    .size = .{ .width = rt.width, .height = rt.height, .depth_or_array_layers = 1 },
                    .format = .rgba8_unorm,
                    .mip_level_count = 1,
                    .sample_count = 1,
                });
                if (rt.blur_texture) |t|
                    rt.blur_view = t.createView(null);
            }

            const blur_tex = rt.blur_texture orelse continue;
            const blur_view = rt.blur_view orelse continue;

            // Temp intermediate (rgba8unorm, texture_binding | storage_binding)
            const temp = state.transient_pool.acquire(.{
                .width = rt.width,
                .height = rt.height,
                .format = .rgba8_unorm,
                .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.storage_binding,
                .debug_name = "zigote blur temp",
            }) catch continue;
            defer state.transient_pool.releaseTexture(temp.texture);

            const blur_enc = state.device.createCommandEncoder(&.{
                .label = wgpu.StringView.fromSlice("zigote blur encoder"),
            }) orelse continue;
            defer blur_enc.release();

            gb.dispatch(blur_enc, rt.texture, temp.texture, blur_tex, blur.sigma) catch |err| {
                std.log.err("zigote: blur dispatch failed: {}", .{err});
                continue;
            };

            var blur_cmd = blur_enc.finish(&.{}) orelse continue;
            defer blur_cmd.release();
            state.queue.submit(&.{blur_cmd});

            // Register blurred texture in image_cache so CMD_IMAGE shows blurred result
            const bgl = state.gpu_ui.text.bindGroupLayout();
            if (state.gpu_ui.image_cache.fetchRemove(rt.cache_key)) |old| old.value.bind_group.release();
            if (state.device.createBindGroup(&.{
                .label = wgpu.StringView.fromSlice("zigote rt blur bg"),
                .layout = bgl,
                .entries = &.{
                    .{ .binding = 0, .texture_view = blur_view },
                    .{ .binding = 1, .sampler = state.gpu_ui.text.getSampler() },
                },
                .entry_count = 2,
            })) |bg| {
                state.gpu_ui.image_cache.put(rt.cache_key, .{
                    .texture = blur_tex,
                    .texture_view = blur_view,
                    .bind_group = bg,
                    // Same owner as the unblurred entry above: rt.blur_texture / rt.blur_view.
                    .pinned = true,
                    .borrowed = true,
                }) catch bg.release();
            }
        }
    }
    state.blur_requests.clearRetainingCapacity();
}

/// Scene3DBegin — (re)creates the offscreen scene_color target on resize, opens the shared
/// scene command encoder, and runs the scene prepare step (lights/env/depth). Sets
/// `scene_rendered` so the dependent scene passes know whether to run this frame.
fn passScene3dBegin(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    state.scene_rendered = false;
    if (state.pending_scene_w == 0 or state.pending_scene_h == 0) return;
    // First 3D frame creates the renderer. A latched init failure returns plainly (not an error) so
    // the render graph doesn't log a failed pass every frame — scene_rendered stays false and every
    // dependent scene pass skips.
    const g3d = ensure3d(state) orelse return;
    const sw = state.pending_scene_w;
    const sh = state.pending_scene_h;
    const internal_w: u32 = sw;
    const internal_h: u32 = sh;

    // Resize offscreen texture if needed
    var need_recreate = false;
    if (state.offscreen_3d_texture) |tex| {
        if (tex.getWidth() != sw or tex.getHeight() != sh) need_recreate = true;
    } else {
        need_recreate = true;
    }

    if (need_recreate) {
        if (state.offscreen_3d_view) |v| v.release();
        if (state.offscreen_3d_texture) |t| t.release();
        state.offscreen_3d_texture = state.device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("scene_color"),
            // copy_dst: receives the resolved TAA frame (taa_output → here). copy_src kept for the
            // UI backdrop capture. render_attachment for the non-TAA tonemap path drawing here.
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.copy_src | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = sw, .height = sh, .depth_or_array_layers = 1 },
            .format = state.wgpu_config.format,
            .mip_level_count = 1,
            .sample_count = 1,
        });
        if (state.offscreen_3d_texture == null) return error.SceneColorCreateFailed;
        state.offscreen_3d_view = state.offscreen_3d_texture.?.createView(null);
        if (state.offscreen_3d_view == null) return error.SceneColorViewCreateFailed;

        // Update gpu_ui image cache with new bind group (BackdropCapture)
        const magic_key: u64 = 0x3D3D3D3D3D3D3D3D;
        if (state.gpu_ui.image_cache.fetchRemove(magic_key)) |kv| {
            kv.value.bind_group.release();
        }
        if (state.device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("scene_color_bg"),
            .layout = state.gpu_ui.text.bindGroupLayout(),
            .entries = &.{
                .{ .binding = 0, .texture_view = state.offscreen_3d_view.? },
                .{ .binding = 1, .sampler = state.gpu_ui.text.getSampler() },
            },
            .entry_count = 2,
        })) |bg| {
            state.gpu_ui.image_cache.put(magic_key, .{
                .texture = state.offscreen_3d_texture.?,
                .texture_view = state.offscreen_3d_view.?,
                .bind_group = bg,
                .pinned = true,
                .borrowed = true,
            }) catch bg.release();
        }
    }

    const scene_enc = state.device.createCommandEncoder(&.{}) orelse return error.EncoderFailed;
    errdefer scene_enc.release();
    state.world.updateTransforms();
    // Render-graph path renders to an offscreen destination that can receive the resolved TAA
    // copy, so the resolve (and thus jitter) may run this frame.
    g3d.taa_path_supported = true;
    try g3d.beginScene(&state.world, state.queue, scene_enc, internal_w, internal_h, state.selected_node_ptr);
    state.scene_enc = scene_enc;
    state.scene_rendered = true;
}

/// Scene3DShadow — directional shadow-map depth pass. Writes `shadow_depth`.
fn passScene3dShadow(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    if (!state.scene_rendered) return;
    // gpu_3d is non-null whenever scene_rendered: Scene3DBegin ensured it (same in Sky/Geometry/Post).
    try state.gpu_3d.?.renderShadowPass(&state.world, state.queue, state.scene_enc.?);
}

/// Scene3DSky — full-screen sky into the (MSAA-resolved) HDR scene buffer. Writes `scene_color`.
fn passScene3dSky(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    if (!state.scene_rendered) return;
    try state.gpu_3d.?.renderSkyPass(state.scene_enc.?, state.world.active_camera != null);
}

/// Scene3DGeometry — opaque + transparent meshes (and optional 2D layer). Reads
/// `shadow_depth`, writes `scene_color` (HDR) + `scene_depth`. Reports drawn objects.
fn passScene3dGeometry(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    if (!state.scene_rendered) return;
    const g3d = state.gpu_3d.?;
    const drawn = try g3d.renderSceneGeometry(
        &state.world,
        state.queue,
        state.scene_enc.?,
        g3d.depth_width,
        g3d.depth_height,
    );
    ctx.graph.frame_scene_objects += drawn;
}

/// Scene3DPost — bloom + ACES tonemap of the HDR scene buffer into the LDR offscreen
/// texture the UI composite samples. Reads `scene_color` (HDR), writes `scene_color` (the
/// LDR offscreen). Records into the shared scene encoder.
fn passScene3dPost(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    if (!state.scene_rendered) return;
    state.gpu_3d.?.renderPostProcess(state.queue, state.scene_enc.?, state.offscreen_3d_view.?, state.offscreen_3d_texture);
}

/// Scene3DSubmit — finish and submit the shared scene encoder opened in Scene3DBegin.
fn passScene3dSubmit(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    if (!state.scene_rendered) return;
    const enc = state.scene_enc orelse return;
    state.scene_enc = null;
    defer enc.release();
    var scene_cmd = enc.finish(&.{}) orelse return error.EncoderFinishFailed;
    defer scene_cmd.release();
    state.queue.submit(&.{scene_cmd});
}

/// CompositeAndPresent — UIPass + BackdropCapture over the swapchain, then present.
/// When the 3D scene ran, the scene color seeds the backdrop so glass UI can sample it;
/// otherwise it falls back to the plain UI render path. Reads `scene_color`, writes the
/// `swapchain_color`. On an unavailable swapchain it flags the frame as dropped so the
/// frame tail does not advance state.
fn passCompositePresent(ctx: *PassContext) anyerror!void {
    const state = rgState(ctx);
    const scale = state.pending_scale;
    const w = state.wgpu_config.width;
    const h = state.wgpu_config.height;

    if (state.scene_rendered) {
        const base_tex = state.offscreen_3d_texture orelse return error.SceneColorMissing;
        var surface_texture = wgpu.SurfaceTexture{
            .next_in_chain = null,
            .texture = null,
            .status = .@"error",
        };
        state.surface.getCurrentTexture(&surface_texture);
        switch (surface_texture.status) {
            .success_optimal, .success_suboptimal => {},
            .timeout, .outdated, .lost, .occluded => {
                state.frame_dropped = true;
                return;
            },
            .@"error" => return error.WgpuSurfaceTextureUnavailable,
            else => return error.WgpuSurfaceTextureUnavailable,
        }
        defer if (surface_texture.texture) |t| t.release();

        const color_view = surface_texture.texture.?.createView(null) orelse
            return error.WgpuTextureViewUnavailable;
        defer color_view.release();

        var encoder = state.device.createCommandEncoder(&.{}) orelse return error.EncoderFailed;
        defer encoder.release();

        try wgpu_renderer.wgpu.renderFrameOverlayWithBaseTexture(
            encoder,
            base_tex,
            color_view,
            state.device,
            state.queue,
            &state.gpu_ui,
            state.paint_list,
            w,
            h,
            scale,
            state.frame_index,
        );

        if (state.overlay_paint_list.commands.items.len > 0) {
            try wgpu_renderer.wgpu.renderFrameOverlay(
                encoder,
                color_view,
                state.device,
                state.queue,
                &state.gpu_ui,
                state.overlay_paint_list,
                w,
                h,
                scale,
                state.frame_index,
            );
        }

        var final_cmd = encoder.finish(&.{}) orelse return error.EncoderFinishFailed;
        defer final_cmd.release();
        state.queue.submit(&.{final_cmd});
        if (state.surface.present() != .success) return error.WgpuPresentFailed;
    } else {
        // No 3D scene — fall back to standard UI render
        const opt_overlay: ?zg.PaintList = if (state.overlay_paint_list.commands.items.len > 0)
            state.overlay_paint_list
        else
            null;
        try wgpu_renderer.wgpu.renderFrame(
            state.surface,
            state.device,
            state.queue,
            &state.gpu_ui,
            state.paint_list,
            opt_overlay,
            w,
            h,
            scale,
            state.frame_index,
            state.pending_damage[0..state.pending_damage_count],
            true, // main window: keeps a persistent scene texture for partial repaint + glass backdrop
        );
    }
}

/// Register the per-frame pipeline as render-graph passes, executed in this order:
///   RTPrePass → BlurPass → Scene3D{Begin,Shadow,Sky,Geometry,Submit} → CompositeAndPresent
/// Resource reads/writes are declared for lifetime tracking, validation and introspection;
/// passes own their command encoders, preserving the existing multi-submit frame structure.
fn buildRenderGraph(state: *EngineState) !void {
    const g = &state.render_graph;
    try g.addPass(.{ .name = "RTPrePass", .pass_type = .resource_upload, .execute = passRtPrepass });
    try g.addPass(.{ .name = "BlurPass", .pass_type = .resource_upload, .writes = &.{.blur_temp_a}, .execute = passBlur });
    try g.addPass(.{ .name = "Scene3DBegin", .pass_type = .scene_3d, .writes = &.{.scene_color}, .execute = passScene3dBegin });
    try g.addPass(.{ .name = "Scene3DShadow", .pass_type = .scene_3d, .writes = &.{.shadow_depth}, .execute = passScene3dShadow });
    try g.addPass(.{ .name = "Scene3DSky", .pass_type = .scene_3d, .writes = &.{.scene_color}, .execute = passScene3dSky });
    try g.addPass(.{ .name = "Scene3DGeometry", .pass_type = .scene_3d, .reads = &.{.shadow_depth}, .writes = &.{ .scene_color, .scene_depth }, .execute = passScene3dGeometry });
    try g.addPass(.{ .name = "Scene3DPost", .pass_type = .scene_3d, .reads = &.{.scene_color}, .writes = &.{.scene_color}, .execute = passScene3dPost });
    try g.addPass(.{ .name = "Scene3DSubmit", .pass_type = .scene_3d, .reads = &.{.scene_color}, .execute = passScene3dSubmit });
    try g.addPass(.{ .name = "CompositeAndPresent", .pass_type = .ui, .reads = &.{.scene_color}, .writes = &.{.swapchain_color}, .execute = passCompositePresent });
    _ = g.validate();
}

/// Execute the render graph for one frame, then advance per-frame state.
fn renderFrameV2Impl(state: *EngineState) !void {
    state.frame_dropped = false;
    state.scene_rendered = false;

    // Before any batch is built: a frame handed over by zigote_update_texture_rgba lands in its
    // texture here, so this frame draws it rather than the one before it.
    drainImageUpdates(state);

    var frame = render_mod.FrameContext{
        .frame_index = state.frame_index,
        .surface_width = state.wgpu_config.width,
        .surface_height = state.wgpu_config.height,
        .dpi_scale = state.pending_scale,
        .delta_time = state.pending_dt,
        .scene_viewport_w = @floatFromInt(state.pending_scene_w),
        .scene_viewport_h = @floatFromInt(state.pending_scene_h),
        .transient_pool = &state.transient_pool,
    };
    state.render_graph.frame_paint_commands = @intCast(state.paint_list.commands.items.len);

    var ctx = PassContext{
        .frame = &frame,
        .graph = &state.render_graph,
        .user = state,
    };
    state.render_graph.execute(&ctx);

    // On a dropped frame (swapchain unavailable) keep the overlay and frame index so the
    // next frame retries cleanly — matches the pre-graph early-return behaviour.
    if (!state.frame_dropped) {
        state.frame_index +%= 1;
        state.overlay_paint_list.clearRetainingCapacity(state.allocator);
        // Dev tooling: optional one-shot framebuffer dump (ZIGOTE_SHOT env var).
        tryAutoCapture(state);
        // Dev tooling: optional periodic GPU-memory log (ZIGOTE_GPU_MEM env var).
        logGpuMem(state);
    }
}

/// End the frame — clear transient per-frame data.
export fn zigote_end_frame(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    // Image lifecycle, both halves, at the one point in the frame where the command encoder is
    // closed and submitted: drop CPU copies of everything now on the GPU, then free what the app
    // released during the frame.
    drainImageUploads(state);
    drainImageReleases(state);
    state.pending_scene_w = 0;
    state.pending_scene_h = 0;
    state.paint_list.clearRetainingCapacity(state.allocator);
    state.overlay_paint_list.clearRetainingCapacity(state.allocator);
    // Clear per-frame RT paint lists (RT textures themselves persist)
    var rt_it = state.render_textures.valueIterator();
    while (rt_it.next()) |rt| {
        rt.pending_paint.clearRetainingCapacity(state.allocator);
    }
    state.blur_requests.clearRetainingCapacity();
}

// ── Render texture API ────────────────────────────────────────────────────────

/// Create a render texture. Returns 0 on failure.
export fn zigote_render_texture_create(handle: u64, width: u32, height: u32) u64 {
    const state = stateFromHandle(handle) orelse return 0;
    if (width == 0 or height == 0) return 0;

    const tex = state.device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote render texture"),
        .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_src,
        .dimension = .@"2d",
        .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
        .format = state.wgpu_config.format,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return 0;

    const view = tex.createView(null) orelse {
        tex.release();
        return 0;
    };

    // From the shared counter: the handle doubles as this texture's key in the image cache, which
    // decoded images key too. See nextGpuHandle.
    const rt_handle = nextGpuHandle(state);

    state.render_textures.put(rt_handle, .{
        .width = width,
        .height = height,
        .texture = tex,
        .view = view,
        .cache_key = rt_handle,
    }) catch {
        view.release();
        tex.release();
        return 0;
    };
    return rt_handle;
}

/// Destroy a render texture created via zigote_render_texture_create.
export fn zigote_render_texture_destroy(handle: u64, rt_handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    if (state.render_textures.fetchRemove(rt_handle)) |entry| {
        if (state.gpu_ui.image_cache.fetchRemove(rt_handle)) |img| img.value.bind_group.release();
        var rt = entry.value;
        rt.deinit(state.allocator);
    }
}

/// Returns the image cache key for a render texture. Pass to CMD_IMAGE / AddImage as cacheKey.
export fn zigote_render_texture_cache_key(handle: u64, rt_handle: u64) u64 {
    _ = handle;
    return rt_handle; // cache_key == rt_handle by design
}

/// Read a render texture back as tightly-packed RGBA8, top-down, sRGB-encoded bytes — the closing
/// end of a GPU image pipeline: paint sources and shader passes into the RT, render the frame, then
/// pull the processed pixels out for encoding or export. Synchronous (blocks on the GPU copy), so it
/// is a capture-time call, not a per-frame one. `out` must hold width*height*4 bytes.
export fn zigote_render_texture_read_rgba(
    handle: u64,
    rt_handle: u64,
    out_ptr: [*c]u8,
    out_len: usize,
) bool {
    const state = stateFromHandle(handle) orelse return false;
    const rt = state.render_textures.getPtr(rt_handle) orelse return false;
    if (out_ptr == null) return false;

    const w = rt.width;
    const h = rt.height;
    const needed = @as(usize, w) * @as(usize, h) * 4;
    if (out_len < needed) return false;

    const fmt = state.wgpu_config.format;
    const is_bgra = (fmt == .bgra8_unorm or fmt == .bgra8_unorm_srgb);

    // Same readback shape as captureTextureBmp: 256-aligned row stride, mapAsync + poll.
    const unpadded: u32 = w * 4;
    const padded: u32 = (unpadded + 255) & ~@as(u32, 255);
    const buf_size: u64 = @as(u64, padded) * @as(u64, h);

    const rb = state.device.createBuffer(&.{
        .label = wgpu.StringView.fromSlice("rt readback"),
        .usage = wgpu.BufferUsages.map_read | wgpu.BufferUsages.copy_dst,
        .size = buf_size,
    }) orelse return false;
    defer rb.release();

    const encoder = state.device.createCommandEncoder(&.{}) orelse return false;
    encoder.copyTextureToBuffer(
        &.{ .texture = rt.texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &.{ .buffer = rb, .layout = .{ .offset = 0, .bytes_per_row = padded, .rows_per_image = h } },
        &.{ .width = w, .height = h, .depth_or_array_layers = 1 },
    );
    const cmd = encoder.finish(&.{}) orelse {
        encoder.release();
        return false;
    };
    encoder.release();
    state.queue.submit(&.{cmd});
    cmd.release();

    const MapCtx = struct { done: bool = false, ok: bool = false };
    var mc = MapCtx{};
    const cb = struct {
        fn f(status: wgpu.MapAsyncStatus, _: wgpu.StringView, ud1: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
            const c: *MapCtx = @ptrCast(@alignCast(ud1.?));
            c.done = true;
            c.ok = (status == .success);
        }
    }.f;
    _ = rb.mapAsync(wgpu.MapModes.read, 0, @intCast(buf_size), .{ .callback = cb, .userdata1 = &mc });
    var guard: u32 = 0;
    while (!mc.done and guard < 500000) : (guard += 1) {
        _ = state.device.poll(true, null);
    }
    if (!mc.ok) return false;
    const mapped = rb.getConstMappedRange(0, @intCast(buf_size)) orelse return false;
    defer rb.unmap();
    const src: [*]const u8 = @ptrCast(mapped);

    var y: u32 = 0;
    while (y < h) : (y += 1) {
        const src_row = src + @as(usize, y) * padded;
        const dst_row = out_ptr + @as(usize, y) * unpadded;
        if (is_bgra) {
            var x: u32 = 0;
            while (x < w) : (x += 1) {
                const s = src_row + @as(usize, x) * 4;
                const d = dst_row + @as(usize, x) * 4;
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        } else {
            @memcpy(dst_row[0..unpadded], src_row[0..unpadded]);
        }
    }
    return true;
}

/// Convenience: begin + render + end in one call for simple use-cases.
/// Equivalent to zigote_begin_frame + zigote_render_frame_v2 + zigote_end_frame.
export fn zigote_frame_begin(handle: u64, scene_w: u32, scene_h: u32, scale: f32, delta_time: f32) void {
    const state = stateFromHandle(handle) orelse return;
    state.pending_scene_w = scene_w;
    state.pending_scene_h = scene_h;
    state.pending_scale = if (scale > 0) scale else 1.0;
    state.pending_dt = delta_time;
    state.transient_pool.resetFrame();
}

/// Execute the render graph and present; then clear per-frame state.
/// Replaces the three-call sequence zigote_render_frame_v2 + zigote_end_frame.
export fn zigote_frame_end(handle: u64) ZgResult {
    const state = stateFromHandle(handle) orelse return .err;
    renderFrameV2Impl(state) catch |err| {
        std.log.err("zigote_frame_end render failed: {}", .{err});
        return .err;
    };
    state.pending_scene_w = 0;
    state.pending_scene_h = 0;
    state.paint_list.clearRetainingCapacity(state.allocator);
    state.overlay_paint_list.clearRetainingCapacity(state.allocator);
    var rt_it = state.render_textures.valueIterator();
    while (rt_it.next()) |rt| {
        rt.pending_paint.clearRetainingCapacity(state.allocator);
    }
    state.blur_requests.clearRetainingCapacity();
    return .ok;
}

/// Update render settings.
/// enable_glass: 1 = glass effects active, 0 = disabled.
/// enable_debug: 1 = debug overlays active, 0 = disabled.
export fn zigote_set_render_settings(handle: u64, enable_glass: u8, enable_debug: u8) void {
    const state = stateFromHandle(handle) orelse return;
    state.render_settings.enable_glass_effects = enable_glass != 0;
    state.render_settings.enable_debug_overlays = enable_debug != 0;
}

/// Flat C-ABI mirror of wgpu_3d.Settings3D. Layout must match ZgRenderSettings3D in C#.
/// All fields f32 (70 floats); colours are linear rgb, sun angles in degrees.
pub const ZgRenderSettings3D = extern struct {
    ambient_intensity: f32,
    sky_horizon_r: f32,
    sky_horizon_g: f32,
    sky_horizon_b: f32,
    sky_zenith_r: f32,
    sky_zenith_g: f32,
    sky_zenith_b: f32,
    sky_ground_r: f32,
    sky_ground_g: f32,
    sky_ground_b: f32,
    env_avg_r: f32,
    env_avg_g: f32,
    env_avg_b: f32,
    sun_azimuth_deg: f32,
    sun_elevation_deg: f32,
    sun_intensity: f32,
    overhead: f32,
    horizon_glow: f32,
    sun_sharpness: f32,
    exposure: f32,
    contrast: f32,
    saturation: f32,
    shadow_strength: f32,
    shadow_bias: f32,
    shadow_softness: f32,
    clearcoat: f32,
    bloom_threshold: f32,
    bloom_knee: f32,
    bloom_intensity: f32,
    ssao_radius: f32,
    ssao_bias: f32,
    ssao_strength: f32,
    ssao_power: f32,
    ssr_intensity: f32,
    ssr_max_distance: f32,
    ssr_thickness: f32,
    ssr_steps: f32,
    taa_enabled: f32,
    taa_feedback: f32,
    diagnostic_mode: f32,
    debug_view: f32,
    // Depth of field: enable + fallback focus distance + aperture + max blur radius.
    dof_enabled: f32,
    dof_focus_distance: f32,
    dof_f_stop: f32,
    dof_max_coc: f32,
    // Wireframe render debug mode (0/1): draw all geometry as flat line edges.
    wireframe: f32,
    // Atmospheric fog: density (0 = off), colour rgb, height base, height falloff, sun in-scatter, anisotropy g.
    fog_density: f32,
    fog_color_r: f32,
    fog_color_g: f32,
    fog_color_b: f32,
    fog_height: f32,
    fog_height_falloff: f32,
    fog_sun_inscatter: f32,
    fog_anisotropy: f32,
    // Auto-exposure: enabled (0/1), key value, min/max metered luminance, adaptation speed.
    auto_exposure_enabled: f32,
    auto_exposure_key: f32,
    auto_exposure_min: f32,
    auto_exposure_max: f32,
    auto_exposure_speed: f32,
    // Photographic grade (post-AgX look). Exposed so film-stock emulation + the physical camera can drive
    // them (previously baked as Settings3D defaults). Consumed by the tonemap shader.
    agx_look: f32,
    wb_temperature: f32,
    wb_tint: f32,
    vignette_strength: f32,
    vignette_softness: f32,
    grain_amount: f32,
    chromatic_aberration: f32,
    // Lens optics (physical-camera native effects). Radial distortion applied as a UV remap in tonemap.
    lens_distortion_k1: f32,
    lens_distortion_k2: f32,
    // Aperture bokeh shape (extends the DoF gather): blade count (0/<3 = circular) + anamorphic squeeze.
    bokeh_blades: f32,
    bokeh_anamorphic: f32,
};

comptime {
    // ABI guard: this struct is passed BY VALUE to C# (ZgRenderSettings3D in ZgStructs.cs) with no
    // per-field check, so a field inserted on one side silently shifts every downstream setting. Pin
    // the size here and report it through ZgAbiInfo (abi_version 7) so a mismatch fails loudly at
    // startup instead of corrupting the render settings. Keep 70 f32 in lockstep with the C# struct.
    if (@sizeOf(ZgRenderSettings3D) != 70 * @sizeOf(f32))
        @compileError("ZgRenderSettings3D must be 70 f32 (280 bytes) — keep field count in sync with C#");
}

fn settingsToWire(s: wgpu_renderer.wgpu_3d.Settings3D) ZgRenderSettings3D {
    return .{
        .ambient_intensity = s.ambient_intensity,
        .sky_horizon_r = s.sky_horizon[0],
        .sky_horizon_g = s.sky_horizon[1],
        .sky_horizon_b = s.sky_horizon[2],
        .sky_zenith_r = s.sky_zenith[0],
        .sky_zenith_g = s.sky_zenith[1],
        .sky_zenith_b = s.sky_zenith[2],
        .sky_ground_r = s.sky_ground[0],
        .sky_ground_g = s.sky_ground[1],
        .sky_ground_b = s.sky_ground[2],
        .env_avg_r = s.env_avg[0],
        .env_avg_g = s.env_avg[1],
        .env_avg_b = s.env_avg[2],
        .sun_azimuth_deg = s.sun_azimuth_deg,
        .sun_elevation_deg = s.sun_elevation_deg,
        .sun_intensity = s.sun_intensity,
        .overhead = s.overhead,
        .horizon_glow = s.horizon_glow,
        .sun_sharpness = s.sun_sharpness,
        .exposure = s.exposure,
        .contrast = s.contrast,
        .saturation = s.saturation,
        .shadow_strength = s.shadow_strength,
        .shadow_bias = s.shadow_bias,
        .shadow_softness = s.shadow_softness,
        .clearcoat = s.clearcoat,
        .bloom_threshold = s.bloom_threshold,
        .bloom_knee = s.bloom_knee,
        .bloom_intensity = s.bloom_intensity,
        .ssao_radius = s.ssao_radius,
        .ssao_bias = s.ssao_bias,
        .ssao_strength = s.ssao_strength,
        .ssao_power = s.ssao_power,
        .ssr_intensity = s.ssr_intensity,
        .ssr_max_distance = s.ssr_max_distance,
        .ssr_thickness = s.ssr_thickness,
        .ssr_steps = s.ssr_steps,
        .taa_enabled = s.taa_enabled,
        .taa_feedback = s.taa_feedback,
        .diagnostic_mode = s.diagnostic_mode,
        .debug_view = s.debug_view,
        .dof_enabled = s.dof_enabled,
        .dof_focus_distance = s.dof_focus_distance,
        .dof_f_stop = s.dof_f_stop,
        .dof_max_coc = s.dof_max_coc,
        .wireframe = s.wireframe,
        .fog_density = s.fog_density,
        .fog_color_r = s.fog_color[0],
        .fog_color_g = s.fog_color[1],
        .fog_color_b = s.fog_color[2],
        .fog_height = s.fog_height,
        .fog_height_falloff = s.fog_height_falloff,
        .fog_sun_inscatter = s.fog_sun_inscatter,
        .fog_anisotropy = s.fog_anisotropy,
        .auto_exposure_enabled = s.auto_exposure_enabled,
        .auto_exposure_key = s.auto_exposure_key,
        .auto_exposure_min = s.auto_exposure_min,
        .auto_exposure_max = s.auto_exposure_max,
        .auto_exposure_speed = s.auto_exposure_speed,
        .agx_look = s.agx_look,
        .wb_temperature = s.wb_temperature,
        .wb_tint = s.wb_tint,
        .vignette_strength = s.vignette_strength,
        .vignette_softness = s.vignette_softness,
        .grain_amount = s.grain_amount,
        .chromatic_aberration = s.chromatic_aberration,
        .lens_distortion_k1 = s.lens_distortion_k1,
        .lens_distortion_k2 = s.lens_distortion_k2,
        .bokeh_blades = s.bokeh_blades,
        .bokeh_anamorphic = s.bokeh_anamorphic,
    };
}

/// Read the current 3D render settings (use to initialise the editor's Settings tab).
export fn zigote_get_render_settings_3d(handle: u64, out_settings: *ZgRenderSettings3D) void {
    const state = stateFromHandle(handle) orelse {
        out_settings.* = settingsToWire(.{});
        return;
    };
    // Pre-creation reads serve the pending copy (== Gpu3d's own defaults until edited) — reading
    // settings must never instantiate the 3D renderer.
    out_settings.* = settingsToWire(if (state.gpu_3d) |g| g.settings else state.pending_settings_3d);
}

/// Decode the flat C-ABI wire struct into renderer settings. The single decode shared by the live
/// and pre-creation (pending) branches of zigote_set_render_settings_3d so the two never drift.
fn settingsFromWire(w: ZgRenderSettings3D) wgpu_renderer.wgpu_3d.Settings3D {
    return .{
        .ambient_intensity = w.ambient_intensity,
        .sky_horizon = .{ w.sky_horizon_r, w.sky_horizon_g, w.sky_horizon_b },
        .sky_zenith = .{ w.sky_zenith_r, w.sky_zenith_g, w.sky_zenith_b },
        .sky_ground = .{ w.sky_ground_r, w.sky_ground_g, w.sky_ground_b },
        .env_avg = .{ w.env_avg_r, w.env_avg_g, w.env_avg_b },
        .sun_azimuth_deg = w.sun_azimuth_deg,
        .sun_elevation_deg = w.sun_elevation_deg,
        .sun_intensity = w.sun_intensity,
        .overhead = w.overhead,
        .horizon_glow = w.horizon_glow,
        .sun_sharpness = w.sun_sharpness,
        .exposure = w.exposure,
        .contrast = w.contrast,
        .saturation = w.saturation,
        .shadow_strength = w.shadow_strength,
        .shadow_bias = w.shadow_bias,
        .shadow_softness = w.shadow_softness,
        .clearcoat = w.clearcoat,
        .bloom_threshold = w.bloom_threshold,
        .bloom_knee = w.bloom_knee,
        .bloom_intensity = w.bloom_intensity,
        .ssao_radius = w.ssao_radius,
        .ssao_bias = w.ssao_bias,
        .ssao_strength = w.ssao_strength,
        .ssao_power = w.ssao_power,
        .ssr_intensity = w.ssr_intensity,
        .ssr_max_distance = w.ssr_max_distance,
        .ssr_thickness = w.ssr_thickness,
        .ssr_steps = w.ssr_steps,
        .taa_enabled = w.taa_enabled,
        .taa_feedback = w.taa_feedback,
        .diagnostic_mode = w.diagnostic_mode,
        .debug_view = w.debug_view,
        .dof_enabled = w.dof_enabled,
        .dof_focus_distance = w.dof_focus_distance,
        .dof_f_stop = w.dof_f_stop,
        .dof_max_coc = w.dof_max_coc,
        .wireframe = w.wireframe,
        .fog_density = w.fog_density,
        .fog_color = .{ w.fog_color_r, w.fog_color_g, w.fog_color_b },
        .fog_height = w.fog_height,
        .fog_height_falloff = w.fog_height_falloff,
        .fog_sun_inscatter = w.fog_sun_inscatter,
        .fog_anisotropy = w.fog_anisotropy,
        .auto_exposure_enabled = w.auto_exposure_enabled,
        .auto_exposure_key = w.auto_exposure_key,
        .auto_exposure_min = w.auto_exposure_min,
        .auto_exposure_max = w.auto_exposure_max,
        .auto_exposure_speed = w.auto_exposure_speed,
        .agx_look = w.agx_look,
        .wb_temperature = w.wb_temperature,
        .wb_tint = w.wb_tint,
        .vignette_strength = w.vignette_strength,
        .vignette_softness = w.vignette_softness,
        .grain_amount = w.grain_amount,
        .chromatic_aberration = w.chromatic_aberration,
        .lens_distortion_k1 = w.lens_distortion_k1,
        .lens_distortion_k2 = w.lens_distortion_k2,
        .bokeh_blades = w.bokeh_blades,
        .bokeh_anamorphic = w.bokeh_anamorphic,
    };
}

/// Apply 3D render settings from the editor's Settings tab.
export fn zigote_set_render_settings_3d(handle: u64, settings: ZgRenderSettings3D) void {
    const state = stateFromHandle(handle) orelse return;
    const new_settings = settingsFromWire(settings);
    const g = state.gpu_3d orelse {
        // Must NOT create the renderer — the debug-menu Renderer panel writes settings in UI-only
        // apps too. Remember them; ensure3d applies the pending copy at creation (env_dirty
        // defaults true there, so sky/sun inputs bake on the first 3D render without a diff).
        state.pending_settings_3d = new_settings;
        return;
    };
    const old = g.settings;
    g.settings = new_settings;
    // Only rebake the environment cubemap when a sky/sun/studio input actually changed. The rebake (GGX
    // prefilter, 6 faces × 6 mips) is expensive, and per-frame callers — the physical camera updates DoF/
    // exposure/grade every frame — would otherwise force a full rebake each frame and tank the frame rate.
    const ns = g.settings;
    const env_changed =
        old.ambient_intensity != ns.ambient_intensity or
        !std.meta.eql(old.sky_horizon, ns.sky_horizon) or
        !std.meta.eql(old.sky_zenith, ns.sky_zenith) or
        !std.meta.eql(old.sky_ground, ns.sky_ground) or
        !std.meta.eql(old.env_avg, ns.env_avg) or
        old.sun_azimuth_deg != ns.sun_azimuth_deg or
        old.sun_elevation_deg != ns.sun_elevation_deg or
        old.sun_intensity != ns.sun_intensity or
        old.overhead != ns.overhead or
        old.horizon_glow != ns.horizon_glow or
        old.sun_sharpness != ns.sun_sharpness;
    if (env_changed) g.env_dirty = true;
}

/// Switch IBL to an HDRI panorama. `data` is an encoded image (PNG/JPEG/etc.); it is decoded,
/// uploaded, GGX-prefiltered into the cubemap, and used for all reflections until cleared.
export fn zigote_set_environment_hdri(handle: u64, data_ptr: [*]const u8, data_len: usize) void {
    const state = stateFromHandle(handle) orelse return;
    // Environment bakes into Gpu3d-owned resources — only 3D hosts call this, so creating here is right.
    const g3d = ensure3d(state) orelse return;
    const file_data = data_ptr[0..data_len];
    var w: u32 = 0;
    var h: u32 = 0;

    // Radiance .hdr (RGBE) → true float HDR path (bright sources survive into reflections).
    if (isRadianceHdr(file_data)) {
        const pixels = decodeRadianceHdr(state.allocator, file_data, &w, &h) orelse {
            std.log.err("zigote: Radiance HDR decode failed", .{});
            return;
        };
        defer state.allocator.free(pixels);
        g3d.setEnvironmentHdriFloat(state.queue, pixels, w, h) catch |err| {
            std.log.err("zigote: setEnvironmentHdriFloat failed: {}", .{err});
        };
        std.log.info("zigote: HDR environment loaded {d}x{d}", .{ w, h });
        return;
    }

    // LDR (PNG/JPEG/WebP) equirect → clamped [0,1] environment.
    const pixels = loadTextureBytes(state.allocator, file_data, &w, &h) orelse {
        std.log.err("zigote: environment image decode failed", .{});
        return;
    };
    defer state.allocator.free(pixels);
    g3d.setEnvironmentHdri(state.queue, pixels, w, h) catch |err| {
        std.log.err("zigote: setEnvironmentHdri failed: {}", .{err});
    };
}

fn isRadianceHdr(data: []const u8) bool {
    return std.mem.startsWith(u8, data, "#?RADIANCE") or std.mem.startsWith(u8, data, "#?RGBE");
}

/// Decode a Radiance RGBE (.hdr) equirectangular image into tightly-packed rgba16-float pixels
/// (8 bytes/texel, alpha = 1). Handles the new-format RLE scanlines that DCC tools (Blender) emit,
/// with a fallback to flat (non-RLE) rows. Returns null on a malformed file.
fn decodeRadianceHdr(allocator: std.mem.Allocator, data: []const u8, out_w: *u32, out_h: *u32) ?[]u8 {
    var i: usize = 0;
    // ── Header: text lines until a blank line, then the resolution line ──
    var width: usize = 0;
    var height: usize = 0;
    while (i < data.len) {
        const line_start = i;
        while (i < data.len and data[i] != '\n') i += 1;
        const line = data[line_start..i];
        if (i < data.len) i += 1; // consume '\n'
        if (line.len == 0) {
            // Blank line → next line is the resolution.
            const rs = i;
            while (i < data.len and data[i] != '\n') i += 1;
            const res = data[rs..i];
            if (i < data.len) i += 1;
            parseResolution(res, &width, &height);
            break;
        }
    }
    if (width == 0 or height == 0) return null;

    const pixel_count = width * height;
    var rgbe = allocator.alloc(u8, pixel_count * 4) catch return null;
    defer allocator.free(rgbe);

    // ── Scanlines ──
    var y: usize = 0;
    while (y < height) : (y += 1) {
        const row = rgbe[(y * width) * 4 ..][0 .. width * 4];
        if (!decodeHdrScanline(data, &i, width, row)) return null;
    }

    // ── RGBE → rgba16f ──
    var out = allocator.alloc(u8, pixel_count * 8) catch return null;
    var p: usize = 0;
    while (p < pixel_count) : (p += 1) {
        const e = rgbe[p * 4 + 3];
        var r: f32 = 0;
        var g: f32 = 0;
        var b: f32 = 0;
        if (e != 0) {
            const f = std.math.ldexp(@as(f32, 1.0), @as(i32, e) - (128 + 8));
            // Clamp below f16's max (65504): an HDRI sun can exceed it, and @floatCast to f16 would
            // produce +inf → NaN in the prefilter → blue/black speckle artifacts in reflections.
            const max_hdr: f32 = 60000.0;
            r = @min(@as(f32, @floatFromInt(rgbe[p * 4 + 0])) * f, max_hdr);
            g = @min(@as(f32, @floatFromInt(rgbe[p * 4 + 1])) * f, max_hdr);
            b = @min(@as(f32, @floatFromInt(rgbe[p * 4 + 2])) * f, max_hdr);
        }
        writeF16(out[p * 8 + 0 ..], r);
        writeF16(out[p * 8 + 2 ..], g);
        writeF16(out[p * 8 + 4 ..], b);
        writeF16(out[p * 8 + 6 ..], 1.0);
    }

    out_w.* = @intCast(width);
    out_h.* = @intCast(height);
    return out;
}

fn writeF16(dst: []u8, value: f32) void {
    const h: f16 = @floatCast(value);
    const bits: u16 = @bitCast(h);
    dst[0] = @truncate(bits);
    dst[1] = @truncate(bits >> 8);
}

fn parseResolution(res: []const u8, w: *usize, h: *usize) void {
    // Format: "-Y <height> +X <width>" (most common). Parse the two integers in order.
    var it = std.mem.tokenizeAny(u8, res, " \t\r");
    var nums: [2]usize = .{ 0, 0 };
    var n: usize = 0;
    while (it.next()) |tok| {
        if (std.fmt.parseInt(usize, tok, 10)) |v| {
            if (n < 2) {
                nums[n] = v;
                n += 1;
            }
        } else |_| {}
    }
    // First number after -Y is height, second after +X is width.
    h.* = nums[0];
    w.* = nums[1];
}

fn decodeHdrScanline(data: []const u8, i: *usize, width: usize, row: []u8) bool {
    // New-format RLE marker: 2, 2, (width hi), (width lo).
    if (i.* + 4 > data.len) return false;
    const b0 = data[i.*];
    const b1 = data[i.* + 1];
    const b2 = data[i.* + 2];
    const b3 = data[i.* + 3];
    const new_rle = b0 == 2 and b1 == 2 and ((@as(usize, b2) << 8) | b3) == width and width >= 8 and width <= 0x7fff;
    if (!new_rle) {
        // Flat scanline: width × RGBE quadruplets, copied verbatim.
        if (i.* + width * 4 > data.len) return false;
        @memcpy(row, data[i.*..][0 .. width * 4]);
        i.* += width * 4;
        return true;
    }
    i.* += 4;
    // Each of the 4 channels is RLE-encoded across the row, then interleaved into row[].
    var c: usize = 0;
    while (c < 4) : (c += 1) {
        var x: usize = 0;
        while (x < width) {
            if (i.* >= data.len) return false;
            const count = data[i.*];
            i.* += 1;
            if (count > 128) {
                // Run: repeat the next byte (count-128) times.
                if (i.* >= data.len) return false;
                const val = data[i.*];
                i.* += 1;
                var k: usize = 0;
                while (k < count - 128 and x < width) : (k += 1) {
                    row[x * 4 + c] = val;
                    x += 1;
                }
            } else {
                // Dump: copy `count` literal bytes.
                var k: usize = 0;
                while (k < count and x < width) : (k += 1) {
                    if (i.* >= data.len) return false;
                    row[x * 4 + c] = data[i.*];
                    i.* += 1;
                    x += 1;
                }
            }
        }
    }
    return true;
}

/// Revert IBL to the built-in procedural studio environment.
export fn zigote_set_environment_procedural(handle: u64) void {
    const state = stateFromHandle(handle) orelse return;
    // Procedural is already Gpu3d's default, but a host toggling back FROM an HDRI needs the call
    // to land on a live renderer — and only 3D hosts call this.
    const g3d = ensure3d(state) orelse return;
    g3d.setEnvironmentProcedural();
}

/// Set the reflection-probe box (EEVEE-style box-projected env reflection). World-space centre +
/// half-extents; all-zero extents clears it (reverts to infinite env). Box-projects the env cubemap
/// so reflections appear anchored to a finite room rather than an infinitely-distant sky.
export fn zigote_set_reflection_probe(handle: u64, cx: f32, cy: f32, cz: f32, ex: f32, ey: f32, ez: f32) void {
    const state = stateFromHandle(handle) orelse return;
    const g3d = ensure3d(state) orelse return;
    g3d.setReflectionProbe(.{ cx, cy, cz }, .{ ex, ey, ez });
}

/// Per-frame engine statistics for the debug overlay/profiler (design doc §14.1). Cheap snapshot —
/// no allocation, no GPU stall. Mirrors ZgEngineStats in C#.
pub const ZgEngineStats = extern struct {
    frame_index: u64,
    draw_calls: u32,
    triangles: u32,
    render_passes: u32,
    visible_objects: u32,
    gpu_buffer_memory: u64,
    gpu_texture_memory: u64,
};

export fn zigote_debug_get_engine_stats(handle: u64, out_stats: *ZgEngineStats) void {
    const state = stateFromHandle(handle) orelse {
        out_stats.* = std.mem.zeroes(ZgEngineStats);
        return;
    };
    var out = std.mem.zeroes(ZgEngineStats);

    // 3D counters + memory, only when the 3D renderer is live. CRITICAL: must NOT create it —
    // DebugStats.cs polls this in every App-based host (gallery included, every 0.4 s); an ensure3d
    // here would silently defeat the lazy init.
    if (state.gpu_3d) |g3d| {
        const s = g3d.stats;
        out.frame_index = s.frame_index;
        out.draw_calls = s.draw_calls;
        out.triangles = s.triangles;
        out.render_passes = s.render_passes;
        out.visible_objects = s.visible_objects;
        // Render targets (shadow/env/G-buffer/post/TAA) are the dominant GPU texture cost; mesh
        // vertex/index/edge buffers are the tracked buffer cost.
        out.gpu_texture_memory += g3d.targetMemoryBytes().total;
        out.gpu_buffer_memory += g3d.meshBufferBytes();
    }

    // 2D UI GPU memory is always present (the 2D renderer is created eagerly), so a pure-2D app still
    // gets real buffer/texture numbers: glyph + emoji atlases, the glass capture targets (textures)
    // and the paint vertex-buffer rings (buffers).
    const um = state.gpu_ui.memoryBytes();
    out.gpu_texture_memory += um.coverage_atlas + um.emoji_atlas + um.scene + um.backdrop;
    out.gpu_buffer_memory += um.vertex_buffers;

    // Cached images count too. Leaving them out made this number a poor guide in exactly the app
    // that needs it most: in anything art-heavy — a music library, a photo grid — the image cache
    // dwarfs every atlas here, so a stats panel reporting only atlases showed a flat few megabytes
    // while the process grew by hundreds. Walked rather than tracked incrementally because the
    // registry is the authority on what is actually resident.
    {
        state.image_lock.lock();
        defer state.image_lock.unlock();
        var it = state.image_registry.iterator();
        while (it.next()) |entry| {
            if (state.gpu_ui.image_cache.contains(entry.key_ptr.*))
                out.gpu_texture_memory += @as(u64, entry.value_ptr.width) * entry.value_ptr.height * 4;
        }
    }

    out_stats.* = out;
}

/// Turn an already-decoded image into an owned RGBA8 buffer, *moving* rather than copying.
///
/// Both branches used to end in `allocator.dupe`, which meant a full-size second copy of an image
/// that had just been allocated and was about to be thrown away — the single largest transient in
/// the whole load path. The storage the decoder (or the converter) produced is already exactly the
/// buffer we want, so it is handed over and the source union is set to `.invalid`, whose `deinit`
/// is a no-op. Ownership of the returned slice passes to the caller.
fn toRgbaOwned(allocator: std.mem.Allocator, img: *zigimg.Image) ?[]u8 {
    // RGBA8 PNGs already decode straight to rgba32. Converting rgba32 -> rgba32 trips an error in
    // zigimg's PixelFormatConverter, which previously dropped every truecolour+alpha texture (it
    // rendered as flat white), so this case must be taken before the converter.
    if (img.pixels == .rgba32) {
        const bytes = std.mem.sliceAsBytes(img.pixels.rgba32);
        img.pixels = .{ .invalid = {} };
        return bytes;
    }

    var rgba = zigimg.PixelFormatConverter.convert(allocator, &img.pixels, .rgba32) catch |err| {
        std.log.err("zigote: pixel-format convert to rgba32 failed: {}", .{err});
        return null;
    };
    const bytes = std.mem.sliceAsBytes(rgba.rgba32);
    rgba = .{ .invalid = {} };
    return bytes;
}

fn loadTextureBytes(allocator: std.mem.Allocator, file_data: []const u8, out_w: *u32, out_h: *u32) ?[]u8 {
    if (isWebP(file_data)) {
        return decodeWebP(allocator, file_data, out_w, out_h);
    } else {
        // PNG/JPG/GIF/etc. all decode via zigimg (GIF used to use a C giflib path; zigimg handles it).
        var img = zigimg.Image.fromMemory(allocator, file_data) catch |err| {
            std.log.err("zigote: image decode failed: {}", .{err});
            return null;
        };
        defer img.deinit(allocator);

        out_w.* = @intCast(img.width);
        out_h.* = @intCast(img.height);
        return toRgbaOwned(allocator, &img);
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

// The audio handle table is written from a loader thread (a host opens a track off the UI thread
// so the container-header parse does not hitch the frame) and read by the frame loop, and
// zigote_audio_reopen frees the whole AudioState out from under both. Every entry point must take
// audio_lock, and the failure mode of forgetting one is not a test failure somewhere else — it is
// a rare use-after-free in a release build hours into playback. So the check is on the source: a
// new zigote_audio_* export that does not lock fails here, at the moment it is written.
test "every audio FFI export takes audio_lock" {
    const src = @embedFile("root.zig");
    // Stateless and slow (a whole-file decode): deliberately outside the lock, see the exports.
    const exempt = [_][]const u8{ "zigote_audio_decode_file", "zigote_audio_decode_free" };

    const decl = "export fn zigote_audio_";
    var checked: usize = 0;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, src, at, decl)) |found| {
        at = found + decl.len;
        // Only a real declaration, which starts a line. Skips this test's own mention of the string.
        if (found != 0 and src[found - 1] != '\n') continue;

        const name_start = found + "export fn ".len;
        const name = src[name_start..std.mem.indexOfScalarPos(u8, src, name_start, '(').?];

        const exempted = for (exempt) |e| {
            if (std.mem.eql(u8, name, e)) break true;
        } else false;
        if (exempted) continue;

        // The lock must come before anything else touches the state, so look only at the head of
        // the body — far enough for the stateFromHandle line, short enough that a lock buried after
        // a state access does not pass.
        const head = src[found..@min(src.len, found + 320)];
        if (std.mem.indexOf(u8, head, "state.audio_lock.lock();") == null) {
            std.debug.print("unguarded audio export: {s}\n", .{name});
            return error.AudioExportNotLocked;
        }
        checked += 1;
    }

    // A rename that silently matches nothing would otherwise make this test vacuously pass.
    try std.testing.expect(checked >= 40);
}

// Every GPU handle an app can hold — a decoded image, a render texture — is a key in the one image
// cache, so they must come from the one counter. Two counters is what this was, both starting at 1,
// and the collision stayed silent until it freed a live texture: whichever registered second
// replaced the other's cache entry, and releasing the image released the render texture's GPU
// memory, which surfaced as a wgpu panic in the blur pass a frame after a resize. Nothing about a
// second counter looks wrong at the call site, and no test without a GPU can catch the consequence
// — so the check is on the source, where adding one is visible.
test "one counter hands out every GPU handle" {
    const src = @embedFile("root.zig");
    const decl = "\n    next_";

    var counters: usize = 0;
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, src, at, decl)) |found| {
        at = found + decl.len;
        // A field declaration is `    name: type,` — anything else on the line is not one.
        const name_start = found + decl.len - "next_".len;
        const colon = std.mem.indexOfScalarPos(u8, src, name_start, ':') orelse continue;
        const name = src[name_start..colon];
        if (std.mem.indexOfScalar(u8, name, ' ') != null) continue;
        if (!std.mem.endsWith(u8, name, "_handle")) continue;

        if (!std.mem.eql(u8, name, "next_gpu_handle")) {
            std.debug.print("second GPU handle counter: {s}\n", .{name});
            return error.SplitHandleSpace;
        }
        counters += 1;
    }

    // The counter must exist, or a rename makes this pass by matching nothing at all.
    try std.testing.expectEqual(@as(usize, 1), counters);
}

// downsampleFromPixels reads the decoder's own buffer rather than a converted RGBA copy, so the
// two things that can silently go wrong are channel order (every image tinted) and source
// indexing (every image mirrored or sheared). Both are invisible in a thumbnail unless checked:
// a 4×1 strip of red|red|blue|blue must halve to exactly red|blue.
test "downsampleFromPixels preserves channel order and position" {
    const allocator = std.testing.allocator;

    const red = zigimg.color.Rgb24{ .r = 200, .g = 20, .b = 10 };
    const blue = zigimg.color.Rgb24{ .r = 10, .g = 20, .b = 200 };
    var strip = [_]zigimg.color.Rgb24{ red, red, blue, blue };
    const storage = zigimg.color.PixelStorage{ .rgb24 = strip[0..] };

    const scaled = downsampleFromPixels(allocator, &storage, 4, 1, 2) orelse
        return error.ExpectedScaling;
    defer allocator.free(scaled.pixels);

    try std.testing.expectEqual(@as(u32, 2), scaled.width);
    try std.testing.expectEqual(@as(u32, 1), scaled.height);
    // Left destination pixel is the average of two reds, i.e. red — not blue, and not (200,20,200).
    try std.testing.expectEqualSlices(u8, &.{ 200, 20, 10, 255 }, scaled.pixels[0..4]);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 200, 255 }, scaled.pixels[4..8]);
}

// rgba32 sources must keep their alpha, and an image already within max_dim must report "nothing
// to do" so the caller moves the decoded buffer out whole instead of copying it.
test "downsampleFromPixels handles alpha and declines when no scaling is needed" {
    const allocator = std.testing.allocator;

    var pixels = [_]zigimg.color.Rgba32{
        .{ .r = 10, .g = 20, .b = 30, .a = 40 },
        .{ .r = 10, .g = 20, .b = 30, .a = 40 },
    };
    const storage = zigimg.color.PixelStorage{ .rgba32 = pixels[0..] };

    const scaled = downsampleFromPixels(allocator, &storage, 2, 1, 1) orelse
        return error.ExpectedScaling;
    defer allocator.free(scaled.pixels);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 40 }, scaled.pixels[0..4]);

    // Already small enough, and a zero axis, both decline rather than allocating or indexing.
    try std.testing.expect(downsampleFromPixels(allocator, &storage, 2, 1, 64) == null);
    try std.testing.expect(downsampleFromPixels(allocator, &storage, 2, 0, 1) == null);
}
