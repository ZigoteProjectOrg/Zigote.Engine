//! GPU adapter selection for multi-GPU machines.
//!
//! A laptop with an integrated + a discrete GPU (and a desktop with two cards) exposes several
//! adapters, often across several graphics APIs — the same physical GPU shows up once per backend
//! the instance was created with (e.g. Vulkan *and* D3D12 on Windows). wgpu's own
//! `requestAdapter(power_preference)` picks one for us, but it gives no way to see the list, no way
//! to override the pick, and its notion of "high performance" can't be steered per app.
//!
//! So we enumerate instead: list every adapter, score it against the app's intent, and hand the
//! winner to `Device` creation directly. The backend follows the adapter — picking an adapter
//! *is* picking the API, which is what a user selecting "NVIDIA (Vulkan)" over "NVIDIA (D3D12)"
//! expects.
//!
//! Intent comes from the host at init:
//!   - `.performance` — 3D apps (editor, player, games): the fastest GPU, discrete first.
//!   - `.efficiency`  — 2D/UI apps: the most power-efficient, integrated first. On a laptop this is
//!     the difference between a UI that sips battery and one that spins up the discrete card to
//!     draw some rectangles.
//!
//! Developers/QA override the pick without a rebuild via `ZIGOTE_GPU` (index or name substring) and
//! `ZIGOTE_GPU_POWER` (`performance` / `efficiency`); the editor exposes the same choice as a
//! setting. Every override is advisory — an adapter that can't present to our surface is skipped,
//! and if nothing survives we fall back to wgpu's own `requestAdapter`, so a bad value degrades to
//! the default instead of failing to boot.

const std = @import("std");
const wgpu = @import("wgpu");

/// How to rank adapters. Mirrors C# `GpuPowerPreference` and `GpuPower` in ffi/root.zig.
pub const Power = enum(u32) {
    /// No app-level intent — rank as `.performance` (what the engine did before this existed).
    auto = 0,
    /// Fastest GPU available: discrete over integrated.
    performance = 1,
    /// Most power-efficient GPU: integrated over discrete.
    efficiency = 2,

    pub fn fromU32(v: u32) Power {
        return switch (v) {
            1 => .performance,
            2 => .efficiency,
            else => .auto,
        };
    }

    /// The equivalent wgpu hint, for the fallback `requestAdapter` path.
    pub fn toWgpu(self: Power) wgpu.PowerPreference {
        return switch (self) {
            .efficiency => .low_power,
            .auto, .performance => .high_performance,
        };
    }
};

/// Max adapters we keep. Far above any real machine (a dual-GPU box across two backends is 4).
pub const max_gpus = 16;

/// One enumerated adapter, as reported to the host. Layout must match ZgGpuInfo in ZgStructs.cs.
pub const GpuInfo = extern struct {
    /// Device name, UTF-8, NUL-padded. Truncated rather than dropped if a driver reports a long one.
    name: [128]u8,
    /// wgpu `BackendType` — the graphics API this adapter drives (Vulkan / D3D12 / Metal / GL).
    backend: u32,
    /// wgpu `AdapterType` — discrete / integrated / cpu / unknown.
    device_type: u32,
    vendor_id: u32,
    device_id: u32,
};

comptime {
    // The engine memcpy's this straight into the host's buffer, so the layout is a hard contract
    // with C# ZgGpuInfo (see AbiLayoutTests). Catch a drift here rather than as a corrupt GPU list.
    std.debug.assert(@sizeOf(GpuInfo) == 144);
    std.debug.assert(@offsetOf(GpuInfo, "backend") == 128);
    std.debug.assert(@offsetOf(GpuInfo, "device_type") == 132);
    std.debug.assert(@offsetOf(GpuInfo, "vendor_id") == 136);
    std.debug.assert(@offsetOf(GpuInfo, "device_id") == 140);
}

/// Fill a GpuInfo from a live adapter. `getInfo` allocates the strings, so free them right after.
fn describe(adapter: *wgpu.Adapter) GpuInfo {
    var out = GpuInfo{
        .name = [_]u8{0} ** 128,
        .backend = 0,
        .device_type = 0,
        .vendor_id = 0,
        .device_id = 0,
    };
    var info: wgpu.AdapterInfo = undefined;
    if (adapter.getInfo(&info) != .success) return out;
    defer info.freeMembers();

    const device = info.device.toSlice() orelse return out;
    const len = @min(device.len, out.name.len - 1); // keep the NUL terminator
    @memcpy(out.name[0..len], device[0..len]);
    out.backend = @intFromEnum(info.backend_type);
    out.device_type = @intFromEnum(info.adapter_type);
    out.vendor_id = info.vendor_id;
    out.device_id = info.device_id;
    return out;
}

/// Rank a graphics API. wgpu itself splits these into "primary" (a full, well-supported feature
/// set) and "secondary" (a fallback path), and we follow that: the SAME physical GPU enumerates
/// once per API, and picking its OpenGL entry over its Vulkan one is a downgrade every time —
/// fewer features, worse performance, and adapters that report themselves as `unknown` because the
/// GL path can't classify them.
fn backendRank(backend: wgpu.BackendType) u8 {
    return switch (backend) {
        .metal, .vulkan, .d3d12 => 2, // primary
        .d3d11, .opengl, .opengl_es => 1, // secondary — fallback only
        else => 0, // undefined / null / webgpu: no reason to prefer it
    };
}

/// Rank an adapter for the requested intent. Higher wins. Three tiers, most significant first:
///
///  1. **Real hardware beats software.** A CPU adapter is a rasterizer running on the CPU; it is
///     orders of magnitude slower than any GPU, so it loses to one no matter which API each uses.
///     This tier is why it is weighted above the API: a machine can expose llvmpipe under Vulkan
///     while its actual GPU is only reachable through OpenGL (Mesa on a Radeon 780M does exactly
///     that), and ranking the API first would quietly hand that machine software rendering.
///  2. **A primary API beats a secondary one** (see `backendRank`) — the same GPU under Vulkan is a
///     better target than under OpenGL.
///  3. **Device type**, per the caller's intent: discrete-first for performance, integrated-first
///     for efficiency.
fn score(backend: wgpu.BackendType, device_type: wgpu.AdapterType, power: Power) u8 {
    const is_hardware: u8 = if (device_type == .cpu) 0 else 1;
    const by_device: u8 = switch (power) {
        .efficiency => switch (device_type) {
            .integrated_gpu => 3,
            .discrete_gpu => 2,
            .unknown => 1,
            .cpu => 0,
        },
        .auto, .performance => switch (device_type) {
            .discrete_gpu => 3,
            .integrated_gpu => 2,
            .unknown => 1,
            .cpu => 0,
        },
    };
    return is_hardware * 16 + backendRank(backend) * 4 + by_device;
}

/// Can this adapter actually drive our window?
///
/// Status alone is not enough: on a Radeon 780M the OpenGL adapter answers `.success` with ZERO
/// surface formats, and picking it takes the engine down at boot with WgpuNoSurfaceFormats. An
/// adapter with no format to present in is not a candidate, whatever it says about itself.
fn canPresent(surface: *wgpu.Surface, adapter: *wgpu.Adapter) bool {
    var caps: wgpu.SurfaceCapabilities = undefined;
    if (surface.getCapabilities(adapter, &caps) != .success) return false;
    defer caps.freeMembers();
    return caps.format_count > 0;
}

/// Does `name` (NUL-padded) contain `needle`, case-insensitively? Used for `ZIGOTE_GPU=nvidia`.
fn nameMatches(name: []const u8, needle: []const u8) bool {
    const end = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
    const haystack = name[0..end];
    if (needle.len == 0 or needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

/// Whether 2× rgba16float MSAA may be used on this adapter.
///
/// `has_format_features` (TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) is necessary but NOT sufficient:
/// it only says adapter-specific format features are exposed, and *which* sample counts those
/// include is per-adapter with no way to query it through wgpu's C API. Mesa's llvmpipe grants the
/// feature and supports [1, 4, 8] — requesting 2 there fails pipeline validation, and wgpu reports
/// a validation error inside an FFI call as a non-unwinding panic, aborting the process rather than
/// returning something we could recover from. So this answers "no" unless we know otherwise: Metal
/// (Apple Silicon) is the backend 2× is verified on. Everything else keeps the spec-guaranteed 4.
pub fn allowsMsaa2(backend: wgpu.BackendType, has_format_features: bool) bool {
    return has_format_features and backend == .metal;
}

/// The outcome of selection: the adapter to create the device on, plus the enumerated list so the
/// host can show what was available and which one won.
pub const Selection = struct {
    adapter: *wgpu.Adapter,
    gpus: [max_gpus]GpuInfo,
    count: u32,
    /// Index into `gpus` of the adapter in use, or -1 when the fallback path produced it (the
    /// fallback adapter is not necessarily one we enumerated).
    active: i32,
};

/// Enumerate every adapter the instance can see, pick one for `power` (honoring an explicit
/// `want_index` or the env overrides), and release the rest.
///
/// Adapters that cannot present to `surface` are never selected — on a multi-GPU laptop the
/// discrete card is often not connected to the panel the window is on, and creating a device on it
/// would produce a swapchain that cannot be presented.
pub fn select(
    instance: *wgpu.Instance,
    surface: *wgpu.Surface,
    backends: wgpu.InstanceBackend,
    requested_power: Power,
    requested_index: i32,
) !Selection {
    var power = requested_power;
    var want_index = requested_index;

    // Env overrides win over the host's request — they exist so a developer can retest on the other
    // GPU without touching code. Unparseable values are ignored, not fatal.
    if (std.c.getenv("ZIGOTE_GPU_POWER")) |raw_c| {
        const raw = std.mem.span(raw_c);
        if (std.ascii.eqlIgnoreCase(raw, "efficiency") or std.ascii.eqlIgnoreCase(raw, "low")) {
            power = .efficiency;
        } else if (std.ascii.eqlIgnoreCase(raw, "performance") or std.ascii.eqlIgnoreCase(raw, "high")) {
            power = .performance;
        } else {
            std.log.warn("zigote: ZIGOTE_GPU_POWER='{s}' not understood (want performance|efficiency)", .{raw});
        }
    }

    // ZIGOTE_GPU takes either an index ("1") or a name substring ("nvidia") — a name is far easier
    // to use when you don't already know the enumeration order.
    var want_name: ?[]const u8 = null;
    if (std.c.getenv("ZIGOTE_GPU")) |raw_c| {
        const raw = std.mem.span(raw_c);
        if (std.fmt.parseInt(i32, raw, 10)) |idx| {
            want_index = idx;
        } else |_| {
            want_name = raw;
        }
    }

    // Filter with the SAME mask the instance was created with, not InstanceBackends.all — `all` is
    // 0x00000000, which is unambiguous for InstanceExtras but is a zero *filter* mask here, and
    // whether wgpu-native reads that as "everything" or "nothing" is not something to leave to
    // chance when the answer decides which GPU we run on.
    var enum_opts = wgpu.EnumerateAdapterOptions{ .backends = backends };
    var adapters: [max_gpus]*wgpu.Adapter = undefined;
    const found = @min(instance.enumerateAdapters(&enum_opts, &adapters), max_gpus);

    var result = Selection{
        .adapter = undefined,
        .gpus = [_]GpuInfo{std.mem.zeroes(GpuInfo)} ** max_gpus,
        .count = @intCast(found),
        .active = -1,
    };

    // Pass 1: describe every adapter. Done for ALL of them before any selection so the list the
    // host shows is complete even when the pick is decided on the first entry.
    //
    // Every candidate is logged with its score and whether it can present. Which GPU an app ends up
    // on is otherwise invisible until something is mysteriously slow or fails to boot, and the
    // interesting cases (a GPU reachable only through OpenGL, a software adapter masquerading under
    // Vulkan) are impossible to diagnose from the winner alone.
    std.log.info("zigote: {d} GPU adapter(s) enumerated:", .{found});
    for (adapters[0..found], 0..) |adapter, i| {
        result.gpus[i] = describe(adapter);
        const g = result.gpus[i];
        std.log.info("zigote:   [{d}] '{s}' backend={d} type={d} present={} score={d}", .{
            i,
            std.mem.sliceTo(&g.name, 0),
            g.backend,
            g.device_type,
            canPresent(surface, adapter),
            score(@enumFromInt(g.backend), @enumFromInt(g.device_type), power),
        });
    }

    // Pass 2: choose. An explicit index or name pins the choice; otherwise rank by intent. Either
    // way an adapter that cannot drive our surface is never eligible.
    const has_override = want_index >= 0 or want_name != null;
    var best: ?usize = null;
    var best_score: u8 = 0;
    for (adapters[0..found], 0..) |adapter, i| {
        if (!canPresent(surface, adapter)) continue;

        if (has_override) {
            const pinned = (want_index == @as(i32, @intCast(i))) or
                (want_name != null and nameMatches(&result.gpus[i].name, want_name.?));
            if (pinned) {
                best = i;
                break;
            }
            continue; // an override is set — nothing else is eligible
        }

        const s = score(
            @enumFromInt(result.gpus[i].backend),
            @enumFromInt(result.gpus[i].device_type),
            power,
        );
        if (best == null or s > best_score) {
            best = i;
            best_score = s;
        }
    }

    if (best) |idx| {
        result.adapter = adapters[idx];
        result.active = @intCast(idx);
        for (adapters[0..found], 0..) |adapter, i| {
            if (i != idx) adapter.release();
        }
        const g = result.gpus[idx];
        std.log.info("zigote: GPU '{s}' (backend {d}, type {d}) selected for {s}", .{
            std.mem.sliceTo(&g.name, 0), g.backend, g.device_type, @tagName(power),
        });
        // Scoring ranks CPU adapters last, so landing on one means it was pinned or it was the only
        // adapter that could present. Either way it is a software rasterizer — say so, because
        // otherwise the only symptom is that everything is inexplicably slow.
        if (@as(wgpu.AdapterType, @enumFromInt(g.device_type)) == .cpu) {
            std.log.warn("zigote: '{s}' is a SOFTWARE renderer — expect very low frame rates", .{
                std.mem.sliceTo(&g.name, 0),
            });
        }
        return result;
    }

    // Nothing usable was enumerated (or an override matched nothing) — let wgpu choose. This is the
    // pre-existing behaviour, so a machine where enumeration misbehaves still boots.
    for (adapters[0..found]) |adapter| adapter.release();
    if (want_index >= 0 or want_name != null) {
        std.log.warn("zigote: requested GPU not found or cannot present; falling back to the default", .{});
    }
    var opts = wgpu.RequestAdapterOptions{
        .power_preference = power.toWgpu(),
        .compatible_surface = surface,
    };
    const resp = instance.requestAdapterSync(&opts, 1_000_000);
    result.adapter = resp.adapter orelse return error.WgpuAdapterUnavailable;
    return result;
}

// ── Tests ─────────────────────────────────────────────────────────────────────
// Scoring and name matching are the parts that can silently do the wrong thing (send a game to the
// integrated chip, or a UI to the discrete one) and they need no GPU to exercise.

test "performance prefers discrete, efficiency prefers integrated" {
    // Within one API, device type decides.
    try std.testing.expect(
        score(.vulkan, .discrete_gpu, .performance) > score(.vulkan, .integrated_gpu, .performance),
    );
    try std.testing.expect(
        score(.vulkan, .integrated_gpu, .efficiency) > score(.vulkan, .discrete_gpu, .efficiency),
    );
    // auto is performance — the historical default.
    try std.testing.expectEqual(
        score(.vulkan, .discrete_gpu, .performance),
        score(.vulkan, .discrete_gpu, .auto),
    );
}

test "real hardware always beats a software adapter, whatever the API" {
    // The regression this guards: Mesa exposes llvmpipe under VULKAN while the actual Radeon 780M
    // is reachable only through OPENGL. Ranking the API above hardware-vs-software would hand this
    // machine software rendering while a real GPU sat idle.
    for ([_]Power{ .auto, .performance, .efficiency }) |p| {
        try std.testing.expect(score(.opengl, .unknown, p) > score(.vulkan, .cpu, p));
        try std.testing.expect(score(.opengl, .integrated_gpu, p) > score(.vulkan, .cpu, p));
        try std.testing.expect(score(.vulkan, .integrated_gpu, p) > score(.vulkan, .cpu, p));
    }
}

test "among hardware adapters, a primary API beats a secondary one" {
    // The same GPU enumerates once per API; its Vulkan entry should win over its OpenGL one even
    // when the GL entry reports a flattering device type (GL often can't classify, hence unknown).
    for ([_]Power{ .auto, .performance, .efficiency }) |p| {
        try std.testing.expect(
            score(.vulkan, .integrated_gpu, p) > score(.opengl, .unknown, p),
        );
        try std.testing.expect(
            score(.vulkan, .unknown, p) > score(.opengl, .discrete_gpu, p),
        );
    }
}

test "primary and secondary APIs are ranked as wgpu classifies them" {
    try std.testing.expectEqual(@as(u8, 2), backendRank(.vulkan));
    try std.testing.expectEqual(@as(u8, 2), backendRank(.metal));
    try std.testing.expectEqual(@as(u8, 2), backendRank(.d3d12));
    try std.testing.expectEqual(@as(u8, 1), backendRank(.opengl));
    try std.testing.expectEqual(@as(u8, 1), backendRank(.opengl_es));
    try std.testing.expectEqual(@as(u8, 1), backendRank(.d3d11));
    try std.testing.expectEqual(@as(u8, 0), backendRank(.undefined));
}

test "2x MSAA is only allowed where it is verified" {
    // The regression this guards: llvmpipe (Vulkan) grants the format-features flag but supports
    // [1, 4, 8], and asking it for 2 aborts the process in pipeline validation.
    try std.testing.expect(!allowsMsaa2(.vulkan, true));
    try std.testing.expect(!allowsMsaa2(.d3d12, true));
    try std.testing.expect(!allowsMsaa2(.opengl, true));
    try std.testing.expect(!allowsMsaa2(.undefined, true));
    // Metal is the one backend it is verified on — and only when the feature is actually granted.
    try std.testing.expect(allowsMsaa2(.metal, true));
    try std.testing.expect(!allowsMsaa2(.metal, false));
}

test "name matching is case-insensitive and NUL-padded-safe" {
    var name = [_]u8{0} ** 128;
    const text = "NVIDIA GeForce RTX 4080";
    @memcpy(name[0..text.len], text);

    try std.testing.expect(nameMatches(&name, "nvidia"));
    try std.testing.expect(nameMatches(&name, "RTX"));
    try std.testing.expect(nameMatches(&name, "4080"));
    try std.testing.expect(!nameMatches(&name, "radeon"));
    try std.testing.expect(!nameMatches(&name, "")); // empty must not match everything
    // Must not read past the NUL into the zero padding.
    try std.testing.expect(!nameMatches(&name, text ++ "x"));
}
