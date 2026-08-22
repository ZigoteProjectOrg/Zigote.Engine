//! Backend identity for the Zigote renderer.
//!
//! **wgpu-native is the backend.** It is portable across Metal/Vulkan/D3D12/GL under the hood,
//! and it is the only implementation. What used to live here — a `GpuBackend` vtable, `Upscaler`
//! and `RayTracer` sub-interfaces, and `vulkan`/`d3d12` enum arms — was a contract with zero
//! implementations that the hot path did not dispatch through anyway (it called
//! `surface.getCurrentTexture` / `surface.present` directly). See docs/v2-design.md §2.4.
//!
//! If a native Vulkan/D3D12 renderer is ever wanted it arrives as a second `renderer_*/`
//! directory selected at **build time** (`-Drenderer=`), resolved by Zig's comptime module
//! switch — not as a runtime vtable and not as capability bits nobody reports.

/// Which GPU backend the engine drives. Chosen once at init; mirrors C# `RenderBackend`.
pub const BackendId = enum(u32) {
    /// Pick the best available backend for the platform. Resolves to `.wgpu`.
    auto = 0,
    /// wgpu-native — the backend.
    wgpu = 1,

    /// Decode a raw FFI value. Anything unrecognised — including the retired `vulkan` (3) and
    /// `d3d12` (4) values an older host may still send — decodes to `.auto`, which resolves to
    /// `.wgpu`: the same backend those arms fell back to when they existed.
    pub fn fromU32(v: u32) BackendId {
        return switch (v) {
            1 => .wgpu,
            else => .auto,
        };
    }
};

/// Resolve a requested backend to the one actually used on this build/platform.
pub fn resolve(requested: BackendId) BackendId {
    return switch (requested) {
        .auto, .wgpu => .wgpu,
    };
}

/// Runtime capabilities of the active backend, queried by the host after init. Mirrors the
/// `ZgRendererCaps` extern struct in `ffi/root.zig` and C# `ZgRendererCaps` — keep in sync.
///
/// wgpu-native 29's C API exposes no vendor upscaler and no acceleration-structure host API
/// (verified against the linked headers), so those two fields are constant `false`/`0` today.
/// They stay in the struct because they are part of the wire contract the host already reads.
pub const Caps = struct {
    /// The backend actually selected.
    active_backend: BackendId = .wgpu,
    /// Bitset of supported upscalers. Always 0 on wgpu.
    upscalers: u32 = 0,
    /// Hardware ray tracing available. Always false on wgpu.
    raytracing: bool = false,
    /// Ray tracing usable from raster fragment shaders. Always false on wgpu.
    raytracing_from_render: bool = false,
};

test "retired backend ids resolve to wgpu" {
    const std = @import("std");
    // 3 and 4 were `vulkan` and `d3d12`; an older host may still send them.
    for ([_]u32{ 0, 1, 3, 4, 99 }) |raw| {
        try std.testing.expectEqual(BackendId.wgpu, resolve(BackendId.fromU32(raw)));
    }
}
