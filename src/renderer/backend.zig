//! GPU backend abstraction (the RHI seam) for the Zigote renderer.
//!
//! Zigote's default — and today **only** — GPU path is **wgpu-native** (portable across
//! Metal/Vulkan/D3D12/GL under the hood). The seam is kept so future native backends
//! (Vulkan/D3D12) can be added to reach vendor features wgpu does not expose — **hardware ray
//! tracing** and **vendor upscalers** (DLSS / FSR / XeSS). (Verified: the linked wgpu-native 29
//! C API has no acceleration-structure host API and no external texture import.) The `Upscaler`/
//! `RayTracer` sub-interfaces below are the reserved contract; no backend implements them today.
//!
//! This file defines the *contract* every backend implements. wgpu is the only
//! implementation (`wgpu_backend.zig`).
//!
//! Backend selection happens **once at init** (the GPU device is created once and never
//! recreated), so it is an init-time choice, never a per-frame setting.
//!
//! Levels:
//!   - **Level 1 — `GpuBackend`** (this vtable): device/surface/frame lifecycle.
//!   - **Level 2 — renderers** (`GpuUi`, `Gpu3d`): per-backend, built on Level-1 primitives.
//!   - **Optional sub-interfaces** — `Upscaler`, `RayTracer`: implemented by native backends
//!     in later stages, gated by `Caps`.

/// Which GPU backend the engine drives. Chosen once at init. Mirrors C# `RenderBackend`.
pub const BackendId = enum(u32) {
    /// Pick the best available backend for the platform/hardware. Resolves to `.wgpu` (the only
    /// implemented backend today).
    auto = 0,
    /// wgpu-native — the portable default and only implemented backend. No host-side ray tracing
    /// or vendor upscaler access.
    wgpu = 1,
    /// Vulkan (Linux/Windows) — DLSS/FSR/XeSS + KHR ray tracing. (planned, reserved)
    vulkan = 3,
    /// Direct3D 12 (Windows) — DLSS/FSR/XeSS + DXR. (planned, reserved)
    d3d12 = 4,

    /// Decode a raw FFI value, defaulting to `.auto` on anything unrecognised.
    pub fn fromU32(v: u32) BackendId {
        return switch (v) {
            1 => .wgpu,
            3 => .vulkan,
            4 => .d3d12,
            else => .auto,
        };
    }
};

/// Resolve a requested backend to the one actually used on this build/platform. Only wgpu is
/// implemented today, so every request resolves to `.wgpu` (a downgrade is logged once by the
/// caller). As native backends land, their arms return themselves.
pub fn resolve(requested: BackendId) BackendId {
    return switch (requested) {
        .auto, .wgpu => .wgpu,
        // Not implemented yet — fall back to the portable default.
        .vulkan, .d3d12 => .wgpu,
    };
}

/// Temporal-upscaler families. A backend reports which it can provide via `Caps.upscalers`;
/// they all share one contract (color + depth + motion vectors + jitter → upscaled color).
/// Reserved for future native backends — wgpu provides none.
pub const UpscalerKind = enum(u32) {
    dlss = 1, // NVIDIA DLSS via Streamline (Vulkan/D3D12)
    fsr = 2, // AMD FidelityFX Super Resolution (any)
    xess = 3, // Intel XeSS (Vulkan/D3D12)

    /// This kind's bit within `Caps.upscalers`.
    pub fn bit(self: UpscalerKind) u32 {
        return @as(u32, 1) << @intCast(@intFromEnum(self));
    }
};

/// Runtime capabilities of the active backend, queried by the host **after** init (once the
/// device exists and its real features are known). Mirrors the `ZgRendererCaps` extern struct
/// in `ffi/root.zig` and C# `ZgRendererCaps` — keep all three in sync.
pub const Caps = struct {
    /// The backend actually selected (`auto` may have fallen back to wgpu).
    active_backend: BackendId = .wgpu,
    /// Bitset of supported `UpscalerKind` (OR of `UpscalerKind.bit()`). 0 = none available.
    upscalers: u32 = 0,
    /// Hardware ray tracing available (acceleration structures + intersection).
    raytracing: bool = false,
    /// Ray tracing usable from raster fragment shaders (Apple-silicon class), not only compute.
    raytracing_from_render: bool = false,

    pub fn supportsUpscaler(self: Caps, kind: UpscalerKind) bool {
        return (self.upscalers & kind.bit()) != 0;
    }
};

/// Opaque per-frame color target (a swapchain image / drawable view). Backend-defined; the
/// host treats it as a token to hand to the active backend's 2D/3D composite path.
pub const ColorTarget = *anyopaque;

/// Result of acquiring the current frame's color target.
pub const Acquire = union(enum) {
    /// Got a target to render the frame into.
    target: ColorTarget,
    /// Swapchain unavailable this frame (timeout / outdated / occluded). Skip and retry next
    /// frame without advancing frame state — matches the existing dropped-frame behaviour.
    skip: void,
};

/// **Level 1** — device/surface/frame lifecycle. One implementation per backend; the active
/// instance lives on the engine state and is chosen once at init. Renderers (Level 2) build
/// on these primitives. A native backend additionally drives 2D+3D and presents the whole
/// frame; wgpu remains the default.
pub const GpuBackend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Reconfigure the swapchain to a new pixel size (window resize).
        configure: *const fn (ptr: *anyopaque, width: u32, height: u32) void,
        /// Acquire the current frame's color target, or signal `skip` if unavailable.
        acquire: *const fn (ptr: *anyopaque) Acquire,
        /// Present the acquired frame. Returns false on present failure.
        present: *const fn (ptr: *anyopaque) bool,
        /// Report runtime capabilities (upscalers / ray tracing / active backend).
        caps: *const fn (ptr: *anyopaque) Caps,
        /// Release backend-owned resources. The host releases shared handles it owns.
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub inline fn configure(self: GpuBackend, w: u32, h: u32) void {
        self.vtable.configure(self.ptr, w, h);
    }
    pub inline fn acquire(self: GpuBackend) Acquire {
        return self.vtable.acquire(self.ptr);
    }
    pub inline fn present(self: GpuBackend) bool {
        return self.vtable.present(self.ptr);
    }
    pub inline fn caps(self: GpuBackend) Caps {
        return self.vtable.caps(self.ptr);
    }
    pub inline fn deinit(self: GpuBackend) void {
        self.vtable.deinit(self.ptr);
    }
};

/// **Optional sub-interface** — temporal upscaler (MetalFX / DLSS / FSR / XeSS). Implemented
/// by native backends in a later stage; gated by `Caps.upscalers`. Defined here so the
/// contract is uniform across backends. Inputs mirror what the renderer already produces for
/// TAA (jittered color + depth + screen-space motion vectors).
pub const Upscaler = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const EncodeInfo = struct {
        color: ColorTarget,
        depth: ColorTarget,
        motion: ColorTarget,
        output: ColorTarget,
        gbuf_pos: ColorTarget,
        jitter_x: f32,
        jitter_y: f32,
        exposure: f32,
        cur_view_proj: [16]f32,
        prev_view_proj: [16]f32,
        inv_view: [16]f32,
        /// Discard temporal history this frame (camera cut / first frame).
        reset: bool,
    };

    pub const VTable = struct {
        encode: *const fn (ptr: *anyopaque, cmd: *anyopaque, info: EncodeInfo) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub inline fn encode(self: Upscaler, cmd: *anyopaque, info: EncodeInfo) void {
        self.vtable.encode(self.ptr, cmd, info);
    }
    pub inline fn deinit(self: Upscaler) void {
        self.vtable.deinit(self.ptr);
    }
};

/// **Optional sub-interface** — hardware ray tracing (Metal RT / KHR-RT / DXR). Implemented by
/// native backends in a later stage; gated by `Caps.raytracing`. Hybrid model: build a TLAS
/// over scene instances, trace over the raster G-buffer (shadows → AO → reflections). Defined
/// here so the contract is uniform across backends.
pub const RayTracer = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// (Re)build acceleration structures for the visible instance set.
        buildScene: *const fn (ptr: *anyopaque) void,
        /// Refit the TLAS from updated instance transforms (cheap per-frame path).
        refit: *const fn (ptr: *anyopaque) void,
        /// Trace over the current G-buffer into the RT output target.
        trace: *const fn (ptr: *anyopaque, cmd: *anyopaque, gbuffer: ColorTarget, out: ColorTarget) void,
        deinit: *const fn (ptr: *anyopaque) void,
    };

    pub inline fn buildScene(self: RayTracer) void {
        self.vtable.buildScene(self.ptr);
    }
    pub inline fn refit(self: RayTracer) void {
        self.vtable.refit(self.ptr);
    }
    pub inline fn trace(self: RayTracer, cmd: *anyopaque, gbuffer: ColorTarget, out: ColorTarget) void {
        self.vtable.trace(self.ptr, cmd, gbuffer, out);
    }
    pub inline fn deinit(self: RayTracer) void {
        self.vtable.deinit(self.ptr);
    }
};
