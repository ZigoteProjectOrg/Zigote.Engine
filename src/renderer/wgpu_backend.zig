//! wgpu-native implementation of the `GpuBackend` contract (`backend.zig`).
//!
//! This is the **default** backend and the only one that ships today. It borrows the wgpu
//! handles that `EngineState` creates and owns (instance/surface/adapter/device/queue) plus a
//! pointer to the live `SurfaceConfiguration` (so resize stays coherent with the host). The
//! host still releases those shared handles in `zigote_shutdown`; this backend's `deinit` only
//! drops any per-frame objects it is holding.
//!
//! `caps()` reports **no** upscaler and **no** ray tracing — wgpu-native 29 exposes neither
//! from its C API (verified). Those capabilities arrive with the native backends.

const std = @import("std");
const wgpu = @import("wgpu");
const backend = @import("backend.zig");

pub const WgpuBackend = struct {
    surface: *wgpu.Surface,
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    /// Points at `EngineState.wgpu_config` (owned there) so `configure` updates the same struct.
    config: *wgpu.SurfaceConfiguration,

    // Per-frame state held between `acquire` and `present`. The wgpu composite path renders
    // the UI into `current_view` (the swapchain image view) before `present` is called.
    current_surface_texture: ?wgpu.SurfaceTexture = null,
    current_view: ?*wgpu.TextureView = null,

    pub fn init(
        surface: *wgpu.Surface,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        config: *wgpu.SurfaceConfiguration,
    ) WgpuBackend {
        return .{ .surface = surface, .device = device, .queue = queue, .config = config };
    }

    /// View this backend as the generic `GpuBackend` interface. The pointer must remain stable
    /// for the engine's lifetime (it lives inside `EngineState`).
    pub fn asGpuBackend(self: *WgpuBackend) backend.GpuBackend {
        return .{ .ptr = self, .vtable = &vtable };
    }

    const vtable = backend.GpuBackend.VTable{
        .configure = configure,
        .acquire = acquire,
        .present = present,
        .caps = caps,
        .deinit = deinitImpl,
    };

    fn configure(ptr: *anyopaque, width: u32, height: u32) void {
        const self: *WgpuBackend = @ptrCast(@alignCast(ptr));
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(self.config);
    }

    fn acquire(ptr: *anyopaque) backend.Acquire {
        const self: *WgpuBackend = @ptrCast(@alignCast(ptr));
        var st = wgpu.SurfaceTexture{ .next_in_chain = null, .texture = null, .status = .@"error" };
        self.surface.getCurrentTexture(&st);
        switch (st.status) {
            .success_optimal, .success_suboptimal => {},
            .timeout, .outdated, .lost, .occluded => return .skip,
            else => return .skip,
        }
        const tex = st.texture orelse return .skip;
        const view = tex.createView(null) orelse {
            tex.release();
            return .skip;
        };
        self.current_surface_texture = st;
        self.current_view = view;
        return .{ .target = @ptrCast(view) };
    }

    fn present(ptr: *anyopaque) bool {
        const self: *WgpuBackend = @ptrCast(@alignCast(ptr));
        if (self.current_view) |v| {
            v.release();
            self.current_view = null;
        }
        const ok = self.surface.present() == .success;
        if (self.current_surface_texture) |st| {
            if (st.texture) |t| t.release();
            self.current_surface_texture = null;
        }
        return ok;
    }

    fn caps(ptr: *anyopaque) backend.Caps {
        _ = ptr;
        return .{
            .active_backend = .wgpu,
            .upscalers = 0, // wgpu-native exposes no vendor upscaler
            .raytracing = false, // ...and no acceleration-structure host API
            .raytracing_from_render = false,
        };
    }

    fn deinitImpl(ptr: *anyopaque) void {
        const self: *WgpuBackend = @ptrCast(@alignCast(ptr));
        if (self.current_view) |v| {
            v.release();
            self.current_view = null;
        }
        if (self.current_surface_texture) |st| {
            if (st.texture) |t| t.release();
            self.current_surface_texture = null;
        }
        // Shared handles (surface/device/queue) are released by the host in zigote_shutdown.
    }
};
