const render_resource = @import("render_resource.zig");

/// Per-frame state shared across all render passes.
pub const FrameContext = struct {
    frame_index: u32,
    surface_width: u32,
    surface_height: u32,
    dpi_scale: f32,
    delta_time: f32,

    /// Viewport within the surface used by the 3D scene (in logical pixels).
    /// Width/height of 0 means no 3D scene this frame.
    scene_viewport_x: f32 = 0,
    scene_viewport_y: f32 = 0,
    scene_viewport_w: f32 = 0,
    scene_viewport_h: f32 = 0,

    transient_pool: *render_resource.TransientPool,
};
