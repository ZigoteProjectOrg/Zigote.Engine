pub const resource = @import("render_resource.zig");
pub const frame_ctx = @import("frame_context.zig");
pub const graph = @import("render_graph.zig");

pub const TransientPool = resource.TransientPool;
pub const TextureDesc = resource.TextureDesc;
pub const ResourceKind = resource.ResourceKind;
pub const FrameContext = frame_ctx.FrameContext;
pub const RenderGraph = graph.RenderGraph;
pub const RenderSettings = graph.RenderSettings;
pub const PassType = graph.PassType;

// Pull the submodules' `test` blocks into the engine test binary (see scene/root.zig for why
// referencing this namespace alone is not enough).
test {
    _ = resource;
    _ = frame_ctx;
    _ = graph;
}
