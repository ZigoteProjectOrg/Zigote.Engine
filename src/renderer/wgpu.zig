const std = @import("std");
const wgpu = @import("wgpu");
const zg = @import("../root.zig");
const ft_text = @import("freetype_text.zig");
const ui_shaders = @import("wgpu_ui_shaders.zig");
const ui_util = @import("wgpu_ui_util.zig");

// Leaf helpers live in wgpu_ui_util.zig; alias them so existing call sites are unchanged.
const ensureVertexBuffer = ui_util.ensureVertexBuffer;
const growBufferCapacity = ui_util.growBufferCapacity;
const releaseBuffer = ui_util.releaseBuffer;
const applyScissor = ui_util.applyScissor;
const intersectRects = ui_util.intersectRects;

// Color is stored as unorm8x4 (4 B, not 16 B): the source is an 8-bit-per-channel geometry.Color, so
// this is lossless, and the wgpu `unorm8x4` vertex format still delivers a normalized vec4<f32> to
// WGSL — the shaders are unchanged. Duplicated across all 6 corners of a quad, so shrinking it is
// 12 B/vertex of per-frame CPU fill + queue.writeBuffer saved on the hottest UI path.
pub const ShapeVertex = extern struct {
    position: [2]f32,
    color: [4]u8,
    local_pos: [2]f32,
    rect_size: [2]f32,
    radius: f32,
    border_width: f32,
    blur_radius: f32,
};

pub const LiquidGlassVertex = extern struct {
    position: [2]f32,
    color: [4]u8,
    local_pos: [2]f32,
    rect_size: [2]f32,
    radius: f32,
    thickness: f32,
    glow_pos: [2]f32,
    pinch_strength: f32,
    /// 0 = fully clear glass, 1 = full color tint. RGB comes from `color`.
    clear_tint: f32,
    /// Adaptive-luminance strength: <0 anchors the backdrop dark (light content),
    /// >0 anchors it light (dark content), 0 = off. Magnitude is strength.
    adapt: f32,
};

pub const ImageVertex = extern struct {
    position: [2]f32,
    uv: [2]f32,
    color: [4]u8,
};

pub const ShaderEffectVertex = extern struct {
    position: [2]f32, // NDC
    uv: [2]f32, // screen UV [0,1]
    params_a: [4]f32, // user params 0-3
    params_b: [4]f32, // user params 4-7
};

/// A registered custom shader effect. wants_image is decided at registration — the WGSL declares
/// @group(1) or it does not — and gates both the pipeline layout and the draw-time requirement
/// that a batch actually carries an image bind group.
pub const CustomShader = struct {
    pipeline: *wgpu.RenderPipeline,
    wants_image: bool,
};

pub const CachedImage = struct {
    texture: *wgpu.Texture,
    texture_view: *wgpu.TextureView,
    bind_group: *wgpu.BindGroup,
    /// Owned by an app-supplied handle (zigote_load_texture*, render textures): the app decides
    /// when it dies, so the cap below never evicts it and its CPU pixels can be dropped after
    /// upload. Unpinned entries are keyed by a hash of inline pixels that arrive again with every
    /// paint command, so re-uploading one is always possible.
    pinned: bool = false,
    /// The texture and the view belong to something else — a render texture, the 3D offscreen
    /// target — and this entry only points at them so a draw command can find them by key. Removing
    /// the entry must then release the bind group and stop there: the owner frees the rest, and
    /// freeing it here leaves the owner holding a dead handle.
    ///
    /// That is not hypothetical. Render-texture handles and image handles were counted separately
    /// and both started at 1, so they named the same cache slot; releasing image 1 freed render
    /// texture 1's GPU memory, and the next frame's blur pass died inside wgpu on a texture that no
    /// longer existed. The two spaces are one counter now (see `nextGpuHandle`) — this flag is what
    /// makes the ownership legible at the place where the sharing happens.
    borrowed: bool = false,
};

/// Device-pixel rect (scoped backdrop capture — see backdropCopyRegion).
pub const PixelRect = struct { x: u32, y: u32, w: u32, h: u32 };

pub const GpuUi = struct {
    shape_shader: *wgpu.ShaderModule,
    shape_pipeline_layout: *wgpu.PipelineLayout,
    shape_pipeline: *wgpu.RenderPipeline,
    // Rounded-clip slot ring: a dynamic-offset uniform bound on the shape (g0), text (g1) and
    // image (g1) pipelines. Buffer + bind group are (re)created on demand in ensureClipBuffer;
    // slot 0 is always zeroed (= no rounded clip). See the ClipUniformEntry block below.
    clip_bgl: *wgpu.BindGroupLayout,
    clip_buffer: ?*wgpu.Buffer = null,
    clip_buffer_capacity: usize = 0,
    clip_bind_group: ?*wgpu.BindGroup = null,
    liquid_glass_shader: *wgpu.ShaderModule,
    liquid_glass_pipeline_layout: *wgpu.PipelineLayout,
    liquid_glass_pipeline: *wgpu.RenderPipeline,
    image_shader: *wgpu.ShaderModule,
    image_pipeline_layout: *wgpu.PipelineLayout,
    image_pipeline: *wgpu.RenderPipeline,
    text: ft_text.FreeTypeTextRenderer,
    allocator: std.mem.Allocator,
    image_cache: std.AutoHashMap(u64, CachedImage),
    /// Keys whose GPU texture was created this frame from an app-owned handle. The FFI layer
    /// drains this at end-of-frame and frees the CPU-side pixel copy: once the texture exists,
    /// holding a second full RGBA buffer per image doubles the memory an image costs for nothing.
    uploaded_keys: std.ArrayListUnmanaged(u64) = .empty,
    scratch: std.heap.ArenaAllocator,
    /// Cap on *unpinned* (inline-pixel) images only — pinned entries are the app's to release.
    max_cached_images: usize = 192,
    unpinned_cached: usize = 0,
    shape_vertex_buffer: ?*wgpu.Buffer = null,
    shape_vertex_buffer_size: usize = 0,
    liquid_glass_vertex_buffer: ?*wgpu.Buffer = null,
    liquid_glass_vertex_buffer_size: usize = 0,
    text_vertex_buffer: ?*wgpu.Buffer = null,
    text_vertex_buffer_size: usize = 0,
    image_vertex_buffer: ?*wgpu.Buffer = null,
    image_vertex_buffer_size: usize = 0,
    // Backdrop capture for liquid glass refraction
    surface_format: wgpu.TextureFormat,
    // Transparent-window mode (CSD rounded corners): the frame clears to alpha 0 so pixels the
    // app leaves uncovered show the desktop through the premultiplied surface. Set once after
    // init by the FFI layer when the window really got an alpha channel.
    transparent_clear: bool = false,
    backdrop_bgl: *wgpu.BindGroupLayout,
    backdrop_sampler: *wgpu.Sampler,
    scene_texture: ?*wgpu.Texture = null,
    scene_texture_view: ?*wgpu.TextureView = null,
    backdrop_texture: ?*wgpu.Texture = null,
    backdrop_texture_view: ?*wgpu.TextureView = null,
    backdrop_bind_group: ?*wgpu.BindGroup = null,
    blit_bind_group: ?*wgpu.BindGroup = null,
    blit_vertex_buffer: ?*wgpu.Buffer = null,
    blit_vertex_buffer_size: usize = 0,
    scene_frame_width: u32 = 0,
    scene_frame_height: u32 = 0,
    // Grow-only allocation for the scene/backdrop pair during live resizes. The textures are
    // allocated at alloc dims (≥ frame dims, rounded up in AllocStep chunks while a drag grows the
    // window) and the frame renders into their top-left frame-sized region via a viewport — so a
    // resize drag reallocates the W×H×4 pair a handful of times instead of every frame. Once the
    // size holds still for SettleFrames frames, the pair is re-created at the exact size, restoring
    // the alloc == frame steady state every sampler is exact in.
    scene_alloc_width: u32 = 0,
    scene_alloc_height: u32 = 0,
    scene_stable_frames: u32 = 0,
    // The scene texture's pixels did not survive the last (re)allocation. Partial repaint replays
    // damage rects over the preserved scene, which is garbage right after a realloc — the frame
    // that observes this flag falls back to a full redraw and clears it. Size-change frames get a
    // full redraw from the host anyway; this exists for the settle realloc, which happens at a
    // STABLE size where the host has no reason to widen the damage itself.
    scene_contents_lost: bool = false,
    // UV extent currently written in the blit quad buffer (frame/alloc; 1,1 in steady state).
    blit_quad_u1: f32 = 1.0,
    blit_quad_v1: f32 = 1.0,
    custom_shader_pipelines: std.AutoHashMap(u32, CustomShader) = undefined,
    shader_effect_vertex_buffer: ?*wgpu.Buffer = null,
    shader_effect_vertex_buffer_size: usize = 0,
    // Shared quad index pattern for the text/image pipelines (both are pure quads) — see
    // ensureQuadIndexBuffer. Capacity is tracked in quads and only ever grows.
    quad_index_buffer: ?*wgpu.Buffer = null,
    quad_index_quads: usize = 0,
    // Scratch scene target for renderToTexture, persisted across calls and recreated only on a
    // size change — see ensureRtSceneTexture.
    rt_scene_texture: ?*wgpu.Texture = null,
    rt_scene_view: ?*wgpu.TextureView = null,
    rt_blit_bind_group: ?*wgpu.BindGroup = null,
    rt_scene_width: u32 = 0,
    rt_scene_height: u32 = 0,

    /// GPU memory held by this window's 2D renderer, in bytes, attributed per subsystem. Diagnostic
    /// only (no allocation). scene/backdrop are the (lazy) glass-backdrop capture targets — null on a
    /// flat UI, where the swapchain (owned by wgpu, not counted here) is the big surface-sized cost.
    pub const UiMem = struct {
        coverage_atlas: u64,
        emoji_atlas: u64,
        scene: u64,
        backdrop: u64,
        vertex_buffers: u64,
        image_count: u32,
        total: u64,
    };
    pub fn memoryBytes(self: *const GpuUi) UiMem {
        const atlas = self.text.atlasBytes();
        const surf_bpp: u64 = 4; // bgra8/rgba8 surface format
        const scene: u64 = if (self.scene_texture != null)
            @as(u64, self.scene_alloc_width) * self.scene_alloc_height * surf_bpp
        else
            0;
        const backdrop: u64 = if (self.backdrop_texture != null)
            @as(u64, self.scene_alloc_width) * self.scene_alloc_height * surf_bpp
        else
            0;
        const vbuf: u64 = self.shape_vertex_buffer_size + self.liquid_glass_vertex_buffer_size +
            self.text_vertex_buffer_size + self.image_vertex_buffer_size +
            self.blit_vertex_buffer_size + self.shader_effect_vertex_buffer_size +
            self.quad_index_quads * 6 * @sizeOf(u32);
        return .{
            .coverage_atlas = atlas.coverage,
            .emoji_atlas = atlas.emoji,
            .scene = scene,
            .backdrop = backdrop,
            .vertex_buffers = vbuf,
            .image_count = self.image_cache.count(),
            .total = atlas.coverage + atlas.emoji + scene + backdrop + vbuf,
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        device: *wgpu.Device,
        format: wgpu.TextureFormat,
        fonts: []const zg.FontAsset,
        default_font_family: []const u8,
    ) !GpuUi {
        // Rounded-clip BGL first — the shape/text/image pipeline layouts all reference it.
        const clip_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{
                .binding = 0,
                .visibility = wgpu.ShaderStages.fragment,
                .buffer = .{
                    .type = .uniform,
                    .has_dynamic_offset = @intFromBool(true),
                    .min_binding_size = @sizeOf(ClipUniformEntry),
                },
            },
        };
        const clip_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote rounded clip bind group layout"),
            .entry_count = clip_bgl_entries.len,
            .entries = &clip_bgl_entries,
        }) orelse return error.WgpuBindGroupLayoutUnavailable;
        errdefer clip_bgl.release();

        var shader_desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote shape shader",
            .code = ui_shaders.shape_shader_source,
        });

        const shape_shader = device.createShaderModule(&shader_desc) orelse {
            return error.WgpuShaderUnavailable;
        };
        errdefer shape_shader.release();

        const attributes = [_]wgpu.VertexAttribute{
            .{
                .format = .float32x2,
                .offset = @offsetOf(ShapeVertex, "position"),
                .shader_location = 0,
            },
            .{
                .format = .unorm8x4,
                .offset = @offsetOf(ShapeVertex, "color"),
                .shader_location = 1,
            },
            .{
                .format = .float32x2,
                .offset = @offsetOf(ShapeVertex, "local_pos"),
                .shader_location = 2,
            },
            .{
                .format = .float32x2,
                .offset = @offsetOf(ShapeVertex, "rect_size"),
                .shader_location = 3,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(ShapeVertex, "radius"),
                .shader_location = 4,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(ShapeVertex, "border_width"),
                .shader_location = 5,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(ShapeVertex, "blur_radius"),
                .shader_location = 6,
            },
        };

        const vertex_buffer_layout = wgpu.VertexBufferLayout{
            .array_stride = @sizeOf(ShapeVertex),
            .attribute_count = attributes.len,
            .attributes = &attributes,
        };

        const color_target = wgpu.ColorTargetState{
            .format = format,
            .blend = &wgpu.BlendState.alpha_blending,
        };

        const fragment = wgpu.FragmentState{
            .module = shape_shader,
            .entry_point = wgpu.StringView.fromSlice("fs_main"),
            .target_count = 1,
            .targets = @ptrCast(&color_target),
        };

        const shape_bgls = [_]*wgpu.BindGroupLayout{clip_bgl};
        const shape_pipeline_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote shape pipeline layout"),
            .bind_group_layout_count = shape_bgls.len,
            .bind_group_layouts = &shape_bgls,
        }) orelse return error.WgpuPipelineUnavailable;
        errdefer shape_pipeline_layout.release();

        const pipeline_desc = wgpu.RenderPipelineDescriptor{
            .label = wgpu.StringView.fromSlice("zigote shape pipeline"),
            .layout = shape_pipeline_layout,
            .vertex = .{
                .module = shape_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buffer_layout),
            },
            .primitive = .{},
            .multisample = .{},
            .fragment = &fragment,
        };

        const shape_pipeline = device.createRenderPipeline(&pipeline_desc) orelse {
            return error.WgpuPipelineUnavailable;
        };
        errdefer shape_pipeline.release();

        var shader_desc_lg = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote liquid glass shader",
            .code = ui_shaders.liquid_glass_shader_source,
        });

        const liquid_glass_shader = device.createShaderModule(&shader_desc_lg) orelse {
            return error.WgpuShaderUnavailable;
        };
        errdefer liquid_glass_shader.release();

        const attributes_lg = [_]wgpu.VertexAttribute{
            .{
                .format = .float32x2,
                .offset = @offsetOf(LiquidGlassVertex, "position"),
                .shader_location = 0,
            },
            .{
                .format = .unorm8x4,
                .offset = @offsetOf(LiquidGlassVertex, "color"),
                .shader_location = 1,
            },
            .{
                .format = .float32x2,
                .offset = @offsetOf(LiquidGlassVertex, "local_pos"),
                .shader_location = 2,
            },
            .{
                .format = .float32x2,
                .offset = @offsetOf(LiquidGlassVertex, "rect_size"),
                .shader_location = 3,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(LiquidGlassVertex, "radius"),
                .shader_location = 4,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(LiquidGlassVertex, "thickness"),
                .shader_location = 5,
            },
            .{
                .format = .float32x2,
                .offset = @offsetOf(LiquidGlassVertex, "glow_pos"),
                .shader_location = 6,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(LiquidGlassVertex, "pinch_strength"),
                .shader_location = 7,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(LiquidGlassVertex, "clear_tint"),
                .shader_location = 8,
            },
            .{
                .format = .float32,
                .offset = @offsetOf(LiquidGlassVertex, "adapt"),
                .shader_location = 9,
            },
        };

        const vertex_buffer_layout_lg = wgpu.VertexBufferLayout{
            .array_stride = @sizeOf(LiquidGlassVertex),
            .attribute_count = attributes_lg.len,
            .attributes = &attributes_lg,
        };

        // Bind group layout for the backdrop texture (texture + sampler, same as text BGL)
        const backdrop_bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{
                .binding = 0,
                .visibility = wgpu.ShaderStages.fragment,
                .texture = .{ .sample_type = .float, .view_dimension = .@"2d", .multisampled = 0 },
            },
            .{
                .binding = 1,
                .visibility = wgpu.ShaderStages.fragment,
                .sampler = .{ .type = .filtering },
            },
        };
        const backdrop_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote backdrop bind group layout"),
            .entry_count = backdrop_bgl_entries.len,
            .entries = &backdrop_bgl_entries,
        }) orelse return error.WgpuBindGroupLayoutUnavailable;
        errdefer backdrop_bgl.release();

        const backdrop_sampler = device.createSampler(&.{
            .label = wgpu.StringView.fromSlice("zigote backdrop sampler"),
            .address_mode_u = .clamp_to_edge,
            .address_mode_v = .clamp_to_edge,
            .address_mode_w = .clamp_to_edge,
            .mag_filter = .linear,
            .min_filter = .linear,
            .mipmap_filter = .nearest,
        }) orelse return error.WgpuSamplerUnavailable;
        errdefer backdrop_sampler.release();

        const lg_pipeline_layout = createTexturePipelineLayout(device, backdrop_bgl) orelse
            return error.WgpuPipelineUnavailable;
        errdefer lg_pipeline_layout.release();

        const fragment_lg = wgpu.FragmentState{
            .module = liquid_glass_shader,
            .entry_point = wgpu.StringView.fromSlice("fs_main"),
            .target_count = 1,
            .targets = @ptrCast(&color_target),
        };

        const pipeline_desc_lg = wgpu.RenderPipelineDescriptor{
            .label = wgpu.StringView.fromSlice("zigote liquid glass pipeline"),
            .layout = lg_pipeline_layout,
            .vertex = .{
                .module = liquid_glass_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vertex_buffer_layout_lg),
            },
            .primitive = .{},
            .multisample = .{},
            .fragment = &fragment_lg,
        };

        const liquid_glass_pipeline = device.createRenderPipeline(&pipeline_desc_lg) orelse {
            return error.WgpuPipelineUnavailable;
        };
        errdefer liquid_glass_pipeline.release();

        var image_shader_desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote image shader",
            .code = ui_shaders.image_shader_source,
        });

        const image_shader = device.createShaderModule(&image_shader_desc) orelse {
            return error.WgpuShaderUnavailable;
        };
        errdefer image_shader.release();

        const image_attributes = [_]wgpu.VertexAttribute{
            .{
                .format = .float32x2,
                .offset = @offsetOf(ImageVertex, "position"),
                .shader_location = 0,
            },
            .{
                .format = .float32x2,
                .offset = @offsetOf(ImageVertex, "uv"),
                .shader_location = 1,
            },
            .{
                .format = .unorm8x4,
                .offset = @offsetOf(ImageVertex, "color"),
                .shader_location = 2,
            },
        };

        const image_vertex_buffer_layout = wgpu.VertexBufferLayout{
            .array_stride = @sizeOf(ImageVertex),
            .attribute_count = image_attributes.len,
            .attributes = &image_attributes,
        };

        const image_color_target = wgpu.ColorTargetState{
            .format = format,
            .blend = &wgpu.BlendState.alpha_blending,
        };

        const image_fragment = wgpu.FragmentState{
            .module = image_shader,
            .entry_point = wgpu.StringView.fromSlice("fs_main"),
            .target_count = 1,
            .targets = @ptrCast(&image_color_target),
        };

        const text = try ft_text.FreeTypeTextRenderer.init(
            allocator,
            device,
            format,
            clip_bgl,
            fonts,
            default_font_family,
        );
        errdefer {
            var mutable_text = text;
            mutable_text.deinit();
        }

        const image_bgls = [_]*wgpu.BindGroupLayout{ text.bindGroupLayout(), clip_bgl };
        const image_pipeline_layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote image pipeline layout"),
            .bind_group_layout_count = image_bgls.len,
            .bind_group_layouts = &image_bgls,
        }) orelse return error.WgpuPipelineUnavailable;
        errdefer image_pipeline_layout.release();

        const image_pipeline = device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote image pipeline"),
            .vertex = .{
                .module = image_shader,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&image_vertex_buffer_layout),
            },
            .primitive = .{},
            .multisample = .{},
            .fragment = &image_fragment,
            .layout = image_pipeline_layout,
        }) orelse return error.WgpuPipelineUnavailable;
        errdefer image_pipeline.release();

        return .{
            .shape_shader = shape_shader,
            .shape_pipeline_layout = shape_pipeline_layout,
            .shape_pipeline = shape_pipeline,
            .clip_bgl = clip_bgl,
            .liquid_glass_shader = liquid_glass_shader,
            .liquid_glass_pipeline_layout = lg_pipeline_layout,
            .liquid_glass_pipeline = liquid_glass_pipeline,
            .surface_format = format,
            .backdrop_bgl = backdrop_bgl,
            .backdrop_sampler = backdrop_sampler,
            .image_shader = image_shader,
            .image_pipeline_layout = image_pipeline_layout,
            .image_pipeline = image_pipeline,
            .text = text,
            .allocator = allocator,
            .image_cache = std.AutoHashMap(u64, CachedImage).init(allocator),
            .scratch = std.heap.ArenaAllocator.init(allocator),
            .custom_shader_pipelines = std.AutoHashMap(u32, CustomShader).init(allocator),
        };
    }

    /// Overwrite a cached image's texels in place, keeping its texture, view and bind group.
    ///
    /// The upload path in `createImageBatch` is create-once: a cache hit returns the existing GPU
    /// resources and never looks at the pixels again, which is exactly right for a photo and exactly
    /// wrong for a video frame. Recreating the texture every frame would churn a 1080p allocation
    /// and a bind group at 60 Hz; this re-uses both and pays only for the texel copy.
    ///
    /// Render thread only, and only between frames: `queue.writeTexture` is ordered before the
    /// submits that follow it, so a caller draining at the top of a frame sees the new pixels in
    /// that same frame. Returns false when the key has no texture yet — the caller should leave the
    /// CPU copy in place and let the normal first-upload path take it.
    /// `pixels` is always the whole image; only rows [y0, y1) are uploaded, so a producer that
    /// knows its damage (a webview blinking a caret, a partially-updated surface) pays for the
    /// rows it changed instead of the whole texture. y0 = 0, y1 = height is the full-frame case.
    pub fn updateCachedImage(
        self: *GpuUi,
        queue: *wgpu.Queue,
        key: u64,
        pixels: []const u8,
        width: u32,
        height: u32,
        stride: u32,
        y0: u32,
        y1: u32,
    ) bool {
        const cached = self.image_cache.get(key) orelse return false;
        if (width == 0 or height == 0) return false;

        // The producer's own row stride, which is what makes the no-staging path reachable for a
        // padded surface: Cairo pads to 32 bytes, wgpu wants 256, and a 1280-px webview is already
        // both — so its rows go straight from the producer's memory to the GPU.
        const src_stride = if (stride != 0) @as(usize, stride) else @as(usize, width) * 4;
        if (src_stride < @as(usize, width) * 4) return false;
        if (pixels.len < src_stride * @as(usize, height)) return false;

        const top = @min(y0, height);
        const bottom = @min(y1, height);
        if (bottom <= top) return true; // nothing damaged — the texture already holds these rows
        const rows: u32 = bottom - top;
        const band = pixels[@as(usize, top) * src_stride ..][0 .. @as(usize, rows) * src_stride];

        // wgpu wants rows aligned to 256 bytes; a frame whose width already satisfies that (every
        // multiple of 64 px — 1280, 1920, 3840) uploads straight from the caller's buffer with no
        // staging copy at all.
        const bytes_per_row = std.mem.alignForward(usize, src_stride, 256);
        if (bytes_per_row == src_stride) {
            writeImageRows(queue, cached.texture, band, bytes_per_row, width, rows, top);
            return true;
        }

        // Not aligned: repack into a staging buffer whose rows are. `src_stride` may carry the
        // producer's padding, so copy width*4 out of each row and leave the rest zeroed.
        const row_bytes = @as(usize, width) * 4;

        const staging = self.allocator.alloc(u8, bytes_per_row * @as(usize, rows)) catch return false;
        defer self.allocator.free(staging);
        var row: usize = 0;
        while (row < rows) : (row += 1) {
            const src_start = row * src_stride;
            const dst = staging[row * bytes_per_row ..][0..bytes_per_row];
            @memcpy(dst[0..row_bytes], band[src_start..][0..row_bytes]);
            // Zero only the padding tail — the old full-buffer @memset cleared bytes the row
            // copies above immediately overwrote.
            @memset(dst[row_bytes..], 0);
        }
        writeImageRows(queue, cached.texture, staging, bytes_per_row, width, rows, top);
        return true;
    }

    /// Drop a cached image's GPU resources. Only valid between frames — mid-frame the texture may
    /// still be recorded in an open command encoder. Returns true if an entry was removed.
    pub fn releaseCachedImage(self: *GpuUi, key: u64) bool {
        const kv = self.image_cache.fetchRemove(key) orelse return false;
        if (!kv.value.pinned and self.unpinned_cached > 0) self.unpinned_cached -= 1;
        // The bind group is always ours — it is built here, per entry. The texture and view are only
        // ours when the entry is not borrowed; see CachedImage.borrowed.
        kv.value.bind_group.release();
        if (kv.value.borrowed) return true;
        kv.value.texture_view.release();
        kv.value.texture.release();
        return true;
    }

    pub fn registerShader(self: *GpuUi, device: *wgpu.Device, id: u32, wgsl: []const u8) !void {
        // The WGSL itself is the declaration: a shader that samples an app texture says
        // @group(1), and the pipeline layout must match what the module declares.
        const wants_image = std.mem.indexOf(u8, wgsl, "@group(1)") != null;
        const pipeline = try createCustomShaderPipeline(
            device,
            self.backdrop_bgl,
            if (wants_image) self.text.bindGroupLayout() else null,
            self.surface_format,
            wgsl,
        );
        const old = try self.custom_shader_pipelines.fetchPut(id, .{ .pipeline = pipeline, .wants_image = wants_image });
        if (old) |kv| kv.value.pipeline.release();
    }

    pub fn deinit(self: *GpuUi) void {
        releaseBuffer(self.shape_vertex_buffer);
        releaseBuffer(self.liquid_glass_vertex_buffer);
        releaseBuffer(self.text_vertex_buffer);
        releaseBuffer(self.image_vertex_buffer);
        releaseBuffer(self.blit_vertex_buffer);
        releaseBuffer(self.shader_effect_vertex_buffer);
        releaseBuffer(self.quad_index_buffer);
        if (self.rt_blit_bind_group) |bg| bg.release();
        if (self.rt_scene_view) |tv| tv.release();
        if (self.rt_scene_texture) |t| t.release();
        if (self.clip_bind_group) |bg| bg.release();
        releaseBuffer(self.clip_buffer);
        var shader_it = self.custom_shader_pipelines.valueIterator();
        while (shader_it.next()) |shader| shader.pipeline.release();
        self.custom_shader_pipelines.deinit();
        if (self.blit_bind_group) |bg| bg.release();
        if (self.backdrop_bind_group) |bg| bg.release();
        if (self.backdrop_texture_view) |tv| tv.release();
        if (self.backdrop_texture) |t| t.release();
        if (self.scene_texture_view) |tv| tv.release();
        if (self.scene_texture) |t| t.release();
        self.backdrop_sampler.release();
        self.backdrop_bgl.release();
        var it = self.image_cache.valueIterator();
        while (it.next()) |cached| {
            cached.bind_group.release();
            cached.texture_view.release();
            cached.texture.release();
        }
        self.image_cache.deinit();
        self.uploaded_keys.deinit(self.allocator);
        self.scratch.deinit();
        self.text.deinit();
        self.image_pipeline.release();
        self.image_pipeline_layout.release();
        self.image_shader.release();
        self.liquid_glass_pipeline.release();
        self.liquid_glass_pipeline_layout.release();
        self.liquid_glass_shader.release();
        self.shape_pipeline.release();
        self.shape_pipeline_layout.release();
        self.shape_shader.release();
        self.clip_bgl.release();
    }
};

// ── Rounded clip (per-batch SDF mask) ────────────────────────────────────────
// A rounded clip cannot ride the scissor rect, so a batch under one carries a byte offset into a
// per-frame uniform slot ring (one 256-byte slot per unique rounded clip; slot 0 = all zeros =
// disabled). The shape/text/image fragment shaders multiply coverage by the rounded-box SDF of the
// slot's rect — the same SDF the shape pipeline rasterises with. The scissor still applies the
// clip's bounding rect, so the SDF only ever refines corners the scissor already bounded.

/// wgpu's baseline min_uniform_buffer_offset_alignment — every dynamic offset must be a multiple.
const CLIP_SLOT: u32 = 256;

/// One uniform slot. Physical (framebuffer) pixels, matching @builtin(position) in the shaders.
/// Mirrors the WGSL `RoundedClip { rect: vec4<f32>, radius: vec4<f32> }` (32 bytes).
const ClipUniformEntry = extern struct {
    center: [2]f32,
    half: [2]f32,
    radius: f32,
    _pad: [3]f32 = .{ 0, 0, 0 },
};

pub const ShapeBatch = struct {
    vertex_offset: u32,
    vertex_count: u32,
    clip_rect: ?zg.Rect,
    clip_offset: u32 = 0,
};

pub const TextBatch = struct {
    vertex_offset: u32,
    vertex_count: u32,
    clip_rect: ?zg.Rect,
    clip_offset: u32 = 0,
};

pub const ImageBatch = struct {
    vertex_offset: u32,
    vertex_count: u32,
    clip_rect: ?zg.Rect,
    clip_offset: u32 = 0,
    texture: *wgpu.Texture,
    texture_view: *wgpu.TextureView,
    bind_group: *wgpu.BindGroup,
    owns_resources: bool = false,

    pub fn deinit(self: *ImageBatch) void {
        if (!self.owns_resources) return;
        self.bind_group.release();
        self.texture_view.release();
        self.texture.release();
    }
};

pub const ShaderEffectBatch = struct {
    shader_id: u32,
    vertex_offset: u32,
    vertex_count: u32,
    clip_rect: ?zg.Rect,
    /// Bound at @group(1) when the shader was registered wanting one (see CustomShader).
    image_bind_group: ?*wgpu.BindGroup = null,
    /// This effect is a filter in a chain: refresh the backdrop capture before it so it sees the
    /// previous effect's output. Off = share the capture with neighbours, which is what a lens
    /// (glass, backdrop blur) wants and what keeps a row of them from copying the frame each.
    chains_backdrop: bool = false,
    // Carried for uniformity but NOT applied: custom shader-effect pipelines have a fixed,
    // user-facing bind-group contract (g0 = backdrop), so a rounded clip degrades to the
    // bounding-rect scissor here. Same fallback for liquid glass (its ShapeBatch offset is
    // ignored by drawLiquidGlassOp).
    clip_offset: u32 = 0,
};

pub const DrawOp = union(enum) {
    shape: ShapeBatch,
    liquid_glass: ShapeBatch,
    text: TextBatch,
    image: ImageBatch,
    shader_effect: ShaderEffectBatch,
};

const FramePaint = struct {
    shape_vertices: std.ArrayList(ShapeVertex) = .empty,
    liquid_glass_vertices: std.ArrayList(LiquidGlassVertex) = .empty,
    text_vertices: std.ArrayList(ft_text.TextVertex) = .empty,
    image_vertices: std.ArrayList(ImageVertex) = .empty,
    shader_effect_vertices: std.ArrayList(ShaderEffectVertex) = .empty,
    // Unique rounded clips this frame; entry i lives at slot (i+1)·CLIP_SLOT (slot 0 = disabled).
    rounded_clips: std.ArrayList(ClipUniformEntry) = .empty,
    ops: std.ArrayList(DrawOp) = .empty,

    fn deinit(self: *FramePaint, allocator: std.mem.Allocator) void {
        for (self.ops.items) |*op| {
            switch (op.*) {
                .image => |*batch| batch.deinit(),
                else => {},
            }
        }
        self.ops.deinit(allocator);
        self.rounded_clips.deinit(allocator);
        self.shader_effect_vertices.deinit(allocator);
        self.image_vertices.deinit(allocator);
        self.text_vertices.deinit(allocator);
        self.liquid_glass_vertices.deinit(allocator);
        self.shape_vertices.deinit(allocator);
    }
};

const UploadedFrameVertices = struct {
    shape: ?*wgpu.Buffer = null,
    shape_bytes_len: usize = 0,
    liquid_glass: ?*wgpu.Buffer = null,
    liquid_glass_bytes_len: usize = 0,
    text: ?*wgpu.Buffer = null,
    text_bytes_len: usize = 0,
    image: ?*wgpu.Buffer = null,
    image_bytes_len: usize = 0,
    shader_effect: ?*wgpu.Buffer = null,
    shader_effect_bytes_len: usize = 0,
};

pub fn renderFrame(
    surface: *wgpu.Surface,
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    paint_list: zg.PaintList,
    // Optional overlay is appended after the main paint list and rendered using
    // the same ordered barrier system, so glass in the overlay sees everything
    // already painted by the main list.
    overlay_paint_list: ?zg.PaintList,
    frame_width: u32,
    frame_height: u32,
    scale_factor: f32,
    frame_index: u32,
    // Sub-rectangle partial-repaint regions (absolute logical px). When non-empty AND the frame has no
    // backdrop-sampling op (Liquid Glass / custom shader), the scene texture is preserved (loadOp=load)
    // and only these disjoint rects are redrawn. Empty = full clear + redraw (the default).
    damage: []const zg.Rect,
    // Whether this caller preserves the scene texture across frames (partial-repaint persistence) or
    // samples it as a Liquid-Glass backdrop. True for the main window. Secondary windows pass false —
    // they always full-redraw with empty damage and emit no glass, so their frame can go straight to
    // the swapchain, skipping the per-window offscreen scene texture + blit pass.
    persistent_scene: bool,
) !void {
    var surface_texture = wgpu.SurfaceTexture{
        .next_in_chain = null,
        .texture = null,
        .status = .@"error",
    };

    surface.getCurrentTexture(&surface_texture);

    switch (surface_texture.status) {
        .success_optimal, .success_suboptimal => {},
        .timeout, .outdated, .lost, .occluded => return,
        .@"error" => return error.WgpuSurfaceTextureUnavailable,
        else => return error.WgpuSurfaceTextureUnavailable,
    }

    const texture = surface_texture.texture orelse return error.WgpuSurfaceTextureUnavailable;
    defer texture.release();

    const view = texture.createView(null) orelse return error.WgpuTextureViewUnavailable;
    defer view.release();

    const scratch_allocator = gpu_ui.scratch.allocator();
    defer _ = gpu_ui.scratch.reset(.retain_capacity);

    var frame = FramePaint{};
    defer frame.deinit(scratch_allocator);

    // Planned BEFORE buildFrameOps: shader-effect UVs normalize over the alloc extents, so the ops
    // must be built against the dims ensureSceneTexture will materialize later this frame.
    const alloc = planSceneAlloc(gpu_ui, frame_width, frame_height);

    try buildFrameOps(
        device,
        queue,
        scratch_allocator,
        &frame,
        gpu_ui,
        paint_list,
        overlay_paint_list,
        @floatFromInt(frame_width),
        @floatFromInt(frame_height),
        @floatFromInt(alloc.w),
        @floatFromInt(alloc.h),
        scale_factor,
    );

    gpu_ui.text.uploadAtlasIfDirty(queue);
    gpu_ui.text.uploadColorAtlasIfDirty(queue);

    // The backdrop gate must run AFTER buildFrameOps has appended BOTH paint lists — glass
    // dialogs/popovers live in the overlay list, and evaluating the predicate earlier would
    // miss them. Apps that never emit a glass/shader-effect op never allocate the backdrop.
    const has_backdrop = frameHasBackdropOp(&frame);

    // Direct-to-swapchain fast path: when the caller never persists the scene across frames
    // (persistent_scene = false ⇒ no partial repaint) AND no op samples the scene as a backdrop, render
    // the frame's ops straight into the swapchain view and skip the offscreen scene texture (W×H×4 B)
    // plus the full-window blit pass entirely. The main window and any glass frame keep the
    // persistent-texture path (needed for next-frame loadOp=load persistence and backdrop refraction).
    const direct = !persistent_scene and !has_backdrop;

    var scene_view: *wgpu.TextureView = view;
    var scene_tex: ?*wgpu.Texture = null;
    if (!direct) {
        try ensureSceneTexture(gpu_ui, device, frame_width, frame_height, alloc);
        scene_view = gpu_ui.scene_texture_view orelse return error.WgpuTextureViewUnavailable;
        scene_tex = gpu_ui.scene_texture orelse return error.WgpuTextureUnavailable;
    }
    // Strictly after ensureSceneTexture: an allocation change releases + nulls the backdrop trio
    // there (scene and backdrop extents must match), so ensuring the backdrop first would validate
    // a stale-size trio that the scene recreation is about to null — and the first resized frame
    // with glass on screen would then fail with WgpuBackdropUnavailable. (has_backdrop ⇒ !direct,
    // so the scene branch above always ran.)
    if (has_backdrop) try ensureBackdropTexture(gpu_ui, device);
    const backdrop_tex = gpu_ui.backdrop_texture;
    const backdrop_bg = gpu_ui.backdrop_bind_group;

    const uploaded = try uploadFrameVertices(device, queue, gpu_ui, &frame);

    const encoder = device.createCommandEncoder(&.{
        .label = wgpu.StringView.fromSlice("zigote ui command encoder"),
    }) orelse return error.WgpuCommandEncoderUnavailable;
    defer encoder.release();

    // Static dark clear. This is only ever visible when the opaque-full-screen-root contract is
    // violated (or transiently while a live resize outruns relayout) — it used to be animated as a
    // debug tell, but the color cycling read as a rainbow shimmer around every window during
    // resizes. A fixed dark fill still exposes contract violations without flashing.
    _ = frame_index;
    // Transparent windows clear to alpha 0 — the rounded-corner cutouts the app's clip leaves
    // uncovered composite as see-through instead of the debug fill.
    const clear = if (gpu_ui.transparent_clear)
        wgpu.Color{ .r = 0, .g = 0, .b = 0, .a = 0 }
    else
        wgpu.Color{ .r = 0.08, .g = 0.10, .b = 0.16, .a = 1.0 };

    // Partial repaint: when C# supplied damage rects, preserve the persistent scene texture
    // (loadOp=load) and redraw only the damaged rects — including any backdrop-sampling op a rect
    // hits, which re-captures its refraction source mid-sweep (see replayFramePaintDamage).
    // Otherwise clear + redraw the whole frame.
    //
    // CORRECTNESS CONTRACT: this relies on the framework's mandated opaque, full-screen root background
    // (op[0]; see the "Root background must be opaque" rule in the C# ThemeData/UiApp docs). Scissored to a
    // damage rect, that opaque op re-fills the region before the semi-transparent ops composite over it, so
    // the loaded (already-composited) pixels are discarded, not double-blended — and the animated clear
    // colour it covers is never visible, so no seam appears against the preserved region. An app that
    // violates this contract (non-opaque / non-full-screen root) should turn off `render.partial_repaint`.
    // scene_contents_lost: the texture was just (re)allocated, so the "preserved" pixels damage
    // replay would composite over are garbage — the settle realloc is the case the host cannot
    // know to widen damage for. One full redraw resets the persistence baseline.
    if (damage.len > 0 and !gpu_ui.scene_contents_lost) {
        // Unreachable in direct mode: direct ⇒ !persistent_scene ⇒ the caller passes empty damage.
        try replayFramePaintDamage(
            encoder,
            scene_view,
            gpu_ui,
            &frame,
            uploaded,
            scene_tex,
            backdrop_tex,
            backdrop_bg,
            damage,
            scale_factor,
            frame_width,
            frame_height,
        );
    } else {
        try replayFramePaint(
            encoder,
            scene_view,
            scene_tex,
            backdrop_tex,
            backdrop_bg,
            gpu_ui,
            &frame,
            uploaded,
            clear,
            scale_factor,
            frame_width,
            frame_height,
        );
    }
    // Either path just repainted the full frame (or valid damage over intact contents) — the
    // persistence baseline is whole again.
    gpu_ui.scene_contents_lost = false;

    // Blit scene_texture -> swapchain. Skipped in direct mode — the ops were rendered straight into the
    // swapchain view above, so there is nothing to copy.
    //
    // Always a full clear + full-surface quad, deliberately: a damage-scissored blit with
    // loadOp=load was tried and REVERTED — WebGPU exposes no swapchain buffer age, the surface
    // rotates an unknown (platform-dependent, sometimes lazily recreated) set of images, and any
    // fixed "union the last N frames' damage" window mis-guesses N on some stack (observed on
    // Wayland/Mesa as stale/black regions outside the union). Do not reintroduce without a real
    // buffer-age API. Bandwidth savings live in the scene pass's damage scissor instead.
    if (!direct) {
        const attachment = wgpu.ColorAttachment{
            .view = view,
            .load_op = .clear,
            .store_op = .store,
            // Alpha 0 under transparent windows so the corner cutouts survive the blit pass.
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = if (gpu_ui.transparent_clear) 0.0 else 1.0 },
        };
        const pass_descriptor = wgpu.RenderPassDescriptor{
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&attachment),
        };
        const pass = encoder.beginRenderPass(&pass_descriptor) orelse return error.WgpuRenderPassUnavailable;
        defer pass.release();
        try blitSceneToSwapchain(device, queue, pass, gpu_ui, frame_width, frame_height);
        pass.end();
    }

    const command = encoder.finish(&.{
        .label = wgpu.StringView.fromSlice("zigote ui command buffer"),
    }) orelse return error.WgpuCommandBufferUnavailable;
    defer command.release();

    const submission = queue.submitForIndex(&.{command});

    if (surface.present() != .success) {
        return error.WgpuPresentFailed;
    }

    _ = device.poll(false, &submission);
}

/// Append paint ops for a main (and optional overlay) paint list, rebuilding the whole frame if the
/// glyph atlas grew or was reset mid-build: quads emitted before the change carry UVs normalized
/// against the old atlas extents and would draw garbled — permanently so on the partial-repaint
/// path, which never redraws this frame's damage rects again. A rebuild is cheap (the triggering
/// glyphs are already cached in the fresh atlas) and bounded: growth occurs at most twice
/// (1024→2048→4096), so the loop settles in ≤ 4 attempts even with an overflow reset.
fn buildFrameOps(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    scratch_allocator: std.mem.Allocator,
    frame: *FramePaint,
    gpu_ui: *GpuUi,
    paint_list: zg.PaintList,
    overlay_paint_list: ?zg.PaintList,
    frame_width: f32,
    frame_height: f32,
    // Extents of the backdrop texture shader-effect UVs normalize over — the scene ALLOCATION this
    // frame will render with (== frame dims except while resize headroom is live; always frame
    // dims for renderToTexture, whose targets are exact).
    backdrop_width: f32,
    backdrop_height: f32,
    scale_factor: f32,
) !void {
    var attempts: u32 = 0;
    while (true) {
        const gen_before = gpu_ui.text.atlas_generation;
        try appendPaintOps(
            device,
            queue,
            scratch_allocator,
            frame,
            &gpu_ui.text,
            gpu_ui,
            paint_list,
            frame_width,
            frame_height,
            backdrop_width,
            backdrop_height,
            scale_factor,
        );
        if (overlay_paint_list) |overlay| {
            if (overlay.commands.items.len > 0) {
                try appendPaintOps(
                    device,
                    queue,
                    scratch_allocator,
                    frame,
                    &gpu_ui.text,
                    gpu_ui,
                    overlay,
                    frame_width,
                    frame_height,
                    backdrop_width,
                    backdrop_height,
                    scale_factor,
                );
            }
        }
        attempts += 1;
        if (gpu_ui.text.atlas_generation == gen_before or attempts >= 4) return;
        frame.deinit(scratch_allocator);
        frame.* = FramePaint{};
    }
}

/// (Re)create the rounded-clip slot ring so every batch's clip_offset lands inside it, and upload
/// this frame's unique rounded clips. Slot 0 is written as all zeros exactly when the buffer is
/// (re)created — it never changes, and a zero radius disables the mask in the shaders. The bind
/// group only needs recreating alongside the buffer (dynamic offsets do the per-batch addressing).
fn ensureClipBuffer(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    rounded_clips: []const ClipUniformEntry,
) !void {
    const needed: usize = (rounded_clips.len + 1) * CLIP_SLOT;
    if (gpu_ui.clip_buffer == null or gpu_ui.clip_buffer_capacity < needed) {
        const next_size = growBufferCapacity(gpu_ui.clip_buffer_capacity, needed);
        const buffer = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("zigote rounded clip slots"),
            .usage = wgpu.BufferUsages.copy_dst | wgpu.BufferUsages.uniform,
            .size = @intCast(next_size),
        }) orelse return error.WgpuClipBufferUnavailable;

        releaseBuffer(gpu_ui.clip_buffer);
        gpu_ui.clip_buffer = buffer;
        gpu_ui.clip_buffer_capacity = next_size;

        const bg_entry = wgpu.BindGroupEntry{
            .binding = 0,
            .buffer = buffer,
            .offset = 0,
            .size = @sizeOf(ClipUniformEntry),
        };
        const bind_group = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("zigote rounded clip bind group"),
            .layout = gpu_ui.clip_bgl,
            .entry_count = 1,
            .entries = @ptrCast(&bg_entry),
        }) orelse return error.WgpuClipBindGroupUnavailable;
        if (gpu_ui.clip_bind_group) |old| old.release();
        gpu_ui.clip_bind_group = bind_group;

        const disabled = ClipUniformEntry{ .center = .{ 0, 0 }, .half = .{ 0, 0 }, .radius = 0 };
        queue.writeBuffer(buffer, 0, @ptrCast(&disabled), @sizeOf(ClipUniformEntry));
    }

    const buffer = gpu_ui.clip_buffer.?;
    for (rounded_clips, 0..) |entry, i| {
        queue.writeBuffer(buffer, (i + 1) * CLIP_SLOT, @ptrCast(&entry), @sizeOf(ClipUniformEntry));
    }
}

/// (Re)create the shared index buffer holding the repeating per-quad pattern {0,1,2, 1,3,2} for at
/// least `quad_count` quads. The text and image pipelines draw nothing but whole quads (4 vertices
/// each, corners TL,TR,BL,BR), so one persistent pattern buffer serves every batch via drawIndexed
/// and only regrows when a frame's quad count exceeds all previous frames.
fn ensureQuadIndexBuffer(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    quad_count: usize,
) !void {
    if (gpu_ui.quad_index_buffer != null and gpu_ui.quad_index_quads >= quad_count) return;
    const quads = @max(quad_count, @max(gpu_ui.quad_index_quads * 2, 256));
    const bytes = quads * 6 * @sizeOf(u32);
    const buffer = device.createBuffer(&.{
        .label = wgpu.StringView.fromSlice("zigote quad indices"),
        .usage = wgpu.BufferUsages.copy_dst | wgpu.BufferUsages.index,
        .size = @intCast(bytes),
    }) orelse return error.WgpuIndexBufferUnavailable;

    const indices = try gpu_ui.allocator.alloc(u32, quads * 6);
    defer gpu_ui.allocator.free(indices);
    for (0..quads) |q| {
        const v: u32 = @intCast(q * 4);
        indices[q * 6 + 0] = v;
        indices[q * 6 + 1] = v + 1;
        indices[q * 6 + 2] = v + 2;
        indices[q * 6 + 3] = v + 1;
        indices[q * 6 + 4] = v + 3;
        indices[q * 6 + 5] = v + 2;
    }
    queue.writeBuffer(buffer, 0, std.mem.sliceAsBytes(indices).ptr, bytes);

    releaseBuffer(gpu_ui.quad_index_buffer);
    gpu_ui.quad_index_buffer = buffer;
    gpu_ui.quad_index_quads = quads;
}

fn quadIndexBytes(gpu_ui: *const GpuUi) u64 {
    return @intCast(gpu_ui.quad_index_quads * 6 * @sizeOf(u32));
}

fn uploadFrameVertices(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    frame: *const FramePaint,
) !UploadedFrameVertices {
    var uploaded = UploadedFrameVertices{};

    // Always ensured (even with zero rounded clips) — every shape/text/image draw and the blit
    // bind the clip group, so the buffer + slot 0 must exist whenever anything renders.
    try ensureClipBuffer(device, queue, gpu_ui, frame.rounded_clips.items);

    // Quad index pattern for the text/image batches; the blit paths draw one quad, so at least
    // one entry is kept even on frames with no text or image content.
    const quad_count = @max(frame.text_vertices.items.len / 4, @max(frame.image_vertices.items.len / 4, 1));
    try ensureQuadIndexBuffer(device, queue, gpu_ui, quad_count);

    if (frame.shape_vertices.items.len > 0) {
        const bytes = std.mem.sliceAsBytes(frame.shape_vertices.items);
        const buffer = try ensureVertexBuffer(device, &gpu_ui.shape_vertex_buffer, &gpu_ui.shape_vertex_buffer_size, "zigote shape vertices", bytes.len);
        queue.writeBuffer(buffer, 0, bytes.ptr, bytes.len);
        uploaded.shape = buffer;
        uploaded.shape_bytes_len = bytes.len;
    }

    if (frame.liquid_glass_vertices.items.len > 0) {
        const bytes = std.mem.sliceAsBytes(frame.liquid_glass_vertices.items);
        const buffer = try ensureVertexBuffer(device, &gpu_ui.liquid_glass_vertex_buffer, &gpu_ui.liquid_glass_vertex_buffer_size, "zigote liquid glass vertices", bytes.len);
        queue.writeBuffer(buffer, 0, bytes.ptr, bytes.len);
        uploaded.liquid_glass = buffer;
        uploaded.liquid_glass_bytes_len = bytes.len;
    }

    if (frame.text_vertices.items.len > 0) {
        const bytes = std.mem.sliceAsBytes(frame.text_vertices.items);
        const buffer = try ensureVertexBuffer(device, &gpu_ui.text_vertex_buffer, &gpu_ui.text_vertex_buffer_size, "zigote text vertices", bytes.len);
        queue.writeBuffer(buffer, 0, bytes.ptr, bytes.len);
        uploaded.text = buffer;
        uploaded.text_bytes_len = bytes.len;
    }

    if (frame.image_vertices.items.len > 0) {
        const bytes = std.mem.sliceAsBytes(frame.image_vertices.items);
        const buffer = try ensureVertexBuffer(device, &gpu_ui.image_vertex_buffer, &gpu_ui.image_vertex_buffer_size, "zigote image vertices", bytes.len);
        queue.writeBuffer(buffer, 0, bytes.ptr, bytes.len);
        uploaded.image = buffer;
        uploaded.image_bytes_len = bytes.len;
    }

    if (frame.shader_effect_vertices.items.len > 0) {
        const bytes = std.mem.sliceAsBytes(frame.shader_effect_vertices.items);
        const buffer = try ensureVertexBuffer(device, &gpu_ui.shader_effect_vertex_buffer, &gpu_ui.shader_effect_vertex_buffer_size, "zigote shader effect vertices", bytes.len);
        queue.writeBuffer(buffer, 0, bytes.ptr, bytes.len);
        uploaded.shader_effect = buffer;
        uploaded.shader_effect_bytes_len = bytes.len;
    }

    return uploaded;
}

fn replayFramePaint(
    encoder: *wgpu.CommandEncoder,
    scene_view: *wgpu.TextureView,
    // Null only on the direct-to-swapchain path (scene_view IS the swapchain), which the caller takes
    // only when the frame has NO backdrop op — so the unwrap in the backdrop branch is unreachable
    // there. When a backdrop op exists this is the real offscreen scene texture. Fail soft either way.
    scene_tex: ?*wgpu.Texture,
    backdrop_tex: ?*wgpu.Texture,
    backdrop_bind_group: ?*wgpu.BindGroup,
    gpu_ui: *GpuUi,
    frame: *const FramePaint,
    uploaded: UploadedFrameVertices,
    clear: wgpu.Color,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
) !void {
    var pass = try beginScenePassClear(encoder, scene_view, clear, frame_width, frame_height);

    // One capture serves every consecutive backdrop-sampling op: the capture only goes stale when
    // a NON-backdrop op has drawn into the scene since. Glass deliberately never refracts sibling
    // glass — a blend group's overlapping shapes must read as one lens — so a backdrop op's own
    // output never staled the capture either way, and skipping the copy also skips a full-frame
    // texture copy plus a render-pass break per op, which is what made a row of glass capsules
    // drop frames during live resizes.
    var backdrop_fresh = false;
    for (frame.ops.items, 0..) |*op, op_idx| {
        if (drawOpNeedsBackdrop(op.*)) {
            if (!backdrop_fresh) {
                const bt = backdrop_tex orelse return error.WgpuBackdropUnavailable;
                const st = scene_tex orelse return error.WgpuTextureUnavailable;
                pass.end();
                pass.release();
                // Every op draws in order on this path, so the consecutive-run region is exact.
                const region = backdropCopyRegion(
                    frame,
                    op_idx,
                    @floatFromInt(frame_width),
                    @floatFromInt(frame_height),
                    frame_width,
                    frame_height,
                );
                copySceneToBackdrop(encoder, st, bt, frame_width, frame_height, region);
                pass = try beginScenePassLoad(encoder, scene_view, frame_width, frame_height);
                backdrop_fresh = true;
            }
        } else backdrop_fresh = false;

        try drawOp(
            pass,
            gpu_ui,
            op,
            uploaded,
            backdrop_bind_group,
            scale_factor,
            frame_width,
            frame_height,
            null, // full-frame path: no damage scissor
        );

        // A chained filter must see the previous pass's output, or a multi-pass pipeline
        // (tone -> LUT -> grain) silently collapses to whichever pass ran last. Only effects
        // that ask for it pay the refresh: Liquid Glass and backdrop blur share one capture on
        // purpose, and a copy per op is what made a row of glass capsules drop frames.
        if (op.* == .shader_effect and op.shader_effect.chains_backdrop) backdrop_fresh = false;
    }

    pass.end();
    pass.release();
}

/// Replays an already-built UI frame over the existing contents of scene_texture.
///
/// This is the path used by game/editor overlays: the caller first copies the
/// game viewport into scene_texture, then this function draws UI commands on top.
/// Liquid Glass and custom shader effects still work because every backdrop
/// command acts as a barrier: end pass -> copy scene_texture to backdrop_texture
/// -> resume pass and sample that backdrop.
fn replayFramePaintLoad(
    encoder: *wgpu.CommandEncoder,
    scene_view: *wgpu.TextureView,
    scene_tex: *wgpu.Texture,
    // Optional for the same reason as replayFramePaint — see there.
    backdrop_tex: ?*wgpu.Texture,
    backdrop_bind_group: ?*wgpu.BindGroup,
    gpu_ui: *GpuUi,
    frame: *const FramePaint,
    uploaded: UploadedFrameVertices,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
) !void {
    var pass = try beginScenePassLoad(encoder, scene_view, frame_width, frame_height);

    // Same consecutive-op capture sharing as replayFramePaint — see the comment there.
    var backdrop_fresh = false;
    for (frame.ops.items, 0..) |*op, op_idx| {
        if (drawOpNeedsBackdrop(op.*)) {
            if (!backdrop_fresh) {
                const bt = backdrop_tex orelse return error.WgpuBackdropUnavailable;
                pass.end();
                pass.release();
                const region = backdropCopyRegion(
                    frame,
                    op_idx,
                    @floatFromInt(frame_width),
                    @floatFromInt(frame_height),
                    frame_width,
                    frame_height,
                );
                copySceneToBackdrop(encoder, scene_tex, bt, frame_width, frame_height, region);
                pass = try beginScenePassLoad(encoder, scene_view, frame_width, frame_height);
                backdrop_fresh = true;
            }
        } else backdrop_fresh = false;

        try drawOp(
            pass,
            gpu_ui,
            op,
            uploaded,
            backdrop_bind_group,
            scale_factor,
            frame_width,
            frame_height,
            null, // full-frame path: no damage scissor
        );

        // A chained filter must see the previous pass's output, or a multi-pass pipeline
        // (tone -> LUT -> grain) silently collapses to whichever pass ran last. Only effects
        // that ask for it pay the refresh: Liquid Glass and backdrop blur share one capture on
        // purpose, and a copy per op is what made a row of glass capsules drop frames.
        if (op.* == .shader_effect and op.shader_effect.chains_backdrop) backdrop_fresh = false;
    }

    pass.end();
    pass.release();
}

/// True if any op samples the scene as a backdrop (Liquid Glass / custom shader effect). Gates the
/// lazy backdrop-texture allocation and the direct-to-swapchain fast path; partial repaint handles
/// these ops in-sweep (see replayFramePaintDamage), so it no longer forces a full redraw.
fn frameHasBackdropOp(frame: *const FramePaint) bool {
    for (frame.ops.items) |op| {
        if (drawOpNeedsBackdrop(op)) return true;
    }
    return false;
}

/// Device-pixel AABB of a contiguous NDC vertex range (any vertex type with an NDC `position`).
/// Computed from the final vertex data, so it is exact even under an active 2D transform.
fn vertexRangeAabb(comptime V: type, verts: []const V, fw: f32, fh: f32) ?zg.Rect {
    if (verts.len == 0) return null;
    var min_x = verts[0].position[0];
    var max_x = min_x;
    var min_y = verts[0].position[1];
    var max_y = min_y;
    for (verts[1..]) |v| {
        min_x = @min(min_x, v.position[0]);
        max_x = @max(max_x, v.position[0]);
        min_y = @min(min_y, v.position[1]);
        max_y = @max(max_y, v.position[1]);
    }
    // NDC y is up, device y is down, so max NDC y maps to the device-space top edge.
    return .{
        .x = (min_x + 1.0) * 0.5 * fw,
        .y = (1.0 - max_y) * 0.5 * fh,
        .width = (max_x - min_x) * 0.5 * fw,
        .height = (max_y - min_y) * 0.5 * fh,
    };
}

/// Device-space bounds of one draw op, from its vertex range in the per-kind list. Null = no
/// vertices (the op draws nothing anywhere).
fn opDeviceAabb(frame: *const FramePaint, op: *const DrawOp, fw: f32, fh: f32) ?zg.Rect {
    return switch (op.*) {
        .shape => |b| vertexRangeAabb(ShapeVertex, frame.shape_vertices.items[b.vertex_offset..][0..b.vertex_count], fw, fh),
        .liquid_glass => |b| vertexRangeAabb(LiquidGlassVertex, frame.liquid_glass_vertices.items[b.vertex_offset..][0..b.vertex_count], fw, fh),
        .text => |b| vertexRangeAabb(ft_text.TextVertex, frame.text_vertices.items[b.vertex_offset..][0..b.vertex_count], fw, fh),
        .image => |b| vertexRangeAabb(ImageVertex, frame.image_vertices.items[b.vertex_offset..][0..b.vertex_count], fw, fh),
        .shader_effect => |b| vertexRangeAabb(ShaderEffectVertex, frame.shader_effect_vertices.items[b.vertex_offset..][0..b.vertex_count], fw, fh),
    };
}

/// True when an op's device-space AABB can produce fragments inside the damage region's scissor
/// box. Mirrors applyScissor's clamping (a negative origin SHRINKS the box — shifting would make
/// adjacent damage sweeps overlap and double-blend), with 1 px of slack against its float→integer
/// truncation — a false positive just draws an op whose scissor discards everything, harmless.
fn aabbHitsDamage(aabb: zg.Rect, region: zg.Rect, scale_factor: f32) bool {
    const rx = @max(0.0, region.x * scale_factor);
    const ry = @max(0.0, region.y * scale_factor);
    const rw = @max(0.0, (region.width + @min(0.0, region.x)) * scale_factor);
    const rh = @max(0.0, (region.height + @min(0.0, region.y)) * scale_factor);
    return aabb.x + aabb.width >= rx - 1.0 and aabb.x <= rx + rw + 1.0 and
        aabb.y + aabb.height >= ry - 1.0 and aabb.y <= ry + rh + 1.0;
}

/// Partial-repaint replay: preserve the persistent scene texture (loadOp=load) and redraw the frame's
/// ops scissored to each damaged rect. Runs one op sweep per damage rect; because C# keeps the rects
/// pairwise non-overlapping, no pixel is drawn twice (which for alpha-blended ops would be wrong).
///
/// Backdrop-sampling ops (Liquid Glass / custom shaders) participate too. One whose AABB misses
/// every damage rect keeps its previous pixels — the scene texture persists, so skipping it is
/// exact. When any rect DOES hit one, the sweep degrades to a single merged region: the union of
/// all damage rects plus every hit backdrop op's full bounds (transitively, since the union can
/// reach further glass). Glass never re-renders a partial patch of itself — patches rendered
/// against backdrops from different frames disagree, and the seams read as smears of content that
/// is no longer there (observed as a stale color band on the player bar). Inside the merged region
/// the usual barrier applies (end pass → capture scene → resume), taken AFTER the ops under the
/// glass redrew, so the refraction source is current everywhere the glass can tap. A ticking label
/// inside a glass bar therefore costs a bar-sized redraw, not a full-window one.
fn damageUnion(a: zg.Rect, b: zg.Rect) zg.Rect {
    const x0 = @min(a.x, b.x);
    const y0 = @min(a.y, b.y);
    const x1 = @max(a.x + a.width, b.x + b.width);
    const y1 = @max(a.y + a.height, b.y + b.height);
    return .{ .x = x0, .y = y0, .width = x1 - x0, .height = y1 - y0 };
}
fn replayFramePaintDamage(
    encoder: *wgpu.CommandEncoder,
    scene_view: *wgpu.TextureView,
    gpu_ui: *GpuUi,
    frame: *const FramePaint,
    uploaded: UploadedFrameVertices,
    // Null only when the frame has no backdrop op — then no barrier ever runs and neither is
    // unwrapped.
    scene_tex: ?*wgpu.Texture,
    backdrop_tex: ?*wgpu.Texture,
    backdrop_bind_group: ?*wgpu.BindGroup,
    damage: []const zg.Rect,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
) !void {
    // One device-space AABB per op, so the per-rect sweeps below can skip ops (and their full
    // pipeline/bind-group/vertex-buffer setup) that cannot touch the rect being repainted.
    const fw: f32 = @floatFromInt(frame_width);
    const fh: f32 = @floatFromInt(frame_height);
    const aabbs = try gpu_ui.scratch.allocator().alloc(?zg.Rect, frame.ops.items.len);
    for (frame.ops.items, aabbs) |*op, *aabb| {
        aabb.* = opDeviceAabb(frame, op, fw, fh);
    }

    // Glass involvement check — see the function doc. If any damage rect hits a backdrop op,
    // collapse the sweep to ONE region: the union of every damage rect and every hit backdrop op's
    // full logical bounds, iterated to a fixpoint because the growing union can reach further
    // glass. One region also cannot overlap itself, so nothing double-blends.
    var merged: ?zg.Rect = null;
    {
        var grew = true;
        while (grew) {
            grew = false;
            for (frame.ops.items, aabbs) |*op, aabb| {
                if (!drawOpNeedsBackdrop(op.*)) continue;
                const bounds = aabb orelse continue;
                const hit = blk: {
                    if (merged) |m| break :blk aabbHitsDamage(bounds, m, scale_factor);
                    for (damage) |region| {
                        if (aabbHitsDamage(bounds, region, scale_factor)) break :blk true;
                    }
                    break :blk false;
                };
                if (!hit) continue;
                // Device-space AABB → logical, since damage regions are logical.
                const logical = zg.Rect{
                    .x = bounds.x / scale_factor,
                    .y = bounds.y / scale_factor,
                    .width = bounds.width / scale_factor,
                    .height = bounds.height / scale_factor,
                };
                var m = merged orelse blk: {
                    var acc = damage[0];
                    for (damage[1..]) |region| acc = damageUnion(acc, region);
                    break :blk acc;
                };
                const before = m;
                m = damageUnion(m, logical);
                if (merged == null or m.x != before.x or m.y != before.y or
                    m.width != before.width or m.height != before.height) grew = true;
                merged = m;
            }
        }
    }
    // The merged region's edges are synthetic — device-space op AABBs divided by the scale — so
    // unlike host damage rects they carry no inflation margin, and applyScissor's truncation of a
    // fractional device edge would leave the region's last row/column stale every sweep (a visible
    // hairline exactly on a glass bound under fractional display scales). One logical pixel of
    // outset covers the truncated row and the AA fringe both; being a single region, there is no
    // neighbouring sweep to double-blend with, and applyScissor's shrink-clamp keeps a negative
    // origin from shifting the box.
    // The sweep below is O(rects × ops) of AABB tests. Past a budget, one union region repaints
    // fewer total pixels than the CPU walk costs — collapse exactly like the glass path does
    // (one region cannot overlap itself, so nothing double-blends).
    if (merged == null and damage.len > 1 and
        damage.len * frame.ops.items.len > 32 * 1024)
    {
        var acc = damage[0];
        for (damage[1..]) |region| acc = damageUnion(acc, region);
        merged = acc;
    }

    if (merged) |*m| {
        m.x -= 1;
        m.y -= 1;
        m.width += 2;
        m.height += 2;
    }
    const sweep: []const zg.Rect = if (merged) |*m| m[0..1] else damage;

    var pass = try beginScenePassLoad(encoder, scene_view, frame_width, frame_height);
    defer {
        pass.end();
        pass.release();
    }

    // Same capture-freshness rule as replayFramePaint: one capture serves consecutive backdrop
    // ops; any other op drawn (in any region) stales it.
    var backdrop_fresh = false;

    // Cached: getenv is a linear environ scan and this runs on every partial-repaint frame.
    const debug_replay = blk: {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached == null) S.cached = std.c.getenv("ZIGOTE_DEBUG_REPLAY") != null;
        break :blk S.cached.?;
    };
    for (sweep) |region| {
        var drawn: u32 = 0;
        var culled: u32 = 0;
        for (frame.ops.items, aabbs, 0..) |*op, aabb, op_idx| {
            const bounds = aabb orelse continue;
            if (!aabbHitsDamage(bounds, region, scale_factor)) {
                culled += 1;
                if (debug_replay) std.log.info(
                    "[replay] CULL op#{d} {s} aabb=({d:.1},{d:.1} {d:.1}x{d:.1}) region=({d:.1},{d:.1} {d:.1}x{d:.1})",
                    .{ op_idx, @tagName(op.*), bounds.x, bounds.y, bounds.width, bounds.height, region.x, region.y, region.width, region.height },
                );
                continue;
            }
            if (drawOpNeedsBackdrop(op.*)) {
                if (!backdrop_fresh) {
                    const bt = backdrop_tex orelse return error.WgpuBackdropUnavailable;
                    const st = scene_tex orelse return error.WgpuTextureUnavailable;
                    pass.end();
                    pass.release();
                    // Full capture on purpose: this path CULLS ops, and a culled op does not
                    // stale the capture — so the op-order run scan backdropCopyRegion does is
                    // not sound here (the reuse run can span past a culled non-backdrop op).
                    copySceneToBackdrop(encoder, st, bt, frame_width, frame_height, null);
                    pass = try beginScenePassLoad(encoder, scene_view, frame_width, frame_height);
                    backdrop_fresh = true;
                }
            } else backdrop_fresh = false;
            drawn += 1;
            try drawOp(
                pass,
                gpu_ui,
                op,
                uploaded,
                backdrop_bind_group,
                scale_factor,
                frame_width,
                frame_height,
                region,
            );

            // Same rule as the full-frame path: an opted-in filter must see the previous
            // pass's output.
            if (op.* == .shader_effect and op.shader_effect.chains_backdrop) backdrop_fresh = false;
        }
        if (debug_replay) std.log.info(
            "[replay] region=({d:.1},{d:.1} {d:.1}x{d:.1}) sf={d:.1} fb={d}x{d} drawn={d} culled={d}",
            .{ region.x, region.y, region.width, region.height, scale_factor, frame_width, frame_height, drawn, culled },
        );
    }
}

fn copyTextureToScene(
    encoder: *wgpu.CommandEncoder,
    src_texture: *wgpu.Texture,
    dst_scene_texture: *wgpu.Texture,
    frame_width: u32,
    frame_height: u32,
) void {
    encoder.copyTextureToTexture(
        &.{ .texture = src_texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &.{ .texture = dst_scene_texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
        &.{ .width = frame_width, .height = frame_height, .depth_or_array_layers = 1 },
    );
}

fn beginScenePassClear(
    encoder: *wgpu.CommandEncoder,
    scene_view: *wgpu.TextureView,
    clear: wgpu.Color,
    frame_width: u32,
    frame_height: u32,
) !*wgpu.RenderPassEncoder {
    const attachment = wgpu.ColorAttachment{
        .view = scene_view,
        .load_op = .clear,
        .store_op = .store,
        .clear_value = clear,
    };
    const pass_descriptor = wgpu.RenderPassDescriptor{
        .color_attachment_count = 1,
        .color_attachments = @ptrCast(&attachment),
    };
    const pass = encoder.beginRenderPass(&pass_descriptor) orelse return error.WgpuRenderPassUnavailable;
    // The scene texture can be allocated larger than the frame (grow-only resize headroom); the
    // viewport maps NDC onto the frame-sized top-left region so vertex math stays alloc-agnostic.
    // Exactly the attachment when alloc == frame — the steady state — where this is a no-op.
    pass.setViewport(0, 0, @floatFromInt(frame_width), @floatFromInt(frame_height), 0, 1);
    return pass;
}

fn beginScenePassLoad(
    encoder: *wgpu.CommandEncoder,
    scene_view: *wgpu.TextureView,
    frame_width: u32,
    frame_height: u32,
) !*wgpu.RenderPassEncoder {
    const attachment = wgpu.ColorAttachment{
        .view = scene_view,
        .load_op = .load,
        .store_op = .store,
        .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
    };
    const pass_descriptor = wgpu.RenderPassDescriptor{
        .color_attachment_count = 1,
        .color_attachments = @ptrCast(&attachment),
    };
    const pass = encoder.beginRenderPass(&pass_descriptor) orelse return error.WgpuRenderPassUnavailable;
    // Same frame-region viewport as beginScenePassClear — see there.
    pass.setViewport(0, 0, @floatFromInt(frame_width), @floatFromInt(frame_height), 0, 1);
    return pass;
}

fn copySceneToBackdrop(
    encoder: *wgpu.CommandEncoder,
    scene_tex: *wgpu.Texture,
    backdrop_tex: *wgpu.Texture,
    frame_width: u32,
    frame_height: u32,
    region: ?PixelRect,
) void {
    // Scoped capture: a 120×40 glass chip on a 4K surface does not need a 33 MB full-frame copy.
    // The region (same origin in both textures) must cover every backdrop tap of the ops it
    // serves — see backdropCopyRegion for the reach bound; null = whole frame.
    const r = region orelse PixelRect{ .x = 0, .y = 0, .w = frame_width, .h = frame_height };
    if (r.w == 0 or r.h == 0) return;
    encoder.copyTextureToTexture(
        &.{ .texture = scene_tex, .mip_level = 0, .origin = .{ .x = r.x, .y = r.y, .z = 0 }, .aspect = .all },
        &.{ .texture = backdrop_tex, .mip_level = 0, .origin = .{ .x = r.x, .y = r.y, .z = 0 }, .aspect = .all },
        &.{ .width = r.w, .height = r.h, .depth_or_array_layers = 1 },
    );
}

/// Region a backdrop capture must cover for the consecutive backdrop-op run starting at
/// `ops[start]` — the run's device AABBs inflated by the glass shader's maximum sampling reach
/// (liquid_glass_shader_source.wgsl: |disp| ≤ (h + t·BEND)/0.25 with h ≤ t ⇒ 28·t; frost
/// ≤ (4 + 2.1·t)·(1 + FROST_RIM); +16 px of AA/fringe slack). Null = copy the whole frame: a
/// custom shader effect samples the backdrop arbitrarily, the bounds are unknown, or the union
/// is nearly the frame anyway. ONLY sound on paths that draw every op in order — a path that
/// culls ops must pass null (a culled op does not stale the capture, so the reuse run can span
/// past this op-order scan).
fn backdropCopyRegion(
    frame: *const FramePaint,
    start: usize,
    fw: f32,
    fh: f32,
    frame_width: u32,
    frame_height: u32,
) ?PixelRect {
    var acc: ?zg.Rect = null;
    var i = start;
    while (i < frame.ops.items.len) : (i += 1) {
        const op = &frame.ops.items[i];
        if (!drawOpNeedsBackdrop(op.*)) break;
        const b = switch (op.*) {
            .liquid_glass => |batch| batch,
            else => return null, // shader_effect: arbitrary sampling — full capture
        };
        const verts = frame.liquid_glass_vertices.items[b.vertex_offset..][0..b.vertex_count];
        const aabb = vertexRangeAabb(LiquidGlassVertex, verts, fw, fh) orelse return null;
        var t_max: f32 = 1.0;
        for (verts) |v| t_max = @max(t_max, v.thickness);
        const reach = 28.0 * t_max + ((4.0 + 2.1 * t_max) * 2.8) + 16.0;
        const r = zg.Rect{
            .x = aabb.x - reach,
            .y = aabb.y - reach,
            .width = aabb.width + 2.0 * reach,
            .height = aabb.height + 2.0 * reach,
        };
        acc = if (acc) |a| damageUnion(a, r) else r;
    }

    const a = acc orelse return null;
    const fx0 = @max(0.0, a.x);
    const fy0 = @max(0.0, a.y);
    const fx1 = @min(@as(f32, @floatFromInt(frame_width)), a.x + a.width);
    const fy1 = @min(@as(f32, @floatFromInt(frame_height)), a.y + a.height);
    if (fx1 <= fx0 or fy1 <= fy0) return null;
    var px = PixelRect{
        .x = @intFromFloat(@floor(fx0)),
        .y = @intFromFloat(@floor(fy0)),
        .w = @intFromFloat(@ceil(fx1 - fx0)),
        .h = @intFromFloat(@ceil(fy1 - fy0)),
    };
    px.w = @min(px.w, frame_width - px.x);
    px.h = @min(px.h, frame_height - px.y);
    // Nearly the whole frame: the full copy is simpler and the saving is noise.
    if (@as(u64, px.w) * px.h * 10 >= @as(u64, frame_width) * frame_height * 9) return null;
    return px;
}

fn drawOpNeedsBackdrop(op: DrawOp) bool {
    return switch (op) {
        .liquid_glass, .shader_effect => true,
        else => false,
    };
}

fn drawOp(
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    op: *const DrawOp,
    uploaded: UploadedFrameVertices,
    backdrop_bind_group: ?*wgpu.BindGroup,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
    // Partial-repaint damage region this op is scissored to; null on the full-frame path.
    damage_clip: ?zg.Rect,
) !void {
    switch (op.*) {
        .shape => |batch| try drawShapeOp(pass, gpu_ui, uploaded, batch, scale_factor, frame_width, frame_height, damage_clip),
        .liquid_glass => |batch| try drawLiquidGlassOp(pass, gpu_ui, uploaded, batch, backdrop_bind_group, scale_factor, frame_width, frame_height, damage_clip),
        .text => |batch| try drawTextOp(pass, gpu_ui, uploaded, batch, scale_factor, frame_width, frame_height, damage_clip),
        .image => |batch| try drawImageOp(pass, gpu_ui, uploaded, batch, scale_factor, frame_width, frame_height, damage_clip),
        .shader_effect => |batch| try drawShaderEffectOp(pass, gpu_ui, uploaded, batch, backdrop_bind_group, scale_factor, frame_width, frame_height, damage_clip),
    }
}

fn drawShapeOp(
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    uploaded: UploadedFrameVertices,
    batch: ShapeBatch,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
    damage_clip: ?zg.Rect,
) !void {
    if (batch.vertex_count == 0) return;
    const buffer = uploaded.shape orelse return;
    // Guaranteed by ensureClipBuffer in uploadFrameVertices; skip rather than bind null.
    const clip_bg = gpu_ui.clip_bind_group orelse return;
    if (!applyScissor(pass, batch.clip_rect, damage_clip, scale_factor, frame_width, frame_height)) return;
    pass.setPipeline(gpu_ui.shape_pipeline);
    pass.setBindGroup(0, clip_bg, 1, @ptrCast(&batch.clip_offset));
    pass.setVertexBuffer(0, buffer, 0, uploaded.shape_bytes_len);
    pass.draw(batch.vertex_count, 1, batch.vertex_offset, 0);
}

fn drawLiquidGlassOp(
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    uploaded: UploadedFrameVertices,
    batch: ShapeBatch,
    backdrop_bind_group: ?*wgpu.BindGroup,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
    damage_clip: ?zg.Rect,
) !void {
    if (batch.vertex_count == 0) return;
    const buffer = uploaded.liquid_glass orelse return;
    // Unreachable by construction (backdrop creation is gated on frameHasBackdropOp): skip the op
    // rather than binding null.
    const backdrop_bg = backdrop_bind_group orelse return;
    if (!applyScissor(pass, batch.clip_rect, damage_clip, scale_factor, frame_width, frame_height)) return;
    pass.setPipeline(gpu_ui.liquid_glass_pipeline);
    pass.setBindGroup(0, backdrop_bg, 0, null);
    pass.setVertexBuffer(0, buffer, 0, uploaded.liquid_glass_bytes_len);
    pass.draw(batch.vertex_count, 1, batch.vertex_offset, 0);
}

fn drawTextOp(
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    uploaded: UploadedFrameVertices,
    batch: TextBatch,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
    damage_clip: ?zg.Rect,
) !void {
    if (batch.vertex_count == 0) return;
    const buffer = uploaded.text orelse return;
    const clip_bg = gpu_ui.clip_bind_group orelse return;
    const index_buffer = gpu_ui.quad_index_buffer orelse return;
    if (!applyScissor(pass, batch.clip_rect, damage_clip, scale_factor, frame_width, frame_height)) return;
    pass.setPipeline(gpu_ui.text.pipeline);
    pass.setBindGroup(0, gpu_ui.text.bind_group, 0, null);
    pass.setBindGroup(1, clip_bg, 1, @ptrCast(&batch.clip_offset));
    pass.setVertexBuffer(0, buffer, 0, uploaded.text_bytes_len);
    pass.setIndexBuffer(index_buffer, .uint32, 0, quadIndexBytes(gpu_ui));
    pass.drawIndexed(batch.vertex_count / 4 * 6, 1, batch.vertex_offset / 4 * 6, 0, 0);
}

fn drawImageOp(
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    uploaded: UploadedFrameVertices,
    batch: ImageBatch,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
    damage_clip: ?zg.Rect,
) !void {
    if (batch.vertex_count == 0) return;
    const buffer = uploaded.image orelse return;
    const clip_bg = gpu_ui.clip_bind_group orelse return;
    const index_buffer = gpu_ui.quad_index_buffer orelse return;
    if (!applyScissor(pass, batch.clip_rect, damage_clip, scale_factor, frame_width, frame_height)) return;
    pass.setPipeline(gpu_ui.image_pipeline);
    pass.setBindGroup(0, batch.bind_group, 0, null);
    pass.setBindGroup(1, clip_bg, 1, @ptrCast(&batch.clip_offset));
    pass.setVertexBuffer(0, buffer, 0, uploaded.image_bytes_len);
    pass.setIndexBuffer(index_buffer, .uint32, 0, quadIndexBytes(gpu_ui));
    pass.drawIndexed(batch.vertex_count / 4 * 6, 1, batch.vertex_offset / 4 * 6, 0, 0);
}

fn drawShaderEffectOp(
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    uploaded: UploadedFrameVertices,
    batch: ShaderEffectBatch,
    backdrop_bind_group: ?*wgpu.BindGroup,
    scale_factor: f32,
    frame_width: u32,
    frame_height: u32,
    damage_clip: ?zg.Rect,
) !void {
    if (batch.vertex_count == 0) return;
    const buffer = uploaded.shader_effect orelse return;
    const shader = gpu_ui.custom_shader_pipelines.get(batch.shader_id) orelse return;
    // A pipeline whose layout declares @group(1) cannot legally draw without it bound — a
    // command that failed to resolve its texture is skipped, not drawn broken.
    if (shader.wants_image and batch.image_bind_group == null) return;
    // Unreachable by construction — see drawLiquidGlassOp.
    const backdrop_bg = backdrop_bind_group orelse return;
    if (!applyScissor(pass, batch.clip_rect, damage_clip, scale_factor, frame_width, frame_height)) return;
    pass.setPipeline(shader.pipeline);
    pass.setBindGroup(0, backdrop_bg, 0, null);
    if (shader.wants_image) pass.setBindGroup(1, batch.image_bind_group.?, 0, null);
    pass.setVertexBuffer(0, buffer, 0, uploaded.shader_effect_bytes_len);
    pass.draw(batch.vertex_count, 1, batch.vertex_offset, 0);
}

/// Row-major 2×3 affine: x' = a·x + c·y + tx; y' = b·x + d·y + ty.
const Affine2 = struct {
    a: f32 = 1,
    b: f32 = 0,
    c: f32 = 0,
    d: f32 = 1,
    tx: f32 = 0,
    ty: f32 = 0,

    /// (outer ∘ inner)(p) = outer(inner(p)) — inner applies first.
    fn compose(outer: Affine2, inner: Affine2) Affine2 {
        return .{
            .a = outer.a * inner.a + outer.c * inner.b,
            .b = outer.b * inner.a + outer.d * inner.b,
            .c = outer.a * inner.c + outer.c * inner.d,
            .d = outer.b * inner.c + outer.d * inner.d,
            .tx = outer.a * inner.tx + outer.c * inner.ty + outer.tx,
            .ty = outer.b * inner.tx + outer.d * inner.ty + outer.ty,
        };
    }

    fn apply(self: Affine2, x: f32, y: f32) [2]f32 {
        return .{ self.a * x + self.c * y + self.tx, self.b * x + self.d * y + self.ty };
    }
};

/// Re-express a logical-pixel affine in NDC space so already-emitted NDC vertex positions can be
/// transformed in place (ndc = N(s·p), N(x,y) = (2x/W−1, 1−2y/H); this is N∘S∘M∘S⁻¹∘N⁻¹ expanded).
/// Written term-by-term so an identity `m` yields the EXACT identity affine — every extra term is
/// a product with 0 — keeping identity-transform frames byte-identical to untransformed ones.
fn ndcAffine(m: Affine2, scale: f32, fw: f32, fh: f32) Affine2 {
    const h_over_w = fh / fw;
    const w_over_h = fw / fh;
    return .{
        .a = m.a,
        .c = -(m.c * h_over_w),
        .tx = (m.a - 1.0) + m.c * h_over_w + (2.0 * scale * m.tx) / fw,
        .b = -(m.b * w_over_h),
        .d = m.d,
        .ty = (1.0 - m.d) - m.b * w_over_h - (2.0 * scale * m.ty) / fh,
    };
}

/// Conservative screen-space AABB of a rect mapped through an affine. Rotation makes the true
/// clip region non-axis-aligned and the scissor is an axis-aligned rect, so clip under a
/// transform is the bounding box (never clips content away that should be visible).
fn transformRectAabb(m: Affine2, r: zg.Rect) zg.Rect {
    const p0 = m.apply(r.x, r.y);
    const p1 = m.apply(r.x + r.width, r.y);
    const p2 = m.apply(r.x, r.y + r.height);
    const p3 = m.apply(r.x + r.width, r.y + r.height);
    const min_x = @min(@min(p0[0], p1[0]), @min(p2[0], p3[0]));
    const max_x = @max(@max(p0[0], p1[0]), @max(p2[0], p3[0]));
    const min_y = @min(@min(p0[1], p1[1]), @min(p2[1], p3[1]));
    const max_y = @max(@max(p0[1], p1[1]), @max(p2[1], p3[1]));
    return .{ .x = min_x, .y = min_y, .width = max_x - min_x, .height = max_y - min_y };
}

/// Transform the NDC positions of a just-emitted vertex range in place. Only `position` moves:
/// local_pos / rect_size stay in the shape's local space (the SDF still evaluates there — affine
/// interpolation keeps rounded corners and stroke coverage correct under rotation/scale, with AA
/// feather widths in pre-transform units).
fn applyNdcTransform(comptime V: type, verts: []V, m: Affine2) void {
    for (verts) |*v| {
        v.position = m.apply(v.position[0], v.position[1]);
    }
}

/// Shader-effect vertices also carry the screen UV the backdrop is sampled at — recompute it from
/// the transformed NDC position so the effect keeps sampling the pixels it covers.
fn applyNdcTransformFx(verts: []ShaderEffectVertex, m: Affine2) void {
    for (verts) |*v| {
        const p = m.apply(v.position[0], v.position[1]);
        v.position = p;
        v.uv = .{ (p[0] + 1.0) * 0.5, (1.0 - p[1]) * 0.5 };
    }
}

/// Byte offset of the uniform slot for this rounded clip, adding a slot when the frame hasn't
/// seen the value yet (deduped by exact value — a widget subtree under one ClipRRect produces one
/// slot). Entries are pre-scaled to physical pixels; the shaders test @builtin(position) directly.
fn roundedClipSlot(
    allocator: std.mem.Allocator,
    frame: *FramePaint,
    rect: zg.Rect,
    radius: f32,
    scale_factor: f32,
) !u32 {
    const entry = ClipUniformEntry{
        .center = .{
            (rect.x + rect.width * 0.5) * scale_factor,
            (rect.y + rect.height * 0.5) * scale_factor,
        },
        .half = .{ rect.width * 0.5 * scale_factor, rect.height * 0.5 * scale_factor },
        .radius = radius * scale_factor,
    };
    for (frame.rounded_clips.items, 0..) |existing, i| {
        if (std.meta.eql(existing, entry)) return @intCast((i + 1) * CLIP_SLOT);
    }
    try frame.rounded_clips.append(allocator, entry);
    return @intCast(frame.rounded_clips.items.len * CLIP_SLOT);
}

fn appendPaintOps(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    allocator: std.mem.Allocator,
    frame: *FramePaint,
    text_renderer: *ft_text.FreeTypeTextRenderer,
    gpu_ui: *GpuUi,
    paint_list: zg.PaintList,
    frame_width: f32,
    frame_height: f32,
    backdrop_width: f32,
    backdrop_height: f32,
    scale_factor: f32,
) !void {
    // Scissor rect (intersection of the whole stack) + the active rounded-clip slot offset.
    // The offset is NOT intersected — nested rounded clips keep the innermost rounded rect and
    // rely on the scissor for the outer bounds.
    const ClipState = struct { rect: zg.Rect, offset: u32 };
    var clip_stack: std.ArrayList(ClipState) = .empty;
    defer clip_stack.deinit(allocator);

    var current_clip: ?zg.Rect = null;
    var current_clip_offset: u32 = 0;

    // 2-D transform stack: each entry is the fully-composed logical-space affine plus the same
    // transform re-expressed in NDC. Vertices are emitted untransformed by the per-command
    // tessellators, then the active NDC affine is applied to the ranges each command appended —
    // one interception point that covers every command kind (shape/text/image/glass/effect).
    const XfState = struct { logical: Affine2, ndc: Affine2 };
    var xf_stack: std.ArrayList(XfState) = .empty;
    defer xf_stack.deinit(allocator);
    var xf: ?XfState = null;

    for (paint_list.commands.items) |command| {
        const shape_mark = frame.shape_vertices.items.len;
        const text_mark = frame.text_vertices.items.len;
        const image_mark = frame.image_vertices.items.len;
        const glass_mark = frame.liquid_glass_vertices.items.len;
        const fx_mark = frame.shader_effect_vertices.items.len;

        switch (command) {
            .transform_push => |t| {
                const m = Affine2{ .a = t.a, .b = t.b, .c = t.c, .d = t.d, .tx = t.tx, .ty = t.ty };
                const logical = if (xf) |cur| Affine2.compose(cur.logical, m) else m;
                const state = XfState{
                    .logical = logical,
                    .ndc = ndcAffine(logical, scale_factor, frame_width, frame_height),
                };
                try xf_stack.append(allocator, state);
                xf = state;
            },

            .transform_pop => {
                if (xf_stack.items.len > 0) _ = xf_stack.pop();
                xf = if (xf_stack.items.len > 0) xf_stack.items[xf_stack.items.len - 1] else null;
            },

            .clip_start => |clip| {
                // Under an active transform the clip rect (authored in pre-transform space) maps
                // to its conservative screen AABB — the scissor rect cannot express rotation.
                const screen_rect = if (xf) |cur| transformRectAabb(cur.logical, clip.rect) else clip.rect;
                const new_clip = if (current_clip) |prev| intersectRects(prev, screen_rect) else screen_rect;
                // A rounded clip additionally claims a uniform slot the fragment shaders mask
                // with. Nested rounded clips keep the innermost radius. Under a non-translation
                // transform the axis-aligned SDF rect cannot express the mapped shape, so the
                // radius conservatively degrades to the AABB scissor (matches the scissor rule
                // above: never clip away content that should be visible).
                var offset = current_clip_offset;
                if (clip.radius > 0) {
                    const translation_only = if (xf) |cur|
                        cur.logical.a == 1 and cur.logical.b == 0 and cur.logical.c == 0 and cur.logical.d == 1
                    else
                        true;
                    if (translation_only)
                        offset = try roundedClipSlot(allocator, frame, screen_rect, clip.radius, scale_factor);
                }
                try clip_stack.append(allocator, .{ .rect = new_clip, .offset = offset });
                current_clip = new_clip;
                current_clip_offset = offset;
            },

            .clip_end => {
                if (clip_stack.items.len > 0) _ = clip_stack.pop();
                if (clip_stack.items.len > 0) {
                    const top = clip_stack.items[clip_stack.items.len - 1];
                    current_clip = top.rect;
                    current_clip_offset = top.offset;
                } else {
                    current_clip = null;
                    current_clip_offset = 0;
                }
            },

            .push_opacity => |po| {
                _ = po;
                // TODO: make opacity an ordered offscreen subpass. Keeping this
                // ignored is safer than pretending it works and breaking z-order.
            },

            .pop_opacity => {},

            .liquid_glass => |lg| {
                const start_len = frame.liquid_glass_vertices.items.len;
                try appendLiquidGlass(
                    allocator,
                    &frame.liquid_glass_vertices,
                    .{
                        .x = lg.bounds.x * scale_factor,
                        .y = lg.bounds.y * scale_factor,
                        .width = lg.bounds.width * scale_factor,
                        .height = lg.bounds.height * scale_factor,
                    },
                    lg.color,
                    lg.radius * scale_factor,
                    lg.thickness * scale_factor,
                    lg.glow_x * scale_factor,
                    lg.glow_y * scale_factor,
                    lg.pinch,
                    liquidGlassClearTint(lg, lg.color),
                    lg.adapt,
                    frame_width,
                    frame_height,
                );
                const added = frame.liquid_glass_vertices.items.len - start_len;
                try appendOrMergeLiquidGlassOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });
            },

            .shader_effect => |se| shader_effect: {
                // Resolve the optional @group(1) texture through the same cache the image path
                // uses — createImageBatch caches an app-owned key as pinned, so a texture only
                // ever referenced by shader effects (a LUT) still gets its GPU entry built here
                // on first use. A key that cannot be materialized skips the op: drawing it would
                // require a bind group that does not exist.
                var image_bg: ?*wgpu.BindGroup = null;
                if (se.image_key) |key| {
                    const resolved = createImageBatch(device, queue, allocator, .{
                        .bounds = .{ .x = 0, .y = 0, .width = 0, .height = 0 },
                        .width = se.image_width,
                        .height = se.image_height,
                        .pixels = se.image_pixels,
                        .cache_key = key,
                    }, gpu_ui, null, 0) catch break :shader_effect;
                    image_bg = resolved.bind_group;
                }

                const start_len = frame.shader_effect_vertices.items.len;
                try appendShaderEffect(
                    allocator,
                    &frame.shader_effect_vertices,
                    .{
                        .x = se.bounds.x * scale_factor,
                        .y = se.bounds.y * scale_factor,
                        .width = se.bounds.width * scale_factor,
                        .height = se.bounds.height * scale_factor,
                    },
                    se.params,
                    frame_width,
                    frame_height,
                    backdrop_width,
                    backdrop_height,
                );
                const added = frame.shader_effect_vertices.items.len - start_len;
                try appendOrMergeShaderEffectOp(allocator, frame, .{
                    .shader_id = se.shader_id,
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                    .image_bind_group = image_bg,
                    .chains_backdrop = se.chains_backdrop,
                });
            },

            .bezier => |bz| {
                const start_len = frame.shape_vertices.items.len;
                try appendBezierStroke(
                    allocator,
                    &frame.shape_vertices,
                    bz.x0 * scale_factor,
                    bz.y0 * scale_factor,
                    bz.x1 * scale_factor,
                    bz.y1 * scale_factor,
                    bz.x2 * scale_factor,
                    bz.y2 * scale_factor,
                    bz.x3 * scale_factor,
                    bz.y3 * scale_factor,
                    bz.color,
                    bz.width * scale_factor,
                    frame_width,
                    frame_height,
                );
                const added = frame.shape_vertices.items.len - start_len;
                try appendOrMergeShapeOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });
            },

            .rect => |rect| {
                const start_len = frame.shape_vertices.items.len;
                try appendShape(
                    allocator,
                    &frame.shape_vertices,
                    .{
                        .x = rect.bounds.x * scale_factor,
                        .y = rect.bounds.y * scale_factor,
                        .width = rect.bounds.width * scale_factor,
                        .height = rect.bounds.height * scale_factor,
                    },
                    rect.color,
                    rect.radius * scale_factor,
                    0.0,
                    0.0,
                    frame_width,
                    frame_height,
                );
                const added = frame.shape_vertices.items.len - start_len;
                try appendOrMergeShapeOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });
            },

            .polygon => |poly| {
                if (poly.points.len >= 3) {
                    const start_len = frame.shape_vertices.items.len;
                    try appendPolygonFill(
                        allocator,
                        &frame.shape_vertices,
                        poly.points,
                        poly.color,
                        scale_factor,
                        frame_width,
                        frame_height,
                    );
                    const added = frame.shape_vertices.items.len - start_len;
                    try appendOrMergeShapeOp(allocator, frame, .{
                        .vertex_offset = @intCast(start_len),
                        .vertex_count = @intCast(added),
                        .clip_rect = current_clip,
                        .clip_offset = current_clip_offset,
                    });
                }
            },

            .border => |border| {
                const start_len = frame.shape_vertices.items.len;
                try appendShape(
                    allocator,
                    &frame.shape_vertices,
                    .{
                        .x = border.bounds.x * scale_factor,
                        .y = border.bounds.y * scale_factor,
                        .width = border.bounds.width * scale_factor,
                        .height = border.bounds.height * scale_factor,
                    },
                    border.color,
                    border.radius * scale_factor,
                    border.width * scale_factor,
                    0.0,
                    frame_width,
                    frame_height,
                );
                const added = frame.shape_vertices.items.len - start_len;
                try appendOrMergeShapeOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });
            },

            .shadow => |shadow| {
                const start_len = frame.shape_vertices.items.len;
                const spread = shadow.spread;
                const blur = shadow.blur_radius;
                const expansion = spread + blur * 2.0;

                try appendShape(
                    allocator,
                    &frame.shape_vertices,
                    .{
                        .x = (shadow.bounds.x - expansion) * scale_factor,
                        .y = (shadow.bounds.y - expansion) * scale_factor,
                        .width = (shadow.bounds.width + expansion * 2.0) * scale_factor,
                        .height = (shadow.bounds.height + expansion * 2.0) * scale_factor,
                    },
                    shadow.color,
                    (shadow.radius + spread) * scale_factor,
                    0.0,
                    blur * scale_factor,
                    frame_width,
                    frame_height,
                );
                const added = frame.shape_vertices.items.len - start_len;
                try appendOrMergeShapeOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });
            },

            .text => |text| {
                var scaled_text = text;
                scaled_text.size *= scale_factor;
                scaled_text.line_height *= scale_factor;
                scaled_text.baseline_x *= scale_factor;
                scaled_text.baseline_y *= scale_factor;
                scaled_text.letter_spacing *= scale_factor;
                scaled_text.word_spacing *= scale_factor;

                const start_len = frame.text_vertices.items.len;
                try text_renderer.appendText(
                    allocator,
                    &frame.text_vertices,
                    scaled_text,
                    0.0,
                    0.0,
                    frame_width,
                    frame_height,
                );
                const added = frame.text_vertices.items.len - start_len;
                try appendOrMergeTextOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });

                // Flush any color emoji quads produced by appendText. A quad only exists after a
                // successful color-glyph bake, which implies the lazy color atlas was created —
                // the else arm is unreachable by construction and just drops the quads.
                if (text_renderer.pending_color_quads.items.len > 0) {
                    if (text_renderer.color_gpu) |cg| {
                        const color_start = frame.image_vertices.items.len;
                        for (text_renderer.pending_color_quads.items) |pq| {
                            try appendImage(allocator, &frame.image_vertices, .{ .x = pq.x, .y = pq.y, .width = pq.w, .height = pq.h }, pq.u0, pq.v0, pq.u1, pq.v1, .{ 255, 255, 255, 255 }, 0.0, 0.0, frame_width, frame_height);
                        }
                        const color_added = frame.image_vertices.items.len - color_start;
                        try appendOrMergeImageOp(allocator, frame, .{
                            .vertex_offset = @intCast(color_start),
                            .vertex_count = @intCast(color_added),
                            .clip_rect = current_clip,
                            .clip_offset = current_clip_offset,
                            .texture = cg.texture,
                            .texture_view = cg.view,
                            .bind_group = cg.bind_group,
                            .owns_resources = false,
                        });
                    }
                    text_renderer.pending_color_quads.clearRetainingCapacity();
                }
            },

            .image => |image| {
                var batch = try createImageBatch(device, queue, allocator, image, gpu_ui, current_clip, current_clip_offset);
                errdefer batch.deinit();
                const start_len = frame.image_vertices.items.len;
                try appendImage(
                    allocator,
                    &frame.image_vertices,
                    .{
                        .x = image.bounds.x * scale_factor,
                        .y = image.bounds.y * scale_factor,
                        .width = image.bounds.width * scale_factor,
                        .height = image.bounds.height * scale_factor,
                    },
                    image.u0,
                    image.v0,
                    image.u1,
                    image.v1,
                    .{ 255, 255, 255, 255 },
                    0.0,
                    0.0,
                    frame_width,
                    frame_height,
                );
                const added = frame.image_vertices.items.len - start_len;
                batch.vertex_offset = @intCast(start_len);
                batch.vertex_count = @intCast(added);
                try appendOrMergeImageOp(allocator, frame, batch);
            },

            .text_layout => |tl| {
                const entry = text_renderer.layout_cache.getPtr(tl.handle) orelse continue;
                const start_len = frame.text_vertices.items.len;
                try text_renderer.appendLayoutGlyphs(
                    allocator,
                    entry,
                    &frame.text_vertices,
                    tl.draw_x,
                    tl.draw_y,
                    scale_factor,
                    frame_width,
                    frame_height,
                    colorToU8(tl.color),
                );
                const added = frame.text_vertices.items.len - start_len;
                try appendOrMergeTextOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                });

                // Cached layouts bake color-emoji quads too; flush them exactly like the
                // immediate-text path above (same lazily-created atlas, same image pipeline).
                if (text_renderer.pending_color_quads.items.len > 0) {
                    if (text_renderer.color_gpu) |cg| {
                        const color_start = frame.image_vertices.items.len;
                        for (text_renderer.pending_color_quads.items) |pq| {
                            try appendImage(allocator, &frame.image_vertices, .{ .x = pq.x, .y = pq.y, .width = pq.w, .height = pq.h }, pq.u0, pq.v0, pq.u1, pq.v1, .{ 255, 255, 255, 255 }, 0.0, 0.0, frame_width, frame_height);
                        }
                        const color_added = frame.image_vertices.items.len - color_start;
                        try appendOrMergeImageOp(allocator, frame, .{
                            .vertex_offset = @intCast(color_start),
                            .vertex_count = @intCast(color_added),
                            .clip_rect = current_clip,
                            .clip_offset = current_clip_offset,
                            .texture = cg.texture,
                            .texture_view = cg.view,
                            .bind_group = cg.bind_group,
                            .owns_resources = false,
                        });
                    }
                    text_renderer.pending_color_quads.clearRetainingCapacity();
                }
            },

            .glyph_run => |gr| {
                const cached = gpu_ui.image_cache.get(gr.atlas_handle) orelse continue;
                const start_len = frame.image_vertices.items.len;
                for (gr.quads) |q| {
                    try appendImage(
                        allocator,
                        &frame.image_vertices,
                        .{ .x = q.x * scale_factor, .y = q.y * scale_factor, .width = q.w * scale_factor, .height = q.h * scale_factor },
                        q.u0,
                        q.v0,
                        q.u1,
                        q.v1,
                        colorToU8(gr.tint),
                        0.0,
                        0.0,
                        frame_width,
                        frame_height,
                    );
                }
                const added = frame.image_vertices.items.len - start_len;
                if (added > 0) try appendOrMergeImageOp(allocator, frame, .{
                    .vertex_offset = @intCast(start_len),
                    .vertex_count = @intCast(added),
                    .clip_rect = current_clip,
                    .clip_offset = current_clip_offset,
                    .texture = cached.texture,
                    .texture_view = cached.texture_view,
                    .bind_group = cached.bind_group,
                    .owns_resources = false,
                });
            },
        }

        // Apply the active transform to everything this command just emitted (marks taken at the
        // top of the iteration, so a command that feeds several vertex lists — e.g. text spilling
        // color-emoji quads into image_vertices — is covered wholesale).
        if (xf) |cur| {
            applyNdcTransform(ShapeVertex, frame.shape_vertices.items[shape_mark..], cur.ndc);
            applyNdcTransform(ft_text.TextVertex, frame.text_vertices.items[text_mark..], cur.ndc);
            applyNdcTransform(ImageVertex, frame.image_vertices.items[image_mark..], cur.ndc);
            applyNdcTransform(LiquidGlassVertex, frame.liquid_glass_vertices.items[glass_mark..], cur.ndc);
            applyNdcTransformFx(frame.shader_effect_vertices.items[fx_mark..], cur.ndc);
        }
    }
}

// Adjacent same-kind ops with the same clip write into one monotonically-growing per-kind vertex
// list, so their vertex ranges are always contiguous — coalescing them into one op collapses N
// setPipeline+setVertexBuffer+draw calls into one. This is the whole point of the "OrMerge" names.
// liquid_glass / shader_effect ops are deliberately NOT merged: each is a scene→backdrop copy
// barrier in replayFramePaint, and merging two would make the second sample a stale backdrop.
fn appendOrMergeShapeOp(allocator: std.mem.Allocator, frame: *FramePaint, batch: ShapeBatch) !void {
    if (batch.vertex_count == 0) return;
    if (frame.ops.items.len > 0) {
        const last = &frame.ops.items[frame.ops.items.len - 1];
        if (last.* == .shape) {
            const prev = &last.shape;
            if (prev.vertex_offset + prev.vertex_count == batch.vertex_offset and
                prev.clip_offset == batch.clip_offset and
                std.meta.eql(prev.clip_rect, batch.clip_rect))
            {
                prev.vertex_count += batch.vertex_count;
                return;
            }
        }
    }
    try frame.ops.append(allocator, .{ .shape = batch });
}
fn appendOrMergeLiquidGlassOp(allocator: std.mem.Allocator, frame: *FramePaint, batch: ShapeBatch) !void {
    if (batch.vertex_count == 0) return;
    try frame.ops.append(allocator, .{ .liquid_glass = batch });
}
fn appendOrMergeTextOp(allocator: std.mem.Allocator, frame: *FramePaint, batch: TextBatch) !void {
    if (batch.vertex_count == 0) return;
    if (frame.ops.items.len > 0) {
        const last = &frame.ops.items[frame.ops.items.len - 1];
        if (last.* == .text) {
            const prev = &last.text;
            if (prev.vertex_offset + prev.vertex_count == batch.vertex_offset and
                prev.clip_offset == batch.clip_offset and
                std.meta.eql(prev.clip_rect, batch.clip_rect))
            {
                prev.vertex_count += batch.vertex_count;
                return;
            }
        }
    }
    try frame.ops.append(allocator, .{ .text = batch });
}
// Image ops only merge when they share the same GPU bind group (same atlas/texture) and neither
// owns its resources — an owning batch has a unique texture and its own deinit, so it is never
// merged into or away.
fn appendOrMergeImageOp(allocator: std.mem.Allocator, frame: *FramePaint, batch: ImageBatch) !void {
    if (batch.vertex_count == 0) return;
    if (!batch.owns_resources and frame.ops.items.len > 0) {
        const last = &frame.ops.items[frame.ops.items.len - 1];
        if (last.* == .image) {
            const prev = &last.image;
            if (!prev.owns_resources and prev.bind_group == batch.bind_group and
                prev.vertex_offset + prev.vertex_count == batch.vertex_offset and
                prev.clip_offset == batch.clip_offset and
                std.meta.eql(prev.clip_rect, batch.clip_rect))
            {
                prev.vertex_count += batch.vertex_count;
                return;
            }
        }
    }
    try frame.ops.append(allocator, .{ .image = batch });
}
fn appendOrMergeShaderEffectOp(allocator: std.mem.Allocator, frame: *FramePaint, batch: ShaderEffectBatch) !void {
    if (batch.vertex_count == 0) return;
    try frame.ops.append(allocator, .{ .shader_effect = batch });
}
const SceneAlloc = struct { w: u32, h: u32 };

/// Headroom granularity while a resize drag grows the window: allocation jumps to the next step
/// boundary, so a drag reallocates once per step instead of once per frame.
const scene_alloc_step: u32 = 256;

/// Frames the size must hold still before the pair snaps back to the exact size. About a second —
/// late enough that a paused drag doesn't thrash, early enough that the oversized margin (whose
/// stale texels glass/blur taps can graze at the window's right/bottom edge) is a mid-drag
/// transient rather than a steady state.
const scene_settle_frames: u32 = 60;

fn roundUpStep(v: u32) u32 {
    return (v + scene_alloc_step - 1) / scene_alloc_step * scene_alloc_step;
}

/// Grow-only sizing policy for the scene/backdrop pair — see the field comments on GpuUi. Called
/// once per frame BEFORE buildFrameOps (shader-effect UVs normalize over the alloc extents, so the
/// ops must be built against the dims the textures will actually have), and ensureSceneTexture then
/// materializes exactly what this returned. Ticks the settle counter, so once per frame only.
fn planSceneAlloc(gpu_ui: *GpuUi, frame_width: u32, frame_height: u32) SceneAlloc {
    // No texture yet (first frame, or a direct-mode window that never allocates one): exact size.
    if (gpu_ui.scene_texture == null)
        return .{ .w = frame_width, .h = frame_height };

    if (frame_width != gpu_ui.scene_frame_width or frame_height != gpu_ui.scene_frame_height) {
        gpu_ui.scene_stable_frames = 0;
        // Still fits → keep the allocation, the viewport confines rendering to the frame region.
        // Outgrew it → step up with headroom, never shrinking a dimension mid-drag (the settle
        // pass trues both up once the drag ends).
        if (frame_width <= gpu_ui.scene_alloc_width and frame_height <= gpu_ui.scene_alloc_height)
            return .{ .w = gpu_ui.scene_alloc_width, .h = gpu_ui.scene_alloc_height };
        return .{
            .w = @max(roundUpStep(frame_width), gpu_ui.scene_alloc_width),
            .h = @max(roundUpStep(frame_height), gpu_ui.scene_alloc_height),
        };
    }

    // Stable size, exact allocation: the steady state.
    if (gpu_ui.scene_alloc_width == frame_width and gpu_ui.scene_alloc_height == frame_height)
        return .{ .w = frame_width, .h = frame_height };

    // Stable size, oversized allocation: count down to the settle realloc.
    gpu_ui.scene_stable_frames += 1;
    if (gpu_ui.scene_stable_frames >= scene_settle_frames)
        return .{ .w = frame_width, .h = frame_height };
    return .{ .w = gpu_ui.scene_alloc_width, .h = gpu_ui.scene_alloc_height };
}

/// Create (or size-track) the persistent scene texture + blit bind group at the planned allocation.
/// A frame-dims change within the same allocation rebuilds nothing — the viewport confines the next
/// frame to the new sub-region. An allocation change releases the backdrop trio too: scene and
/// backdrop allocations must stay in sync (the backdrop re-creates lazily at the new dims the next
/// time a frame contains a backdrop-sampling op — see ensureBackdropTexture), and marks the scene's
/// contents lost so partial repaint skips one frame.
fn ensureSceneTexture(
    gpu_ui: *GpuUi,
    device: *wgpu.Device,
    frame_width: u32,
    frame_height: u32,
    alloc: SceneAlloc,
) !void {
    if (gpu_ui.scene_alloc_width == alloc.w and
        gpu_ui.scene_alloc_height == alloc.h and
        gpu_ui.scene_texture != null and
        gpu_ui.scene_texture_view != null and
        gpu_ui.blit_bind_group != null)
    {
        gpu_ui.scene_frame_width = frame_width;
        gpu_ui.scene_frame_height = frame_height;
        return;
    }

    if (gpu_ui.blit_bind_group) |bg| bg.release();
    gpu_ui.blit_bind_group = null;
    if (gpu_ui.backdrop_bind_group) |bg| bg.release();
    gpu_ui.backdrop_bind_group = null;
    if (gpu_ui.backdrop_texture_view) |tv| tv.release();
    gpu_ui.backdrop_texture_view = null;
    if (gpu_ui.backdrop_texture) |t| t.release();
    gpu_ui.backdrop_texture = null;
    if (gpu_ui.scene_texture_view) |tv| tv.release();
    gpu_ui.scene_texture_view = null;
    if (gpu_ui.scene_texture) |t| t.release();
    gpu_ui.scene_texture = null;

    const scene_tex = device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote scene texture"),
        .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.copy_src | wgpu.TextureUsages.copy_dst | wgpu.TextureUsages.texture_binding,
        .dimension = .@"2d",
        .size = .{ .width = alloc.w, .height = alloc.h, .depth_or_array_layers = 1 },
        .format = gpu_ui.surface_format,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return error.WgpuTextureUnavailable;
    errdefer scene_tex.release();

    const scene_view = scene_tex.createView(null) orelse return error.WgpuTextureViewUnavailable;
    errdefer scene_view.release();

    const blit_bg = createImageBindGroup(device, gpu_ui.text.bindGroupLayout(), scene_view, gpu_ui.backdrop_sampler) orelse return error.WgpuBindGroupUnavailable;

    gpu_ui.scene_texture = scene_tex;
    gpu_ui.scene_texture_view = scene_view;
    gpu_ui.blit_bind_group = blit_bg;
    gpu_ui.scene_frame_width = frame_width;
    gpu_ui.scene_frame_height = frame_height;
    gpu_ui.scene_alloc_width = alloc.w;
    gpu_ui.scene_alloc_height = alloc.h;
    gpu_ui.scene_stable_frames = 0;
    gpu_ui.scene_contents_lost = true;
}

/// Lazily create the backdrop texture sampled by Liquid Glass / custom shader-effect ops. Callers
/// gate this on frameHasBackdropOp AFTER ensureSceneTexture, so a UI without glass never pays the
/// full-window texture (−W×H×4 bytes GPU per window). A plain null check suffices because
/// ensureSceneTexture releases + nulls the trio on every size change.
fn ensureBackdropTexture(
    gpu_ui: *GpuUi,
    device: *wgpu.Device,
) !void {
    if (gpu_ui.backdrop_texture != null and
        gpu_ui.backdrop_texture_view != null and
        gpu_ui.backdrop_bind_group != null) return;

    // Sized to the scene ALLOCATION, not the frame: copySceneToBackdrop copies the frame-sized
    // region, and glass shaders map fragment position → texel through textureDimensions, which
    // only stays correct while both textures share their extents.
    const backdrop_tex = device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote backdrop texture"),
        .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
        .dimension = .@"2d",
        .size = .{ .width = gpu_ui.scene_alloc_width, .height = gpu_ui.scene_alloc_height, .depth_or_array_layers = 1 },
        .format = gpu_ui.surface_format,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return error.WgpuTextureUnavailable;
    errdefer backdrop_tex.release();

    const backdrop_view = backdrop_tex.createView(null) orelse return error.WgpuTextureViewUnavailable;
    errdefer backdrop_view.release();

    const backdrop_bg = createImageBindGroup(device, gpu_ui.backdrop_bgl, backdrop_view, gpu_ui.backdrop_sampler) orelse return error.WgpuBindGroupUnavailable;

    gpu_ui.backdrop_texture = backdrop_tex;
    gpu_ui.backdrop_texture_view = backdrop_view;
    gpu_ui.backdrop_bind_group = backdrop_bg;
}

/// (Re)create the scratch scene texture + blit bind group renderToTexture renders into, recreated
/// only on a size change (mirrors ensureSceneTexture). renderToTexture is self-submitting and the
/// queue executes submissions in order, so sequential RT renders can safely share one persistent
/// scratch target instead of allocating and destroying a full-size texture per call per frame.
fn ensureRtSceneTexture(
    gpu_ui: *GpuUi,
    device: *wgpu.Device,
    width: u32,
    height: u32,
) !void {
    if (gpu_ui.rt_scene_width == width and
        gpu_ui.rt_scene_height == height and
        gpu_ui.rt_scene_texture != null and
        gpu_ui.rt_scene_view != null and
        gpu_ui.rt_blit_bind_group != null) return;

    if (gpu_ui.rt_blit_bind_group) |bg| bg.release();
    gpu_ui.rt_blit_bind_group = null;
    if (gpu_ui.rt_scene_view) |tv| tv.release();
    gpu_ui.rt_scene_view = null;
    if (gpu_ui.rt_scene_texture) |t| t.release();
    gpu_ui.rt_scene_texture = null;

    const scene_tex = device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote rt scene"),
        .usage = wgpu.TextureUsages.render_attachment | wgpu.TextureUsages.copy_src | wgpu.TextureUsages.texture_binding,
        .dimension = .@"2d",
        .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
        .format = gpu_ui.surface_format,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return error.RtSceneTexFailed;
    errdefer scene_tex.release();

    const scene_view = scene_tex.createView(null) orelse return error.RtSceneViewFailed;
    errdefer scene_view.release();

    const blit_bg = createImageBindGroup(device, gpu_ui.text.bindGroupLayout(), scene_view, gpu_ui.backdrop_sampler) orelse return error.RtBlitBgFailed;

    gpu_ui.rt_scene_texture = scene_tex;
    gpu_ui.rt_scene_view = scene_view;
    gpu_ui.rt_blit_bind_group = blit_bg;
    gpu_ui.rt_scene_width = width;
    gpu_ui.rt_scene_height = height;
}

const blit_quad_size = @sizeOf(ImageVertex) * 4;

/// Fullscreen quad in NDC (UV top-left=(0,0) bottom-right=(u1,v1)), corners TL,TR,BL,BR to match
/// the shared quad index pattern. u1/v1 = frame/alloc extents of the source texture — (1,1) except
/// while the scene texture carries grow-only resize headroom — so the quad is rewritten only when
/// that ratio (or the buffer itself) changes and stays a cached constant in the steady state.
fn ensureBlitQuad(device: *wgpu.Device, queue: *wgpu.Queue, gpu_ui: *GpuUi, u_max: f32, v_max: f32) !*wgpu.Buffer {
    const need_upload = gpu_ui.blit_vertex_buffer == null or gpu_ui.blit_vertex_buffer_size < blit_quad_size;
    const vbuf = try ensureVertexBuffer(device, &gpu_ui.blit_vertex_buffer, &gpu_ui.blit_vertex_buffer_size, "zigote blit vertices", blit_quad_size);
    if (need_upload or gpu_ui.blit_quad_u1 != u_max or gpu_ui.blit_quad_v1 != v_max) {
        const verts = [_]ImageVertex{
            .{ .position = .{ -1.0, 1.0 }, .uv = .{ 0.0, 0.0 }, .color = .{ 255, 255, 255, 255 } },
            .{ .position = .{ 1.0, 1.0 }, .uv = .{ u_max, 0.0 }, .color = .{ 255, 255, 255, 255 } },
            .{ .position = .{ -1.0, -1.0 }, .uv = .{ 0.0, v_max }, .color = .{ 255, 255, 255, 255 } },
            .{ .position = .{ 1.0, -1.0 }, .uv = .{ u_max, v_max }, .color = .{ 255, 255, 255, 255 } },
        };
        queue.writeBuffer(vbuf, 0, std.mem.sliceAsBytes(&verts).ptr, blit_quad_size);
        gpu_ui.blit_quad_u1 = u_max;
        gpu_ui.blit_quad_v1 = v_max;
    }
    return vbuf;
}

/// Draw the fullscreen blit quad through the image pipeline with `blit_bg` as the source texture.
/// Shared by the swapchain blit and the renderToTexture blit. `u1`/`v1` map the quad onto the
/// frame-sized region of the source (1,1 for an exact-size source — see ensureBlitQuad).
fn drawBlit(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    blit_bg: *wgpu.BindGroup,
    frame_width: u32,
    frame_height: u32,
    u_max: f32,
    v_max: f32,
) !void {
    const vbuf = try ensureBlitQuad(device, queue, gpu_ui, u_max, v_max);
    try ensureQuadIndexBuffer(device, queue, gpu_ui, 1);
    const index_buffer = gpu_ui.quad_index_buffer orelse return;
    // The image pipeline's layout includes the rounded-clip group; bind the disabled slot 0.
    const clip_bg = gpu_ui.clip_bind_group orelse return;
    const no_clip: u32 = 0;
    pass.setPipeline(gpu_ui.image_pipeline);
    pass.setBindGroup(0, blit_bg, 0, null);
    pass.setBindGroup(1, clip_bg, 1, @ptrCast(&no_clip));
    pass.setVertexBuffer(0, vbuf, 0, blit_quad_size);
    pass.setIndexBuffer(index_buffer, .uint32, 0, quadIndexBytes(gpu_ui));
    pass.setScissorRect(0, 0, frame_width, frame_height);
    pass.drawIndexed(6, 1, 0, 0, 0);
}

fn blitSceneToSwapchain(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    pass: *wgpu.RenderPassEncoder,
    gpu_ui: *GpuUi,
    frame_width: u32,
    frame_height: u32,
) !void {
    const blit_bg = gpu_ui.blit_bind_group orelse return;
    // The scene texture may be oversized (grow-only resize headroom): sample only its frame region.
    // Fragment centres land exactly on texel centres either way, so no epsilon is needed.
    const u_max = @as(f32, @floatFromInt(frame_width)) / @as(f32, @floatFromInt(gpu_ui.scene_alloc_width));
    const v_max = @as(f32, @floatFromInt(frame_height)) / @as(f32, @floatFromInt(gpu_ui.scene_alloc_height));
    try drawBlit(device, queue, pass, gpu_ui, blit_bg, frame_width, frame_height, u_max, v_max);
}

fn appendShaderEffect(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(ShaderEffectVertex),
    bounds: zg.Rect,
    params: [8]f32,
    frame_width: f32,
    frame_height: f32,
    backdrop_width: f32,
    backdrop_height: f32,
) !void {
    if (bounds.width <= 0 or bounds.height <= 0) return;

    const x0 = bounds.x / frame_width * 2.0 - 1.0;
    const y0 = 1.0 - bounds.y / frame_height * 2.0;
    const x1 = (bounds.x + bounds.width) / frame_width * 2.0 - 1.0;
    const y1 = 1.0 - (bounds.y + bounds.height) / frame_height * 2.0;

    // UVs normalize over the backdrop TEXTURE's extents, which can exceed the frame while the
    // scene carries grow-only resize headroom (the content sits 1:1 in its top-left region).
    const uv0_x = bounds.x / backdrop_width;
    const uv0_y = bounds.y / backdrop_height;
    const uv1_x = (bounds.x + bounds.width) / backdrop_width;
    const uv1_y = (bounds.y + bounds.height) / backdrop_height;

    const pa = [4]f32{ params[0], params[1], params[2], params[3] };
    const pb = [4]f32{ params[4], params[5], params[6], params[7] };

    try vertices.appendSlice(allocator, &.{
        .{ .position = .{ x0, y0 }, .uv = .{ uv0_x, uv0_y }, .params_a = pa, .params_b = pb },
        .{ .position = .{ x1, y0 }, .uv = .{ uv1_x, uv0_y }, .params_a = pa, .params_b = pb },
        .{ .position = .{ x0, y1 }, .uv = .{ uv0_x, uv1_y }, .params_a = pa, .params_b = pb },
        .{ .position = .{ x1, y0 }, .uv = .{ uv1_x, uv0_y }, .params_a = pa, .params_b = pb },
        .{ .position = .{ x1, y1 }, .uv = .{ uv1_x, uv1_y }, .params_a = pa, .params_b = pb },
        .{ .position = .{ x0, y1 }, .uv = .{ uv0_x, uv1_y }, .params_a = pa, .params_b = pb },
    });
}

fn createCustomShaderPipeline(
    device: *wgpu.Device,
    backdrop_bgl: *wgpu.BindGroupLayout,
    image_bgl: ?*wgpu.BindGroupLayout,
    format: wgpu.TextureFormat,
    wgsl: []const u8,
) !*wgpu.RenderPipeline {
    var shader_desc = wgpu.shaderModuleWGSLDescriptor(.{
        .label = "zigote custom shader",
        .code = wgsl,
    });
    const shader = device.createShaderModule(&shader_desc) orelse return error.WgpuShaderUnavailable;
    defer shader.release();

    const attributes = [_]wgpu.VertexAttribute{
        .{ .format = .float32x2, .offset = @offsetOf(ShaderEffectVertex, "position"), .shader_location = 0 },
        .{ .format = .float32x2, .offset = @offsetOf(ShaderEffectVertex, "uv"), .shader_location = 1 },
        .{ .format = .float32x4, .offset = @offsetOf(ShaderEffectVertex, "params_a"), .shader_location = 2 },
        .{ .format = .float32x4, .offset = @offsetOf(ShaderEffectVertex, "params_b"), .shader_location = 3 },
    };
    const vbl = wgpu.VertexBufferLayout{
        .array_stride = @sizeOf(ShaderEffectVertex),
        .step_mode = .vertex,
        .attribute_count = attributes.len,
        .attributes = &attributes,
    };

    var bgls = [2]*wgpu.BindGroupLayout{ backdrop_bgl, backdrop_bgl };
    var bgl_count: usize = 1;
    if (image_bgl) |bgl| {
        bgls[1] = bgl;
        bgl_count = 2;
    }
    const pipeline_layout = device.createPipelineLayout(&.{
        .label = wgpu.StringView.fromSlice("zigote custom shader layout"),
        .bind_group_layout_count = bgl_count,
        .bind_group_layouts = &bgls,
    }) orelse return error.WgpuPipelineUnavailable;
    defer pipeline_layout.release();

    const blend = wgpu.BlendState{
        .color = .{ .src_factor = .src_alpha, .dst_factor = .one_minus_src_alpha, .operation = .add },
        .alpha = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha, .operation = .add },
    };
    const color_target = wgpu.ColorTargetState{
        .format = format,
        .blend = &blend,
        .write_mask = wgpu.ColorWriteMasks.all,
    };
    const fragment = wgpu.FragmentState{
        .module = shader,
        .entry_point = wgpu.StringView.fromSlice("fs_main"),
        .target_count = 1,
        .targets = @ptrCast(&color_target),
    };

    const pipeline = device.createRenderPipeline(&.{
        .label = wgpu.StringView.fromSlice("zigote custom shader pipeline"),
        .vertex = .{
            .module = shader,
            .entry_point = wgpu.StringView.fromSlice("vs_main"),
            .buffer_count = 1,
            .buffers = @ptrCast(&vbl),
        },
        .primitive = .{},
        .multisample = .{},
        .fragment = &fragment,
        .layout = pipeline_layout,
    }) orelse return error.WgpuPipelineUnavailable;

    return pipeline;
}

fn createImageBatch(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    allocator: std.mem.Allocator,
    image: zg.paint.Image,
    gpu_ui: *GpuUi,
    clip_rect: ?zg.Rect,
    clip_offset: u32,
) !ImageBatch {
    const key = imageKey(image);
    if (gpu_ui.image_cache.get(key)) |cached| {
        return .{
            .vertex_offset = 0,
            .vertex_count = 0,
            .clip_rect = clip_rect,
            .clip_offset = clip_offset,
            .texture = cached.texture,
            .texture_view = cached.texture_view,
            .bind_group = cached.bind_group,
            .owns_resources = false,
        };
    }

    const texture = try createImageTexture(device, queue, allocator, image);
    errdefer texture.release();

    const texture_view = texture.createView(null) orelse return error.WgpuImageTextureViewUnavailable;
    errdefer texture_view.release();

    const bind_group = createImageBindGroup(
        device,
        gpu_ui.text.bindGroupLayout(),
        texture_view,
        gpu_ui.text.getSampler(),
    ) orelse return error.WgpuImageBindGroupUnavailable;
    errdefer bind_group.release();

    // An app-owned handle (image_registry entry, render texture) is pinned: cached unconditionally
    // and kept until zigote_release_texture frees it. Anything else is keyed by a hash of the
    // inline pixels and cached only while under the (generous) cap — over-cap images fall through
    // to owns_resources=true and are released after the frame is submitted (the caller's deinit),
    // safe unlike evicting a cached texture mid-frame, which would free a resource still recorded
    // in the open command encoder.
    const pinned = image.cache_key != null;
    if (pinned or gpu_ui.unpinned_cached < gpu_ui.max_cached_images) {
        try gpu_ui.image_cache.put(key, .{
            .texture = texture,
            .texture_view = texture_view,
            .bind_group = bind_group,
            .pinned = pinned,
        });
        if (pinned) {
            // Best-effort: a failed append only means the CPU copy is freed later (at release or
            // shutdown), never that the texture is wrong.
            gpu_ui.uploaded_keys.append(gpu_ui.allocator, key) catch {};
        } else {
            gpu_ui.unpinned_cached += 1;
        }
        return .{
            .vertex_offset = 0,
            .vertex_count = 0,
            .clip_rect = clip_rect,
            .clip_offset = clip_offset,
            .texture = texture,
            .texture_view = texture_view,
            .bind_group = bind_group,
            .owns_resources = false,
        };
    }

    return .{
        .vertex_offset = 0,
        .vertex_count = 0,
        .clip_rect = clip_rect,
        .clip_offset = clip_offset,
        .texture = texture,
        .texture_view = texture_view,
        .bind_group = bind_group,
        .owns_resources = true,
    };
}

fn createImageTexture(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    allocator: std.mem.Allocator,
    image: zg.paint.Image,
) !*wgpu.Texture {
    if (image.width == 0 or image.height == 0) return error.WgpuImageTextureUnavailable;

    const row_bytes = std.math.mul(usize, @as(usize, image.width), 4) catch
        return error.WgpuImageTextureUnavailable;
    const src_stride = if (image.stride != 0) @as(usize, image.stride) else row_bytes;
    if (src_stride < row_bytes) return error.WgpuImageTextureUnavailable;
    const required_rgba_len = std.math.mul(usize, src_stride, @as(usize, image.height)) catch
        return error.WgpuImageTextureUnavailable;
    if (image.pixels.len < required_rgba_len) return error.WgpuImageTextureUnavailable;

    const texture = device.createTexture(&.{
        .label = wgpu.StringView.fromSlice("zigote image texture"),
        .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
        .dimension = .@"2d",
        .size = .{
            .width = image.width,
            .height = image.height,
            .depth_or_array_layers = 1,
        },
        // The producer's channel order, resolved by the sampler instead of by the CPU. A source
        // that is already BGRA (every Cairo/GTK surface, most video decoders) then needs no
        // conversion pass at all.
        .format = if (image.bgra) .bgra8_unorm_srgb else .rgba8_unorm_srgb,
        .mip_level_count = 1,
        .sample_count = 1,
    }) orelse return error.WgpuImageTextureUnavailable;

    const bytes_per_row = std.mem.alignForward(usize, src_stride, 256);

    // Already 256-aligned (any width that's a multiple of 64 px): upload straight from the decoded
    // pixels — no staging alloc, no zero-fill, no per-row copy.
    if (bytes_per_row == src_stride) {
        queue.writeTexture(
            &.{
                .texture = texture,
                .mip_level = 0,
                .origin = .{ .x = 0, .y = 0, .z = 0 },
                .aspect = .all,
            },
            image.pixels.ptr,
            required_rgba_len,
            &.{
                .offset = 0,
                .bytes_per_row = @intCast(bytes_per_row),
                .rows_per_image = image.height,
            },
            &.{
                .width = image.width,
                .height = image.height,
                .depth_or_array_layers = 1,
            },
        );
        return texture;
    }

    const upload = try allocator.alloc(u8, bytes_per_row * @as(usize, image.height));
    var row: usize = 0;
    while (row < image.height) : (row += 1) {
        const src_start = row * src_stride;
        const dst_start = row * bytes_per_row;
        @memcpy(upload[dst_start .. dst_start + row_bytes], image.pixels[src_start .. src_start + row_bytes]);
        // Zero only the padding tail — the old full @memset cleared the whole buffer and the
        // copies above overwrote most of it again.
        @memset(upload[dst_start + row_bytes .. dst_start + bytes_per_row], 0);
    }

    queue.writeTexture(
        &.{
            .texture = texture,
            .mip_level = 0,
            .origin = .{ .x = 0, .y = 0, .z = 0 },
            .aspect = .all,
        },
        upload.ptr,
        upload.len,
        &.{
            .offset = 0,
            .bytes_per_row = @intCast(bytes_per_row),
            .rows_per_image = image.height,
        },
        &.{
            .width = image.width,
            .height = image.height,
            .depth_or_array_layers = 1,
        },
    );

    return texture;
}

/// The `queue.writeTexture` call shared by the create and update paths. `rows` is already padded to
/// `bytes_per_row`.
fn writeImageRows(
    queue: *wgpu.Queue,
    texture: *wgpu.Texture,
    rows: []const u8,
    bytes_per_row: usize,
    width: u32,
    height: u32,
    y0: u32,
) void {
    queue.writeTexture(
        &.{
            .texture = texture,
            .mip_level = 0,
            .origin = .{ .x = 0, .y = y0, .z = 0 },
            .aspect = .all,
        },
        rows.ptr,
        rows.len,
        &.{
            .offset = 0,
            .bytes_per_row = @intCast(bytes_per_row),
            .rows_per_image = height,
        },
        &.{
            .width = width,
            .height = height,
            .depth_or_array_layers = 1,
        },
    );
}

fn createImageBindGroup(
    device: *wgpu.Device,
    layout: *wgpu.BindGroupLayout,
    texture_view: *wgpu.TextureView,
    sampler: *wgpu.Sampler,
) ?*wgpu.BindGroup {
    const entries = [_]wgpu.BindGroupEntry{
        .{
            .binding = 0,
            .texture_view = texture_view,
        },
        .{
            .binding = 1,
            .sampler = sampler,
        },
    };

    return device.createBindGroup(&.{
        .label = wgpu.StringView.fromSlice("zigote image bind group"),
        .layout = layout,
        .entry_count = entries.len,
        .entries = &entries,
    });
}

fn createTexturePipelineLayout(
    device: *wgpu.Device,
    bind_group_layout: *wgpu.BindGroupLayout,
) ?*wgpu.PipelineLayout {
    const bind_group_layouts = [_]*wgpu.BindGroupLayout{
        bind_group_layout,
    };

    return device.createPipelineLayout(&.{
        .label = wgpu.StringView.fromSlice("zigote image pipeline layout"),
        .bind_group_layout_count = bind_group_layouts.len,
        .bind_group_layouts = &bind_group_layouts,
    });
}

fn appendLiquidGlass(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(LiquidGlassVertex),
    rect: zg.Rect,
    color: zg.Color,
    radius: f32,
    thickness: f32,
    glow_x: f32,
    glow_y: f32,
    pinch: f32,
    clear_tint: f32,
    adapt: f32,
    frame_width: f32,
    frame_height: f32,
) !void {
    if (rect.width <= 0 or rect.height <= 0) return;
    const adapt_c = std.math.clamp(adapt, -1.0, 1.0);

    const x0 = rect.x / frame_width * 2.0 - 1.0;
    const y0 = 1.0 - rect.y / frame_height * 2.0;
    const x1 = (rect.x + rect.width) / frame_width * 2.0 - 1.0;
    const y1 = 1.0 - (rect.y + rect.height) / frame_height * 2.0;

    const c = colorToU8(color);

    const w2 = rect.width / 2.0;
    const h2 = rect.height / 2.0;
    const size = [2]f32{ rect.width, rect.height };

    const glow_pos = [2]f32{ glow_x, glow_y };

    try vertices.appendSlice(allocator, &.{
        // Triangle 1
        .{
            .position = .{ x0, y0 },
            .color = c,
            .local_pos = .{ -w2, -h2 },
            .rect_size = size,
            .radius = radius,
            .thickness = thickness,
            .glow_pos = glow_pos,
            .pinch_strength = pinch,
            .clear_tint = clamp01(clear_tint),
            .adapt = adapt_c,
        },
        .{
            .position = .{ x1, y0 },
            .color = c,
            .local_pos = .{ w2, -h2 },
            .rect_size = size,
            .radius = radius,
            .thickness = thickness,
            .glow_pos = glow_pos,
            .pinch_strength = pinch,
            .clear_tint = clamp01(clear_tint),
            .adapt = adapt_c,
        },
        .{
            .position = .{ x0, y1 },
            .color = c,
            .local_pos = .{ -w2, h2 },
            .rect_size = size,
            .radius = radius,
            .thickness = thickness,
            .glow_pos = glow_pos,
            .pinch_strength = pinch,
            .clear_tint = clamp01(clear_tint),
            .adapt = adapt_c,
        },
        // Triangle 2
        .{
            .position = .{ x1, y0 },
            .color = c,
            .local_pos = .{ w2, -h2 },
            .rect_size = size,
            .radius = radius,
            .thickness = thickness,
            .glow_pos = glow_pos,
            .pinch_strength = pinch,
            .clear_tint = clamp01(clear_tint),
            .adapt = adapt_c,
        },
        .{
            .position = .{ x1, y1 },
            .color = c,
            .local_pos = .{ w2, h2 },
            .rect_size = size,
            .radius = radius,
            .thickness = thickness,
            .glow_pos = glow_pos,
            .pinch_strength = pinch,
            .clear_tint = clamp01(clear_tint),
            .adapt = adapt_c,
        },
        .{
            .position = .{ x0, y1 },
            .color = c,
            .local_pos = .{ -w2, h2 },
            .rect_size = size,
            .radius = radius,
            .thickness = thickness,
            .glow_pos = glow_pos,
            .pinch_strength = pinch,
            .clear_tint = clamp01(clear_tint),
            .adapt = adapt_c,
        },
    });
}

fn appendShape(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(ShapeVertex),
    rect: zg.Rect,
    color: zg.Color,
    radius: f32,
    border_width: f32,
    blur_radius: f32,
    frame_width: f32,
    frame_height: f32,
) !void {
    if (rect.width <= 0 or rect.height <= 0) return;

    const x0 = rect.x / frame_width * 2.0 - 1.0;
    const y0 = 1.0 - rect.y / frame_height * 2.0;
    const x1 = (rect.x + rect.width) / frame_width * 2.0 - 1.0;
    const y1 = 1.0 - (rect.y + rect.height) / frame_height * 2.0;

    const c = colorToU8(color);

    const w2 = rect.width / 2.0;
    const h2 = rect.height / 2.0;
    const size = [2]f32{ rect.width, rect.height };

    try vertices.appendSlice(allocator, &.{
        // Triangle 1
        .{
            .position = .{ x0, y0 },
            .color = c,
            .local_pos = .{ -w2, -h2 },
            .rect_size = size,
            .radius = radius,
            .border_width = border_width,
            .blur_radius = blur_radius,
        },
        .{
            .position = .{ x1, y0 },
            .color = c,
            .local_pos = .{ w2, -h2 },
            .rect_size = size,
            .radius = radius,
            .border_width = border_width,
            .blur_radius = blur_radius,
        },
        .{
            .position = .{ x0, y1 },
            .color = c,
            .local_pos = .{ -w2, h2 },
            .rect_size = size,
            .radius = radius,
            .border_width = border_width,
            .blur_radius = blur_radius,
        },
        // Triangle 2
        .{
            .position = .{ x1, y0 },
            .color = c,
            .local_pos = .{ w2, -h2 },
            .rect_size = size,
            .radius = radius,
            .border_width = border_width,
            .blur_radius = blur_radius,
        },
        .{
            .position = .{ x1, y1 },
            .color = c,
            .local_pos = .{ w2, h2 },
            .rect_size = size,
            .radius = radius,
            .border_width = border_width,
            .blur_radius = blur_radius,
        },
        .{
            .position = .{ x0, y1 },
            .color = c,
            .local_pos = .{ -w2, h2 },
            .rect_size = size,
            .radius = radius,
            .border_width = border_width,
            .blur_radius = blur_radius,
        },
    });
}

/// One stroke-ribbon vertex. Reuses the shape shader: local_pos.x is 0 and rect_size.x is huge,
/// so the rounded-rect SDF collapses to the cross-stream distance |local_pos.y|. `d` is that
/// signed distance (device px); the edge sits at ±width/2 and feathers over ~1px for AA.
fn mkStrokeVertex(sx: f32, sy: f32, d: f32, color: [4]u8, size: [2]f32, fw: f32, fh: f32) ShapeVertex {
    return .{
        .position = .{ sx / fw * 2.0 - 1.0, 1.0 - sy / fh * 2.0 },
        .color = color,
        .local_pos = .{ 0.0, d },
        .rect_size = size,
        .radius = 0.0,
        .border_width = 0.0,
        .blur_radius = 0.0,
    };
}

/// A fully-solid shape vertex: local_pos 0 inside a 1e6-size box makes the rounded-box SDF resolve
/// deep-interior (dist ≈ -5e5) → coverage 1 everywhere, so any triangle is a flat fill. Edges are
/// hard (no SDF feather); callers that want an AA outline stroke the ring separately.
fn mkSolidVertex(sx: f32, sy: f32, color: [4]u8, fw: f32, fh: f32) ShapeVertex {
    return .{
        .position = .{ sx / fw * 2.0 - 1.0, 1.0 - sy / fh * 2.0 },
        .color = color,
        .local_pos = .{ 0.0, 0.0 },
        .rect_size = .{ 1.0e6, 1.0e6 },
        .radius = 0.0,
        .border_width = 0.0,
        .blur_radius = 0.0,
    };
}

/// Triangle-fan a simple polygon ring into solid shape vertices (points in logical pixels).
fn appendPolygonFill(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(ShapeVertex),
    points: []const [2]f32,
    color: zg.Color,
    scale_factor: f32,
    frame_width: f32,
    frame_height: f32,
) !void {
    const c = colorToU8(color);
    const p0x = points[0][0] * scale_factor;
    const p0y = points[0][1] * scale_factor;
    var i: usize = 1;
    while (i + 1 < points.len) : (i += 1) {
        const ax = points[i][0] * scale_factor;
        const ay = points[i][1] * scale_factor;
        const bx = points[i + 1][0] * scale_factor;
        const by = points[i + 1][1] * scale_factor;
        try vertices.appendSlice(allocator, &.{
            mkSolidVertex(p0x, p0y, c, frame_width, frame_height),
            mkSolidVertex(ax, ay, c, frame_width, frame_height),
            mkSolidVertex(bx, by, c, frame_width, frame_height),
        });
    }
}

/// Tessellate a cubic Bézier into a single anti-aliased stroke ribbon appended into the shape
/// vertex buffer. Because the ribbon is one continuous strip (adjacent quads share endpoints,
/// not overlap), translucent strokes blend uniformly — unlike the old overlapping-stamp
/// approximation, which banded on transparency and emitted hundreds of quads per curve.
/// All control points and `width` are in device pixels.
fn appendBezierStroke(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(ShapeVertex),
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    x3: f32,
    y3: f32,
    color: zg.Color,
    width: f32,
    frame_width: f32,
    frame_height: f32,
) !void {
    // Segment count from the control-polygon length (an upper bound on arc length).
    const d01 = @sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
    const d12 = @sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    const d23 = @sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2));
    const total_len = d01 + d12 + d23;

    // Curvature proxy: perpendicular distance of the two handles from the chord. Near-straight
    // segments (chart polylines, straight strokes) need only a couple of quads — the fixed min-8
    // was 4-8× over-tessellating every straight line. Curved beziers keep the smooth budget.
    const chord_dx = x3 - x0;
    const chord_dy = y3 - y0;
    const chord_len = @sqrt(chord_dx * chord_dx + chord_dy * chord_dy);
    var curviness: f32 = 0;
    if (chord_len > 1.0e-3) {
        const h1 = @abs((x1 - x0) * chord_dy - (y1 - y0) * chord_dx) / chord_len;
        const h2 = @abs((x2 - x0) * chord_dy - (y2 - y0) * chord_dx) / chord_len;
        curviness = @max(h1, h2);
    } else {
        curviness = 8.0; // degenerate chord — keep it smooth
    }

    const straight = curviness < 0.35;
    const per_seg: f32 = if (straight) 24.0 else 6.0;
    const min_seg: f32 = if (straight) 1.0 else 8.0;
    const seg_f = std.math.clamp(@ceil(total_len / per_seg), min_seg, @as(f32, 128.0));
    const segments: usize = @intFromFloat(seg_f);

    const w = @max(width, 1.0);
    const half_ext = w * 0.5 + 0.5; // half-width + 1px feather band
    const size = [2]f32{ 1.0e6, w };
    const c = colorToU8(color);

    var px: [129]f32 = undefined;
    var py: [129]f32 = undefined;
    var nx: [129]f32 = undefined;
    var ny: [129]f32 = undefined;

    var i: usize = 0;
    while (i <= segments) : (i += 1) {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(segments));
        const it = 1.0 - t;
        const a = it * it * it;
        const b = 3.0 * it * it * t;
        const cc = 3.0 * it * t * t;
        const e = t * t * t;
        px[i] = a * x0 + b * x1 + cc * x2 + e * x3;
        py[i] = a * y0 + b * y1 + cc * y2 + e * y3;
    }

    // Unit normal (perpendicular to the averaged tangent) at each sample point.
    i = 0;
    while (i <= segments) : (i += 1) {
        var tx: f32 = 0;
        var ty: f32 = 0;
        if (i > 0) {
            tx += px[i] - px[i - 1];
            ty += py[i] - py[i - 1];
        }
        if (i < segments) {
            tx += px[i + 1] - px[i];
            ty += py[i + 1] - py[i];
        }
        const len = @sqrt(tx * tx + ty * ty);
        if (len > 1.0e-6) {
            nx[i] = -ty / len;
            ny[i] = tx / len;
        } else {
            nx[i] = 0;
            ny[i] = 1;
        }
    }

    i = 0;
    while (i < segments) : (i += 1) {
        const aLx = px[i] + nx[i] * half_ext;
        const aLy = py[i] + ny[i] * half_ext;
        const aRx = px[i] - nx[i] * half_ext;
        const aRy = py[i] - ny[i] * half_ext;
        const bLx = px[i + 1] + nx[i + 1] * half_ext;
        const bLy = py[i + 1] + ny[i + 1] * half_ext;
        const bRx = px[i + 1] - nx[i + 1] * half_ext;
        const bRy = py[i + 1] - ny[i + 1] * half_ext;

        try vertices.appendSlice(allocator, &.{
            mkStrokeVertex(aLx, aLy, half_ext, c, size, frame_width, frame_height),
            mkStrokeVertex(aRx, aRy, -half_ext, c, size, frame_width, frame_height),
            mkStrokeVertex(bLx, bLy, half_ext, c, size, frame_width, frame_height),
            mkStrokeVertex(aRx, aRy, -half_ext, c, size, frame_width, frame_height),
            mkStrokeVertex(bRx, bRy, -half_ext, c, size, frame_width, frame_height),
            mkStrokeVertex(bLx, bLy, half_ext, c, size, frame_width, frame_height),
        });
    }
}

fn appendImage(
    allocator: std.mem.Allocator,
    vertices: *std.ArrayList(ImageVertex),
    bounds: zg.Rect,
    uv0_x: f32,
    uv0_y: f32,
    uv1_x: f32,
    uv1_y: f32,
    color: [4]u8,
    offset_x: f32,
    offset_y: f32,
    frame_width: f32,
    frame_height: f32,
) !void {
    if (bounds.width <= 0 or bounds.height <= 0) return;

    const x0 = (bounds.x - offset_x) / frame_width * 2.0 - 1.0;
    const y0 = 1.0 - (bounds.y - offset_y) / frame_height * 2.0;
    const x1 = (bounds.x + bounds.width - offset_x) / frame_width * 2.0 - 1.0;
    const y1 = 1.0 - (bounds.y + bounds.height - offset_y) / frame_height * 2.0;

    // Corners TL,TR,BL,BR — the order the shared quad index pattern (ensureQuadIndexBuffer)
    // expands into the two triangles the old 6-vertex emission produced.
    try vertices.appendSlice(allocator, &.{
        .{ .position = .{ x0, y0 }, .uv = .{ uv0_x, uv0_y }, .color = color },
        .{ .position = .{ x1, y0 }, .uv = .{ uv1_x, uv0_y }, .color = color },
        .{ .position = .{ x0, y1 }, .uv = .{ uv0_x, uv1_y }, .color = color },
        .{ .position = .{ x1, y1 }, .uv = .{ uv1_x, uv1_y }, .color = color },
    });
}

fn imageKey(image: zg.paint.Image) u64 {
    if (image.cache_key) |key| return key;

    var hasher = std.hash.Wyhash.init(0);
    hasher.update(std.mem.asBytes(&image.width));
    hasher.update(std.mem.asBytes(&image.height));
    hasher.update(std.mem.asBytes(&image.pixels.len));
    hasher.update(image.pixels);
    return hasher.final();
}

fn clamp01(value: f32) f32 {
    return @min(1.0, @max(0.0, value));
}

/// Returns Liquid Glass tint amount as 0..1.
///
/// Backward-compatible behavior:
/// - If the paint command has a float field named `clear_tint`, `clearTint`,
///   `tint_strength`, `tint`, or `color_tint`, that value is used.
/// - Otherwise the old API behavior is preserved: `color.a / 255.0` controls tint.
///
/// Recommended command field:
///
/// ```zig
/// clear_tint: f32 = 0.0, // 0 clear, 1 fully color-tinted
/// color: zg.Color = .{ .r = 255, .g = 255, .b = 255, .a = 255 },
/// ```
fn liquidGlassClearTint(lg: anytype, color: zg.Color) f32 {
    const T = @TypeOf(lg);

    if (@hasField(T, "clear_tint")) {
        return clamp01(@as(f32, @floatCast(@field(lg, "clear_tint"))));
    }
    if (@hasField(T, "clearTint")) {
        return clamp01(@as(f32, @floatCast(@field(lg, "clearTint"))));
    }
    if (@hasField(T, "tint_strength")) {
        return clamp01(@as(f32, @floatCast(@field(lg, "tint_strength"))));
    }
    if (@hasField(T, "tint")) {
        return clamp01(@as(f32, @floatCast(@field(lg, "tint"))));
    }
    if (@hasField(T, "color_tint")) {
        return clamp01(@as(f32, @floatCast(@field(lg, "color_tint"))));
    }

    return @as(f32, @floatFromInt(color.a)) / 255.0;
}

fn colorToU8(color: zg.Color) [4]u8 {
    return .{ color.r, color.g, color.b, color.a };
}

/// Renders UI over an already-rendered game/editor viewport while preserving
/// correct Liquid Glass refraction.
///
/// Required texture usage:
/// - base_texture: copy_src
/// - internal scene texture: copy_dst | copy_src | texture_binding | render_attachment
///
/// Pipeline:
/// 1. copy base_texture -> internal scene_texture
/// 2. replay ordered UI commands onto scene_texture
/// 3. before each Liquid Glass/custom shader op, copy scene_texture -> backdrop_texture
/// 4. blit final scene_texture -> target color_view
///
/// Use this for game/editor composition. The older renderFrameOverlay() only
/// receives a TextureView, so it cannot copy the already-rendered game pixels
/// into the Liquid Glass backdrop.
pub fn renderFrameOverlayWithBaseTexture(
    encoder: *wgpu.CommandEncoder,
    base_texture: *wgpu.Texture,
    color_view: *wgpu.TextureView,
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    paint_list: zg.PaintList,
    frame_width: u32,
    frame_height: u32,
    scale_factor: f32,
    frame_index: u32,
) !void {
    _ = frame_index;

    const scratch_allocator = gpu_ui.scratch.allocator();
    defer _ = gpu_ui.scratch.reset(.retain_capacity);

    var frame = FramePaint{};
    defer frame.deinit(scratch_allocator);

    // Planned BEFORE buildFrameOps — see renderFrame.
    const alloc = planSceneAlloc(gpu_ui, frame_width, frame_height);

    try buildFrameOps(
        device,
        queue,
        scratch_allocator,
        &frame,
        gpu_ui,
        paint_list,
        null,
        @floatFromInt(frame_width),
        @floatFromInt(frame_height),
        @floatFromInt(alloc.w),
        @floatFromInt(alloc.h),
        scale_factor,
    );

    gpu_ui.text.uploadAtlasIfDirty(queue);
    gpu_ui.text.uploadColorAtlasIfDirty(queue);

    try ensureSceneTexture(gpu_ui, device, frame_width, frame_height, alloc);
    if (frameHasBackdropOp(&frame)) try ensureBackdropTexture(gpu_ui, device);
    const scene_view = gpu_ui.scene_texture_view orelse return error.WgpuTextureViewUnavailable;
    const scene_tex = gpu_ui.scene_texture orelse return error.WgpuTextureUnavailable;
    const backdrop_tex = gpu_ui.backdrop_texture;
    const backdrop_bg = gpu_ui.backdrop_bind_group;

    const uploaded = try uploadFrameVertices(device, queue, gpu_ui, &frame);

    // Seed the UI scene with the already-rendered game/editor viewport. This is
    // the critical step that makes Liquid Glass sample the game behind it.
    copyTextureToScene(encoder, base_texture, scene_tex, frame_width, frame_height);

    try replayFramePaintLoad(
        encoder,
        scene_view,
        scene_tex,
        backdrop_tex,
        backdrop_bg,
        gpu_ui,
        &frame,
        uploaded,
        scale_factor,
        frame_width,
        frame_height,
    );

    const attachment = wgpu.ColorAttachment{
        .view = color_view,
        .load_op = .load,
        .store_op = .store,
        .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
    };
    const pass_descriptor = wgpu.RenderPassDescriptor{
        .color_attachment_count = 1,
        .color_attachments = @ptrCast(&attachment),
    };

    const pass = encoder.beginRenderPass(&pass_descriptor) orelse return error.WgpuRenderPassUnavailable;
    defer pass.release();
    try blitSceneToSwapchain(device, queue, pass, gpu_ui, frame_width, frame_height);
    pass.end();
}

/// Like renderFrame but uses load_op=.load on the color attachment (no clear).
/// Intended for use after a 3D pass has already cleared and drawn to the same view.
/// The caller owns the encoder and color_view; this function only adds a render pass to the encoder.
pub fn renderFrameOverlay(
    encoder: *wgpu.CommandEncoder,
    color_view: *wgpu.TextureView,
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    paint_list: zg.PaintList,
    frame_width: u32,
    frame_height: u32,
    scale_factor: f32,
    frame_index: u32,
) !void {
    _ = frame_index;

    const scratch_allocator = gpu_ui.scratch.allocator();
    defer _ = gpu_ui.scratch.reset(.retain_capacity);

    var frame = FramePaint{};
    defer frame.deinit(scratch_allocator);

    // Planned BEFORE buildFrameOps — see renderFrame.
    const alloc = planSceneAlloc(gpu_ui, frame_width, frame_height);

    try buildFrameOps(
        device,
        queue,
        scratch_allocator,
        &frame,
        gpu_ui,
        paint_list,
        null,
        @floatFromInt(frame_width),
        @floatFromInt(frame_height),
        @floatFromInt(alloc.w),
        @floatFromInt(alloc.h),
        scale_factor,
    );

    gpu_ui.text.uploadAtlasIfDirty(queue);
    gpu_ui.text.uploadColorAtlasIfDirty(queue);

    // Overlay mode cannot copy the already-rendered external color_view into a
    // backdrop because this API receives only a TextureView. We still use the
    // offscreen scene path so transparent UI composites correctly. Glass samples
    // previously drawn UI in this overlay pass, not the caller's 3D scene.
    try ensureSceneTexture(gpu_ui, device, frame_width, frame_height, alloc);
    if (frameHasBackdropOp(&frame)) try ensureBackdropTexture(gpu_ui, device);
    const scene_view = gpu_ui.scene_texture_view orelse return error.WgpuTextureViewUnavailable;
    const scene_tex = gpu_ui.scene_texture orelse return error.WgpuTextureUnavailable;
    const backdrop_tex = gpu_ui.backdrop_texture;
    const backdrop_bg = gpu_ui.backdrop_bind_group;

    const uploaded = try uploadFrameVertices(device, queue, gpu_ui, &frame);

    try replayFramePaint(
        encoder,
        scene_view,
        scene_tex,
        backdrop_tex,
        backdrop_bg,
        gpu_ui,
        &frame,
        uploaded,
        .{ .r = 0, .g = 0, .b = 0, .a = 0 },
        scale_factor,
        frame_width,
        frame_height,
    );

    const attachment = wgpu.ColorAttachment{
        .view = color_view,
        .load_op = .load,
        .store_op = .store,
        .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
    };
    const pass_descriptor = wgpu.RenderPassDescriptor{
        .color_attachment_count = 1,
        .color_attachments = @ptrCast(&attachment),
    };

    const pass = encoder.beginRenderPass(&pass_descriptor) orelse return error.WgpuRenderPassUnavailable;
    defer pass.release();
    try blitSceneToSwapchain(device, queue, pass, gpu_ui, frame_width, frame_height);
    pass.end();
}

/// Render a paint list into an arbitrary texture view (for render textures).
///
/// Creates temporary scene/backdrop textures so the main gpu_ui shared textures are
/// not touched. Submits its own command encoder immediately — must be called BEFORE
/// the main frame encoder is submitted to preserve GPU queue ordering.
pub fn renderToTexture(
    device: *wgpu.Device,
    queue: *wgpu.Queue,
    gpu_ui: *GpuUi,
    paint_list: zg.PaintList,
    dst_view: *wgpu.TextureView,
    width: u32,
    height: u32,
    scale_factor: f32,
) !void {
    const scratch_allocator = gpu_ui.scratch.allocator();
    defer _ = gpu_ui.scratch.reset(.retain_capacity);

    var frame = FramePaint{};
    defer frame.deinit(scratch_allocator);

    try buildFrameOps(device, queue, scratch_allocator, &frame, gpu_ui, paint_list, null, @floatFromInt(width), @floatFromInt(height), @floatFromInt(width), @floatFromInt(height), scale_factor);

    gpu_ui.text.uploadAtlasIfDirty(queue);
    gpu_ui.text.uploadColorAtlasIfDirty(queue);

    // Persistent scratch scene target, recreated only when the RT size changes.
    try ensureRtSceneTexture(gpu_ui, device, width, height);
    const scene_tex = gpu_ui.rt_scene_texture orelse return error.RtSceneTexFailed;
    const scene_view = gpu_ui.rt_scene_view orelse return error.RtSceneViewFailed;

    // Temporary backdrop trio, created only when this RT frame actually contains a
    // backdrop-sampling (glass / shader-effect) op — plain RT content skips the allocation.
    var backdrop_tex: ?*wgpu.Texture = null;
    defer if (backdrop_tex) |t| t.release();
    var backdrop_view: ?*wgpu.TextureView = null;
    defer if (backdrop_view) |v| v.release();
    var backdrop_bg: ?*wgpu.BindGroup = null;
    defer if (backdrop_bg) |bg| bg.release();
    if (frameHasBackdropOp(&frame)) {
        backdrop_tex = device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("zigote rt backdrop"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = gpu_ui.surface_format,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return error.RtBackdropTexFailed;
        backdrop_view = backdrop_tex.?.createView(null) orelse return error.RtBackdropViewFailed;
        backdrop_bg = createImageBindGroup(device, gpu_ui.backdrop_bgl, backdrop_view.?, gpu_ui.backdrop_sampler) orelse return error.RtBackdropBgFailed;
    }

    // Blit bind group: scene_tex → dst_view (persisted alongside the scratch scene texture)
    const blit_bg = gpu_ui.rt_blit_bind_group orelse return error.RtBlitBgFailed;

    const uploaded = try uploadFrameVertices(device, queue, gpu_ui, &frame);

    const encoder = device.createCommandEncoder(&.{
        .label = wgpu.StringView.fromSlice("zigote rt encoder"),
    }) orelse return error.RtEncoderFailed;
    defer encoder.release();

    try replayFramePaint(
        encoder,
        scene_view,
        scene_tex,
        backdrop_tex,
        backdrop_bg,
        gpu_ui,
        &frame,
        uploaded,
        .{ .r = 0, .g = 0, .b = 0, .a = 0 },
        scale_factor,
        width,
        height,
    );

    // Blit scene_tex → dst_view. Same constant fullscreen quad as blitSceneToSwapchain, sharing
    // the persistent blit_vertex_buffer (see ensureBlitQuad).
    {
        const attachment = wgpu.ColorAttachment{
            .view = dst_view,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 },
        };
        const pass = encoder.beginRenderPass(&.{
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&attachment),
        }) orelse return error.RtBlitPassFailed;
        defer pass.release();
        try drawBlit(device, queue, pass, gpu_ui, blit_bg, width, height, 1.0, 1.0);
        pass.end();
    }

    const rt_cmd = encoder.finish(&.{}) orelse return error.RtCmdFailed;
    defer rt_cmd.release();
    queue.submit(&.{rt_cmd});
}

pub const Backend = struct {
    pub const name = "wgpu";
};

pub const RendererPlan = struct {
    batches_rectangles: bool = true,
    batches_glyphs: bool = true,
    supports_custom_passes: bool = true,
};

test "wgpu binding is visible" {
    try std.testing.expectEqualStrings("wgpu", Backend.name);
    _ = wgpu.StringView;
}

test "vertexRangeAabb maps NDC vertex bounds to device pixels" {
    const fw: f32 = 512;
    const fh: f32 = 512;
    // Quad covering device px 128..256 × 64..192 on a 512×512 frame (corners TL,TR,BL,BR).
    const verts = [_]ImageVertex{
        .{ .position = .{ -0.5, 0.75 }, .uv = .{ 0, 0 }, .color = .{ 255, 255, 255, 255 } },
        .{ .position = .{ 0.0, 0.75 }, .uv = .{ 1, 0 }, .color = .{ 255, 255, 255, 255 } },
        .{ .position = .{ -0.5, 0.25 }, .uv = .{ 0, 1 }, .color = .{ 255, 255, 255, 255 } },
        .{ .position = .{ 0.0, 0.25 }, .uv = .{ 1, 1 }, .color = .{ 255, 255, 255, 255 } },
    };
    const aabb = vertexRangeAabb(ImageVertex, &verts, fw, fh).?;
    try std.testing.expectEqual(@as(f32, 128), aabb.x);
    try std.testing.expectEqual(@as(f32, 64), aabb.y);
    try std.testing.expectEqual(@as(f32, 128), aabb.width);
    try std.testing.expectEqual(@as(f32, 128), aabb.height);
    try std.testing.expect(vertexRangeAabb(ImageVertex, verts[0..0], fw, fh) == null);
}

test "aabbHitsDamage keeps touching ops and culls distant ones" {
    const aabb = zg.Rect{ .x = 128, .y = 64, .width = 128, .height = 128 };
    // Far away on either axis — culled.
    try std.testing.expect(!aabbHitsDamage(aabb, .{ .x = 300, .y = 64, .width = 50, .height = 50 }, 1.0));
    try std.testing.expect(!aabbHitsDamage(aabb, .{ .x = 128, .y = 300, .width = 50, .height = 50 }, 1.0));
    // Overlapping and edge-touching (inside the 1 px slack) — kept.
    try std.testing.expect(aabbHitsDamage(aabb, .{ .x = 200, .y = 100, .width = 50, .height = 50 }, 1.0));
    try std.testing.expect(aabbHitsDamage(aabb, .{ .x = 256, .y = 64, .width = 50, .height = 50 }, 1.0));
    // scale_factor applies to the damage region (logical px), not the device-space AABB.
    try std.testing.expect(!aabbHitsDamage(aabb, .{ .x = 130, .y = 32, .width = 50, .height = 50 }, 2.0));
    try std.testing.expect(aabbHitsDamage(aabb, .{ .x = 60, .y = 32, .width = 50, .height = 50 }, 2.0));
    // Negative origin SHRINKS the effective box (mirrors applyScissor). A shifted box would reach
    // x=100 here and wrongly keep the op; the true extent is [0, 60) — culled.
    try std.testing.expect(!aabbHitsDamage(aabb, .{ .x = -40, .y = 64, .width = 100, .height = 50 }, 1.0));
    try std.testing.expect(aabbHitsDamage(aabb, .{ .x = -40, .y = 64, .width = 170, .height = 50 }, 1.0));
}
