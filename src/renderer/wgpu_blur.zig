//! Two-pass separable Gaussian blur via WebGPU compute shaders.
//! Source texture must have `texture_binding`. Intermediate and destination
//! textures must have `texture_binding | storage_binding` and use `rgba8unorm`
//! format (sRGB formats do not support storage writes in WebGPU).

const std = @import("std");
const wgpu = @import("wgpu");

pub const BlurParams = extern struct {
    sigma: f32,
    width: u32,
    height: u32,
    _pad: u32 = 0,
};

// ── WGSL: horizontal pass (blur along X) ─────────────────────────────────────

const h_wgsl =
    \\struct BlurParams { sigma: f32, width: u32, height: u32, _pad: u32 }
    \\@group(0) @binding(0) var src  : texture_2d<f32>;
    \\@group(0) @binding(1) var<uniform> p : BlurParams;
    \\@group(0) @binding(2) var dst  : texture_storage_2d<rgba8unorm, write>;
    \\@compute @workgroup_size(8, 8, 1)
    \\fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    \\    if (id.x >= p.width || id.y >= p.height) { return; }
    \\    let sigma  = max(p.sigma, 0.001);
    \\    let radius = i32(ceil(3.0 * sigma));
    \\    // Incremental Gaussian (GPU Gems 3 §40): w_i = w_{i-1} * r^(2i-1), r = exp(-1/(2σ²)) —
    \\    // two multiplies per tap instead of an exp(), identical weights.
    \\    let r  = exp(-1.0 / (2.0 * sigma * sigma));
    \\    let r2 = r * r;
    \\    var color = textureLoad(src, vec2<i32>(id.xy), 0);
    \\    var w_sum = 1.0;
    \\    var w     = 1.0;
    \\    var ratio = r;
    \\    for (var i = 1; i <= radius; i++) {
    \\        w = w * ratio;
    \\        ratio = ratio * r2;
    \\        let xl = clamp(i32(id.x) - i, 0, i32(p.width) - 1);
    \\        let xr = clamp(i32(id.x) + i, 0, i32(p.width) - 1);
    \\        color += (textureLoad(src, vec2<i32>(xl, i32(id.y)), 0)
    \\                + textureLoad(src, vec2<i32>(xr, i32(id.y)), 0)) * w;
    \\        w_sum += 2.0 * w;
    \\    }
    \\    textureStore(dst, vec2<i32>(id.xy), color / w_sum);
    \\}
;

// ── WGSL: vertical pass (blur along Y) ───────────────────────────────────────

const v_wgsl =
    \\struct BlurParams { sigma: f32, width: u32, height: u32, _pad: u32 }
    \\@group(0) @binding(0) var src  : texture_2d<f32>;
    \\@group(0) @binding(1) var<uniform> p : BlurParams;
    \\@group(0) @binding(2) var dst  : texture_storage_2d<rgba8unorm, write>;
    \\@compute @workgroup_size(8, 8, 1)
    \\fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    \\    if (id.x >= p.width || id.y >= p.height) { return; }
    \\    let sigma  = max(p.sigma, 0.001);
    \\    let radius = i32(ceil(3.0 * sigma));
    \\    // Same incremental-weight recurrence as the horizontal pass.
    \\    let r  = exp(-1.0 / (2.0 * sigma * sigma));
    \\    let r2 = r * r;
    \\    var color = textureLoad(src, vec2<i32>(id.xy), 0);
    \\    var w_sum = 1.0;
    \\    var w     = 1.0;
    \\    var ratio = r;
    \\    for (var i = 1; i <= radius; i++) {
    \\        w = w * ratio;
    \\        ratio = ratio * r2;
    \\        let yt = clamp(i32(id.y) - i, 0, i32(p.height) - 1);
    \\        let yb = clamp(i32(id.y) + i, 0, i32(p.height) - 1);
    \\        color += (textureLoad(src, vec2<i32>(i32(id.x), yt), 0)
    \\                + textureLoad(src, vec2<i32>(i32(id.x), yb), 0)) * w;
    \\        w_sum += 2.0 * w;
    \\    }
    \\    textureStore(dst, vec2<i32>(id.xy), color / w_sum);
    \\}
;

// ── GaussianBlur ──────────────────────────────────────────────────────────────

pub const GaussianBlur = struct {
    h_pipeline: *wgpu.ComputePipeline,
    v_pipeline: *wgpu.ComputePipeline,
    bgl: *wgpu.BindGroupLayout,
    params_buf: *wgpu.Buffer,
    device: *wgpu.Device,
    queue: *wgpu.Queue,

    pub fn init(device: *wgpu.Device, queue: *wgpu.Queue) !GaussianBlur {
        // ── Bind group layout: src texture, uniform params, dst storage texture ──
        const bgl_entries = [_]wgpu.BindGroupLayoutEntry{
            .{
                .binding = 0,
                .visibility = wgpu.ShaderStages.compute,
                .texture = .{
                    .sample_type = .float,
                    .view_dimension = .@"2d",
                    .multisampled = 0,
                },
            },
            .{
                .binding = 1,
                .visibility = wgpu.ShaderStages.compute,
                .buffer = .{ .type = .uniform },
            },
            .{
                .binding = 2,
                .visibility = wgpu.ShaderStages.compute,
                .storage_texture = .{
                    .access = .write_only,
                    .format = .rgba8_unorm,
                    .view_dimension = .@"2d",
                },
            },
        };

        const bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote blur bgl"),
            .entry_count = bgl_entries.len,
            .entries = &bgl_entries,
        }) orelse return error.BlurBglFailed;
        errdefer bgl.release();

        const layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote blur layout"),
            .bind_group_layout_count = 1,
            .bind_group_layouts = @ptrCast(&bgl),
        }) orelse return error.BlurLayoutFailed;
        defer layout.release();

        const params_buf = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("zigote blur params"),
            .size = @sizeOf(BlurParams),
            .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
        }) orelse return error.BlurBufFailed;
        errdefer params_buf.release();

        // ── H-pass pipeline ──────────────────────────────────────────────────
        var h_desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote blur-h",
            .code = h_wgsl,
        });
        const h_shader = device.createShaderModule(&h_desc) orelse return error.BlurShaderFailed;
        defer h_shader.release();

        const h_pipeline = device.createComputePipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote blur-h pipeline"),
            .layout = layout,
            .compute = .{
                .module = h_shader,
                .entry_point = wgpu.StringView.fromSlice("cs_main"),
            },
        }) orelse return error.BlurHPipelineFailed;
        errdefer h_pipeline.release();

        // ── V-pass pipeline ──────────────────────────────────────────────────
        var v_desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote blur-v",
            .code = v_wgsl,
        });
        const v_shader = device.createShaderModule(&v_desc) orelse return error.BlurShaderFailed;
        defer v_shader.release();

        const v_pipeline = device.createComputePipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote blur-v pipeline"),
            .layout = layout,
            .compute = .{
                .module = v_shader,
                .entry_point = wgpu.StringView.fromSlice("cs_main"),
            },
        }) orelse return error.BlurVPipelineFailed;
        errdefer v_pipeline.release();

        return .{
            .h_pipeline = h_pipeline,
            .v_pipeline = v_pipeline,
            .bgl = bgl,
            .params_buf = params_buf,
            .device = device,
            .queue = queue,
        };
    }

    pub fn deinit(self: *GaussianBlur) void {
        self.v_pipeline.release();
        self.h_pipeline.release();
        self.params_buf.release();
        self.bgl.release();
    }

    /// Blur `src` into `dst` using a two-pass separable Gaussian.
    ///
    /// - `src`  : texture with `texture_binding`; any format readable as `texture_2d<f32>`.
    /// - `temp` : `rgba8unorm` texture with `texture_binding | storage_binding`.
    ///            Used as ping-pong intermediate (H-pass output / V-pass input).
    /// - `dst`  : `rgba8unorm` texture with `storage_binding`. Receives the final result.
    /// - `sigma`: standard deviation in pixels. Values ≤ 0 are clamped to 0.001.
    ///
    /// The encoder must not be submitted until after the wgpu device queue has been
    /// flushed (i.e., the caller submits as normal after recording).
    pub fn dispatch(
        self: *GaussianBlur,
        encoder: *wgpu.CommandEncoder,
        src: *wgpu.Texture,
        temp: *wgpu.Texture,
        dst: *wgpu.Texture,
        sigma: f32,
    ) !void {
        const w = src.getWidth();
        const h = src.getHeight();

        const p = BlurParams{ .sigma = @max(sigma, 0.001), .width = w, .height = h };
        self.queue.writeBuffer(self.params_buf, 0, @ptrCast(&p), @sizeOf(BlurParams));

        const src_view = src.createView(null) orelse return error.BlurViewFailed;
        defer src_view.release();
        const temp_view = temp.createView(null) orelse return error.BlurViewFailed;
        defer temp_view.release();
        const dst_view = dst.createView(null) orelse return error.BlurViewFailed;
        defer dst_view.release();

        // H-pass: src → temp
        {
            const bg = self.device.createBindGroup(&.{
                .layout = self.bgl,
                .entry_count = 3,
                .entries = &[_]wgpu.BindGroupEntry{
                    .{ .binding = 0, .texture_view = src_view },
                    .{ .binding = 1, .buffer = self.params_buf, .size = @sizeOf(BlurParams) },
                    .{ .binding = 2, .texture_view = temp_view },
                },
            }) orelse return error.BlurBgFailed;
            defer bg.release();

            const pass = encoder.beginComputePass(null) orelse return error.BlurPassFailed;
            defer pass.release();
            pass.setPipeline(self.h_pipeline);
            pass.setBindGroup(0, bg, 0, null);
            pass.dispatchWorkgroups((w + 7) / 8, (h + 7) / 8, 1);
            pass.end();
        }

        // V-pass: temp → dst
        {
            const bg = self.device.createBindGroup(&.{
                .layout = self.bgl,
                .entry_count = 3,
                .entries = &[_]wgpu.BindGroupEntry{
                    .{ .binding = 0, .texture_view = temp_view },
                    .{ .binding = 1, .buffer = self.params_buf, .size = @sizeOf(BlurParams) },
                    .{ .binding = 2, .texture_view = dst_view },
                },
            }) orelse return error.BlurBgFailed;
            defer bg.release();

            const pass = encoder.beginComputePass(null) orelse return error.BlurPassFailed;
            defer pass.release();
            pass.setPipeline(self.v_pipeline);
            pass.setBindGroup(0, bg, 0, null);
            pass.dispatchWorkgroups((w + 7) / 8, (h + 7) / 8, 1);
            pass.end();
        }
    }
};
