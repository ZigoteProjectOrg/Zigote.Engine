//! GPU particle billboard renderer for the VFX system, with two feeds:
//!
//!  • CPU feed (`upload`): the host CPU-simulates particles and uploads a flat {pos.xyz, size, rot,
//!    rgba} buffer (9 f32 = 36 B/particle) per emitter; we draw them as camera-facing billboards.
//!  • GPU feed (`computeEmit` + `computeStep`): a compute kernel simulates particles in a persistent
//!    GPU storage buffer and writes the same 9-f32 instance buffer, which the billboard pass then draws
//!    (dead particles wrote size 0 → degenerate quad). The instance buffer is bound storage (compute
//!    write) + vertex (render read) on the same buffer — wgpu barriers between the compute and render
//!    passes on the shared encoder.
//!
//! Both feed the one billboard pass (additive / premultiplied-alpha), drawn in the geometry pass after
//! the transparent meshes (depth-test opaque, no depth write), with the G-buffer MRT targets masked off.
//!
//! Failure-isolated + lazy: pipelines are created on first use and any failure (e.g. a shader naga
//! rejects) sets `failed` and disables the system rather than killing the renderer — a no-op until used.

const std = @import("std");
const wgpu = @import("wgpu");
const shaders3d = @import("wgpu_3d_shaders.zig");

pub const ParticleSystem = struct {
    // ── Billboard render (shared by both feeds) ──────────────────────────────
    initialized: bool = false,
    failed: bool = false,
    shader: ?*wgpu.ShaderModule = null,
    layout: ?*wgpu.PipelineLayout = null,
    pipe_additive: ?*wgpu.RenderPipeline = null,
    pipe_alpha: ?*wgpu.RenderPipeline = null,

    // ── CPU feed ─────────────────────────────────────────────────────────────
    batches: std.AutoHashMapUnmanaged(u64, Batch) = .{},

    // ── GPU compute feed ─────────────────────────────────────────────────────
    compute_ready: bool = false,
    compute_shader: ?*wgpu.ShaderModule = null,
    compute_bgl: ?*wgpu.BindGroupLayout = null,
    compute_layout: ?*wgpu.PipelineLayout = null,
    compute_pipe: ?*wgpu.ComputePipeline = null,
    gpu_batches: std.AutoHashMapUnmanaged(u64, GpuBatch) = .{},

    alloc: ?std.mem.Allocator = null,

    const STRIDE_BYTES: u64 = 36; // instance: 9 f32 (pos.xyz, size, rot, rgba)
    const STATE_BYTES: u64 = 80; // GPU particle state: 5 vec4 (pa, vl, sc, sr, sd)
    const PARAMS_FLOATS: u32 = 112; // emitter params UBO: 28 vec4
    const PARAMS_BYTES: u64 = PARAMS_FLOATS * 4;

    const Batch = struct {
        gpu: ?*wgpu.Buffer = null,
        capacity: u32 = 0,
        count: u32 = 0,
        blend: u32 = 0,
    };

    const GpuBatch = struct {
        state: ?*wgpu.Buffer = null, // persistent particle state (storage)
        inst: ?*wgpu.Buffer = null, // instance buffer (storage + vertex)
        params: ?*wgpu.Buffer = null, // emitter params UBO
        counter: ?*wgpu.Buffer = null, // atomic spawn counter (storage)
        bg: ?*wgpu.BindGroup = null,
        capacity: u32 = 0,
        blend: u32 = 0,
        dispatch: bool = false, // host emitted to it this frame → run the kernel

        fn release(b: *GpuBatch) void {
            if (b.bg) |x| x.release();
            if (b.state) |x| x.release();
            if (b.inst) |x| x.release();
            if (b.params) |x| x.release();
            if (b.counter) |x| x.release();
            b.* = .{};
        }
    };

    pub fn deinit(self: *ParticleSystem) void {
        if (self.alloc) |a| {
            var it = self.batches.valueIterator();
            while (it.next()) |b| if (b.gpu) |buf| buf.release();
            self.batches.deinit(a);
            var git = self.gpu_batches.valueIterator();
            while (git.next()) |b| b.release();
            self.gpu_batches.deinit(a);
        }
        if (self.pipe_additive) |p| p.release();
        if (self.pipe_alpha) |p| p.release();
        if (self.layout) |l| l.release();
        if (self.shader) |s| s.release();
        if (self.compute_pipe) |p| p.release();
        if (self.compute_layout) |l| l.release();
        if (self.compute_bgl) |l| l.release();
        if (self.compute_shader) |s| s.release();
        self.* = .{};
    }

    /// Remove one emitter node's batch (CPU and/or GPU feed).
    pub fn clearNode(self: *ParticleSystem, node: u64) void {
        if (self.batches.fetchRemove(node)) |kv| {
            if (kv.value.gpu) |b| b.release();
        }
        if (self.gpu_batches.fetchRemove(node)) |kv| {
            var v = kv.value;
            v.release();
        }
    }

    /// Drop every batch, both feeds (play stop).
    pub fn clearAll(self: *ParticleSystem) void {
        if (self.alloc) |a| {
            var it = self.batches.valueIterator();
            while (it.next()) |b| if (b.gpu) |buf| buf.release();
            self.batches.clearAndFree(a);
            var git = self.gpu_batches.valueIterator();
            while (git.next()) |b| b.release();
            self.gpu_batches.clearAndFree(a);
        }
    }

    // ── CPU feed: upload a flat instance buffer ──────────────────────────────
    pub fn upload(
        self: *ParticleSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        alloc: std.mem.Allocator,
        node: u64,
        data: [*c]const f32,
        count: u32,
        blend: u32,
    ) void {
        if (self.failed) return;
        self.alloc = alloc;
        const gop = self.batches.getOrPut(alloc, node) catch return;
        if (!gop.found_existing) gop.value_ptr.* = .{};
        const batch = gop.value_ptr;
        batch.blend = blend;
        batch.count = count;
        if (count == 0) return;

        const needed: u64 = @as(u64, count) * STRIDE_BYTES;
        if (batch.gpu == null or batch.capacity < count) {
            if (batch.gpu) |b| b.release();
            const new_cap = std.math.ceilPowerOfTwo(u32, @max(count, 256)) catch count;
            batch.gpu = device.createBuffer(&.{
                .label = wgpu.StringView.fromSlice("particle instance buffer"),
                .usage = wgpu.BufferUsages.vertex | wgpu.BufferUsages.copy_dst,
                .size = @as(u64, new_cap) * STRIDE_BYTES,
            }) orelse {
                batch.gpu = null;
                batch.capacity = 0;
                return;
            };
            batch.capacity = new_cap;
        }
        const bytes: [*]const u8 = @ptrCast(data);
        queue.writeBuffer(batch.gpu.?, 0, bytes, @intCast(needed));
    }

    // ── GPU feed: register/update an emitter's params for the compute kernel ──
    pub fn computeEmit(
        self: *ParticleSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        alloc: std.mem.Allocator,
        node: u64,
        params: [*c]const f32,
        param_count: u32,
        capacity: u32,
        blend: u32,
    ) void {
        if (self.failed or capacity == 0) return;
        self.alloc = alloc;
        const gop = self.gpu_batches.getOrPut(alloc, node) catch return;
        if (!gop.found_existing) gop.value_ptr.* = .{};
        const b = gop.value_ptr;
        b.blend = blend;
        ensureGpuBuffers(device, b, capacity);
        if (b.state == null or b.params == null) return; // allocation failed → skip
        const n = @min(param_count, PARAMS_FLOATS);
        const bytes: [*]const u8 = @ptrCast(params);
        queue.writeBuffer(b.params.?, 0, bytes, @as(usize, n) * 4);
        b.dispatch = true;
    }

    /// Dispatch the compute kernel for every GPU emitter that was emitted to this frame. Records onto the
    /// shared scene encoder BEFORE the geometry render pass reads the instance buffers.
    pub fn computeStep(self: *ParticleSystem, device: *wgpu.Device, queue: *wgpu.Queue, encoder: *wgpu.CommandEncoder) void {
        if (self.failed or self.gpu_batches.count() == 0) return;
        if (!self.compute_ready) self.ensureCompute(device);
        if (self.failed or !self.compute_ready) return;

        var it = self.gpu_batches.valueIterator();
        while (it.next()) |b| {
            if (!b.dispatch or b.state == null) continue;
            if (b.bg == null) {
                b.bg = self.makeComputeBg(device, b);
                if (b.bg == null) continue;
            }

            const zero: u32 = 0;
            queue.writeBuffer(b.counter.?, 0, @ptrCast(&zero), 4); // reset spawn budget

            const pass = encoder.beginComputePass(null) orelse continue;
            pass.setPipeline(self.compute_pipe.?);
            pass.setBindGroup(0, b.bg.?, 0, null);
            pass.dispatchWorkgroups((b.capacity + 63) / 64, 1, 1);
            pass.end();
            pass.release();
            b.dispatch = false;
        }
    }

    /// Draw all batches (both feeds) into the open geometry render pass. Lazily creates pipelines.
    pub fn render(
        self: *ParticleSystem,
        device: *wgpu.Device,
        camera_bgl: *wgpu.BindGroupLayout,
        hdr_format: wgpu.TextureFormat,
        msaa: u32,
        pass: *wgpu.RenderPassEncoder,
        camera_bg: *wgpu.BindGroup,
    ) void {
        if (self.failed) return;
        if (self.batches.count() == 0 and self.gpu_batches.count() == 0) return;
        if (!self.initialized) self.ensureInit(device, camera_bgl, hdr_format, msaa);
        if (self.failed or !self.initialized) return;

        var it = self.batches.valueIterator();
        while (it.next()) |batch| {
            if (batch.count == 0 or batch.gpu == null) continue;
            const pipe = if (batch.blend == 0) self.pipe_additive else self.pipe_alpha;
            if (pipe == null) continue;
            pass.setPipeline(pipe.?);
            pass.setBindGroup(0, camera_bg, 0, null);
            pass.setVertexBuffer(0, batch.gpu.?, 0, wgpu.WGPU_WHOLE_SIZE);
            pass.draw(6, batch.count, 0, 0);
        }

        // GPU compute emitters: draw `capacity` instances; dead particles wrote size 0 (culled in VS).
        var git = self.gpu_batches.valueIterator();
        while (git.next()) |b| {
            if (b.inst == null or b.capacity == 0) continue;
            const pipe = if (b.blend == 0) self.pipe_additive else self.pipe_alpha;
            if (pipe == null) continue;
            pass.setPipeline(pipe.?);
            pass.setBindGroup(0, camera_bg, 0, null);
            pass.setVertexBuffer(0, b.inst.?, 0, wgpu.WGPU_WHOLE_SIZE);
            pass.draw(6, b.capacity, 0, 0);
        }
    }

    fn ensureGpuBuffers(device: *wgpu.Device, b: *GpuBatch, capacity: u32) void {
        if (b.capacity == capacity and b.state != null) return;
        if (b.bg) |x| {
            x.release();
            b.bg = null;
        }
        if (b.state) |x| x.release();
        if (b.inst) |x| x.release();
        if (b.counter) |x| x.release();
        b.state = null;
        b.inst = null;
        b.counter = null;
        b.capacity = 0;

        const cap = @max(capacity, 1);
        // Fresh wgpu buffers are zero-initialised (WebGPU spec), so every state slot starts dead.
        b.state = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("particle state buffer"),
            .usage = wgpu.BufferUsages.storage | wgpu.BufferUsages.copy_dst,
            .size = @as(u64, cap) * STATE_BYTES,
        }) orelse return;
        b.inst = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("particle gpu instance buffer"),
            .usage = wgpu.BufferUsages.storage | wgpu.BufferUsages.vertex | wgpu.BufferUsages.copy_dst,
            .size = @as(u64, cap) * STRIDE_BYTES,
        }) orelse return;
        b.counter = device.createBuffer(&.{
            .label = wgpu.StringView.fromSlice("particle spawn counter"),
            .usage = wgpu.BufferUsages.storage | wgpu.BufferUsages.copy_dst,
            .size = 4,
        }) orelse return;
        if (b.params == null) {
            b.params = device.createBuffer(&.{
                .label = wgpu.StringView.fromSlice("particle params"),
                .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
                .size = PARAMS_BYTES,
            }) orelse return;
        }
        b.capacity = cap;
    }

    fn makeComputeBg(self: *ParticleSystem, device: *wgpu.Device, b: *GpuBatch) ?*wgpu.BindGroup {
        return device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("particle compute bg"),
            .layout = self.compute_bgl.?,
            .entry_count = 4,
            .entries = &[_]wgpu.BindGroupEntry{
                .{ .binding = 0, .buffer = b.state.?, .size = @as(u64, b.capacity) * STATE_BYTES },
                .{ .binding = 1, .buffer = b.inst.?, .size = @as(u64, b.capacity) * STRIDE_BYTES },
                .{ .binding = 2, .buffer = b.params.?, .size = PARAMS_BYTES },
                .{ .binding = 3, .buffer = b.counter.?, .size = 4 },
            },
        });
    }

    fn ensureCompute(self: *ParticleSystem, device: *wgpu.Device) void {
        if (self.compute_ready or self.failed) return;

        var desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote particle compute",
            .code = shaders3d.particle_compute_source,
        });
        const shader = device.createShaderModule(&desc) orelse {
            self.failed = true;
            return;
        };
        self.compute_shader = shader;

        const entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.compute, .buffer = .{ .type = .storage } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.compute, .buffer = .{ .type = .storage } },
            .{ .binding = 2, .visibility = wgpu.ShaderStages.compute, .buffer = .{ .type = .uniform } },
            .{ .binding = 3, .visibility = wgpu.ShaderStages.compute, .buffer = .{ .type = .storage } },
        };
        const bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("particle compute bgl"),
            .entry_count = entries.len,
            .entries = &entries,
        }) orelse {
            self.failed = true;
            return;
        };
        self.compute_bgl = bgl;

        const bgls = [_]*wgpu.BindGroupLayout{bgl};
        const layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("particle compute layout"),
            .bind_group_layout_count = bgls.len,
            .bind_group_layouts = &bgls,
        }) orelse {
            self.failed = true;
            return;
        };
        self.compute_layout = layout;

        self.compute_pipe = device.createComputePipeline(&.{
            .label = wgpu.StringView.fromSlice("particle compute pipeline"),
            .layout = layout,
            .compute = .{ .module = shader, .entry_point = wgpu.StringView.fromSlice("cs_particle") },
        }) orelse {
            self.failed = true;
            return;
        };
        self.compute_ready = true;
    }

    fn ensureInit(
        self: *ParticleSystem,
        device: *wgpu.Device,
        camera_bgl: *wgpu.BindGroupLayout,
        hdr_format: wgpu.TextureFormat,
        msaa: u32,
    ) void {
        if (self.initialized or self.failed) return;

        var desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote particle shader",
            .code = shaders3d.particle_shader_source,
        });
        const shader = device.createShaderModule(&desc) orelse {
            self.failed = true;
            return;
        };
        self.shader = shader;

        const bgls = [_]*wgpu.BindGroupLayout{camera_bgl};
        const layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("zigote particle layout"),
            .bind_group_layout_count = bgls.len,
            .bind_group_layouts = &bgls,
        }) orelse {
            self.failed = true;
            return;
        };
        self.layout = layout;

        self.pipe_additive = createPipeline(device, layout, shader, hdr_format, msaa, true) orelse {
            self.failed = true;
            return;
        };
        self.pipe_alpha = createPipeline(device, layout, shader, hdr_format, msaa, false) orelse {
            self.failed = true;
            return;
        };
        self.initialized = true;
    }

    fn createPipeline(
        device: *wgpu.Device,
        layout: *wgpu.PipelineLayout,
        shader: *wgpu.ShaderModule,
        hdr_format: wgpu.TextureFormat,
        msaa: u32,
        additive: bool,
    ) ?*wgpu.RenderPipeline {
        const blend = if (additive) wgpu.BlendState{
            .color = .{ .src_factor = .one, .dst_factor = .one, .operation = .add },
            .alpha = .{ .src_factor = .one, .dst_factor = .one, .operation = .add },
        } else wgpu.BlendState{
            .color = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha, .operation = .add },
            .alpha = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha, .operation = .add },
        };
        // Must declare ALL of the geometry pass's color attachments (attachment-state compatibility):
        // location 1 = view-position, 2 = view-normal, 3 = albedo G-buffer — all masked off.
        const targets = [_]wgpu.ColorTargetState{
            .{ .format = hdr_format, .blend = &blend, .write_mask = wgpu.ColorWriteMasks.all },
            .{ .format = hdr_format, .write_mask = wgpu.ColorWriteMasks.none },
            .{ .format = hdr_format, .write_mask = wgpu.ColorWriteMasks.none },
            .{ .format = hdr_format, .write_mask = wgpu.ColorWriteMasks.none },
        };

        const inst_attrs = [_]wgpu.VertexAttribute{
            .{ .format = .float32x3, .offset = 0, .shader_location = 0 }, // position
            .{ .format = .float32, .offset = 12, .shader_location = 1 }, // size
            .{ .format = .float32, .offset = 16, .shader_location = 2 }, // rotation
            .{ .format = .float32x4, .offset = 20, .shader_location = 3 }, // color
        };
        const inst_layout = wgpu.VertexBufferLayout{
            .array_stride = STRIDE_BYTES,
            .step_mode = .instance,
            .attribute_count = inst_attrs.len,
            .attributes = &inst_attrs,
        };

        const depth_state = wgpu.DepthStencilState{
            .format = .depth32_float,
            .depth_write_enabled = .false,
            .depth_compare = .less,
            .stencil_front = .{ .compare = .always, .fail_op = .keep, .depth_fail_op = .keep, .pass_op = .keep },
            .stencil_back = .{ .compare = .always, .fail_op = .keep, .depth_fail_op = .keep, .pass_op = .keep },
        };

        return device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("zigote particle pipeline"),
            .layout = layout,
            .vertex = .{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("vs_particle"),
                .buffer_count = 1,
                .buffers = @ptrCast(&inst_layout),
            },
            .primitive = .{ .topology = .triangle_list, .front_face = .ccw, .cull_mode = .none },
            .depth_stencil = &depth_state,
            .multisample = .{ .count = msaa, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = shader,
                .entry_point = wgpu.StringView.fromSlice("fs_particle"),
                .target_count = targets.len,
                .targets = &targets,
            },
        });
    }
};
