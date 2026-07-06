//! 2D sprite renderer: batched, textured, painter's-order instanced quads with per-sprite
//! UV rects (sprite sheets / atlases), tint, rotation and custom WGSL materials.
//!
//! Immediate-mode frame model: the host calls `begin` (two cameras: scene + overlay) once per
//! frame, then `draw` per pre-sorted batch (C# owns sorting layers / order-in-layer; native draws
//! batches strictly in submission order). Instances and material params accumulate in CPU arenas
//! and upload ONCE at first render (grow-at-upload — no mid-frame ring reallocation hazard).
//!
//! Two stages, hooked inside Gpu3d so both the immediate `zigote_render_3d` path and the render
//! graph path get them:
//!   • stage 0 "scene"   — drawn at the end of renderSceneGeometry into the (resolved,
//!     single-sample) HDR scene target: sprites participate in bloom/tonemap like world content.
//!   • stage 1 "overlay" — drawn at the end of renderPostProcess into the LDR destination:
//!     exact colors, no AgX shift, no TAA ghosting.
//! Neither pass has a depth attachment (2D = painter's algorithm); neither writes the G-buffer.
//!
//! Failure-isolated + lazy like wgpu_particles.zig: pipeline/shader failures disable the affected
//! shader (falling back to the default) or the whole system, never the frame.

const std = @import("std");
const wgpu = @import("wgpu");
const shaders3d = @import("wgpu_3d_shaders.zig");

const log = std.log.scoped(.sprites2d);

pub const SpriteSystem = struct {
    // Instance layout: 14 f32 per sprite — pos.xyz, rot, size.xy, uv0.xy, uv1.xy, rgba.
    pub const INSTANCE_FLOATS: u32 = 14;
    const STRIDE_BYTES: u64 = INSTANCE_FLOATS * 4;
    const PARAMS_SLOT: u32 = 256; // wgpu min uniform-buffer dynamic-offset alignment
    const PARAMS_FLOATS: u32 = 16;
    const CAMERA_BYTES: u64 = 80; // mat4 view_proj + vec4 viewport
    const MAX_TEXTURE_DIM: u32 = 8192;

    initialized: bool = false,
    failed: bool = false,

    default_shader: ?*wgpu.ShaderModule = null,
    camera_bgl: ?*wgpu.BindGroupLayout = null,
    texture_bgl: ?*wgpu.BindGroupLayout = null,
    params_bgl: ?*wgpu.BindGroupLayout = null,
    layout: ?*wgpu.PipelineLayout = null,

    // Pipeline cache: key = shader_id << 32 | target_format << 8 | blend << 1 | stage.
    pipelines: std.AutoHashMapUnmanaged(u64, *wgpu.RenderPipeline) = .{},

    // Four shared samplers: [wrap(0=clamp,1=repeat)][filter(0=nearest,1=linear)].
    samplers: [2][2]?*wgpu.Sampler = .{ .{ null, null }, .{ null, null } },

    textures: std.AutoHashMapUnmanaged(u32, Texture) = .{},
    next_texture: u32 = 1,
    white: ?Texture = null, // fallback / secondary-slot default (not in the map)

    shaders: std.AutoHashMapUnmanaged(u32, *wgpu.ShaderModule) = .{},
    next_shader: u32 = 1,

    // Camera UBOs (one per stage) + bind groups; written by begin() via the queue.
    scene_cam_buf: ?*wgpu.Buffer = null,
    overlay_cam_buf: ?*wgpu.Buffer = null,
    scene_cam_bg: ?*wgpu.BindGroup = null,
    overlay_cam_bg: ?*wgpu.BindGroup = null,

    // Frame state: CPU arenas packed by draw(), uploaded once by the first render stage.
    frame_active: bool = false,
    uploaded: bool = false,
    scene_cam: [20]f32 = [_]f32{0} ** 20,
    overlay_cam: [20]f32 = [_]f32{0} ** 20,
    inst_cpu: std.ArrayListUnmanaged(f32) = .empty,
    params_cpu: std.ArrayListUnmanaged(u8) = .empty,
    batches: std.ArrayListUnmanaged(Batch) = .empty,

    // GPU rings, grown at upload time (frame data is fully known then).
    inst_buf: ?*wgpu.Buffer = null,
    inst_cap_bytes: u64 = 0,
    params_buf: ?*wgpu.Buffer = null,
    params_cap_bytes: u64 = 0,
    params_bg: ?*wgpu.BindGroup = null,

    alloc: ?std.mem.Allocator = null,

    const Texture = struct {
        texture: *wgpu.Texture,
        view: *wgpu.TextureView,
        bind_group: *wgpu.BindGroup,
        width: u32,
        height: u32,

        fn release(t: *const Texture) void {
            t.bind_group.release();
            t.view.release();
            t.texture.release();
        }
    };

    const Batch = struct {
        texture: u32,
        texture2: u32,
        shader: u32,
        blend: u32,
        stage: u32,
        inst_byte_off: u64,
        count: u32,
        params_off: u32,
    };

    pub fn deinit(self: *SpriteSystem) void {
        if (self.alloc) |a| {
            var it = self.textures.valueIterator();
            while (it.next()) |t| t.release();
            self.textures.deinit(a);
            var sit = self.shaders.valueIterator();
            while (sit.next()) |m| m.*.release();
            self.shaders.deinit(a);
            var pit = self.pipelines.valueIterator();
            while (pit.next()) |p| p.*.release();
            self.pipelines.deinit(a);
            self.inst_cpu.deinit(a);
            self.params_cpu.deinit(a);
            self.batches.deinit(a);
        }
        if (self.white) |*w| w.release();
        for (&self.samplers) |*row| for (row) |*s| {
            if (s.*) |smp| smp.release();
            s.* = null;
        };
        if (self.scene_cam_bg) |x| x.release();
        if (self.overlay_cam_bg) |x| x.release();
        if (self.scene_cam_buf) |x| x.release();
        if (self.overlay_cam_buf) |x| x.release();
        if (self.params_bg) |x| x.release();
        if (self.params_buf) |x| x.release();
        if (self.inst_buf) |x| x.release();
        if (self.layout) |x| x.release();
        if (self.params_bgl) |x| x.release();
        if (self.texture_bgl) |x| x.release();
        if (self.camera_bgl) |x| x.release();
        if (self.default_shader) |x| x.release();
        self.* = .{};
    }

    // ── Textures ─────────────────────────────────────────────────────────────

    /// Create a sprite texture from tightly-packed RGBA8 pixels. Returns 0 on failure.
    /// filter: 0 nearest (pixel art) / 1 linear; wrap: 0 clamp / 1 repeat (tiling backgrounds);
    /// srgb: color textures 1, data textures (masks/LUTs for custom shaders) 0.
    pub fn createTexture(
        self: *SpriteSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        alloc: std.mem.Allocator,
        pixels: []const u8,
        width: u32,
        height: u32,
        filter: u32,
        srgb: u32,
        wrap: u32,
    ) u32 {
        if (self.failed) return 0;
        if (width == 0 or height == 0 or width > MAX_TEXTURE_DIM or height > MAX_TEXTURE_DIM) return 0;
        if (pixels.len < @as(usize, width) * height * 4) return 0;
        self.alloc = alloc;
        if (!self.initialized) self.ensureInit(device, queue);
        if (self.failed) return 0;

        const tex = self.makeTexture(device, queue, pixels, width, height, filter, srgb, wrap) orelse return 0;
        const id = self.next_texture;
        self.textures.put(alloc, id, tex) catch {
            tex.release();
            return 0;
        };
        self.next_texture += 1;
        return id;
    }

    pub fn destroyTexture(self: *SpriteSystem, id: u32) void {
        if (self.textures.fetchRemove(id)) |kv| kv.value.release();
    }

    fn makeTexture(
        self: *SpriteSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        pixels: []const u8,
        width: u32,
        height: u32,
        filter: u32,
        srgb: u32,
        wrap: u32,
    ) ?Texture {
        const texture = device.createTexture(&.{
            .label = wgpu.StringView.fromSlice("sprite texture"),
            .usage = wgpu.TextureUsages.texture_binding | wgpu.TextureUsages.copy_dst,
            .dimension = .@"2d",
            .size = .{ .width = width, .height = height, .depth_or_array_layers = 1 },
            .format = if (srgb != 0) .rgba8_unorm_srgb else .rgba8_unorm,
            .mip_level_count = 1,
            .sample_count = 1,
        }) orelse return null;
        const view = texture.createView(null) orelse {
            texture.release();
            return null;
        };
        queue.writeTexture(
            &.{ .texture = texture, .mip_level = 0, .origin = .{ .x = 0, .y = 0, .z = 0 }, .aspect = .all },
            pixels.ptr,
            @as(usize, width) * height * 4,
            &.{ .offset = 0, .bytes_per_row = width * 4, .rows_per_image = height },
            &.{ .width = width, .height = height, .depth_or_array_layers = 1 },
        );
        const sampler = self.samplers[@min(wrap, 1)][@min(filter, 1)] orelse return null;
        const entries = [_]wgpu.BindGroupEntry{
            .{ .binding = 0, .texture_view = view },
            .{ .binding = 1, .sampler = sampler },
        };
        const bg = device.createBindGroup(&.{
            .label = wgpu.StringView.fromSlice("sprite texture bg"),
            .layout = self.texture_bgl.?,
            .entry_count = entries.len,
            .entries = &entries,
        }) orelse {
            view.release();
            texture.release();
            return null;
        };
        return .{ .texture = texture, .view = view, .bind_group = bg, .width = width, .height = height };
    }

    pub fn textureSize(self: *SpriteSystem, id: u32, out_w: *u32, out_h: *u32) bool {
        const t = self.textures.get(id) orelse return false;
        out_w.* = t.width;
        out_h.* = t.height;
        return true;
    }

    // ── Custom shaders ───────────────────────────────────────────────────────

    /// Compile a custom sprite shader module (contract documented in sprite_shader_source.wgsl).
    /// The scene-stage alpha pipeline is built eagerly so bad WGSL fails HERE (returns 0)
    /// instead of at first draw; remaining blend/stage variants build lazily.
    pub fn createShader(
        self: *SpriteSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        alloc: std.mem.Allocator,
        wgsl: []const u8,
        scene_format: wgpu.TextureFormat,
    ) u32 {
        if (self.failed) return 0;
        self.alloc = alloc;
        if (!self.initialized) self.ensureInit(device, queue);
        if (self.failed) return 0;

        var desc = wgpu.shaderModuleWGSLDescriptor(.{ .label = "sprite custom shader", .code = wgsl });
        const module = device.createShaderModule(&desc) orelse return 0;
        const id = self.next_shader;

        // Eager validation: naga/wgpu reject invalid modules at pipeline creation.
        const probe = self.buildPipeline(device, module, scene_format, 0) orelse {
            module.release();
            log.warn("sprite shader rejected (pipeline creation failed)", .{});
            return 0;
        };
        self.shaders.put(alloc, id, module) catch {
            probe.release();
            module.release();
            return 0;
        };
        const key = pipeKey(id, scene_format, 0, 0);
        self.pipelines.put(alloc, key, probe) catch probe.release();
        self.next_shader += 1;
        return id;
    }

    // ── Frame API ────────────────────────────────────────────────────────────

    /// Start a sprite frame: reset batches and store both stage cameras
    /// (scene = world camera, overlay = usually pixel-space ortho). 20 floats each:
    /// column-major view_proj + (viewport_w, viewport_h, 0, 0).
    pub fn begin(
        self: *SpriteSystem,
        alloc: std.mem.Allocator,
        scene_vp: *const [16]f32,
        overlay_vp: *const [16]f32,
        viewport_w: f32,
        viewport_h: f32,
    ) void {
        if (self.failed) return;
        self.alloc = alloc;
        self.inst_cpu.clearRetainingCapacity();
        self.params_cpu.clearRetainingCapacity();
        self.batches.clearRetainingCapacity();
        @memcpy(self.scene_cam[0..16], scene_vp);
        @memcpy(self.overlay_cam[0..16], overlay_vp);
        self.scene_cam[16] = viewport_w;
        self.scene_cam[17] = viewport_h;
        self.scene_cam[18] = 0;
        self.scene_cam[19] = 0;
        @memcpy(self.overlay_cam[16..20], self.scene_cam[16..20]);
        self.frame_active = true;
        self.uploaded = false;
    }

    /// Append one pre-sorted batch: `count` sprites × INSTANCE_FLOATS floats, drawn with
    /// `texture` (+ optional `texture2` for custom shaders; 0 = white), `shader` (0 = default),
    /// `blend` (0 alpha / 1 additive / 2 opaque) into `stage` (0 scene / 1 overlay).
    pub fn draw(
        self: *SpriteSystem,
        texture: u32,
        texture2: u32,
        shader: u32,
        blend: u32,
        stage: u32,
        params: []const f32,
        data: []const f32,
        count: u32,
    ) void {
        if (self.failed or !self.frame_active or count == 0) return;
        const a = self.alloc orelse return;
        const floats = @as(usize, count) * INSTANCE_FLOATS;
        if (data.len < floats) return;
        if (!self.textures.contains(texture)) return;

        const inst_byte_off = self.inst_cpu.items.len * 4;
        self.inst_cpu.appendSlice(a, data[0..floats]) catch return;

        // One 256-byte slot per batch; 16 params floats zero-padded.
        const params_off: u32 = @intCast(self.params_cpu.items.len);
        var slot = [_]u8{0} ** PARAMS_SLOT;
        const n = @min(params.len, PARAMS_FLOATS);
        @memcpy(slot[0 .. n * 4], std.mem.sliceAsBytes(params[0..n]));
        self.params_cpu.appendSlice(a, &slot) catch return;

        self.batches.append(a, .{
            .texture = texture,
            .texture2 = texture2,
            .shader = shader,
            .blend = @min(blend, 2),
            .stage = @min(stage, 1),
            .inst_byte_off = inst_byte_off,
            .count = count,
            .params_off = params_off,
        }) catch {
            _ = self.params_cpu.pop();
        };
    }

    // ── Render ───────────────────────────────────────────────────────────────

    /// Stage 0: record the scene-stage sprite pass onto the geometry encoder, targeting the
    /// (already-resolved, single-sample) HDR scene view. Called at the end of renderSceneGeometry.
    pub fn renderScene(
        self: *SpriteSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        target: *wgpu.TextureView,
        hdr_format: wgpu.TextureFormat,
    ) u32 {
        return self.renderStage(device, queue, encoder, target, hdr_format, 0);
    }

    /// Stage 1: record the overlay pass onto the post encoder, targeting the LDR destination —
    /// strictly AFTER renderPostProcess's tonemap/TAA writes on the same encoder. Also ends the
    /// sprite frame (subsequent draws are ignored until the next begin()).
    pub fn renderOverlay(
        self: *SpriteSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        target: *wgpu.TextureView,
        dst_format: wgpu.TextureFormat,
    ) u32 {
        const drawn = self.renderStage(device, queue, encoder, target, dst_format, 1);
        self.frame_active = false;
        return drawn;
    }

    fn renderStage(
        self: *SpriteSystem,
        device: *wgpu.Device,
        queue: *wgpu.Queue,
        encoder: *wgpu.CommandEncoder,
        target: *wgpu.TextureView,
        format: wgpu.TextureFormat,
        stage: u32,
    ) u32 {
        if (self.failed or !self.frame_active or self.batches.items.len == 0) return 0;
        var any = false;
        for (self.batches.items) |b| {
            if (b.stage == stage) {
                any = true;
                break;
            }
        }
        if (!any) return 0;
        if (!self.initialized) self.ensureInit(device, queue);
        if (self.failed or !self.initialized) return 0;
        if (!self.uploaded) self.upload(device, queue);
        if (self.inst_buf == null or self.params_bg == null) return 0;

        const color = wgpu.ColorAttachment{
            .view = target,
            .load_op = .load,
            .store_op = .store,
            .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 1 },
        };
        const pass = encoder.beginRenderPass(&.{
            .label = wgpu.StringView.fromSlice(if (stage == 0) "sprite scene pass" else "sprite overlay pass"),
            .color_attachment_count = 1,
            .color_attachments = @ptrCast(&color),
        }) orelse return 0;
        defer {
            pass.end();
            pass.release();
        }

        const cam_bg = if (stage == 0) self.scene_cam_bg.? else self.overlay_cam_bg.?;
        const white_bg = self.white.?.bind_group;
        var drawn: u32 = 0;
        for (self.batches.items) |b| {
            if (b.stage != stage) continue;
            const pipe = self.getPipeline(device, b.shader, format, b.blend, b.stage) orelse continue;
            const tex = self.textures.get(b.texture) orelse continue;
            const tex2_bg = if (self.textures.get(b.texture2)) |t2| t2.bind_group else white_bg;
            pass.setPipeline(pipe);
            pass.setBindGroup(0, cam_bg, 0, null);
            pass.setBindGroup(1, tex.bind_group, 0, null);
            pass.setBindGroup(2, self.params_bg.?, 1, @ptrCast(&b.params_off));
            pass.setBindGroup(3, tex2_bg, 0, null);
            pass.setVertexBuffer(0, self.inst_buf.?, b.inst_byte_off, @as(u64, b.count) * STRIDE_BYTES);
            pass.draw(6, b.count, 0, 0);
            drawn += 1;
        }
        return drawn;
    }

    /// Upload the whole frame's instances + params in two writeBuffers, growing the GPU rings
    /// first (frame contents are fully known here, so growth can never orphan earlier batches).
    fn upload(self: *SpriteSystem, device: *wgpu.Device, queue: *wgpu.Queue) void {
        self.uploaded = true;
        const inst_bytes: u64 = self.inst_cpu.items.len * 4;
        if (inst_bytes > 0) {
            if (self.inst_buf == null or self.inst_cap_bytes < inst_bytes) {
                if (self.inst_buf) |b| b.release();
                const cap = std.math.ceilPowerOfTwo(u64, @max(inst_bytes, 16 * 1024)) catch inst_bytes;
                self.inst_buf = device.createBuffer(&.{
                    .label = wgpu.StringView.fromSlice("sprite instance ring"),
                    .usage = wgpu.BufferUsages.vertex | wgpu.BufferUsages.copy_dst,
                    .size = cap,
                }) orelse {
                    self.inst_cap_bytes = 0;
                    return;
                };
                self.inst_cap_bytes = cap;
            }
            queue.writeBuffer(self.inst_buf.?, 0, @ptrCast(self.inst_cpu.items.ptr), @intCast(inst_bytes));
        }

        const params_bytes: u64 = self.params_cpu.items.len;
        if (params_bytes > 0) {
            if (self.params_buf == null or self.params_cap_bytes < params_bytes) {
                if (self.params_bg) |bg| bg.release();
                if (self.params_buf) |b| b.release();
                self.params_bg = null;
                const cap = std.math.ceilPowerOfTwo(u64, @max(params_bytes, 16 * 1024)) catch params_bytes;
                self.params_buf = device.createBuffer(&.{
                    .label = wgpu.StringView.fromSlice("sprite params ring"),
                    .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
                    .size = cap,
                }) orelse {
                    self.params_cap_bytes = 0;
                    return;
                };
                self.params_cap_bytes = cap;
                const entry = wgpu.BindGroupEntry{ .binding = 0, .buffer = self.params_buf.?, .offset = 0, .size = PARAMS_FLOATS * 4 };
                self.params_bg = device.createBindGroup(&.{
                    .label = wgpu.StringView.fromSlice("sprite params bg"),
                    .layout = self.params_bgl.?,
                    .entry_count = 1,
                    .entries = @ptrCast(&entry),
                }) orelse return;
            }
            queue.writeBuffer(self.params_buf.?, 0, @ptrCast(self.params_cpu.items.ptr), @intCast(params_bytes));
        }

        queue.writeBuffer(self.scene_cam_buf.?, 0, @ptrCast(&self.scene_cam), CAMERA_BYTES);
        queue.writeBuffer(self.overlay_cam_buf.?, 0, @ptrCast(&self.overlay_cam), CAMERA_BYTES);
    }

    // ── Init / pipelines ─────────────────────────────────────────────────────

    fn ensureInit(self: *SpriteSystem, device: *wgpu.Device, queue: *wgpu.Queue) void {
        if (self.initialized or self.failed) return;

        var desc = wgpu.shaderModuleWGSLDescriptor(.{
            .label = "zigote sprite shader",
            .code = shaders3d.sprite_shader_source,
        });
        self.default_shader = device.createShaderModule(&desc) orelse return self.fail();

        const cam_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.vertex | wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform } },
        };
        self.camera_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("sprite camera bgl"),
            .entry_count = cam_entries.len,
            .entries = &cam_entries,
        }) orelse return self.fail();

        const tex_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.fragment, .texture = .{ .sample_type = .float, .view_dimension = .@"2d" } },
            .{ .binding = 1, .visibility = wgpu.ShaderStages.fragment, .sampler = .{ .type = .filtering } },
        };
        self.texture_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("sprite texture bgl"),
            .entry_count = tex_entries.len,
            .entries = &tex_entries,
        }) orelse return self.fail();

        const params_entries = [_]wgpu.BindGroupLayoutEntry{
            .{ .binding = 0, .visibility = wgpu.ShaderStages.vertex | wgpu.ShaderStages.fragment, .buffer = .{ .type = .uniform, .has_dynamic_offset = @intFromBool(true), .min_binding_size = PARAMS_FLOATS * 4 } },
        };
        self.params_bgl = device.createBindGroupLayout(&.{
            .label = wgpu.StringView.fromSlice("sprite params bgl"),
            .entry_count = params_entries.len,
            .entries = &params_entries,
        }) orelse return self.fail();

        const bgls = [_]*wgpu.BindGroupLayout{ self.camera_bgl.?, self.texture_bgl.?, self.params_bgl.?, self.texture_bgl.? };
        self.layout = device.createPipelineLayout(&.{
            .label = wgpu.StringView.fromSlice("sprite pipeline layout"),
            .bind_group_layout_count = bgls.len,
            .bind_group_layouts = &bgls,
        }) orelse return self.fail();

        // Samplers [wrap][filter].
        for (0..2) |w| for (0..2) |f| {
            const fm: wgpu.FilterMode = if (f == 0) .nearest else .linear;
            const am: wgpu.AddressMode = if (w == 0) .clamp_to_edge else .repeat;
            self.samplers[w][f] = device.createSampler(&.{
                .label = wgpu.StringView.fromSlice("sprite sampler"),
                .address_mode_u = am,
                .address_mode_v = am,
                .address_mode_w = am,
                .mag_filter = fm,
                .min_filter = fm,
                .mipmap_filter = .nearest,
            }) orelse return self.fail();
        };

        // Camera UBOs + bind groups (contents written per frame in upload()).
        inline for (.{ "scene", "overlay" }, 0..) |name, i| {
            const buf = device.createBuffer(&.{
                .label = wgpu.StringView.fromSlice("sprite camera " ++ name),
                .usage = wgpu.BufferUsages.uniform | wgpu.BufferUsages.copy_dst,
                .size = CAMERA_BYTES,
            }) orelse return self.fail();
            const entry = wgpu.BindGroupEntry{ .binding = 0, .buffer = buf, .offset = 0, .size = CAMERA_BYTES };
            const bg = device.createBindGroup(&.{
                .label = wgpu.StringView.fromSlice("sprite camera bg " ++ name),
                .layout = self.camera_bgl.?,
                .entry_count = 1,
                .entries = @ptrCast(&entry),
            }) orelse return self.fail();
            if (i == 0) {
                self.scene_cam_buf = buf;
                self.scene_cam_bg = bg;
            } else {
                self.overlay_cam_buf = buf;
                self.overlay_cam_bg = bg;
            }
        }

        // White 1×1 fallback (secondary-texture default).
        const white_px = [_]u8{ 255, 255, 255, 255 };
        self.white = self.makeTexture(device, queue, &white_px, 1, 1, 0, 0, 0) orelse return self.fail();

        self.initialized = true;
    }

    fn fail(self: *SpriteSystem) void {
        self.failed = true;
        log.err("sprite system init failed — 2D sprite rendering disabled", .{});
    }

    fn pipeKey(shader: u32, format: wgpu.TextureFormat, blend: u32, stage: u32) u64 {
        return (@as(u64, shader) << 32) | (@as(u64, @intFromEnum(format)) << 8) | (@as(u64, blend) << 1) | stage;
    }

    fn getPipeline(self: *SpriteSystem, device: *wgpu.Device, shader: u32, format: wgpu.TextureFormat, blend: u32, stage: u32) ?*wgpu.RenderPipeline {
        const a = self.alloc orelse return null;
        const key = pipeKey(shader, format, blend, stage);
        if (self.pipelines.get(key)) |p| return p;

        const module = if (shader == 0)
            self.default_shader.?
        else
            self.shaders.get(shader) orelse {
                // Unknown/failed shader: fall back to the default pipeline for this blend/stage.
                log.warn("unknown sprite shader {d}; falling back to default", .{shader});
                return self.getPipeline(device, 0, format, blend, stage);
            };

        const pipe = self.buildPipeline(device, module, format, blend) orelse {
            if (shader != 0) return self.getPipeline(device, 0, format, blend, stage);
            return null;
        };
        self.pipelines.put(a, key, pipe) catch {
            pipe.release();
            return null;
        };
        return pipe;
    }

    fn buildPipeline(
        self: *SpriteSystem,
        device: *wgpu.Device,
        module: *wgpu.ShaderModule,
        format: wgpu.TextureFormat,
        blend: u32,
    ) ?*wgpu.RenderPipeline {
        // Shaders output premultiplied alpha.
        const alpha_blend = wgpu.BlendState{
            .color = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha, .operation = .add },
            .alpha = .{ .src_factor = .one, .dst_factor = .one_minus_src_alpha, .operation = .add },
        };
        const additive_blend = wgpu.BlendState{
            .color = .{ .src_factor = .one, .dst_factor = .one, .operation = .add },
            .alpha = .{ .src_factor = .one, .dst_factor = .one, .operation = .add },
        };
        const target = wgpu.ColorTargetState{
            .format = format,
            .blend = switch (blend) {
                0 => &alpha_blend,
                1 => &additive_blend,
                else => null,
            },
            .write_mask = wgpu.ColorWriteMasks.all,
        };

        const attrs = [_]wgpu.VertexAttribute{
            .{ .format = .float32x3, .offset = 0, .shader_location = 0 }, // pos.xyz
            .{ .format = .float32, .offset = 12, .shader_location = 1 }, // rot
            .{ .format = .float32x2, .offset = 16, .shader_location = 2 }, // size
            .{ .format = .float32x2, .offset = 24, .shader_location = 3 }, // uv0
            .{ .format = .float32x2, .offset = 32, .shader_location = 4 }, // uv1
            .{ .format = .float32x4, .offset = 40, .shader_location = 5 }, // color
        };
        const vb_layout = wgpu.VertexBufferLayout{
            .array_stride = STRIDE_BYTES,
            .step_mode = .instance,
            .attribute_count = attrs.len,
            .attributes = &attrs,
        };

        return device.createRenderPipeline(&.{
            .label = wgpu.StringView.fromSlice("sprite pipeline"),
            .layout = self.layout.?,
            .vertex = .{
                .module = module,
                .entry_point = wgpu.StringView.fromSlice("vs_main"),
                .buffer_count = 1,
                .buffers = @ptrCast(&vb_layout),
            },
            .primitive = .{ .topology = .triangle_list, .front_face = .ccw, .cull_mode = .none },
            .depth_stencil = null,
            .multisample = .{ .count = 1, .mask = 0xFFFFFFFF },
            .fragment = &.{
                .module = module,
                .entry_point = wgpu.StringView.fromSlice("fs_main"),
                .target_count = 1,
                .targets = @ptrCast(&target),
            },
        });
    }
};
