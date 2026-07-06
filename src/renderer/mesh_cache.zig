//! Per-primitive GPU geometry cache for the 3D renderer.
//!
//! Owns one vertex/index/edge-index buffer set per (mesh, primitive) key, uploaded lazily on first
//! draw and reused across frames. Extracted from the `Gpu3d` monolith as a leaf struct with clean
//! field ownership (device + allocator + the map). The renderer holds one `MeshCache` by composition
//! and draws through `getOrUpload`.

const std = @import("std");
const wgpu = @import("wgpu");
const zmo = @import("zmesh_opt");
const engine = @import("zigote_engine");
const resources_mod = engine.resources;

// Running total of GPU bytes held by cached mesh vertex/index/edge buffers. A create adds, a
// release subtracts (each buffer set remembers its own byte_size). Reported through ZgEngineStats
// so the devtools GPU panel can show real mesh-buffer memory. Diagnostic-only; single-threaded.
var g_mesh_bytes: u64 = 0;

/// Total GPU bytes currently held by mesh vertex/index/edge buffers (for engine stats).
pub fn gpuMeshBytes() u64 {
    return g_mesh_bytes;
}

pub const MeshGpuBuffers = struct {
    vertex_buf: *wgpu.Buffer,
    index_buf: *wgpu.Buffer,
    index_count: u32,
    // GPU bytes this buffer set holds (vertex + index + optional edge buffer) — tracked so releasing
    // it subtracts exactly what it added from the running `g_mesh_bytes` total.
    byte_size: u64 = 0,
    // Edge index buffer (line-list) derived from the triangle indices: each triangle (a,b,c)
    // expands to the three edges (a,b)(b,c)(c,a). Bound instead of `index_buf` by the wireframe
    // pipeline so the same vertex buffer renders as a wireframe without re-uploading geometry.
    // Built LAZILY on the first wireframe draw (`ensureLineBuffer`) — wireframe is a default-off
    // debug toggle, so a normal load never pays its 24 B/tri (2× the triangle IB) GPU memory,
    // transient CPU alloc, edge-expansion loop, or upload bandwidth. `null` until first needed.
    line_index_buf: ?*wgpu.Buffer = null,
    line_index_count: u32 = 0,
    // Whether `index_buf` holds the meshopt-reordered order (true) or the raw source order
    // (false, meshopt alloc fallback). The lazy edge buffer must match this so its indices
    // reference the vertices as they were uploaded.
    optimized: bool = false,

    pub fn deinit(self: *MeshGpuBuffers) void {
        self.vertex_buf.release();
        self.index_buf.release();
        if (self.line_index_buf) |b| b.release();
        g_mesh_bytes -= self.byte_size;
        self.byte_size = 0;
    }
};

/// Per-mesh list of primitive GPU buffers (indexed by primitive index). `null` = not yet uploaded.
const PrimSlots = std.ArrayListUnmanaged(?MeshGpuBuffers);

pub const MeshCache = struct {
    device: *wgpu.Device,
    allocator: std.mem.Allocator,
    // Dense two-level array [mesh_handle][prim_index] → GPU buffers, replacing an
    // AutoHashMap(PrimKey, ...): per-draw lookup is two bounds-checked loads instead of hashing an
    // 8-byte key and probing. Mesh handles are dense World.meshes indices (always resolved before
    // getOrUpload is reached, so the outer array stays bounded); prim indices are 0-based dense.
    meshes: std.ArrayListUnmanaged(PrimSlots) = .empty,

    pub fn init(device: *wgpu.Device, allocator: std.mem.Allocator) MeshCache {
        return .{ .device = device, .allocator = allocator };
    }

    /// Release every cached buffer + free the arrays. Renderer teardown.
    pub fn deinit(self: *MeshCache) void {
        for (self.meshes.items) |*prims| {
            for (prims.items) |*slot| if (slot.*) |*b| b.deinit();
            prims.deinit(self.allocator);
        }
        self.meshes.deinit(self.allocator);
    }

    /// Release every cached buffer but keep the arrays allocated (scene clear / project switch).
    pub fn clear(self: *MeshCache) void {
        for (self.meshes.items) |*prims| {
            for (prims.items) |*slot| if (slot.*) |*b| b.deinit();
            prims.clearRetainingCapacity();
        }
    }

    /// Return cached GPU buffers for (mesh, prim), uploading + caching on first use.
    /// Returns null on allocation/upload failure so the caller can skip the primitive.
    pub fn getOrUpload(self: *MeshCache, queue: *wgpu.Queue, mesh_handle: u32, prim_index: u32, prim: *const resources_mod.Primitive) ?MeshGpuBuffers {
        if (mesh_handle < self.meshes.items.len) {
            const prims = self.meshes.items[mesh_handle];
            if (prim_index < prims.items.len) {
                if (prims.items[prim_index]) |b| return b;
            }
        }
        return self.upload(queue, mesh_handle, prim_index, prim) catch null;
    }

    /// Drop all cached primitives belonging to `mesh_handle` (re-uploaded on next draw).
    pub fn invalidate(self: *MeshCache, mesh_handle: u32) void {
        if (mesh_handle >= self.meshes.items.len) return;
        const prims = &self.meshes.items[mesh_handle];
        for (prims.items) |*slot| if (slot.*) |*b| {
            b.deinit();
            slot.* = null;
        };
    }

    /// Lazily build (once) + return the wireframe edge index buffer for an already-uploaded primitive.
    /// The wireframe pipeline binds this instead of `index_buf`. Returns null if the primitive was
    /// never uploaded, has no triangles, or on alloc/upload failure (caller skips the wireframe draw).
    /// The edge order matches whatever `index_buf` holds (re-deriving the meshopt order when the slot
    /// was optimized) so it references the vertices exactly as uploaded.
    pub fn ensureLineBuffer(self: *MeshCache, queue: *wgpu.Queue, mesh_handle: u32, prim_index: u32, prim: *const resources_mod.Primitive) ?struct { buf: *wgpu.Buffer, count: u32 } {
        if (mesh_handle >= self.meshes.items.len) return null;
        const prims = &self.meshes.items[mesh_handle];
        if (prim_index >= prims.items.len) return null;
        const slot = &(prims.items[prim_index] orelse return null);
        if (slot.line_index_buf) |b| return .{ .buf = b, .count = slot.line_index_count };

        const tri_count = prim.indices.len / 3;
        if (tri_count == 0) return null;

        // Reproduce the exact index order that was uploaded to index_buf. optimizeMesh is
        // deterministic in `prim`, so when the slot was optimized this yields the same order; if it
        // was a raw fallback, or the re-optimize now OOMs on an optimized slot, use/return safely.
        const opt = if (slot.optimized) self.optimizeMesh(prim) else null;
        defer if (opt) |o| {
            self.allocator.free(o.verts);
            self.allocator.free(o.indices);
        };
        if (slot.optimized and opt == null) return null; // can't reproduce the uploaded order — skip
        const src_indices: []const u32 = if (opt) |o| o.indices else prim.indices;

        const line_count: usize = tri_count * 6;
        const line_indices = self.allocator.alloc(u32, line_count) catch return null;
        defer self.allocator.free(line_indices);
        for (0..tri_count) |t| {
            const a = src_indices[t * 3 + 0];
            const b = src_indices[t * 3 + 1];
            const c = src_indices[t * 3 + 2];
            line_indices[t * 6 + 0] = a;
            line_indices[t * 6 + 1] = b;
            line_indices[t * 6 + 2] = b;
            line_indices[t * 6 + 3] = c;
            line_indices[t * 6 + 4] = c;
            line_indices[t * 6 + 5] = a;
        }
        const lib = self.device.createBuffer(&.{
            .usage = wgpu.BufferUsages.index | wgpu.BufferUsages.copy_dst,
            .size = line_count * @sizeOf(u32),
        }) orelse return null;
        const lib_data = std.mem.sliceAsBytes(line_indices);
        queue.writeBuffer(lib, 0, lib_data.ptr, lib_data.len);
        slot.line_index_buf = lib;
        slot.line_index_count = @intCast(line_count);
        const line_bytes: u64 = line_count * @sizeOf(u32);
        slot.byte_size += line_bytes;
        g_mesh_bytes += line_bytes;
        return .{ .buf = lib, .count = slot.line_index_count };
    }

    /// Reorder indices for post-transform vertex-cache locality, then reorder vertices for
    /// pre-transform fetch locality (rewriting the indices to match). Returns owned (verts, indices)
    /// the caller must free, or null for empty/non-triangle primitives or on alloc failure (caller
    /// falls back to source order — never fatal). Deterministic in `prim`, so re-running it later
    /// reproduces the exact same index order — which is why the lazy edge buffer can rebuild it.
    fn optimizeMesh(self: *MeshCache, prim: *const resources_mod.Primitive) ?struct { verts: []resources_mod.GpuVertex, indices: []u32 } {
        const tri_count = prim.indices.len / 3;
        if (prim.vertices.len == 0 or prim.indices.len == 0 or tri_count * 3 != prim.indices.len)
            return null;
        const oi = self.allocator.alloc(u32, prim.indices.len) catch return null;
        const ov = self.allocator.alloc(resources_mod.GpuVertex, prim.vertices.len) catch {
            self.allocator.free(oi);
            return null;
        };
        zmo.optimizeVertexCache(oi, prim.indices, prim.vertices.len);
        _ = zmo.optimizeVertexFetch(resources_mod.GpuVertex, ov, oi, prim.vertices);
        return .{ .verts = ov, .indices = oi };
    }

    fn upload(self: *MeshCache, queue: *wgpu.Queue, mesh_handle: u32, prim_index: u32, prim: *const resources_mod.Primitive) !MeshGpuBuffers {
        const vb_size = prim.vertices.len * @sizeOf(resources_mod.GpuVertex);
        const ib_size = prim.indices.len * @sizeOf(u32);

        const vb = self.device.createBuffer(&.{
            .usage = wgpu.BufferUsages.vertex | wgpu.BufferUsages.copy_dst,
            .size = vb_size,
        }) orelse return error.BufferCreateFailed;

        const ib = self.device.createBuffer(&.{
            .usage = wgpu.BufferUsages.index | wgpu.BufferUsages.copy_dst,
            .size = ib_size,
        }) orelse {
            vb.release();
            return error.BufferCreateFailed;
        };

        // ── meshopt (one-time, at upload) ─────────────────────────────────────────────────────
        const opt = self.optimizeMesh(prim);
        defer if (opt) |o| {
            self.allocator.free(o.verts);
            self.allocator.free(o.indices);
        };
        // Vertices are already 28 B GpuVertex (packed once at registration — packPrimitive). meshopt
        // only reorders them for fetch locality; the byte content is unchanged, so upload is a straight
        // memcpy with no per-vertex pack. The reorder permutation depends only on the index stream, so
        // reordering the packed vertices is byte-identical to the old pack-after-reorder path.
        const up_verts: []const resources_mod.GpuVertex = if (opt) |o| o.verts else prim.vertices;
        const up_indices: []const u32 = if (opt) |o| o.indices else prim.indices;

        const vb_data = std.mem.sliceAsBytes(up_verts);
        queue.writeBuffer(vb, 0, vb_data.ptr, vb_data.len);
        const ib_data = std.mem.sliceAsBytes(up_indices);
        queue.writeBuffer(ib, 0, ib_data.ptr, ib_data.len);

        const buffers = MeshGpuBuffers{
            .vertex_buf = vb,
            .index_buf = ib,
            .index_count = @intCast(prim.indices.len),
            .optimized = opt != null,
            .byte_size = vb_size + ib_size,
        };
        g_mesh_bytes += vb_size + ib_size;
        // Grow the outer (mesh) array, then the inner (prim) array, then store — mirror of the dense
        // material cache. Grow outer BEFORE taking the inner pointer so the pointer stays valid.
        while (self.meshes.items.len <= mesh_handle)
            try self.meshes.append(self.allocator, .empty);
        const prims = &self.meshes.items[mesh_handle];
        while (prims.items.len <= prim_index)
            try prims.append(self.allocator, null);
        prims.items[prim_index] = buffers;
        return buffers;
    }
};
