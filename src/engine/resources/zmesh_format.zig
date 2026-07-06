//! The `.zmesh` interchange format — the trivial binary the Assimp importer writes into the
//! mesh cache and `zigote_scene_set_mesh_blob` reads back at runtime. Lives here (not in
//! ffi/assimp_loader.zig) because it has no Assimp dependency and the parser must be available
//! in every build — including `-Denable3d=false` game exports that compile the importer out
//! but still load pre-baked `.zmesh` assets.
//!
//! Layout (all little-endian):
//!   u32 magic ('ZMSH') | u32 version | u32 vertex_count | u32 index_count
//!   vertex_count vertices | u32[index_count]
//!
//! Version 2 (current) stores the packed 28 B `GpuVertex` (pos f32x3, normal snorm8x4, uv f32x2,
//! tangent snorm8x4) — exactly what the renderer uploads — so load is a straight memcpy and the file
//! is ~42% smaller than v1. Version 1 stored the 48 B full-float `Vertex`; it is still parsed (packed
//! on load) so pre-existing `.mesh_cache` files keep working without a forced re-import.
const std = @import("std");
const mesh = @import("mesh.zig");
const Vertex = mesh.Vertex;
const GpuVertex = mesh.GpuVertex;
const Mesh = mesh.Mesh;
const Primitive = mesh.Primitive;

const ZMESH_MAGIC: u32 = 0x484D535A; // 'Z','M','S','H'
const ZMESH_VERSION: u32 = 2; // 1 = 48 B Vertex (legacy, still parsed); 2 = 28 B GpuVertex
const ZMESH_HEADER: usize = 16;

/// Serialize a vertex/index buffer into a `.zmesh` blob (caller owns the returned slice). Writes the
/// current v2 format: each full-float `Vertex` is packed to the 28 B `GpuVertex` as it is written.
pub fn writeZmesh(allocator: std.mem.Allocator, vertices: []const Vertex, indices: []const u32) ![]u8 {
    const total = ZMESH_HEADER + vertices.len * @sizeOf(GpuVertex) + indices.len * @sizeOf(u32);
    const buf = try allocator.alloc(u8, total);
    std.mem.writeInt(u32, buf[0..4], ZMESH_MAGIC, .little);
    std.mem.writeInt(u32, buf[4..8], ZMESH_VERSION, .little);
    std.mem.writeInt(u32, buf[8..12], @intCast(vertices.len), .little);
    std.mem.writeInt(u32, buf[12..16], @intCast(indices.len), .little);
    var off: usize = ZMESH_HEADER;
    for (vertices) |v| {
        const g = GpuVertex.fromVertex(v);
        @memcpy(buf[off .. off + @sizeOf(GpuVertex)], std.mem.asBytes(&g));
        off += @sizeOf(GpuVertex);
    }
    const ibytes = std.mem.sliceAsBytes(indices);
    @memcpy(buf[off .. off + ibytes.len], ibytes);
    return buf;
}

/// Parse a `.zmesh` blob into a single-primitive `Mesh` (allocator-owned). Handles both the current
/// v2 (28 B `GpuVertex`, straight memcpy) and legacy v1 (48 B `Vertex`, packed on load) layouts.
pub fn parseZmesh(allocator: std.mem.Allocator, data: []const u8) !Mesh {
    if (data.len < ZMESH_HEADER) return error.InvalidZmesh;
    if (std.mem.readInt(u32, data[0..4], .little) != ZMESH_MAGIC) return error.InvalidZmeshMagic;
    const version = std.mem.readInt(u32, data[4..8], .little);
    const vsize: usize = switch (version) {
        1 => @sizeOf(Vertex),
        2 => @sizeOf(GpuVertex),
        else => return error.UnsupportedZmeshVersion,
    };
    const vcount: usize = std.mem.readInt(u32, data[8..12], .little);
    const icount: usize = std.mem.readInt(u32, data[12..16], .little);
    const vbytes = vcount * vsize;
    const need = ZMESH_HEADER + vbytes + icount * @sizeOf(u32);
    if (data.len < need) return error.TruncatedZmesh;

    const indices = try allocator.alloc(u32, icount);
    errdefer allocator.free(indices);
    @memcpy(std.mem.sliceAsBytes(indices), data[ZMESH_HEADER + vbytes .. need]);

    const prims = try allocator.alloc(Primitive, 1);
    errdefer allocator.free(prims);

    if (version == 2) {
        // Already packed on disk — copy straight into the retained GPU form (no per-vertex pack).
        const gpu = try allocator.alloc(GpuVertex, vcount);
        errdefer allocator.free(gpu);
        @memcpy(std.mem.sliceAsBytes(gpu), data[ZMESH_HEADER .. ZMESH_HEADER + vbytes]);
        prims[0] = .{ .vertices = gpu, .indices = indices };
    } else {
        // Legacy v1: read full-float vertices and pack. packPrimitive frees `vertices` on success;
        // on failure the errdefer here frees it (and the outer errdefers free indices/prims).
        const vertices = try allocator.alloc(Vertex, vcount);
        errdefer allocator.free(vertices);
        @memcpy(std.mem.sliceAsBytes(vertices), data[ZMESH_HEADER .. ZMESH_HEADER + vbytes]);
        prims[0] = try mesh.packPrimitive(allocator, vertices, indices, null);
    }
    return .{ .primitives = prims, .name = "" };
}

test "zmesh v2 round-trip" {
    const a = std.testing.allocator;
    const verts = [_]Vertex{
        .{ .position = .{ 1, 2, 3 }, .normal = .{ 1, 0, 0 }, .uv = .{ 0.25, 0.75 }, .tangent = .{ 0, 1, 0, 1 } },
        .{ .position = .{ 4, 5, 6 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0.5, 0.5 }, .tangent = .{ 1, 0, 0, -1 } },
        .{ .position = .{ 7, 8, 9 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 }, .tangent = .{ 0, 0, 1, 1 } },
    };
    const idx = [_]u32{ 0, 1, 2 };
    const blob = try writeZmesh(a, &verts, &idx);
    defer a.free(blob);
    // v2 header + 28 B/vertex, and version byte is 2.
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, blob[4..8], .little));
    try std.testing.expectEqual(ZMESH_HEADER + verts.len * @sizeOf(GpuVertex) + idx.len * @sizeOf(u32), blob.len);

    const parsed = try parseZmesh(a, blob);
    defer {
        a.free(parsed.primitives[0].vertices);
        a.free(parsed.primitives[0].indices);
        a.free(parsed.primitives);
    }
    try std.testing.expectEqual(@as(usize, 3), parsed.primitives[0].vertices.len);
    try std.testing.expectEqualSlices(u32, &idx, parsed.primitives[0].indices);
    // Parsed v2 vertices are byte-identical to packing the source at load time.
    for (verts, parsed.primitives[0].vertices) |src, got| {
        try std.testing.expectEqual(GpuVertex.fromVertex(src), got);
    }
    try std.testing.expectError(error.InvalidZmeshMagic, parseZmesh(a, blob[4..]));
    try std.testing.expectError(error.TruncatedZmesh, parseZmesh(a, blob[0 .. blob.len - 1]));
}
