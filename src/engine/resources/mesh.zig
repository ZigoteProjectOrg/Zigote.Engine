const std = @import("std");
const math = @import("../math/root.zig");
const Vec3 = math.Vec3;
const Vec2 = math.Vec2;

pub const Vertex = extern struct {
    position: [3]f32,
    normal: [3]f32,
    uv: [2]f32,
    tangent: [4]f32 = .{ 0, 0, 0, 1 },
};

/// Quantize a value in [-1, 1] to snorm8 (i8 in [-127, 127]); the GPU decodes it back to [-1, 1].
inline fn snorm8(x: f32) i8 {
    return @intFromFloat(@round(std.math.clamp(x, -1.0, 1.0) * 127.0));
}

/// GPU-side packed vertex (28 bytes) — what the renderer uploads to the vertex buffer and streams
/// every draw. `normal`/`tangent` are quantized to snorm8x4 (the GPU hardware-normalizes them to
/// [-1, 1] on fetch; 8-bit normals are the industry-standard mesh precision). This is 28 B vs the
/// CPU-side `Vertex`'s 48 B — a ~42% cut to the frame's dominant per-vertex GPU footprint/bandwidth.
/// The CPU-side `Vertex` (full-float, produced by generators/importers/`.zmesh`) is packed into this
/// once, at mesh upload (mesh_cache), so nothing upstream of the GPU changes.
pub const GpuVertex = extern struct {
    position: [3]f32,
    normal: [4]i8, // snorm8x4: xyz unit normal (w unused)
    uv: [2]f32,
    tangent: [4]i8, // snorm8x4: xyz unit tangent + w handedness (+1/-1)

    pub fn fromVertex(v: Vertex) GpuVertex {
        return .{
            .position = v.position,
            .normal = .{ snorm8(v.normal[0]), snorm8(v.normal[1]), snorm8(v.normal[2]), 0 },
            .uv = v.uv,
            .tangent = .{ snorm8(v.tangent[0]), snorm8(v.tangent[1]), snorm8(v.tangent[2]), snorm8(v.tangent[3]) },
        };
    }
};

comptime {
    // The vertex_attrs in wgpu_3d.zig pin these field offsets; keep them in lockstep.
    std.debug.assert(@sizeOf(GpuVertex) == 28);
    std.debug.assert(@offsetOf(GpuVertex, "normal") == 12);
    std.debug.assert(@offsetOf(GpuVertex, "uv") == 16);
    std.debug.assert(@offsetOf(GpuVertex, "tangent") == 24);
}

pub const Primitive = struct {
    // GPU-packed vertices (28 B). The full-float `Vertex` (48 B) is produced only by generators /
    // importers / `.zmesh` parse and packed into this once at registration via `packPrimitive` — it is
    // never retained in `World.meshes`, so a scene holds 28 B/vertex instead of 48 B. `mesh_cache`
    // meshopt-reorders + uploads these directly (no per-vertex pack at draw-time upload).
    vertices: []GpuVertex,
    indices: []u32,
    material_index: ?u32 = null,

    pub fn deinit(self: *Primitive, allocator: std.mem.Allocator) void {
        allocator.free(self.vertices);
        allocator.free(self.indices);
    }
};

/// Pack full-float generation/import vertices into GPU vertices (snorm8x4 normal/tangent) and build a
/// Primitive that owns the packed array + `indices`. Frees the source `verts` (always — the 48 B
/// `Vertex` is transient). This is the single 48 B→28 B pack point, moved here from mesh upload so
/// scenes never retain the full-float copy; upload becomes a straight memcpy of the packed data.
pub fn packPrimitive(allocator: std.mem.Allocator, verts: []Vertex, indices: []u32, material_index: ?u32) !Primitive {
    // Consume `verts` only on success — on alloc failure the caller (which may hold an errdefer on it,
    // e.g. parseZmesh) still owns it, so this never double-frees.
    const gpu = try allocator.alloc(GpuVertex, verts.len);
    for (verts, gpu) |src, *dst| dst.* = GpuVertex.fromVertex(src);
    allocator.free(verts);
    return .{ .vertices = gpu, .indices = indices, .material_index = material_index };
}

/// Recalculate vertex normals from triangle faces (overwrites existing normals). Operates on
/// full-float generation/import `Vertex`es before they are packed by `packPrimitive`.
pub fn recalculateNormals(vertices: []Vertex, indices: []const u32) void {
    for (vertices) |*v| {
        v.normal = .{ 0, 0, 0 };
    }
    var i: usize = 0;
    while (i + 2 < indices.len) : (i += 3) {
        const ia = indices[i];
        const ib = indices[i + 1];
        const ic = indices[i + 2];
        const a = Vec3{ .x = vertices[ia].position[0], .y = vertices[ia].position[1], .z = vertices[ia].position[2] };
        const b = Vec3{ .x = vertices[ib].position[0], .y = vertices[ib].position[1], .z = vertices[ib].position[2] };
        const c = Vec3{ .x = vertices[ic].position[0], .y = vertices[ic].position[1], .z = vertices[ic].position[2] };
        const n = b.sub(a).cross(c.sub(a)).normalize();
        inline for (.{ ia, ib, ic }) |idx| {
            vertices[idx].normal[0] += n.x;
            vertices[idx].normal[1] += n.y;
            vertices[idx].normal[2] += n.z;
        }
    }
    for (vertices) |*v| {
        const nn = Vec3{ .x = v.normal[0], .y = v.normal[1], .z = v.normal[2] };
        const n = nn.normalize();
        v.normal = .{ n.x, n.y, n.z };
    }
}

/// Generate per-vertex tangents (xyz + handedness in w) from positions, UVs and normals using the
/// standard accumulate-then-orthonormalize method (Lengyel). Needed for normal mapping when a mesh
/// ships no TANGENT attribute (procedural primitives, or glTF exported without tangents). Operates on
/// full-float generation/import `Vertex`es before they are packed by `packPrimitive`.
pub fn recalculateTangents(allocator: std.mem.Allocator, vertices: []Vertex, indices: []const u32) !void {
    const n = vertices.len;
    if (n == 0) return;
    const tan = try allocator.alloc(Vec3, n);
    defer allocator.free(tan);
    const bitan = try allocator.alloc(Vec3, n);
    defer allocator.free(bitan);
    @memset(tan, Vec3.zero);
    @memset(bitan, Vec3.zero);

    var i: usize = 0;
    while (i + 2 < indices.len) : (i += 3) {
        const ia = indices[i];
        const ib = indices[i + 1];
        const ic = indices[i + 2];
        const v0 = vertices[ia];
        const v1 = vertices[ib];
        const v2 = vertices[ic];
        const p0 = Vec3{ .x = v0.position[0], .y = v0.position[1], .z = v0.position[2] };
        const p1 = Vec3{ .x = v1.position[0], .y = v1.position[1], .z = v1.position[2] };
        const p2 = Vec3{ .x = v2.position[0], .y = v2.position[1], .z = v2.position[2] };
        const e1 = p1.sub(p0);
        const e2 = p2.sub(p0);
        const du1 = v1.uv[0] - v0.uv[0];
        const dv1 = v1.uv[1] - v0.uv[1];
        const du2 = v2.uv[0] - v0.uv[0];
        const dv2 = v2.uv[1] - v0.uv[1];
        const denom = du1 * dv2 - du2 * dv1;
        const r: f32 = if (@abs(denom) > 1e-8) 1.0 / denom else 0.0;
        const t = Vec3{ .x = (e1.x * dv2 - e2.x * dv1) * r, .y = (e1.y * dv2 - e2.y * dv1) * r, .z = (e1.z * dv2 - e2.z * dv1) * r };
        const bt = Vec3{ .x = (e2.x * du1 - e1.x * du2) * r, .y = (e2.y * du1 - e1.y * du2) * r, .z = (e2.z * du1 - e1.z * du2) * r };
        inline for (.{ ia, ib, ic }) |idx| {
            tan[idx] = tan[idx].add(t);
            bitan[idx] = bitan[idx].add(bt);
        }
    }

    for (vertices, 0..) |*v, idx| {
        const nrm = (Vec3{ .x = v.normal[0], .y = v.normal[1], .z = v.normal[2] }).normalize();
        // Gram-Schmidt: project the accumulated tangent onto the plane perpendicular to N.
        var t = tan[idx].sub(nrm.scale(nrm.dot(tan[idx])));
        if (t.lengthSq() < 1e-12) {
            // No usable UV gradient — pick an arbitrary perpendicular so the TBN stays valid.
            const up = if (@abs(nrm.y) > 0.99) Vec3{ .x = 1, .y = 0, .z = 0 } else Vec3{ .x = 0, .y = 1, .z = 0 };
            t = up.cross(nrm);
        }
        t = t.normalize();
        const w: f32 = if (nrm.cross(t).dot(bitan[idx]) < 0.0) -1.0 else 1.0;
        v.tangent = .{ t.x, t.y, t.z, w };
    }
}

pub const Mesh = struct {
    primitives: []Primitive,
    name: []const u8 = "",
    // Local-space bounding sphere (centre + radius), computed once by computeBounds. Used for
    // frustum culling. Default radius 1 so an un-measured mesh is never wrongly culled.
    bounds_center: Vec3 = Vec3.zero,
    bounds_radius: f32 = 1.0,

    /// Compute the local bounding sphere from all primitive vertices (AABB → centre + half-diagonal).
    pub fn computeBounds(self: *Mesh) void {
        var min = Vec3{ .x = std.math.floatMax(f32), .y = std.math.floatMax(f32), .z = std.math.floatMax(f32) };
        var max = Vec3{ .x = -std.math.floatMax(f32), .y = -std.math.floatMax(f32), .z = -std.math.floatMax(f32) };
        var any = false;
        for (self.primitives) |prim| {
            for (prim.vertices) |v| {
                const p = v.position;
                min.x = @min(min.x, p[0]);
                min.y = @min(min.y, p[1]);
                min.z = @min(min.z, p[2]);
                max.x = @max(max.x, p[0]);
                max.y = @max(max.y, p[1]);
                max.z = @max(max.z, p[2]);
                any = true;
            }
        }
        if (!any) {
            self.bounds_center = Vec3.zero;
            self.bounds_radius = 1.0;
            return;
        }
        self.bounds_center = min.add(max).scale(0.5);
        self.bounds_radius = max.sub(self.bounds_center).length();
    }

    pub fn deinit(self: *Mesh, allocator: std.mem.Allocator) void {
        for (self.primitives) |*prim| {
            prim.deinit(allocator);
        }
        allocator.free(self.primitives);
    }

    /// Create a unit cube mesh (1x1x1 centered at origin).
    pub fn createCube(allocator: std.mem.Allocator) !Mesh {
        const verts = [_]Vertex{
            // +X face
            .{ .position = .{ 0.5, -0.5, -0.5 }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 1 } },
            .{ .position = .{ 0.5, 0.5, -0.5 }, .normal = .{ 1, 0, 0 }, .uv = .{ 0, 0 } },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 0 } },
            .{ .position = .{ 0.5, -0.5, 0.5 }, .normal = .{ 1, 0, 0 }, .uv = .{ 1, 1 } },
            // -X face
            .{ .position = .{ -0.5, -0.5, 0.5 }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 1 } },
            .{ .position = .{ -0.5, 0.5, 0.5 }, .normal = .{ -1, 0, 0 }, .uv = .{ 0, 0 } },
            .{ .position = .{ -0.5, 0.5, -0.5 }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 0 } },
            .{ .position = .{ -0.5, -0.5, -0.5 }, .normal = .{ -1, 0, 0 }, .uv = .{ 1, 1 } },
            // +Y face
            .{ .position = .{ -0.5, 0.5, -0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 } },
            .{ .position = .{ -0.5, 0.5, 0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 } },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 } },
            .{ .position = .{ 0.5, 0.5, -0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 } },
            // -Y face
            .{ .position = .{ -0.5, -0.5, 0.5 }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 1 } },
            .{ .position = .{ -0.5, -0.5, -0.5 }, .normal = .{ 0, -1, 0 }, .uv = .{ 0, 0 } },
            .{ .position = .{ 0.5, -0.5, -0.5 }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 0 } },
            .{ .position = .{ 0.5, -0.5, 0.5 }, .normal = .{ 0, -1, 0 }, .uv = .{ 1, 1 } },
            // +Z face
            .{ .position = .{ 0.5, -0.5, 0.5 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 1 } },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 } },
            .{ .position = .{ -0.5, 0.5, 0.5 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 } },
            .{ .position = .{ -0.5, -0.5, 0.5 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 1 } },
            // -Z face
            .{ .position = .{ -0.5, -0.5, -0.5 }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 1 } },
            .{ .position = .{ -0.5, 0.5, -0.5 }, .normal = .{ 0, 0, -1 }, .uv = .{ 0, 0 } },
            .{ .position = .{ 0.5, 0.5, -0.5 }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 0 } },
            .{ .position = .{ 0.5, -0.5, -0.5 }, .normal = .{ 0, 0, -1 }, .uv = .{ 1, 1 } },
        };

        const face_indices = [_]u32{ 0, 1, 2, 0, 2, 3 };
        var indices: [6 * 6]u32 = undefined;
        for (0..6) |face| {
            for (face_indices, 0..) |idx, i| {
                indices[face * 6 + i] = @intCast(face * 4 + idx);
            }
        }

        const prim_verts = try allocator.dupe(Vertex, &verts);
        const prim_indices = try allocator.dupe(u32, &indices);

        const prims = try allocator.alloc(Primitive, 1);
        try recalculateTangents(allocator, prim_verts, prim_indices);
        prims[0] = try packPrimitive(allocator, prim_verts, prim_indices, null);

        return .{ .primitives = prims, .name = "cube" };
    }

    /// Create a unit quad on the XZ plane (horizontal floor) centered at the origin, normal pointing +Y.
    pub fn createQuad(allocator: std.mem.Allocator) !Mesh {
        const verts = [_]Vertex{
            .{ .position = .{ -0.5, 0.0, 0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 }, .tangent = .{ 1, 0, 0, 1 } },
            .{ .position = .{ 0.5, 0.0, 0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 0 }, .tangent = .{ 1, 0, 0, 1 } },
            .{ .position = .{ 0.5, 0.0, -0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 1, 1 }, .tangent = .{ 1, 0, 0, 1 } },
            .{ .position = .{ -0.5, 0.0, -0.5 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 1 }, .tangent = .{ 1, 0, 0, 1 } },
        };
        const indices = [_]u32{ 0, 1, 2, 0, 2, 3 };

        const prim_verts = try allocator.dupe(Vertex, &verts);
        const prim_indices = try allocator.dupe(u32, &indices);

        const prims = try allocator.alloc(Primitive, 1);
        prims[0] = try packPrimitive(allocator, prim_verts, prim_indices, null);

        return .{ .primitives = prims, .name = "quad" };
    }

    /// UV sphere.
    pub fn createSphere(allocator: std.mem.Allocator, rings: u32, segments: u32) !Mesh {
        const vert_count = (rings + 1) * (segments + 1);
        const idx_count = rings * segments * 6;
        const verts = try allocator.alloc(Vertex, vert_count);
        const indices = try allocator.alloc(u32, idx_count);

        var vi: usize = 0;
        for (0..rings + 1) |ring| {
            const phi = math.pi * @as(f32, @floatFromInt(ring)) / @as(f32, @floatFromInt(rings));
            for (0..segments + 1) |seg| {
                const theta = 2.0 * math.pi * @as(f32, @floatFromInt(seg)) / @as(f32, @floatFromInt(segments));
                const x = @sin(phi) * @cos(theta);
                const y = @cos(phi);
                const z = @sin(phi) * @sin(theta);
                verts[vi] = .{
                    .position = .{ x * 0.5, y * 0.5, z * 0.5 },
                    .normal = .{ x, y, z },
                    .uv = .{
                        @as(f32, @floatFromInt(seg)) / @as(f32, @floatFromInt(segments)),
                        @as(f32, @floatFromInt(ring)) / @as(f32, @floatFromInt(rings)),
                    },
                };
                vi += 1;
            }
        }

        var ii: usize = 0;
        for (0..rings) |ring| {
            for (0..segments) |seg| {
                const a: u32 = @intCast(ring * (segments + 1) + seg);
                const b: u32 = a + 1;
                const c: u32 = @intCast((ring + 1) * (segments + 1) + seg);
                const d: u32 = c + 1;
                // CCW winding (matches glTF) so outward faces are front-facing under
                // front_face=.ccw + back-cull. (Was a,c,b / b,c,d — inverted, which back-face-culled
                // the visible hemisphere so every procedural primitive read as grey chrome.)
                indices[ii] = a;
                indices[ii + 1] = b;
                indices[ii + 2] = c;
                indices[ii + 3] = b;
                indices[ii + 4] = d;
                indices[ii + 5] = c;
                ii += 6;
            }
        }

        const prims = try allocator.alloc(Primitive, 1);
        try recalculateTangents(allocator, verts, indices);
        prims[0] = try packPrimitive(allocator, verts, indices, null);
        return .{ .primitives = prims, .name = "sphere" };
    }

    /// Unit cylinder along the Y axis (radius 0.5, height 1) with side + both caps. Used for wheels:
    /// scale + lay on its side to make a tire.
    pub fn createCylinder(allocator: std.mem.Allocator, segments: u32) !Mesh {
        const seg = segments;
        const side_v = 2 * (seg + 1);
        const cap_v = 1 + (seg + 1);
        const vert_count = side_v + 2 * cap_v;
        const idx_count = seg * 6 + 2 * (seg * 3);
        const verts = try allocator.alloc(Vertex, vert_count);
        const indices = try allocator.alloc(u32, idx_count);

        const r: f32 = 0.5;
        const hy: f32 = 0.5;
        var vi: usize = 0;
        var ii: usize = 0;

        // ── Side wall ──
        for (0..seg + 1) |s| {
            const t = 2.0 * math.pi * @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(seg));
            const cx = @cos(t);
            const cz = @sin(t);
            const u = @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(seg));
            verts[vi] = .{ .position = .{ cx * r, hy, cz * r }, .normal = .{ cx, 0, cz }, .uv = .{ u, 0 } };
            vi += 1;
            verts[vi] = .{ .position = .{ cx * r, -hy, cz * r }, .normal = .{ cx, 0, cz }, .uv = .{ u, 1 } };
            vi += 1;
        }
        for (0..seg) |s| {
            const a: u32 = @as(u32, @intCast(s)) * 2;
            indices[ii] = a;
            indices[ii + 1] = a + 1;
            indices[ii + 2] = a + 2;
            indices[ii + 3] = a + 2;
            indices[ii + 4] = a + 1;
            indices[ii + 5] = a + 3;
            ii += 6;
        }

        // ── Top cap (fan) ──
        const top_center: u32 = @intCast(vi);
        verts[vi] = .{ .position = .{ 0, hy, 0 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0.5, 0.5 } };
        vi += 1;
        const top_ring: u32 = @intCast(vi);
        for (0..seg + 1) |s| {
            const t = 2.0 * math.pi * @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(seg));
            verts[vi] = .{ .position = .{ @cos(t) * r, hy, @sin(t) * r }, .normal = .{ 0, 1, 0 }, .uv = .{ 0.5 + @cos(t) * 0.5, 0.5 + @sin(t) * 0.5 } };
            vi += 1;
        }
        for (0..seg) |s| {
            indices[ii] = top_center;
            indices[ii + 1] = top_ring + @as(u32, @intCast(s));
            indices[ii + 2] = top_ring + @as(u32, @intCast(s)) + 1;
            ii += 3;
        }

        // ── Bottom cap (fan, reversed winding) ──
        const bot_center: u32 = @intCast(vi);
        verts[vi] = .{ .position = .{ 0, -hy, 0 }, .normal = .{ 0, -1, 0 }, .uv = .{ 0.5, 0.5 } };
        vi += 1;
        const bot_ring: u32 = @intCast(vi);
        for (0..seg + 1) |s| {
            const t = 2.0 * math.pi * @as(f32, @floatFromInt(s)) / @as(f32, @floatFromInt(seg));
            verts[vi] = .{ .position = .{ @cos(t) * r, -hy, @sin(t) * r }, .normal = .{ 0, -1, 0 }, .uv = .{ 0.5 + @cos(t) * 0.5, 0.5 + @sin(t) * 0.5 } };
            vi += 1;
        }
        for (0..seg) |s| {
            indices[ii] = bot_center;
            indices[ii + 1] = bot_ring + @as(u32, @intCast(s)) + 1;
            indices[ii + 2] = bot_ring + @as(u32, @intCast(s));
            ii += 3;
        }

        const prims = try allocator.alloc(Primitive, 1);
        try recalculateTangents(allocator, verts, indices);
        prims[0] = try packPrimitive(allocator, verts, indices, null);
        return .{ .primitives = prims, .name = "cylinder" };
    }
};
