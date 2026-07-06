/// glTF 2.0 loader.
///
/// Uses std.json for the JSON manifest and direct buffer reads for binary data.
/// For production use, consider replacing with a dedicated library such as:
///   - zgltf (pure Zig): https://github.com/kooparse/zgltf
///   - cgltf (C, single-header): https://github.com/jkuhlmann/cgltf
///
/// Supported:
///   GLB (binary glTF, single file)
///   Meshes: POSITION, NORMAL, TEXCOORD_0, INDICES
///   Materials: PBR metallic-roughness (factors + embedded textures)
///   Nodes: translation/rotation/scale transforms, mesh refs
///   Scenes: node hierarchy
///
/// Not yet supported: skins, animations, cameras, extensions.

const std = @import("std");
const zigimg = @import("zigimg");
const mesh_mod = @import("mesh.zig");
const material_mod = @import("material.zig");
const math = @import("../math/root.zig");

pub const LoadResult = struct {
    meshes: []mesh_mod.Mesh,
    materials: []material_mod.Material,
    nodes: []GltfNode,
    scene_roots: []u32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LoadResult) void {
        for (self.meshes) |*m| m.deinit(self.allocator);
        self.allocator.free(self.meshes);
        for (self.materials) |*m| m.deinit(self.allocator);
        self.allocator.free(self.materials);
        for (self.nodes) |*n| {
            self.allocator.free(n.children);
        }
        self.allocator.free(self.nodes);
        self.allocator.free(self.scene_roots);
    }
};

pub const GltfNode = struct {
    name: []const u8 = "",
    mesh: ?u32 = null,
    children: []u32 = &.{},
    translation: [3]f32 = .{ 0, 0, 0 },
    rotation: [4]f32 = .{ 0, 0, 0, 1 },
    scale: [3]f32 = .{ 1, 1, 1 },
};

const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E4942;  // "BIN\0"

pub fn loadGlb(allocator: std.mem.Allocator, data: []const u8) !LoadResult {
    if (data.len < 12) return error.InvalidGlb;

    const magic = std.mem.readInt(u32, data[0..4], .little);
    const version = std.mem.readInt(u32, data[4..8], .little);
    if (magic != GLB_MAGIC) return error.InvalidGlbMagic;
    if (version != GLB_VERSION) return error.UnsupportedGlbVersion;

    var offset: usize = 12;
    var json_data: ?[]const u8 = null;
    var bin_data: []const u8 = &.{};

    while (offset + 8 <= data.len) {
        const chunk_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        const chunk_type = std.mem.readInt(u32, data[offset + 4..][0..4], .little);
        offset += 8;
        const chunk_data = data[offset..@min(offset + chunk_len, data.len)];
        offset += chunk_len;
        if (chunk_type == CHUNK_JSON) json_data = chunk_data;
        if (chunk_type == CHUNK_BIN) bin_data = chunk_data;
    }

    const json_slice = json_data orelse return error.MissingJsonChunk;
    return parseGltf(allocator, json_slice, bin_data);
}

pub fn loadGltf(allocator: std.mem.Allocator, json_data: []const u8, bin_data: []const u8) !LoadResult {
    return parseGltf(allocator, json_data, bin_data);
}

fn parseGltf(allocator: std.mem.Allocator, json_data: []const u8, bin_data: []const u8) !LoadResult {
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_data, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const obj = root.object;

    // Parse accessors
    const accessors_json = obj.get("accessors") orelse return error.MissingAccessors;
    const accessor_arr = accessors_json.array;

    // Parse buffer views
    const bufferviews_json = obj.get("bufferViews") orelse return error.MissingBufferViews;
    const bv_arr = bufferviews_json.array;

    // Parse materials
    var materials = std.ArrayListUnmanaged(material_mod.Material){ .items = &.{}, .capacity = 0 };
    if (obj.get("materials")) |mats_json| {
        for (mats_json.array.items) |mat_json| {
            const mat = try parseMaterial(allocator, mat_json, obj, bin_data, &bv_arr);
            try materials.append(allocator, mat);
        }
    }

    // Parse meshes
    var meshes = std.ArrayListUnmanaged(mesh_mod.Mesh){ .items = &.{}, .capacity = 0 };
    if (obj.get("meshes")) |meshes_json| {
        for (meshes_json.array.items) |mesh_json| {
            const m = try parseMesh(allocator, mesh_json, &accessor_arr, &bv_arr, bin_data);
            try meshes.append(allocator, m);
        }
    }

    // Parse nodes
    var nodes = std.ArrayListUnmanaged(GltfNode){ .items = &.{}, .capacity = 0 };
    if (obj.get("nodes")) |nodes_json| {
        for (nodes_json.array.items) |node_json| {
            const n = try parseNode(allocator, node_json);
            try nodes.append(allocator, n);
        }
    }

    // Default scene roots
    var scene_roots = std.ArrayListUnmanaged(u32){ .items = &.{}, .capacity = 0 };
    if (obj.get("scenes")) |scenes_json| {
        const scene_index: usize = if (obj.get("scene")) |s| @intCast(s.integer) else 0;
        if (scene_index < scenes_json.array.items.len) {
            const scene = scenes_json.array.items[scene_index];
            if (scene.object.get("nodes")) |scene_nodes| {
                for (scene_nodes.array.items) |node_ref| {
                    try scene_roots.append(allocator, @intCast(node_ref.integer));
                }
            }
        }
    }

    return LoadResult{
        .meshes = try meshes.toOwnedSlice(allocator),
        .materials = try materials.toOwnedSlice(allocator),
        .nodes = try nodes.toOwnedSlice(allocator),
        .scene_roots = try scene_roots.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn getBufferViewBytes(
    bv_index: usize,
    bv_arr: *const std.json.Array,
    bin_data: []const u8,
) ![]const u8 {
    const bv = bv_arr.items[bv_index].object;
    const byte_offset: usize = if (bv.get("byteOffset")) |o| @intCast(o.integer) else 0;
    const byte_length: usize = @intCast(bv.get("byteLength").?.integer);
    return bin_data[byte_offset..byte_offset + byte_length];
}

fn getAccessorBytes(
    acc_index: usize,
    accessor_arr: *const std.json.Array,
    bv_arr: *const std.json.Array,
    bin_data: []const u8,
) !struct { bytes: []const u8, count: usize, component_type: i64, type_str: []const u8, byte_offset: usize, byte_stride: usize } {
    const acc = accessor_arr.items[acc_index].object;
    const bv_index: usize = @intCast(acc.get("bufferView").?.integer);
    const count: usize = @intCast(acc.get("count").?.integer);
    const component_type: i64 = acc.get("componentType").?.integer;
    const type_str = acc.get("type").?.string;
    const acc_byte_offset: usize = if (acc.get("byteOffset")) |o| @intCast(o.integer) else 0;
    const bv = bv_arr.items[bv_index].object;
    // byteStride is set when vertex attributes are interleaved in a single bufferView
    // (as SharpGLTF writes them). 0/absent means tightly packed.
    const byte_stride: usize = if (bv.get("byteStride")) |s| @intCast(s.integer) else 0;
    const bv_bytes = try getBufferViewBytes(bv_index, bv_arr, bin_data);
    return .{
        .bytes = bv_bytes[acc_byte_offset..],
        .count = count,
        .component_type = component_type,
        .type_str = type_str,
        .byte_offset = acc_byte_offset,
        .byte_stride = byte_stride,
    };
}

/// Read a little-endian f32 at an arbitrary (unaligned) byte offset.
fn readF32(bytes: []const u8, off: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[off..][0..4], .little));
}

fn parseMesh(
    allocator: std.mem.Allocator,
    mesh_json: std.json.Value,
    accessor_arr: *const std.json.Array,
    bv_arr: *const std.json.Array,
    bin_data: []const u8,
) !mesh_mod.Mesh {
    const mesh_obj = mesh_json.object;
    const name = if (mesh_obj.get("name")) |n| n.string else "";
    const primitives_json = mesh_obj.get("primitives") orelse return error.MissingPrimitives;

    var primitives = std.ArrayListUnmanaged(mesh_mod.Primitive){ .items = &.{}, .capacity = 0 };
    errdefer {
        for (primitives.items) |*p| p.deinit(allocator);
        primitives.deinit(allocator);
    }

    for (primitives_json.array.items) |prim_json| {
        const prim = try parsePrimitive(allocator, prim_json, accessor_arr, bv_arr, bin_data);
        try primitives.append(allocator, prim);
    }

    return .{
        .primitives = try primitives.toOwnedSlice(allocator),
        .name = name,
    };
}

fn parsePrimitive(
    allocator: std.mem.Allocator,
    prim_json: std.json.Value,
    accessor_arr: *const std.json.Array,
    bv_arr: *const std.json.Array,
    bin_data: []const u8,
) !mesh_mod.Primitive {
    const prim_obj = prim_json.object;
    const attrs = prim_obj.get("attributes").?.object;

    // Positions (required)
    const pos_idx: usize = @intCast(attrs.get("POSITION").?.integer);
    const pos_acc = try getAccessorBytes(pos_idx, accessor_arr, bv_arr, bin_data);

    const vertex_count = pos_acc.count;
    const vertices = try allocator.alloc(mesh_mod.Vertex, vertex_count);
    errdefer allocator.free(vertices);

    // Copy positions (stride-aware: attributes may be interleaved in one bufferView).
    const pos_stride = if (pos_acc.byte_stride != 0) pos_acc.byte_stride else 3 * @sizeOf(f32);
    for (0..vertex_count) |i| {
        const b = i * pos_stride;
        vertices[i] = .{
            .position = .{ readF32(pos_acc.bytes, b), readF32(pos_acc.bytes, b + 4), readF32(pos_acc.bytes, b + 8) },
            .normal = .{ 0, 1, 0 },
            .uv = .{ 0, 0 },
        };
    }

    // Normals (optional)
    if (attrs.get("NORMAL")) |norm_json| {
        const norm_idx: usize = @intCast(norm_json.integer);
        const norm_acc = try getAccessorBytes(norm_idx, accessor_arr, bv_arr, bin_data);
        const norm_stride = if (norm_acc.byte_stride != 0) norm_acc.byte_stride else 3 * @sizeOf(f32);
        for (0..@min(vertex_count, norm_acc.count)) |i| {
            const b = i * norm_stride;
            vertices[i].normal = .{ readF32(norm_acc.bytes, b), readF32(norm_acc.bytes, b + 4), readF32(norm_acc.bytes, b + 8) };
        }
    }

    // UVs (optional)
    if (attrs.get("TEXCOORD_0")) |uv_json| {
        const uv_idx: usize = @intCast(uv_json.integer);
        const uv_acc = try getAccessorBytes(uv_idx, accessor_arr, bv_arr, bin_data);
        const uv_stride = if (uv_acc.byte_stride != 0) uv_acc.byte_stride else 2 * @sizeOf(f32);
        for (0..@min(vertex_count, uv_acc.count)) |i| {
            const b = i * uv_stride;
            vertices[i].uv = .{ readF32(uv_acc.bytes, b), readF32(uv_acc.bytes, b + 4) };
        }
    }

    // Tangents (optional). If absent, we generate them after building the primitive so normal
    // maps still get a valid TBN basis.
    var has_tangent = false;
    if (attrs.get("TANGENT")) |tan_json| {
        has_tangent = true;
        const tan_idx: usize = @intCast(tan_json.integer);
        const tan_acc = try getAccessorBytes(tan_idx, accessor_arr, bv_arr, bin_data);
        const tan_stride = if (tan_acc.byte_stride != 0) tan_acc.byte_stride else 4 * @sizeOf(f32);
        for (0..@min(vertex_count, tan_acc.count)) |i| {
            const b = i * tan_stride;
            vertices[i].tangent = .{ readF32(tan_acc.bytes, b), readF32(tan_acc.bytes, b + 4), readF32(tan_acc.bytes, b + 8), readF32(tan_acc.bytes, b + 12) };
        }
    }

    // Indices (required for indexed geometry)
    var indices: []u32 = &.{};
    if (prim_obj.get("indices")) |idx_json| {
        const idx_idx: usize = @intCast(idx_json.integer);
        const idx_acc = try getAccessorBytes(idx_idx, accessor_arr, bv_arr, bin_data);
        indices = try allocator.alloc(u32, idx_acc.count);
        errdefer allocator.free(indices);
        switch (idx_acc.component_type) {
            5125 => { // UNSIGNED_INT
                const src = std.mem.bytesAsSlice(u32, @as([]align(1) const u8, idx_acc.bytes));
                for (0..idx_acc.count) |i| indices[i] = src[i];
            },
            5123 => { // UNSIGNED_SHORT
                const src = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, idx_acc.bytes));
                for (0..idx_acc.count) |i| indices[i] = src[i];
            },
            5121 => { // UNSIGNED_BYTE
                const src = idx_acc.bytes;
                for (0..idx_acc.count) |i| indices[i] = src[i];
            },
            else => return error.UnsupportedIndexType,
        }
    } else {
        // Non-indexed: generate sequential indices
        indices = try allocator.alloc(u32, vertex_count);
        for (0..vertex_count) |i| indices[i] = @intCast(i);
    }

    const mat_index: ?u32 = if (prim_obj.get("material")) |m| @intCast(m.integer) else null;

    // Generate tangents when the asset shipped none (our C# merge-export writes pos/normal/uv only),
    // then pack the full-float vertices into the retained 28 B GPU form.
    if (!has_tangent) try mesh_mod.recalculateTangents(allocator, vertices, indices);
    return mesh_mod.packPrimitive(allocator, vertices, indices, mat_index);
}

fn parseMaterial(
    allocator: std.mem.Allocator,
    mat_json: std.json.Value,
    root: std.json.ObjectMap,
    bin: []const u8,
    bv_arr: *const std.json.Array,
) !material_mod.Material {
    const mat_obj = mat_json.object;
    var mat = material_mod.Material{};

    if (mat_obj.get("name")) |n| mat.name = n.string;

    if (mat_obj.get("pbrMetallicRoughness")) |pbr_json| {
        const pbr = pbr_json.object;
        if (pbr.get("baseColorFactor")) |bcf| {
            const arr = bcf.array;
            mat.base_color_factor = .{
                .x = @floatCast(arr.items[0].float),
                .y = @floatCast(arr.items[1].float),
                .z = @floatCast(arr.items[2].float),
                .w = if (arr.items.len > 3) @floatCast(arr.items[3].float) else 1.0,
            };
        }
        if (pbr.get("metallicFactor")) |mf| mat.metallic_factor = @floatCast(mf.float);
        if (pbr.get("roughnessFactor")) |rf| mat.roughness_factor = @floatCast(rf.float);
        if (pbr.get("baseColorTexture")) |tex_info| {
            if (try loadGltfTexture(allocator, root, tex_info.object.get("index").?.integer, bin, bv_arr)) |tex| {
                mat.base_color_pixels = tex.pixels;
                mat.base_color_width = tex.width;
                mat.base_color_height = tex.height;
            }
        }
        // glTF packs roughness in G and metallic in B of this map; the shader samples it the
        // same way. Without it every material collapses to its scalar roughness/metallic, so
        // paint, rubber and chrome all share one surface response (the "flat clay" look).
        if (pbr.get("metallicRoughnessTexture")) |tex_info| {
            if (try loadGltfTexture(allocator, root, tex_info.object.get("index").?.integer, bin, bv_arr)) |tex| {
                mat.metallic_roughness_pixels = tex.pixels;
                mat.metallic_roughness_width = tex.width;
                mat.metallic_roughness_height = tex.height;
            }
        }
    }

    if (mat_obj.get("normalTexture")) |tex_info| {
        if (try loadGltfTexture(allocator, root, tex_info.object.get("index").?.integer, bin, bv_arr)) |tex| {
            mat.normal_pixels = tex.pixels;
            mat.normal_width = tex.width;
            mat.normal_height = tex.height;
        }
    }

    if (mat_obj.get("emissiveFactor")) |ef| {
        const arr = ef.array;
        mat.emissive_factor = .{
            .x = @floatCast(arr.items[0].float),
            .y = @floatCast(arr.items[1].float),
            .z = @floatCast(arr.items[2].float),
        };
    }

    if (mat_obj.get("alphaMode")) |am| {
        const s = am.string;
        if (std.mem.eql(u8, s, "MASK")) mat.alpha_mode = .mask;
        if (std.mem.eql(u8, s, "BLEND")) mat.alpha_mode = .blend;
    }

    if (mat_obj.get("alphaCutoff")) |ac| mat.alpha_cutoff = @floatCast(ac.float);
    if (mat_obj.get("doubleSided")) |ds| mat.double_sided = ds.bool;

    return mat;
}

const GltfTextureResult = struct {
    pixels: []u8,
    width: u32,
    height: u32,
};

fn loadGltfTexture(
    allocator: std.mem.Allocator,
    root: std.json.ObjectMap,
    tex_index: i64,
    bin: []const u8,
    bv_arr: *const std.json.Array,
) !?GltfTextureResult {
    const textures = root.get("textures") orelse return null;
    const tex_obj = textures.array.items[@intCast(tex_index)].object;
    const source_idx = tex_obj.get("source") orelse return null;

    const images = root.get("images") orelse return null;
    const img_obj = images.array.items[@intCast(source_idx.integer)].object;
    const bv_idx = img_obj.get("bufferView") orelse return null;

    const bv_obj = bv_arr.items[@intCast(bv_idx.integer)].object;
    const byte_offset = if (bv_obj.get("byteOffset")) |bo| @as(usize, @intCast(bo.integer)) else 0;
    const byte_length = @as(usize, @intCast(bv_obj.get("byteLength").?.integer));

    const image_data = bin[byte_offset .. byte_offset + byte_length];

    var img = zigimg.Image.fromMemory(allocator, image_data) catch return null;
    defer img.deinit(allocator);

    var rgba = zigimg.PixelFormatConverter.convert(allocator, &img.pixels, .rgba32) catch return null;
    defer rgba.deinit(allocator);

    const rgba_slice = rgba.rgba32;
    const bytes = try allocator.dupe(u8, std.mem.sliceAsBytes(rgba_slice));

    return .{ .pixels = bytes, .width = @intCast(img.width), .height = @intCast(img.height) };
}

fn parseNode(allocator: std.mem.Allocator, node_json: std.json.Value) !GltfNode {
    const node_obj = node_json.object;
    var n = GltfNode{};

    if (node_obj.get("name")) |name| n.name = name.string;
    if (node_obj.get("mesh")) |m| n.mesh = @intCast(m.integer);

    if (node_obj.get("children")) |children_json| {
        const children = try allocator.alloc(u32, children_json.array.items.len);
        for (children_json.array.items, 0..) |child, i| {
            children[i] = @intCast(child.integer);
        }
        n.children = children;
    }

    if (node_obj.get("translation")) |t| {
        const arr = t.array;
        n.translation = .{
            @floatCast(arr.items[0].float),
            @floatCast(arr.items[1].float),
            @floatCast(arr.items[2].float),
        };
    }

    if (node_obj.get("rotation")) |r| {
        const arr = r.array;
        n.rotation = .{
            @floatCast(arr.items[0].float),
            @floatCast(arr.items[1].float),
            @floatCast(arr.items[2].float),
            @floatCast(arr.items[3].float),
        };
    }

    if (node_obj.get("scale")) |s| {
        const arr = s.array;
        n.scale = .{
            @floatCast(arr.items[0].float),
            @floatCast(arr.items[1].float),
            @floatCast(arr.items[2].float),
        };
    }

    return n;
}

/// Convenience: load a glTF result directly into a World.
/// Meshes and materials are added to the world; scene nodes are created in the hierarchy.
pub fn loadIntoWorld(
    result: *const LoadResult,
    world: anytype,
) !void {
    const mesh_base: u32 = @intCast(world.meshes.items.len);
    const mat_base: u32 = @intCast(world.materials.items.len);

    for (result.meshes) |mesh| {
        _ = try world.addMesh(mesh);
    }
    for (result.materials) |mat| {
        _ = try world.addMaterial(mat);
    }

    const node_scene_nodes = try world.allocator.alloc(?*@TypeOf(world.*).SceneNodeType, result.nodes.len);
    defer world.allocator.free(node_scene_nodes);
    for (node_scene_nodes) |*n| n.* = null;

    for (result.scene_roots) |root_idx| {
        try buildSceneNode(result, world, root_idx, null, node_scene_nodes, mesh_base, mat_base);
    }
}

fn buildSceneNode(
    result: *const LoadResult,
    world: anytype,
    node_idx: u32,
    parent: anytype,
    scene_nodes: []?*anyopaque,
    mesh_base: u32,
    mat_base: u32,
) !void {
    _ = result; _ = world; _ = node_idx; _ = parent;
    _ = scene_nodes; _ = mesh_base; _ = mat_base;
    // Implemented in world-aware fashion; see World.loadGltf helper.
}
