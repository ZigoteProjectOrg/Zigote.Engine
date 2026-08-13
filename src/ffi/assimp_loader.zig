/// Assimp-backed model importer for the C# editor.
///
/// Imports any format Assimp understands (glTF/GLB, FBX, OBJ, DAE, PLY, STL, 3DS, …),
/// processes the geometry (triangulate, generate normals/tangents), and emits:
///   • one `.zmesh` binary per mesh (engine Vertex layout) written into the cache dir
///   • extracted embedded textures into the cache dir
///   • a JSON manifest describing the node tree, materials, lights and animations
///
/// The C# `GltfLoader` consumes the manifest to build its `SceneNode` tree, applying the
/// same material heuristics it always has. Geometry never round-trips through C#: native
/// writes the `.zmesh` files and C# uploads them back via `zigote_scene_set_mesh_blob`.
const std = @import("std");
const zg = @import("zigote");
const Vertex = zg.resources.Vertex;
const Mat4 = zg.Mat4;
const Vec3 = zg.Vec3;

const c = @cImport({
    @cInclude("assimp/cimport.h");
    @cInclude("assimp/scene.h");
    @cInclude("assimp/postprocess.h");
    @cInclude("assimp/material.h");
});

// .zmesh interchange format — lives in engine/resources/zmesh_format.zig (Assimp-free) so
// the parser stays available in `-Denable3d=false` builds that compile this importer out.
const writeZmesh = zg.resources.writeZmesh;

// ── JSON manifest schema (serialized via std.json.Stringify) ────────────────

const JCounts = struct {
    meshes: usize,
    materials: usize,
    textures: usize,
    lights: usize,
    animations: usize,
    nodes: usize,
    primitives: usize,
};

const JMaterial = struct {
    name: []const u8,
    baseColor: [4]f32,
    metallic: f32,
    roughness: f32,
    hasMetallicRoughness: bool,
    emissive: [3]f32,
    emissiveStrength: f32,
    alphaMode: []const u8, // "OPAQUE" | "MASK" | "BLEND"
    alphaCutoff: f32,
    doubleSided: bool,
    unlit: bool,
    clearcoat: f32,
    clearcoatRoughness: f32,
    ior: f32,
    specular: f32,
    transmission: f32,
    // KHR_materials_volume / _sheen — carried in the manifest for forward-compat (the sheen lobe
    // is not rendered yet; thickness may refine the glass refraction later).
    thickness: f32,
    sheenColor: [3]f32,
    sheenRoughness: f32,
    baseColorTexture: ?[]const u8,
    metallicRoughnessTexture: ?[]const u8,
    normalTexture: ?[]const u8,
    emissiveTexture: ?[]const u8,
    occlusionTexture: ?[]const u8,
};

/// A mesh reference inside a hierarchical (animated) node: a `.zmesh` cache path + material index.
const JMeshRef = struct { cache: []const u8, material: i64 };

/// A node in the preserved hierarchy (animated import). Local TRS; channels bind by `name`.
const JNode = struct {
    name: []const u8,
    translation: [3]f32,
    rotation: [4]f32, // x, y, z, w
    scale: [3]f32,
    meshes: []const JMeshRef,
    children: []const JNode,
};

/// A flattened-by-material mesh node (static import). World-space geometry baked into the `.zmesh`.
const JMeshNode = struct { name: []const u8, cache: []const u8, material: i64 };

const JLight = struct {
    name: []const u8,
    kind: []const u8, // "directional" | "point" | "spot"
    color: [3]f32,
    intensity: f32,
    range: f32,
    position: [3]f32,
    rotation: [4]f32,
};

const JChannel = struct {
    node: []const u8,
    path: []const u8, // "translation" | "rotation" | "scale"
    interpolation: []const u8, // "LINEAR" | "STEP"
    times: []const f32,
    values: []const f32, // flattened: vec3 → 3/key, quat (x,y,z,w) → 4/key
};

const JAnimation = struct { name: []const u8, channels: []const JChannel };

const JManifest = struct {
    animated: bool,
    counts: JCounts,
    warnings: []const []const u8,
    nodes: []const JNode, // populated when animated
    meshNodes: []const JMeshNode, // populated when static
    materials: []const JMaterial,
    lights: []const JLight,
    animations: []const JAnimation,
};

// ── Import context ──────────────────────────────────────────────────────────

const Ctx = struct {
    arena: std.mem.Allocator,
    io: std.Io,
    scene: *const c.aiScene,
    cache_dir: []const u8,
    model_dir: []const u8,
    base: []const u8,
    warnings: *std.ArrayListUnmanaged([]const u8),
    node_count: usize = 0,
    prim_count: usize = 0,
    mesh_seq: usize = 0,
};

fn warn(ctx: *Ctx, comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.allocPrint(ctx.arena, fmt, args) catch return;
    ctx.warnings.append(ctx.arena, s) catch {};
}

// ── Assimp helpers ──────────────────────────────────────────────────────────

fn aiStr(s: *const c.aiString) []const u8 {
    const p: [*]const u8 = @ptrCast(&s.data);
    return p[0..@as(usize, s.length)];
}

inline fn meshAt(sc: *const c.aiScene, i: usize) *const c.aiMesh {
    return @ptrCast(sc.mMeshes[i]);
}

inline fn nodeChild(n: *const c.aiNode, i: usize) *const c.aiNode {
    return @ptrCast(n.mChildren[i]);
}

/// Convert a (row-major) Assimp matrix into the engine's column-major `Mat4`.
fn mat4FromAi(m: *const c.aiMatrix4x4) Mat4 {
    var out = Mat4.identity;
    out.setElement(0, 0, m.a1);
    out.setElement(1, 0, m.a2);
    out.setElement(2, 0, m.a3);
    out.setElement(3, 0, m.a4);
    out.setElement(0, 1, m.b1);
    out.setElement(1, 1, m.b2);
    out.setElement(2, 1, m.b3);
    out.setElement(3, 1, m.b4);
    out.setElement(0, 2, m.c1);
    out.setElement(1, 2, m.c2);
    out.setElement(2, 2, m.c3);
    out.setElement(3, 2, m.c4);
    out.setElement(0, 3, m.d1);
    out.setElement(1, 3, m.d2);
    out.setElement(2, 3, m.d3);
    out.setElement(3, 3, m.d4);
    return out;
}

/// Find the world transform (as an Assimp matrix) of the first node named `name`, or null.
fn searchWorldAi(node: *const c.aiNode, parent: c.aiMatrix4x4, name: []const u8) ?c.aiMatrix4x4 {
    var world = parent;
    c.aiMultiplyMatrix4(&world, &node.mTransformation); // world = parent * local
    if (std.mem.eql(u8, aiStr(&node.mName), name)) return world;
    for (0..node.mNumChildren) |i| {
        if (searchWorldAi(nodeChild(node, i), world, name)) |w| return w;
    }
    return null;
}

// ── Geometry ────────────────────────────────────────────────────────────────

const Geo = struct { verts: []Vertex, idx: []u32 };

const Accum = struct {
    verts: std.ArrayListUnmanaged(Vertex) = .{ .items = &.{}, .capacity = 0 },
    idx: std.ArrayListUnmanaged(u32) = .{ .items = &.{}, .capacity = 0 },
    used: bool = false,
};

/// Convert one `aiMesh` into engine vertices/indices (local space). Returns null for empty or
/// non-triangle meshes (points/lines are skipped — only triangle lists are rendered).
fn convertLocal(ctx: *Ctx, m: *const c.aiMesh) !?Geo {
    const n: usize = m.mNumVertices;
    if (n == 0 or m.mVertices == null) return null;

    const has_normals = m.mNormals != null;
    const has_uv = m.mTextureCoords[0] != null;
    const has_tan = m.mTangents != null;
    const has_bitan = m.mBitangents != null;

    const verts = try ctx.arena.alloc(Vertex, n);
    for (0..n) |i| {
        var v = Vertex{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 1, 0 }, .uv = .{ 0, 0 }, .tangent = .{ 0, 0, 0, 1 } };
        const p = m.mVertices[i];
        v.position = .{ p.x, p.y, p.z };
        if (has_normals) {
            const nn = m.mNormals[i];
            v.normal = .{ nn.x, nn.y, nn.z };
        }
        if (has_uv) {
            const t = m.mTextureCoords[0][i];
            v.uv = .{ t.x, t.y };
        }
        if (has_tan) {
            const tg = m.mTangents[i];
            var w: f32 = 1.0;
            if (has_bitan and has_normals) {
                const nv = Vec3{ .x = m.mNormals[i].x, .y = m.mNormals[i].y, .z = m.mNormals[i].z };
                const tv = Vec3{ .x = tg.x, .y = tg.y, .z = tg.z };
                const bv = Vec3{ .x = m.mBitangents[i].x, .y = m.mBitangents[i].y, .z = m.mBitangents[i].z };
                if (nv.cross(tv).dot(bv) < 0.0) w = -1.0;
            }
            v.tangent = .{ tg.x, tg.y, tg.z, w };
        }
        verts[i] = v;
    }

    var idx: std.ArrayListUnmanaged(u32) = .{ .items = &.{}, .capacity = 0 };
    // aiProcess_Triangulate makes every face a triangle, so this reserves the exact index count up
    // front — the append loop below never grows-and-copies. (A hint: appends still work if it's short.)
    idx.ensureTotalCapacity(ctx.arena, @as(usize, m.mNumFaces) * 3) catch {};
    for (0..m.mNumFaces) |f| {
        const face = m.mFaces[f];
        if (face.mNumIndices != 3) continue;
        try idx.append(ctx.arena, face.mIndices[0]);
        try idx.append(ctx.arena, face.mIndices[1]);
        try idx.append(ctx.arena, face.mIndices[2]);
    }
    if (idx.items.len == 0) return null;
    return Geo{ .verts = verts, .idx = try idx.toOwnedSlice(ctx.arena) };
}

// ── Static path: bake world transforms, group by material ───────────────────

/// Write a `.zmesh` cache file, staging the blob in a short-lived allocation freed immediately after
/// the write — so peak import memory holds one mesh blob at a time instead of retaining every group's
/// blob in the import-lifetime arena until the whole import returns.
fn writeMeshCache(io: std.Io, path: []const u8, verts: []const Vertex, idx: []const u32) !void {
    const blob = try writeZmesh(std.heap.c_allocator, verts, idx);
    defer std.heap.c_allocator.free(blob);
    try writeFileBytes(io, path, blob);
}

/// Sum per-material vertex/index totals over the node tree (mirrors walkStatic's traversal, counting
/// each instanced mesh reference once per referencing node) so the Accum group buffers can be pre-sized
/// once instead of grown geometrically. Over-counts only meshes convertLocal later skips (rare after
/// triangulation) — harmless, since the counts feed ensureTotalCapacity as a hint.
fn countStatic(ctx: *Ctx, node: *const c.aiNode, vcount: []usize, icount: []usize) void {
    for (0..node.mNumMeshes) |i| {
        const mi: usize = @intCast(node.mMeshes[i]);
        const m = meshAt(ctx.scene, mi);
        const mat_idx: usize = @intCast(m.mMaterialIndex);
        if (mat_idx >= vcount.len) continue;
        vcount[mat_idx] += m.mNumVertices;
        icount[mat_idx] += @as(usize, m.mNumFaces) * 3;
    }
    for (0..node.mNumChildren) |ci| countStatic(ctx, nodeChild(node, ci), vcount, icount);
}

fn walkStatic(ctx: *Ctx, node: *const c.aiNode, parent_world: Mat4, groups: []Accum) void {
    ctx.node_count += 1;
    const world = parent_world.mul(mat4FromAi(&node.mTransformation));
    const normal_mat = world.inverse().transpose();

    for (0..node.mNumMeshes) |i| {
        const mi: usize = @intCast(node.mMeshes[i]);
        const m = meshAt(ctx.scene, mi);
        const geo = (convertLocal(ctx, m) catch null) orelse continue;
        ctx.prim_count += 1;
        const mat_idx: usize = @intCast(m.mMaterialIndex);
        if (mat_idx >= groups.len) continue;
        const g = &groups[mat_idx];
        const base_v: u32 = @intCast(g.verts.items.len);
        for (geo.verts) |v| {
            var wv = v;
            const wp = world.mulPoint(.{ .x = v.position[0], .y = v.position[1], .z = v.position[2] });
            wv.position = .{ wp.x, wp.y, wp.z };
            const wn = normal_mat.mulDirection(.{ .x = v.normal[0], .y = v.normal[1], .z = v.normal[2] }).normalize();
            wv.normal = .{ wn.x, wn.y, wn.z };
            const wt = world.mulDirection(.{ .x = v.tangent[0], .y = v.tangent[1], .z = v.tangent[2] }).normalize();
            wv.tangent = .{ wt.x, wt.y, wt.z, v.tangent[3] };
            g.verts.append(ctx.arena, wv) catch return;
        }
        for (geo.idx) |ix| g.idx.append(ctx.arena, base_v + ix) catch return;
        g.used = true;
    }

    for (0..node.mNumChildren) |ci| walkStatic(ctx, nodeChild(node, ci), world, groups);
}

fn emitMeshNodes(ctx: *Ctx, groups: []Accum, materials: []const JMaterial) ![]const JMeshNode {
    var out: std.ArrayListUnmanaged(JMeshNode) = .{ .items = &.{}, .capacity = 0 };
    for (groups, 0..) |*g, mi| {
        if (!g.used or g.verts.items.len == 0) continue;
        const cache_path = matCachePath(ctx, mi) catch continue;
        writeMeshCache(ctx.io, cache_path, g.verts.items, g.idx.items) catch |e| {
            warn(ctx, "failed to write mesh cache {s}: {}", .{ cache_path, e });
            continue;
        };
        const name = if (mi < materials.len and materials[mi].name.len > 0)
            materials[mi].name
        else
            std.fmt.allocPrint(ctx.arena, "Material {d}", .{mi}) catch "Material";
        try out.append(ctx.arena, .{ .name = name, .cache = cache_path, .material = @intCast(mi) });
    }
    return out.toOwnedSlice(ctx.arena);
}

// ── Animated path: preserve hierarchy, local-space per-primitive meshes ──────

fn buildNode(ctx: *Ctx, node: *const c.aiNode) anyerror!JNode {
    ctx.node_count += 1;
    var scaling: c.aiVector3D = undefined;
    var rotation: c.aiQuaternion = undefined;
    var position: c.aiVector3D = undefined;
    c.aiDecomposeMatrix(&node.mTransformation, &scaling, &rotation, &position);

    var meshes: std.ArrayListUnmanaged(JMeshRef) = .{ .items = &.{}, .capacity = 0 };
    for (0..node.mNumMeshes) |i| {
        const mi: usize = @intCast(node.mMeshes[i]);
        const m = meshAt(ctx.scene, mi);
        const geo = (convertLocal(ctx, m) catch null) orelse continue;
        ctx.prim_count += 1;
        ctx.mesh_seq += 1;
        const cache_path = primCachePath(ctx, ctx.mesh_seq) catch continue;
        writeMeshCache(ctx.io, cache_path, geo.verts, geo.idx) catch continue;
        try meshes.append(ctx.arena, .{ .cache = cache_path, .material = @intCast(m.mMaterialIndex) });
    }

    var children: std.ArrayListUnmanaged(JNode) = .{ .items = &.{}, .capacity = 0 };
    for (0..node.mNumChildren) |ci| {
        try children.append(ctx.arena, try buildNode(ctx, nodeChild(node, ci)));
    }

    return JNode{
        .name = nodeName(ctx, node),
        .translation = .{ position.x, position.y, position.z },
        .rotation = .{ rotation.x, rotation.y, rotation.z, rotation.w },
        .scale = .{ scaling.x, scaling.y, scaling.z },
        .meshes = try meshes.toOwnedSlice(ctx.arena),
        .children = try children.toOwnedSlice(ctx.arena),
    };
}

fn nodeName(ctx: *Ctx, node: *const c.aiNode) []const u8 {
    const s = aiStr(&node.mName);
    if (s.len > 0) return ctx.arena.dupe(u8, s) catch "Node";
    return "Node";
}

// ── Materials ───────────────────────────────────────────────────────────────

fn matColor4(mat: *const c.aiMaterial, key: [*c]const u8) ?[4]f32 {
    var col: c.aiColor4D = undefined;
    if (c.aiGetMaterialColor(mat, key, 0, 0, &col) == c.aiReturn_SUCCESS)
        return .{ col.r, col.g, col.b, col.a };
    return null;
}

fn matFloat(mat: *const c.aiMaterial, key: [*c]const u8) ?f32 {
    var v: f32 = undefined;
    var n: c_uint = 1;
    if (c.aiGetMaterialFloatArray(mat, key, 0, 0, &v, &n) == c.aiReturn_SUCCESS) return v;
    return null;
}

fn matInt(mat: *const c.aiMaterial, key: [*c]const u8) ?i32 {
    var v: c_int = undefined;
    var n: c_uint = 1;
    if (c.aiGetMaterialIntegerArray(mat, key, 0, 0, &v, &n) == c.aiReturn_SUCCESS) return @intCast(v);
    return null;
}

fn matString(ctx: *Ctx, mat: *const c.aiMaterial, key: [*c]const u8) ?[]const u8 {
    var s: c.aiString = undefined;
    if (c.aiGetMaterialString(mat, key, 0, 0, &s) == c.aiReturn_SUCCESS)
        return ctx.arena.dupe(u8, aiStr(&s)) catch null;
    return null;
}

fn matTexture(ctx: *Ctx, mat: *const c.aiMaterial, tex_type: c.aiTextureType) ?[]const u8 {
    if (c.aiGetMaterialTextureCount(mat, tex_type) == 0) return null;
    var path: c.aiString = undefined;
    if (c.aiGetMaterialTexture(mat, tex_type, 0, &path, null, null, null, null, null, null) != c.aiReturn_SUCCESS)
        return null;
    return resolveTexture(ctx, &path);
}

fn buildMaterial(ctx: *Ctx, mat: *const c.aiMaterial) JMaterial {
    const metal_opt = matFloat(mat, "$mat.metallicFactor");
    const rough_opt = matFloat(mat, "$mat.roughnessFactor");
    const emissive4 = matColor4(mat, "$clr.emissive") orelse [4]f32{ 0, 0, 0, 0 };

    const mr_tex = matTexture(ctx, mat, c.aiTextureType_METALNESS) orelse
        matTexture(ctx, mat, c.aiTextureType_DIFFUSE_ROUGHNESS) orelse
        matTexture(ctx, mat, c.aiTextureType_UNKNOWN);

    return JMaterial{
        .name = matString(ctx, mat, "?mat.name") orelse "",
        .baseColor = matColor4(mat, "$clr.base") orelse matColor4(mat, "$clr.diffuse") orelse [4]f32{ 1, 1, 1, 1 },
        .metallic = metal_opt orelse 1.0,
        .roughness = rough_opt orelse 1.0,
        .hasMetallicRoughness = metal_opt != null or rough_opt != null or mr_tex != null,
        .emissive = .{ emissive4[0], emissive4[1], emissive4[2] },
        .emissiveStrength = matFloat(mat, "$mat.emissiveIntensity") orelse 1.0,
        .alphaMode = matString(ctx, mat, "$mat.gltf.alphaMode") orelse "OPAQUE",
        .alphaCutoff = matFloat(mat, "$mat.gltf.alphaCutoff") orelse 0.5,
        .doubleSided = (matInt(mat, "$mat.twosided") orelse 0) != 0,
        .unlit = (matInt(mat, "$mat.shadingm") orelse 0) == c.aiShadingMode_Unlit,
        .clearcoat = matFloat(mat, "$mat.clearcoat.factor") orelse 0.0,
        .clearcoatRoughness = matFloat(mat, "$mat.clearcoat.roughnessFactor") orelse 0.0,
        .ior = matFloat(mat, "$mat.refracti") orelse 1.5,
        .specular = matFloat(mat, "$mat.specularFactor") orelse 1.0,
        .transmission = matFloat(mat, "$mat.transmission.factor") orelse 0.0,
        .thickness = matFloat(mat, "$mat.volume.thicknessFactor") orelse 0.0,
        .sheenColor = blk: {
            const sc = matColor4(mat, "$clr.sheen.factor") orelse [4]f32{ 0, 0, 0, 0 };
            break :blk .{ sc[0], sc[1], sc[2] };
        },
        .sheenRoughness = matFloat(mat, "$mat.sheen.roughnessFactor") orelse 0.0,
        .baseColorTexture = matTexture(ctx, mat, c.aiTextureType_BASE_COLOR) orelse matTexture(ctx, mat, c.aiTextureType_DIFFUSE),
        .metallicRoughnessTexture = mr_tex,
        .normalTexture = matTexture(ctx, mat, c.aiTextureType_NORMALS),
        .emissiveTexture = matTexture(ctx, mat, c.aiTextureType_EMISSION_COLOR) orelse
            matTexture(ctx, mat, c.aiTextureType_EMISSIVE),
        // glTF occlusion imports as LIGHTMAP in Assimp; AMBIENT_OCCLUSION covers FBX/other formats.
        .occlusionTexture = matTexture(ctx, mat, c.aiTextureType_LIGHTMAP) orelse
            matTexture(ctx, mat, c.aiTextureType_AMBIENT_OCCLUSION),
    };
}

fn buildMaterials(ctx: *Ctx) ![]const JMaterial {
    var out: std.ArrayListUnmanaged(JMaterial) = .{ .items = &.{}, .capacity = 0 };
    for (0..ctx.scene.mNumMaterials) |i| {
        const mat: *const c.aiMaterial = @ptrCast(ctx.scene.mMaterials[i]);
        try out.append(ctx.arena, buildMaterial(ctx, mat));
    }
    return out.toOwnedSlice(ctx.arena);
}

// ── Textures ────────────────────────────────────────────────────────────────

fn resolveTexture(ctx: *Ctx, path_ai: *const c.aiString) ?[]const u8 {
    const p = aiStr(path_ai);
    if (p.len == 0) return null;
    if (p[0] == '*') {
        const idx = std.fmt.parseInt(usize, p[1..], 10) catch return null;
        if (idx >= ctx.scene.mNumTextures) return null;
        const tex: *const c.aiTexture = @ptrCast(ctx.scene.mTextures[idx]);
        return writeEmbedded(ctx, idx, tex);
    }
    const resolved = if (std.fs.path.isAbsolute(p))
        (ctx.arena.dupe(u8, p) catch return null)
    else
        (std.fs.path.join(ctx.arena, &.{ ctx.model_dir, p }) catch return null);
    std.Io.Dir.cwd().access(ctx.io, resolved, .{}) catch {
        warn(ctx, "texture '{s}' not found", .{p});
        return null;
    };
    return resolved;
}

fn writeEmbedded(ctx: *Ctx, idx: usize, tex: *const c.aiTexture) ?[]const u8 {
    if (tex.mHeight != 0) {
        // Raw uncompressed ARGB texels — rare (almost all embedded textures are PNG/JPG). Skip
        // rather than pull in an encoder; the surface just loses that one map.
        warn(ctx, "embedded texture {d} is raw (uncompressed); skipped", .{idx});
        return null;
    }
    const hint_arr: *const [9]u8 = @ptrCast(&tex.achFormatHint);
    var ext: []const u8 = std.mem.sliceTo(hint_arr, 0);
    if (ext.len == 0 or ext.len > 4) ext = "png";
    const fname = std.fmt.allocPrint(ctx.arena, "{s}_tex{d}.{s}", .{ ctx.base, idx, ext }) catch return null;
    const path = std.fs.path.join(ctx.arena, &.{ ctx.cache_dir, fname }) catch return null;
    const len: usize = @intCast(tex.mWidth);
    const bytes = @as([*]const u8, @ptrCast(tex.pcData))[0..len];
    writeFileBytes(ctx.io, path, bytes) catch return null;
    return path;
}

// ── Lights ──────────────────────────────────────────────────────────────────

fn buildLights(ctx: *Ctx) ![]const JLight {
    var out: std.ArrayListUnmanaged(JLight) = .{ .items = &.{}, .capacity = 0 };
    var id: c.aiMatrix4x4 = undefined;
    c.aiIdentityMatrix4(&id);

    for (0..ctx.scene.mNumLights) |i| {
        const L: *const c.aiLight = @ptrCast(ctx.scene.mLights[i]);
        var kind: []const u8 = undefined;
        if (L.mType == c.aiLightSource_DIRECTIONAL) {
            kind = "directional";
        } else if (L.mType == c.aiLightSource_POINT) {
            kind = "point";
        } else if (L.mType == c.aiLightSource_SPOT) {
            kind = "spot";
        } else continue;

        const name = ctx.arena.dupe(u8, aiStr(&L.mName)) catch "";
        const world = searchWorldAi(@ptrCast(ctx.scene.mRootNode), id, aiStr(&L.mName)) orelse id;
        var scaling: c.aiVector3D = undefined;
        var rotation: c.aiQuaternion = undefined;
        var position: c.aiVector3D = undefined;
        c.aiDecomposeMatrix(&world, &scaling, &rotation, &position);

        // Assimp folds the glTF intensity into the diffuse colour; split it back out into a
        // unit colour + scalar intensity so the editor can calibrate per-light.
        const cd = L.mColorDiffuse;
        var inten = @max(cd.r, @max(cd.g, cd.b));
        var color = [3]f32{ 1, 1, 1 };
        if (inten > 1e-6) {
            color = .{ cd.r / inten, cd.g / inten, cd.b / inten };
        } else {
            inten = 1;
        }

        try out.append(ctx.arena, .{
            .name = name,
            .kind = kind,
            .color = color,
            .intensity = std.math.clamp(inten, 0.0, 20.0),
            .range = 0,
            .position = .{ position.x, position.y, position.z },
            .rotation = .{ rotation.x, rotation.y, rotation.z, rotation.w },
        });
    }
    return out.toOwnedSlice(ctx.arena);
}

// ── Animations ──────────────────────────────────────────────────────────────

fn vecChannel(ctx: *Ctx, node: []const u8, path: []const u8, keys: [*c]c.aiVectorKey, count: c_uint, tps: f64) !JChannel {
    const n: usize = @intCast(count);
    const times = try ctx.arena.alloc(f32, n);
    const values = try ctx.arena.alloc(f32, n * 3);
    for (0..n) |k| {
        times[k] = @floatCast(keys[k].mTime / tps);
        const v = keys[k].mValue;
        values[k * 3 + 0] = v.x;
        values[k * 3 + 1] = v.y;
        values[k * 3 + 2] = v.z;
    }
    return .{ .node = node, .path = path, .interpolation = "LINEAR", .times = times, .values = values };
}

fn quatChannel(ctx: *Ctx, node: []const u8, keys: [*c]c.aiQuatKey, count: c_uint, tps: f64) !JChannel {
    const n: usize = @intCast(count);
    const times = try ctx.arena.alloc(f32, n);
    const values = try ctx.arena.alloc(f32, n * 4);
    for (0..n) |k| {
        times[k] = @floatCast(keys[k].mTime / tps);
        const q = keys[k].mValue; // aiQuaternion: w, x, y, z
        values[k * 4 + 0] = q.x;
        values[k * 4 + 1] = q.y;
        values[k * 4 + 2] = q.z;
        values[k * 4 + 3] = q.w;
    }
    return .{ .node = node, .path = "rotation", .interpolation = "LINEAR", .times = times, .values = values };
}

fn buildAnimations(ctx: *Ctx) ![]const JAnimation {
    var out: std.ArrayListUnmanaged(JAnimation) = .{ .items = &.{}, .capacity = 0 };
    for (0..ctx.scene.mNumAnimations) |ai| {
        const a: *const c.aiAnimation = @ptrCast(ctx.scene.mAnimations[ai]);
        const tps: f64 = if (a.mTicksPerSecond != 0) a.mTicksPerSecond else 1.0;

        var channels: std.ArrayListUnmanaged(JChannel) = .{ .items = &.{}, .capacity = 0 };
        for (0..a.mNumChannels) |ci| {
            const ch: *const c.aiNodeAnim = @ptrCast(a.mChannels[ci]);
            const nname = ctx.arena.dupe(u8, aiStr(&ch.mNodeName)) catch continue;
            if (ch.mNumPositionKeys > 0)
                try channels.append(ctx.arena, try vecChannel(ctx, nname, "translation", ch.mPositionKeys, ch.mNumPositionKeys, tps));
            if (ch.mNumScalingKeys > 0)
                try channels.append(ctx.arena, try vecChannel(ctx, nname, "scale", ch.mScalingKeys, ch.mNumScalingKeys, tps));
            if (ch.mNumRotationKeys > 0)
                try channels.append(ctx.arena, try quatChannel(ctx, nname, ch.mRotationKeys, ch.mNumRotationKeys, tps));
        }
        if (channels.items.len == 0) continue;

        const nm = if (aiStr(&a.mName).len > 0)
            (ctx.arena.dupe(u8, aiStr(&a.mName)) catch "Animation")
        else
            (std.fmt.allocPrint(ctx.arena, "Animation {d}", .{ai}) catch "Animation");
        try out.append(ctx.arena, .{ .name = nm, .channels = try channels.toOwnedSlice(ctx.arena) });
    }
    return out.toOwnedSlice(ctx.arena);
}

// ── Cache file helpers ──────────────────────────────────────────────────────

fn matCachePath(ctx: *Ctx, idx: usize) ![]const u8 {
    const fname = try std.fmt.allocPrint(ctx.arena, "{s}_mat{d}.zmesh", .{ ctx.base, idx });
    return std.fs.path.join(ctx.arena, &.{ ctx.cache_dir, fname });
}

fn primCachePath(ctx: *Ctx, seq: usize) ![]const u8 {
    const fname = try std.fmt.allocPrint(ctx.arena, "{s}_p{d}.zmesh", .{ ctx.base, seq });
    return std.fs.path.join(ctx.arena, &.{ ctx.cache_dir, fname });
}

fn writeFileBytes(io: std.Io, path: []const u8, bytes: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

// ── FFI entry point ─────────────────────────────────────────────────────────

/// Duplicate a byte slice into a libc-`malloc`'d, NUL-terminated buffer so the C# side can
/// own it across the FFI boundary and release it with `zigote_model_free` (libc `free`).
fn dupeC(s: []const u8) ?[*:0]u8 {
    const buf: [*c]u8 = @ptrCast(std.c.malloc(s.len + 1) orelse return null);
    @memcpy(buf[0..s.len], s);
    buf[s.len] = 0;
    return @ptrCast(buf);
}

/// Import `path_z`, writing caches under `cache_dir_z`, and return a JSON manifest as a
/// malloc'd C string (caller frees via `zigote_model_free`). Returns null on failure.
pub fn importModelJson(path_z: [*:0]const u8, cache_dir_z: [*:0]const u8) ?[*:0]u8 {
    // aiProcess_FlipUVs: Assimp normalises every format to a bottom-left UV origin (V-up). Our
    // renderer — like glTF and the procedural primitives — uses a top-left origin (V-down), so flip
    // V back on import. Without this, imported textures come in vertically mirrored.
    const flags: c_uint = @intCast(c.aiProcess_Triangulate | c.aiProcess_GenSmoothNormals |
        c.aiProcess_CalcTangentSpace | c.aiProcess_JoinIdenticalVertices |
        c.aiProcess_SortByPType | c.aiProcess_GenUVCoords | c.aiProcess_RemoveRedundantMaterials |
        c.aiProcess_FlipUVs);

    const scene_opt = c.aiImportFile(path_z, flags);
    if (scene_opt == null) {
        std.log.warn("assimp import failed: {s}", .{c.aiGetErrorString()});
        return null;
    }
    const scene: *const c.aiScene = @ptrCast(scene_opt);
    defer c.aiReleaseImport(scene_opt);

    var arena_state = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var io_state = std.Io.Threaded.init(std.heap.c_allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();

    const path = std.mem.span(path_z);
    const cache_dir = std.mem.span(cache_dir_z); // caller (C#) guarantees this directory exists

    var warnings: std.ArrayListUnmanaged([]const u8) = .{ .items = &.{}, .capacity = 0 };
    var ctx = Ctx{
        .arena = arena,
        .io = io,
        .scene = scene,
        .cache_dir = cache_dir,
        .model_dir = std.fs.path.dirname(path) orelse ".",
        .base = std.fs.path.stem(path),
        .warnings = &warnings,
    };

    const materials = buildMaterials(&ctx) catch return null;
    const animated = scene.mNumAnimations > 0;

    var nodes: []const JNode = &.{};
    var mesh_nodes: []const JMeshNode = &.{};
    if (animated) {
        var roots: std.ArrayListUnmanaged(JNode) = .{ .items = &.{}, .capacity = 0 };
        const root = buildNode(&ctx, @as(*const c.aiNode, @ptrCast(scene.mRootNode))) catch return null;
        roots.append(arena, root) catch return null;
        nodes = roots.toOwnedSlice(arena) catch return null;
    } else {
        const ngroups: usize = if (scene.mNumMaterials == 0) 1 else scene.mNumMaterials;
        const groups = arena.alloc(Accum, ngroups) catch return null;
        for (groups) |*g| g.* = .{};
        const root_node: *const c.aiNode = @ptrCast(scene.mRootNode);
        // Pre-size each material group to its vertex/index total so walkStatic's append loops never
        // grow-and-copy (geometric doubling stranded ~2x the final size in the import-lifetime arena).
        {
            const vcount = arena.alloc(usize, ngroups) catch return null;
            const icount = arena.alloc(usize, ngroups) catch return null;
            @memset(vcount, 0);
            @memset(icount, 0);
            countStatic(&ctx, root_node, vcount, icount);
            for (groups, vcount, icount) |*g, v, ix| {
                g.verts.ensureTotalCapacity(arena, v) catch {};
                g.idx.ensureTotalCapacity(arena, ix) catch {};
            }
        }
        walkStatic(&ctx, root_node, Mat4.identity, groups);
        mesh_nodes = emitMeshNodes(&ctx, groups, materials) catch return null;
    }

    const lights = buildLights(&ctx) catch &[_]JLight{};
    const animations = buildAnimations(&ctx) catch &[_]JAnimation{};

    const manifest = JManifest{
        .animated = animated,
        .counts = .{
            .meshes = @intCast(scene.mNumMeshes),
            .materials = @intCast(scene.mNumMaterials),
            .textures = @intCast(scene.mNumTextures),
            .lights = lights.len,
            .animations = @intCast(scene.mNumAnimations),
            .nodes = ctx.node_count,
            .primitives = ctx.prim_count,
        },
        .warnings = warnings.toOwnedSlice(arena) catch &[_][]const u8{},
        .nodes = nodes,
        .meshNodes = mesh_nodes,
        .materials = materials,
        .lights = lights,
        .animations = animations,
    };

    const json = std.json.Stringify.valueAlloc(arena, manifest, .{}) catch return null;
    return dupeC(json);
}
