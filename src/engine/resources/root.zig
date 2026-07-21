pub const mesh = @import("mesh.zig");
pub const material = @import("material.zig");
pub const zmesh_format = @import("zmesh_format.zig");

pub const Mesh = mesh.Mesh;
pub const Primitive = mesh.Primitive;
pub const Vertex = mesh.Vertex;
pub const GpuVertex = mesh.GpuVertex;
pub const Material = material.Material;
pub const RenderEffect = material.RenderEffect;
pub const AlphaMode = material.AlphaMode;

pub const writeZmesh = zmesh_format.writeZmesh;
pub const parseZmesh = zmesh_format.parseZmesh;

test {
    _ = zmesh_format;
}
