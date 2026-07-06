// Public aggregate module for the C# FFI bridge (src/ffi/root.zig).
// The Zig widget framework has been ported to C# (ZigoteCS/Zigote.UI).
// This module now exports only what the FFI layer and 2D/3D renderers need.

pub const core = @import("zigote_core");

const ui_mod = @import("zigote_ui");
pub const geometry   = ui_mod.geometry;
pub const text_style = ui_mod.text;
pub const paint      = ui_mod.render.paint;

pub const Color      = geometry.Color;
pub const Rect       = geometry.Rect;
pub const Size       = geometry.Size;
pub const Constraints = geometry.Constraints;
pub const EdgeInsets = geometry.EdgeInsets;
pub const PaintList  = paint.PaintList;
pub const FontAsset  = text_style.FontAsset;
pub const TextStyle  = text_style.TextStyle;

const engine_mod = @import("zigote_engine");
pub const math3d    = engine_mod.math;
pub const scene     = engine_mod.scene;
pub const resources = engine_mod.resources;
pub const Vec2      = math3d.Vec2;
pub const Vec3      = math3d.Vec3;
pub const Vec4      = math3d.Vec4;
pub const Mat4      = math3d.Mat4;
pub const Quat      = math3d.Quat;
pub const Ray       = math3d.Ray;
pub const Frustum   = math3d.Frustum;
pub const World     = scene.World;
pub const SceneNode = scene.SceneNode;
pub const Mesh      = resources.Mesh;
pub const Material  = resources.Material;

pub const renderer = @import("renderer/root.zig");
pub const render = @import("render/root.zig");

test {
    _ = core;
    _ = geometry;
    _ = text_style;
    _ = paint;
    _ = math3d;
    _ = scene;
    _ = resources;
    _ = renderer;
}
