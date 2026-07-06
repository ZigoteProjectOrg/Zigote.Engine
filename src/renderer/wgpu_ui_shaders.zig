//! WGSL shader sources for the 2D UI renderer (wgpu.zig).
//! Extracted verbatim so the renderer module stays focused on pipeline wiring.

pub const shape_shader_source = @embedFile("shaders/shape_shader_source.wgsl");

// Liquid glass shader: physically-based refraction + rim lighting.
// Samples a backdrop texture (captured before glass pass) at refracted UV offsets
// to create the real "looking through glass" effect. Ported from the
// reference implementation (whynotmake.it / Tim Lehmann, 2025).
pub const liquid_glass_shader_source = @embedFile("shaders/liquid_glass_shader_source.wgsl");

pub const image_shader_source = @embedFile("shaders/image_shader_source.wgsl");
