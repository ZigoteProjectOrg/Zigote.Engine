//! Object-level transparency ordering. Matrices and bounds use the same world-space contract
//! as the vertex shader; instance matrices already include the node transform.
const std = @import("std");
const math = @import("../math/root.zig");

pub fn depth(view: math.Mat4, model: math.Mat4, center: math.Vec3) f32 {
    const z = -view.mulPoint(model.mulPoint(center)).z;
    // Invalid transforms must not violate the sort comparator's ordering contract.
    return if (std.math.isFinite(z)) z else -std.math.inf(f32);
}

test "transparency orders by view depth rather than radial distance" {
    var lateral = math.Mat4.identity;
    lateral.cols[3] = .{ .x = 100, .y = 0, .z = -2, .w = 1 };
    var distant = math.Mat4.identity;
    distant.cols[3].z = -10;
    try std.testing.expect(depth(.identity, distant, .zero) > depth(.identity, lateral, .zero));
}

test "transparency depth includes mesh bounds and camera transform" {
    var view = math.Mat4.identity;
    view.cols[3].z = 3;
    var model = math.Mat4.identity;
    model.cols[2].z = 2;
    model.cols[3].z = -10;
    try std.testing.expectEqual(@as(f32, 11), depth(view, model, .{ .x = 0, .y = 0, .z = -2 }));
    model.cols[3].z = std.math.nan(f32);
    try std.testing.expectEqual(-std.math.inf(f32), depth(view, model, .zero));
}
