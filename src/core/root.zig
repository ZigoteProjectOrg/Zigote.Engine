// Remaining after Zig UI cleanup: collections and async modules were deleted
// (superseded by C#). image was removed (used deleted async_mod).

pub const math = @import("math/root.zig");

pub const Color       = math.Color;
pub const Constraints = math.Constraints;
pub const EdgeInsets  = math.EdgeInsets;
pub const Offset      = math.Offset;
pub const Rect        = math.Rect;
pub const Size        = math.Size;

test {
    _ = math;
}
