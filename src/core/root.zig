//! Engine-neutral primitives shared by the UI, renderer and FFI layers.
//!
//! Reaching one `Rect` used to cost six files: geometry.zig → core/math/root.zig →
//! core/root.zig → ui/geometry.zig → ui/root.zig → root.zig, four of which contained nothing
//! but the same six-line alias list. The definitions live in geometry.zig; this is the module
//! root the build wires as `zigote_core`. See docs/v2-design.md §2.4.

pub const geometry = @import("geometry.zig");
pub const handle = @import("handle.zig");
pub const sync = @import("sync.zig");

pub const Color = geometry.Color;
pub const Constraints = geometry.Constraints;
pub const EdgeInsets = geometry.EdgeInsets;
pub const Offset = geometry.Offset;
pub const Rect = geometry.Rect;
pub const Size = geometry.Size;
pub const SpinLock = sync.SpinLock;
pub const HandleTable = handle.HandleTable;
pub const Handle = handle.Handle;

test {
    _ = geometry;
    _ = handle;
    _ = sync;
}
