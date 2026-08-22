// Minimal UI module — only the types still needed by src/renderer/ and src/ffi/.
// The widget framework, reconciler, and all layout code have been ported to C#
// (ZigoteCS/Zigote.UI) and are no longer compiled in Zig.

// Re-exported so `zigote_ui` consumers need not also depend on `zigote_core`.
pub const geometry = @import("zigote_core").geometry;
pub const text = @import("text.zig");
pub const render = struct {
    pub const paint = @import("render/paint.zig");
};

test {
    _ = geometry;
    _ = text;
    _ = render.paint;
}
