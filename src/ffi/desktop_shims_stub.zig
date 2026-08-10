//! No-op stands-in for the macOS-only Objective-C platform shims (native menu bar, OS drag-out),
//! compiled on every NON-macOS target so the FFI surface is identical everywhere.
//!
//! Why this exists rather than "the caller just doesn't call them": the C# bindings are generated
//! from the export list and are therefore platform-independent, and on iOS the engine is linked
//! STATICALLY into the app binary, where every declared P/Invoke must resolve at link time
//! (DllImport names `__Internal`). A missing symbol is a link error even if no code path reaches
//! it. The managed side already gates these behind OperatingSystem.IsMacOS(), so these bodies are
//! never entered — they exist to satisfy the linker.
//!
//! Behaviour matches "the platform has no such feature": menu operations do nothing, menu
//! creation yields a null handle, and the OS drag-out reports that it did not take the drag (0),
//! which is exactly what the C# side treats as "fall back to the in-app drag".

const std = @import("std");

export fn zigote_macmenu_set_handler(cb: ?*const fn (i32) callconv(.c) void) void {
    _ = cb;
}

export fn zigote_macmenu_reset(app_name: [*c]const u8) void {
    _ = app_name;
}

export fn zigote_macmenu_add_menu(title: [*c]const u8) ?*anyopaque {
    _ = title;
    return null;
}

export fn zigote_macmenu_add_submenu(parent_menu: ?*anyopaque, title: [*c]const u8) ?*anyopaque {
    _ = parent_menu;
    _ = title;
    return null;
}

export fn zigote_macmenu_add_item(
    parent_menu: ?*anyopaque,
    title: [*c]const u8,
    tag: i32,
    key: [*c]const u8,
    mod_mask: u32,
    enabled: c_int,
    sf_symbol: [*c]const u8,
    checked_state: c_int,
) void {
    _ = parent_menu;
    _ = title;
    _ = tag;
    _ = key;
    _ = mod_mask;
    _ = enabled;
    _ = sf_symbol;
    _ = checked_state;
}

export fn zigote_macmenu_add_separator(parent_menu: ?*anyopaque) void {
    _ = parent_menu;
}

export fn zigote_macmenu_commit() void {}

export fn zigote_macmenu_show_standard_about() void {}

export fn zigote_macmenu_set_menu_role(menu_ptr: ?*anyopaque, role: i32) void {
    _ = menu_ptr;
    _ = role;
}

/// 0 = the OS did not take the drag (the app should run its own in-app drag).
export fn zigote_macdrag_begin(text: [*c]const u8, files_nl: [*c]const u8) i32 {
    _ = text;
    _ = files_nl;
    return 0;
}
