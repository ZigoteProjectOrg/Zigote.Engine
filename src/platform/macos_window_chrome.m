// macOS unified-titlebar window chrome.
//
// Extends the content view under the titlebar (NSWindowStyleMaskFullSizeContentView) with a
// transparent titlebar and hidden title text, keeping the NATIVE close/minimize/zoom traffic
// lights floating over the app's own titlebar strip — the standard modern-macOS "unified"
// look. Driven from src/ffi/chrome.zig; the app-side titlebar widget reserves the top-left
// inset for the lights and provides the drag region (SDL hit-test).

#import <Cocoa/Cocoa.h>
#include <stdint.h>

#define ZEXPORT __attribute__((visibility("default")))

// Diagnostic readback: 1 = unified applied, 0 = not applied, -1 = no NSWindow.
ZEXPORT int32_t zigote_macwin_get_unified(void *nswindowPtr) {
    NSWindow *window = (NSWindow *)nswindowPtr;
    if (window == nil) return -1;
    BOOL fullSize = (window.styleMask & NSWindowStyleMaskFullSizeContentView) != 0;
    return (fullSize && window.titlebarAppearsTransparent) ? 1 : 0;
}

ZEXPORT void zigote_macwin_set_unified(void *nswindowPtr, int enabled) {
    NSWindow *window = (NSWindow *)nswindowPtr;
    if (window == nil) return;
    if (enabled) {
        window.styleMask |= NSWindowStyleMaskFullSizeContentView;
        window.titlebarAppearsTransparent = YES;
        window.titleVisibility = NSWindowTitleHidden;
    } else {
        window.styleMask &= ~NSWindowStyleMaskFullSizeContentView;
        window.titlebarAppearsTransparent = NO;
        window.titleVisibility = NSWindowTitleVisible;
    }
}
