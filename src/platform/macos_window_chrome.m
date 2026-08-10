// macOS window chrome — the two in-app titlebar looks.
//
// Unified: extends the content view under the titlebar (NSWindowStyleMaskFullSizeContentView)
// with a transparent titlebar and hidden title text, keeping the NATIVE close/minimize/zoom
// traffic lights floating over the app's own titlebar strip — the standard modern-macOS look.
//
// CSD: a borderless window, where the app draws its own traffic lights. Its rounded corners are
// masked in CoreAnimation rather than clipped by the renderer, so they come out antialiased, cast
// the system shadow along the curve, and — the reason it must be done here — do not depend on the
// GPU surface having negotiated a premultiplied-alpha composite mode, which Metal does not offer.
//
// Both are driven from src/ffi/chrome.zig; the app-side titlebar provides the drag region (SDL
// hit-test) and, under unified chrome, the top-left inset for the native lights.

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

// Round (or unround, with enabled = 0) a borderless CSD window's corners. Masking the content
// view's layer clips every sublayer under it, the renderer's CAMetalLayer included, so the cut is
// made after the frame is composited and needs nothing from the swapchain's alpha mode. The
// window must be non-opaque for the cut-away corners to show the desktop; SDL already made it so
// when it granted SDL_WINDOW_TRANSPARENT, and this asserts it for the case where it did not.
ZEXPORT void zigote_macwin_set_csd(void *nswindowPtr, int enabled, float radius) {
    NSWindow *window = (NSWindow *)nswindowPtr;
    if (window == nil) return;
    NSView *content = window.contentView;
    if (content == nil) return;
    content.wantsLayer = YES;
    CALayer *layer = content.layer;
    if (layer == nil) return;

    if (enabled) {
        layer.cornerRadius = radius;
        // The continuous ("squircle") curve every other macOS window corner uses — a plain
        // circular arc next to real windows is the tell that the corner was drawn by hand.
        if (@available(macOS 10.15, *)) layer.cornerCurve = kCACornerCurveContinuous;
        layer.masksToBounds = YES;
        window.backgroundColor = NSColor.clearColor;
        window.opaque = NO;
        window.hasShadow = YES;
    } else {
        layer.cornerRadius = 0.0;
        layer.masksToBounds = NO;
    }
}
