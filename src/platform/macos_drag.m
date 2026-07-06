// macOS native drag-OUT (app → OS).
//
// SDL3 exposes no portable "begin a drag as a source" API, so dragging content out of the window
// (e.g. a file onto Finder, or text into another app) is done here with Cocoa's NSDraggingSession.
//
// This is best-effort and macOS-only. It works by grabbing [NSApp currentEvent] — which, right after
// SDL has pumped and dispatched the OS mouse-drag event, is that NSEvent — and starting a dragging
// session from the key window's content view. If no left-drag event is current (i.e. C# called this
// outside a live pointer drag) it no-ops and returns 0, so the caller can fall back gracefully.
//
// Plain C entry point exported from libzigote.dylib; C# drives it via P/Invoke (NativeEngine, guarded
// by OperatingSystem.IsMacOS). Built with -fno-objc-arc like macos_menu.m; the single source object is
// intentionally leaked for the app's lifetime.

#import <Cocoa/Cocoa.h>
#include <stdint.h>

#define ZEXPORT __attribute__((visibility("default")))

@interface ZigDragSource : NSObject <NSDraggingSource>
@end

@implementation ZigDragSource
- (NSDragOperation)draggingSession:(NSDraggingSession *)session
    sourceOperationMaskForDraggingContext:(NSDraggingContext)context {
    return NSDragOperationCopy | NSDragOperationGeneric;
}
@end

static ZigDragSource *g_dragSource = nil;

// Begin a system drag carrying UTF-8 `text` (plain-text item) and/or `filesNL` (newline-separated
// absolute paths, each dragged as a file URL). Either may be NULL/empty. Returns 1 if a dragging
// session started, 0 otherwise (no live mouse-drag event, no key window, or nothing to drag).
ZEXPORT int32_t zigote_macdrag_begin(const char *text, const char *filesNL) {
    NSEvent *ev = [NSApp currentEvent];
    if (ev == nil) return 0;
    NSEventType t = [ev type];
    if (t != NSEventTypeLeftMouseDragged && t != NSEventTypeLeftMouseDown) return 0;

    NSWindow *win = [ev window];
    if (win == nil) win = [NSApp keyWindow];
    if (win == nil) return 0;
    NSView *view = [win contentView];
    if (view == nil) return 0;

    NSMutableArray<NSDraggingItem *> *items = [NSMutableArray array];

    if (filesNL != NULL && filesNL[0] != '\0') {
        NSString *joined = [NSString stringWithUTF8String:filesNL];
        for (NSString *p in [joined componentsSeparatedByString:@"\n"]) {
            if ([p length] == 0) continue;
            NSURL *url = [NSURL fileURLWithPath:p];
            NSDraggingItem *it = [[NSDraggingItem alloc] initWithPasteboardWriter:url];
            [items addObject:it];
        }
    }
    if (text != NULL && text[0] != '\0') {
        NSString *s = [NSString stringWithUTF8String:text];
        NSDraggingItem *it = [[NSDraggingItem alloc] initWithPasteboardWriter:s];
        [items addObject:it];
    }
    if ([items count] == 0) return 0;

    // Place each item's drag image near the cursor so the session has a non-degenerate frame.
    NSPoint loc = [ev locationInWindow];
    NSRect frame = NSMakeRect(loc.x - 16.0, loc.y - 16.0, 32.0, 32.0);
    for (NSDraggingItem *it in items) {
        [it setDraggingFrame:frame contents:nil];
    }

    if (g_dragSource == nil) g_dragSource = [[ZigDragSource alloc] init];

    @try {
        [view beginDraggingSessionWithItems:items event:ev source:g_dragSource];
    } @catch (NSException *e) {
        return 0;
    }
    return 1;
}
