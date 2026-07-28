// macOS native file/folder dialogs (NSOpenPanel/NSSavePanel).
//
// The full-featured macOS backend behind the zigote_file_dialog_* FFI (src/ffi/dialogs.zig
// dispatches here instead of SDL's dialog subsystem on macOS). What this adds over SDL:
// a visible title (panels ignore `title`; the `message` line is what users see), save-panel
// file-name prefill, accept-button ("prompt") labels, hidden-files / create-directories
// switches, the folder-pick "Choose" convention, and a native Format popup on save panels
// when there is more than one filter — the pieces that make the dialogs feel first-party.
//
// Threading: begin is main-thread only (dialogs.zig guarantees it); completion handlers run on
// the main thread via the normal run loop, which SDL's event pump drives. One dialog at a time
// (enforced by dialogs.zig). Compiled without ARC like the other platform files; the panel is
// retained for the duration of its session and released in the completion handler.

#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <stdint.h>

#define ZEXPORT __attribute__((visibility("default")))

// Must match the flag bits in src/ffi/dialogs.zig / Zigote.Core FileDialog.
enum {
    ZIG_DLG_MANY = 1,
    ZIG_DLG_SHOW_HIDDEN = 2,
    ZIG_DLG_NO_CREATE_DIRS = 4,
};

// outcome: 0 = selected (pathsNL = newline-joined absolute paths), 1 = cancelled, 2 = error.
typedef void (*ZigDlgDone)(const char *pathsNL, int32_t outcome);

static ZigDlgDone g_done = NULL;
static NSSavePanel *g_panel = nil; // retained while a session is up

/// Apply one filter's extension list to the panel. An empty list means "all files".
static void ApplyTypes(NSSavePanel *panel, NSArray<NSString *> *exts) {
    if (exts.count == 0) {
        if (@available(macOS 11.0, *)) {
            panel.allowedContentTypes = @[]; // empty = any type
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            panel.allowedFileTypes = nil;
#pragma clang diagnostic pop
        }
        panel.allowsOtherFileTypes = YES;
        return;
    }
    if (@available(macOS 11.0, *)) {
        NSMutableArray<UTType *> *types = [NSMutableArray array];
        for (NSString *ext in exts) {
            UTType *t = [UTType typeWithFilenameExtension:ext];
            if (t != nil) [types addObject:t];
        }
        panel.allowedContentTypes = types;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        panel.allowedFileTypes = exts;
#pragma clang diagnostic pop
    }
    panel.allowsOtherFileTypes = NO;
}

// Target for the save panel's Format popup: switches the allowed types to the chosen filter,
// which also makes the panel rewrite the name field's extension — the native format-picker UX.
@interface ZigDlgFormatTarget : NSObject
@property(assign) NSSavePanel *panel; // the panel retains the accessory chain; assign avoids a cycle
@property(retain) NSArray<NSArray<NSString *> *> *extLists;
- (void)formatChanged:(NSPopUpButton *)sender;
@end

@implementation ZigDlgFormatTarget
- (void)formatChanged:(NSPopUpButton *)sender {
    NSInteger idx = sender.indexOfSelectedItem;
    if (self.panel == nil || idx < 0 || (NSUInteger)idx >= self.extLists.count) return;
    ApplyTypes(self.panel, self.extLists[idx]);
}
@end

static ZigDlgFormatTarget *g_formatTarget = nil;

/// Parse the FFI filter spec — newline-separated "Name|ext1;ext2" ("*" pattern = all files) —
/// into parallel name / extension-list arrays (an empty list encodes "all files").
static void ParseFilters(const char *filtersC, NSMutableArray<NSString *> **outNames,
                         NSMutableArray<NSArray<NSString *> *> **outExts) {
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    NSMutableArray<NSArray<NSString *> *> *exts = [NSMutableArray array];
    if (filtersC != NULL && filtersC[0] != '\0') {
        NSString *spec = [NSString stringWithUTF8String:filtersC];
        for (NSString *line in [spec componentsSeparatedByString:@"\n"]) {
            NSRange bar = [line rangeOfString:@"|"];
            if (line.length == 0 || bar.location == NSNotFound) continue;
            NSString *name = [line substringToIndex:bar.location];
            NSString *pattern = [line substringFromIndex:bar.location + 1];
            if ([pattern isEqualToString:@"*"]) {
                [names addObject:(name.length > 0 ? name : @"All Files")];
                [exts addObject:@[]];
                continue;
            }
            NSMutableArray<NSString *> *list = [NSMutableArray array];
            for (NSString *ext in [pattern componentsSeparatedByString:@";"]) {
                if (ext.length > 0) [list addObject:ext];
            }
            if (list.count == 0) continue;
            [names addObject:name];
            [exts addObject:list];
        }
    }
    *outNames = names;
    *outExts = exts;
}

/// "Format:" label + popup accessory for save panels with more than one filter.
static NSView *MakeFormatAccessory(NSSavePanel *panel, NSArray<NSString *> *names,
                                   NSArray<NSArray<NSString *> *> *extLists) {
    if (g_formatTarget == nil) g_formatTarget = [[ZigDlgFormatTarget alloc] init];
    g_formatTarget.panel = panel;
    g_formatTarget.extLists = extLists;

    NSTextField *label = [NSTextField labelWithString:@"Format:"];
    [label sizeToFit];
    NSPopUpButton *popup = [[[NSPopUpButton alloc] initWithFrame:NSMakeRect(0, 0, 220, 25)
                                                       pullsDown:NO] autorelease];
    [popup addItemsWithTitles:names];
    popup.target = g_formatTarget;
    popup.action = @selector(formatChanged:);

    const CGFloat pad = 8.0;
    CGFloat labelW = NSWidth(label.frame);
    NSView *container =
        [[[NSView alloc] initWithFrame:NSMakeRect(0, 0, labelW + pad + 220 + 40, 34)] autorelease];
    label.frame = NSMakeRect(20, 9, labelW, NSHeight(label.frame));
    popup.frame = NSMakeRect(20 + labelW + pad, 4, 220, 25);
    [container addSubview:label];
    [container addSubview:popup];
    return container;
}

// Move a file/folder to the user's Trash (NSFileManager — undoable via Finder, unlike unlink).
// Returns 1 on success. Backs the cross-platform zigote_file_trash export.
ZEXPORT int32_t zigote_mac_trash_item(const char *pathC) {
    if (pathC == NULL || pathC[0] == '\0') return 0;
    NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:pathC]];
    NSError *error = nil;
    BOOL ok = [[NSFileManager defaultManager] trashItemAtURL:url
                                           resultingItemURL:nil
                                                      error:&error];
    return ok ? 1 : 0;
}

// kind: 0 = open file, 1 = pick folder, 2 = save file. Strings are optional (NULL/empty);
// they are copied into Cocoa objects before returning. nswindowPtr: NSWindow* to sheet onto
// (NULL = key window, standalone panel when there is none). Returns 0 when the panel is up.
ZEXPORT int32_t zigote_macdlg_begin(int32_t kind, const char *titleC, const char *dirC,
                                    const char *nameC, const char *filtersC, const char *acceptC,
                                    uint32_t flags, void *nswindowPtr, ZigDlgDone done) {
    if (NSApp == nil) [NSApplication sharedApplication];
    g_done = done;

    BOOL isSave = (kind == 2);
    NSSavePanel *panel;
    if (isSave) {
        panel = [NSSavePanel savePanel];
    } else {
        NSOpenPanel *open = [NSOpenPanel openPanel];
        open.canChooseFiles = (kind == 0);
        open.canChooseDirectories = (kind == 1);
        open.allowsMultipleSelection = (flags & ZIG_DLG_MANY) != 0;
        open.resolvesAliases = YES;
        panel = open;
    }

    // Modern panels drop `title`; `message` is the line users actually see. Set both — title
    // still captions a standalone (non-sheet) panel window.
    if (titleC != NULL && titleC[0] != '\0') {
        NSString *title = [NSString stringWithUTF8String:titleC];
        panel.title = title;
        panel.message = title;
    }
    if (acceptC != NULL && acceptC[0] != '\0') {
        panel.prompt = [NSString stringWithUTF8String:acceptC];
    } else if (kind == 1) {
        panel.prompt = @"Choose"; // the folder-pick convention ("Open" reads wrong for folders)
    }

    if (dirC != NULL && dirC[0] != '\0') {
        NSString *dir = [NSString stringWithUTF8String:dirC];
        BOOL isDir = NO;
        if ([[NSFileManager defaultManager] fileExistsAtPath:dir isDirectory:&isDir] && isDir) {
            panel.directoryURL = [NSURL fileURLWithPath:dir isDirectory:YES];
        }
    }
    if (isSave && nameC != NULL && nameC[0] != '\0') {
        panel.nameFieldStringValue = [NSString stringWithUTF8String:nameC];
    }

    panel.showsHiddenFiles = (flags & ZIG_DLG_SHOW_HIDDEN) != 0;
    if (isSave || kind == 1) {
        // Save panels and folder picks offer New Folder unless the caller opted out; open-file
        // panels keep the native default (no folder creation).
        panel.canCreateDirectories = (flags & ZIG_DLG_NO_CREATE_DIRS) == 0;
    }

    if (kind != 1) {
        NSMutableArray<NSString *> *names;
        NSMutableArray<NSArray<NSString *> *> *extLists;
        ParseFilters(filtersC, &names, &extLists);
        if (extLists.count > 0) {
            if (isSave) {
                // Native save UX: default to the first format, switchable via the Format popup.
                ApplyTypes(panel, extLists[0]);
                if (extLists.count > 1) {
                    panel.accessoryView = MakeFormatAccessory(panel, names, extLists);
                }
            } else {
                // Open panels filter on the union of every format (no popup — macOS convention).
                NSMutableArray<NSString *> *all = [NSMutableArray array];
                BOOL allowAny = NO;
                for (NSArray<NSString *> *list in extLists) {
                    if (list.count == 0) allowAny = YES;
                    [all addObjectsFromArray:list];
                }
                ApplyTypes(panel, allowAny ? @[] : all);
            }
            panel.canSelectHiddenExtension = YES;
            panel.extensionHidden = NO; // a developer tool — show real file names
        }
    }

    [g_panel release];
    g_panel = [panel retain];

    void (^handler)(NSModalResponse) = ^(NSModalResponse result) {
        if (result == NSModalResponseOK) {
            NSMutableString *joined = [NSMutableString string];
            if (isSave) {
                [joined appendString:(panel.URL.path ?: @"")];
            } else {
                BOOL first = YES;
                for (NSURL *url in ((NSOpenPanel *)panel).URLs) {
                    if (!first) [joined appendString:@"\n"];
                    [joined appendString:url.path];
                    first = NO;
                }
            }
            if (g_done != NULL) g_done([joined UTF8String], joined.length > 0 ? 0 : 1);
        } else {
            if (g_done != NULL) g_done(NULL, 1);
        }
        if (g_formatTarget != nil) g_formatTarget.panel = nil;
        [g_panel release];
        g_panel = nil;
    };

    NSWindow *parent = (NSWindow *)nswindowPtr;
    if (parent == nil) parent = [NSApp keyWindow];
    if (parent != nil) {
        [panel beginSheetModalForWindow:parent completionHandler:handler];
    } else {
        [panel beginWithCompletionHandler:handler];
    }
    return 0;
}
