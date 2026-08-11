// macOS status-bar item (NSStatusItem) — the tray icon.
//
// Plain C entry points (exported from libzigote.dylib) that C# drives via P/Invoke, same
// shape as macos_menu.m: item clicks come back through one callback carrying the item's
// integer tag, and the C# side maps that tag to an Action. Main thread only. The status
// item and its menu are intentionally leaked for the app's lifetime (no ARC), which is also
// what keeps the menu alive while it is open.

#import <Cocoa/Cocoa.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ZEXPORT __attribute__((visibility("default")))

typedef void (*ZigTrayCallback)(int32_t tag);
static ZigTrayCallback g_handler = NULL;

@interface ZigTrayTarget : NSObject
- (void)zigTrayAction:(id)sender;
@end

@implementation ZigTrayTarget
- (void)zigTrayAction:(id)sender {
    if (g_handler != NULL) {
        g_handler((int32_t)[(NSMenuItem *)sender tag]);
    }
}
@end

static ZigTrayTarget *g_target = nil;
static NSStatusItem *g_item = nil;

ZEXPORT void zigote_mactray_set_handler(ZigTrayCallback cb) {
    g_handler = cb;
}

// Create the status item, if it is not already there. The image is the app's own icon,
// scaled to the menu-bar height — an app that ships an icon never has to hand one over.
ZEXPORT void zigote_mactray_show(const char *tooltipC) {
    if (g_item == nil) {
        g_target = [[ZigTrayTarget alloc] init];
        g_item = [[NSStatusBar systemStatusBar]
            statusItemWithLength:NSSquareStatusItemLength];

        NSImage *icon = [NSApp applicationIconImage];
        if (icon != nil) {
            NSImage *scaled = [icon copy];
            [scaled setSize:NSMakeSize(18, 18)];
            g_item.button.image = scaled;
        } else {
            g_item.button.title = @"♪";
        }
    }

    if (tooltipC != NULL) {
        g_item.button.toolTip = [NSString stringWithUTF8String:tooltipC];
    }
}

ZEXPORT void zigote_mactray_set_tooltip(const char *tooltipC) {
    if (g_item != nil && tooltipC != NULL) {
        g_item.button.toolTip = [NSString stringWithUTF8String:tooltipC];
    }
}

// Replace the whole menu. One line per item, "tag\tlabel\tenabled"; an empty line is a
// separator. Wholesale rather than incremental because the labels that change (Play/Pause)
// change together with the enabled states, so there is never a partial update to make.
ZEXPORT void zigote_mactray_set_menu(const char *specC) {
    if (g_item == nil) {
        return;
    }

    NSMenu *menu = [[NSMenu alloc] init];
    [menu setAutoenablesItems:NO];

    if (specC != NULL) {
        char *spec = strdup(specC);
        char *linePos = spec;
        char *line;
        while ((line = strsep(&linePos, "\n")) != NULL) {
            if (line[0] == '\0') {
                [menu addItem:[NSMenuItem separatorItem]];
                continue;
            }

            char *fieldPos = line;
            const char *tagField = strsep(&fieldPos, "\t");
            const char *labelField = strsep(&fieldPos, "\t");
            const char *enabledField = strsep(&fieldPos, "\t");
            if (labelField == NULL) {
                continue;
            }

            NSMenuItem *item = [[NSMenuItem alloc]
                initWithTitle:[NSString stringWithUTF8String:labelField]
                       action:@selector(zigTrayAction:)
                keyEquivalent:@""];
            item.target = g_target;
            item.tag = (NSInteger)atoi(tagField);
            item.enabled = enabledField == NULL || atoi(enabledField) != 0;
            [menu addItem:item];
        }
        free(spec);
    }

    g_item.menu = menu;
}

ZEXPORT void zigote_mactray_hide(void) {
    if (g_item != nil) {
        [[NSStatusBar systemStatusBar] removeStatusItem:g_item];
        g_item = nil;
    }
}
