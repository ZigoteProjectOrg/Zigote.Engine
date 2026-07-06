// macOS native menu bar (NSMenu).
//
// Plain C entry points (exported from libzigote.dylib) that C# drives via P/Invoke
// to build the application's global menu bar. Menu-item clicks are delivered back to
// C# through a single callback that carries the item's integer tag; the C# side maps
// that tag to an Action. Built lazily/occasionally, so menus are intentionally leaked
// for the app's lifetime (no ARC).

#import <Cocoa/Cocoa.h>
#include <stdint.h>

#define ZEXPORT __attribute__((visibility("default")))

typedef void (*ZigMenuCallback)(int32_t tag);
static ZigMenuCallback g_handler = NULL;

// Reserved tag for the app menu's About item (C# item tags start at 1). Must match
// MacMenu.AboutTag on the C# side.
#define ZIG_MENU_ABOUT_TAG (-1)

@interface ZigMenuTarget : NSObject
- (void)zigMenuAction:(id)sender;
- (void)zigAboutAction:(id)sender;
@end

@implementation ZigMenuTarget
- (void)zigMenuAction:(id)sender {
    if (g_handler != NULL) {
        g_handler((int32_t)[(NSMenuItem *)sender tag]);
    }
}

// About is routed through the C# handler (reserved tag) so apps can show their own about
// screen; without a handler it falls back to the standard Cocoa panel.
- (void)zigAboutAction:(id)sender {
    if (g_handler != NULL) {
        g_handler((int32_t)ZIG_MENU_ABOUT_TAG);
    } else {
        [NSApp orderFrontStandardAboutPanel:sender];
    }
}
@end

static ZigMenuTarget *g_target = nil;
static NSMenu *g_mainMenu = nil;

static NSString *MakeString(const char *c) {
    return c ? [NSString stringWithUTF8String:c] : @"";
}

ZEXPORT void zigote_macmenu_set_handler(ZigMenuCallback cb) {
    g_handler = cb;
}

// Start a fresh main menu with the standard application menu (item 0): About, Hide, Quit.
ZEXPORT void zigote_macmenu_reset(const char *appNameC) {
    if (g_target == nil) {
        g_target = [[ZigMenuTarget alloc] init];
    }

    NSString *appName = (appNameC && appNameC[0]) ? MakeString(appNameC) : @"App";
    g_mainMenu = [[NSMenu alloc] init];

    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    [g_mainMenu addItem:appItem];

    NSMenu *appMenu = [[NSMenu alloc] init];
    [appItem setSubmenu:appMenu];
    NSMenuItem *aboutItem =
        [appMenu addItemWithTitle:[@"About " stringByAppendingString:appName]
                           action:@selector(zigAboutAction:)
                    keyEquivalent:@""];
    [aboutItem setTarget:g_target];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[@"Hide " stringByAppendingString:appName]
                       action:@selector(hide:)
                keyEquivalent:@"h"];
    NSMenuItem *hideOthers = [appMenu addItemWithTitle:@"Hide Others"
                                                action:@selector(hideOtherApplications:)
                                         keyEquivalent:@"h"];
    [hideOthers setKeyEquivalentModifierMask:(NSEventModifierFlagCommand |
                                              NSEventModifierFlagOption)];
    [appMenu addItemWithTitle:@"Show All"
                       action:@selector(unhideAllApplications:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[@"Quit " stringByAppendingString:appName]
                       action:@selector(terminate:)
                keyEquivalent:@"q"];
}

// Append a top-level menu (e.g. File) and return its NSMenu* as an opaque handle.
ZEXPORT void *zigote_macmenu_add_menu(const char *titleC) {
    NSString *title = MakeString(titleC);
    NSMenuItem *item = [[NSMenuItem alloc] init];
    [item setTitle:title];
    [g_mainMenu addItem:item];

    NSMenu *menu = [[NSMenu alloc] initWithTitle:title];
    [menu setAutoenablesItems:NO];
    [item setSubmenu:menu];
    return (void *)menu;
}

// Append a submenu item to parent and return the child NSMenu* handle.
ZEXPORT void *zigote_macmenu_add_submenu(void *parentMenu, const char *titleC) {
    NSMenu *parent = (NSMenu *)parentMenu;
    NSString *title = MakeString(titleC);

    NSMenuItem *item = [[NSMenuItem alloc] init];
    [item setTitle:title];
    [parent addItem:item];

    NSMenu *sub = [[NSMenu alloc] initWithTitle:title];
    [sub setAutoenablesItems:NO];
    [item setSubmenu:sub];
    return (void *)sub;
}

// sfSymbolC: optional SF Symbol name rendered as the item image (macOS 11+; unknown names are
// silently ignored). checkedState: 0 = off, 1 = on (the leading checkmark).
ZEXPORT void zigote_macmenu_add_item(void *parentMenu, const char *titleC, int32_t tag,
                                     const char *keyC, uint32_t modMask, int enabled,
                                     const char *sfSymbolC, int checkedState) {
    NSMenu *parent = (NSMenu *)parentMenu;
    NSString *title = MakeString(titleC);
    NSString *key = MakeString(keyC);

    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title
                                                  action:@selector(zigMenuAction:)
                                           keyEquivalent:key];
    [item setTarget:g_target];
    [item setTag:(NSInteger)tag];
    [item setEnabled:(enabled != 0)];
    if ([key length] > 0) {
        [item setKeyEquivalentModifierMask:(NSEventModifierFlags)modMask];
    }
    if (sfSymbolC && sfSymbolC[0]) {
        if (@available(macOS 11.0, *)) {
            NSImage *img = [NSImage imageWithSystemSymbolName:MakeString(sfSymbolC)
                                     accessibilityDescription:nil];
            if (img != nil) {
                [item setImage:img];
            }
        }
    }
    [item setState:(checkedState != 0 ? NSControlStateValueOn : NSControlStateValueOff)];
    [parent addItem:item];
}

ZEXPORT void zigote_macmenu_add_separator(void *parentMenu) {
    NSMenu *parent = (NSMenu *)parentMenu;
    [parent addItem:[NSMenuItem separatorItem]];
}

// Install the assembled menu as the application's main menu.
ZEXPORT void zigote_macmenu_commit(void) {
    if (NSApp == nil) {
        [NSApplication sharedApplication];
    }
    [NSApp setMainMenu:g_mainMenu];
}

// C#-side fallback for the About item when no app-specific about screen is registered.
ZEXPORT void zigote_macmenu_show_standard_about(void) {
    [NSApp orderFrontStandardAboutPanel:nil];
}

// Mark a menu as one of the standard AppKit roles so the OS decorates it:
// role 1 = the windows menu (AppKit appends Minimize/Zoom + the live window list),
// role 2 = the help menu (AppKit adds the search field).
ZEXPORT void zigote_macmenu_set_menu_role(void *menuPtr, int32_t role) {
    NSMenu *menu = (NSMenu *)menuPtr;
    if (NSApp == nil) {
        [NSApplication sharedApplication];
    }
    if (role == 1) {
        [NSApp setWindowsMenu:menu];
    } else if (role == 2) {
        [NSApp setHelpMenu:menu];
    }
}
