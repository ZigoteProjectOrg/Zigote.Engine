# Native file & folder dialogs — design

Status: **implemented** (desktop). This document is the design record for the cross-platform native
file/folder picker: the platform survey behind the decisions, the FFI surface, the C# API, and the
extension path for mobile.

## Problem

The editor's "Open Project…", "Save Scene As…" and "Load HDRI" flows used either a custom in-window
file browser (`Zigote.UI.Material` `FilePickerDialog`) or a bare text field for typing a path.
Neither matches user expectations on any desktop OS: no OS-side favorites/recents, no iCloud/network
volumes, no OS-level permission integration (macOS sandbox bookmarks, Linux portals), no
overwrite-confirmation on save, and no folder picking at all.

Goal: one **universal, async, engine-level API** for open-file / open-files / pick-folder /
save-file that presents the *native* dialog of the host OS, is the default in the editor, degrades
gracefully where a native dialog cannot be shown, and can be extended to mobile without reshaping
the API.

## What "native best practice" means per platform

| Platform | Best-practice dialog | Presentation | Async model |
|----------|---------------------|--------------|-------------|
| macOS | `NSOpenPanel` / `NSSavePanel` | Document-modal **sheet** on the parent window (falls back to app-modal panel) | Completion handler on the main run loop |
| Windows | COM `IFileOpenDialog` / `IFileSaveDialog` (Vista+; `FOS_PICKFOLDERS` for folders) | Owner-window modal | Blocking `Show()` — must run off the render thread |
| Linux | `xdg-desktop-portal` `FileChooser` over D-Bus (renders the *desktop's* dialog — GTK on GNOME, KDE's on Plasma; also the only sandbox-correct path under Flatpak/Snap), with a `zenity` subprocess as fallback | Portal-owned window, parented via window export | Inherently async (D-Bus response signal) |
| iOS (future) | `UIDocumentPickerViewController` | Sheet | Delegate callback |
| Android (future) | Storage Access Framework (`ACTION_OPEN_DOCUMENT`) | Activity | Activity result; returns `content://` URIs, not paths |

Two structural facts fall out of this survey and shape the whole design:

1. **The API must be asynchronous.** Linux portals, macOS sheets, and both mobile pickers are
   callback/async by nature; only Windows is blocking, and there the blocking call must be moved off
   the render thread anyway. A synchronous API would have to fake it everywhere.
2. **Results are an opaque list of location strings.** Desktop returns absolute paths; Android
   returns URIs. Returning strings (not a richer file abstraction) keeps the API mobile-extensible —
   a future `StorageFile` layer can wrap the strings without breaking desktop callers.

## Decision: custom NSOpenPanel/NSSavePanel on macOS, SDL3's dialog subsystem elsewhere

The engine already embeds SDL3 for windowing/input, and SDL3 ships a dialog subsystem
(`SDL_ShowOpenFileDialog` / `SDL_ShowSaveFileDialog` / `SDL_ShowOpenFolderDialog` /
`SDL_ShowFileDialogWithProperties`) whose per-platform implementations follow the best-practice
column above: `NSOpenPanel`/`NSSavePanel` on macOS, `IFileDialog` COM on its own thread on
Windows, and `xdg-desktop-portal` with `zenity` fallback on Linux. The symbols are already
compiled into `libzigote`, so it is the zero-dependency baseline backend.

SDL is a lowest-common-denominator layer, though, and on macOS (the primary dev platform) that
showed: panels had **no visible title** (modern macOS ignores `title`; the visible line is
`message`, which SDL never sets), **no save-name prefill** (SDL folds directory + name into one
"location" and doesn't split it back), no accept-button labels, no hidden-files or New Folder
control, and no format picker. So the design's escape hatch was exercised for macOS:
[`src/platform/macos_file_dialog.m`](../src/platform/macos_file_dialog.m) implements the panels
directly (same pipeline as `macos_menu.m`), and `ffi/dialogs.zig` dispatches to it on macOS while
keeping SDL for Windows/Linux. What the custom backend adds:

- `message` + `title` set from the request title — the dialog finally *says* what it's for.
- Save panels: `nameFieldStringValue` prefill, and with >1 filter a native **Format popup**
  (accessory view) that swaps the enforced `UTType`/extension — the TextEdit-style save UX.
- Accept-button (`prompt`) labels; folder picks default to the "Choose" convention.
- `showsHiddenFiles` and `canCreateDirectories` (New Folder) switches; extensions kept visible
  (`extensionHidden = NO` — it's a developer tool).
- Sheet parenting via the real `NSWindow` (resolved from the SDL window id), falling back to the
  key window.

Remaining SDL-path (Windows/Linux) limitations, accepted:

- Title/accept-label/multi-select map through SDL properties; the hidden-files and
  create-directories flags do not (no SDL equivalent) and are ignored there.
- Filters are extension-based only (no MIME/UTType); the "selected filter index" is not surfaced.
- One dialog at a time is the model (the C# layer serializes requests; OS dialogs are modal anyway).
- Windows could get the same treatment later (an `IFileDialog` `.c` backend behind the same
  exports) if its SDL dialog proves limiting.

## Architecture

```
Editor / app code                          (await FileDialog.OpenFileAsync(...))
        │
Zigote.Core  FileDialog (public, Task-based, request queue, pumped per frame)
        │            P/Invoke (bindings generated from src/ffi/*.zig)
        ▼
Zigote.Engine  src/ffi/dialogs.zig   zigote_file_dialog_* exports
        │            one outstanding native request + mutex-guarded result slot
        ▼
SDL3 dialog subsystem
   macOS: NSOpenPanel/NSSavePanel sheet   Windows: IFileDialog (own thread)   Linux: portal → zenity
```

### Completion is **poll-based**, not callback-based

SDL invokes the dialog callback *on an arbitrary thread* (main thread on macOS, the dialog's worker
thread on Windows). Instead of marshaling a managed callback across threads, the Zig layer stores
the outcome in a mutex-guarded slot and C# polls it once per frame from the UI thread. This choice
is what makes the design uniform:

- `App.Frame` already runs at least every 16 ms even when fully idle (`WaitEvents` has a 16 ms
  timeout), so polling adds no wake-up machinery and at most one frame of latency.
- Task completion always happens on the UI thread → `await` continuations run inline on the UI
  thread, so editor code can touch widgets after `await` with no dispatcher.
- No `[UnmanagedCallersOnly]` reverse P/Invoke from foreign threads, no managed/native lifetime
  races. The native side is a dumb slot; all sequencing lives in one place (the C# pump).

### FFI surface (`src/ffi/dialogs.zig`)

Five exports, auto-bound by `ZigoteBindingGenerator` like every other `src/ffi` file. Additive only
— **no ABI version bump** (the version guards struct layouts and event codes; no structs or events
change here).

| Export | Signature | Notes |
|--------|-----------|-------|
| `zigote_file_dialog_supported` | `() bool` | Compile-time: true on macOS/Windows/Linux builds. Linux runtime absence (no portal, no zenity) surfaces as a completion error instead — it cannot be known up front. |
| `zigote_file_dialog_begin` | `(kind u32, title, directory, file_name, filters, accept_label [*c]const u8, flags u32, parent_window_id u32) bool` | Starts the native dialog; false if unsupported, a request is already outstanding, or the backend rejects it. `file_name` prefills the save name; `accept_label` renames the OK button; flags: 1 = multi-select, 2 = show hidden, 4 = no New Folder. Main thread only. |
| `zigote_file_dialog_status` | `() i32` | 0 idle · 1 pending · 2 done-selected · 3 done-cancelled · 4 done-error |
| `zigote_file_dialog_result` | `() [*c]const u8` | Newline-joined UTF-8 locations; non-null only in state 2. Valid until `consume`/next `begin`. |
| `zigote_file_dialog_consume` | `() void` | Frees the result + request storage, returns to idle. |

Encoding rules:

- `kind`: 0 = open file, 1 = pick folder, 2 = save file.
- `title`/`directory`/`file_name`/`accept_label`: optional NUL-terminated UTF-8 (null → platform
  default). The SDL path re-joins directory + file_name into its single "location"; the macOS
  backend uses them separately (which is what makes the save-name prefill work).
- `filters`: newline-separated `Name|pattern` entries, pattern in SDL form (`ext1;ext2` or `*`),
  e.g. `"Zigote Project|zigoteproj\nAll Files|*"`. Null → no filtering. Ignored for folders.
- `parent_window_id`: SDL window id (`ZgEvent.window_id` domain); 0 → unparented. Parenting is what
  turns the macOS panel into a sheet and anchors the Windows dialog.
- Result paths are joined with `\n` (the same convention as `zigote_macdrag_begin`'s file list;
  `\n` cannot appear in a path on any supported OS).
- Request storage (title/location/filter strings, the SDL properties group) stays alive until
  `consume` — SDL requires the filter array to outlive the callback.
- Allocation uses `std.heap.c_allocator` (thread-safe; the completion callback may run off-thread).

### C# API (`Zigote.Core.Engine.FileDialog`)

```csharp
public readonly struct FileDialogFilter(string name, params string[] extensions);

public static class FileDialog
{
    public static bool Enabled { get; set; }         // app-level opt-out (preference, default on)
    public static bool PlatformSupported { get; }    // engine up + platform has a backend
    public static bool IsSupported { get; }          // Enabled && PlatformSupported — what call sites gate on
    public static uint DefaultParentWindow { get; set; }  // App sets this to the main window id

    public static Func<uint>? ParentWindowProvider { get; set; }  // App: "the focused window"
    public static Func<FileDialogRequest, Task<string[]>>? ManagedBackend { get; set; } // in-app impl
    public static bool CanShowDialogs { get; }       // native OR managed exists — gates Browse buttons

    public static Task<string?>  OpenFileAsync(string? title = null, string? startDirectory = null,
                                               FileDialogFilter[]? filters = null, uint parentWindow = 0,
                                               string? acceptLabel = null, bool showHidden = false);
    public static Task<string[]> OpenFilesAsync(...);           // empty array = cancelled
    public static Task<string?>  PickFolderAsync(string? title = null, string? startDirectory = null,
                                               uint parentWindow = 0, string? acceptLabel = null,
                                               bool showHidden = false, bool canCreateDirectories = true);
    public static Task<string?>  SaveFileAsync(string? title = null, string? startDirectory = null,
                                               string? suggestedName = null,
                                               FileDialogFilter[]? filters = null, uint parentWindow = 0,
                                               string? acceptLabel = null, bool canCreateDirectories = true);
}
```

Semantics:

- **UI-thread only** (both the `*Async` calls and the pump). Completions run inline on the UI
  thread, so `await` + widget mutation is safe with no dispatcher.
- `null` (or empty array) result = user cancelled — never an exception. A platform failure (no
  portal/zenity on Linux, SDL error) faults the task with `FileDialogException`; callers that have
  an in-app fallback catch it and degrade.
- Requests are **queued and serialized** — the native layer holds one outstanding dialog; a second
  request simply waits for the first (OS dialogs are modal; concurrent dialogs are a UX bug, not a
  feature).
- `parentWindow` 0 resolves through `ParentWindowProvider` — `Zigote.UI.Host.App` sets it to "the
  focused OS window", so every dialog sheets onto the window the user actually triggered it from
  (main or a secondary like Settings) with no per-call-site code — then falls back to
  `DefaultParentWindow` (the main window).
- `App.Frame` calls the internal `FileDialog.Pump()` right after the native event poll, in both the
  with-root and root-less paths, so dialogs also resolve for apps that only have secondary windows
  and for `GameApp` (which drives `App.Frame`).

Why `Zigote.Core` and not `Zigote.UI`: the API has no widget dependencies, and hosts other than the
widget `App` (players, tools) need it too. `Zigote.UI` only contributes the per-frame pump call and
the default parent window.

### Editor policy — which picker where

Native OS dialogs are for **OS-level flows** (paths outside the project, or writes):

- **Open Project…** → native open-file, filter `.zigoteproj` (welcome screen + File menu).
- **New Project** → dialog keeps the name field; a **Browse…** button picks the parent directory
  natively.
- **Save Scene As…** → native save-file, filter `.scene`, prefilled with the current name; results
  under the project root are stored project-relative.
- **Load HDRI / Environment** (viewport settings) → native open-file with image filters.

Native is the default but user-overridable: **Settings → Developer → "Native file dialogs"**
(persisted as `EditorConfig.NativeFileDialogs`, applied to `FileDialog.Enabled` at boot and live on
toggle). The row is only shown when `FileDialog.PlatformSupported` is true.

## The in-app cross-platform browser (`FileBrowserDialog`)

`Zigote.UI.Material` ships a full widget-based file dialog and registers it as
`FileDialog.ManagedBackend` via a module initializer — any app referencing Material gets it with
zero wiring (the DevTools auto-install philosophy). Routing is automatic and callers never branch:
native when available/enabled, the browser otherwise, **including when a native dialog fails at
show time** (e.g. a Linux desktop with no portal and no zenity). Only when neither exists does a
call fault with `FileDialogException`.

**Presentation: a separate OS window.** `ShowAsync` opens the browser as its own secondary window
(`App.CreateWindow`), centered over the invoking window — movable, resizable, and it never covers
the content being picked for, matching how OS dialogs behave. The OS titlebar carries the dialog
title (the in-content title row only appears in the fallback presentation). Cancel paths: Cancel
button, Esc (via the new root-`IDismissableOverlay` hook in `App.HandleEscape` — a window whose
root is dismissable closes on Esc), or the titlebar ✕ (`CloseRequested` → cancelled task). The
window is non-modal (like portal dialogs); when a secondary window cannot be created the browser
falls back to the previous in-window modal overlay automatically.

**Window chrome (OS-integrated titlebars) — app-wide.** Every app window (main window included,
plus Settings, torn-out panels and dialog windows) follows the host desktop's look via
`App.ApplyWindowChrome` + `WindowChrome.Resolve()` (`Zigote.UI.Host`): macOS gets a **unified
titlebar** — the content view extends under a transparent titlebar and the *native*
close/minimize/zoom traffic lights float over the app-drawn strip
(`src/platform/macos_window_chrome.m`); GNOME-family desktops get **Adwaita-style client-side
decorations** — a borderless window with in-app minimize/maximize/close circle buttons
(`WindowTitleBar` widget, `Zigote.UI`); Windows and KDE keep system decorations.

Presentation per style: **MacUnified hides the titlebar entirely** — no in-app strip either; the
content extends to the top of the window and the app's own top row (the editor toolbar, the
browser's navigation bar) IS the titlebar band, leading with `App.TitleBarLeftInset` (78px) for
the native traffic lights (`App.TitleBarTopInset` for windows with no toolbar, e.g. Settings).
**AdwaitaCsd composes a headerbar** (`WindowChromeHost` wraps the root — the Root setter
re-wraps transparently on every assignment) because something must carry the CSD buttons.
`ApplyWindowChrome` cascades to open secondary windows and new windows inherit at
`CreateWindow`; the CSD close button routes through `App.RequestClose()` so it behaves exactly
like the OS ✕ (CloseRequested → destroy).

**Dragging is decided by the app, per pointer position**: the SDL hit-test calls back into the
managed drag arbiter (`zigote_window_chrome_set_hit_provider` → `App.DragHitTest`), which walks
the widget under the point — interactive widgets (focusable, or overriding pointer virtuals;
reflection cached per type) stay clickable, everything else in the titlebar band drags the
window. Static drag rects remain as the no-provider fallback, and borderless windows get
synthesized 8px resize edges. macOS drops the unified styleMask bit on fullscreen/zoom
round-trips — `zigote_window_chrome_sync`, called from resize events, re-asserts it (no-op when
intact or fullscreen). The editor applies the mode at boot and exposes it under **Settings →
Developer → "Window chrome"** (`EditorConfig.WindowChromeMode`: auto/system/mac/adwaita, applied
live to all open windows) so any look can be forced on any OS for testing — `mac` is refused
off-macOS at the native layer and degrades to System per window.

Feature set (deliberately matched to what OS dialogs offer): places sidebar (pinned location,
Home/Desktop/Documents/Downloads, per-OS volumes — `FileBrowserPlaces`), back/forward/up history,
clickable breadcrumbs (collapsed in the middle when deep), sortable Name/Size/Modified columns
(folders always group first), search-as-you-type, a filter dropdown (open dialogs get an implicit
"All Files" entry; save dialogs enforce their format list and swap the name's extension on filter
change), hidden-files toggle, New Folder, Cmd/Ctrl- and Shift-multi-select, a save mode with name
prefill + overwrite confirmation, and full keyboard control (arrows/Home/End/PageUp/PageDown,
Enter activates, Backspace goes up, Space toggles, type-ahead jumps by prefix, Esc cancels). The
row list is clip-virtualized (the TreeView direct-paint pattern), so huge directories stay cheap.
Names sort naturally (`file2` before `file10`). Beyond parity: a **right-click context menu**
(Open, Rename…, Move to Trash — recoverable via `FileOperations.MoveToTrash`: NSFileManager /
shell recycle / XDG Trash per OS — Copy Path, Reveal in Finder/Explorer, New Folder) and a
**preview pane** for a single selected image, decoded through the engine's own loader
(`FileBrowserPreview`) — so it previews `.hdr`/`.tga` engine content OS dialogs can't, with
dimensions/size/date metadata.
The view pipeline lives in the widget-free `FileBrowserModel` (filters → sort → visible +
selection + history), unit-tested in `FileBrowserModelTests`.

Two structural roles beyond fallback:

1. **Asset-reference pickers** (Inspector mesh/texture fields) use it through the
   `FilePickerDialog.Show` shim with `LockRoot`: navigation is clamped to the project subtree, the
   places sidebar is hidden, and results come back project-relative — an OS dialog can't express
   any of that. Same policy as other engines: in-app pickers for assets, OS dialogs for
   import/export/open/save.
2. **Embeddable**: `FileBrowserDialog` is a plain widget (`Options` + `Result`), so it can also be
   hosted outside a modal (a future asset-browser panel).

## Mobile extension path (design headroom, not implemented)

- `kind` + async + string-list results map 1:1 onto `UIDocumentPickerViewController` and Android
  SAF. SDL3 already routes Android dialogs to SAF (results are `content://` URIs — one more reason
  results are opaque strings).
- `FileDialogFilter` extensions would translate to UTTypes/MIME types in a mobile backend; the
  public shape doesn't change.
- Save-file has no SAF equivalent (`ACTION_CREATE_DOCUMENT` differs semantically); a mobile backend
  may legitimately fault save requests with `FileDialogException`, which callers already handle.

## Testing

- Filter-spec building and result-splitting are pure and unit-tested (`Zigote.Tests`
  `FileDialogTests`); the Zig-side spec parser has `zig build test` coverage.
- The dialog itself is OS-owned UI — not automatable headlessly, and irrelevant to golden images.
  Manual smoke per platform: open/cancel each of the four editor flows, multi-select, save-overwrite
  prompt, sheet parenting on macOS.
- Failure injection: the C# queue treats `begin() == false` and status 4 as faults; both paths are
  unit-testable without a native dialog.

## Rejected alternatives

- **Hand-rolled per-OS backends first** (NSOpenPanel `.m`, IFileDialog COM `.c`, portal D-Bus):
  strictly more code and three new platform surfaces to maintain, for features v1 doesn't need. The
  FFI is shaped so any single platform can move to this later without touching callers.
- **Callback-based completion into C#**: crosses threads on Windows/Linux, needs re-dispatch to the
  UI thread anyway, and adds reverse-P/Invoke lifetime hazards. Polling costs one branch per frame.
- **Event-queue delivery (`EVT_*`)**: `ZgEvent` is a fixed 44-byte struct — variable-length path
  lists don't fit, and adding a side-channel would bump the ABI for no gain over the result slot.
- **In-app picker as the default**: rejected by requirement — it can't see OS favorites/volumes and
  will never match platform conventions; it stays as fallback + asset-scoped picker.
