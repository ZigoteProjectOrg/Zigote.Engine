# FFI / ABI reference

Zigote.Engine exposes a flat C ABI: **286 exported `zigote_*` functions** (the count `nm` reports
from the built library), all `callconv(.c)`, defined across `src/ffi/` — **204** in
[`ffi/root.zig`](../src/ffi/root.zig) (which also holds the audio and physics wrappers), **49** in
[`ffi/ecs.zig`](../src/ffi/ecs.zig), **11** in [`ffi/chrome.zig`](../src/ffi/chrome.zig), **10** in
`ffi/desktop_shims_stub.zig` (non-macOS only), **6** each in
[`ffi/dialogs.zig`](../src/ffi/dialogs.zig) and `ffi/channel.zig`.
The C# frontend consumes them via `[DllImport]` P/Invoke; the bindings are **generated** from these `export fn`s — don't hand-write
them, regenerate.

`ffi/root.zig` is the authoritative export list. This document is a map, not a replacement for it.

## ABI versioning — the safety contract

The C# side pins struct sizes and an ABI version against this backend. At startup it calls
`zigote_get_renderer_abi_info` and asserts the returned sizes equal its own compile-time `sizeof`
values — a mismatch fails loudly instead of corrupting memory.

**Current ABI version: `9`** (`.abi_version = 9` in `zigote_get_renderer_abi_info`,
[`ffi/root.zig`](../src/ffi/root.zig)).

**Bump the ABI version whenever you change field order/size or add fields** to `ZgPaintCommand`,
`ZgEvent`, or `ZgRenderSettings3D`. The C# counterpart is `RendererAbiInfo.ExpectedAbiVersion` +
`AbiLayoutTests`.

### `ZgAbiInfo` (20 bytes) — returned by `zigote_get_renderer_abi_info`

| Offset | Field | Type |
|-------:|-------|------|
| 0 | `abi_version` | u32 |
| 4 | `paint_command_size` | u32 |
| 8 | `event_size` | u32 |
| 12 | `handle_size` | u32 (opaque resource handle = usize) |
| 16 | `render_settings_3d_size` | u32 |

### `ZgRendererCaps` (12 bytes) — returned by `zigote_get_renderer_caps` (after init)

Runtime capabilities of the backend actually selected. Kept **separate** from `ZgAbiInfo` (whose size
is a fixed compile-time guard). Features gate on this, never on a backend-name check.

| Offset | Field | Type | Meaning |
|-------:|-------|------|---------|
| 0 | `active_backend` | u32 | `BackendId` in use (`auto` may fall back to wgpu) |
| 4 | `upscalers` | u32 | always 0 (wgpu exposes no vendor upscaler) |
| 8 | `raytracing` | u8 | hardware RT available |
| 9 | `raytracing_from_render` | u8 | RT usable from fragment shaders, not only compute |
| 10 | `pad` | [2]u8 | pad to 12 |

> `upscalers` / `raytracing` are **reserved contracts** — no backend reports them today.

## Wire structs

### `ZgPaintCommand` (112 bytes)

One flat paint command. Fields are ordered large→small (8-byte pointers first) so the struct packs
with a single 3-byte hole instead of ~11 padding bytes (120→112 B — meaningful across the
thousands of commands a frame streams). Every offset is pinned by a `comptime @offsetOf` assert here
and by `AbiLayoutTests` on the C# side.

| Offset | Field | Type | Notes |
|-------:|-------|------|-------|
| 0 | `kind` | u8 | `CMD_*` discriminant (0–17) |
| 1 | `font_style` | u8 | 0 = normal, 1 = italic |
| 2 | `font_weight` | u16 | 100–900 |
| 4 | `has_cache_key` | u8 | |
| 5 | `pad0` | [3]u8 | the single packing hole |
| 8 | `text_ptr` | ptr | text bytes; also the `ZgGlyphRunQuad` array for `CMD_GLYPH_RUN` |
| 16 | `pixels_ptr` | ptr | image pixels / font-family bytes / **polygon points** |
| 24 | `rect_x/y/w/h` | 4×f32 | |
| 40 | `color_r/g/b/a` | 4×f32 | |
| 56 | `radius` | f32 | aliased: image u0 / shader id (`@bitCast`) |
| 60 | `border_width` | f32 | aliased: image v0 |
| 64 | `baseline_x` | f32 | aliased: image u1 |
| 68 | `baseline_y` | f32 | aliased: image v1 |
| 72 | `font_size` | f32 | |
| 76 | `line_height` | f32 | |
| 80 | `letter_spacing` | f32 | |
| 84 | `word_spacing` | f32 | |
| 88 | `img_pixel_w` | u32 | |
| 92 | `img_pixel_h` | u32 | |
| 96 | `cache_key_lo` | u32 | |
| 100 | `cache_key_hi` | u32 | |
| 104 | `text_len` | u32 | |
| 108 | `pixels_len` | u32 | |

**`CMD_*` command kinds:** 0 `RECT` · 1 `BORDER` · 2 `TEXT` · 3 `IMAGE` · 4 `CLIP_START` ·
5 `CLIP_END` · 6 `PUSH_OPACITY` · 7 `POP_OPACITY` · 8 `SHADOW` · 9 `LIQUID_GLASS` ·
10 `SHADER_EFFECT` · 11 `TEXT_LAYOUT` · 12 `GLYPH_RUN` · 13 `RENDER_TEXTURE_BEGIN` ·
14 `RENDER_TEXTURE_END` · 15 `BLUR` · 16 `BEZIER` · 17 `POLYGON`.

`ZgGlyphRunQuad` (32 bytes, for `CMD_GLYPH_RUN`): `{ x, y, w, h, u0, v0, u1, v1 }` (8×f32).

See [rendering.md](rendering.md) for what each command draws.

### `ZgEvent` (44 bytes)

One flat input event. The text-input / IME / drop UTF-8 **payload lives out of band**: it is appended
to the engine's per-poll `poll_text` buffer and the event carries only `(text_off, text_len)` into it
— so the common flood of mouse/key events costs 44 B, not ~288 B, and IME pre-edit is never
truncated. C# reads the buffer via `zigote_poll_text_ptr` right after polling (valid until the next
poll; single-threaded drain-decode).

| Offset | Field | Type | Notes |
|-------:|-------|------|-------|
| 0 | `kind` | u8 | `EVT_*` |
| 1 | `button` | u8 | mouse button; for key events 1 = OS auto-repeat |
| 2 | `modifiers` | u8 | `MOD_*` bitset |
| 3 | `key_char` | u8 | ASCII, 0 if not printable |
| 4 | `key_scancode` | u32 | raw SDL scancode |
| 8 | `x`, `y` | 2×f32 | pointer position |
| 16 | `scroll_x`, `scroll_y` | 2×f32 | |
| 24 | `resize_w` | u32 | `TEXT_EDITING`: IME composition **start** |
| 28 | `resize_h` | u32 | `TEXT_EDITING`: IME composition **length** |
| 32 | `text_off` | u32 | byte offset into `poll_text` |
| 36 | `text_len` | u32 | byte length in `poll_text` |
| 40 | `window_id` | u32 | SDL window id; 0 = unknown → treated as main window |

**`EVT_*` event kinds:** 0 `MOUSE_MOVE` · 1 `MOUSE_DOWN` · 2 `MOUSE_UP` · 3 `SCROLL` · 4 `KEY_DOWN` ·
5 `KEY_UP` · 6 `QUIT` · 7 `RESIZE` · 8 `TEXT_INPUT` · 9 `TEXT_EDITING` (IME) ·
10 `WINDOW_FOCUS` (button: 1 gained / 0 lost) · 11 `WINDOW_CLOSE` · 12 `SYSTEM_THEME` (button: 0
unknown / 1 light / 2 dark) · 13 `DROP_BEGIN` · 14 `DROP_FILE` · 15 `DROP_TEXT` · 16 `DROP_POSITION` ·
17 `DROP_COMPLETE` · 18 `TOUCH_DOWN` · 19 `TOUCH_MOVE` · 20 `TOUCH_UP` · 21 `TOUCH_CANCEL` ·
22 `APP_BACKGROUND` · 23 `APP_FOREGROUND` · 24 `LOW_MEMORY` · 25 `SCREEN_KEYBOARD_SHOWN` · 26 `SCREEN_KEYBOARD_HIDDEN` (mobile on-screen keyboard; occlusion is handled natively — the backend pans the view against the SetTextInputArea rect).

A file/text drop arrives as `DROP_BEGIN` → N × `DROP_FILE`/`DROP_TEXT` → `DROP_COMPLETE`, with the
payload out-of-band in `poll_text` exactly like text input. **New drop event kinds reused existing
`ZgEvent` fields — no ABI bump** (still ABI 9).

Touchscreen fingers (direct touch devices only — trackpads stay wheel/cursor; SDL's touch↔mouse
synthesis is pinned off at init so nothing fires twice) arrive as `TOUCH_DOWN` → N × `TOUCH_MOVE` →
`TOUCH_UP`/`TOUCH_CANCEL`, reusing existing fields: `x`/`y` = window-local position (de-normalized to
the mouse coordinate space), `key_scancode` = compact finger slot (0–9, stable per contact),
`scroll_x` = pressure 0..1. `TOUCH_CANCEL` means the OS took the gesture — abandon, don't commit.
App lifecycle: `APP_BACKGROUND` (SDL `will_enter_background` — stop GPU work before the app
suspends), `APP_FOREGROUND` (`did_enter_foreground`), `LOW_MEMORY`; `terminating` maps to `QUIT`.
**All reuse existing `ZgEvent` fields — no ABI bump** (still ABI 9). The main-window safe-area
insets (notch/home indicator; zero on desktop) are queryable via `zigote_get_safe_area(handle,
insets*4f32)` as `[left, top, right, bottom]` in window coordinates.

**`MOD_*`:** 1 `SHIFT` · 2 `CTRL` · 4 `ALT` · 8 `GUI` (⌘ on macOS, Super/Win elsewhere).
**`BTN_*`:** 0 `LEFT` · 1 `RIGHT` · 2 `MIDDLE`.

## Export map

Grouped by subsystem. Anchor functions are named; the exhaustive list is the `export fn`s in the
source. `ffi/root.zig` for everything except ECS, which is in `ffi/ecs.zig`.

| Group | Anchors | Notes |
|-------|---------|-------|
| **Lifecycle / frame** | `zigote_init`, `zigote_shutdown`, `zigote_begin_frame`, `zigote_submit_paint_commands`, `zigote_submit_overlay_commands`, `zigote_render_frame_v2`, `zigote_end_frame` | The per-frame contract (see [architecture.md](architecture.md)). `zigote_render_3d` is the immediate-mode 3D entry the editor uses. |
| **Events / input** | `zigote_poll_events`, `zigote_poll_text_ptr`, `zigote_wait_events`, `zigote_input_gamepad_*`, `zigote_get_system_theme` | Events → `ZgEvent` buffer; text/IME/drops via the out-of-band `poll_text` buffer. |
| **Windowing / multi-window** | `zigote_window_create`, `zigote_window_render`, `zigote_window_submit_paint`, `zigote_main_window_*`, `zigote_get_size`, `zigote_get_scale` | Secondary OS windows are UI-only, each with its own `GpuUi` sharing the main device. |
| **Text** | `zigote_load_font`, `zigote_add_emoji_font`, `zigote_measure_text`, `zigote_text_layout_*`, `zigote_text_reset_caches` | `zigote_measure_text` is headless (no GPU). See [subsystems.md](subsystems.md#text). |
| **Clipboard** | `zigote_get_clipboard`, `zigote_set_clipboard` | Two-way; `get` re-queries with a heap buffer for text over 8 KB (never truncates). |
| **2D paint / capture** | `zigote_submit_frame_damage`, `zigote_upload_glyph_atlas`, `zigote_capture_ui_bmp`, `zigote_render_texture_*` | `zigote_capture_ui_bmp` is the 2D golden-image seam (counterpart of the 3D `ZIGOTE_SHOT`). |
| **Textures (2D)** | `zigote_load_texture`, `zigote_load_texture_from_memory[_scaled]`, `zigote_load_texture_mask` | |
| **3D scene resources** | `zigote_scene_clear`, `zigote_scene_add_child_node`, `zigote_scene_update_node`, `zigote_scene_set_mesh_blob`, `zigote_scene_set_mesh_instances`, `zigote_scene_set_mesh_*` (color/roughness/emissive/textures/volume/alpha/double-sided/occlusion), `zigote_scene_set_light_properties`, `zigote_set_environment_hdri`, `zigote_set_reflection_probe` | Uploading these lazily creates the 3D stack. |
| **3D render / settings** | `zigote_get_render_settings_3d`, `zigote_set_render_settings_3d`, `zigote_render_set_frustum_cull`, `zigote_set_vsync`, `zigote_debug_get_engine_stats`, `zigote_debug_gpu_allocated_bytes`, `zigote_get_renderer_abi_info`, `zigote_get_renderer_caps` | **Settings/stats/cull FFI must never create the 3D stack** — they read/write pending state. |
| **Sprites** | `zigote_sprites_begin`, `zigote_sprites_draw`, `zigote_sprites_texture_create[_file]`, `zigote_sprites_shader_create`, `zigote_sprites_texture_destroy` | Native batched 2D pass. |
| **Particles** | `zigote_particles_upload`, `zigote_particles_clear[_all]`, `zigote_particles_compute_*` | Native billboard pass. |
| **Model import** | `zigote_model_import`, `zigote_model_free` | Assimp → `.zmesh` + JSON manifest. Compiled to a stub when `-Denable3d=false`. |
| **Audio** (24) | `zigote_audio_beep[_3d]`, `zigote_audio_sound_create_file`, `zigote_audio_sound_create_tone`, `zigote_audio_sound_{play,stop,set_position,set_attenuation,…}`, `zigote_audio_group_*`, `zigote_audio_set_listener`, `zigote_audio_set_master_volume`, `zigote_audio_update` | miniaudio `ma_engine`. Wrappers in `root.zig` over `ffi/audio.zig`. |
| **Physics** (23) | `zigote_physics_init`, `zigote_physics_step`, `zigote_physics_create_body`/`add_body`, `zigote_physics_add_force[_at_point]`, `zigote_physics_add_torque`, `zigote_physics_{get,set}_*velocity`, `zigote_physics_raycast_closest`, `zigote_physics_set_gravity` | Jolt. Wrappers in `root.zig` over `ffi/physics.zig`; a no-op `physics_stub.zig` when `-Dphysics3d=false`. |
| **ECS** (55, in `ffi/ecs.zig`) | `zigote_ecs_world_create`, `zigote_ecs_entity_create[_named]`, `zigote_ecs_{add,remove,set,get,has}`, `zigote_ecs_query_create`/`iter`/`next`, `zigote_ecs_system_create`, `zigote_ecs_observer_create`, `zigote_ecs_set_parent`, `zigote_ecs_new_prefab`, `zigote_ecs_builtin_*` | flecs. Whole module compiled out when `-Decs=false`. |
| **File dialogs** (6, in `ffi/dialogs.zig`) | `zigote_file_dialog_begin`, `zigote_file_dialog_status`, `zigote_file_dialog_result`, `zigote_file_dialog_consume`, `zigote_file_dialog_supported`, `zigote_file_trash` | Native OS open/save/folder dialogs (custom NSOpenPanel/NSSavePanel backend on macOS, SDL3 dialogs elsewhere), poll-based (one request outstanding). `file_trash` = macOS NSFileManager trash (managed code covers Windows/Linux). Main-thread only; C# façade is `Zigote.Core.Engine.FileDialog`. Design: [file-dialogs.md](file-dialogs.md). |
| **Window chrome** (7, in `ffi/chrome.zig`) | `zigote_window_chrome_set`, `zigote_window_chrome_drag_rects`, `zigote_window_chrome_set_hit_provider`, `zigote_window_chrome_minimize`, `zigote_window_chrome_toggle_maximize`, `zigote_window_chrome_sync`, `zigote_window_chrome_probe` | In-app titlebars: macOS unified (full-size content + native traffic lights, `macos_window_chrome.m`) and borderless CSD (Adwaita-style buttons); drag decided by the app-side hit provider (static rects fallback) + synthesized resize edges via SDL hit-test; `sync` re-asserts unified chrome after fullscreen/zoom drops it. Takes SDL window ids. Main-thread only. |

## Adding a binding

1. Add `export fn zigote_x(...) callconv(.c) ...` in [`ffi/root.zig`](../src/ffi/root.zig) (or the
   subsystem's own `src/ffi/*.zig` — ECS in `ecs.zig`, dialogs in `dialogs.zig`; a new file must be
   force-referenced from a `comptime` block in `root.zig` or its exports won't be compiled).
2. Regenerate the C# P/Invoke in `Zigote.Core/Native/NativeEngine.cs` — don't hand-write it.
3. Wrap it publicly in `ZigoteEngine` (C#) — never expose `NativeEngine` members directly.
4. If it stores into a `Gpu3d` cache, call `ensure3d(...)` first. If it only reads/invalidates, it
   must **not** create the 3D stack.
5. If you touched a wire struct, bump `.abi_version` and update the C# `AbiLayoutTests`.

GPU UBO/param structs live in [`src/renderer/uniforms.zig`](../src/renderer/uniforms.zig) with comptime
`@offsetOf`/`@sizeOf` guards pinning the field offsets the shaders depend on — the same pattern as the
wire structs above.
