# Subsystems

Beyond rendering, Zigote.Engine bundles the OS-facing and native-compute subsystems the C# side needs.
Each is exposed through the `zigote_*` FFI ([ffi-reference.md](ffi-reference.md)) and, in the C#
solution, wrapped by a static provider (`Audio`, `Physics`, `Input`, …). This document covers what the
**native** side does; the C# API shape lives in the parent [CLAUDE.md](../../CLAUDE.md).

## Windowing & input (SDL3)

`zigote_init` opens the main SDL3 window + wgpu device + the UI renderer. Input is polled once per
frame by `zigote_poll_events` into a `ZgEvent` buffer; text / IME / drop payloads ride the out-of-band
`poll_text` buffer read via `zigote_poll_text_ptr`.

- **Multi-window** — `zigote_window_create` opens a real secondary OS window. Secondary windows are
  **UI-only**: each owns its SDL window + wgpu surface + its own `GpuUi` (per-target vertex buffers /
  glyph atlas), sharing the main device/queue. The 3D scene stays bound to the main window. Events
  carry `window_id`; the C# main loop polls once and routes by id.
- **IME** — `zigote_start_text_input` / `zigote_stop_text_input` + `zigote_set_text_input_area` (per
  window variants exist) drive SDL text input; composition arrives as `EVT_TEXT_EDITING` with the
  start/length packed into the event's `resize_w`/`resize_h`.
- **Gamepad** — `zigote_input_gamepad_*` (connected / button / axis) over SDL's gamepad subsystem,
  which is initialized **lazily and failure-isolated** (`ensureGamepad`) on first query — never in the
  boot init, so a controller SDL can't open disables input rather than killing window creation.
- **System theme** — `zigote_get_system_theme` + `EVT_SYSTEM_THEME` surface the OS light/dark
  appearance.
- **Clipboard** — `zigote_get_clipboard` / `zigote_set_clipboard` bridge SDL3's system clipboard;
  `get` re-queries with a heap buffer when text exceeds 8 KB, so it never truncates.
- **Drag & drop** — OS→app drops arrive as the `EVT_DROP_*` run (begin → files/text → complete);
  app→OS drag-out is macOS-only best-effort via `platform/macos_drag.m` (`NSDraggingSession`), since
  SDL3 has no portable drag-source API.
- **Native menu bar** — macOS maps the C# `AppMenu` to a real `NSMenu` via `platform/macos_menu.m`.

## Text (FreeType + HarfBuzz)

`renderer/freetype_text.zig` owns glyph rasterization and shaping. `zigote_load_font` registers a
family (broadcast to every window's atlas); `zigote_add_emoji_font` registers a colour-emoji face.

- **Measurement is headless** — `zigote_measure_text` shapes and measures without a GPU, so C# layout
  can run before any device exists.
- **Two caches** keep a per-frame-rendering host fast: a glyph coverage atlas (1024² R8, grows ×2 to
  4096² on overflow) and a **shaped-quad cache** keyed by `(text, family, size)` so repeated text
  skips `hb_shape`. A colour-emoji atlas materializes lazily on the first colour glyph.
- **Layout objects** — `zigote_text_layout_*` build a retained native layout with hit-testing and
  caret navigation (used by `TextField`/`CodeEditor`); these bake against the **main** window's atlas.
- `zigote_text_reset_caches` drops every native + managed text cache — required after a wholesale
  sizing change or a face swap.

See [rendering.md](rendering.md#text) for how text commands are drawn.

## Audio (miniaudio `ma_engine`)

`ffi/audio.zig` creates one miniaudio high-level `ma_engine` (vendored `zaudio`). The engine owns the
playback device, mixing graph, resource manager (WAV/FLAC/MP3/OGG decode), and — for "surround" — the
built-in `ma_spatializer`, which pans every 3D sound across the device's **native** channel map
(stereo / 5.1 / 7.1) with distance attenuation. It opens with `channels = 0` so the OS picks the
layout and miniaudio spatialises into it.

The high-level engine API is **main-thread-safe**, so there is **no Zig code on the audio thread** —
the old raw-device callback, its command ring, and the real-time-safety hazard are gone.

Two layers, exposed through `zigote_audio_*` (wrappers in `root.zig` over `ffi/audio.zig`):

- **fire-and-forget one-shots** — `zigote_audio_beep` (2D UI click) / `zigote_audio_beep_3d`
  (positioned ping). Pooled, round-robin, reaped by `zigote_audio_update(dt)`.
- **addressable sources** — a u32-handle table of `Sound`s (procedural `Waveform`/`Noise` or a
  decoded/streamed file) created / positioned / destroyed by the caller: `zigote_audio_sound_create_*`,
  `_play` / `_stop`, `_set_position` / `_velocity` / `_attenuation` / `_pitch` / `_looping` /
  `_spatial`. **Mixer groups** (`zigote_audio_group_*`) route buses; `zigote_audio_set_listener`
  follows the camera.

Init is **lazy + failure-isolated** (`ensureAudio`): the device opens on first use, and a machine with
no audio device runs silently rather than crashing. `zaudio.init` is process-global "init once", so it
is guarded and never deinit-ed — safe across engine re-creation (project switching / tests).

## Physics (Jolt, `zphysics`)

`ffi/physics.zig` wraps Jolt (via the `joltc` C API); the `zigote_physics_*` C exports are thin
wrappers in `root.zig` over it. Gated by `-Dphysics3d` — when off, `root.zig` imports
`physics_stub.zig` instead, which provides the same functions as no-ops so the exports still compile
and link without the JoltC static lib.

The FFI is a thin rigid-body API, driven from C# only in play mode:

- **World** — `zigote_physics_init` / `_shutdown` / `_step(dt)`, `_set_gravity`,
  `_optimize_broadphase` (call after static bodies are registered).
- **Bodies** — `zigote_physics_create_body` / `_add_body` / `_remove_body` / `_destroy_body`;
  get/set position + rotation (Euler and quaternion) + linear/angular velocity.
- **Forces** — `_add_force`, `_add_force_at_point`, `_add_impulse`, `_add_torque`. A force **activates a
  sleeping body** (a Jolt body that sleeps otherwise silently drops applied forces).
- **Queries** — `zigote_physics_raycast_closest` (ray normal is world-up until a body-lock variant
  lands).

## ECS (flecs, `zflecs`)

`ffi/ecs.zig` (55 exports) wraps flecs v4. Gated by `-Decs` — when off, the whole module is dropped
and the exports do not exist. It is a general-purpose ECS the C# `Zigote.ECS` package builds on; in
play mode flecs is the entity store that the `World` gameplay façade mirrors.

Surface: world lifecycle (`zigote_ecs_world_create` / `_destroy` / `_progress`), entities
(`_entity_create[_named]` / `_destroy` / `_instantiate`), components (`_component_register`,
`_add` / `_remove` / `_set` / `_get` / `_get_mut` / `_has` / `_modified`), relationships and hierarchy
(`_add_pair`, `_set_parent`, `_builtin_childof` / `_isa` / `_prefab` …), queries + systems + observers
(`_query_create` / `_iter` / `_next`, `_system_create`, `_observer_create`), and prefabs
(`_new_prefab`).

**Binding note:** C# keywords in Zig parameter names break the C# binding generator — e.g. `event`
must be renamed `evt`. Per-entity `get`/`set` FFI is ~64 ns a call; batch through the iterator field
arrays (`_iter_field` / `_iter_entities`) for hot loops rather than per-entity round-trips.

## Model import (Assimp)

`ffi/assimp_loader.zig` imports any Assimp-supported format (glTF/GLB, FBX, OBJ, DAE, PLY, STL, …).
Gated by `-Denable3d` — when off, `zigote_model_import` is a stub returning null (games ship pre-baked
`.zmesh` and never import at runtime).

`zigote_model_import(path, cacheDir)`:

1. parses the file with Assimp (`aiImportFile`: triangulate + gen normals/tangents +
   `aiProcess_FlipUVs` so V matches the renderer's top-left origin),
2. writes one **`.zmesh`** binary per mesh into a `.mesh_cache/` dir next to the source (engine vertex
   layout: pos/normal/uv/tangent) and extracts embedded textures,
3. returns a **JSON manifest** (node tree, materials — the full glTF PBR set — lights, animations).

The C# `GltfLoader` builds the scene tree from the manifest and applies material heuristics. `.zmesh`
loads back through `zigote_scene_set_mesh_blob`. The `.zmesh` parser itself
(`engine/resources/zmesh_format.zig`) is **ungated** — exported games with `-Denable3d=false` still
load baked meshes.