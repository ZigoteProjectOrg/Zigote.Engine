# Architecture

Zigote.Engine is the **native half** of the Zigote engine. It is a Zig library that owns the GPU and
every OS-facing resource, and exposes a flat C ABI (`zigote_*` functions) that the C# frontend
(`Zigote.Core`) drives via `[DllImport]` P/Invoke, generated at compile time from the `export fn`s.

```
┌─────────────────────────────────────────────────────────────┐
│  C# solution (Zigote.Core, Zigote.UI, Zigote.Game, Editor…)  │  product logic
│    widgets · scenes · scripting · physics coordination       │
└───────────────────────────┬─────────────────────────────────┘
                            │  [DllImport] P/Invoke — the zigote_* C ABI
┌───────────────────────────▼─────────────────────────────────┐
│  Zigote.Engine (this repo, Zig)                              │  native backend
│    SDL3 window · wgpu renderer · FreeType/HarfBuzz text      │
│    Jolt physics · flecs ECS · Assimp import · miniaudio      │
└─────────────────────────────────────────────────────────────┘
```

## The split, and why

| Layer | Owns | Reason |
|-------|------|--------|
| **Zig (this repo)** | Windowing, GPU device + render passes, text shaping, physics, ECS, model import, audio | Deterministic performance, direct GPU access, no GC on the frame's hot path |
| **C#** | Widgets, layout, scenes, scripting, the editor, all gameplay | Ergonomic composition, hot reload, rich tooling |

**C# owns the frame loop.** Zig is a passive library — it never spins its own render thread. Each
frame the C# side polls events, builds a paint-command list, and calls into Zig to render. The one
exception is the audio device (miniaudio's own callback thread) and SDL's internal event queue.

The design of the whole solution lives in the parent repository's
[docs/architecture.md](../../docs/architecture.md) — this repo documents only the native backend.

## Source layout

```text
src/
  root.zig                 internal aggregate module (re-exports the sub-modules)

  core/                    engine-neutral primitives
    geometry.zig           Color, Rect, Size, Offset, Constraints, EdgeInsets
    sync.zig               SpinLock

  ui/                      headless UI support (no GPU)
    text.zig               text measurement + text styles
    geometry.zig           2D geometry helpers
    render/paint.zig       the renderer's internal command list (the wire format is abi.zig)

  engine/                  3D foundation. NOT legacy: `World` is the live mesh/material/transform
                           store the 3D renderer walks every frame. C# drives tree editing; Zig
                           owns the storage and traversal.
    math/                  Vec2/3/4, Mat4, Quat, Ray
    scene/                 World, SceneNode, components
    resources/             Mesh, Material, zmesh_format (the .zmesh binary parser)

  renderer/                the wgpu implementation
    wgpu.zig               2D UI renderer + paint-command tessellation
    wgpu_3d.zig            3D forward+ renderer (the EEVEE-style pipeline)
    wgpu_3d_shaders.zig    3D pipeline/shader-module construction
    wgpu_sprites.zig       native batched 2D sprite pass
    wgpu_particles.zig     native billboard particle pass
    wgpu_blur.zig          separable blur helper
    wgpu_ui_shaders.zig    UI shader-module construction
    wgpu_ui_util.zig       UI tessellation helpers
    freetype_text.zig      FreeType glyph atlas + HarfBuzz shaping + shaped-quad cache
    mesh_cache.zig         GPU mesh buffer cache
    backend.zig            BackendId + Caps (wgpu is the only backend; the vtable is gone)
    frame.zig              the ordered per-frame pass list, FrameContext, FrameStats
    transient.zig          transient (per-frame) texture pool
    shader_prelude.zig     WGSL fragments comptime-prepended to shader sources
    uniforms.zig           GPU UBO/param structs (offsets pinned by comptime asserts)
    shaders/               WGSL sources + common_*.wgsl prelude fragments
                           (validated by `zig build check-gpu`)

  ffi/                     the C ABI export layer — everything C# calls
    root.zig               ~163 zigote_* exports: lifecycle, window, paint, scene, render, sprites…
    audio.zig              miniaudio (ma_engine) spatial/surround audio exports
    physics.zig            Jolt physics exports (physics_stub.zig when -Dphysics3d=false)
    ecs.zig                flecs ECS exports (compiled out when -Decs=false)
    assimp_loader.zig      Assimp model import → .zmesh + JSON manifest

  platform/                per-OS native glue (Objective-C)
    macos_menu.m           NSMenu system menu bar
    macos_drag.m           NSDraggingSession app→OS drag-out
```

`root.zig` at each level aggregates its folder into a Zig module. The build wires four public modules
(`zigote_core`, `zigote_ui`, `zigote_engine`, `zigote`) plus the vendored dependency modules, then
compiles `ffi/root.zig` into the shared library. See [building.md](building.md).

## Module dependency direction

```
ffi/*  ──►  renderer/*  ──►  engine/* ────────────────┐
  │            │                                      ├──►  core/*
  └────────────┴──────────────►  ui/*  ───────────────┘
```

`ffi/` is the top: it references the renderer, the render graph, the engine scaffold, text, and the
subsystem FFI modules, and turns them into C exports. Nothing below `ffi/` knows about the C ABI —
the wire structs (the tagged paint-op records, `ZgEvent`, `ZgAbiInfo`, …) are defined in
[`src/abi.zig`](../src/abi.zig), which imports nothing but `std`, and `ffi/` decodes them into
plain Zig data for the layers below.

## What is *not* here

- **No widgets, layout, or scene authoring** — those are C# (`Zigote.UI`, `Zigote.Runtime`). The C#
  scene graph drives tree edits through the FFI; the native `engine/` `World` stores and walks the
  resulting meshes, materials and transforms.
- **No gameplay** — game code lives in each game's own project, outside this repository.
- **No software rendering fallback** — the path is always GPU (wgpu). This is a GPU-first engine.

## Frame flow (the contract with C#)

Each frame the C# host (`Zigote.UI.App`) runs this sequence against the FFI:

```
1. zigote_wait_events(ms)           only when idle: block until input, a post, or a timeout
2. zigote_poll_events()             SDL3 poll → ZgEvent buffer (+ zigote_poll_text_ptr for text/IME/drops)
   … C# runs effects, layout, input dispatch and animation, and paints into a PaintList …
3. zigote_frame_acquire()           optional: take the next swapchain image before late input, so
                                    the vsync wait stays out of input latency
4. zigote_begin_frame(dt, …)        store this frame's parameters
5. zigote_submit_paint_commands()   hand over the tagged paint stream (+ _overlay_commands, _frame_damage)
6. zigote_frame_end(window)         run the frame's pass list and present; then
                                    zigote_frame_was_dropped() says whether the image was skipped
```

`zigote_render_frame_v2` + `zigote_end_frame` are the older pair for the same work and remain for
hosts that have not moved to `frame_end`.

3D content, sprites, and particles are uploaded through their own resource FFI (`zigote_scene_*`,
`zigote_sprites_*`, `zigote_particles_*`) between frames or before the render call; the 3D pipeline is
composited under the 2D UI. See [rendering.md](rendering.md) for the pass-level detail and
[ffi-reference.md](ffi-reference.md) for the full export surface.
