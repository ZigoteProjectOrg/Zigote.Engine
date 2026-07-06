# Architecture

Zigote.Engine is the **native half** of the Zigote engine. It is a Zig library that owns the GPU and
every OS-facing resource, and exposes a flat C ABI (`zigote_*` functions) that the C# frontend
(`Zigote.Core`) drives via `[LibraryImport]` P/Invoke.

```
┌─────────────────────────────────────────────────────────────┐
│  C# solution (Zigote.Core, Zigote.UI, Zigote.Game, Editor…)  │  product logic
│    widgets · scenes · scripting · physics coordination       │
└───────────────────────────┬─────────────────────────────────┘
                            │  [LibraryImport] P/Invoke — the zigote_* C ABI
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

The full design rationale for the whole engine lives in the root [CLAUDE.md](../../CLAUDE.md) of the
parent C# solution — this repo documents only the native backend.

## Source layout

```text
src/
  root.zig                 internal aggregate module (re-exports the sub-modules)

  core/                    engine-neutral primitives
    math/                  Color, Rect, Size

  ui/                      headless UI support (no GPU)
    text.zig               text measurement + text styles
    geometry.zig           2D geometry helpers
    render/paint.zig       (C ABI paint command structs live in ffi/root.zig today)

  engine/                  3D foundation (mostly legacy scaffold; the live scene lives in C#)
    math/                  Vec2/3/4, Mat4, Quat, Ray
    scene/                 World, SceneNode, components
    resources/             Mesh, Material, zmesh_format (the .zmesh binary parser)

  render/                  backend-agnostic frame orchestration
    render_graph.zig       RenderGraph: passes with reads/writes, topological execution
    frame_context.zig      per-frame context passed to passes
    render_resource.zig    transient resource pool + handles

  renderer/                the wgpu implementation
    wgpu.zig               2D UI renderer + paint-command tessellation
    wgpu_3d.zig            3D forward+ renderer (the EEVEE-style pipeline)
    wgpu_3d_shaders.zig    3D pipeline/shader-module construction
    wgpu_sprites.zig       native batched 2D sprite pass
    wgpu_particles.zig     native billboard particle pass
    wgpu_blur.zig          separable blur helper
    wgpu_backend.zig       WgpuBackend: device/queue/surface lifecycle
    wgpu_ui_shaders.zig    UI shader-module construction
    wgpu_ui_util.zig       UI tessellation helpers
    freetype_text.zig      FreeType glyph atlas + HarfBuzz shaping + shaped-quad cache
    mesh_cache.zig         GPU mesh buffer cache
    backend.zig            GpuBackend vtable, BackendId, Caps, Upscaler/RayTracer seams
    uniforms.zig           GPU UBO/param structs (offsets pinned by comptime asserts)
    shaders/               21 WGSL shader sources (canonical; naga-validated at runtime)

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
ffi/*  ──►  renderer/*  ──►  render/*  ──►  engine/* ─┐
  │            │                                      ├──►  core/*
  └────────────┴──────────────►  ui/*  ───────────────┘
```

`ffi/` is the top: it references the renderer, the render graph, the engine scaffold, text, and the
subsystem FFI modules, and turns them into C exports. Nothing below `ffi/` knows about the C ABI —
the wire structs (`ZgPaintCommand`, `ZgEvent`) are defined in `ffi/root.zig` and passed down as plain
Zig data.

## What is *not* here

- **No widgets, layout, or scenes** — those are C# (`Zigote.UI`, `Zigote.Game`). The `engine/scene`
  Zig scaffold is legacy; the live scene graph is C# and pushes resources down through the FFI.
- **No gameplay** — game code lives in the C# side's `examples/<project>/`.
- **No software rendering fallback** — the path is always GPU (wgpu). This is a GPU-first engine.

## Frame flow (the contract with C#)

Each frame the C# host runs this exact sequence against the FFI (from the header of `ffi/root.zig`):

```
1. zigote_poll_events()             SDL3 poll → ZgEvent buffer  (+ zigote_poll_text_ptr for text/IME/drops)
   … C# builds the widget tree, runs layout, produces a flat paint-command list …
3. zigote_begin_frame()             store this frame's parameters
4. zigote_submit_paint_commands()   hand over the root paint list
5. zigote_submit_overlay_commands() hand over the overlay paint list (optional)
6. zigote_render_frame_v2()         execute the render graph + present
7. zigote_end_frame()               reset per-frame state
```

3D content, sprites, and particles are uploaded through their own resource FFI (`zigote_scene_*`,
`zigote_sprites_*`, `zigote_particles_*`) between frames or before the render call; the 3D pipeline is
composited under the 2D UI. See [rendering.md](rendering.md) for the pass-level detail and
[ffi-reference.md](ffi-reference.md) for the full export surface.
