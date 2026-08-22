# Zigote.Engine V2 — refactoring, optimization and modularization design

Status: **Phases 0, 1, 2, 5 complete; 3 and 4 partial (remainder measured out — see §10, §12)** · Date: 2026-08-22 · See §9 for what landed · Scope: `Zigote.Engine` (Zig) + its C# binding seam
(`Zigote.Generators`, `Zigote.Core/Native`, `build/Zigote.Native.targets`).

V2 **breaks ABI and source compatibility**. The C# side is rebuilt against it in the same change
series; there is no migration shim.

---

## 0. TL;DR

Today the engine is ~28.5k lines of first-party Zig, 70 % of it in four files (`ffi/root.zig` 7687,
`wgpu_3d.zig` 4967, `wgpu.zig` 3988, `freetype_text.zig` 3272). It works, it is fast enough, and it
has been grown one export at a time. The cost of that growth is now structural:

| Problem | Evidence |
|---|---|
| The frame orchestrator lives in the FFI layer | 9 render passes registered and bodied in `ffi/root.zig:6388-6761` |
| Two scene graphs | C# `SceneNode` diffs fields and pushes them into Zig `zg.World.SceneNode` which stores them again; 14 per-node setters + 4 copy-pasted texture loaders + a `[SuppressGCTransition]` allowlist exist to make that cheap |
| Paint commands are transcoded on arrival | `fillPaintList` (`root.zig:2696-3053`) converts every 112 B `ZgPaintCommand` into a second tagged union with per-text heap copies, every frame |
| Five error conventions, four handle schemes, three boolean encodings | 157 `void` exports fail silently; `ZgResult` used by 8/295; ecs handles are unchecked `@ptrFromInt` |
| Abstractions nobody dispatches through | `GpuBackend` vtable: 2/5 methods used; `RenderGraph`: reads/writes/lifetimes/setEnabled unused; `Upscaler`/`RayTracer`/`vulkan`/`d3d12` have zero implementations |
| Per-frame CPU waste | whole-image Wyhash per image command per frame; full UI re-tessellation + 5 whole-buffer uploads even for a static frame; 192-image cache cliff with no eviction |
| Build knowledge encoded 3× | target matrix in `build.zig.zon`, `linkWgpuNative`, and `Zigote.Native.targets`; iOS framework list in 3 places |
| Docs are wrong about the code | "217 exports" (real: 295), "LibraryImport" (real: DllImport), "engine/scene is legacy scaffold" (real: it is the live mesh store) |

V2 is organized around **five decisions**, then a file-level plan, then a phased migration.

---

## 1. Goals and non-goals

**Goals**
1. One convention per concern: errors, handles, booleans, strings, callbacks, stubs.
2. The renderer owns the frame. `ffi/` becomes a thin, generated-friendly shell.
3. One scene store. Zig stores; C# sends commands. No field mirroring.
4. Zero-copy paint ingestion; retained 2D geometry; no per-frame hashing of bulk data.
5. Delete every one-implementation seam and every reserved-for-later contract.
6. Build: one source of truth for the target matrix; every option combination test-built.
7. Every file that survives carries tests that `zig build test` actually runs.

**Non-goals**
- A second GPU backend. wgpu is the backend. (A future native backend would be a separate
  renderer, not a vtable — see §2.4.)
- Changing the C#/Zig ownership split (C# drives the loop; Zig is passive).
- Rewriting vendored C libraries.

---

## 2. The five decisions

### 2.1 One ABI contract

**Errors.** Every export returns `ZgStatus` (`i32`, `0 = ok`, negative = error code from one enum)
*unless* it is a pure getter of a scalar that cannot fail. Handles and counts are returned through
an `out_` parameter, never as a `0 = failure` sentinel. Failure detail goes through the existing log
callback, with the status code attached. The C# wrapper throws on non-zero; no more
"logged a warning and dropped the handle" (`root.zig:809-813`).

Delete: `bool`/`u8`/`u32` boolean returns, `i32` status ladders in dialogs (`-3..1`), `u64 0`
sentinels (32 exports), `ZgResult` (superseded).

**Handles.** One `ZgHandle = u64` = `(generation: u32, index: u32)` into a typed slot table.
One generic `HandleTable(T)` in `core/handle.zig` replaces the five ad-hoc schemes
(pointer-as-handle + `AutoHashMap(u64, void)` for nodes, `image_registry`, `render_textures`,
`windows`, and the *unchecked* ecs `@ptrFromInt`). Stale handles fail with `ZgStatus.stale_handle`
everywhere, including ECS. The `engine: u64` first parameter is dropped from every export — there is
exactly one engine per process today and the atomic `live_engine` check already proves it
(`root.zig:824-840`). If multi-engine ever matters it is a `zigote_engine_select`, not 295 parameters.

**Booleans** are `bool` (C `_Bool`). Not `u8`, not `u32`.

**Strings in** are `ZgStr { ptr, len }` (UTF-8, not NUL-terminated). C# passes `ReadOnlySpan<byte>`
via one helper; no stackalloc-and-NUL at 40 call sites. **Strings/blobs out** are always
caller-buffer `(buf, cap, out_len)`; the one `std.c.free` path (`zigote_model_free`) dies —
`zigote_model_import` writes the manifest into a caller buffer in two calls (size, fill) like
clipboard already does.

**Callbacks** are `(fn ptr, userdata: ?*anyopaque)` pairs. The ecs `callback: usize` hack goes
away once the generator maps `?*anyopaque` (§2.5).

**Wire structs.** `ZgPaintCommand` stops being a union-in-disguise. V2 uses a **tagged command
stream**: a `u32 kind` + `u32 size` header followed by a kind-specific `extern struct`. Rect
commands are 48 B instead of 112; text commands carry their own fields; no `@bitCast` of a float
into a shader id, no `text_len` that is secretly a quad count. Strings and pixel blobs ride a
**side buffer** referenced by `(offset, len)` — the same trick `ZgEvent` already uses for `poll_text`
and which C# `PaintList` already does for its pinned-object-heap strings. One pinned `byte[]` for
commands, one for blobs, two pointers per frame.

`ZgEvent` becomes a tagged union the same way (`kind` + per-kind struct); no more
`scroll_x` meaning relative motion on move and pressure on touch.

`ZgRenderSettings3D` is generated (§2.5), so `settingsToWire`/`settingsFromWire`
(`root.zig:7085`, `7173`, 136 hand-written lines) are deleted.

ABI version restarts at **100**. The startup size check stays and now covers every wire struct.

### 2.2 The renderer owns the frame

New module `renderer/frame.zig` holds the frame sequence that currently lives in
`ffi/root.zig:6307-6980` (`begin_frame` → 9 passes → present → `end_frame`). `ffi/` calls
`Frame.begin(params)`, `Frame.submitUi(stream)`, `Frame.end()` and nothing else.

**RenderGraph is deleted.** Evidence: `reads`/`writes` inert, `computeLifetimes` test-only,
`setEnabled` and `validate()` results unused, 4/7 `PassType` and 4/9 `ResourceKind` arms never
referenced, `TransientPool` has one acquire site, `FrameContext.scene_viewport_*` written and never
read. What remains after deletion is a `for` over an ordered list of pass functions, which is what
`frame.zig` is. The one real need — a temp texture for blur — is a field on the blur pass.

**One command encoder per frame.** Blur, scene-3D submit, and composite each open and submit their
own encoder today (`root.zig:6480-6493`, `passScene3dSubmit`, `passCompositePresent`). V2 records
all passes into one encoder and submits once; the frame-tail `device.poll(false)` stays.

**The two 3D entry points collapse.** `zigote_render_3d` (`root.zig:4139`, 152 lines, pre-graph)
and the `Scene3DPass` inside `render_frame_v2` do the same thing via different code; `Gpu3d.render()`
(`wgpu_3d.zig:3550`) is a third copy with zero callers. V2 has one `Scene3d.render(target)`; the
editor's change-gated viewport is the same function with an offscreen target and a dirty flag. The
`0x3D3D3D3D3D3D3D3D` magic cache key (`root.zig:2836`) becomes an ordinary render-texture handle.

**The two frame APIs collapse.** `begin_frame/render_frame_v2/end_frame` and
`frame_begin/frame_end` (the latter had drifted, `root.zig:6968-6971`; `zigote_frame_begin` has no
C# caller) become `zigote_frame_begin` + `zigote_frame_end`. Secondary windows use the **same**
frame API with a window handle parameter; their parallel 3-call path (`root.zig:4900-4930`) is
deleted.

### 2.3 One scene store

Today `Zigote.Runtime/Scene/SceneNode.cs` keeps Position/Rotation/Scale/light/material and
diff-pushes them through 14 setters into `zg.World.SceneNode`, which keeps them again. That mirror
is the root cause of: chatty per-node P/Invoke (3–6 calls per node on rebuild), the
`[SuppressGCTransition]` allowlist in the generator, four copy-pasted 45-line texture loaders,
and the `node_handles` hashmap that exists only to survive stale pointers.

V2: **Zig is the authority for transforms, lights, materials and meshes. C# sends a command
buffer**, exactly like the paint stream:

```
zigote_scene_apply(stream: ZgStr, blobs: ZgStr) -> ZgStatus
```

with commands `node_create`, `node_destroy`, `node_set_transform`, `node_set_parent`,
`node_set_mesh`, `material_set` (one struct, all PBR factors), `material_set_texture(slot, handle)`,
`light_set`, `visibility_set`. One call per frame regardless of how many nodes changed. C#
`SceneNode` keeps only what C# needs for gameplay (its own transform) and marks dirty; a
`SceneSync` flushes dirty nodes into the stream. The `_pPos.ApproxEquals` diffing moves to one
place.

Textures: `material_set_texture` takes a texture **handle**; loading is a separate async
`zigote_texture_load(path) -> handle` (the `TextureDecodeJob` thread pool at `root.zig:3867`
already exists). No export does a synchronous 256 MB file read on the caller's thread. The four
`*_texture_file` setters become one.

`src/engine/scene` + `src/engine/resources` are **kept and renamed** to `scene/` — the docs calling
them "legacy scaffold" were wrong; they are the live store the 3D renderer walks every frame.
`engine/math` stays as `math/`; `zmath` (220 KB vendored SIMD) is dropped — it backs four call sites
in `mat.zig:15-55`, which get a hand-written 4×4 multiply with `@Vector`.

### 2.4 Delete one-implementation abstractions

| Delete | Why |
|---|---|
| `backend.zig` `GpuBackend` vtable, `BackendId.vulkan/d3d12`, `Upscaler`, `RayTracer`, `Caps.upscalers/raytracing` | Hot path bypasses the vtable (`acquire`/`present`/`configure` dead in `wgpu_backend.zig:55-95`); every reserved arm resolves to `.wgpu`; zero implementations |
| `render/` (RenderGraph, TransientPool, FrameContext) | §2.2 |
| `ZgRendererCaps` | only field that varies is `active_backend`, which is always wgpu |
| `Backend`/`RendererPlan` marker structs (`wgpu.zig:3940`) | tautological test only |
| ecs.zig `pub fn` layer (lines 18–380) | 55 pairs of fn + one-line export wrapper; the reason cited in its header no longer holds |
| audio/physics wrapper blocks in root.zig (`:2283-2690`, `:6041-6265`, ~650 lines) | `audio.zig`/`physics.zig` export directly, like ecs/chrome/dialogs already do |
| `zigote_frame_begin`, `zigote_render_3d`, `Gpu3d.render`, `renderFrameOverlay*` dead `frame_index` params | dead or duplicate |
| 9 exports with no C# caller (`ecs_ensure`, `ecs_make_pair`, `ecs_remove_pair`, `ecs_has_pair`, `ecs_new_w_pair`, `ecs_builtin_prefab`, `get_relative_mouse_mode`, `warp_mouse_in_window`, `debug_gpu_allocated_bytes`) | unused |
| `shaders/text_shader_source.wgsl` | dead and already diverged from the live inline copy at `freetype_text.zig:2860` |
| `libraries/zflecs/src/*.zig`, `libraries/zphysics/src/zphysics.zig`, `libraries/zmesh/{libs/cgltf,libs/par_shapes,src/*}` except `zmeshoptimizer.zig`, `libraries/zpool`, `libraries/zmath` | vendored Zig bindings never imported (both libs are used via `@cImport`); zpool backs one `Pool()` instantiation → replaced by `HandleTable`; zmath per §2.3 |
| `src/core/math/root.zig`, `src/ui/geometry.zig`, `src/ui/root.zig`, `src/core/root.zig` alias chains | six files to reach one `Rect`; four contain only the same alias list |
| `engine/math/ray.zig`, `RigidBody`, `ProjectionKind` in `components.zig` | zero references |
| three hand-rolled `SpinLock`s (`root.zig:551`, `netstream.zig:41`, `channel.zig:61`) | one in `core/sync.zig` |

If a native Metal/Vulkan renderer is ever wanted, it is a second `renderer_*/` directory selected
at **build time** (`-Drenderer=wgpu|metal`) through Zig's comptime module switch — no runtime
vtable, no `Caps` bits nobody reports.

### 2.5 The binding generator grows up

`ZigoteBindingGenerator.cs` scans for `export fn ` at column 0 and substring-slices types. V2
replaces the scanner with a small **Zig-side manifest**: a `zig build ffi-manifest` step that
`@typeInfo`-walks every `export fn` and every `extern struct`/`enum` in `ffi/abi.zig` and emits a
JSON description (names, param types, return type, struct fields with `@offsetOf`/`@sizeOf`, enum
values, doc comments). The C# generator consumes the JSON.

What this buys, concretely:
- Structs and enums are **generated**, not hand-mirrored (`ZgStructs.cs`, 506 lines, goes away;
  `AbiLayoutTests` becomes a generated assert per struct).
- `?*anyopaque`, `ZgStr`, `ZgHandle`, `bool`, fn-pointer-with-userdata map properly; the
  `desktop_shims_stub.zig` exclusion in `Zigote.Core.csproj:19-20` and the hand-written
  `NativeMenu.cs` go away.
- `out_` name-prefix magic is replaced by an explicit `*T` → `out T` rule for pointer params of
  non-blob structs; blobs are `ZgStr`.
- `[SuppressGCTransition]` becomes an attribute-like doc tag on the Zig export
  (`/// ffi: leaf`) instead of a hardcoded list in the C# generator.
- `[LibraryImport]` becomes possible (the generated file is source, not generator output, if the
  manifest step writes `NativeEngine.g.cs` to disk) — faster marshalling stubs, no runtime IL emit.

---

## 3. Performance plan

Each item names the smell, the fix, and the expected effect. None requires new dependencies.

### 3.1 2D UI path (`wgpu.zig`)

| # | Today | V2 |
|---|---|---|
| P1 | `imageKey()` Wyhashes the whole pixel buffer per image command per frame (`wgpu.zig:3594`) | Image commands carry a mandatory texture handle; pixels are uploaded once through `zigote_texture_create`. No per-frame hashing of bulk data, anywhere. |
| P2 | `max_cached_images = 192` cliff: past it, every image is texture+view+bindgroup+upload+destroy per frame (`wgpu.zig:2991`) | Handle table + LRU with byte budget; eviction is explicit and observable in `zigote_debug_engine_stats` |
| P3 | Full re-tessellation + 5 unconditional whole-buffer `writeBuffer`s per frame (`wgpu.zig:878`, `1196-1254`) | Command stream carries a per-subtree **version**; tessellated vertex ranges are retained per subtree and re-emitted only when the version changes. Damage rects (already sent) then also skip tessellation, not just GPU fill. |
| P4 | `ShapeVertex` 40 B × 6 per rect = 240 B/rect (`wgpu.zig:19`, `3332`) | Rects/borders/shadows become **instanced** (one 64 B instance per rect, 4-vertex shared quad via existing `quad_index_buffer`). −75 % vertex bytes; text and images already do this. |
| P5 | One `writeBuffer` per rounded clip per frame (`wgpu.zig:1147`) | Stage the clip ring, one write. Same for 3D shadow slices (`wgpu_3d.zig:3701/3727/3794`) and post params (`:3908-4019`, 7 writes, unconditional) — the `flushModelRing` pattern already in the codebase, applied everywhere. |
| P6 | Blur creates 3 views + 2 bind groups + own encoder + own submit per request (`wgpu_blur.zig:218-262`) | Cache bind groups on `(src,temp,dst)`; record into the frame encoder. |
| P7 | `fillPaintList` transcodes the wire format into a second union with per-text heap dupes (`root.zig:2696-3053`) | Tessellate **directly from the wire stream**. `ui/render/paint.zig`'s union is deleted. |
| P8 | Shaped-run cache hashes the full text per run per frame (`freetype_text.zig:1037`) | Text command carries a caller-supplied `u64 text_id` (C# already interns strings on the POH; the address is stable and free). Hash only the id + style. |

### 3.2 3D path (`wgpu_3d.zig`)

| # | Today | V2 |
|---|---|---|
| P9 | Per-draw `worldMatrix()` + `normalMatrix()` + 7-field repack with no change detection (`wgpu_3d.zig:4767-4805`, duplicated at `4843-4870`) | `scene/` keeps a packed `ModelUniforms` per node, recomputed on `node_set_transform`/`material_set` only (it is the authority now — §2.3). Draw = copy. |
| P10 | Chatty per-node FFI (3–6 calls per node) | `zigote_scene_apply` stream, 1 call/frame |
| P11 | Bloom mip0 allocated full-res, never used (`wgpu_3d.zig:568`) | Drop it |
| P12 | `Gpu3d.init` is 1417 lines with `@setEvalBranchQuota(4000)` | Split per §4; also enables lazy per-feature pipeline creation (SSR pipelines only when SSR is on) rather than the all-22-pipelines spike |

### 3.3 FFI path

| # | Today | V2 |
|---|---|---|
| P13 | `windowFromSdlId` linear-scans the window map per event (`root.zig:854`) | `HandleTable` indexed by SDL id |
| P14 | `image_lock` spinlock taken twice per image command in the paint loop (`root.zig:2848`, `2907`) | Images are handles; the paint loop touches no lock |
| P15 | `audio_lock` around every audio call including file opens (`root.zig:779`) | Lock only the handle table; decode/open outside |
| P16 | Two identical 500k-iteration `mapAsync` spin-polls (`root.zig:4355`, `6938`) | One `readbackSync` helper; fine to keep synchronous, it's capture-time |

---

## 4. Module layout

```
src/
  abi.zig                 every extern struct / enum / ZgStatus / ZgHandle / ZgStr — the wire contract,
                          the single input of the manifest generator; comptime layout asserts live here
  core/
    geometry.zig          Color, Rect, Size, Offset, Constraints, EdgeInsets  (was core/math/geometry.zig)
    handle.zig            HandleTable(T)  (generational slots; replaces 5 schemes + zpool)
    sync.zig              SpinLock (one), readbackSync
    arena.zig             per-frame arena
  math/                   Vec, Mat4 (@Vector), Quat, Frustum   (was engine/math; ray.zig deleted)
  scene/                  World, Node, Material, Mesh, zmesh_format, packed ModelUniforms  (was engine/*)
                          scene/apply.zig — the command-stream decoder (§2.3)
  text/
    faces.zig             FreeType face/strike management, font registry
    shape.zig             HarfBuzz shaping + bidi + run cache        (bidi.zig, bidi_table.zig move here)
    atlas.zig             coverage + colour atlas packing
    layout.zig            retained TextLayout, hit-test, caret
  gpu/                    thin wgpu helpers shared by everything below
    device.zig            instance/adapter/device/surface (was wgpu_backend.zig, minus the vtable)
    texture.zig           ONE upload path with 256-B row alignment (was 4 copies)
    pipeline.zig          ONE render-pipeline builder (was 75 call sites of boilerplate)
    staging.zig           StagingRing — the flushModelRing pattern, generic
    shaders/
      common.wgsl         fullscreen VS, rounded_clip_coverage, srgb_decode — comptime-prepended
      *.wgsl              every shader is a file; zero inline Zig string shaders
  ui/                     2D renderer
    stream.zig            wire-stream cursor (reads abi.zig command headers)
    tessellate.zig        stream → retained per-subtree vertex ranges (P3, P4, P7)
    images.zig            texture handle table + LRU (P1, P2)
    pass.zig              clip ring, draw replay, damage scissor
    blur.zig
  scene3d/                3D renderer (was wgpu_3d.zig, split)
    pipelines.zig         pipeline/bind-group-layout construction (was Gpu3d.init)
    targets.zig           G-buffer / post targets, lazy allocation (was ensurePostTargets)
    shadows.zig
    environment.zig       sky + IBL bake
    geometry_pass.zig     renderLayer, instancing, model staging ring
    post.zig              ssao/ssr/bloom/dof/tonemap/taa
    sprites.zig, particles.zig   (no longer embedded in Gpu3d; take a *Scene3d context)
    uniforms.zig
  renderer/
    frame.zig             THE frame: begin → passes → one submit → present  (§2.2)
    window.zig            main + secondary windows, one code path
  platform/
    sdl_events.zig        poll_events split out of root.zig (344 lines → own file)
    input.zig             gamepad, touch slots, cursors, IME
    macos/*.m             unchanged ObjC
    stubs.zig             ONE stub file covering menu, drag, tray, chrome, dialogs on non-macOS (§5.2)
  ffi/
    engine.zig            init/shutdown/frame/window/input/text exports — thin shells over the above
    scene.zig             zigote_scene_apply + texture_load
    audio.zig, physics.zig, ecs.zig, dialogs.zig, chrome.zig, channel.zig, model_import.zig
                          each exports directly; no wrapper layer in a root file
    root.zig              ~40 lines: comptime force-references only
```

Rules enforced by `zig build test` (a test that walks `@import` graph via `std.zig.Ast` — cheap):
- `abi.zig` imports nothing.
- `ffi/*` may import anything; nothing imports `ffi/*`.
- `ui/`, `scene3d/`, `text/` import `gpu/` and `core/`; they do not import each other or
  `renderer/`. (Today `wgpu.zig:3` imports the engine root — the circular weld goes away.)
- Every file is reachable from a `test {}` block in its directory's `root.zig` — so the current
  situation where `wgpu_3d.zig` tests would not run even if written (`renderer/root.zig:11-16`)
  cannot recur. A test asserts the list matches `ls`.

No function over ~300 lines; `appendPaintOps` (490), `renderLayer` (333), `beginScene` (237),
`ensurePostTargets` (410), `renderPostProcess` (196) are split along the module lines above.

---

## 5. Build and platform

### 5.1 One target matrix

`build.zig` gets a single `const Targets = [_]TargetSpec{…}` table (os, arch, abi, wgpu dep name,
system libs, frameworks). `linkWgpuNative`, `buildMiniaudio`'s framework block, and the Apple
framework block become lookups into it. `zig build print-targets` emits the same table as JSON;
`Zigote.Native.targets` reads that instead of re-deriving `_RidOS`/`_ZigTriple`/`ZigNativeLib`
(`Native.targets:51-119`) and the iOS `<Frameworks>` string.

The two hand-maintained 10-module lists (`build.zig:625-628`, `675-678`) become one
`all_modules` array. `buildWebp`/`buildMiniaudio`/`buildFlecs`/`buildMeshoptimizer` become one
`vendoredStaticLib(spec)`.

### 5.2 One stub strategy

Today non-macOS builds have three strategies for macOS-only symbols: a Zig no-op stub (menu/drag),
comptime-pruned `extern` (chrome/dialogs), and **nothing** (`zigote_mactray_*` — P/Invoked from
`NativeMenu.cs:76-102`, undefined in `libzigote.so` on Linux/Windows). V2: `platform/stubs.zig`
covers every macOS-only export on every other OS, and the manifest generator (§2.5) reads the real
signatures so the stub file no longer needs excluding from the binding build. This is also what
lets `-Decs=false` work on iOS (`Native.targets:128-132`): ecs gets a stub too, generated from the
same manifest.

### 5.3 Test every configuration

`zig build test` runs the module tests for the default config **and** `zig build check-matrix`
compiles `shared-lib` for `{enable3d, physics3d, ecs} ∈ {on,off}³` (8 builds, cached, seconds)
plus a naga validation of every `.wgsl` through wgpu-native's `wgpuDeviceCreateShaderModule` on a
headless adapter where available. Today the mobile and export configurations ship untested.

### 5.4 Toolchain

Pin Zig via a `.zigversion` read by `Native.targets` (today: bare `zig` from `$PATH`, 0.16.0
assumed). Keep the two documented `ponytail:` workarounds (`_FORTIFY_SOURCE`, `use_llvm`) with
their issue links; they are not design debt.

---

## 6. C# side changes (the seam)

- `Zigote.Core/Native/ZgStructs.cs` → deleted (generated).
- `ZigoteEngine.cs` (2814 lines): the ~40 UTF-8/stackalloc string call sites collapse to one
  `ZgStr.From(ReadOnlySpan<byte>)`; error handling becomes `Check(status)` in the generated stub.
- `PaintList` writes the V2 tagged stream + blob buffer directly (it already keeps POH strings and
  a blob fix-up list; the change is the record layout).
- `SceneNode` loses its `_p*` mirror fields; `SceneSync` builds the scene stream.
- `App.cs:1656/1704` mixes old `BeginFrame` with new `FrameEnd`; V2 has one pair.

---

## 7. Migration plan

Each phase is independently shippable and leaves the tree green. ABI version bumps once, at the
end of Phase 2; until then V1 and V2 coexist behind `-Dabi=v1|v2` only for the paint/scene streams.

| Phase | Work | Deletes | Risk |
|---|---|---|---|
| **0. Mechanical split** (1–2 wk) | Move code into the §4 layout without behaviour change; `gpu/texture.zig`, `gpu/pipeline.zig`, `common.wgsl`; fix `root.zig` test reach; `all_modules`; `vendoredStaticLib` | dead items from §2.4 table that have zero callers (backend seams, RenderGraph, wrapper layers, unused vendored trees, alias chains) | Low — BMP golden diffs (`ZIGOTE_SHOT`, `capture_ui_bmp`) gate every step |
| **1. ABI contract** (1–2 wk) | `abi.zig`, `ZgStatus`, `HandleTable`, `ZgStr`, manifest generator, generated C# structs, drop `engine` param, one stub file | `ZgStructs.cs`, `NativeMenu.cs` hand bindings, sentinel returns | Medium — every C# call site touched, but mechanically |
| **2. Streams** (2–3 wk) | Tagged paint stream + direct tessellation (P7), tagged events, `zigote_scene_apply` + texture handles (P1, P10), one frame API for all windows | `fillPaintList`, `paint.zig` union, 14 node setters, 4 texture loaders, `render_3d`, frame-API duplicates | High — this is the compat break; ship behind the flag, flip default, delete V1 |
| **3. Retained 2D** (1–2 wk) | Subtree versions, retained vertex ranges, instanced rects, staged clip ring (P3–P5) | — | Medium — golden diffs must be byte-identical |
| **4. 3D hygiene** (1 wk) | Packed uniforms in `scene/` (P9), lazy per-feature pipelines (P12), staged shadow/post params (P5), bloom mip0 | — | Low |
| **5. Build matrix + docs** (3 d) | `Targets` table, `print-targets`, `check-matrix`, `.zigversion`; rewrite `docs/` from the manifest (export counts generated, never hand-typed again) | `_RidOS`/`_ZigTriple` derivations in MSBuild | Low |

**Verification throughout**: the existing `ZIGOTE_SHOT` + `tools/bmpdiff.py` byte-exact gate for 3D,
`zigote_capture_ui_bmp` for 2D, `AbiLayoutTests` (generated), and a new `Zigote.SmokeTest` frame
benchmark asserting P/Invoke count per frame (≤ 4 on a static UI frame, ≤ 5 with a 3D scene).

---

## 8. Open questions (decided by default unless overridden)

1. **Keep `ffi_mod` as the root for both `shared-lib` and `static-lib`?** Default: yes; iOS static
   needs the same symbol set, and the stub strategy (§5.2) removes the reason they could diverge.
2. **Instance-vs-indexed rects (P4).** Default: instanced; the sprite pass already proves the
   vertex-pulling pattern on this wgpu version.
3. **Does `Zigote.ECS` keep flecs, or does `scene/` become the ECS?** Out of scope for V2; the
   `HandleTable` + stream pattern makes either future choice local to `ffi/ecs.zig`.
4. **Drop `zmath`** (§2.3) assumes `@Vector` codegen is adequate for a 4×4 multiply on all five
   targets. Measure in Phase 0; if it regresses on Android x86_64, keep zmath for `mat.zig` only.

---

## Appendix A — corrections to existing docs

- Export count is **295** (207 root, 55 ecs, 11 chrome, 10 shims, 6 channel, 6 dialogs), not 217/230.
- Bindings are `[DllImport]`, not `[LibraryImport]` (`ZigoteBindingGenerator.cs:88-91`).
- `src/engine/scene` is the **live** mesh/material/transform store (`EngineState.world`,
  `wgpu_3d.zig` walks it every frame), not a legacy scaffold.
- `RenderGraph` does not topologically order anything; execution is registration order
  (`render_graph.zig:9-10`).
- `zigote_mactray_*` has no non-macOS definition.

---

## 9. Implementation log

### Phase 0 — complete

Landed, each step gated by: `zig build test`, `zig build check-gpu`, an `nm` diff of the exported
symbol list, a C# solution build, and a **byte-exact** 3D golden capture (`ZIGOTE_SHOT` +
`tools/bmpdiff.py`) against the pre-refactor engine. The capture stayed byte-identical throughout —
across a rewritten pass system, a deleted backend vtable, ten restructured shaders and three
rewritten texture-upload paths.

| Done | Notes |
|---|---|
| `RenderGraph` → `renderer/frame.zig` | Comptime pass array; `FrameContext` trimmed to fields anything reads; `run()` reports failures so the skip path is testable |
| `GpuBackend` vtable, `Upscaler`, `RayTracer`, `vulkan`/`d3d12`, `wgpu_backend.zig` | Deleted; retired backend ids still decode to `.wgpu`, with a test |
| ecs.zig `pub fn` + `export fn` pairs | 110 functions → 55; exported signatures byte-identical |
| Three hand-rolled spinlocks | One `core/sync.zig`, with a real contention test |
| Geometry alias chain | Six files to reach one `Rect` → the definition plus module roots |
| `renderer/root.zig` completeness | Seven modules were missing, so their tests never ran; a test now reads the directory back and fails if one is missing |
| Shader prelude | 8 duplicate fullscreen VS + 3 `rounded_clip_coverage` + 2 `srgb_decode` → `shaders/common_*.wgsl` |
| **`zig build check-gpu`** (new) | Compiles all 23 shaders headlessly; asserts the texture row-pitch contract |
| `build.zig` | One `all_modules`; one `vendoredStaticLib` |
| 9 uncalled exports | Removed; 295 → 286 |
| **Texture staging** (not in the original plan) | See below |
| Engine docs | `architecture.md`, `rendering.md`, `ffi-reference.md`, `building.md`, `README.md` rewritten to match |

**Unplanned win — the texture staging.** Three upload paths repacked any image whose row pitch was
not a multiple of 256 into an aligned staging buffer: an allocation, a per-row `memcpy` and a
`memset` per upload, for every width not a multiple of 64 px (down a mip chain, most levels). That
alignment rule governs buffer↔texture *copies*, not `queue.writeTexture`. The codebase already
disagreed with itself — the glyph atlas and sprite paths have always written unaligned rows
directly on every shipping backend. Confirmed empirically rather than from the spec (write at a
400-byte pitch, read back through `copyTextureToBuffer`, compare: 0 errors, 0 mismatches of 4000),
and that probe is now a permanent part of `check-gpu`. `createImageTexture` allocates nothing.

### Corrections to this document, found by checking before deleting

1. **`ProjectionKind` is not dead** (§2.4). `Camera` is live (`ffi/root.zig`, `wgpu_3d.zig` read it
   every frame) and `ProjectionKind` is its field type. `RigidBody` *was* dead and is gone.
2. **`libraries/zphysics/src/zphysics.zig` is not dead** (§2.4). It is never `@import`ed by the
   engine, but it is the root module of zphysics's own `build.zig`, which we consume as a path
   dependency. Deleting it breaks the build.
3. **`zmath` is not a four-call-site wrapper** (§2.3, open question 4). It backs the SIMD 4×4
   inverse (`inverseDet`) as well as `mul`/`transpose`. Hand-rolling a 4×4 inverse to shed a
   vendored dependency is a bad trade; it stays. Dropping `zpool` still stands — it backs one
   `Pool()` instantiation that `HandleTable` replaces in Phase 1.
4. **The vendored-tree pruning is deferred.** The genuinely unreferenced trees
   (`libraries/zflecs/src`, `zmesh`'s cgltf/par_shapes) are not compiled, so removing them saves
   disk and nothing else, while making re-vendoring from upstream harder. Low value, non-zero cost.

### Phase 1 — started

| Done | Notes |
|---|---|
| `core/handle.zig` — `HandleTable(T)` | Generational `(generation, index)` handles; `0` is never valid. 8 tests including a churn workload. |
| ECS handles validated | `ffi/ecs.zig`'s world/query/iterator handles were bare `@ptrFromInt` with **no validation** — a stale handle was a use-after-free. Now table indices; ABI unchanged (still opaque `u64`), so no C# change. Destroy untracks first, making a double destroy a no-op. |
| `zpool` dropped | Its one `Pool()` instantiation (per-entity instance buffers) is now a `HandleTable`. Note the behavioural difference handled explicitly: zpool auto-called each value's 0-arg `deinit` on release; the table returns the value and the caller frees. |

Remaining in Phase 1: `abi.zig` + `ZgStatus`, `ZgStr`, dropping the decorative `engine: u64`
parameter, the Zig-side manifest generator and generated C# structs, and the single stub file.
These change every export signature and the whole C# call surface, so they want to land as one
series rather than piecemeal.

### Phase 4 — partial

| Done | Notes |
|---|---|
| P11 bloom level 0 | The chain was allocated full-res with level 0 marked "allocated, unused" — ~16 MB of rgba16float at 1080p, ~¾ of the bloom allocation, on every 3D app. Chain now starts half-res; golden capture byte-identical, which is the proof the level was dead. |
| P12 (partial) — `Gpu3d.init` | 1417 → 1011 lines. The post-processing chain (sampler, 7 layouts, 8 pipelines, 7 buffers) is now `createPostResources`, the one block that reads nothing else init builds. |

**Splitting stopped there deliberately.** The next candidates — environment IBL, shadow cascade,
point shadow, sky — share bind-group layouts, the depth-stencil state and the vertex layout with the
mesh pipelines above them: 11 inbound locals for one block. Threading those through a parameter
struct would make `init` harder to read, not easier. Split where there is a seam; don't manufacture
one. The remaining size is inherent coupling, and `@setEvalBranchQuota` stays for it.

Also found: `Gpu3d.init` has **no `errdefer` anywhere**, so a failure part-way through leaks
everything created before it. Bounded — `ensure3d` latches `gpu_3d_failed`, so it happens at most
once per process — and now documented at the seam. Worth fixing deliberately, not as a side effect.

Not done: P9 (packed model uniforms) depends on the scene-authority change in §2.3. P5's remaining
items (staged shadow-slice and post-param uploads) are ~20 small `writeBuffer` calls per frame —
measurable only as noise against the numbers in §10.

### Phase 5 — complete

| Done | Notes |
|---|---|
| `zig build check-matrix` | All 8 `-Denable3d/-Dphysics3d/-Decs` combinations. Found two real defects (below). |
| `ecs_stub.zig` | `-Decs=false` dropped 49 exports the generated C# bindings still declared — the reason ECS was documented as un-disableable on iOS. |
| Tray stubs | `zigote_mactray_*` was P/Invoked unconditionally but defined only on macOS. |
| `target_specs` + `print-targets` | The nine-way wgpu matrix is data; a test checks every row against `build.zig.zon`. |
| `.zigversion` | Pinned and enforced at comptime. |

All 8 configurations now export an **identical 291-symbol set** — the invariant that makes
platform-independent generated bindings correct, and which was violated before. Two source-level
tests keep it that way.

`Zigote.Native.targets`' RID→triple derivation is deliberately left alone: it drives iOS/Android
paths that cannot be exercised on this machine, and rewriting it blind would trade a documented
duplication for an untestable regression.

---

## 10. §3 re-scoped: what measurement changed

The performance plan in §3 was written from an audit that located code without establishing whether
it runs. Three of its headline items turn out to be cold, and the conclusion matters more than the
individual corrections: **Phases 2 and 3, as specified, are largely premature optimisation.**

| Claim | What is actually true |
|---|---|
| **P1** — "`imageKey()` Wyhashes the whole pixel buffer per image command per frame… 1 MiB hashed per frame". Called the "worst offender". | `imageKey` returns immediately when `cache_key` is set, and **every** real caller passes one with `pixels: null` (`AsyncImage`, `ImagePreviewProvider`, `TilePalettePanel`, `ViewportPanel`, `GameViewport`, …). The hash is a correctness fallback for unkeyed raw pixels, and production never takes it. |
| **P10** — "chatty per-node FFI, 3–6 calls per node" | Every one of the 14 setters in `SceneNode.cs` is dirty-gated behind `first \|\| <value changed>`. 3–6 calls happen on the first push after (re)creation; a static node costs **zero** calls per frame, a moving one costs **one**. |
| **P3** — "a static UI pays 100% of tessellation + upload cost at 60 Hz" | `App.Tick` returns before painting: `if (!_repaint.AnyDirty && !ContinuousUpdate && !ForceContinuousRender) return;`. An idle UI renders **no frames at all**, so it pays no tessellation, no uploads and no hashing. |

What survives: re-tessellation is real on frames that *do* change (one blinking caret rebuilds the
whole list), and P4's unindexed 40-byte × 6 vertices per rect is real whenever anything animates.
Those are worth doing on their own merits — but they do not justify the ABI break and stream
redesign of §2.1/§2.2/§2.3, which were sold largely on P1/P3/P7/P10.

**Recommendation.** Do not execute Phases 2–3 as written. The parts worth keeping are:
- **P4** instanced rects — a contained change inside `wgpu.zig`, no ABI impact.
- **P12** splitting `wgpu_3d.zig` / `Gpu3d.init`, and the §4 module layout — maintainability, no ABI impact.
- The `ZgStatus`/`ZgStr`/`engine`-parameter sweep from §2.1 — genuinely better, but it is a
  coordinated break across ~290 exports and the whole C# call surface, and its strongest
  justification (silent ABI drift) is now closed by the manifest test in §9.

### The benchmark, and what it says

`ZIGOTE_SMOKE_PAINT=<rects>` (added to `Zigote.SmokeTest`) builds a synthetic paint list each frame,
submits and renders it, and reports the median split three ways. The engine had no rendering
benchmark before — which is exactly how §3 came to overstate three of its own items.

It disables vsync, which turned out to dominate everything else: **with vsync on, 500 rects and 8000
rects both measure 6.99 ms/frame**, because the present blocks and the number is the display, not
the engine. Anyone tuning against that figure would have been chasing noise.

Uncapped (RADV / Radeon 780M), linear at ~0.38 µs/command:

| commands | total | C# list build | native transcode | native render |
|---:|---:|---:|---:|---:|
| 2 000 | 1.03 ms | 0.21 | 0.11 | 0.72 |
| 8 000 | 3.20 ms | 0.78 | 0.39 | 2.03 |
| 20 000 | 7.53 ms | 1.99 | 1.08 | 4.46 |

A heavy 2 000-command UI frame costs **~1.0 ms, of which ~0.8 ms is native** — under 5 % of a 60 Hz
budget. `fillPaintList`'s transcode, which P7 proposes deleting the intermediate representation to
remove, is **14 %** of that, so P7's ceiling is roughly 0.1 ms on such a frame. P4's instanced rects
target part of the render column. Neither justifies an ABI break; both are worth doing on their own
merits, and can now be measured rather than argued.

### Phase 2 — complete

| Done | Notes |
|---|---|
| Scene command stream | `zigote_scene_apply`; 14 per-node setters → 1 call. Validate-then-apply, so a malformed batch changes nothing. |
| One frame API for every window | The `zigote_window_submit_paint/_overlay/_render` trio deleted; `window` handle selects, 0 = main. |
| **Tagged paint stream** | 20 kinds × one flat 112-byte struct → per-kind records. A rect is 48 bytes. Nothing aliases. |
| ABI version 100 | |

**Deliberate deviation — blobs stay pointer+len.** §2.1 proposed moving text/pixels/points into a
side buffer referenced by `(offset, len)`. The host already memoises UTF-8 on the pinned object
heap and passes stable pointers, so those cross at **zero copy** today; a side buffer would copy
every string every frame, and §11 puts C# list building at ~26% of a paint frame. The stated goal
(fewer pinned objects) is already met by the POH.

**Deliberate deviation — `ZgEvent` stays a flat 44-byte record.** §2.1 wanted it tagged like the
paint stream. The hazard that justified tagging the paint command does not exist here: one flat
struct served 20 paint kinds with *cross-kind* aliasing (a text shadow's colour in the rectangle
fields), whereas each event kind is consumed by a `switch (kind)` that immediately decodes into the
typed `InputEvent` union, and every aliased slot already has a named accessor at a fixed offset
(`TouchFinger`, `TouchPressure`, `CompositionStart`). Tagging it would mean ~8 record types and a
new poll ABI to remove aliasing that is already mediated, on a path carrying dozens of 44-byte
events per frame.

### Phase 3 — partial

| Done | Notes |
|---|---|
| P5 staged clip ring | N `writeBuffer` calls per frame → one staged write, with a per-slot fallback on allocation failure (dropping clips would render *unclipped* content, not merely slower content). |

**P4 (instanced rects) attempted and reverted — see §12.** P3 (retained geometry / subtree
versions) is not done: §10 established that an idle UI renders no frames at all, so its benefit
applies only to frames that already changed.

---

## 12. P4: why instanced rects were reverted

The plan was "rects/borders/shadows become instanced (one 64 B instance per rect, 4-vertex shared
quad)" — 240 B/rect down to ~40 B. It was implemented far enough to compile, then reverted, for a
reason the design did not account for:

**`shape_vertices` is not homogeneous.** Rect, border and shadow emit six-vertex quads into it —
but so do **polygon fills** (triangle-fanned rings, `mkSolidVertex`) and **bezier strokes**
(triangle strips, `mkStrokeVertex`), as arbitrary triangles that no quad instancing can express.
They share the buffer, the vertex format and the SDF shader. Instancing therefore needs a *second*
pipeline and buffer, plus batch splitting that preserves painter's-algorithm order between the two
— the op list can express that, but it doubles the most-used draw path.

Against that, the measured payoff: at 20 000 rects the vertex upload is ~4.8 MB/frame falling to
~0.8 MB, worth roughly 0.4 ms of a 4.6 ms render column — and a realistic 2 000-command frame is
1.03 ms total, so the saving is ~0.04 ms, about 4% of a path that is already under 5% of a 60 Hz
budget. Doubling the shape path for 0.2% of a frame is not a trade worth making, and the constraint
that forces the doubling is structural rather than incidental.

Recorded rather than silently skipped: if the shape path is ever split for another reason, the
instancing falls out of it cheaply. The `gpu/pipeline.zig` builder and the large-file splits (`wgpu_3d.zig`, the 1417-line
`Gpu3d.init`) remain from Phase 0's §4 layout; they are mechanical but large, and are best done
alongside the rest of Phase 1 rather than as a separate pass over the same files.

### A note on the C# suite

5–7 tests in `Zigote.Tests` fail both with and without these changes, and the failing set rotates
between runs. `SvgAssetTests` (4) need `cargo`, which is not installed on this machine; the rest are
order-dependent flakes in the Reactive and World areas (they pass in isolation). Confirmed
pre-existing by running the full suite against the engine with the changes stashed. Not addressed
here — it is unrelated to the engine — but worth fixing separately, because it makes the suite a
weak gate for exactly this kind of work.

---

## 11. Measured results

Same machine (Radeon 780M / RADV, Zig 0.16, .NET 10). "Before" is the pre-refactor tree
(`2c2279f` / engine `9fe3962`) built and run from a worktree, so both sides are measured the same
way rather than compared against remembered numbers.

### GPU memory — the one real win

Same scene, 640×480, from the engine's own `ZIGOTE_GPU_MEM` diagnostic:

| | before | after |
|---|---:|---:|
| HDR targets (scene + refraction + bloom chain) | 7.8 MB | **5.5 MB** |
| 3D targets, total | 147.6 MB | **145.2 MB** |

The difference is exactly one full-res `rgba16float` — the bloom chain's level 0, which was
allocated and never read. It is linear in pixel count, so **~16.6 MB at 1080p**, on every 3D app
including both mobile heads.

This is also a small cautionary tale. The first before/after run reported an *identical* 147.6 MB,
because `targetMemoryBytes` estimated the bloom chain with a hardcoded `4/3` constant that was only
true while level 0 existed. The allocation had changed; the diagnostic had not. Had I trusted it,
I would have concluded the change did nothing.

### Frame time — unchanged, as expected

3D scene, 300 frames, five runs each, median of medians:

| | before | after |
|---|---:|---:|
| ms/frame | 7.81 | 7.87 |

Statistically identical. That is the correct outcome: the removed bloom level was never *read*, so
no GPU work went with it, and nothing else in this series touched per-frame work on the 3D path.

### 2D paint throughput (new measurement, no "before")

`ZIGOTE_SMOKE_PAINT`, vsync off — the benchmark did not exist before, so there is nothing to compare
against; these are the numbers future work is measured against.

| commands | total | C# build | native transcode | native render |
|---:|---:|---:|---:|---:|
| 2 000 | 1.03 ms | 0.23 | 0.10 | 0.70 |
| 8 000 | 3.38 ms | 0.92 | 0.39 | 2.07 |
| 20 000 | 8.39 ms | 2.66 | 1.12 | 4.61 |

~0.42 µs/command. A heavy 2 000-command UI frame is ~1 ms, under 5 % of a 60 Hz budget.

### Structure

| | before | after |
|---|---:|---:|
| Exported symbols | 295 | 289 |
| `zig build test` assertions (test blocks) | 70 | **91** |
| `ffi/ecs.zig` | 556 | **464** |
| `Native/ZgStructs.cs` (hand-mirrored layout) | 506 | **381** |
| `Gpu3d.init` | 1417 | **1011** |

Engine diff over the series: **73 files, +4964 / −5264**.

The first-party Zig line count went *up* slightly (32 031 → 32 865), and that is worth stating
plainly rather than hiding behind the deletion figure. This was not primarily a deletion exercise:
roughly 1 500 lines of dead or duplicated code went (the render graph, the backend vtable, 55
ECS forwarder pairs, three spinlocks, four alias files, a dead shader, texture staging, `zpool`),
and rather more arrived as things that did not exist before — a generational handle table, the wire
contract, a shader validator, an FFI manifest generator, an ECS stub, the scene command stream, and
21 new tests.

### What did not get faster

Nothing in §3's original list, because — as §10 records — those costs were already avoided by
dirty-gating, cache keys and the idle-frame early return. The wins here are memory, safety
(generational handles, validated ABI layout, stub parity), and coverage (headless shader
validation, all-command golden, config matrix), not throughput.
