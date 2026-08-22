# Zigote.Engine

The native **Zig + wgpu rendering backend** for [Zigote](../README.md). This repository builds
`libzigote` (`.dylib` / `.dll` / `.so`) and exports a C ABI through
[`src/ffi/root.zig`](src/ffi/root.zig) that the C# frontend (`Zigote.Core`) consumes via
`[LibraryImport]` P/Invoke.

> **The split:** Zig owns the GPU — SDL3 windowing, wgpu rendering, text shaping, physics, ECS, audio,
> model import. C# owns everything above the GPU — widgets, scenes, scripting, and the editor. This
> repo is only the native half; all high-level product logic lives in the parent C# solution.
>
> Architecture, APIs, and coding patterns for the whole engine live in the root. Deeper native-backend guides live in [
`docs/`](docs/README.md):
> [architecture](docs/architecture.md) · [rendering](docs/rendering.md) ·
> [FFI/ABI reference](docs/ffi-reference.md) · [building](docs/building.md) ·
> [subsystems](docs/subsystems.md).

---

## What it provides

- **Rendering (pure wgpu-native)** — a forward+ EEVEE-style path: shadows → sky → geometry + MRT
  G-buffer → SSAO/GTAO + contact → SSR → bloom → AgX tonemap → TAA → UI → composite, over a
  fixed per-frame pass list. Plus a batched 2D UI paint path, native sprite and particle passes,
  and glass refraction. wgpu is the *sole* backend; on macOS it renders through Metal under the hood.
- **Text** — FreeType glyph atlas + HarfBuzz shaping, with a shaped-quad cache keyed by
  `(text, family, size)`.
- **Windowing & input** — SDL3 windows (incl. secondary OS windows), keyboard/mouse/IME/gamepad,
  clipboard, and drag-and-drop, marshalled as `ZgEvent` structs.
- **3D foundation** — Jolt physics (`zphysics`), flecs ECS (`zflecs`), Assimp model import
  (glTF/FBX/OBJ/… → `.zmesh` caches + a JSON manifest), and spatial/surround audio (`zaudio`/miniaudio).
- **Math & scene primitives** — Vec/Mat/Quat/Ray, Color/Rect/Size, and a legacy scene/resource scaffold.

Everything is exposed as `zigote_*` C functions (**286 exports** today — 204 in
[`src/ffi/root.zig`](src/ffi/root.zig), 49 in [`src/ffi/ecs.zig`](src/ffi/ecs.zig), the rest in
chrome/dialogs/channel and the non-macOS shims). The C# P/Invoke
layer is **generated** from these `export fn`s — don't hand-write bindings; regenerate. See the
[FFI/ABI reference](docs/ffi-reference.md) for the full map.

---

## Building

```bash
zig build shared-lib     # build libzigote.dylib / zigote.dll / libzigote.so
zig build test           # module tests (math, scene, resources)
```

> **Use `zig build shared-lib`, not plain `zig build`, to validate.** WGSL shaders are only checked at
> runtime by wgpu-native's embedded naga — a broken shader compiles fine and fails on the first frame.

You normally don't run this by hand: `Zigote.Core.csproj` runs `zig build` as a pre-build step and
copies the resulting library into its output directory. **Native (Zig/shader) changes require a full
process restart** — they apply at load, not via C# hot reload.

### Build options

All default **on**; disable them for lean 2D/UI apps and game exports.

| Option        | Default           | Effect                                                                                   |
|---------------|-------------------|------------------------------------------------------------------------------------------|
| `-Denable3d`  | `true`            | Build the Assimp model importer. Exports/2D apps disable it (the `.zmesh` parser stays). |
| `-Dphysics3d` | `true`            | Build the Jolt 3D physics FFI.                                                           |
| `-Decs`       | `true`            | Build the flecs ECS FFI.                                                                 |
| `-Dstrip`     | ReleaseFast/Small | Strip debug info from the shared library.                                                |

Example lean 2D build: `zig build shared-lib -Denable3d=false -Dphysics3d=false -Decs=false -Doptimize=ReleaseFast`.

### Dependencies

**All native dependencies build from source — no Homebrew, no system libraries.** Freetype, HarfBuzz,
WebP, SDL3, Assimp, Jolt (`zphysics`), flecs (`zflecs`), and miniaudio (`zaudio`) are vendored under
`libraries/` or fetched by the Zig package manager. The one exception is **wgpu-native**, pinned as
per-platform prebuilt static binaries (gfx-rs/wgpu-native **v29.0.1.1**), fetched lazily so only the
archive matching the build target downloads. On macOS, Metal/QuartzCore/Foundation are linked
transitively — needed by wgpu's own Metal backend, not a separate native-Metal path (that was removed).

---

## Layout

```text
src/
  root.zig                 internal aggregate module
  core/                    shared geometry (Color, Rect, Size) + SpinLock
  ui/                      headless text measurement + C ABI paint command structs
  engine/                  3D foundation: math (Vec/Mat/Quat/Ray), scene, resources
  renderer/
    wgpu.zig               2D UI renderer + paint-command tessellation
    wgpu_3d.zig            3D forward+ renderer
    wgpu_sprites.zig       native batched 2D sprite pass
    wgpu_particles.zig     native billboard particle pass
    freetype_text.zig      FreeType/HarfBuzz glyph atlas + shaping
    backend.zig            BackendId + Caps (wgpu is the only backend)
    frame.zig              the ordered per-frame pass list, FrameContext, FrameStats
    transient.zig          transient per-frame texture pool
    shader_prelude.zig     shared WGSL fragments, comptime-prepended
    uniforms.zig           GPU UBO/param structs (offset-pinned via comptime asserts)
    shaders/               WGSL sources + common_*.wgsl prelude fragments
                           (validated by `zig build check-gpu`)
  ffi/
    root.zig               the C ABI export layer (all zigote_* functions)
    assimp_loader.zig      Assimp import → .zmesh + JSON manifest
    audio.zig              miniaudio (ma_engine) spatial/surround audio
    physics.zig            Jolt physics FFI  (physics_stub.zig when -Dphysics3d=false)
    ecs.zig                flecs ECS FFI
  platform/
    macos_menu.m           NSMenu native menu bar
    macos_drag.m           NSDraggingSession app→OS drag-out
```

---

## FFI ABI contract

The C# side pins struct sizes and an ABI version against this backend — they **must** match, or a
mismatch fails loudly at startup (`zigote_get_renderer_abi_info`).

| Struct           | Size      | Note                                                                                                          |
|------------------|-----------|---------------------------------------------------------------------------------------------------------------|
| `ZgPaintCommand` | 112 bytes | Explicit layout; offsets asserted on both sides (comptime `@offsetOf` here, `AbiLayoutTests` in C#).          |
| `ZgEvent`        | 44 bytes  | Text/IME/drop payloads ride the out-of-band `zigote_poll_text` buffer; `WindowId` routes multi-window events. |

Current ABI version: **9** (`abi_version` in [`src/ffi/root.zig`](src/ffi/root.zig)). Bump it whenever
you change field order/size or add fields to these structs.

### Adding an FFI binding

1. `export fn zigote_x(...)` in [`src/ffi/root.zig`](src/ffi/root.zig).
2. Regenerate the C# P/Invoke in `Zigote.Core/Native/NativeEngine.cs` (don't hand-write it).
3. Wrap it publicly in `ZigoteEngine` — never expose `NativeEngine` members directly.

GPU UBO/param structs live in [`src/renderer/uniforms.zig`](src/renderer/uniforms.zig) with comptime
`@offsetOf`/`@sizeOf` guards pinning the field invariants the shaders depend on.