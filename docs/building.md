# Building

Requires **Zig 0.16+** (matching the toolchain the parent solution pins). Two build steps:

```bash
zig build shared-lib     # build libzigote.dylib / zigote.dll / libzigote.so
zig build test           # run the module tests (math, scene, resources)
```

> **Validate with `zig build shared-lib`, not plain `zig build`.** WGSL shaders are only checked at
> runtime by wgpu-native's embedded naga — a broken shader compiles fine and fails on the first frame.
> A bare `zig build` in Debug can also mask errors behind a stale cache; prefer the explicit step.

You normally never run this by hand. `Zigote.Core.csproj` runs `zig build` as a pre-build step and
copies the resulting library into its output directory. **Native (Zig/shader) changes require a full
process restart** — they apply at load, not via C# hot reload, and the `zig build` pre-step doesn't
re-run mid-session.

## Build options

All default **on** (the full engine). Disable them for lean 2D/UI apps and game exports. They are
independent — in particular `physics3d` and `ecs` are **not** gated by `enable3d` (a game export
passes `-Denable3d=false` yet still needs Jolt and flecs at runtime).

| Option        | Default                                           | Effect                                                                                                                                                                                                                                                                         |
|---------------|---------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-Denable3d`  | `true`                                            | Build the **Assimp** model importer (editor-only). Games load pre-baked `.zmesh`; the `.zmesh` parser stays ungated in `engine/resources/zmesh_format.zig`. The wgpu 3D renderer itself stays either way — its targets are allocated lazily, so a 2D app pays no runtime cost. |
| `-Dphysics3d` | `true`                                            | Build the **Jolt** 3D physics FFI (`ffi/physics.zig` + the JoltC static lib). `false` swaps in `physics_stub.zig` — the `zigote_physics_*` wrappers still compile and link, as no-ops.                                                                                         |
| `-Decs`       | `true`                                            | Build the **flecs** ECS FFI (`ffi/ecs.zig` + the flecs static lib). `false` drops the whole module — the `zigote_ecs_*` exports vanish, so a caller must never invoke them.                                                                                                    |
| `-Dstrip`     | `true` for ReleaseFast/ReleaseSmall, else `false` | Strip debug info from the shared library. ELF embeds DWARF for every static dep straight into the `.so` (~90 MB of a 123 MB Linux build); stripping cuts it dramatically. Override with `-Dstrip=false` for a symbolized profiling build.                                      |

Standard Zig options apply too: `-Doptimize=Debug|ReleaseSafe|ReleaseFast|ReleaseSmall` and
`-Dtarget=<triple>` for cross-compilation.

**Lean 2D/UI build** (what a pure-widget app or a 2D game export uses):

```bash
zig build shared-lib -Denable3d=false -Dphysics3d=false -Decs=false -Doptimize=ReleaseFast
```

**Game export build** (needs Jolt + flecs at runtime, drops Assimp, strips symbols):

```bash
zig build shared-lib -Denable3d=false -Doptimize=ReleaseFast -Dstrip
```

## Dependencies

**Every native dependency builds from source — no Homebrew, no system libraries.** (The old CLAUDE.md
claim that macOS deps come from `/opt/homebrew` is obsolete.) They are vendored under `libraries/` or
fetched by the Zig package manager into `zig-pkg/`:

| Dependency                  | Role                          | Source                                                                                |
|-----------------------------|-------------------------------|---------------------------------------------------------------------------------------|
| **FreeType**                | glyph rasterization           | Zig package (`freetype`, with libpng)                                                 |
| **HarfBuzz**                | text shaping                  | Zig package (`harfbuzz`)                                                              |
| **SDL3**                    | windowing / input / clipboard | Zig package (`sdl3`)                                                                  |
| **libwebp**                 | WebP decode                   | vendored, built by `buildWebp`                                                        |
| **Assimp**                  | model import                  | Zig package (`zig_assimp`, allyourcodebase, formats trimmed) — only when `-Denable3d` |
| **Jolt** (`zphysics`)       | 3D physics                    | Zig package (`zphysics` → `joltc`) — only when `-Dphysics3d`                          |
| **flecs** (`zflecs`)        | ECS                           | vendored, built by `buildFlecs` — only when `-Decs`                                   |
| **miniaudio** (`zaudio`)    | spatial/surround audio        | vendored, built by `buildMiniaudio`                                                   |
| **meshoptimizer** (`zmesh`) | mesh optimization             | vendored, built by `buildMeshoptimizer`                                               |
| **zmath / zpool**           | SIMD math / handle pools      | vendored                                                                              |
| **zigimg**                  | image decode                  | Zig package                                                                           |

### wgpu-native — the one exception

wgpu-native is **not** built from source. It is pinned as per-platform **prebuilt static** binaries,
gfx-rs/wgpu-native **v29.0.1.1**, fetched lazily so only the archive matching the build target
downloads. Five platforms are wired in `build.zig.zon`:

- `wgpu_macos_aarch64`, `wgpu_macos_x86_64`
- `wgpu_windows_x86_64` (MSVC)
- `wgpu_linux_x86_64`, `wgpu_linux_aarch64`

Adding a new desktop target = add its prebuilt archive here + wire the link in `build.zig`.

On **macOS**, Metal / QuartzCore / Foundation / CoreFoundation frameworks + `objc`/`iconv` are linked
transitively — these are needed by **wgpu's own Metal backend**, not a separate native-Metal path
(that was removed). On **Windows**: `ws2_32`, `winmm`, `dbghelp`. On **Linux**: `unwind`. Audio adds
CoreAudio/AudioUnit/AudioToolbox on macOS and `pthread`/`m`/`dl` on Linux.

### wgpu binding drift — read before touching extern structs

The Zig wgpu binding under `libraries/wgpu` is **hand-matched** to the pinned v29.0.1.1
prebuilt headers, with no cross-boundary compiler check. When touching any `extern struct` there, diff
it against `zig-pkg/<wgpu-prebuilt>/include/webgpu/{webgpu,wgpu}.h` first. Drift here feeds wgpu stack
garbage and fails only at runtime.

## Cross-compilation

Zig cross-compiles from any host. The lazy wgpu fetch means only the target's prebuilt archive
downloads. Example:

```bash
zig build shared-lib -Dtarget=x86_64-linux-gnu  -Doptimize=ReleaseFast -Denable3d=false
zig build shared-lib -Dtarget=aarch64-linux-gnu -Doptimize=ReleaseFast -Denable3d=false
```

Native cross-compilation of all five targets from macOS is verified; the Windows/Linux libraries are
built but not yet run on their own hardware. **NativeAOT cross-OS is unsupported by .NET** — the C#
game-export flow only produces AOT binaries for host-OS RIDs and falls back to a self-contained JIT
publish for others (see the parent solution's export docs).
