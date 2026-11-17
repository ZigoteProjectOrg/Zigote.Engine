# Changelog

All notable changes to the Zigote native engine (`libzigote`) are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

The engine and the C# SDK in [Zigote](https://github.com/ZigoteProjectOrg/Zigote) ship under the
**same version number**: `Zigote.Engine.runtime.<rid>` 0.1.0 is this repository at tag `v0.1.0`.
The FFI ABI has its own number (`abi_version` in `src/ffi/root.zig`, checked by C# at startup) and
is listed per release below — a release that bumps it is not binary-compatible with the previous
SDK, whatever the semver says.

## [Unreleased]

### Fixed
- The Radiance `.hdr` decoder bounds the image dimensions. An unchecked `width * height` wrapped
  for a crafted header, and the scanline copy then wrote past a too-small buffer.
- `zigote_audio_decode_file` computes the decoded size with checked arithmetic and refuses more
  than 2 GiB; a frame count taken from the file could wrap the allocation size.
- A glyph wider than the atlas no longer spills into the rows below it (or past the pixel store):
  the reservation fails and the atlas grows or resets, as it already did for a glyph too tall.
- The release workflow publishes only when the module tests passed, and accepts prerelease tags
  (`vX.Y.Z-rc.N`) against the base version in `build.zig.zon`.

## [0.1.0] - 2026-11-04

First public release. ABI version **100** (the V2 contract).

### Rendering
- wgpu-native (v29.0.1.1, prebuilt per platform) is the sole GPU backend: Metal on macOS and iOS,
  Vulkan on Linux and Android, D3D12 on Windows, WebGPU in the browser.
- Forward+ 3D path over a fixed per-frame pass list: cascaded shadows, sky, MRT G-buffer,
  SSAO/GTAO and contact shadows, SSR, bloom, AgX tonemapping, TAA, UI, composite. Post-processing
  targets are allocated lazily, so a 2D app never pays for them.
- Batched 2D UI renderer with instanced rects, borders and shadows, a transient texture pool,
  rounded clips staged in one write, and chained backdrop captures.
- Liquid glass: per-pixel adaptive luminance against a legibility anchor, adaptive probes,
  hairline bevels, outer-only shadows, linear-blend alpha.
- Native sprite and billboard particle passes; the UI can be captured straight into an image texture.
- Frame diagnostics: dropped-frame reporting, a timing split per frame, scene texture capture, and
  wgpu validation errors surfaced in the log.

### Text
- FreeType rasterisation and HarfBuzz shaping with a shaped-quad cache keyed by text, family and size.
- Synthetic bold and oblique for faces that lack the style, the platform's colour emoji, per-cluster
  tracking, subpixel glyph positioning, and a light-on-dark weight matched to Skia.

### Windowing and input
- SDL3 windows, including secondary OS windows (Wayland multi-window included), fullscreen,
  always-on-top and aspect-ratio lock.
- Keyboard, mouse, IME composition, gamepad, clipboard, drag and drop, native file dialogs,
  client-side decorations, and a native `NSMenu` bar on macOS.

### Subsystems
- Jolt 3D physics (`-Dphysics3d`), flecs ECS (`-Decs`), miniaudio spatial and surround audio,
  Assimp model import to `.zmesh` caches with a JSON manifest (`-Denable3d`).
- Platform channels (`zigote_channel_register` / `zigote_channel_send`) for plugins.

### FFI and ABI
- The V2 wire contract: a tagged paint-command stream and a scene command stream instead of flat
  structs, generational `u64` handles, one `ZgStatus` code for every fallible export, and one frame
  API for every window.
- `zigote_get_renderer_abi_info` lets the managed side refuse to run on a mismatch;
  `zig build ffi-manifest` writes the real sizes and offsets for the C# mirror tests.
- SIGTRAP is no longer reported as a native crash.

### Targets
- macOS (arm64, x64), Linux (x64, arm64), Windows (x64), Android (arm64, x64, linked at 16 KB page
  alignment), iOS device and simulator (static archive), and the browser (`wasm32-emscripten` with
  a WebGPU JavaScript backend under `web/`).

### Build
- Zig 0.17.0, pinned in `.zigversion` and enforced by `build.zig`.
- Every dependency builds from source except the pinned wgpu-native binaries; no system libraries.
- `shared-lib`, `static-lib`, `test`, `check-gpu` (headless shader compilation), `check-matrix`
  (every feature-flag combination), `ffi-manifest` and `print-targets` build steps.

### Known limitations
- Android and iOS build the lean feature set: no Assimp import and no Jolt physics yet.
- Cross-compiling for Windows from macOS or Linux produces the GNU ABI; the MSVC ABI needs a
  Windows host.
- The browser build is single-threaded and has no video or network audio streams.

[Unreleased]: https://github.com/ZigoteProjectOrg/Zigote.Engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ZigoteProjectOrg/Zigote.Engine/releases/tag/v0.1.0
