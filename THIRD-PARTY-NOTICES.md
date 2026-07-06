# Third-Party Notices

The Zigote native engine (`libzigote`) is licensed under the [MIT License](LICENSE).
It incorporates the third-party components below, all of which are compiled into the
`libzigote` shared library. Each entry lists the license and where the full license text
lives in this repository (or in the referenced build-time package). If you redistribute
`libzigote` or an application containing it, these notices must accompany the distribution.

| Component | Version / pin | License | License text |
|---|---|---|---|
| [SDL3](https://libsdl.org) (via the `sdl3` Zig binding) | binding 0.2.1 | zlib | `zig-pkg/sdl3-*/LICENSE` |
| [wgpu-native](https://github.com/gfx-rs/wgpu-native) (prebuilt binaries) | v29.0.1.1 | MIT OR Apache-2.0 | `libraries/wgpu/wgpu-native.LICENSE.MIT`, `wgpu-native.LICENSE.APACHE` (the Zig binding under `libraries/wgpu` is first-party, MIT like the rest of Zigote) |
| [FreeType](https://freetype.org) (built from source via allyourcodebase/freetype) | 2.14.3 | FreeType License (FTL) | fetched at build time (license text in the downloaded package); see the attribution below |
| [HarfBuzz](https://harfbuzz.github.io) (built from source via allyourcodebase/harfbuzz) | 14.1.0 | MIT (“Old MIT”) | fetched at build time (license text in the downloaded package) |
| [Jolt Physics](https://github.com/jrouwe/JoltPhysics) (bundled by zphysics) | vendored | MIT © 2021 Jorrit Rouwe | SPDX headers in `libraries/zphysics/libs/Jolt/`; the zphysics wrapper is MIT (`libraries/zphysics/LICENSE`) |
| [flecs](https://github.com/SanderMertens/flecs) (bundled by zflecs) | vendored | MIT | `libraries/zflecs/libs/flecs/LICENSE`; the zflecs wrapper is MIT (`libraries/zflecs/LICENSE`) |
| [miniaudio](https://miniaud.io) (bundled by zaudio) | vendored | public domain (unlicense) OR MIT-0 (dual) | embedded at the end of `libraries/zaudio/libs/miniaudio/miniaudio.h`; the zaudio wrapper is MIT (`libraries/zaudio/LICENSE`) |
| [Assimp](https://github.com/assimp/assimp) (built from the upstream v5.3.1 source archive) | 5.3.1 | BSD-3-Clause | in the fetched source archive; the Zig build wrapper (forked from allyourcodebase/assimp) is BSD-3-Clause © Felix Queißner (`libraries/assimp/LICENCE`) |
| [zlib](https://zlib.net) (built from source via allyourcodebase/zlib; used by Assimp and FreeType) | 1.3.2 | zlib | fetched at build time (license text in the downloaded package) |
| [libwebp](https://chromium.googlesource.com/webm/libwebp) | vendored | BSD-3-Clause © Google | `libraries/libwebp/COPYING` |
| [zmath](https://github.com/zig-gamedev/zmath) | vendored | MIT © Michal Ziulek / zig-gamedev | `libraries/zmath/LICENSE` |
| [zmesh](https://github.com/zig-gamedev/zmesh) (bundles [par_shapes](https://github.com/prideout/par) © Philip Rideout, [cgltf](https://github.com/jkuhlmann/cgltf), and [meshoptimizer](https://github.com/zeux/meshoptimizer) © 2016-2022 Arseny Kapoulkine — all MIT) | vendored | MIT © Michal Ziulek / zig-gamedev | `libraries/zmesh/LICENSE`; bundled libs carry license headers in their sources under `libraries/zmesh/libs/` |
| [zpool](https://github.com/zig-gamedev/zpool) | vendored | MIT © zig-gamedev contributors | `libraries/zpool/LICENSE` |
| [zigimg](https://github.com/zigimg/zigimg) | 0.1.0 | MIT | `zig-pkg/zigimg-*/LICENSE` |
| SDL Linux build deps (X11/Wayland/ALSA headers etc.) | fetched | various (permissive) | fetched at build time (`sdl_linux_deps-*/LICENSES` in the downloaded package) |

Vendored zig-gamedev libraries carry a `ZIGOTE_VENDOR.txt` recording the upstream revision
and any local patches.

## FreeType attribution

Portions of this software are copyright © The FreeType Project (www.freetype.org).
All rights reserved.
