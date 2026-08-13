# Rendering

Zigote.Engine renders through **wgpu-native** and nothing else. The former native-Metal backend,
Metal ray tracing, and the MetalFX upscaler were removed — this is a single, lean wgpu path. On macOS
wgpu runs *through* Metal under the hood, which is why the Metal/QuartzCore/Foundation frameworks are
still linked; that is wgpu's own backend, not a separate native-Metal path.

There are three cooperating render layers:

1. **2D UI** — a batched paint-command renderer (`renderer/wgpu.zig`). Always present.
2. **3D scene** — a forward+ EEVEE-style pipeline (`renderer/wgpu_3d.zig`). Created lazily.
3. **Sprites & particles** — native batched passes hooked inside the 3D geometry/post stages.

## Backend selection

`renderer/backend.zig` defines the abstraction *seam*: a `GpuBackend` vtable, a `BackendId`
(`auto` / `wgpu` / `vulkan` / `d3d12` — `.metal` was removed), a `Caps` struct, and reserved
`Upscaler` / `RayTracer` sub-interfaces. **Only wgpu is implemented today** — `auto`, `vulkan`, and
`d3d12` all resolve to `.wgpu`. The seam exists so a future native backend can slot in; features gate
on `Caps`, never on backend-name checks.

Selecting a backend re-creates the device, so it is an init-time choice. The instance is created with
per-OS backends (`macOS → Metal`; `Windows → DX12|Vulkan`; else `Vulkan|GL`) rather than the
all-backends default, so adapter creation never probes a GPU stack the engine won't use.

## The RenderGraph

`render/render_graph.zig` is a small backend-agnostic pass scheduler. A `Pass` declares its
`pass_type`, its `reads` and `writes` resource handles, and an `execute` callback; `RenderGraph`
topologically orders passes by their dependencies and runs them. `setEnabled(pass_type, on)` toggles a
whole class of pass. This is what `zigote_render_frame_v2` drives. The immediate-mode
`zigote_render_3d` entry point drives the same underlying pipeline directly for the editor's
change-gated viewport.

## The 3D pipeline (forward+, EEVEE-style)

The full-quality path, in order:

```
shadow  →  sky  →  geometry + MRT G-buffer  →  SSAO/GTAO + contact  →  SSR
        →  bloom  →  AgX tonemap  →  TAA  →  UI  →  composite
```

| Stage | What it does |
|-------|--------------|
| **Shadow** | Cascaded directional shadow map + spot/point shadow maps. Shadow-map allocation is **lazy/grow-on-demand** — a scene with no spot/point lights never pays for those atlases. |
| **Sky** | Procedural sky / environment radiance. |
| **Geometry (MRT)** | Forward-lit meshes writing a G-buffer: HDR colour + normal + albedo + view-space position/depth. Supports the full glTF PBR material set (base/MR/normal/emissive/occlusion, IOR, transmission/glass, clearcoat, double-sided, alpha cutoff). |
| **SSAO / GTAO** | Horizon-based ambient occlusion + contact shadows; optional SSGI single-bounce gathered in the same pass. |
| **SSR** | Screen-space reflections (ray-marched, thickness-tested). |
| **Bloom** | Mip-chain bloom (downsample → tent upsample), threshold above the sky's peak so only genuine highlights bloom. |
| **DoF** | EEVEE-style gather bokeh on linear HDR before tonemap (lazy-allocated targets). |
| **Tonemap** | AgX tonemap → LDR; folds in bloom, AO, SSGI, SSR, exposure. Auto-exposure adapts a 1×1 multiplier per frame. |
| **TAA** | Temporal anti-aliasing (history reprojection + feedback). |
| **UI / composite** | The 2D UI paint list is drawn over the tonemapped scene and composited to the surface. |

Tunables (bloom threshold/knee/intensity, SSAO radius/bias/strength, SSR intensity/distance/steps,
TAA feedback, DoF, exposure, ambient, and a 16-channel **debug-view** selector) are carried in the
`ZgRenderSettings3D` struct and set through `zigote_set_render_settings_3d`. A `diagnostic_mode`
forces a stable physically-plausible baseline (bloom/effects off) for regression comparison.

### G-buffer / bind-group contract

The material bind groups (documented in `renderer/wgpu_3d.zig` / the shaders):

- `g0` — camera + light UBO
- `g1` — per-instance model UBO (transform + PBR factors)
- `g2` — base / normal / metallic-roughness / emissive textures + samplers
- `g3` — shadow map + environment cube (+ refraction source + G-buffer position for glass)

GPU UBO/param structs live in `renderer/uniforms.zig` with `comptime @offsetOf`/`@sizeOf` guards
pinning every field offset the shaders depend on. The per-frame **model-UBO staging ring**
(`pushModel` → `flushModelRing`) uploads the whole written range in one `queue.writeBuffer` per stage
instead of one write per draw.

### Wireframe

wgpu-native 29 exposes no polygon-fill mode, so wireframe is a dedicated **line-list pipeline**: each
triangle `(a,b,c)` is expanded to edges `(a,b)(b,c)(c,a)` in a per-primitive edge index buffer built
once at mesh upload, and the mesh shader emits a flat unlit colour when the wireframe flag is set.

## The 2D UI paint path

The C# side streams a flat array of `ZgPaintCommand` structs (see
[ffi-reference.md](ffi-reference.md)). `renderer/wgpu.zig` parses them into a `paint.Command` union
and tessellates them. Command kinds (`CMD_*`):

| Code | Command | Notes |
|------|---------|-------|
| 0 | `RECT` | filled rounded rect (rounded-box SDF) |
| 1 | `BORDER` | stroked rect |
| 2 | `TEXT` | shaped text run |
| 3 | `IMAGE` | textured quad |
| 4 / 5 | `CLIP_START` / `CLIP_END` | scissor clip push/pop |
| 6 / 7 | `PUSH_OPACITY` / `POP_OPACITY` | layer alpha |
| 8 | `SHADOW` | soft drop shadow |
| 9 | `LIQUID_GLASS` | opt-in translucent backdrop-sampling material |
| 10 | `SHADER_EFFECT` | custom shader effect |
| 11 | `TEXT_LAYOUT` | pre-shaped native text layout handle |
| 12 | `GLYPH_RUN` | explicit glyph-quad array (`ZgGlyphRunQuad`, 32 B each) |
| 13 / 14 | `RENDER_TEXTURE_BEGIN` / `_END` | render-to-texture region |
| 15 | `BLUR` | separable blur region |
| 16 | `BEZIER` | stroked bézier (the only "line" primitive) |
| 17 | `POLYGON` | filled simple/convex polygon — point ring rides the `pixels` side-channel, triangle-fanned into the shape pipeline |

`CMD_POLYGON` is what backs `Zigote.UI.Charts` filled symbols, sector wedges, and seam-free area
fills. It adds no shader and no ABI change — it reuses existing command fields with a new `kind`.

### Text

Text is shaped by HarfBuzz and rasterized into a FreeType glyph atlas (`renderer/freetype_text.zig`).
Two caches keep a per-frame-rendering editor fast:

- a **glyph coverage atlas** (starts 1024² R8, grows ×2 to 4096² on overflow), and
- a **shaped-quad cache** keyed by `(text, family, size, synth-style, spacing-flag)` so repeated
  text skips `hb_shape` entirely.

Weight and italic requests the resolved face cannot provide itself are **synthesized** — outline
embolden proportional to the weight gap (read from OS/2 `usWeightClass`, so real Medium/SemiBold
faces are never double-bolded) and a 12° shear for oblique. Letter-spacing applies per HarfBuzz
*cluster* (marks stay attached, ligatures under tracking shape with `liga/clig` off), and the same
run-splitting drives paint, measurement, cached layouts and caret stops, so they can never disagree.

A colour-emoji atlas materializes lazily on the first colour glyph. Fixed-strike colour fonts
(Apple Color Emoji sbix, Noto CBDT) select the nearest strike via `FT_Select_Size` and scale the
quad; the atlas is keyed by strike so a size ramp reuses one bake. On overflow it resets and
repacks (generation bump re-bakes cached layouts). `zigote_add_emoji_font` probes that the face
actually produces BGRA bitmaps and refuses ones that don't (COLR-outline-only), so hosts can fall
back to a monochrome emoji face. `zigote_text_reset_caches` drops every native + managed text
cache (needed after a wholesale sizing change or face swap).

## Sprites and particles

Both are native batched passes owned by the 3D stack and hooked into the pipeline so the immediate and
render-graph paths both get them:

- **Sprites** (`renderer/wgpu_sprites.zig`) — immediate-mode per-frame model: `zigote_sprites_begin`
  once, then `zigote_sprites_draw` per pre-sorted batch (C# owns sorting; native draws in submission
  order = painter's algorithm). Instances are vertex-pulled quads (`draw(6, count)`, no quad/index
  buffer). Two stages: **scene** (in the geometry pass, gets bloom/AgX) and **overlay** (after post,
  exact colours, no TAA ghosting). Custom WGSL materials are supported.
- **Particles** (`renderer/wgpu_particles.zig`) — the host uploads CPU-simulated particles via
  `zigote_particles_upload`; the shader vertex-pulls a camera-facing quad and blends additive or alpha.
  Drawn inside the geometry pass after transparent meshes (depth-tests the opaque scene, no depth
  write), with the G-buffer MRT targets masked off.

Both are **lazy + failure-isolated**: pipelines are created on first draw and any failure (e.g. a
shader naga rejects) disables that system instead of crashing the frame.

## Lazy memory footprint

The 3D stack (`Gpu3d`: ~22 pipelines, ~14 shader modules, a 2048²-array shadow map, a 512² env
cubemap) is created **lazily** by `ensure3d(...)` on the first real 3D use — a UI-only app never pays
for it, and skips the cold-start shader-compile spike. **Read-only, settings-only, and
invalidation-only FFI must never trigger creation** — `zigote_debug_get_engine_stats`,
`zigote_get/set_render_settings_3d`, and `zigote_render_set_frustum_cull` read/write pending state
applied at creation time. A failed init latches (`gpu_3d_failed`) and the app keeps running UI-only.

## Validating a render change

- Build with `zig build shared-lib`, **not** plain `zig build` — WGSL is only checked at runtime by
  wgpu-native's embedded naga, so a broken shader compiles fine and fails on the first frame.
- Native/shader changes need a **full process restart** — they apply at load, not via C# hot reload.
- The parent solution's `ZIGOTE_SHOT=/path/out.bmp` env var dumps the tonemapped 3D viewport to a
  24-bit BMP deterministically at a fixed frame; `tools/bmpdiff.py` gives a byte-exact before/after
  gate. `zigote_capture_ui_bmp` is the 2D-paint counterpart.
