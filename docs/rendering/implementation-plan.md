# Rendering improvement plan

Goal: predictable physically based materials, fast rendering, and controls that behave consistently across imported assets, editor previews, and runtime. Unity/Unreal-level quality is a long-term target, not a claim of current feature parity.

## Delivery rules

Each increment includes implementation, relevant CPU tests, runtime shader/pipeline validation, and documented limits. Visual changes additionally need fixed-camera image comparisons. Performance claims require before/after timings on named hardware and resolution. Preserve the current ABI unless a feature requires an explicit versioned extension.

## 1. Material routing and instance correctness — in progress

- Classify instance batches by material before selecting a pipeline.
- Provide opaque/masked, blended, and glass instance pipelines, each with single/double-sided culling.
- Keep opaque/masked instances in one draw per primitive. Expand transparent instances into draw records sorted together with ordinary transparent objects; upload each instance buffer once per layer, not once per instance.
- Use camera-space depth of transformed mesh bounds for ordering instead of distance from the node origin. Preserve empty instance batches as empty.
- Validate normals under nonuniform scale. Keep mirrored-transform handling explicit.
- Centralize native pass routing; define the relationship between alpha coverage and transmission, including masked transmission and explicit zero versus legacy glass defaults.

Acceptance: instanced leaves retain mask/culling, blended cards interleave with ordinary objects, glass samples the post-copy source, and empty batches emit no geometry. CPU ordering tests, native build, GPU validation, and a rendered comparison scene. Initial sorting is object/instance-level; intersecting transparent triangles require a later transparency solution. Instanced shadow casting is a separate follow-up.

## 2. Reproducible rendering lab

Create a deterministic material scene with fixed camera, lighting, exposure, and seed: metallic/roughness grid, clear/frosted/tinted glass, foreground occluders, overlapping panes, alpha masks, normal-mapped back faces, instanced variants, and nonuniform transforms. Add screenshot comparison tooling with tolerances and named baselines. Capture square/widescreen, TAA on/off, and fog on/off.

Acceptance: one command produces captures and a comparison report; GPU timings distinguish opaque geometry, transmission, shadows, and post processing. Establish budgets after measuring current behavior; do not invent FPS targets without a target device/resolution.

## 3. Glass and transparency architecture

Introduce authored thickness and attenuation with persistence/import/editor support. Separate thin panes from solid volumes. Replace the hard-coded depth-gap bend; handle exit refraction, total internal reflection, rough transmission, and off-screen fallback. Correct fog composition and transparent foreground ordering. Choose and benchmark a policy for overlapping layers (bounded depth peeling or another measured approach).

Acceptance: panes, bottles, tinted volumes, and layered glass remain stable during camera movement; image tests cover silhouettes, screen edges, fog, and MSAA. Document approximation limits and cost per quality tier.

## 4. End-to-end material coverage

Publish a matrix for managed API, native resource, importer, editor, persistence, and shader support. Implement separate AO, normal strength, UV transforms/sets, and extension textures first; then volume, anisotropy, sheen, subsurface, and displacement. Remove importer heuristics that overwrite authored values where explicit material data exists. Report unsupported features during import.

Acceptance: every supported property survives import/save/load and produces a reference-tested result. The standalone managed material API matches the supported runtime contract.

## 5. Renderer structure and performance

Extract pipeline creation, material upload, draw preparation, shadows, transmission, and post processing from the large renderer into modules with explicit resource lifetimes. Add pass timings and allocation/draw/upload counters. Use measurements to prioritize pipeline/material cache reuse, clustered lighting, visibility/LOD, and streaming. Add quality presets with independently tunable features.

Acceptance: no regression against the lab baseline; measured frame-time and memory improvements; no per-frame allocations after warm-up for steady scenes. Settings have documented ranges, costs, and defaults.

## 6. Lighting and temporal quality

Improve reflection/GI fallbacks, shadow stability, transparent motion/history handling, volumetrics, and dynamic resolution. Add instanced shadow casting and deformation-aware motion data. Test animation, camera cuts, moving lights, and disocclusion rather than only still frames.

Acceptance: temporal image sequences and GPU timings at each quality tier, with a backend support/fallback matrix. Advanced features ship only when they fit the chosen performance budgets.

## Current increment

Start with item 1's pipeline routing and transparent instance ordering. Preserve opaque batching. Add CPU tests for ordering and empty batches, compile all shaders, build the native library, and use the renderer lab work as the next visual acceptance gate. Layered glass, transparent self-intersection, and instanced shadows remain explicit limitations until their own increments pass validation.

## First increment delivered

Implemented alpha-mode routing before instance batching, six instance pipeline variants (three passes × two culling modes), per-instance transparent draw records interleaved with ordinary objects, view-depth ordering of transformed mesh bounds, and one transform upload per batch per layer. Empty instance batches remain empty. Opaque/masked batches retain one draw per primitive; transparent instances use one draw per primitive per instance.

Validation: 20 native CPU tests passed; native shared library built; 24 WGSL shaders compiled; the real renderer pipelines initialized on the GPU. The headless GPU scene exercises all four alpha modes and both sidedness settings with two instances each, plus empty batches and an ordinary blend object between the instances. It checks pass membership, ordering/instance indices, 13 geometry draw calls, 34 triangles, and submits the commands without GPU validation errors.

This establishes runtime correctness coverage, not a visual baseline or a performance claim. The remaining item-1 tasks are nonuniform/mirrored transform handling and transmission/coverage semantics. Next acceptance work is image readback/comparison for the lab scene. Instanced shadows, layered glass, and intersecting transparent triangles remain unsupported by this increment.
