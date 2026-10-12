# Material renderer: correctness pass and next milestones

This is an incremental renderer improvement, not Unity/Unreal feature parity.

## Implemented in this pass

- One native material-to-GPU payload builder for regular and instanced draws; bounds and finite-value checks at upload without rewriting authored materials or changing the uniform ABI.
- Back-face shading normals for double-sided materials, including normal maps.
- Stable GGX denominators that preserve smooth specular peaks; no direct light from the back hemisphere; full clearcoat roughness range.
- Glass direct highlights use the base material specular response instead of an unrelated clearcoat lobe. Glass reflection respects the specular factor; metallic surfaces suppress transmission.
- Transmission tint uses base RGB independently of coverage alpha.
- Depth rejection for foreground refraction samples and frosted-glass taps; half-texel bounds; aspect-correct blur footprint.
- Smooth glass can use the one-color-sample path instead of always taking seven samples because of the BRDF roughness floor. Frosted glass adds depth fetches for rejection, so its cost needs measurement.

Existing materials authored around the old highlight and inverse-alpha tint behavior may need retuning. No frame-time or visual-quality improvement is claimed without scene measurements.

## Verified / verification commands

Material payload regressions cover default/legacy glass, partial transmission, clearcoat, IOR, selection, malformed factors, HDR emission, and preservation of authored values. The native engine suite and `zig build check-gpu` exercise CPU behavior and runtime WGSL compilation respectively. On this macOS setup the GPU check needs the explicit Xcode SDK and the generated `zig-out/.zig-libc-macos.txt` configuration used by the managed native build.

Shader compilation is not an image regression test. Visually compare a fixed-camera scene containing clear and frosted glass, colored panes with different alpha, a foreground occluder, a metallic/roughness grid, rough clearcoat, and back-facing normal-mapped cards. Capture both square and widescreen views, with TAA on/off and fog on/off. Measure GPU time separately for opaque, clear glass, and frosted glass.

## Remaining renderer work, in order

1. **Reproducible image and performance baseline.** Add an automated material test scene with fixed lighting/exposure, image captures, tolerance-based comparisons, and GPU timings at target resolutions/hardware. Include imported assets and instancing, not just procedural spheres.
2. **Material routing and instancing.** Alpha classification, opaque/masked batching, blend/glass instance variants, double-sided culling, and global per-instance transparent ordering are implemented and covered by headless GPU draw validation. Remaining: nonuniform/mirrored instance transforms, instanced shadows, native transmission routing consistent with the editor importer, and explicit zero transmission separate from the legacy glass-mode default. See [implementation plan](implementation-plan.md) for acceptance status.
3. **Transmission architecture.** Glass currently samples one pre-glass scene copy and writes nearest-surface depth. It cannot correctly transmit multiple overlapping glass layers. Replace the hard-coded depth-gap refraction band with authored thickness and attenuation; support solid vs thin surfaces, exit refraction, total internal reflection, transmission roughness, and a defined off-screen fallback. Composite fog once across the transmitted path. Account for transparent objects in front of glass and depth/MSAA edge disagreement.
4. **Material feature contract.** Publish a support matrix spanning import, editor controls, persistence, managed APIs, native resources, and shaders. The standalone `Zigote.Game.Material3D` model is narrower than the native/editor material path. Add separate AO, normal scale, UV sets/transforms, extension textures, volume, anisotropy, sheen, subsurface, and displacement only with end-to-end tests and explicit unsupported-import diagnostics.
5. **Renderer decomposition and scaling.** Split the large renderer into material, geometry, lighting, transmission, and post-process modules with explicit resource dependencies. Profile before introducing clustered lights, GPU-driven visibility, geometry streaming, or additional quality tiers.
6. **Advanced lighting and temporal stability.** Establish tested contracts for reflections/GI, transparent motion vectors/history rejection, shadow quality, volumetrics, and dynamic resolution. Specify budgets and fallback behavior for each supported backend.

Do not call a feature supported solely because a shader parameter or editor slider exists. Acceptance requires import/persistence round trips, rendered reference cases, runtime validation, and a measured cost.
