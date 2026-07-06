# Zigote.Engine documentation

Deeper guides for the native Zig + wgpu backend. Start with the repo [README](../README.md) for the
overview; these go one level down. Architecture, APIs, and coding patterns for the **whole** engine
(the C# side included) live in the parent solution's [CLAUDE.md](../../CLAUDE.md).

| Guide                                | What it covers                                                                                                               |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| [architecture.md](architecture.md)   | The Zig/C# split, the source layout, module dependency direction, and the per-frame contract.                                |
| [rendering.md](rendering.md)         | The wgpu render path — RenderGraph, the forward+ 3D pipeline, the 2D paint commands, sprites & particles, lazy memory.       |
| [ffi-reference.md](ffi-reference.md) | The C ABI: ABI versioning, the `ZgPaintCommand` / `ZgEvent` wire structs, and the categorized `zigote_*` export map.         |
| [building.md](building.md)           | Build steps, the `-Denable3d`/`-Dphysics3d`/`-Decs`/`-Dstrip` options, dependencies, wgpu pinning, and cross-compilation.    |
| [subsystems.md](subsystems.md)       | Windowing/input (SDL3), text (FreeType/HarfBuzz), audio (miniaudio), physics (Jolt), ECS (flecs), and model import (Assimp). |