const std = @import("std");

/// Cross-compiling to an Apple target (`-Dtarget=x86_64-macos` / `-Dtarget=aarch64-ios-simulator`
/// with `--sysroot "$(xcrun --sdk … --show-sdk-path)"`) does not derive the SDK search paths from
/// the sysroot the way a native build auto-detects them, so every module that compiles against or
/// links Apple frameworks needs them added explicitly. No-op when no `--sysroot` is passed
/// (native builds). iOS builds are always cross builds and always need this.
fn addAppleSdkPaths(b: *std.Build, mod: *std.Build.Module) void {
    const sysroot = b.sysroot orelse return;
    mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sysroot, "System/Library/Frameworks" }) });
    // Xcode 26 SDKs split some framework dependencies (e.g. UIKit's UIUtilities) into a separate
    // SubFrameworks directory that is on the default search path in Xcode but not for us.
    mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sysroot, "System/Library/SubFrameworks" }) });
    mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sysroot, "usr/include" }) });
    // The linker prepends the sysroot to -L paths itself, so this one stays sysroot-relative
    // (an absolute path would be doubled into <sdk>/<sdk>/usr/lib).
    mod.addLibraryPath(.{ .cwd_relative = "/usr/lib" });
}

fn linkWgpuNative(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    const os = target.result.os.tag;
    const arch = target.result.cpu.arch;

    const dep_name: []const u8 = switch (os) {
        .macos => switch (arch) {
            .aarch64 => "wgpu_macos_aarch64",
            .x86_64 => "wgpu_macos_x86_64",
            else => std.debug.panic("wgpu-native: unsupported macOS arch {s}", .{@tagName(arch)}),
        },
        .windows => switch (arch) {
            .x86_64 => "wgpu_windows_x86_64",
            else => std.debug.panic("wgpu-native: unsupported Windows arch {s}", .{@tagName(arch)}),
        },
        // Android is not its own OS tag — it is the `android` ABI of linux (aarch64-linux-android).
        .linux => if (target.result.abi.isAndroid()) switch (arch) {
            .aarch64 => "wgpu_android_aarch64",
            .x86_64 => "wgpu_android_x86_64", // emulator
            else => std.debug.panic("wgpu-native: unsupported Android arch {s}", .{@tagName(arch)}),
        } else switch (arch) {
            .x86_64 => "wgpu_linux_x86_64",
            .aarch64 => "wgpu_linux_aarch64",
            else => std.debug.panic("wgpu-native: unsupported Linux arch {s}", .{@tagName(arch)}),
        },
        // Device and simulator are distinct platforms with distinct archives; the simulator is
        // selected by the `simulator` target ABI (aarch64-ios-simulator).
        .ios => switch (arch) {
            .aarch64 => if (target.result.abi == .simulator)
                "wgpu_ios_aarch64_simulator"
            else
                "wgpu_ios_aarch64",
            else => std.debug.panic("wgpu-native: unsupported iOS arch {s}", .{@tagName(arch)}),
        },
        else => std.debug.panic("wgpu-native: unsupported OS {s}", .{@tagName(os)}),
    };

    // The Zig binding (libraries/wgpu) is pure `extern fn` — no @cImport — so only the static
    // archive is needed to resolve symbols; the bundled headers are not added to the include path.
    if (b.lazyDependency(dep_name, .{})) |dep| {
        // Windows links wgpu-native's DLL import lib, not its static .lib: the static MSVC archive
        // drags in the full MSVC CRT + WinRT/d3dcompiler import libs MinGW lacks, so it can't link a
        // `windows-gnu` cross-build (the only Windows target that cross-compiles from macOS/Linux,
        // since `windows-msvc` needs the MSVC SDK). The import lib pulls only the C exports;
        // wgpu_native.dll self-resolves d3d12/dxgi/CRT at load and is installed next to zigote.dll.
        const lib_path = if (os == .windows) "lib/wgpu_native.dll.lib" else "lib/libwgpu_native.a";
        mod.addObjectFile(dep.path(lib_path));
        // wgpu_native.dll itself is installed next to zigote.dll by `installWgpuDll` in build().
    }

    // Transitive system dependencies of the wgpu-native static archive.
    switch (os) {
        .macos => {
            // From `otool -L libwgpu_native.dylib`: Metal/QuartzCore/Foundation/CoreFoundation +
            // libobjc + libiconv. (Metal/QuartzCore/Foundation are also linked for the Metal backend;
            // frameworks dedupe.)
            mod.linkFramework("Metal", .{});
            mod.linkFramework("QuartzCore", .{});
            mod.linkFramework("Foundation", .{});
            mod.linkFramework("CoreFoundation", .{});
            mod.linkSystemLibrary("objc", .{});
            mod.linkSystemLibrary("iconv", .{});
            addAppleSdkPaths(b, mod);
        },
        .linux => if (target.result.abi.isAndroid()) {
            // Vulkan is dlopen'd at runtime; the archive's hard deps are the NDK log and
            // android (ANativeWindow) libs, resolved against the NDK sysroot.
            mod.linkSystemLibrary("log", .{});
            mod.linkSystemLibrary("android", .{});
        } else {
            // libc (link_libc) already covers pthread/dl/m; Rust's panic/unwind path needs unwind.
            mod.linkSystemLibrary("unwind", .{});
        },
        .windows => {
            // Using wgpu_native.dll's import lib (above), so the DLL resolves its own d3d12/dxgi/
            // WinRT/CRT dependencies at load — nothing extra to link statically here.
        },
        .ios => {
            // Same Metal stack as macOS, resolved against the iOS SDK sysroot. No iconv — the
            // iOS archive doesn't pull it.
            mod.linkFramework("Metal", .{});
            mod.linkFramework("QuartzCore", .{});
            mod.linkFramework("Foundation", .{});
            mod.linkFramework("CoreFoundation", .{});
            mod.linkSystemLibrary("objc", .{});
            addAppleSdkPaths(b, mod);
        },
        else => {},
    }
}

/// Decode-only source set for the vendored libwebp (libraries/libwebp), taken from upstream's
/// Makefile.am (dsp COMMON + per-arch SIMD decode variants, full dec/, the decode utils subset, and
/// demux). The SIMD files self-gate on compiler arch macros (WEBP_USE_SSE2/NEON/…), so compiling all
/// of them is cross-safe — the wrong-arch ones become empty objects. Encode + sharpyuv are omitted.
const webp_decode_srcs = [_][]const u8{
    // dsp — common
    "src/dsp/alpha_processing.c",         "src/dsp/cpu.c",                    "src/dsp/dec.c",
    "src/dsp/dec_clip_tables.c",          "src/dsp/filters.c",                "src/dsp/lossless.c",
    "src/dsp/rescaler.c",                 "src/dsp/upsampling.c",             "src/dsp/yuv.c",
    // dsp — SSE2 / SSE4.1 / NEON decode
    "src/dsp/alpha_processing_sse2.c",    "src/dsp/dec_sse2.c",               "src/dsp/filters_sse2.c",
    "src/dsp/lossless_sse2.c",            "src/dsp/rescaler_sse2.c",          "src/dsp/upsampling_sse2.c",
    "src/dsp/yuv_sse2.c",                 "src/dsp/alpha_processing_sse41.c", "src/dsp/dec_sse41.c",
    "src/dsp/lossless_sse41.c",           "src/dsp/upsampling_sse41.c",       "src/dsp/yuv_sse41.c",
    "src/dsp/alpha_processing_neon.c",    "src/dsp/dec_neon.c",               "src/dsp/filters_neon.c",
    "src/dsp/lossless_neon.c",            "src/dsp/rescaler_neon.c",          "src/dsp/upsampling_neon.c",
    "src/dsp/yuv_neon.c",
    // dec — full decoder
                    "src/dec/alpha_dec.c",              "src/dec/buffer_dec.c",
    "src/dec/frame_dec.c",                "src/dec/idec_dec.c",               "src/dec/io_dec.c",
    "src/dec/quant_dec.c",                "src/dec/tree_dec.c",               "src/dec/vp8_dec.c",
    "src/dec/vp8l_dec.c",                 "src/dec/webp_dec.c",
    // utils — decode subset
                  "src/utils/bit_reader_utils.c",
    "src/utils/color_cache_utils.c",      "src/utils/filters_utils.c",        "src/utils/huffman_utils.c",
    "src/utils/quant_levels_dec_utils.c", "src/utils/rescaler_utils.c",       "src/utils/random_utils.c",
    "src/utils/thread_utils.c",           "src/utils/utils.c",
    "src/utils/palette.c", // GetColorPalette, referenced by utils.c's WebPGetColorPalette
    // demux — container / animation
    "src/demux/anim_decode.c",
    "src/demux/demux.c",
};

/// Build the vendored libwebp (libraries/libwebp) as a self-contained static decode library. Replaces
/// the Homebrew `webp` system lib. Hand-rolled (no upstream build.zig) — libwebp uses root-relative
/// includes ("src/dsp/dsp.h"), so the compile include root is the libwebp root.
fn buildWebp(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "webp",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    lib.root_module.addIncludePath(b.path("libraries/libwebp"));
    lib.root_module.addCSourceFiles(.{
        .root = b.path("libraries/libwebp"),
        .files = &webp_decode_srcs,
        .flags = &.{"-fno-sanitize=undefined"},
    });
    return lib;
}

/// Build the vendored zaudio/miniaudio (libraries/zaudio) as a self-contained static library — the
/// audio backend, replacing the former SDL3 software-synth device layer. miniaudio talks to CoreAudio /
/// WASAPI / ALSA directly (runtime-linked where applicable), so the only link-time deps are the macOS
/// audio frameworks and libc/pthread on Linux. Hand-rolled (we don't use zaudio's own build.zig) so the
/// build stays Homebrew-free AND avoids zaudio's `system_sdk` remote dependency — we link the macOS
/// frameworks directly, mirroring `buildWebp`. The Zig binding (libraries/zaudio/src/zaudio.zig) is pure
/// `extern fn`, so it only needs link-time symbol resolution against this archive. Defines mirror
/// upstream zaudio's build.zig (drop WebAudio/null/JACK/DSound/WinMM backends; macOS uses CoreAudio, so
/// no runtime backend loading there).
fn buildMiniaudio(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "miniaudio",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    lib.root_module.addIncludePath(b.path("libraries/zaudio/libs/miniaudio"));
    lib.root_module.addCSourceFile(.{
        .file = b.path("libraries/zaudio/src/zaudio.c"),
        .flags = &.{ "-std=c99", "-fno-sanitize=undefined" },
    });
    lib.root_module.addCSourceFile(.{
        // On iOS the implementation must be compiled as Objective-C (miniaudio manages the
        // AVAudioSession there) — the .m wrapper includes the same miniaudio.c; a `-x` flag
        // can't do it because zig places per-file flags after the input file.
        .file = if (target.result.os.tag == .ios)
            b.path("libraries/zaudio/src/miniaudio_objc.m")
        else
            b.path("libraries/zaudio/libs/miniaudio/miniaudio.c"),
        .flags = &.{
            "-DMA_NO_WEBAUDIO",
            "-DMA_NO_NULL",
            "-DMA_NO_JACK",
            "-DMA_NO_DSOUND",
            "-DMA_NO_WINMM",
            "-fno-sanitize=undefined",
            if (target.result.os.tag.isDarwin()) "-DMA_NO_RUNTIME_LINKING" else "",
            // ObjC (iOS) compiles without -std=c99 (miniaudio detects ARC itself); C elsewhere.
            if (target.result.os.tag == .ios) "" else "-std=c99",
        },
    });
    switch (target.result.os.tag) {
        .macos => {
            lib.root_module.linkFramework("CoreAudio", .{});
            lib.root_module.linkFramework("CoreFoundation", .{});
            lib.root_module.linkFramework("AudioUnit", .{});
            lib.root_module.linkFramework("AudioToolbox", .{});
            addAppleSdkPaths(b, lib.root_module);
        },
        .ios => {
            lib.root_module.linkFramework("CoreAudio", .{});
            lib.root_module.linkFramework("CoreFoundation", .{});
            lib.root_module.linkFramework("AudioToolbox", .{});
            lib.root_module.linkFramework("AVFoundation", .{});
            lib.root_module.linkFramework("Foundation", .{});
            addAppleSdkPaths(b, lib.root_module);
        },
        .linux => {
            lib.root_module.linkSystemLibrary("pthread", .{});
            lib.root_module.linkSystemLibrary("m", .{});
            lib.root_module.linkSystemLibrary("dl", .{});
        },
        else => {}, // Windows: WASAPI is runtime-linked (LoadLibrary), nothing to link.
    }
    return lib;
}

/// Build the vendored flecs (libraries/zflecs/libs/flecs) as a self-contained static library.
/// flecs is compiled with FLECS_CUSTOM_BUILD + the addons needed for our C# ECS surface:
/// META (runtime component registration), SYSTEM+PIPELINE+MODULE (systems/progress/phases),
/// QUERY_DSL+PARSER (cached queries), OS_API_IMPL (default OS timer/threads), TIMER (pipeline needs it).
/// Mirrors buildMiniaudio/buildWebp — single C file, no Homebrew, no remote deps.
fn buildFlecs(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "flecs",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    lib.root_module.addIncludePath(b.path("libraries/zflecs/libs/flecs"));
    lib.root_module.addCSourceFile(.{
        .file = b.path("libraries/zflecs/libs/flecs/flecs.c"),
        .flags = &.{
            "-fno-sanitize=undefined",
            // Always use OS allocator (mirrors zflecs's forced behavior).
            "-DFLECS_USE_OS_ALLOC",
            // Custom build so only our listed addons compile in (no HTTP/REST/JSON dead weight).
            "-DFLECS_CUSTOM_BUILD",
            // Addons required for the zigote_ecs_* FFI surface:
            "-DFLECS_META", // ecs_component_init with runtime size/alignment
            "-DFLECS_SYSTEM", // ecs_system_init
            "-DFLECS_PIPELINE", // ecs_progress, built-in phase entities (EcsOnUpdate etc.)
            "-DFLECS_MODULE", // pipeline depends on module
            "-DFLECS_PARSER", // query DSL depends on parser
            "-DFLECS_QUERY_DSL", // cached queries
            "-DFLECS_OS_API_IMPL", // default OS timer/thread layer (needed by ecs_progress)
            "-DFLECS_TIMER", // timers used by pipeline scheduling
            "-DFLECS_STATS", // runtime statistics (lightweight, useful for diagnostics)
        },
    });
    if (target.result.os.tag == .windows) {
        // flecs OS API impl on Windows needs socket + debug helpers.
        lib.root_module.linkSystemLibrary("ws2_32", .{});
        lib.root_module.linkSystemLibrary("winmm", .{});
        lib.root_module.linkSystemLibrary("dbghelp", .{});
    }
    return lib;
}

/// Build the vendored meshoptimizer (libraries/zmesh/libs/meshoptimizer) as a self-contained static
/// library. Only the meshoptimizer C++ TUs are compiled — par_shapes/cgltf (the rest of zmesh) are not
/// referenced by the engine, so they're omitted (lean). Backs the `zmesh_opt` binding used at mesh
/// upload to reorder indices/vertices for GPU cache locality.
fn buildMeshoptimizer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const lib = b.addLibrary(.{
        .name = "meshoptimizer",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        }),
    });
    const dir = "libraries/zmesh/libs/meshoptimizer/";
    const srcs = [_][]const u8{
        "allocator.cpp",      "clusterizer.cpp",      "indexcodec.cpp",
        "indexgenerator.cpp", "overdrawanalyzer.cpp", "overdrawoptimizer.cpp",
        "simplifier.cpp",     "spatialorder.cpp",     "stripifier.cpp",
        "vcacheanalyzer.cpp", "vcacheoptimizer.cpp",  "vertexcodec.cpp",
        "vertexfilter.cpp",   "vfetchanalyzer.cpp",   "vfetchoptimizer.cpp",
    };
    inline for (srcs) |s| {
        lib.root_module.addCSourceFile(.{ .file = b.path(dir ++ s), .flags = &.{"-std=c++11"} });
    }
    return lib;
}

/// Every dependency static archive linked into the engine, collected so the `static-lib` step can
/// merge them into ONE self-contained libzigote.a. A static library does not link its dependencies
/// — it is just an archive of its own objects — so an app linking libzigote.a would otherwise face
/// thousands of undefined SDL/freetype/harfbuzz/… symbols. (wgpu-native needs no entry here: it is
/// attached with addObjectFile, which zig already archives into the output.)
const StaticDepArchive = struct { name: []const u8, path: std.Build.LazyPath };
var static_dep_archives: std.ArrayListUnmanaged(StaticDepArchive) = .empty;

/// linkLibrary + remember the archive for the static-lib merge.
fn linkAndCollect(b: *std.Build, mod: *std.Build.Module, lib: *std.Build.Step.Compile) void {
    mod.linkLibrary(lib);
    static_dep_archives.append(b.allocator, .{ .name = lib.name, .path = lib.getEmittedBin() }) catch @panic("OOM");
}

/// Remember a dependency's static archive by artifact name, chosen from its install list (rather
/// than `Dependency.artifact`, which panics when a name is registered more than once).
fn collectNamedStaticLib(b: *std.Build, dep: *std.Build.Dependency, name: []const u8) void {
    for (dep.builder.install_tls.step.dependencies.items) |step| {
        const install = step.cast(std.Build.Step.InstallArtifact) orelse continue;
        const lib = install.artifact;
        if (!std.mem.eql(u8, lib.name, name)) continue;
        if (lib.isDynamicLibrary()) continue;
        static_dep_archives.append(b.allocator, .{ .name = lib.name, .path = lib.getEmittedBin() }) catch @panic("OOM");
        return;
    }
    std.debug.panic("static-lib merge: no static artifact named '{s}' in dependency", .{name});
}

/// Wire the renderer/core native dependencies onto `mod`: freetype + harfbuzz (allyourcodebase, built
/// from source) and the pinned static wgpu-native. linkLibrary propagates freetype/harfbuzz headers for
/// freetype_text.zig's @cImport. Image decode (webp) and model import (assimp) are wired on the FFI
/// module instead — `src/ffi/root.zig`/`assimp_loader.zig` are their only consumers. Nothing Homebrew.
fn linkNativeDeps(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    // freetype + harfbuzz. `enable-libpng = true` matches the option harfbuzz passes to its own freetype
    // dependency, so both resolve to a single shared freetype artifact (no duplicate symbols). harfbuzz
    // is built with freetype support (default), providing the hb_ft_* used by freetype_text.zig.
    const freetype_dep = b.dependency("freetype", .{ .target = target, .optimize = optimize, .@"enable-libpng" = true });
    const harfbuzz_dep = b.dependency("harfbuzz", .{ .target = target, .optimize = optimize });
    linkAndCollect(b, mod, freetype_dep.artifact("freetype"));
    linkAndCollect(b, mod, harfbuzz_dep.artifact("harfbuzz"));
    // freetype's own dependencies: libpng (embedded-bitmap decode) and zlib. They are separate
    // archives — freetype.a does not absorb them — so a static consumer needs them listed too.
    // Reached through freetype's OWN builder with the same arguments its build.zig passes, so
    // these resolve to the very instances it linked (not fresh copies).
    const ft_zlib = freetype_dep.builder.dependency("zlib", .{ .target = target, .optimize = optimize });
    const ft_libpng = freetype_dep.builder.dependency("libpng", .{ .target = target, .optimize = optimize });
    static_dep_archives.append(b.allocator, .{ .name = "z", .path = ft_zlib.artifact("z").getEmittedBin() }) catch @panic("OOM");
    static_dep_archives.append(b.allocator, .{ .name = "png", .path = ft_libpng.artifact("png").getEmittedBin() }) catch @panic("OOM");

    // wgpu-native: pinned prebuilt static binary, selected per target.
    linkWgpuNative(b, mod, target);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── Build options ─────────────────────────────────────────────────────────
    // enable_3d gates the Assimp model importer — an editor-only subsystem (games load
    // pre-baked .zmesh; parsing lives ungated in engine/resources/zmesh_format.zig). Game
    // exports and lean 2D apps build with -Denable3d=false; the wgpu 3D renderer itself
    // stays in either way (render targets are allocated lazily, so a 2D app pays no
    // runtime cost for it). Default true = full engine, unchanged.
    const enable_3d = b.option(bool, "enable3d", "Build the Assimp model importer (editor-only). Game exports and lean 2D apps disable it.") orelse true;

    // enable_physics3d gates the Jolt 3D-physics FFI (src/ffi/physics.zig + the JoltC static lib).
    // enable_ecs gates the flecs ECS FFI (src/ffi/ecs.zig + the flecs static lib). Both default true
    // (full engine, unchanged) and are INDEPENDENT of enable_3d: game exports pass -Denable3d=false
    // yet still need Jolt (3D physics) and flecs (the runtime World entity store) at runtime. A pure
    // 2D/UI app — 2D physics is pure-C# Zigote.Physics2D, and a widget app never touches World/ECS —
    // can drop both (~several MB): `-Dphysics3d=false -Decs=false`.
    const enable_physics3d = b.option(bool, "physics3d", "Build the Jolt 3D physics FFI. Pure-2D/UI apps can disable it.") orelse true;
    const enable_ecs = b.option(bool, "ecs", "Build the flecs ECS FFI. Pure-UI apps that never use World/ECS can disable it.") orelse true;

    // Strip debug info from the shared library. ELF embeds DWARF for every statically-linked dep
    // (wgpu-native, Jolt, Assimp, SDL3, …) straight into the .so — ~90 MB of a 123 MB Linux build —
    // whereas Windows splits it into a discarded .pdb. Default: strip the speed-optimized release
    // modes (what game exports ship), keep symbols in Debug/ReleaseSafe dev builds. Override with
    // -Dstrip=false for a symbolized ReleaseFast (profiling) build.
    const strip = b.option(bool, "strip", "Strip debug info from the shared library (default: true for ReleaseFast/ReleaseSmall).") orelse
        (optimize == .ReleaseFast or optimize == .ReleaseSmall);

    const build_opts = b.addOptions();
    build_opts.addOption(bool, "enable_3d", enable_3d);
    build_opts.addOption(bool, "enable_physics3d", enable_physics3d);
    build_opts.addOption(bool, "enable_ecs", enable_ecs);

    const sdl3_dep = if (target.result.os.tag == .ios and b.sysroot != null) b.dependency("sdl3", .{
        .target = target,
        .optimize = optimize,
        // The binding's translate-c of the SDL headers needs the SDK's libc headers
        // (AvailabilityMacros.h & co) — same paths addAppleSdkPaths supplies to compiles.
        .sdl_system_include_path = @as(std.Build.LazyPath, .{ .cwd_relative = b.pathJoin(&.{ b.sysroot.?, "usr/include" }) }),
    }) else b.dependency("sdl3", .{
        .target = target,
        .optimize = optimize,
    });
    // The SDL3 C library itself (the binding module links it internally) — needed by the
    // static-lib merge just like the rest. Not via `artifact("SDL3")`: the binding registers the
    // name more than once and that helper panics on ambiguity, so walk its install list and take
    // the static archive.
    collectNamedStaticLib(b, sdl3_dep, "SDL3");
    const zigimg_dep = b.dependency("zigimg", .{
        .target = target,
        .optimize = optimize,
    });
    const wgpu_mod = b.addModule("wgpu", .{
        .root_source_file = b.path("libraries/wgpu/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // zaudio (miniaudio) binding module — the audio backend. Pure `extern fn`, so it needs no include
    // path; the symbols resolve against the buildMiniaudio static archive linked onto ffi_mod below.
    const zaudio_mod = b.addModule("zaudio", .{
        .root_source_file = b.path("libraries/zaudio/src/zaudio.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ── zig-gamedev SIMD math, handle pools, job system ─────────────────────────
    // All vendored under libraries/ (pure-Zig, Zig-0.16-native, no system deps).
    //   zmath — @Vector(4,f32) linear algebra backing the engine math hot paths
    //           (Mat4.mul / mulVec4 / inverse). (Frustum.intersectsSphere uses a plain 6-lane
    //           @Vector dot-and-compare in engine/math/frustum.zig, not zmath.)
    //   zpool — generational handle pools for the renderer's per-entity GPU caches.
    //   zjobs — persistent worker pool for the parallel texture loader.
    const zmath_opts = b.addOptions();
    zmath_opts.addOption(bool, "enable_cross_platform_determinism", true);
    const zmath_mod = b.addModule("zmath", .{
        .root_source_file = b.path("libraries/zmath/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zmath_options", .module = zmath_opts.createModule() },
        },
    });
    const zpool_mod = b.addModule("zpool", .{
        .root_source_file = b.path("libraries/zpool/src/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    // zmesh's meshoptimizer binding only (extern + inline wrappers; std-only imports). Points
    // straight at zmeshoptimizer.zig so par_shapes/cgltf aren't pulled in. Symbols resolve against
    // the buildMeshoptimizer static archive linked onto ffi_mod.
    const zmesh_opt_mod = b.addModule("zmesh_opt", .{
        .root_source_file = b.path("libraries/zmesh/src/zmeshoptimizer.zig"),
        .target = target,
        .optimize = optimize,
    });

    const core_mod = b.addModule("zigote_core", .{
        .root_source_file = b.path("src/core/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const ui_mod = b.addModule("zigote_ui", .{
        .root_source_file = b.path("src/ui/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigote_core", .module = core_mod },
        },
    });

    const engine_mod = b.addModule("zigote_engine", .{
        .root_source_file = b.path("src/engine/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigote_core", .module = core_mod },
            .{ .name = "zigote_ui", .module = ui_mod },
            .{ .name = "zigimg", .module = zigimg_dep.module("zigimg") },
            .{ .name = "zmath", .module = zmath_mod },
        },
    });

    const zigote_mod = b.addModule("zigote", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "sdl3", .module = sdl3_dep.module("sdl3") },
            .{ .name = "wgpu", .module = wgpu_mod },
            .{ .name = "zigimg", .module = zigimg_dep.module("zigimg") },
            .{ .name = "zigote_core", .module = core_mod },
            .{ .name = "zigote_ui", .module = ui_mod },
            .{ .name = "zigote_engine", .module = engine_mod },
            .{ .name = "zmath", .module = zmath_mod },
            .{ .name = "zpool", .module = zpool_mod },
            .{ .name = "zmesh_opt", .module = zmesh_opt_mod },
        },
    });

    zigote_mod.link_libc = true;
    // All native GPU/text/image deps — freetype, harfbuzz, webp, wgpu-native — from source or pinned
    // binaries, nothing from Homebrew. On `zigote_mod` so they resolve for both the FFI shared library
    // (which imports this module) and `mod_tests`. (assimp is wired on ffi_mod below, gated -Denable3d.)
    linkNativeDeps(b, zigote_mod, target, optimize);

    // Standalone Zig executables (showcase, devtools, game_example) have been
    // superseded by the C# editor (ZigoteCS/Zigote.Editor) and are no longer built.

    // ── C# FFI shared library ─────────────────────────────────────────────────
    const ffi_mod = b.createModule(.{
        .root_source_file = b.path("src/ffi/root.zig"),
        .target = target,
        .optimize = optimize,
        // Root-module strip applies at link time, so DWARF from the static archives is dropped too.
        .strip = strip,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "sdl3", .module = sdl3_dep.module("sdl3") },
            .{ .name = "wgpu", .module = wgpu_mod },
            .{ .name = "zaudio", .module = zaudio_mod },
            .{ .name = "zigimg", .module = zigimg_dep.module("zigimg") },
            .{ .name = "zigote", .module = zigote_mod },
            .{ .name = "zmath", .module = zmath_mod },
        },
    });
    ffi_mod.addOptions("build_options", build_opts);
    // miniaudio static archive — backs `src/ffi/audio.zig`'s zaudio.Device usage.
    linkAndCollect(b, ffi_mod, buildMiniaudio(b, target, optimize));
    // flecs static archive — backs `src/ffi/ecs.zig`'s @cImport("flecs.h") usage. Gated by -Decs:
    // when off, ecs.zig isn't compiled (see root.zig) so neither the header nor the archive is needed.
    if (enable_ecs) {
        linkAndCollect(b, ffi_mod, buildFlecs(b, target, optimize));
        ffi_mod.addIncludePath(b.path("libraries/zflecs/libs/flecs"));
    }
    // meshoptimizer static archive — backs `zmesh_opt`'s mesh-upload cache optimization.
    linkAndCollect(b, ffi_mod, buildMeshoptimizer(b, target, optimize));
    // Jolt (JoltC) 3D physics — backs `src/ffi/physics.zig`. Gated by -Dphysics3d: when off, root.zig
    // swaps in physics_stub.zig, so neither the JoltC header nor its archive is needed.
    if (enable_physics3d) {
        const zphysics_dep = b.dependency("zphysics", .{
            .target = target,
            .optimize = optimize,
            .enable_asserts = optimize == .Debug,
        });
        ffi_mod.addIncludePath(zphysics_dep.path("libs/JoltC"));
        linkAndCollect(b, ffi_mod, zphysics_dep.artifact("joltc"));
    }
    // freetype/harfbuzz + wgpu-native arrive (statically, from source) via `zigote_mod` which this
    // module imports — no Homebrew link here. webp + assimp are this module's own loaders:
    // webp decode (vendored libraries/libwebp, from source) — root.zig's image loader is its only
    // consumer; the src/ include resolves @cInclude("webp/decode.h").
    linkAndCollect(b, ffi_mod, buildWebp(b, target, optimize));
    ffi_mod.addIncludePath(b.path("libraries/libwebp/src"));
    // Assimp (Open Asset Import Library) — model import for every supported format. Built from source
    // (allyourcodebase/assimp), formats trimmed to what GltfLoader imports. linkLibrary propagates its
    // headers for assimp_loader.zig's @cImport. Gated by -Denable3d: lean 2D builds skip the C++ build.
    if (enable_3d) {
        const assimp_dep = b.dependency("zig_assimp", .{
            .target = target,
            .optimize = optimize,
            .formats = @as([]const u8, "glTF2,glTF,FBX,Obj,Collada,Ply,STL"),
        });
        linkAndCollect(b, ffi_mod, assimp_dep.artifact("assimp"));
    }
    if (target.result.os.tag == .macos) {
        // Native macOS menu bar (NSMenu) — Objective-C, linked against Cocoa.
        ffi_mod.addCSourceFile(.{
            .file = b.path("src/platform/macos_menu.m"),
            .flags = &.{"-fno-objc-arc"},
        });
        // Native macOS drag-out (NSDraggingSession) — Objective-C, linked against Cocoa.
        ffi_mod.addCSourceFile(.{
            .file = b.path("src/platform/macos_drag.m"),
            .flags = &.{"-fno-objc-arc"},
        });
        // Native macOS file dialogs (NSOpenPanel/NSSavePanel) — the full-featured backend
        // src/ffi/dialogs.zig dispatches to on macOS (UniformTypeIdentifiers arrives weak-linked
        // via SDL; UTType use is @available-guarded).
        ffi_mod.addCSourceFile(.{
            .file = b.path("src/platform/macos_file_dialog.m"),
            .flags = &.{"-fno-objc-arc"},
        });
        // macOS unified titlebar (full-size content view + native traffic lights) — backs
        // src/ffi/chrome.zig's mac_unified style.
        ffi_mod.addCSourceFile(.{
            .file = b.path("src/platform/macos_window_chrome.m"),
            .flags = &.{"-fno-objc-arc"},
        });
        ffi_mod.linkFramework("Cocoa", .{});
        addAppleSdkPaths(b, ffi_mod);
        // Metal/QuartzCore/Foundation arrive transitively via SDL + wgpu-native (wgpu renders
        // through Metal on macOS) — no explicit links needed now the native Metal backend is gone.
    }

    const shared_lib = b.addLibrary(.{
        .name = "zigote",
        .root_module = ffi_mod,
        .linkage = .dynamic,
        .version = .{ .major = 0, .minor = 1, .patch = 0 },
    });
    // Mach-O: leave room for install-name rewrites. The .NET iOS packager runs
    // install_name_tool on bundled dylibs and fails outright when the header has no padding.
    if (target.result.os.tag.isDarwin()) shared_lib.headerpad_max_install_names = true;
    const install_lib = b.addInstallArtifact(shared_lib, .{});

    const lib_step = b.step("shared-lib", "Build the shared library for C# FFI");
    lib_step.dependOn(&install_lib.step);

    // Static archive of the same FFI module. iOS forbids loading unsigned dylibs from the app
    // sandbox, so the engine is linked INTO the host binary there (C# side resolves DllImport
    // via "__Internal" under the ZIGOTE_STATIC_NATIVE define). Usable on any platform, but iOS
    // is the customer.
    const static_lib = b.addLibrary(.{
        .name = "zigote",
        .root_module = ffi_mod,
        .linkage = .static,
    });
    const install_static = b.addInstallArtifact(static_lib, .{});
    const static_step = b.step("static-lib", "Build the static library (iOS embeds the engine in the app binary)");
    static_step.dependOn(&install_static.step);
    // A static library archives only its OWN objects — linkLibrary dependencies are recorded, not
    // absorbed — so the host app must link the dependency archives too. Install each one next to
    // libzigote.a; the app build links every zig-out/lib/*.a it finds. (Merging them into a single
    // archive with libtool silently dropped members, so they stay separate and explicit.)
    for (static_dep_archives.items) |dep| {
        const install_dep = b.addInstallLibFile(dep.path, b.fmt("lib{s}.a", .{dep.name}));
        static_step.dependOn(&install_dep.step);
    }

    // On Windows, ship wgpu_native.dll alongside zigote.dll (we link its import lib, not the static
    // MSVC archive — see linkWgpuNative). Installed into zig-out/lib so the C# build copies it too.
    if (target.result.os.tag == .windows) {
        if (b.lazyDependency("wgpu_windows_x86_64", .{})) |wgpu_win| {
            const dll_install = b.addInstallBinFile(wgpu_win.path("lib/wgpu_native.dll"), "wgpu_native.dll");
            lib_step.dependOn(&dll_install.step);
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────────────
    const core_tests = b.addTest(.{ .root_module = core_mod });
    const run_core_tests = b.addRunArtifact(core_tests);

    const ui_tests = b.addTest(.{ .root_module = ui_mod });
    const run_ui_tests = b.addRunArtifact(ui_tests);

    const engine_tests = b.addTest(.{ .root_module = engine_mod });
    const run_engine_tests = b.addRunArtifact(engine_tests);

    const mod_tests = b.addTest(.{ .root_module = zigote_mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_core_tests.step);
    test_step.dependOn(&run_ui_tests.step);
    test_step.dependOn(&run_engine_tests.step);
    test_step.dependOn(&run_mod_tests.step);
}
