// WebGPU for the engine in the browser: the webgpu.h entry points libzigote imports (the set
// `llvm-nm -u libzigote.a | grep wgpu` lists), implemented over navigator.gpu. Linked by emcc as a
// --js-library together with the generated zigote_webgpu_layout.js, which supplies the wasm32
// struct offsets and enum tables reflected out of the Zig binding.
//
// The engine requests its adapter and device synchronously (wgpu-native style: request, then poll
// until the callback fired). A browser can only resolve them asynchronously, so the page requests
// both BEFORE the .NET runtime starts and leaves them on globalThis.zigoteGpu = { adapter, device };
// the request functions here answer from that, invoking the callback before they return.
//
// Handles are small integers into WGPU.objs; 0 stays null.
mergeInto(LibraryManager.library, {
  $WGPU__deps: ['$WGPU_LAYOUT', '$UTF8ToString', '$stringToUTF8', '$lengthBytesUTF8', '$getWasmTableEntry', 'malloc', 'free'],
  $WGPU: {
    objs: [null], // slot 0 is the null handle ([undefined] would serialize as [])
    freeIds: [],
    submitIndex: 0,
    emptySV: 0,
    formatNames: null, // enum value -> 'rgba8unorm'
    formatValues: null, // 'rgba8unorm' -> enum value
    add(o) {
      const id = WGPU.freeIds.length ? WGPU.freeIds.pop() : WGPU.objs.length;
      WGPU.objs[id] = o;
      return id;
    },
    get(id) { return id ? WGPU.objs[id] : undefined; },
    rel(id) {
      if (!id) return;
      WGPU.objs[id] = undefined;
      WGPU.freeIds.push(id);
    },

    u32(p) { return HEAPU32[p >> 2]; },
    i32(p) { return HEAP32[p >> 2]; },
    f32(p) { return HEAPF32[p >> 2]; },
    f64(p) { return HEAPF64[p >> 3]; },
    // A u64 as a Number (sizes, offsets, flags: all far below 2^53).
    u64(p) { return HEAPU32[p >> 2] + HEAPU32[(p + 4) >> 2] * 4294967296; },
    // u64 "undefined" sentinels (WGPU_WHOLE_SIZE) read as all-ones in both halves.
    u64opt(p) { return (HEAPU32[p >> 2] === 0xFFFFFFFF && HEAPU32[(p + 4) >> 2] === 0xFFFFFFFF) ? undefined : WGPU.u64(p); },
    u32opt(p) { const v = HEAPU32[p >> 2]; return v === 0xFFFFFFFF ? undefined : v; },
    setU64(p, v) { HEAPU32[p >> 2] = v % 4294967296; HEAPU32[(p + 4) >> 2] = Math.floor(v / 4294967296); },

    // StringView at p. length == SIZE_MAX means NUL-terminated.
    str(p) {
      const data = HEAPU32[p >> 2], len = HEAPU32[(p + 4) >> 2];
      if (!data) return undefined;
      return len === 0xFFFFFFFF ? UTF8ToString(data) : UTF8ToString(data, len);
    },
    // A heap StringView holding `s` (free with freeSV).
    newSV(s) {
      const n = lengthBytesUTF8(s);
      const sv = _malloc(8 + n + 1);
      stringToUTF8(s, sv + 8, n + 1);
      HEAPU32[sv >> 2] = sv + 8;
      HEAPU32[(sv + 4) >> 2] = n;
      return sv;
    },
    // Write a malloc'd string into the StringView field at p (AdapterInfo members).
    putSV(p, s) {
      s = s || '';
      const n = lengthBytesUTF8(s);
      const data = _malloc(n + 1);
      stringToUTF8(s, data, n + 1);
      HEAPU32[p >> 2] = data;
      HEAPU32[(p + 4) >> 2] = n;
    },
    empty() {
      if (!WGPU.emptySV) { WGPU.emptySV = _malloc(8); HEAPU32[WGPU.emptySV >> 2] = 0; HEAPU32[(WGPU.emptySV + 4) >> 2] = 0; }
      return WGPU.emptySV;
    },

    // Enum value -> WebGPU JS string; undefined for the "undefined"/"binding not used" values so
    // the JS default applies.
    tagFixups: { s_int: 'sint', u_int: 'uint', src_1: 'src1', one_minus_src_1: 'one-minus-src1', src_1_alpha: 'src1-alpha', one_minus_src_1_alpha: 'one-minus-src1-alpha' },
    en(type, v) {
      const tag = WGPU_LAYOUT.E[type][v];
      if (tag === undefined || tag === 'undefined' || tag === 'binding_not_used') return undefined;
      if (type === 'TextureFormat') return WGPU.fmt(v);
      return WGPU.tagFixups[tag] || tag.replace(/_/g, '-');
    },
    initFormats() {
      if (WGPU.formatNames) return;
      WGPU.formatNames = {};
      WGPU.formatValues = {};
      const depth = { depth16_unorm: 'depth16unorm', depth24_plus: 'depth24plus', depth24_plus_stencil8: 'depth24plus-stencil8', depth32_float: 'depth32float', depth32_float_stencil8: 'depth32float-stencil8', stencil8: 'stencil8' };
      for (const [v, tag] of Object.entries(WGPU_LAYOUT.E.TextureFormat)) {
        if (tag === 'undefined') continue;
        let name = depth[tag];
        if (!name) {
          const m = /^(r|rg|rgba|bgra|rgb10a2|rg11b10|rgb9e5)(\d*)_(unorm|snorm|uint|sint|float|ufloat)(_srgb)?$/.exec(tag);
          name = m ? m[1] + m[2] + m[3] + (m[4] ? '-srgb' : '') : tag.replace(/_/g, '-');
        }
        WGPU.formatNames[v] = name;
        WGPU.formatValues[name] = +v;
      }
    },
    fmt(v) { WGPU.initFormats(); return WGPU.formatNames[v]; },
    fmtValue(name) { WGPU.initFormats(); return WGPU.formatValues[name] || 0; },

    // Walk a chain looking for sType; returns the struct pointer or 0.
    chain(p, sType) {
      while (p) {
        if (HEAPU32[(p + WGPU_LAYOUT.S.ChainedStruct.s_type) >> 2] === sType) return p;
        p = HEAPU32[(p + WGPU_LAYOUT.S.ChainedStruct.next) >> 2];
      }
      return 0;
    },

    call(fnPtr, ...args) { if (fnPtr) getWasmTableEntry(fnPtr)(...args); },
    callWithMessage(fnPtr, before, message, after) {
      if (!fnPtr) return;
      const sv = message ? WGPU.newSV(message) : WGPU.empty();
      try { getWasmTableEntry(fnPtr)(...before, sv, ...after); } finally { if (message) _free(sv); }
    },

    constants(count, p) {
      if (!count) return undefined;
      const S = WGPU_LAYOUT.S.ConstantEntry, out = {};
      for (let i = 0; i < count; i++) {
        const e = p + i * S.__size;
        out[WGPU.str(e + S.key)] = WGPU.f64(e + S.value);
      }
      return out;
    },
    texelCopyTexture(p) {
      const S = WGPU_LAYOUT.S.TexelCopyTextureInfo, O = WGPU_LAYOUT.S.Origin3D;
      return {
        texture: WGPU.get(WGPU.u32(p + S.texture)),
        mipLevel: WGPU.u32(p + S.mip_level),
        origin: { x: WGPU.u32(p + S.origin + O.x), y: WGPU.u32(p + S.origin + O.y), z: WGPU.u32(p + S.origin + O.z) },
        aspect: WGPU.en('TextureAspect', WGPU.u32(p + S.aspect)),
      };
    },
    texelLayout(p, base) {
      const S = WGPU_LAYOUT.S.TexelCopyBufferLayout;
      return { offset: (base || 0) + WGPU.u64(p + S.offset), bytesPerRow: WGPU.u32opt(p + S.bytes_per_row), rowsPerImage: WGPU.u32opt(p + S.rows_per_image) };
    },
    extent(p) {
      const S = WGPU_LAYOUT.S.Extent3D;
      return { width: WGPU.u32(p + S.width), height: WGPU.u32(p + S.height), depthOrArrayLayers: WGPU.u32(p + S.depth_or_array_layers) };
    },
    stage(p, S) {
      return {
        module: WGPU.get(WGPU.u32(p + S.module)),
        entryPoint: WGPU.str(p + S.entry_point),
        constants: WGPU.constants(WGPU.u32(p + S.constant_count), WGPU.u32(p + S.constants)),
      };
    },
    blendComponent(p) {
      const S = WGPU_LAYOUT.S.BlendComponent;
      return {
        operation: WGPU.en('BlendOperation', WGPU.u32(p + S.operation)),
        srcFactor: WGPU.en('BlendFactor', WGPU.u32(p + S.src_factor)),
        dstFactor: WGPU.en('BlendFactor', WGPU.u32(p + S.dst_factor)),
      };
    },
    stencilFace(p) {
      const S = WGPU_LAYOUT.S.StencilFaceState;
      return {
        compare: WGPU.en('CompareFunction', WGPU.u32(p + S.compare)),
        failOp: WGPU.en('StencilOperation', WGPU.u32(p + S.fail_op)),
        depthFailOp: WGPU.en('StencilOperation', WGPU.u32(p + S.depth_fail_op)),
        passOp: WGPU.en('StencilOperation', WGPU.u32(p + S.pass_op)),
      };
    },
    gpu() {
      const g = globalThis.zigoteGpu;
      if (!g || !g.device) throw new Error('zigote: globalThis.zigoteGpu.{adapter,device} must be requested before the runtime starts');
      return g;
    },
  },

  // ── Logging / instance ─────────────────────────────────────────────────────
  wgpuSetLogCallback: (cb, ud) => {},
  wgpuSetLogLevel: (level) => {},
  wgpuCreateInstance__deps: ['$WGPU'],
  wgpuCreateInstance: (desc) => WGPU.add({ instance: true }),
  wgpuInstanceRelease: (id) => WGPU.rel(id),
  wgpuInstanceProcessEvents: (id) => {}, // callbacks fire as their promises settle

  wgpuInstanceCreateSurface: (inst, desc) => {
    const S = WGPU_LAYOUT.S;
    const src = WGPU.chain(WGPU.u32(desc + S.SurfaceDescriptor.next_in_chain), 0x00040000);
    const selector = src ? WGPU.str(src + S.EmscriptenSurfaceSourceCanvasHTMLSelector.selector) : '#canvas';
    const canvas = (typeof Module != 'undefined' && Module.canvas) || document.querySelector(selector);
    if (!canvas) { err(`zigote: no canvas for ${selector}`); return 0; }
    return WGPU.add({ canvas, ctx: canvas.getContext('webgpu'), viewFormat: undefined });
  },
  wgpuSurfaceRelease: (id) => WGPU.rel(id),

  wgpuInstanceEnumerateAdapters: (inst, opts, out) => {
    if (out) HEAPU32[out >> 2] = WGPU.add(WGPU.gpu().adapter);
    return 1;
  },
  wgpuInstanceRequestAdapter: (inst, opts, info) => {
    const S = WGPU_LAYOUT.S.RequestAdapterCallbackInfo;
    const g = globalThis.zigoteGpu;
    const ok = !!(g && g.adapter);
    WGPU.callWithMessage(WGPU.u32(info + S.callback), [ok ? 1 : 3, ok ? WGPU.add(g.adapter) : 0],
      ok ? '' : 'WebGPU adapter unavailable', [WGPU.u32(info + S.userdata1), WGPU.u32(info + S.userdata2)]);
    return 1n;
  },

  // ── Adapter / device ───────────────────────────────────────────────────────
  wgpuAdapterRelease: (id) => WGPU.rel(id),
  wgpuAdapterHasFeature: (id, feature) => {
    if (feature >= 0x00030000) return 0; // wgpu-native extensions do not exist in the browser
    const name = WGPU.en('FeatureName', feature);
    return name && WGPU.get(id).features.has(name) ? 1 : 0;
  },
  wgpuAdapterGetInfo: (id, out) => {
    const S = WGPU_LAYOUT.S.AdapterInfo;
    const info = WGPU.get(id).info || {};
    WGPU.putSV(out + S.vendor, info.vendor);
    WGPU.putSV(out + S.architecture, info.architecture);
    WGPU.putSV(out + S.device, info.device);
    WGPU.putSV(out + S.description, info.description);
    HEAPU32[(out + S.backend_type) >> 2] = 2; // webgpu
    HEAPU32[(out + S.adapter_type) >> 2] = 4; // unknown
    HEAPU32[(out + S.vendor_id) >> 2] = 0;
    HEAPU32[(out + S.device_id) >> 2] = 0;
    return 1;
  },
  wgpuAdapterInfoFreeMembers: (info) => {
    const S = WGPU_LAYOUT.S.AdapterInfo;
    for (const f of [S.vendor, S.architecture, S.device, S.description]) _free(HEAPU32[(info + f) >> 2]);
  },
  wgpuAdapterRequestDevice: (adapter, desc, info) => {
    const S = WGPU_LAYOUT.S;
    const device = WGPU.gpu().device;
    if (desc) {
      const ue = desc + S.DeviceDescriptor.uncaptured_error_callback_info;
      const cb = WGPU.u32(ue + S.UncapturedErrorCallbackInfo.callback);
      const ud1 = WGPU.u32(ue + S.UncapturedErrorCallbackInfo.userdata1);
      const ud2 = WGPU.u32(ue + S.UncapturedErrorCallbackInfo.userdata2);
      device.onuncapturederror = (e) => {
        console.error(`zigote: WebGPU ${e.error.constructor.name}: ${e.error.message}`);
        const type = e.error instanceof GPUValidationError ? 2 : e.error instanceof GPUOutOfMemoryError ? 3 : 5;
        WGPU.callWithMessage(cb, [0, type], e.error.message, [ud1, ud2]);
      };
    }
    const ci = S.RequestDeviceCallbackInfo;
    WGPU.callWithMessage(WGPU.u32(info + ci.callback), [1, WGPU.add(device)], '', [WGPU.u32(info + ci.userdata1), WGPU.u32(info + ci.userdata2)]);
    return 1n;
  },
  wgpuDeviceRelease: (id) => WGPU.rel(id),
  wgpuDeviceGetQueue: (id) => WGPU.add(WGPU.get(id).queue),
  wgpuDevicePoll: (id, wait, index) => 1,
  wgpuQueueRelease: (id) => WGPU.rel(id),

  // ── Surface ────────────────────────────────────────────────────────────────
  wgpuSurfaceGetCapabilities: (surface, adapter, out) => {
    const S = WGPU_LAYOUT.S.SurfaceCapabilities;
    const pref = navigator.gpu.getPreferredCanvasFormat();
    // The canvas itself only takes the non-sRGB format; the -srgb one is reached through
    // viewFormats (see wgpuSurfaceConfigure), which is what the engine picks.
    const formats = [WGPU.fmtValue(pref + '-srgb'), WGPU.fmtValue(pref)];
    const list = (vals) => { const p = _malloc(vals.length * 4); vals.forEach((v, i) => HEAPU32[(p >> 2) + i] = v); return p; };
    WGPU.setU64(out + S.usages, GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING);
    HEAPU32[(out + S.format_count) >> 2] = formats.length;
    HEAPU32[(out + S.formats) >> 2] = list(formats);
    HEAPU32[(out + S.present_mode_count) >> 2] = 1;
    HEAPU32[(out + S.present_modes) >> 2] = list([1]); // fifo
    // Opaque only: a tab has no desktop behind it, so a "transparent window" (CSD corners)
    // would just composite the frame over the page.
    HEAPU32[(out + S.alpha_mode_count) >> 2] = 1;
    HEAPU32[(out + S.alpha_modes) >> 2] = list([1]);
    return 1;
  },
  wgpuSurfaceCapabilitiesFreeMembers: (caps) => {
    const S = WGPU_LAYOUT.S.SurfaceCapabilities;
    for (const f of [S.formats, S.present_modes, S.alpha_modes]) _free(HEAPU32[(caps + f) >> 2]);
  },
  wgpuSurfaceConfigure: (id, config) => {
    const S = WGPU_LAYOUT.S.SurfaceConfiguration;
    const surf = WGPU.get(id);
    const format = WGPU.fmt(WGPU.u32(config + S.format));
    const base = format.replace(/-srgb$/, '');
    const width = WGPU.u32(config + S.width), height = WGPU.u32(config + S.height);
    if (surf.canvas.width !== width) surf.canvas.width = width;
    if (surf.canvas.height !== height) surf.canvas.height = height;
    surf.viewFormat = format;
    surf.ctx.configure({
      device: WGPU.get(WGPU.u32(config + S.device)),
      format: base,
      usage: WGPU.u64(config + S.usage),
      viewFormats: format !== base ? [format] : [],
      alphaMode: WGPU.u32(config + S.alpha_mode) === 2 ? 'premultiplied' : 'opaque',
    });
  },
  wgpuSurfaceUnconfigure: (id) => WGPU.get(id).ctx.unconfigure(),
  wgpuSurfaceGetCurrentTexture: (id, out) => {
    const S = WGPU_LAYOUT.S.SurfaceTexture;
    const surf = WGPU.get(id);
    const tex = surf.ctx.getCurrentTexture();
    tex.zgViewFormat = surf.viewFormat;
    HEAPU32[(out + S.texture) >> 2] = WGPU.add(tex);
    HEAPU32[(out + S.status) >> 2] = 1; // success_optimal
  },
  wgpuSurfacePresent: (id) => 1, // the browser presents when the task yields

  // ── Resources ──────────────────────────────────────────────────────────────
  wgpuDeviceCreateBuffer: (dev, desc) => {
    const S = WGPU_LAYOUT.S.BufferDescriptor;
    return WGPU.add(WGPU.get(dev).createBuffer({
      label: WGPU.str(desc + S.label),
      usage: WGPU.u64(desc + S.usage),
      size: WGPU.u64(desc + S.size),
      mappedAtCreation: !!WGPU.u32(desc + S.mapped_at_creation),
    }));
  },
  // ponytail: Release drops the handle only and the GC frees the GPU object; webgpu.h semantics
  // keep a released object alive while bind groups/views reference it, which destroy() would break.
  wgpuBufferRelease: (id) => WGPU.rel(id),
  wgpuBufferMapAsync: (id, mode, offset, size, info) => {
    const S = WGPU_LAYOUT.S.BufferMapCallbackInfo;
    const cb = WGPU.u32(info + S.callback), ud1 = WGPU.u32(info + S.userdata1), ud2 = WGPU.u32(info + S.userdata2);
    WGPU.get(id).mapAsync(Number(mode), offset, size === 0xFFFFFFFF ? undefined : size).then(
      () => WGPU.callWithMessage(cb, [1], '', [ud1, ud2]),
      (e) => WGPU.callWithMessage(cb, [3], String(e && e.message || e), [ud1, ud2]));
    return 1n;
  },
  wgpuBufferGetConstMappedRange: (id, offset, size) => {
    const buf = WGPU.get(id);
    const range = new Uint8Array(buf.getMappedRange(offset, size === 0xFFFFFFFF ? undefined : size));
    const p = _malloc(range.length);
    HEAPU8.set(range, p);
    (buf.zgCopies ||= []).push(p);
    return p;
  },
  wgpuBufferUnmap: (id) => {
    const buf = WGPU.get(id);
    (buf.zgCopies || []).forEach((p) => _free(p));
    buf.zgCopies = null;
    buf.unmap();
  },

  wgpuDeviceCreateTexture: (dev, desc) => {
    const S = WGPU_LAYOUT.S.TextureDescriptor;
    const viewFormats = [];
    for (let i = 0, n = WGPU.u32(desc + S.view_format_count), p = WGPU.u32(desc + S.view_formats); i < n; i++) viewFormats.push(WGPU.fmt(WGPU.u32(p + i * 4)));
    return WGPU.add(WGPU.get(dev).createTexture({
      label: WGPU.str(desc + S.label),
      usage: WGPU.u64(desc + S.usage),
      dimension: WGPU.en('TextureDimension', WGPU.u32(desc + S.dimension)),
      size: WGPU.extent(desc + S.size),
      format: WGPU.fmt(WGPU.u32(desc + S.format)),
      mipLevelCount: WGPU.u32(desc + S.mip_level_count),
      sampleCount: WGPU.u32(desc + S.sample_count),
      viewFormats,
    }));
  },
  wgpuTextureRelease: (id) => WGPU.rel(id),
  wgpuTextureGetWidth: (id) => WGPU.get(id).width,
  wgpuTextureGetHeight: (id) => WGPU.get(id).height,
  wgpuTextureGetFormat: (id) => { const t = WGPU.get(id); return WGPU.fmtValue(t.zgViewFormat || t.format); },
  wgpuTextureCreateView: (id, desc) => {
    const t = WGPU.get(id);
    if (!desc) return WGPU.add(t.createView(t.zgViewFormat ? { format: t.zgViewFormat } : undefined));
    const S = WGPU_LAYOUT.S.TextureViewDescriptor;
    return WGPU.add(t.createView({
      label: WGPU.str(desc + S.label),
      format: WGPU.en('TextureFormat', WGPU.u32(desc + S.format)) || t.zgViewFormat,
      dimension: WGPU.en('ViewDimension', WGPU.u32(desc + S.dimension)),
      baseMipLevel: WGPU.u32(desc + S.base_mip_level),
      mipLevelCount: WGPU.u32opt(desc + S.mip_level_count),
      baseArrayLayer: WGPU.u32(desc + S.base_array_layer),
      arrayLayerCount: WGPU.u32opt(desc + S.array_layer_count),
      aspect: WGPU.en('TextureAspect', WGPU.u32(desc + S.aspect)),
    }));
  },
  wgpuTextureViewRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreateSampler: (dev, desc) => {
    if (!desc) return WGPU.add(WGPU.get(dev).createSampler());
    const S = WGPU_LAYOUT.S.SamplerDescriptor;
    return WGPU.add(WGPU.get(dev).createSampler({
      label: WGPU.str(desc + S.label),
      addressModeU: WGPU.en('AddressMode', WGPU.u32(desc + S.address_mode_u)),
      addressModeV: WGPU.en('AddressMode', WGPU.u32(desc + S.address_mode_v)),
      addressModeW: WGPU.en('AddressMode', WGPU.u32(desc + S.address_mode_w)),
      magFilter: WGPU.en('FilterMode', WGPU.u32(desc + S.mag_filter)),
      minFilter: WGPU.en('FilterMode', WGPU.u32(desc + S.min_filter)),
      mipmapFilter: WGPU.en('MipmapFilterMode', WGPU.u32(desc + S.mipmap_filter)),
      lodMinClamp: WGPU.f32(desc + S.lod_min_clamp),
      lodMaxClamp: WGPU.f32(desc + S.lod_max_clamp),
      compare: WGPU.en('CompareFunction', WGPU.u32(desc + S.compare)),
      maxAnisotropy: HEAPU16[(desc + S.max_anisotropy) >> 1] || 1,
    }));
  },
  wgpuSamplerRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreateBindGroupLayout: (dev, desc) => {
    const L = WGPU_LAYOUT.S, S = L.BindGroupLayoutEntry;
    const entries = [];
    for (let i = 0, n = WGPU.u32(desc + L.BindGroupLayoutDescriptor.entry_count), base = WGPU.u32(desc + L.BindGroupLayoutDescriptor.entries); i < n; i++) {
      const e = base + i * S.__size;
      const entry = { binding: WGPU.u32(e + S.binding), visibility: WGPU.u64(e + S.visibility) };
      const b = e + S.buffer, s = e + S.sampler, t = e + S.texture, st = e + S.storage_texture;
      const bufType = WGPU.u32(b + L.BufferBindingLayout.type);
      if (bufType) entry.buffer = {
        type: WGPU.en('BufferBindingType', bufType),
        hasDynamicOffset: !!WGPU.u32(b + L.BufferBindingLayout.has_dynamic_offset),
        minBindingSize: WGPU.u64(b + L.BufferBindingLayout.min_binding_size),
      };
      const samplerType = WGPU.u32(s + L.SamplerBindingLayout.type);
      if (samplerType) entry.sampler = { type: WGPU.en('SamplerBindingType', samplerType) };
      const sampleType = WGPU.u32(t + L.TextureBindingLayout.sample_type);
      if (sampleType) entry.texture = {
        sampleType: WGPU.en('SampleType', sampleType),
        viewDimension: WGPU.en('ViewDimension', WGPU.u32(t + L.TextureBindingLayout.view_dimension)),
        multisampled: !!WGPU.u32(t + L.TextureBindingLayout.multisampled),
      };
      const access = WGPU.u32(st + L.StorageTextureBindingLayout.access);
      if (access) entry.storageTexture = {
        access: WGPU.en('StorageTextureAccess', access),
        format: WGPU.fmt(WGPU.u32(st + L.StorageTextureBindingLayout.format)),
        viewDimension: WGPU.en('ViewDimension', WGPU.u32(st + L.StorageTextureBindingLayout.view_dimension)),
      };
      entries.push(entry);
    }
    return WGPU.add(WGPU.get(dev).createBindGroupLayout({ label: WGPU.str(desc + L.BindGroupLayoutDescriptor.label), entries }));
  },
  wgpuBindGroupLayoutRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreateBindGroup: (dev, desc) => {
    const D = WGPU_LAYOUT.S.BindGroupDescriptor, S = WGPU_LAYOUT.S.BindGroupEntry;
    const entries = [];
    for (let i = 0, n = WGPU.u32(desc + D.entry_count), base = WGPU.u32(desc + D.entries); i < n; i++) {
      const e = base + i * S.__size;
      const buffer = WGPU.u32(e + S.buffer), sampler = WGPU.u32(e + S.sampler), view = WGPU.u32(e + S.texture_view);
      const resource = buffer ? { buffer: WGPU.get(buffer), offset: WGPU.u64(e + S.offset), size: WGPU.u64opt(e + S.size) }
        : sampler ? WGPU.get(sampler) : WGPU.get(view);
      entries.push({ binding: WGPU.u32(e + S.binding), resource });
    }
    return WGPU.add(WGPU.get(dev).createBindGroup({ label: WGPU.str(desc + D.label), layout: WGPU.get(WGPU.u32(desc + D.layout)), entries }));
  },
  wgpuBindGroupRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreatePipelineLayout: (dev, desc) => {
    const S = WGPU_LAYOUT.S.PipelineLayoutDescriptor;
    const layouts = [];
    for (let i = 0, n = WGPU.u32(desc + S.bind_group_layout_count), p = WGPU.u32(desc + S.bind_group_layouts); i < n; i++) layouts.push(WGPU.get(WGPU.u32(p + i * 4)));
    return WGPU.add(WGPU.get(dev).createPipelineLayout({ label: WGPU.str(desc + S.label), bindGroupLayouts: layouts }));
  },
  wgpuPipelineLayoutRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreateShaderModule: (dev, desc) => {
    const S = WGPU_LAYOUT.S;
    const wgsl = WGPU.chain(WGPU.u32(desc + S.ShaderModuleDescriptor.next_in_chain), 0x00000002);
    if (!wgsl) { err('zigote: only WGSL shader modules exist on the web'); return 0; }
    const label = WGPU.str(desc + S.ShaderModuleDescriptor.label);
    const module = WGPU.get(dev).createShaderModule({ label, code: WGPU.str(wgsl + S.ShaderSourceWGSL.code) });
    // Tint and naga disagree at the edges; surface the browser's verdict per module.
    module.getCompilationInfo().then((info) => info.messages.forEach((m) => {
      if (m.type === 'error') console.error(`zigote: WGSL ${label || ''} ${m.lineNum}:${m.linePos}: ${m.message}`);
    }));
    return WGPU.add(module);
  },
  wgpuShaderModuleRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreateRenderPipeline: (dev, desc) => {
    const L = WGPU_LAYOUT.S, D = L.RenderPipelineDescriptor;
    const v = desc + D.vertex, VS = L.VertexState;
    const buffers = [];
    for (let i = 0, n = WGPU.u32(v + VS.buffer_count), base = WGPU.u32(v + VS.buffers); i < n; i++) {
      const b = base + i * L.VertexBufferLayout.__size, B = L.VertexBufferLayout, A = L.VertexAttribute;
      const attributes = [];
      for (let j = 0, m = WGPU.u32(b + B.attribute_count), ab = WGPU.u32(b + B.attributes); j < m; j++) {
        const a = ab + j * A.__size;
        attributes.push({ format: WGPU.en('VertexFormat', WGPU.u32(a + A.format)), offset: WGPU.u64(a + A.offset), shaderLocation: WGPU.u32(a + A.shader_location) });
      }
      buffers.push({ arrayStride: WGPU.u64(b + B.array_stride), stepMode: WGPU.en('VertexStepMode', WGPU.u32(b + B.step_mode)), attributes });
    }
    const p = desc + D.primitive, P = L.PrimitiveState;
    const out = {
      label: WGPU.str(desc + D.label),
      layout: WGPU.get(WGPU.u32(desc + D.layout)) || 'auto',
      vertex: { ...WGPU.stage(v, VS), buffers },
      primitive: {
        topology: WGPU.en('PrimitiveTopology', WGPU.u32(p + P.topology)),
        stripIndexFormat: WGPU.en('IndexFormat', WGPU.u32(p + P.strip_index_format)),
        frontFace: WGPU.en('FrontFace', WGPU.u32(p + P.front_face)),
        cullMode: WGPU.en('CullMode', WGPU.u32(p + P.cull_mode)),
        unclippedDepth: !!WGPU.u32(p + P.unclipped_depth) || undefined,
      },
      multisample: {
        count: WGPU.u32(desc + D.multisample + L.MultisampleState.count) || 1,
        mask: WGPU.u32(desc + D.multisample + L.MultisampleState.mask),
        alphaToCoverageEnabled: !!WGPU.u32(desc + D.multisample + L.MultisampleState.alpha_to_coverage_enabled),
      },
    };
    const ds = WGPU.u32(desc + D.depth_stencil);
    if (ds) {
      const S = L.DepthStencilState, dw = WGPU.u32(ds + S.depth_write_enabled);
      out.depthStencil = {
        format: WGPU.fmt(WGPU.u32(ds + S.format)),
        depthWriteEnabled: dw === 2 ? undefined : !!dw,
        depthCompare: WGPU.en('CompareFunction', WGPU.u32(ds + S.depth_compare)),
        stencilFront: WGPU.stencilFace(ds + S.stencil_front),
        stencilBack: WGPU.stencilFace(ds + S.stencil_back),
        stencilReadMask: WGPU.u32(ds + S.stencil_read_mask),
        stencilWriteMask: WGPU.u32(ds + S.stencil_write_mask),
        depthBias: WGPU.i32(ds + S.depth_bias),
        depthBiasSlopeScale: WGPU.f32(ds + S.depth_bias_slope_scale),
        depthBiasClamp: WGPU.f32(ds + S.depth_bias_clamp),
      };
    }
    const fs = WGPU.u32(desc + D.fragment);
    if (fs) {
      const F = L.FragmentState, T = L.ColorTargetState, targets = [];
      for (let i = 0, n = WGPU.u32(fs + F.target_count), base = WGPU.u32(fs + F.targets); i < n; i++) {
        const t = base + i * T.__size, blend = WGPU.u32(t + T.blend);
        const format = WGPU.fmt(WGPU.u32(t + T.format));
        targets.push(format ? {
          format,
          blend: blend ? { color: WGPU.blendComponent(blend + L.BlendState.color), alpha: WGPU.blendComponent(blend + L.BlendState.alpha) } : undefined,
          writeMask: WGPU.u64(t + T.write_mask),
        } : null);
      }
      out.fragment = { ...WGPU.stage(fs, F), targets };
    }
    return WGPU.add(WGPU.get(dev).createRenderPipeline(out));
  },
  wgpuRenderPipelineRelease: (id) => WGPU.rel(id),

  wgpuDeviceCreateComputePipeline: (dev, desc) => {
    const D = WGPU_LAYOUT.S.ComputePipelineDescriptor;
    return WGPU.add(WGPU.get(dev).createComputePipeline({
      label: WGPU.str(desc + D.label),
      layout: WGPU.get(WGPU.u32(desc + D.layout)) || 'auto',
      compute: WGPU.stage(desc + D.compute, WGPU_LAYOUT.S.ProgrammableStageDescriptor),
    }));
  },
  wgpuComputePipelineRelease: (id) => WGPU.rel(id),

  // ── Commands ───────────────────────────────────────────────────────────────
  wgpuDeviceCreateCommandEncoder: (dev, desc) =>
    WGPU.add(WGPU.get(dev).createCommandEncoder(desc ? { label: WGPU.str(desc + WGPU_LAYOUT.S.CommandEncoderDescriptor.label) } : undefined)),
  wgpuCommandEncoderRelease: (id) => WGPU.rel(id),
  wgpuCommandEncoderFinish: (id, desc) => WGPU.add(WGPU.get(id).finish()),
  wgpuCommandBufferRelease: (id) => WGPU.rel(id),
  wgpuCommandEncoderBeginRenderPass: (id, desc) => {
    const L = WGPU_LAYOUT.S, D = L.RenderPassDescriptor, C = L.ColorAttachment, K = L.Color;
    const colorAttachments = [];
    for (let i = 0, n = WGPU.u32(desc + D.color_attachment_count), base = WGPU.u32(desc + D.color_attachments); i < n; i++) {
      const a = base + i * C.__size, view = WGPU.u32(a + C.view);
      if (!view) { colorAttachments.push(null); continue; }
      const c = a + C.clear_value;
      colorAttachments.push({
        view: WGPU.get(view),
        depthSlice: WGPU.u32opt(a + C.depth_slice),
        resolveTarget: WGPU.get(WGPU.u32(a + C.resolve_target)),
        loadOp: WGPU.en('LoadOp', WGPU.u32(a + C.load_op)),
        storeOp: WGPU.en('StoreOp', WGPU.u32(a + C.store_op)),
        clearValue: { r: WGPU.f64(c + K.r), g: WGPU.f64(c + K.g), b: WGPU.f64(c + K.b), a: WGPU.f64(c + K.a) },
      });
    }
    const pass = { label: WGPU.str(desc + D.label), colorAttachments };
    const ds = WGPU.u32(desc + D.depth_stencil_attachment);
    if (ds) {
      const S = L.DepthStencilAttachment;
      const depthLoadOp = WGPU.en('LoadOp', WGPU.u32(ds + S.depth_load_op));
      const stencilLoadOp = WGPU.en('LoadOp', WGPU.u32(ds + S.stencil_load_op));
      // Ops for an aspect the format lacks are a validation error in the browser, and the engine
      // leaves them undefined there — so undefined ops are simply omitted.
      pass.depthStencilAttachment = {
        view: WGPU.get(WGPU.u32(ds + S.view)),
        depthLoadOp, depthStoreOp: WGPU.en('StoreOp', WGPU.u32(ds + S.depth_store_op)),
        depthClearValue: depthLoadOp === 'clear' ? WGPU.f32(ds + S.depth_clear_value) : undefined,
        depthReadOnly: !!WGPU.u32(ds + S.depth_read_only),
        stencilLoadOp, stencilStoreOp: WGPU.en('StoreOp', WGPU.u32(ds + S.stencil_store_op)),
        stencilClearValue: WGPU.u32(ds + S.stencil_clear_value),
        stencilReadOnly: !!WGPU.u32(ds + S.stencil_read_only),
      };
    }
    return WGPU.add(WGPU.get(id).beginRenderPass(pass));
  },
  wgpuCommandEncoderBeginComputePass: (id, desc) => WGPU.add(WGPU.get(id).beginComputePass()),
  wgpuCommandEncoderCopyTextureToBuffer: (id, src, dst, size) => {
    const B = WGPU_LAYOUT.S.TexelCopyBufferInfo;
    WGPU.get(id).copyTextureToBuffer(WGPU.texelCopyTexture(src),
      { buffer: WGPU.get(WGPU.u32(dst + B.buffer)), ...WGPU.texelLayout(dst + B.layout) }, WGPU.extent(size));
  },
  wgpuCommandEncoderCopyTextureToTexture: (id, src, dst, size) =>
    WGPU.get(id).copyTextureToTexture(WGPU.texelCopyTexture(src), WGPU.texelCopyTexture(dst), WGPU.extent(size)),

  wgpuRenderPassEncoderSetViewport: (id, x, y, w, h, min, max) => WGPU.get(id).setViewport(x, y, w, h, min, max),
  wgpuRenderPassEncoderSetScissorRect: (id, x, y, w, h) => WGPU.get(id).setScissorRect(x, y, w, h),
  wgpuRenderPassEncoderSetPipeline: (id, p) => WGPU.get(id).setPipeline(WGPU.get(p)),
  wgpuRenderPassEncoderSetBindGroup: (id, index, group, count, offsets) =>
    WGPU.get(id).setBindGroup(index, WGPU.get(group) || null, count ? HEAPU32.subarray(offsets >> 2, (offsets >> 2) + count) : []),
  wgpuRenderPassEncoderSetVertexBuffer: (id, slot, buf, offset, size) =>
    WGPU.get(id).setVertexBuffer(slot, WGPU.get(buf) || null, Number(offset), size === 0xFFFFFFFFFFFFFFFFn ? undefined : Number(size)),
  wgpuRenderPassEncoderSetIndexBuffer: (id, buf, format, offset, size) =>
    WGPU.get(id).setIndexBuffer(WGPU.get(buf), WGPU.en('IndexFormat', format), Number(offset), size === 0xFFFFFFFFFFFFFFFFn ? undefined : Number(size)),
  wgpuRenderPassEncoderDraw: (id, vc, ic, fv, fi) => WGPU.get(id).draw(vc, ic, fv, fi),
  wgpuRenderPassEncoderDrawIndexed: (id, ic, inst, first, base, fi) => WGPU.get(id).drawIndexed(ic, inst, first, base, fi),
  wgpuRenderPassEncoderEnd: (id) => WGPU.get(id).end(),
  wgpuRenderPassEncoderRelease: (id) => WGPU.rel(id),

  wgpuComputePassEncoderSetPipeline: (id, p) => WGPU.get(id).setPipeline(WGPU.get(p)),
  wgpuComputePassEncoderSetBindGroup: (id, index, group, count, offsets) =>
    WGPU.get(id).setBindGroup(index, WGPU.get(group) || null, count ? HEAPU32.subarray(offsets >> 2, (offsets >> 2) + count) : []),
  wgpuComputePassEncoderDispatchWorkgroups: (id, x, y, z) => WGPU.get(id).dispatchWorkgroups(x, y, z),
  wgpuComputePassEncoderEnd: (id) => WGPU.get(id).end(),
  wgpuComputePassEncoderRelease: (id) => WGPU.rel(id),

  // ── Queue ──────────────────────────────────────────────────────────────────
  wgpuQueueSubmit: (id, count, cmds) => {
    const list = [];
    for (let i = 0; i < count; i++) list.push(WGPU.get(HEAPU32[(cmds >> 2) + i]));
    WGPU.get(id).submit(list);
  },
  wgpuQueueSubmitForIndex: (id, count, cmds) => {
    _wgpuQueueSubmit(id, count, cmds);
    return BigInt(++WGPU.submitIndex);
  },
  wgpuQueueSubmitForIndex__deps: ['wgpuQueueSubmit'],
  wgpuQueueWriteBuffer: (id, buf, offset, data, size) =>
    WGPU.get(id).writeBuffer(WGPU.get(buf), Number(offset), HEAPU8, data, size),
  wgpuQueueWriteTexture: (id, dst, data, dataSize, layout, size) =>
    WGPU.get(id).writeTexture(WGPU.texelCopyTexture(dst), HEAPU8.subarray(data, data + dataSize), WGPU.texelLayout(layout), WGPU.extent(size)),
});

// Every entry point needs the shared state.
for (const name of Object.keys(LibraryManager.library)) {
  if (name.startsWith('wgpu') && !name.includes('__')) {
    const deps = LibraryManager.library[name + '__deps'] || [];
    if (!deps.includes('$WGPU')) LibraryManager.library[name + '__deps'] = deps.concat('$WGPU');
  }
}
