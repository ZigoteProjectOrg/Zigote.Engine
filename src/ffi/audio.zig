const std = @import("std");
const zaudio = @import("zaudio");

// Procedural data sources synthesise at this internal rate; the engine resamples to the device rate, so
// the value only affects oscillator stepping precision, never pitch (frequency is given in Hz).
const synth_rate: u32 = 48000;

// Round-robin pool of fire-and-forget one-shots. Plenty for UI + gameplay pings; the oldest is reaped
// when the pool wraps. Matches the spirit of the old transient voice bank.
const max_oneshots: usize = 32;

// Back-compat sustained channels for `zigote_audio_voice` (held tones whose pitch/volume change live).
const voice_channels: usize = 16;

pub const Waveform = enum(u8) {
    sine = 0,
    square = 1,
    triangle = 2,
    sawtooth = 3,
    noise = 4,

    pub fn fromU8(v: u8) Waveform {
        return switch (v) {
            0 => .sine,
            1 => .square,
            2 => .triangle,
            3 => .sawtooth,
            else => .noise,
        };
    }
};

/// Map our waveform code to miniaudio's `ma_waveform` type. `.noise` is handled by a separate `Noise`
/// data source, so it never reaches here (callers branch first); fall back to sine if it ever does.
fn maWaveType(w: Waveform) zaudio.Waveform.Type {
    return switch (w) {
        .sine => .sine,
        .square => .square,
        .triangle => .triangle,
        .sawtooth => .sawtooth,
        .noise => .sine,
    };
}

/// A live `Sound` plus the procedural data source backing it (null for file sounds). The data source
/// must outlive the sound and is destroyed *after* it.
const Instance = struct {
    sound: *zaudio.Sound,
    wave: ?*zaudio.Waveform = null,
    noise: ?*zaudio.Noise = null,
    wave_code: Waveform = .sine,
};

const OneShot = struct {
    active: bool = false,
    sound: ?*zaudio.Sound = null,
    wave: ?*zaudio.Waveform = null,
    noise: ?*zaudio.Noise = null,
    remaining: f32 = 0, // seconds until reaped
};

// zaudio's allocator + allocation tracking is process-global and "init once"; the engine handle can be
// recreated within one process (project switching / tests), so guard `zaudio.init` and never `deinit`
// it (mirrors the original device-layer note).
var za_inited: bool = false;

pub const AudioState = struct {
    allocator: std.mem.Allocator,
    engine: *zaudio.Engine,

    oneshots: [max_oneshots]OneShot = [_]OneShot{.{}} ** max_oneshots,
    next_oneshot: usize = 0,

    instances: std.AutoHashMap(u32, Instance),
    next_id: u32 = 1,

    // Mixer buses (miniaudio sound groups). Group-lifetime = audio-state lifetime: miniaudio must not
    // destroy a group node while sounds route through it, and "a handful of buses created once" is the
    // model — so there is no per-group destroy, only teardown in deinit.
    groups: std.AutoHashMap(u32, *zaudio.SoundGroup),
    next_group_id: u32 = 1,

    // Back-compat sustained channels → handle ids in `instances` (0 = empty slot).
    voices: [voice_channels]u32 = [_]u32{0} ** voice_channels,
};

// ── lifecycle ───────────────────────────────────────────────────────────────────

/// Open the engine on the default device using its native channel layout (channels = 0 → surround when
/// the device offers it). Returns null (sound disabled) on any failure — never fatal.
pub fn init(allocator: std.mem.Allocator) ?*AudioState {
    if (!za_inited) {
        zaudio.init(allocator);
        za_inited = true;
    }

    const self = allocator.create(AudioState) catch return null;
    self.* = .{
        .allocator = allocator,
        .engine = undefined,
        .instances = std.AutoHashMap(u32, Instance).init(allocator),
        .groups = std.AutoHashMap(u32, *zaudio.SoundGroup).init(allocator),
    };

    var config = zaudio.Engine.Config.init();
    config.channels = 0; // native device layout (stereo / 5.1 / 7.1) → surround spatialisation
    config.listener_count = 1;
    config.sample_rate = 0; // native device rate

    const engine = zaudio.Engine.create(config) catch {
        std.log.warn("zigote: failed to create audio engine; sound disabled", .{});
        self.instances.deinit();
        self.groups.deinit();
        allocator.destroy(self);
        return null;
    };
    self.engine = engine;

    engine.start() catch {
        std.log.warn("zigote: failed to start audio engine; sound disabled", .{});
        engine.destroy();
        self.instances.deinit();
        self.groups.deinit();
        allocator.destroy(self);
        return null;
    };

    std.log.info("zigote: audio ready (miniaudio engine, spatial/surround)", .{});
    return self;
}

pub fn deinit(self: *AudioState) void {
    // Stop + free every live source before tearing down the engine.
    var it = self.instances.valueIterator();
    while (it.next()) |inst| destroyInstance(inst.*);
    self.instances.deinit();

    for (&self.oneshots) |*os| {
        if (os.active) destroyOneShot(os);
    }

    // Groups after sounds (nothing routes through them any more), before the engine that owns them.
    var git = self.groups.valueIterator();
    while (git.next()) |g| g.*.destroy();
    self.groups.deinit();

    self.engine.destroy();
    // Intentionally not calling zaudio.deinit (see `za_inited`).
    self.allocator.destroy(self);
}

fn destroyInstance(inst: Instance) void {
    inst.sound.destroy();
    if (inst.wave) |w| w.destroy();
    if (inst.noise) |n| n.destroy();
}

fn destroyOneShot(os: *OneShot) void {
    if (os.sound) |s| s.destroy();
    if (os.wave) |w| w.destroy();
    if (os.noise) |n| n.destroy();
    os.* = .{};
}

/// Age + reap fire-and-forget one-shots. Called once per frame by the host (PlaySession / App.Frame).
pub fn update(self: *AudioState, dt: f32) void {
    for (&self.oneshots) |*os| {
        if (!os.active) continue;
        os.remaining -= dt;
        if (os.remaining <= 0) destroyOneShot(os);
    }
}

// ── procedural source construction ────────────────────────────────────────────

/// Build a `Sound` over a fresh procedural data source. `spatial` toggles 3D positioning. Returns the
/// pieces so the caller decides where to store them (one-shot pool vs. handle table).
fn makeProcedural(self: *AudioState, freq: f32, wave: Waveform, spatial: bool) ?Instance {
    if (wave == .noise) {
        const cfg = zaudio.Noise.Config.init(.float32, 1, .white, 0, 0.5);
        const n = zaudio.Noise.create(cfg) catch return null;
        const snd = self.engine.createSoundFromDataSource(n.asDataSourceMut(), .{}, null) catch {
            n.destroy();
            return null;
        };
        snd.setSpatializationEnabled(spatial);
        return .{ .sound = snd, .noise = n, .wave_code = wave };
    }

    const cfg = zaudio.Waveform.Config.init(.float32, 1, synth_rate, maWaveType(wave), 0.5, @floatCast(freq));
    const w = zaudio.Waveform.create(cfg) catch return null;
    const snd = self.engine.createSoundFromDataSource(w.asDataSourceMut(), .{}, null) catch {
        w.destroy();
        return null;
    };
    snd.setSpatializationEnabled(spatial);
    return .{ .sound = snd, .wave = w, .wave_code = wave };
}

fn pushOneShot(self: *AudioState, inst: Instance, duration: f32) void {
    const slot = &self.oneshots[self.next_oneshot];
    self.next_oneshot = (self.next_oneshot + 1) % max_oneshots;
    if (slot.active) destroyOneShot(slot);
    slot.* = .{
        .active = true,
        .sound = inst.sound,
        .wave = inst.wave,
        .noise = inst.noise,
        .remaining = @max(0.01, duration),
    };
    inst.sound.start() catch {};
}

// ── listener + master ─────────────────────────────────────────────────────────

pub fn setListener(self: *AudioState, px: f32, py: f32, pz: f32, fx: f32, fy: f32, fz: f32, ux: f32, uy: f32, uz: f32) void {
    self.engine.setListenerPosition(0, .{ px, py, pz });
    self.engine.setListenerDirection(0, .{ fx, fy, fz });
    self.engine.setListenerWorldUp(0, .{ ux, uy, uz });
}

pub fn setMasterVolume(self: *AudioState, volume: f32) void {
    self.engine.setVolume(std.math.clamp(volume, 0.0, 4.0)) catch {};
}

// ── fire-and-forget one-shots ─────────────────────────────────────────────────

/// 2D UI one-shot (no spatialisation): the old `zigote_audio_beep`.
pub fn beep(self: *AudioState, freq: f32, duration_s: f32, volume: f32, wave: Waveform) void {
    if (freq <= 0 and wave != .noise) return;
    const inst = makeProcedural(self, freq, wave, false) orelse return;
    inst.sound.setVolume(std.math.clamp(volume, 0.0, 1.0));
    pushOneShot(self, inst, duration_s);
}

/// Positioned procedural one-shot (a ping at a world point), spatialised + attenuated.
pub fn beep3d(self: *AudioState, px: f32, py: f32, pz: f32, freq: f32, duration_s: f32, volume: f32, wave: Waveform, min_dist: f32, max_dist: f32, rolloff: f32) void {
    if (freq <= 0 and wave != .noise) return;
    const inst = makeProcedural(self, freq, wave, true) orelse return;
    const s = inst.sound;
    s.setVolume(std.math.clamp(volume, 0.0, 1.0));
    s.setPosition(.{ px, py, pz });
    if (min_dist > 0) s.setMinDistance(min_dist);
    if (max_dist > 0) s.setMaxDistance(max_dist);
    if (rolloff > 0) s.setRolloff(rolloff);
    pushOneShot(self, inst, duration_s);
}

// ── back-compat sustained channels ────────────────────────────────────────────

/// `zigote_audio_voice`: a held 2D tone on a channel. volume<=0 or freq<=0 silences (frees) it.
pub fn setVoice(self: *AudioState, channel: usize, freq: f32, volume: f32, wave: Waveform) void {
    const ch = channel % voice_channels;
    const id = self.voices[ch];

    const silence = volume <= 0 or (freq <= 0 and wave != .noise);
    if (silence) {
        if (id != 0) {
            destroyHandle(self, id);
            self.voices[ch] = 0;
        }
        return;
    }

    if (id == 0) {
        const new_id = createTone(self, freq, wave) orelse return;
        self.voices[ch] = new_id;
        setVolume(self, new_id, volume);
        play(self, new_id);
        return;
    }

    // Update the live channel. A noise↔tone switch needs a rebuild; otherwise just retune.
    const inst = self.instances.getPtr(id) orelse return;
    if (inst.wave_code != wave and (inst.wave_code == .noise or wave == .noise)) {
        destroyHandle(self, id);
        self.voices[ch] = 0;
        const new_id = createTone(self, freq, wave) orelse return;
        self.voices[ch] = new_id;
        setVolume(self, new_id, volume);
        play(self, new_id);
        return;
    }
    if (inst.wave) |w| {
        w.setFrequency(@floatCast(freq)) catch {};
        w.setType(maWaveType(wave)) catch {};
        inst.wave_code = wave;
    }
    inst.sound.setVolume(std.math.clamp(volume, 0.0, 1.0));
}

/// Silence + free everything (one-shots, sustained channels, and all handle sources).
pub fn stopAll(self: *AudioState) void {
    for (&self.oneshots) |*os| {
        if (os.active) destroyOneShot(os);
    }
    var it = self.instances.valueIterator();
    while (it.next()) |inst| destroyInstance(inst.*);
    self.instances.clearRetainingCapacity();
    self.voices = [_]u32{0} ** voice_channels;
}

// ── addressable source handles (editor AudioSource nodes + scripting Audio API) ──

fn register(self: *AudioState, inst: Instance) ?u32 {
    const id = self.next_id;
    self.next_id +%= 1;
    if (self.next_id == 0) self.next_id = 1;
    self.instances.put(id, inst) catch {
        destroyInstance(inst);
        return null;
    };
    return id;
}

/// Create a sustained procedural-tone source (not started). 0 on failure.
pub fn createTone(self: *AudioState, freq: f32, wave: Waveform) ?u32 {
    const inst = makeProcedural(self, freq, wave, false) orelse return null;
    return register(self, inst);
}

/// Create a source from a decoded/streamed audio file (not started). 0 on failure.
pub fn createFile(self: *AudioState, path: [:0]const u8, streaming: bool) ?u32 {
    const snd = self.engine.createSoundFromFile(path, .{ .flags = .{ .stream = streaming } }) catch {
        std.log.warn("zigote: failed to load audio file '{s}'", .{path});
        return null;
    };
    return register(self, .{ .sound = snd });
}

fn get(self: *AudioState, id: u32) ?*Instance {
    if (id == 0) return null;
    return self.instances.getPtr(id);
}

pub fn play(self: *AudioState, id: u32) void {
    const inst = get(self, id) orelse return;
    inst.sound.start() catch {};
}

pub fn stop(self: *AudioState, id: u32) void {
    const inst = get(self, id) orelse return;
    inst.sound.stop() catch {};
    inst.sound.seekToPcmFrame(0) catch {};
}

pub fn destroyHandle(self: *AudioState, id: u32) void {
    if (id == 0) return;
    if (self.instances.fetchRemove(id)) |kv| destroyInstance(kv.value);
}

pub fn setVolume(self: *AudioState, id: u32, volume: f32) void {
    const inst = get(self, id) orelse return;
    inst.sound.setVolume(std.math.clamp(volume, 0.0, 4.0));
}

pub fn setPitch(self: *AudioState, id: u32, pitch: f32) void {
    const inst = get(self, id) orelse return;
    inst.sound.setPitch(std.math.clamp(pitch, 0.01, 8.0));
}

pub fn setLooping(self: *AudioState, id: u32, looping: bool) void {
    const inst = get(self, id) orelse return;
    inst.sound.setLooping(looping);
}

pub fn setSpatial(self: *AudioState, id: u32, enabled: bool) void {
    const inst = get(self, id) orelse return;
    inst.sound.setSpatializationEnabled(enabled);
}

pub fn setPosition(self: *AudioState, id: u32, x: f32, y: f32, z: f32) void {
    const inst = get(self, id) orelse return;
    inst.sound.setPosition(.{ x, y, z });
}

pub fn setVelocity(self: *AudioState, id: u32, x: f32, y: f32, z: f32) void {
    const inst = get(self, id) orelse return;
    inst.sound.setVelocity(.{ x, y, z });
}

pub fn setAttenuation(self: *AudioState, id: u32, min_dist: f32, max_dist: f32, rolloff: f32) void {
    const inst = get(self, id) orelse return;
    if (min_dist > 0) inst.sound.setMinDistance(min_dist);
    if (max_dist > 0) inst.sound.setMaxDistance(max_dist);
    if (rolloff >= 0) inst.sound.setRolloff(rolloff);
}

pub fn isPlaying(self: *AudioState, id: u32) bool {
    const inst = get(self, id) orelse return false;
    return inst.sound.isPlaying();
}

// ── mixer buses (sound groups) ──────────────────────────────────────────────────

// zaudio does not bind the node-graph endpoint getter; declare the miniaudio symbol directly so a
// sound can be re-routed back to the master output (group 0).
extern fn ma_node_graph_get_endpoint(graph: *zaudio.NodeGraph) *zaudio.Node;

/// Create a mixer bus. 0 on failure. Buses live until the audio state is torn down.
pub fn groupCreate(self: *AudioState) ?u32 {
    const group = self.engine.createSoundGroup(.{}, null) catch return null;
    const id = self.next_group_id;
    self.next_group_id +%= 1;
    if (self.next_group_id == 0) self.next_group_id = 1;
    self.groups.put(id, group) catch {
        group.destroy();
        return null;
    };
    return id;
}

pub fn groupSetVolume(self: *AudioState, id: u32, volume: f32) void {
    const g = self.groups.get(id) orelse return;
    g.setVolume(@max(0.0, volume));
}

pub fn groupSetPitch(self: *AudioState, id: u32, pitch: f32) void {
    const g = self.groups.get(id) orelse return;
    g.setPitch(@max(0.01, pitch));
}

/// Route a sound's output through a bus (group 0 = back to the engine's master endpoint).
pub fn soundSetGroup(self: *AudioState, sound_id: u32, group_id: u32) void {
    const inst = get(self, sound_id) orelse return;
    if (group_id == 0) {
        const endpoint = ma_node_graph_get_endpoint(self.engine.asNodeGraphMut());
        inst.sound.asNodeMut().attachOutputBus(0, endpoint, 0) catch {};
        return;
    }
    const g = self.groups.get(group_id) orelse return;
    inst.sound.asNodeMut().attachOutputBus(0, g.asNodeMut(), 0) catch {};
}
