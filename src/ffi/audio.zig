const std = @import("std");
const zaudio = @import("zaudio");
const netstream = @import("netstream.zig");

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
    /// Push source backing this sound (see `createStream`), or null for a file/procedural one.
    /// Like the procedural sources above it must outlive the sound and is freed after it.
    stream: ?*netstream.NetStream = null,
    // Equalizer chain this sound routes through (0 = straight to the master endpoint). Kept on the
    // instance so the chain can re-attach its sounds after its head node is replaced.
    eq_id: u32 = 0,
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

    // Equalizer chains (see the equalizer section). Like buses these are few and long-lived, but
    // unlike buses they *are* destroyable — a music player tears one down when its EQ is disabled.
    equalizers: std.AutoHashMap(u32, Equalizer),
    next_eq_id: u32 = 1,
};

// ── lifecycle ───────────────────────────────────────────────────────────────────

/// Open the engine on the default device using its native channel layout (channels = 0 → surround when
/// the device offers it) and its native rate. Returns null (sound disabled) on any failure — never fatal.
pub fn init(allocator: std.mem.Allocator) ?*AudioState {
    return initWithRate(allocator, 0);
}

/// As `init`, but asks the device for a specific sample rate (0 = whatever the device prefers).
///
/// This is what high-resolution playback needs: miniaudio resamples every source to the device rate,
/// so a 192 kHz file played through a device opened at 48 kHz is downsampled no matter how good the
/// file is. Opening the device at the source's rate is the only way to avoid that. The rate is a
/// device-creation property, hence a whole new engine rather than a setter — see `reopen`.
pub fn initWithRate(allocator: std.mem.Allocator, sample_rate: u32) ?*AudioState {
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
        .equalizers = std.AutoHashMap(u32, Equalizer).init(allocator),
    };

    var config = zaudio.Engine.Config.init();
    config.channels = 0; // native device layout (stereo / 5.1 / 7.1) → surround spatialisation
    config.listener_count = 1;
    config.sample_rate = sample_rate; // 0 = whatever the device prefers

    const engine = zaudio.Engine.create(config) catch {
        std.log.warn("zigote: failed to create audio engine; sound disabled", .{});
        self.instances.deinit();
        self.groups.deinit();
        self.equalizers.deinit();
        allocator.destroy(self);
        return null;
    };
    self.engine = engine;

    engine.start() catch {
        std.log.warn("zigote: failed to start audio engine; sound disabled", .{});
        engine.destroy();
        self.instances.deinit();
        self.groups.deinit();
        self.equalizers.deinit();
        allocator.destroy(self);
        return null;
    };

    std.log.info(
        "zigote: audio ready (miniaudio engine, spatial/surround, {d} Hz)",
        .{engine.getSampleRate()},
    );
    return self;
}

// zaudio binds the engine's clock for sounds and groups but not for the engine itself; declare the
// miniaudio symbol directly, as this file already does for the node-graph endpoint.
extern fn ma_engine_get_time_in_pcm_frames(engine: *zaudio.Engine) u64;

/// Start a sound at an exact point on the audio clock, `seconds_from_now` ahead of now.
///
/// This is what makes gapless playback actually gapless: the next track is queued to begin on the
/// frame the current one ends, decided by the audio thread rather than by whenever the UI thread
/// next notices the track finished. Polling can only ever be as tight as the frame interval.
pub fn scheduleStart(self: *AudioState, id: u32, seconds_from_now: f32) void {
    const inst = get(self, id) orelse return;
    const rate: f64 = @floatFromInt(self.engine.getSampleRate());
    const ahead: f64 = @max(0.0, @as(f64, @floatCast(seconds_from_now))) * rate;
    const now = ma_engine_get_time_in_pcm_frames(self.engine);
    inst.sound.setStartTimeInPcmFrames(now +% @as(u64, @intFromFloat(ahead)));
    inst.sound.start() catch {};
}

/// The rate the output device is actually running at. Compare against a source's rate to know
/// whether it is being resampled.
pub fn outputRate(self: *AudioState) u32 {
    return self.engine.getSampleRate();
}

pub fn deinit(self: *AudioState) void {
    // Stop + free every live source before tearing down the engine.
    var it = self.instances.valueIterator();
    while (it.next()) |inst| destroyInstance(inst.*);
    self.instances.deinit();

    for (&self.oneshots) |*os| {
        if (os.active) destroyOneShot(os);
    }

    // Groups and equalizer chains after sounds (nothing routes through them any more), before the
    // engine whose node graph owns them.
    var git = self.groups.valueIterator();
    while (git.next()) |g| g.*.destroy();
    self.groups.deinit();

    var eit = self.equalizers.valueIterator();
    while (eit.next()) |eq| destroyBands(eq);
    self.equalizers.deinit();

    self.engine.destroy();
    // Intentionally not calling zaudio.deinit (see `za_inited`).
    self.allocator.destroy(self);
}

fn destroyInstance(inst: Instance) void {
    inst.sound.destroy();
    if (inst.wave) |w| w.destroy();
    if (inst.noise) |n| n.destroy();
    // After the sound: the mixer must not be able to reach a data source that is being freed.
    if (inst.stream) |ns| netstream.destroy(ns);
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

// ── push streams (network radio, and anything else the host feeds by hand) ──────
//
// A file source pulls; a stream source is pushed. Everything downstream — volume, the equalizer
// chain, scheduling, the transport — is identical, because what `createStream` registers is an
// ordinary sound over a `ma_data_source` (see core/netstream.zig).

/// Create a sound fed by `streamPush` rather than by a file (not started). 0 on failure.
pub fn createStream(self: *AudioState) ?u32 {
    const ns = netstream.create(self.allocator, self.engine.getSampleRate()) catch {
        std.log.warn("zigote: failed to create audio stream source", .{});
        return null;
    };
    const snd = self.engine.createSoundFromDataSource(netstream.asDataSource(ns), .{}, null) catch {
        netstream.destroy(ns);
        return null;
    };
    snd.setSpatializationEnabled(false);
    return register(self, .{ .sound = snd, .stream = ns });
}

/// Hand encoded bytes to a stream source. Returns how many were accepted — a short count means
/// its queue is full and the caller should stop reading until it drains.
pub fn streamPush(self: *AudioState, id: u32, bytes: []const u8) usize {
    const inst = get(self, id) orelse return 0;
    const ns = inst.stream orelse return 0;
    return netstream.push(ns, bytes);
}

/// No more bytes are coming. What is queued still plays out, then the sound reports end-of-stream.
pub fn streamFinish(self: *AudioState, id: u32) void {
    const inst = get(self, id) orelse return;
    if (inst.stream) |ns| netstream.finish(ns);
}

/// 0 connecting, 1 playing, 2 undecodable, 3 ended; 2 for an id that is not a stream at all.
pub fn streamState(self: *AudioState, id: u32) u32 {
    const inst = get(self, id) orelse return @intFromEnum(netstream.State.failed);
    const ns = inst.stream orelse return @intFromEnum(netstream.State.failed);
    return @intFromEnum(netstream.state(ns));
}

/// Decoded audio held ahead of the mixer, in seconds — what a "Buffering…" indicator shows.
pub fn streamBuffered(self: *AudioState, id: u32) f32 {
    const inst = get(self, id) orelse return 0;
    const ns = inst.stream orelse return 0;
    return netstream.bufferedSeconds(ns);
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

// ── transport (music playback: seek, position, end-of-stream) ───────────────────
//
// The spatial API above is built for game one-shots, which start and are forgotten. A media player
// needs the opposite: a long file whose cursor it scrubs and whose end it must notice. miniaudio
// already speaks seconds here, so nothing converts frames on either side of the FFI.

/// Seek to an absolute position in seconds. Values past the end leave the sound at its end.
pub fn seekSeconds(self: *AudioState, id: u32, seconds: f32) void {
    const inst = get(self, id) orelse return;
    inst.sound.seekToSecond(@max(0, seconds)) catch {};
}

/// Playback cursor in seconds, or -1 for a source that cannot report one (procedural tones).
pub fn cursorSeconds(self: *AudioState, id: u32) f32 {
    const inst = get(self, id) orelse return -1;
    return inst.sound.getCursorInSeconds() catch -1;
}

/// Total length in seconds, or -1 when unknown (procedural tones, unseekable streams).
pub fn durationSeconds(self: *AudioState, id: u32) f32 {
    const inst = get(self, id) orelse return -1;
    return inst.sound.getLengthInSeconds() catch -1;
}

/// The source decoded past its last frame — the auto-advance signal for a playlist. Distinct from
/// `!isPlaying`, which is also true for a sound that was merely paused.
pub fn atEnd(self: *AudioState, id: u32) bool {
    const inst = get(self, id) orelse return false;
    return inst.sound.isAtEnd();
}

// ── equalizer chains ────────────────────────────────────────────────────────────
//
// A chain of biquad filter nodes spliced between a sound and the master endpoint:
//
//     sound ─→ band[0] ─→ band[1] ─→ … ─→ band[n-1] ─→ endpoint
//
// The three band types are exactly the three filter shapes a parametric EQ needs, and exactly the
// three AutoEq emits in its ParametricEQ profiles (PK / LSC / HSC) — so a downloaded profile maps
// onto a chain one filter per band with no conversion beyond Q → shelf slope below.
//
// Retuning a band (dragging a gain slider) reconfigures its node in place, which recomputes the
// biquad coefficients without touching the graph links or the filter's running state: no clicks,
// no re-linking. Only changing a band's *type* replaces a node, and that is a rare, deliberate act
// (applying a different profile).

pub const BandKind = enum(u8) {
    peak = 0,
    low_shelf = 1,
    high_shelf = 2,

    pub fn fromU8(v: u8) BandKind {
        return switch (v) {
            1 => .low_shelf,
            2 => .high_shelf,
            else => .peak,
        };
    }
};

// AutoEq profiles are 10 bands; the headroom covers hand-built chains without making the fixed
// per-chain array meaningfully large.
const max_eq_bands: usize = 16;

/// One filter slot. `node` is the concrete zaudio node erased to its base type; `kind` records how
/// to reconfigure and destroy it.
const Band = struct {
    kind: BandKind = .peak,
    node: ?*zaudio.Node = null,
};

const Equalizer = struct {
    count: u32 = 0,
    bands: [max_eq_bands]Band = [_]Band{.{}} ** max_eq_bands,
    enabled: bool = true,
};

/// AutoEq (like every parametric EQ UI) specifies shelves by Q, but miniaudio's RBJ shelves take
/// the slope S. Equating the two alpha terms — `sin(w)/(2Q)` for a peak against
/// `sin(w)/2·√((A+1/A)(1/S−1)+2)` for a shelf — gives S = 1 / (1 + (1/Q² − 2)/(A + 1/A)).
/// Q = 0.7071 lands exactly on S = 1 at any gain, which is what AutoEq emits for its shelves.
fn shelfSlope(gain_db: f64, q: f64) f64 {
    const a = std.math.pow(f64, 10, gain_db / 40.0);
    const s = 1.0 / (1.0 + (1.0 / (q * q) - 2.0) / (a + 1.0 / a));
    // Outside (0,1] the RBJ square root goes imaginary and miniaudio would bake NaN coefficients
    // into the biquad — a silent, permanent dropout. Clamp instead.
    return std.math.clamp(s, 0.05, 1.0);
}

fn makeBandNode(self: *AudioState, kind: BandKind, freq: f32, gain_db: f32, q: f32) ?*zaudio.Node {
    const graph = self.engine.asNodeGraphMut();
    const channels = self.engine.asNodeGraph().getChannels();
    const rate = self.engine.getSampleRate();
    const f: f64 = @max(1.0, @as(f64, @floatCast(freq)));
    const g: f64 = @floatCast(gain_db);
    const qq: f64 = @max(0.05, @as(f64, @floatCast(q)));

    return switch (kind) {
        .peak => blk: {
            const n = graph.createPeakNode(
                zaudio.PeakNode.Config.init(channels, rate, g, qq, f),
            ) catch break :blk null;
            break :blk n.asNodeMut();
        },
        .low_shelf => blk: {
            const n = graph.createLoshelfNode(
                zaudio.LoshelfNode.Config.init(channels, rate, g, shelfSlope(g, qq), f),
            ) catch break :blk null;
            break :blk n.asNodeMut();
        },
        .high_shelf => blk: {
            const n = graph.createHishelfNode(
                zaudio.HishelfNode.Config.init(channels, rate, g, shelfSlope(g, qq), f),
            ) catch break :blk null;
            break :blk n.asNodeMut();
        },
    };
}

fn reconfigureBand(self: *AudioState, band: Band, freq: f32, gain_db: f32, q: f32) void {
    const node = band.node orelse return;
    const channels = self.engine.asNodeGraph().getChannels();
    const rate = self.engine.getSampleRate();
    const f: f64 = @max(1.0, @as(f64, @floatCast(freq)));
    const g: f64 = @floatCast(gain_db);
    const qq: f64 = @max(0.05, @as(f64, @floatCast(q)));

    switch (band.kind) {
        .peak => @as(*zaudio.PeakNode, @ptrCast(node)).reconfigure(.{
            .format = .float32,
            .channels = channels,
            .sample_rate = rate,
            .gain_db = g,
            .q = qq,
            .frequency = f,
        }) catch {},
        .low_shelf => @as(*zaudio.LoshelfNode, @ptrCast(node)).reconfigure(.{
            .format = .float32,
            .channels = channels,
            .sample_rate = rate,
            .gain_db = g,
            .shelf_slope = shelfSlope(g, qq),
            .frequency = f,
        }) catch {},
        .high_shelf => @as(*zaudio.HishelfNode, @ptrCast(node)).reconfigure(.{
            .format = .float32,
            .channels = channels,
            .sample_rate = rate,
            .gain_db = g,
            .shelf_slope = shelfSlope(g, qq),
            .frequency = f,
        }) catch {},
    }
}

fn destroyBandNode(band: Band) void {
    const node = band.node orelse return;
    switch (band.kind) {
        .peak => @as(*zaudio.PeakNode, @ptrCast(node)).destroy(),
        .low_shelf => @as(*zaudio.LoshelfNode, @ptrCast(node)).destroy(),
        .high_shelf => @as(*zaudio.HishelfNode, @ptrCast(node)).destroy(),
    }
}

fn destroyBands(eq: *Equalizer) void {
    for (0..eq.count) |i| destroyBandNode(eq.bands[i]);
    eq.* = .{};
}

/// Wire band[i] → band[i+1] → … → endpoint. Re-run whenever a slot's node is replaced.
fn relink(self: *AudioState, eq: *Equalizer) void {
    const endpoint = ma_node_graph_get_endpoint(self.engine.asNodeGraphMut());
    for (0..eq.count) |i| {
        const node = eq.bands[i].node orelse continue;
        const next = if (i + 1 < eq.count) (eq.bands[i + 1].node orelse endpoint) else endpoint;
        node.attachOutputBus(0, next, 0) catch {};
    }
}

/// Where a sound routed through `inst.eq_id` should send its output right now: the chain head, or
/// the master endpoint when the chain is bypassed or gone.
fn eqTarget(self: *AudioState, eq_id: u32) *zaudio.Node {
    const endpoint = ma_node_graph_get_endpoint(self.engine.asNodeGraphMut());
    const eq = self.equalizers.getPtr(eq_id) orelse return endpoint;
    if (!eq.enabled) return endpoint;
    return eq.bands[0].node orelse endpoint;
}

/// Re-point every sound on this chain at its current head. Needed when band 0 is replaced (the head
/// moved) or when the chain is bypassed/re-enabled.
fn reattachSounds(self: *AudioState, eq_id: u32) void {
    const target = eqTarget(self, eq_id);
    var it = self.instances.valueIterator();
    while (it.next()) |inst| {
        if (inst.eq_id != eq_id) continue;
        inst.sound.asNodeMut().attachOutputBus(0, target, 0) catch {};
    }
}

/// Create a chain of `band_count` flat (0 dB peak) filters. 0 on failure. Starting flat means the
/// chain is fully wired before it carries audio, so configuring bands later never re-links.
pub fn eqCreate(self: *AudioState, band_count: u32) ?u32 {
    const n: u32 = std.math.clamp(band_count, 1, @as(u32, max_eq_bands));
    var eq = Equalizer{ .count = n };
    for (0..n) |i| {
        eq.bands[i] = .{
            .kind = .peak,
            .node = makeBandNode(self, .peak, 1000, 0, 0.7071) orelse {
                eq.count = @intCast(i);
                destroyBands(&eq);
                return null;
            },
        };
    }

    const id = self.next_eq_id;
    self.next_eq_id +%= 1;
    if (self.next_eq_id == 0) self.next_eq_id = 1;
    self.equalizers.put(id, eq) catch {
        destroyBands(&eq);
        return null;
    };
    // getPtr only after the put: the map may have rehashed and moved the value.
    relink(self, self.equalizers.getPtr(id).?);
    return id;
}

pub fn eqSetBand(self: *AudioState, eq_id: u32, index: u32, kind: BandKind, freq: f32, gain_db: f32, q: f32) void {
    const eq = self.equalizers.getPtr(eq_id) orelse return;
    if (index >= eq.count) return;
    const band = &eq.bands[index];

    // Same filter type: retune in place, no graph surgery.
    if (band.node != null and band.kind == kind) {
        reconfigureBand(self, band.*, freq, gain_db, q);
        return;
    }

    // Type change: build the replacement first and only unlink the old node once the chain has
    // been re-linked around the new one, so the audio thread never sees a dangling head.
    const fresh = makeBandNode(self, kind, freq, gain_db, q) orelse return;
    const stale = band.*;
    band.* = .{ .kind = kind, .node = fresh };
    relink(self, eq);
    if (index == 0) reattachSounds(self, eq_id);
    destroyBandNode(stale);
}

/// Bypass or re-engage the chain. Bypassing re-routes the sounds straight to the endpoint rather
/// than tearing the chain down, so toggling is instant and the band settings survive — which is
/// what makes an A/B compare button possible.
pub fn eqSetEnabled(self: *AudioState, eq_id: u32, enabled: bool) void {
    const eq = self.equalizers.getPtr(eq_id) orelse return;
    if (eq.enabled == enabled) return;
    eq.enabled = enabled;
    reattachSounds(self, eq_id);
}

pub fn eqDestroy(self: *AudioState, eq_id: u32) void {
    if (eq_id == 0) return;
    if (self.equalizers.getPtr(eq_id)) |eq| eq.enabled = false;
    // Send the sounds back to the endpoint *before* the nodes go away.
    reattachSounds(self, eq_id);
    var it = self.instances.valueIterator();
    while (it.next()) |inst| {
        if (inst.eq_id == eq_id) inst.eq_id = 0;
    }
    if (self.equalizers.fetchRemove(eq_id)) |kv| {
        var eq = kv.value;
        destroyBands(&eq);
    }
}

/// Route a sound through an equalizer chain (eq_id 0 = dry, straight to the master output).
pub fn soundSetEq(self: *AudioState, sound_id: u32, eq_id: u32) void {
    const inst = get(self, sound_id) orelse return;
    inst.eq_id = eq_id;
    inst.sound.asNodeMut().attachOutputBus(0, eqTarget(self, eq_id), 0) catch {};
}

// ── offline file decoding ───────────────────────────────────────────────────────

/// Decode a whole audio file to interleaved f32 at its native rate and channel count. For callers
/// that need the samples themselves rather than playback — waveform overviews, loudness analysis,
/// sampler/IR loading. Loader threads only; never call this from an audio callback.
///
/// The buffer is C-allocated so it can be freed through `decodeFree` with nothing but the pointer,
/// which is what keeps the FFI signature free of slice bookkeeping. Returns null on any failure.
pub fn decodeFile(path: [:0]const u8, out_channels: *u32, out_rate: *u32, out_frames: *u64) ?[*]f32 {
    // channels/rate 0 → keep the file's native layout and rate; only the sample format is forced.
    const decoder = zaudio.Decoder.createFromFile(
        path,
        zaudio.Decoder.Config.init(.float32, 0, 0),
    ) catch {
        std.log.warn("zigote: failed to decode audio file '{s}'", .{path});
        return null;
    };
    defer decoder.destroy();

    var channels: u32 = 0;
    var rate: u32 = 0;
    decoder.getDataFormat(null, &channels, &rate, null) catch return null;
    if (channels == 0) return null;

    const frames = decoder.getLengthInPCMFrames() catch return null;
    if (frames == 0) return null;

    const samples = frames * channels;
    const bytes = samples * @sizeOf(f32);
    const raw = std.c.malloc(bytes) orelse return null;
    const buf: [*]f32 = @ptrCast(@alignCast(raw));

    const read = decoder.readPCMFrames(raw, frames) catch {
        std.c.free(raw);
        return null;
    };
    // A short read is not fatal (some encoders overstate the length); zero the tail so the caller
    // never sees uninitialised samples, and report what actually decoded.
    if (read < frames) {
        const got: usize = @intCast(read * channels);
        @memset(buf[got..@intCast(samples)], 0);
    }

    out_channels.* = channels;
    out_rate.* = rate;
    out_frames.* = read;
    return buf;
}

pub fn decodeFree(frames: [*]f32) void {
    std.c.free(@ptrCast(frames));
}

// ── tests (pure math only — the node graph needs a device) ──────────────────────

test "shelfSlope: Q 0.7071 is slope 1 at any gain" {
    // The identity AutoEq relies on — every shelf it emits is Q 0.70, and a slope of exactly 1 is
    // the maximally-steep shelf with no overshoot. If this drifts, every AutoEq profile is subtly
    // wrong in its bass and treble.
    for ([_]f64{ -12, -6, 0, 5.5, 12 }) |gain|
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), shelfSlope(gain, 0.7071067811865476), 1e-9);
}

test "shelfSlope: stays inside the RBJ domain for extreme Q" {
    // Outside (0,1] miniaudio's sqrt term goes imaginary and bakes NaN coefficients into the
    // biquad — a permanent silent dropout rather than a loud failure. Nothing may escape the clamp.
    for ([_]f64{ 0.05, 0.5, 1.0, 4.0, 12.0 }) |q| {
        for ([_]f64{ -20, 0, 20 }) |gain| {
            const s = shelfSlope(gain, q);
            try std.testing.expect(s > 0 and s <= 1.0);
            try std.testing.expect(!std.math.isNan(s));
        }
    }
}

test "shelfSlope: low Q gives a gentler shelf than high Q" {
    // Slope tracks Q monotonically below 0.7071; the clamp flattens everything above it.
    try std.testing.expect(shelfSlope(6, 0.3) < shelfSlope(6, 0.5));
    try std.testing.expect(shelfSlope(6, 0.5) < shelfSlope(6, 0.7071067811865476));
}
