//! A sound whose bytes arrive from the outside — a push source.
//!
//! Every other audio source in the engine pulls: miniaudio opens a path and reads it at its own
//! pace. Nothing that arrives over a socket can be opened that way, which is why a media player
//! built on `createFile` can play a 60 MB FLAC and not a 128 kbit/s radio station. This closes
//! that gap without teaching the engine about HTTP: the host pushes encoded bytes in, and what
//! comes out is an ordinary `ma_data_source` that mixes, routes through an equalizer chain and
//! obeys the transport exactly like a file does.
//!
//! Three stages, in the order the bytes travel:
//!
//! ```text
//!   host thread            decode thread                audio thread
//!   push(bytes) ──→ [encoded queue] ──→ ma_decoder ──→ [pcm ring] ──→ onRead ──→ mixer
//! ```
//!
//! The split exists because the audio callback may never block or allocate, and decoding cannot
//! promise either — a stalled network read inside the callback is a system-wide dropout, not a
//! gap in one song. So decoding happens on its own thread and hands over a ring of finished PCM;
//! the callback only ever memcpys, and covers an underrun with silence rather than by waiting.
//! That is also why a stall sounds like a brief gap and not a crash.
//!
//! The decoder itself is fed through read/seek callbacks over the encoded queue, which is what
//! lets miniaudio's own MP3/FLAC/WAV/Vorbis backends do the actual decoding — the engine adds a
//! transport, not a codec.

const std = @import("std");
const builtin = @import("builtin");
const zaudio = @import("zaudio");

/// A spin lock and a sleep, because this zig version has no `std.Thread.Mutex` and `std.Io.Mutex`
/// wants an `Io` handle the engine has no reason to hold. Every critical section below is a
/// memcpy of a few kilobytes, and the one place that waits for a while (the decode thread with an
/// empty queue) sleeps *outside* the lock rather than spinning inside it.
///
/// It backs off rather than spinning indefinitely. `push` is called from the host's **UI thread**
/// once a frame, and the longest critical section here — compacting the encoded queue in
/// `onDecoderRead` — is a memmove of up to `max_encoded`. A pure spin would burn that whole
/// memmove out of the frame budget; yielding hands the core to the thread actually holding the
/// lock, which is the one that can end the wait.
const Lock = struct {
    locked: std.atomic.Value(bool) = .init(false),

    /// Uncontended is the overwhelming case, so spin briefly before paying for a syscall — long
    /// enough to cover a short memcpy, short enough not to matter next to a scheduler round.
    const spins_before_yield: u32 = 64;

    fn lock(self: *Lock) void {
        var spins: u32 = 0;
        while (self.locked.cmpxchgWeak(false, true, .acquire, .monotonic) != null) {
            spins += 1;
            if (spins < spins_before_yield) {
                std.atomic.spinLoopHint();
            } else {
                spins = 0;
                std.Thread.yield() catch std.atomic.spinLoopHint();
            }
        }
    }

    fn unlock(self: *Lock) void {
        self.locked.store(false, .release);
    }
};

/// How long a waiter naps before looking again. Irrelevant next to network latency, and it keeps
/// an idle decode thread off the CPU.
const poll_ms: u64 = 5;

// Declared here rather than taken from std: zig 0.16 dropped both `std.os.windows.kernel32.Sleep`
// and `std.Thread.sleep` (sleeping moved behind the new std.Io interface, which this FFI layer has
// no Io instance to reach). One extern is smaller than threading an Io through for a 5 ms nap.
extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.winapi) void;

fn nap() void {
    if (builtin.os.tag == .windows) {
        Sleep(@intCast(poll_ms));
    } else {
        var ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(poll_ms * std.time.ns_per_ms) };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// How much encoded audio to gather before letting the decoder look at it. Container detection
/// reads a header and seeks back; too small a window and it decides the stream is truncated.
const probe_bytes: usize = 64 * 1024;

/// Ceiling on the encoded queue. Reached only when the decode thread is stalled or the source is
/// far faster than real time; `push` reports the shortfall so the host can stop reading rather
/// than grow this without bound.
const max_encoded: usize = 4 * 1024 * 1024;

/// Decoded audio held ahead of the mixer, in seconds. Covers ordinary network jitter; the deeper
/// buffer is the encoded queue, which holds the same seconds for a tenth of the memory.
const ring_seconds: usize = 4;

/// Fixed output shape. Every decoder is configured to convert to this, so the data source can
/// answer `onGetDataFormat` before a single byte has arrived — which it must, since miniaudio
/// asks at attach time.
const channels: u32 = 2;

pub const State = enum(u32) {
    /// No decoder yet: too few bytes to identify the container.
    connecting = 0,
    /// Decoding. The ring may still be filling — see `bufferedSeconds`.
    playing = 1,
    /// Nothing here can decode this stream (an AAC station is the usual reason).
    failed = 2,
    /// The source ended and the ring drained. `atEnd` on the sound follows.
    ended = 3,
};

/// The first field of what miniaudio sees. `ma_data_source_init` writes into `base`, and the
/// callbacks get a pointer to it back — so it is kept in an `extern struct` whose layout Zig may
/// not reorder, with a back-pointer to the Zig side that has no layout requirements at all.
const Head = extern struct {
    base: zaudio.DataSourceBase,
    owner: *anyopaque,
};

pub const NetStream = struct {
    head: Head,
    allocator: std.mem.Allocator,
    sample_rate: u32,

    // ── encoded queue: host thread → decode thread ────────────────────────────
    lock: Lock = .{},
    encoded: std.ArrayList(u8),
    /// How far the decoder has read into `encoded`. Bytes before it are dropped once the
    /// container has been identified, and retained until then so init-time seeks can go back.
    read_pos: usize = 0,
    /// The host will send no more bytes.
    finished: bool = false,
    /// Teardown: unblocks the decode thread wherever it is waiting.
    stop: bool = false,

    // ── pcm ring: decode thread → audio thread ───────────────────────────────
    // One producer, one consumer, monotonic counters — so the audio thread takes no lock.
    pcm: []f32,
    capacity: u64,
    write_frames: std.atomic.Value(u64) = .init(0),
    read_frames: std.atomic.Value(u64) = .init(0),

    state: std.atomic.Value(u32) = .init(@intFromEnum(State.connecting)),
    /// Frames handed to the mixer, silence included — the cursor a player displays.
    delivered: std.atomic.Value(u64) = .init(0),

    thread: ?std.Thread = null,
    decoder: ?*zaudio.Decoder = null,
};

const vtable = zaudio.DataSource.VTable{
    .onRead = onRead,
    .onSeek = onSeek,
    .onGetDataFormat = onGetDataFormat,
    .onGetCursor = onGetCursor,
    .onGetLength = onGetLength,
    .onSetLooping = null,
    .flags = .{},
};

fn fromDataSource(ds: *zaudio.DataSource) *NetStream {
    const head: *Head = @ptrCast(@alignCast(ds));
    return @ptrCast(@alignCast(head.owner));
}

/// Allocate a push stream and start its decode thread. The returned data source can be handed to
/// `createSoundFromDataSource` immediately — it answers as silence until bytes arrive.
pub fn create(allocator: std.mem.Allocator, sample_rate: u32) !*NetStream {
    const rate = if (sample_rate == 0) 48000 else sample_rate;
    const capacity: u64 = @as(u64, rate) * ring_seconds;

    const self = try allocator.create(NetStream);
    errdefer allocator.destroy(self);

    const pcm = try allocator.alloc(f32, @intCast(capacity * channels));
    errdefer allocator.free(pcm);

    self.* = .{
        .head = .{ .base = undefined, .owner = undefined },
        .allocator = allocator,
        .sample_rate = rate,
        .encoded = .empty,
        .pcm = pcm,
        .capacity = capacity,
    };
    self.head.owner = @ptrCast(self);

    var config = zaudio.DataSource.Config.init();
    config.vtable = &vtable;
    _ = try zaudio.DataSource.create(config, &self.head.base);

    self.thread = try std.Thread.spawn(.{}, decodeLoop, .{self});
    return self;
}

pub fn asDataSource(self: *NetStream) *zaudio.DataSource {
    return @ptrCast(@alignCast(&self.head));
}

/// Stop the decode thread and release everything. The sound built on this must already be gone —
/// the mixer must not be able to reach a data source that is being freed.
pub fn destroy(self: *NetStream) void {
    {
        self.lock.lock();
        defer self.lock.unlock();
        self.stop = true;
    }
    if (self.thread) |t| t.join();
    if (self.decoder) |d| d.destroy();

    const allocator = self.allocator;
    self.encoded.deinit(allocator);
    allocator.free(self.pcm);
    allocator.destroy(self);
}

/// Hand over encoded bytes. Returns how many were taken — a short count means the queue is full
/// and the host should stop reading until it drains, rather than buffer a radio station forever.
pub fn push(self: *NetStream, bytes: []const u8) usize {
    self.lock.lock();
    defer self.lock.unlock();
    if (self.stop or self.finished) return 0;

    const room = max_encoded -| (self.encoded.items.len - self.read_pos);
    const take = @min(room, bytes.len);
    if (take == 0) return 0;

    self.encoded.appendSlice(self.allocator, bytes[0..take]) catch return 0;
    return take;
}

/// No more bytes are coming. What is already queued still plays out, and the sound reports
/// end-of-stream once it does.
pub fn finish(self: *NetStream) void {
    self.lock.lock();
    defer self.lock.unlock();
    self.finished = true;
}

pub fn state(self: *NetStream) State {
    return @enumFromInt(self.state.load(.acquire));
}

/// Decoded audio held ahead of the mixer. What a "Buffering…" indicator shows, and what tells a
/// player it is safe to start.
pub fn bufferedSeconds(self: *NetStream) f32 {
    const w = self.write_frames.load(.acquire);
    const r = self.read_frames.load(.acquire);
    return @as(f32, @floatFromInt(w -| r)) / @as(f32, @floatFromInt(self.sample_rate));
}

// ── decode thread ───────────────────────────────────────────────────────────────

fn setState(self: *NetStream, value: State) void {
    self.state.store(@intFromEnum(value), .release);
}

fn decodeLoop(self: *NetStream) void {
    const decoder = openDecoder(self) orelse {
        // `connecting` → `failed` unless we are simply being torn down, in which case the sound
        // is going away anyway and the state nobody will read does not matter.
        if (!stopped(self)) setState(self, .failed);
        return;
    };
    self.decoder = decoder;
    setState(self, .playing);

    // Sized so a read is a meaningful chunk of work without outrunning the ring in one go.
    var scratch: [4096 * channels]f32 = undefined;
    const scratch_frames: u64 = scratch.len / channels;

    while (!stopped(self)) {
        const w = self.write_frames.load(.acquire);
        const r = self.read_frames.load(.acquire);
        const room = self.capacity - (w -| r);
        if (room == 0) {
            // The mixer is behind: the ring is full and there is nothing useful to do. Sleeping
            // beats spinning — five milliseconds is a thousandth of the buffer.
            nap();
            continue;
        }

        const want = @min(room, scratch_frames);

        // Bound directly rather than through zaudio's wrapper, which checks the result before it
        // returns the count: miniaudio reports the last read of a source as MA_AT_END *with* the
        // frames it managed to decode, and the wrapper turns that into an error that throws them
        // away. For an MP3 that is not a clipped tail but the whole stream — one read consumed the
        // rest of the file, so the frames dropped were all of them.
        var got: u64 = 0;
        const result = ma_decoder_read_pcm_frames(decoder, &scratch, want, &got);
        if (got > 0) writeRing(self, scratch[0..@intCast(got * channels)]);

        // Anything but a clean read is the end: an empty queue blocks in `onDecoderRead` rather
        // than returning short, so a shortfall here can only mean the source really ran out.
        if (result != .success or got == 0) {
            setState(self, .ended);
            return;
        }
    }
}

/// Wait for enough bytes to identify the container, then build a decoder that converts whatever
/// it finds to our fixed output shape. Null when the stream ended first, or when no backend
/// recognises it.
fn openDecoder(self: *NetStream) ?*zaudio.Decoder {
    while (true) {
        self.lock.lock();
        const ready = self.stop or self.finished or self.encoded.items.len >= probe_bytes;
        const give_up = self.stop or (self.finished and self.encoded.items.len == 0);
        self.lock.unlock();
        if (give_up) return null;
        if (ready) break;
        nap();
    }

    var config = zaudio.Decoder.Config.init(.float32, channels, self.sample_rate);
    config.encoding_format = .unknown;

    var decoder: ?*zaudio.Decoder = null;
    const result = zaudioDecoderCreate(onDecoderRead, onDecoderSeek, @ptrCast(self), &config, &decoder);
    if (result == .success and decoder != null) return decoder.?;
    {
        // ponytail: miniaudio decodes MP3/FLAC/WAV/Vorbis, so an AAC station lands here. Adding a
        // backend means a custom decoding vtable, not a change to this file.
        std.log.warn("zigote: no decoder backend recognised this stream", .{});
        return null;
    }
}

// zaudio declares these callbacks as function *types* rather than pointers, which C calling
// convention forbids as a parameter — its `Decoder.create` wrapper does not compile. The C shim
// underneath it is fine, so bind that directly (as the equalizer does for the node-graph endpoint).
extern fn ma_decoder_read_pcm_frames(
    decoder: *zaudio.Decoder,
    frames_out: *anyopaque,
    frame_count: u64,
    frames_read: ?*u64,
) zaudio.Result;

extern fn zaudioDecoderCreate(
    on_read: *const fn (*zaudio.Decoder, *anyopaque, usize, *usize) callconv(.c) zaudio.Result,
    on_seek: *const fn (*zaudio.Decoder, i64, zaudio.Vfs.SeekOrigin) callconv(.c) zaudio.Result,
    user_data: *anyopaque,
    config: *const zaudio.Decoder.Config,
    out_handle: ?*?*zaudio.Decoder,
) zaudio.Result;

fn stopped(self: *NetStream) bool {
    self.lock.lock();
    defer self.lock.unlock();
    return self.stop;
}

fn writeRing(self: *NetStream, samples: []const f32) void {
    const w = self.write_frames.load(.acquire);
    var offset: u64 = 0;
    const frames: u64 = samples.len / channels;

    while (offset < frames) {
        const at: usize = @intCast(((w + offset) % self.capacity) * channels);
        const run = @min(frames - offset, self.capacity - ((w + offset) % self.capacity));
        const count: usize = @intCast(run * channels);
        @memcpy(
            self.pcm[at .. at + count],
            samples[@intCast(offset * channels)..@intCast((offset + run) * channels)],
        );
        offset += run;
    }

    // Published last: until this store lands the consumer must not see these frames.
    self.write_frames.store(w + frames, .release);
}

// ── decoder callbacks (decode thread) ───────────────────────────────────────────
//
// These run on our own thread, so unlike the audio callback they are free to block — which is
// exactly what a stream needs: an empty queue means "not yet", never "the end". Returning short
// here would make miniaudio believe the file was truncated and stop the stream for good.

fn onDecoderRead(
    decoder: *zaudio.Decoder,
    buffer_out: *anyopaque,
    bytes_to_read: usize,
    bytes_read: *usize,
) callconv(.c) zaudio.Result {
    const self: *NetStream = @ptrCast(@alignCast(zaudio.Decoder.getUserData(decoder).?));
    const out: [*]u8 = @ptrCast(buffer_out);

    // Wait for bytes rather than returning short: a short read tells miniaudio the file was
    // truncated, and it stops the stream for good. Only `finish` or teardown ends this.
    while (true) {
        self.lock.lock();
        if (self.stop or self.finished or self.encoded.items.len > self.read_pos) break;
        self.lock.unlock();
        nap();
    }
    defer self.lock.unlock();

    const available = self.encoded.items.len - self.read_pos;
    const take = @min(available, bytes_to_read);
    if (take == 0) {
        bytes_read.* = 0;
        return .at_end;
    }

    @memcpy(out[0..take], self.encoded.items[self.read_pos .. self.read_pos + take]);
    self.read_pos += take;
    bytes_read.* = take;

    // Once a backend is chosen nothing seeks backwards again, so consumed bytes can go. Held
    // until then because container detection rewinds to zero between attempts. Compact only when
    // the consumed prefix outweighs the remainder — shifting the whole tail down on EVERY read
    // was O(n²) over the stream's life; amortized it is now O(n).
    if (self.decoder != null and self.read_pos > probe_bytes and
        self.read_pos > self.encoded.items.len / 2)
    {
        const remaining = self.encoded.items.len - self.read_pos;
        std.mem.copyForwards(u8, self.encoded.items[0..remaining], self.encoded.items[self.read_pos..]);
        self.encoded.shrinkRetainingCapacity(remaining);
        self.read_pos = 0;
    }

    return .success;
}

fn onDecoderSeek(
    decoder: *zaudio.Decoder,
    byte_offset: i64,
    origin: zaudio.Vfs.SeekOrigin,
) callconv(.c) zaudio.Result {
    const self: *NetStream = @ptrCast(@alignCast(zaudio.Decoder.getUserData(decoder).?));

    self.lock.lock();
    defer self.lock.unlock();

    const base: i64 = switch (origin) {
        .start => 0,
        .current => @intCast(self.read_pos),
        // A live stream has no end to seek from, and no backend asks for one during detection.
        .end => return .generic_error,
    };
    const target = base + byte_offset;
    if (target < 0 or target > @as(i64, @intCast(self.encoded.items.len))) return .generic_error;
    self.read_pos = @intCast(target);
    return .success;
}

// ── data source callbacks (audio thread) ────────────────────────────────────────
//
// No locks, no allocation, no waiting. An underrun is filled with silence and reported as a
// normal read: a stall must sound like a gap, not like the end of the track.

fn onRead(
    ds: *zaudio.DataSource,
    frames_out: ?*anyopaque,
    frame_count: u64,
    frames_read: *u64,
) callconv(.c) zaudio.Result {
    const self = fromDataSource(ds);
    // A null output buffer means "consume these frames and throw them away" — miniaudio reads that
    // way when it skips forward. The frames still have to leave the ring, they just go nowhere, so
    // this is an optional pointer and not a `.?`: under ReleaseFast an unchecked unwrap is not a
    // panic but a store through a wild pointer, from the audio thread.
    const out: ?[*]f32 = if (frames_out) |p| @ptrCast(@alignCast(p)) else null;

    const w = self.write_frames.load(.acquire);
    const r = self.read_frames.load(.acquire);
    const available = w -| r;

    if (available == 0) {
        const current: State = @enumFromInt(self.state.load(.acquire));
        if (current == .ended or current == .failed) {
            frames_read.* = 0;
            return .at_end;
        }
        // Still connecting, or the network fell behind: silence, and try again next callback.
        if (out) |dst| @memset(dst[0..@intCast(frame_count * channels)], 0);
        frames_read.* = frame_count;
        _ = self.delivered.fetchAdd(frame_count, .monotonic);
        return .success;
    }

    const take = @min(available, frame_count);
    var offset: u64 = 0;
    while (offset < take) {
        const at: usize = @intCast(((r + offset) % self.capacity) * channels);
        const run = @min(take - offset, self.capacity - ((r + offset) % self.capacity));
        const count: usize = @intCast(run * channels);
        if (out) |dst| @memcpy(dst[@intCast(offset * channels)..][0..count], self.pcm[at .. at + count]);
        offset += run;
    }

    self.read_frames.store(r + take, .release);
    frames_read.* = take;
    _ = self.delivered.fetchAdd(take, .monotonic);
    return .success;
}

/// A live stream has no position to seek to. Reported as unsupported rather than silently
/// ignored, so a transport can grey out its scrubber instead of pretending.
fn onSeek(ds: *zaudio.DataSource, frame_index: u64) callconv(.c) zaudio.Result {
    _ = ds;
    _ = frame_index;
    return .not_implemented;
}

fn onGetDataFormat(
    ds: *zaudio.DataSource,
    format: ?*zaudio.Format,
    out_channels: ?*u32,
    sample_rate: ?*u32,
    channel_map: ?[*]zaudio.Channel,
    channel_map_cap: usize,
) callconv(.c) zaudio.Result {
    const self = fromDataSource(ds);
    if (format) |f| f.* = .float32;
    if (out_channels) |c| c.* = channels;
    if (sample_rate) |r| r.* = self.sample_rate;
    if (channel_map) |m| {
        if (channel_map_cap >= 2) {
            m[0] = 1; // MA_CHANNEL_FRONT_LEFT
            m[1] = 2; // MA_CHANNEL_FRONT_RIGHT
        }
    }
    return .success;
}

/// How long we have been playing, silence included — the only honest answer for a source with no
/// beginning to measure from.
fn onGetCursor(ds: *zaudio.DataSource, cursor: ?*u64) callconv(.c) zaudio.Result {
    const self = fromDataSource(ds);
    if (cursor) |c| c.* = self.delivered.load(.monotonic);
    return .success;
}

fn onGetLength(ds: *zaudio.DataSource, length: ?*u64) callconv(.c) zaudio.Result {
    _ = ds;
    _ = length;
    return .not_implemented;
}

// ── self-check ──────────────────────────────────────────────────────────────────

test "pushed bytes decode, play out, and then report end of stream" {
    // Both the data source and the decoder are allocated through zaudio's process-global
    // allocator, so it has to exist first. In the app `AudioState.init` has already done this.
    zaudio.init(std.testing.allocator);
    // The app never unwinds this (see `za_inited`), but a leak-checked test has to.
    defer zaudio.deinit();

    const rate: u32 = 44100;
    const frames: u32 = 4000;

    // A 16-bit stereo WAV built by hand: the smallest thing miniaudio will recognise, so the test
    // exercises the real decoder rather than a stub, and needs no device and no file on disk.
    var wav: std.ArrayList(u8) = .empty;
    defer wav.deinit(std.testing.allocator);
    const data_bytes: u32 = frames * 2 * 2;

    const put = struct {
        fn le(list: *std.ArrayList(u8), comptime T: type, value: T) !void {
            var bytes: [@sizeOf(T)]u8 = undefined;
            std.mem.writeInt(T, &bytes, value, .little);
            try list.appendSlice(std.testing.allocator, &bytes);
        }
    }.le;

    try wav.appendSlice(std.testing.allocator, "RIFF");
    try put(&wav, u32, 36 + data_bytes);
    try wav.appendSlice(std.testing.allocator, "WAVEfmt ");
    try put(&wav, u32, 16); // fmt chunk size
    try put(&wav, u16, 1); // PCM
    try put(&wav, u16, 2); // channels
    try put(&wav, u32, rate);
    try put(&wav, u32, rate * 4); // byte rate
    try put(&wav, u16, 4); // block align
    try put(&wav, u16, 16); // bits per sample
    try wav.appendSlice(std.testing.allocator, "data");
    try put(&wav, u32, data_bytes);
    for (0..frames) |i| {
        const sample: i16 = if (i % 100 < 50) 8000 else -8000;
        try put(&wav, i16, sample);
        try put(&wav, i16, sample);
    }

    const self = try create(std.testing.allocator, rate);
    const ds = asDataSource(self);

    try std.testing.expectEqual(wav.items.len, push(self, wav.items));
    finish(self);

    var out: [512 * channels]f32 = undefined;
    var loud = false;
    var ended = false;
    // Bounded: 2000 × 5 ms is ten seconds, far past the tenth of a second of audio pushed above.
    for (0..2000) |_| {
        var read: u64 = 0;
        const result = onRead(ds, &out, 512, &read);
        if (result == .at_end) {
            ended = true;
            break;
        }
        for (out[0..@intCast(read * channels)]) |sample| {
            if (@abs(sample) > 0.1) loud = true;
        }
        nap();
    }

    try std.testing.expect(loud); // the square wave came back out
    try std.testing.expect(ended); // and the stream closed rather than looping or hanging
    destroy(self);
}

test "a discard read consumes frames without an output buffer" {
    // miniaudio skips forward by reading with a null output buffer. The frames must still leave the
    // ring, and nothing may be written — in a safe build an unchecked unwrap panics here, and in the
    // ReleaseFast build the app ships it would store through a wild pointer instead.
    zaudio.init(std.testing.allocator);
    defer zaudio.deinit();

    const rate: u32 = 44100;
    const self = try create(std.testing.allocator, rate);
    const ds = asDataSource(self);

    // Straight into the ring, so the test needs no decoder and no real container.
    const frames: u64 = 256;
    writeRing(self, &[_]f32{0.5} ** @intCast(frames * channels));

    var read: u64 = 0;
    try std.testing.expectEqual(zaudio.Result.success, onRead(ds, null, frames, &read));
    try std.testing.expectEqual(frames, read);
    // Consumed: the ring is empty, so the next read is the underrun path — also with no buffer.
    try std.testing.expectEqual(@as(u64, 0), self.write_frames.load(.acquire) -|
        self.read_frames.load(.acquire));
    try std.testing.expectEqual(zaudio.Result.success, onRead(ds, null, frames, &read));

    destroy(self);
}

test "a real mp3 decodes end to end through the push path" {
    // An eight-kilobyte second of sine, encoded by ffmpeg. A synthetic WAV cannot stand in for it:
    // MP3 is what internet radio actually sends, and it is the format whose decoder reads far
    // ahead — which is how it exposed miniaudio reporting its *last* read as an error carrying a
    // full buffer of audio. Decoded through the wrong wrapper that was the entire stream, silently.
    const bytes = @embedFile("netstream_test.mp3");

    zaudio.init(std.testing.allocator);
    defer zaudio.deinit();

    const rate: u32 = 44100;
    const self = try create(std.testing.allocator, rate);
    const ds = asDataSource(self);

    // Fed in pieces, the way a socket delivers it, rather than as one buffer.
    var at: usize = 0;
    while (at < bytes.len) {
        const take = @min(bytes.len - at, 2048);
        var sent: usize = 0;
        while (sent < take) {
            const n = push(self, bytes[at + sent .. at + take]);
            if (n == 0) {
                nap();
                continue;
            }
            sent += n;
        }
        at += take;
    }
    finish(self);

    var out: [512 * channels]f32 = undefined;
    var frames: u64 = 0;
    var peak: f32 = 0;
    for (0..4000) |_| {
        var read: u64 = 0;
        if (onRead(ds, &out, 512, &read) == .at_end) break;
        frames += read;
        for (out[0..@intCast(read * channels)]) |sample| peak = @max(peak, @abs(sample));
        nap();
    }

    // A second of audio, within a few frames — a truncated tail or a dropped buffer shows up here
    // as a shortfall rather than as a stream that merely sounds odd.
    try std.testing.expect(frames > rate * 9 / 10);
    try std.testing.expect(frames < rate * 12 / 10);
    try std.testing.expect(peak > 0.01); // and it is the sine, not silence
    destroy(self);
}
