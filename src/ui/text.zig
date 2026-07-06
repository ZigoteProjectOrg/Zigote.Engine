const std = @import("std");
const geometry = @import("geometry.zig");

pub const FontStyle = enum {
    normal,
    italic,
};

pub const FontAsset = struct {
    name: []const u8,
    source: Source,

    pub const Source = union(enum) {
        embedded: []const u8,
        platform_path: []const u8,
    };

    pub fn fromMemory(name: []const u8, data: []const u8) FontAsset {
        return .{ .name = name, .source = .{ .embedded = data } };
    }

    pub fn fromPlatform(name: []const u8, path: []const u8) FontAsset {
        return .{ .name = name, .source = .{ .platform_path = path } };
    }
};

pub const FontWeight = enum(u16) {
    w100 = 100,
    w200 = 200,
    w300 = 300,
    w400 = 400,
    w500 = 500,
    w600 = 600,
    w700 = 700,
    w800 = 800,
    w900 = 900,

    pub const normal: FontWeight = .w400;
    pub const bold: FontWeight = .w700;
};

pub const TextBaseline = enum {
    alphabetic,
    ideographic,
};

pub const TextDecoration = enum(u8) {
    none = 0,
    underline = 1,
    overline = 2,
    lineThrough = 4,
};

pub const TextDecorationStyle = enum {
    solid,
    double,
    dotted,
    dashed,
    wavy,
};

pub const TextLeadingDistribution = enum {
    even,
    proportional,
};

pub const TextOverflow = enum {
    clip,
    fade,
    ellipsis,
    visible,
};

pub const Shadow = struct {
    color: geometry.Color = geometry.Color.rgba(0, 0, 0, 255),
    offset: geometry.Offset = .{},
    blur_radius: f32 = 0,
};

pub const FontFeature = struct {
    tag: u32,
    value: u32 = 1,
};

pub const FontVariation = struct {
    axis: u32,
    value: f32,
};

pub const TextStyle = struct {
    inherit: bool = true,
    color: geometry.Color = geometry.Color.rgb(28, 31, 36),
    background_color: ?geometry.Color = null,
    size: f32 = 14,
    font_weight: FontWeight = .normal,
    font_style: FontStyle = .normal,
    letter_spacing: f32 = 0,
    word_spacing: f32 = 0,
    text_baseline: ?TextBaseline = null,
    line_height: f32 = 0,
    leading_distribution: ?TextLeadingDistribution = null,
    locale: ?[]const u8 = null,
    foreground: ?geometry.Color = null,
    background: ?geometry.Color = null,
    shadows: []const Shadow = &.{},
    font_features: []const FontFeature = &.{},
    font_variations: []const FontVariation = &.{},
    decoration: ?TextDecoration = null,
    decoration_color: ?geometry.Color = null,
    decoration_style: ?TextDecorationStyle = null,
    decoration_thickness: ?f32 = null,
    debug_label: ?[]const u8 = null,
    font_family: ?[]const u8 = null,
    font_family_fallback: []const []const u8 = &.{},
    package: ?[]const u8 = null,
    overflow: ?TextOverflow = null,

    pub fn resolvedLineHeight(self: TextStyle) f32 {
        return if (self.line_height > 0) self.line_height else self.size * 1.25;
    }
};

pub fn lineHeight(style: TextStyle) f32 {
    return style.resolvedLineHeight();
}

pub const Metrics = struct {
    size: geometry.Size,
    line_count: usize,
    line_height: f32,
};

pub fn measure(value: []const u8, style: TextStyle, max_width: f32) Metrics {
    const lh = lineHeight(style);
    const ch_width = textWidth(1, style.size);
    var max_line_width: f32 = 0;
    var line_count: usize = 1;

    var current_line_width: f32 = 0;
    var word_width: f32 = 0;

    for (value) |ch| {
        if (ch == '\n') {
            current_line_width += word_width;
            max_line_width = @max(max_line_width, current_line_width);
            current_line_width = 0;
            word_width = 0;
            line_count += 1;
            continue;
        }

        if (ch == ' ') {
            current_line_width += word_width + ch_width;
            word_width = 0;
            if (current_line_width > max_width and max_width > 0) {
                max_line_width = @max(max_line_width, current_line_width - ch_width);
                current_line_width = 0;
                line_count += 1;
            }
        } else {
            word_width += ch_width;
            if (max_width > 0 and current_line_width + word_width > max_width) {
                if (current_line_width > 0) {
                    // Flush the current line; word_width carries to the next line.
                    max_line_width = @max(max_line_width, current_line_width);
                    current_line_width = 0;
                    line_count += 1;
                } else {
                    // Word alone exceeds max_width; force-break at boundary.
                    max_line_width = @max(max_line_width, word_width - ch_width);
                    word_width = ch_width;
                    line_count += 1;
                }
            }
        }
    }

    current_line_width += word_width;
    max_line_width = @max(max_line_width, current_line_width);

    return .{
        .size = .{
            .width = max_line_width,
            .height = @as(f32, @floatFromInt(line_count)) * lh,
        },
        .line_count = line_count,
        .line_height = lh,
    };
}

pub fn textWidth(chars: usize, size: f32) f32 {
    const base_char_width: f32 = 8;
    const scale = size / 14.0;
    return @as(f32, @floatFromInt(chars)) * base_char_width * scale;
}
