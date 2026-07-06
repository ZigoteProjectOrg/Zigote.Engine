const std = @import("std");

pub const Color = packed struct(u32) {
    r: u8,
    g: u8,
    b: u8,
    a: u8 = 255,

    // Material Colors.
    // Values are 32-bit ARGB: 0xAARRGGBB.

    pub const transparent: Color = hex(0x00000000);

    pub const black: Color = hex(0xFF000000);
    pub const black_87: Color = hex(0xDD000000);
    pub const black_54: Color = hex(0x8A000000);
    pub const black_45: Color = hex(0x73000000);
    pub const black_38: Color = hex(0x61000000);
    pub const black_26: Color = hex(0x42000000);
    pub const black_12: Color = hex(0x1F000000);

    pub const white: Color = hex(0xFFFFFFFF);
    pub const white_70: Color = hex(0xB3FFFFFF);
    pub const white_60: Color = hex(0x99FFFFFF);
    pub const white_54: Color = hex(0x8AFFFFFF);
    pub const white_38: Color = hex(0x62FFFFFF);
    pub const white_30: Color = hex(0x4DFFFFFF);
    pub const white_24: Color = hex(0x3DFFFFFF);
    pub const white_12: Color = hex(0x1FFFFFFF);
    pub const white_10: Color = hex(0x1AFFFFFF);

    pub const red: Color = red_500;
    pub const red_50: Color = hex(0xFFFFEBEE);
    pub const red_100: Color = hex(0xFFFFCDD2);
    pub const red_200: Color = hex(0xFFEF9A9A);
    pub const red_300: Color = hex(0xFFE57373);
    pub const red_400: Color = hex(0xFFEF5350);
    pub const red_500: Color = hex(0xFFF44336);
    pub const red_600: Color = hex(0xFFE53935);
    pub const red_700: Color = hex(0xFFD32F2F);
    pub const red_800: Color = hex(0xFFC62828);
    pub const red_900: Color = hex(0xFFB71C1C);

    pub const red_accent: Color = red_accent_200;
    pub const red_accent_100: Color = hex(0xFFFF8A80);
    pub const red_accent_200: Color = hex(0xFFFF5252);
    pub const red_accent_400: Color = hex(0xFFFF1744);
    pub const red_accent_700: Color = hex(0xFFD50000);

    pub const pink: Color = pink_500;
    pub const pink_50: Color = hex(0xFFFCE4EC);
    pub const pink_100: Color = hex(0xFFF8BBD0);
    pub const pink_200: Color = hex(0xFFF48FB1);
    pub const pink_300: Color = hex(0xFFF06292);
    pub const pink_400: Color = hex(0xFFEC407A);
    pub const pink_500: Color = hex(0xFFE91E63);
    pub const pink_600: Color = hex(0xFFD81B60);
    pub const pink_700: Color = hex(0xFFC2185B);
    pub const pink_800: Color = hex(0xFFAD1457);
    pub const pink_900: Color = hex(0xFF880E4F);

    pub const pink_accent: Color = pink_accent_200;
    pub const pink_accent_100: Color = hex(0xFFFF80AB);
    pub const pink_accent_200: Color = hex(0xFFFF4081);
    pub const pink_accent_400: Color = hex(0xFFF50057);
    pub const pink_accent_700: Color = hex(0xFFC51162);

    pub const purple: Color = purple_500;
    pub const purple_50: Color = hex(0xFFF3E5F5);
    pub const purple_100: Color = hex(0xFFE1BEE7);
    pub const purple_200: Color = hex(0xFFCE93D8);
    pub const purple_300: Color = hex(0xFFBA68C8);
    pub const purple_400: Color = hex(0xFFAB47BC);
    pub const purple_500: Color = hex(0xFF9C27B0);
    pub const purple_600: Color = hex(0xFF8E24AA);
    pub const purple_700: Color = hex(0xFF7B1FA2);
    pub const purple_800: Color = hex(0xFF6A1B9A);
    pub const purple_900: Color = hex(0xFF4A148C);

    pub const purple_accent: Color = purple_accent_200;
    pub const purple_accent_100: Color = hex(0xFFEA80FC);
    pub const purple_accent_200: Color = hex(0xFFE040FB);
    pub const purple_accent_400: Color = hex(0xFFD500F9);
    pub const purple_accent_700: Color = hex(0xFFAA00FF);

    pub const deep_purple: Color = deep_purple_500;
    pub const deep_purple_50: Color = hex(0xFFEDE7F6);
    pub const deep_purple_100: Color = hex(0xFFD1C4E9);
    pub const deep_purple_200: Color = hex(0xFFB39DDB);
    pub const deep_purple_300: Color = hex(0xFF9575CD);
    pub const deep_purple_400: Color = hex(0xFF7E57C2);
    pub const deep_purple_500: Color = hex(0xFF673AB7);
    pub const deep_purple_600: Color = hex(0xFF5E35B1);
    pub const deep_purple_700: Color = hex(0xFF512DA8);
    pub const deep_purple_800: Color = hex(0xFF4527A0);
    pub const deep_purple_900: Color = hex(0xFF311B92);

    pub const deep_purple_accent: Color = deep_purple_accent_200;
    pub const deep_purple_accent_100: Color = hex(0xFFB388FF);
    pub const deep_purple_accent_200: Color = hex(0xFF7C4DFF);
    pub const deep_purple_accent_400: Color = hex(0xFF651FFF);
    pub const deep_purple_accent_700: Color = hex(0xFF6200EA);

    pub const indigo: Color = indigo_500;
    pub const indigo_50: Color = hex(0xFFE8EAF6);
    pub const indigo_100: Color = hex(0xFFC5CAE9);
    pub const indigo_200: Color = hex(0xFF9FA8DA);
    pub const indigo_300: Color = hex(0xFF7986CB);
    pub const indigo_400: Color = hex(0xFF5C6BC0);
    pub const indigo_500: Color = hex(0xFF3F51B5);
    pub const indigo_600: Color = hex(0xFF3949AB);
    pub const indigo_700: Color = hex(0xFF303F9F);
    pub const indigo_800: Color = hex(0xFF283593);
    pub const indigo_900: Color = hex(0xFF1A237E);

    pub const indigo_accent: Color = indigo_accent_200;
    pub const indigo_accent_100: Color = hex(0xFF8C9EFF);
    pub const indigo_accent_200: Color = hex(0xFF536DFE);
    pub const indigo_accent_400: Color = hex(0xFF3D5AFE);
    pub const indigo_accent_700: Color = hex(0xFF304FFE);

    pub const blue: Color = blue_500;
    pub const blue_50: Color = hex(0xFFE3F2FD);
    pub const blue_100: Color = hex(0xFFBBDEFB);
    pub const blue_200: Color = hex(0xFF90CAF9);
    pub const blue_300: Color = hex(0xFF64B5F6);
    pub const blue_400: Color = hex(0xFF42A5F5);
    pub const blue_500: Color = hex(0xFF2196F3);
    pub const blue_600: Color = hex(0xFF1E88E5);
    pub const blue_700: Color = hex(0xFF1976D2);
    pub const blue_800: Color = hex(0xFF1565C0);
    pub const blue_900: Color = hex(0xFF0D47A1);

    pub const blue_accent: Color = blue_accent_200;
    pub const blue_accent_100: Color = hex(0xFF82B1FF);
    pub const blue_accent_200: Color = hex(0xFF448AFF);
    pub const blue_accent_400: Color = hex(0xFF2979FF);
    pub const blue_accent_700: Color = hex(0xFF2962FF);

    pub const light_blue: Color = light_blue_500;
    pub const light_blue_50: Color = hex(0xFFE1F5FE);
    pub const light_blue_100: Color = hex(0xFFB3E5FC);
    pub const light_blue_200: Color = hex(0xFF81D4FA);
    pub const light_blue_300: Color = hex(0xFF4FC3F7);
    pub const light_blue_400: Color = hex(0xFF29B6F6);
    pub const light_blue_500: Color = hex(0xFF03A9F4);
    pub const light_blue_600: Color = hex(0xFF039BE5);
    pub const light_blue_700: Color = hex(0xFF0288D1);
    pub const light_blue_800: Color = hex(0xFF0277BD);
    pub const light_blue_900: Color = hex(0xFF01579B);

    pub const light_blue_accent: Color = light_blue_accent_200;
    pub const light_blue_accent_100: Color = hex(0xFF80D8FF);
    pub const light_blue_accent_200: Color = hex(0xFF40C4FF);
    pub const light_blue_accent_400: Color = hex(0xFF00B0FF);
    pub const light_blue_accent_700: Color = hex(0xFF0091EA);

    pub const cyan: Color = cyan_500;
    pub const cyan_50: Color = hex(0xFFE0F7FA);
    pub const cyan_100: Color = hex(0xFFB2EBF2);
    pub const cyan_200: Color = hex(0xFF80DEEA);
    pub const cyan_300: Color = hex(0xFF4DD0E1);
    pub const cyan_400: Color = hex(0xFF26C6DA);
    pub const cyan_500: Color = hex(0xFF00BCD4);
    pub const cyan_600: Color = hex(0xFF00ACC1);
    pub const cyan_700: Color = hex(0xFF0097A7);
    pub const cyan_800: Color = hex(0xFF00838F);
    pub const cyan_900: Color = hex(0xFF006064);

    pub const cyan_accent: Color = cyan_accent_200;
    pub const cyan_accent_100: Color = hex(0xFF84FFFF);
    pub const cyan_accent_200: Color = hex(0xFF18FFFF);
    pub const cyan_accent_400: Color = hex(0xFF00E5FF);
    pub const cyan_accent_700: Color = hex(0xFF00B8D4);

    pub const teal: Color = teal_500;
    pub const teal_50: Color = hex(0xFFE0F2F1);
    pub const teal_100: Color = hex(0xFFB2DFDB);
    pub const teal_200: Color = hex(0xFF80CBC4);
    pub const teal_300: Color = hex(0xFF4DB6AC);
    pub const teal_400: Color = hex(0xFF26A69A);
    pub const teal_500: Color = hex(0xFF009688);
    pub const teal_600: Color = hex(0xFF00897B);
    pub const teal_700: Color = hex(0xFF00796B);
    pub const teal_800: Color = hex(0xFF00695C);
    pub const teal_900: Color = hex(0xFF004D40);

    pub const teal_accent: Color = teal_accent_200;
    pub const teal_accent_100: Color = hex(0xFFA7FFEB);
    pub const teal_accent_200: Color = hex(0xFF64FFDA);
    pub const teal_accent_400: Color = hex(0xFF1DE9B6);
    pub const teal_accent_700: Color = hex(0xFF00BFA5);

    pub const green: Color = green_500;
    pub const green_50: Color = hex(0xFFE8F5E9);
    pub const green_100: Color = hex(0xFFC8E6C9);
    pub const green_200: Color = hex(0xFFA5D6A7);
    pub const green_300: Color = hex(0xFF81C784);
    pub const green_400: Color = hex(0xFF66BB6A);
    pub const green_500: Color = hex(0xFF4CAF50);
    pub const green_600: Color = hex(0xFF43A047);
    pub const green_700: Color = hex(0xFF388E3C);
    pub const green_800: Color = hex(0xFF2E7D32);
    pub const green_900: Color = hex(0xFF1B5E20);

    pub const green_accent: Color = green_accent_200;
    pub const green_accent_100: Color = hex(0xFFB9F6CA);
    pub const green_accent_200: Color = hex(0xFF69F0AE);
    pub const green_accent_400: Color = hex(0xFF00E676);
    pub const green_accent_700: Color = hex(0xFF00C853);

    pub const light_green: Color = light_green_500;
    pub const light_green_50: Color = hex(0xFFF1F8E9);
    pub const light_green_100: Color = hex(0xFFDCEDC8);
    pub const light_green_200: Color = hex(0xFFC5E1A5);
    pub const light_green_300: Color = hex(0xFFAED581);
    pub const light_green_400: Color = hex(0xFF9CCC65);
    pub const light_green_500: Color = hex(0xFF8BC34A);
    pub const light_green_600: Color = hex(0xFF7CB342);
    pub const light_green_700: Color = hex(0xFF689F38);
    pub const light_green_800: Color = hex(0xFF558B2F);
    pub const light_green_900: Color = hex(0xFF33691E);

    pub const light_green_accent: Color = light_green_accent_200;
    pub const light_green_accent_100: Color = hex(0xFFCCFF90);
    pub const light_green_accent_200: Color = hex(0xFFB2FF59);
    pub const light_green_accent_400: Color = hex(0xFF76FF03);
    pub const light_green_accent_700: Color = hex(0xFF64DD17);

    pub const lime: Color = lime_500;
    pub const lime_50: Color = hex(0xFFF9FBE7);
    pub const lime_100: Color = hex(0xFFF0F4C3);
    pub const lime_200: Color = hex(0xFFE6EE9C);
    pub const lime_300: Color = hex(0xFFDCE775);
    pub const lime_400: Color = hex(0xFFD4E157);
    pub const lime_500: Color = hex(0xFFCDDC39);
    pub const lime_600: Color = hex(0xFFC0CA33);
    pub const lime_700: Color = hex(0xFFAFB42B);
    pub const lime_800: Color = hex(0xFF9E9D24);
    pub const lime_900: Color = hex(0xFF827717);

    pub const lime_accent: Color = lime_accent_200;
    pub const lime_accent_100: Color = hex(0xFFF4FF81);
    pub const lime_accent_200: Color = hex(0xFFEEFF41);
    pub const lime_accent_400: Color = hex(0xFFC6FF00);
    pub const lime_accent_700: Color = hex(0xFFAEEA00);

    pub const yellow: Color = yellow_500;
    pub const yellow_50: Color = hex(0xFFFFFDE7);
    pub const yellow_100: Color = hex(0xFFFFF9C4);
    pub const yellow_200: Color = hex(0xFFFFF59D);
    pub const yellow_300: Color = hex(0xFFFFF176);
    pub const yellow_400: Color = hex(0xFFFFEE58);
    pub const yellow_500: Color = hex(0xFFFFEB3B);
    pub const yellow_600: Color = hex(0xFFFDD835);
    pub const yellow_700: Color = hex(0xFFFBC02D);
    pub const yellow_800: Color = hex(0xFFF9A825);
    pub const yellow_900: Color = hex(0xFFF57F17);

    pub const yellow_accent: Color = yellow_accent_200;
    pub const yellow_accent_100: Color = hex(0xFFFFFF8D);
    pub const yellow_accent_200: Color = hex(0xFFFFFF00);
    pub const yellow_accent_400: Color = hex(0xFFFFEA00);
    pub const yellow_accent_700: Color = hex(0xFFFFD600);

    pub const amber: Color = amber_500;
    pub const amber_50: Color = hex(0xFFFFF8E1);
    pub const amber_100: Color = hex(0xFFFFECB3);
    pub const amber_200: Color = hex(0xFFFFE082);
    pub const amber_300: Color = hex(0xFFFFD54F);
    pub const amber_400: Color = hex(0xFFFFCA28);
    pub const amber_500: Color = hex(0xFFFFC107);
    pub const amber_600: Color = hex(0xFFFFB300);
    pub const amber_700: Color = hex(0xFFFFA000);
    pub const amber_800: Color = hex(0xFFFF8F00);
    pub const amber_900: Color = hex(0xFFFF6F00);

    pub const amber_accent: Color = amber_accent_200;
    pub const amber_accent_100: Color = hex(0xFFFFE57F);
    pub const amber_accent_200: Color = hex(0xFFFFD740);
    pub const amber_accent_400: Color = hex(0xFFFFC400);
    pub const amber_accent_700: Color = hex(0xFFFFAB00);

    pub const orange: Color = orange_500;
    pub const orange_50: Color = hex(0xFFFFF3E0);
    pub const orange_100: Color = hex(0xFFFFE0B2);
    pub const orange_200: Color = hex(0xFFFFCC80);
    pub const orange_300: Color = hex(0xFFFFB74D);
    pub const orange_400: Color = hex(0xFFFFA726);
    pub const orange_500: Color = hex(0xFFFF9800);
    pub const orange_600: Color = hex(0xFFFB8C00);
    pub const orange_700: Color = hex(0xFFF57C00);
    pub const orange_800: Color = hex(0xFFEF6C00);
    pub const orange_900: Color = hex(0xFFE65100);

    pub const orange_accent: Color = orange_accent_200;
    pub const orange_accent_100: Color = hex(0xFFFFD180);
    pub const orange_accent_200: Color = hex(0xFFFFAB40);
    pub const orange_accent_400: Color = hex(0xFFFF9100);
    pub const orange_accent_700: Color = hex(0xFFFF6D00);

    pub const deep_orange: Color = deep_orange_500;
    pub const deep_orange_50: Color = hex(0xFFFBE9E7);
    pub const deep_orange_100: Color = hex(0xFFFFCCBC);
    pub const deep_orange_200: Color = hex(0xFFFFAB91);
    pub const deep_orange_300: Color = hex(0xFFFF8A65);
    pub const deep_orange_400: Color = hex(0xFFFF7043);
    pub const deep_orange_500: Color = hex(0xFFFF5722);
    pub const deep_orange_600: Color = hex(0xFFF4511E);
    pub const deep_orange_700: Color = hex(0xFFE64A19);
    pub const deep_orange_800: Color = hex(0xFFD84315);
    pub const deep_orange_900: Color = hex(0xFFBF360C);

    pub const deep_orange_accent: Color = deep_orange_accent_200;
    pub const deep_orange_accent_100: Color = hex(0xFFFF9E80);
    pub const deep_orange_accent_200: Color = hex(0xFFFF6E40);
    pub const deep_orange_accent_400: Color = hex(0xFFFF3D00);
    pub const deep_orange_accent_700: Color = hex(0xFFDD2C00);

    pub const brown: Color = brown_500;
    pub const brown_50: Color = hex(0xFFEFEBE9);
    pub const brown_100: Color = hex(0xFFD7CCC8);
    pub const brown_200: Color = hex(0xFFBCAAA4);
    pub const brown_300: Color = hex(0xFFA1887F);
    pub const brown_400: Color = hex(0xFF8D6E63);
    pub const brown_500: Color = hex(0xFF795548);
    pub const brown_600: Color = hex(0xFF6D4C41);
    pub const brown_700: Color = hex(0xFF5D4037);
    pub const brown_800: Color = hex(0xFF4E342E);
    pub const brown_900: Color = hex(0xFF3E2723);

    pub const grey: Color = grey_500;
    pub const grey_50: Color = hex(0xFFFAFAFA);
    pub const grey_100: Color = hex(0xFFF5F5F5);
    pub const grey_200: Color = hex(0xFFEEEEEE);
    pub const grey_300: Color = hex(0xFFE0E0E0);
    pub const grey_350: Color = hex(0xFFD6D6D6);
    pub const grey_400: Color = hex(0xFFBDBDBD);
    pub const grey_500: Color = hex(0xFF9E9E9E);
    pub const grey_600: Color = hex(0xFF757575);
    pub const grey_700: Color = hex(0xFF616161);
    pub const grey_800: Color = hex(0xFF424242);
    pub const grey_850: Color = hex(0xFF303030);
    pub const grey_900: Color = hex(0xFF212121);

    pub const blue_grey: Color = blue_grey_500;
    pub const blue_grey_50: Color = hex(0xFFECEFF1);
    pub const blue_grey_100: Color = hex(0xFFCFD8DC);
    pub const blue_grey_200: Color = hex(0xFFB0BEC5);
    pub const blue_grey_300: Color = hex(0xFF90A4AE);
    pub const blue_grey_400: Color = hex(0xFF78909C);
    pub const blue_grey_500: Color = hex(0xFF607D8B);
    pub const blue_grey_600: Color = hex(0xFF546E7A);
    pub const blue_grey_700: Color = hex(0xFF455A64);
    pub const blue_grey_800: Color = hex(0xFF37474F);
    pub const blue_grey_900: Color = hex(0xFF263238);

    pub fn rgb(r: u8, g: u8, b: u8) Color {
        return .{ .r = r, .g = g, .b = b, .a = 255 };
    }

    pub fn rgba(r: u8, g: u8, b: u8, a: u8) Color {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    pub fn hex(hex_color: u32) Color {
        const a: u8 = ((hex_color >> 24) & 0xFF);
        const r: u8 = ((hex_color >> 16) & 0xFF);
        const g: u8 = ((hex_color >> 8) & 0xFF);
        const b: u8 = (hex_color & 0xFF);

        return .{ .r = r, .g = g, .b = b, .a = a };
    }
};

pub const Size = struct {
    width: f32 = 0,
    height: f32 = 0,
};

pub const Offset = struct {
    x: f32 = 0,
    y: f32 = 0,
};

pub const Rect = struct {
    x: f32 = 0,
    y: f32 = 0,
    width: f32 = 0,
    height: f32 = 0,

    pub fn contains(self: Rect, x: f32, y: f32) bool {
        return x >= self.x and y >= self.y and x < self.x + self.width and y < self.y + self.height;
    }

    pub fn inset(self: Rect, edges: EdgeInsets) Rect {
        return .{
            .x = self.x + edges.left,
            .y = self.y + edges.top,
            .width = @max(0, self.width - edges.left - edges.right),
            .height = @max(0, self.height - edges.top - edges.bottom),
        };
    }
};

pub const Constraints = struct {
    min_width: f32 = 0,
    max_width: f32 = std.math.floatMax(f32),
    min_height: f32 = 0,
    max_height: f32 = std.math.floatMax(f32),

    pub fn tight(width: f32, height: f32) Constraints {
        return .{ .min_width = width, .max_width = width, .min_height = height, .max_height = height };
    }

    pub fn loose(width: f32, height: f32) Constraints {
        return .{ .max_width = width, .max_height = height };
    }

    pub fn constrain(self: Constraints, size: Size) Size {
        return .{
            .width = std.math.clamp(size.width, self.min_width, self.max_width),
            .height = std.math.clamp(size.height, self.min_height, self.max_height),
        };
    }

    pub fn deflate(self: Constraints, edges: EdgeInsets) Constraints {
        const horizontal = edges.left + edges.right;
        const vertical = edges.top + edges.bottom;
        return .{
            .min_width = @max(0, self.min_width - horizontal),
            .max_width = @max(0, self.max_width - horizontal),
            .min_height = @max(0, self.min_height - vertical),
            .max_height = @max(0, self.max_height - vertical),
        };
    }
};

pub const EdgeInsets = struct {
    left: f32 = 0,
    top: f32 = 0,
    right: f32 = 0,
    bottom: f32 = 0,

    pub const zero: EdgeInsets = .{};

    pub fn all(value: f32) EdgeInsets {
        return .{ .left = value, .top = value, .right = value, .bottom = value };
    }

    pub fn symmetric(horizontal_value: f32, vertical_value: f32) EdgeInsets {
        return .{ .left = horizontal_value, .right = horizontal_value, .top = vertical_value, .bottom = vertical_value };
    }

    pub fn horizontal(self: EdgeInsets) f32 {
        return self.left + self.right;
    }

    pub fn vertical(self: EdgeInsets) f32 {
        return self.top + self.bottom;
    }
};

test "rect contains points" {
    const rect = Rect{ .x = 10, .y = 20, .width = 50, .height = 30 };
    try std.testing.expect(rect.contains(10, 20));
    try std.testing.expect(rect.contains(59, 49));
    try std.testing.expect(!rect.contains(60, 50));
}
