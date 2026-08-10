// iOS builds compile miniaudio's implementation as Objective-C (it drives AVAudioSession
// there). Zig's per-file C flags can't switch the language (-x lands after the input file),
// so this .m wrapper pulls the same translation unit in under the ObjC compiler instead.
// Selected by build.zig for .ios targets only; every other platform compiles miniaudio.c.
#include "miniaudio.c"
