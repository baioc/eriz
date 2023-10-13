//! Data structures and procedures ... with good documentation.

const std = @import("std");

pub const btree = @import("btree.zig");
pub const math = @import("math.zig");

test {
    std.testing.refAllDecls(btree);
    std.testing.refAllDecls(math);
}
