//! Data structures and procedures ... with good documentation.

const std = @import("std");

pub const btree = @import("btree.zig");
pub const math = @import("math.zig");

test {
    std.testing.refAllDecls(btree);
    std.testing.refAllDecls(math);
}

const stdlib = @cImport({
    @cInclude("stdlib.h");
    @cInclude("stdio.h");
});
const Allocator = std.mem.Allocator;
const Accumulator = math.Accumulator(f64);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 3) return error.NotEnoughArgs;
    const element = args[1][0..args[1].len];
    const n = try std.fmt.parseUnsigned(u64, args[2][0..args[2].len], 10);
    if (n == 0) return error.InvalidPayloadLength;

    if (std.mem.eql(u8, element, "int")) {
        return setBenchmarks(BTree(i32, btree.AutoContext(i32)), i32, alloc, n);
    } else if (std.mem.eql(u8, element, "String32")) {
        return setBenchmarks(BTree([32]u8, String32Context), [32]u8, alloc, n);
    } else {
        return error.InvalidElementType;
    }
}

fn BTree(comptime T: type, comptime Context: type) type {
    return struct {
        b3: btree.BTree(.{ .Element = T }),
        ctx: Context,
        alloc: Allocator,

        const Self = @This();

        fn init(allocator: Allocator) Self {
            return .{ .b3 = .{}, .ctx = .{}, .alloc = allocator };
        }

        fn deinit(self: *Self) void {
            self.b3.clear(self.alloc);
        }

        fn len(self: Self) usize {
            return self.b3.len();
        }

        fn upsert(self: *Self, x: T) !void {
            const lookup = self.b3.lookup(x, self.ctx);
            switch (lookup) {
                .found => |cursor| cursor.get().* = x,
                .not_found => _ = try self.b3.insert(self.alloc, lookup, x, self.ctx),
            }
        }

        fn contains(self: Self, x: T) bool {
            return switch (self.b3.lookup(x, self.ctx)) {
                .found => true,
                .not_found => false,
            };
        }

        fn remove(self: *Self, x: T) void {
            const lookup = self.b3.lookup(x, self.ctx);
            switch (lookup) {
                .found => |cursor| self.b3.delete(self.alloc, cursor),
                .not_found => {},
            }
        }
    };
}

const String32Context = struct {
    pub fn cmp(_: @This(), lhs: [32]u8, rhs: [32]u8) std.math.Order {
        for (0..lhs.len) |i| {
            if (lhs[i] < rhs[i]) return .lt;
            if (lhs[i] > rhs[i]) return .gt;
        }
        return .eq;
    }
};

fn randomize(comptime T: type, output: *T) !void {
    if (T == i32) {
        output.* = stdlib.rand();
    } else if (T == [32]u8) {
        _ = stdlib.snprintf(output.ptr, output.len, "%031d", stdlib.rand());
    } else {
        @compileError("randomize not implemented for type " ++ @typeName(T));
    }
}

fn benchmark(context: anytype) !Accumulator {
    var acc = Accumulator{};
    for (0..30) |_| {
        var begin: i64 = undefined;
        var end: i64 = undefined;
        try context.code(&begin, &end);
        const elapsed = end - begin;
        acc.add(@floatFromInt(elapsed));
    }
    return acc;
}

fn setBenchmarks(comptime Set: type, comptime Element: type, allocator: Allocator, n: u64) !void {
    const n2 = n * 2;

    const buffer = try allocator.alloc(Element, n2);
    defer allocator.free(buffer);

    stdlib.srand(0);
    for (buffer) |*x| try randomize(Element, x);

    const xs = buffer[0..n];
    const xs2 = buffer[0..n2];

    const upsert = try benchmark(
        struct {
            xs: []const Element,
            allocator: Allocator,
            fn code(self: @This(), begin: *i64, end: *i64) !void {
                var set = Set.init(self.allocator);
                defer set.deinit();
                begin.* = std.time.microTimestamp();
                for (self.xs) |x| try set.upsert(x);
                end.* = std.time.microTimestamp();
            }
        }{ .xs = xs, .allocator = allocator },
    );

    const lookup_find = try benchmark(
        struct {
            xs: []const Element,
            allocator: Allocator,
            fn code(self: @This(), begin: *i64, end: *i64) !void {
                var set = Set.init(self.allocator);
                defer set.deinit();
                for (self.xs) |x| try set.upsert(x);
                begin.* = std.time.microTimestamp();
                for (self.xs) |x| {
                    if (!set.contains(x)) return error.NotFound;
                }
                end.* = std.time.microTimestamp();
            }
        }{ .xs = xs, .allocator = allocator },
    );

    var rss: usize = 0;
    const lookup_fail = try benchmark(struct {
        xs: []const Element,
        xs2: []const Element,
        allocator: Allocator,
        rss: *usize,
        fn code(self: @This(), begin: *i64, end: *i64) !void {
            var set = Set.init(self.allocator);
            defer set.deinit();
            for (self.xs2) |x| try set.upsert(x);
            self.rss.* = @max(self.rss.*, set.len());
            for (self.xs) |x| set.remove(x);
            begin.* = std.time.microTimestamp();
            for (self.xs) |x| {
                if (set.contains(x)) return error.Found;
            }
            end.* = std.time.microTimestamp();
        }
    }{ .xs = xs, .xs2 = xs2, .allocator = allocator, .rss = &rss });

    const remove = try benchmark(
        struct {
            xs: []const Element,
            allocator: Allocator,
            fn code(self: @This(), begin: *i64, end: *i64) !void {
                var set = Set.init(self.allocator);
                defer set.deinit();
                for (self.xs) |x| try set.upsert(x);
                begin.* = std.time.microTimestamp();
                for (self.xs) |x| set.remove(x);
                end.* = std.time.microTimestamp();
            }
        }{ .xs = xs, .allocator = allocator },
    );

    try printBenchmark(@typeName(Element), "upsert", n, upsert, rss);
    try printBenchmark(@typeName(Element), "lookupFind", n, lookup_find, rss);
    try printBenchmark(@typeName(Element), "lookupFail", n, lookup_fail, rss);
    try printBenchmark(@typeName(Element), "remove", n, remove, rss);
}

fn printBenchmark(
    comptime element: []const u8,
    comptime operation: []const u8,
    n: u64,
    op: Accumulator,
    rss: usize,
) !void {
    const stdout = std.io.getStdOut().writer();
    const fmt = comptime "container=eriz.BTree" ++
        "\telement=" ++ element ++
        "\toperation=" ++ operation ++
        "\tn={d}\tres={d}\tavg={d:.3}\tstd={d:.3}\tmin={d:.3}\tmax={d:.3}\n";
    try stdout.print(fmt, .{ n, rss, op.mean(), @sqrt(op.variance()), op.min(), op.max() });
}
