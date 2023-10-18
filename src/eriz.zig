//! Data structures and procedures ... with good documentation.

const std = @import("std");

pub const btree = @import("btree.zig");
pub const math = @import("math.zig");

test {
    std.testing.refAllDecls(btree);
    std.testing.refAllDecls(math);
}

const Allocator = std.mem.Allocator;
const Arena = std.heap.ArenaAllocator;
const Accumulator = math.Accumulator(f64);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = false }){};
    const alloc = gpa.allocator();

    const args = try std.process.argsAlloc(alloc);
    defer std.process.argsFree(alloc, args);

    if (args.len < 3) return error.NotEnoughArgs;
    const element = args[1][0..args[1].len];
    const n = try std.fmt.parseUnsigned(u64, args[2][0..args[2].len], 10);
    if (n == 0) return error.InvalidPayloadLength;

    const order = if (args.len >= 4)
        try std.fmt.parseUnsigned(u16, args[3][0..args[3].len], 10)
    else
        0;

    if (args.len > 4) return error.TooManyArgs;

    if (std.mem.eql(u8, element, "int")) {
        const Context = btree.AutoContext(i32);
        return switch (order) {
            8 => setBenchmarks(BTree(i32, 8, Context), i32, alloc, n),
            16 => setBenchmarks(BTree(i32, 16, Context), i32, alloc, n),
            32 => setBenchmarks(BTree(i32, 32, Context), i32, alloc, n),
            64 => setBenchmarks(BTree(i32, 64, Context), i32, alloc, n),
            128 => setBenchmarks(BTree(i32, 128, Context), i32, alloc, n),
            else => error.InvalidBTreeOrder,
        };
    } else if (std.mem.eql(u8, element, "String32")) {
        return switch (order) {
            8 => setBenchmarks(BTree([32]u8, 8, String32Context), [32]u8, alloc, n),
            16 => setBenchmarks(BTree([32]u8, 16, String32Context), [32]u8, alloc, n),
            32 => setBenchmarks(BTree([32]u8, 32, String32Context), [32]u8, alloc, n),
            64 => setBenchmarks(BTree([32]u8, 64, String32Context), [32]u8, alloc, n),
            128 => setBenchmarks(BTree([32]u8, 128, String32Context), [32]u8, alloc, n),
            else => error.InvalidBTreeOrder,
        };
    } else {
        return error.InvalidElementType;
    }
}

fn BTree(comptime T: type, comptime order_: usize, comptime Context: type) type {
    return struct {
        const order = order_;

        b3: btree.BTree(.{ .Element = T, .slots_per_node = order }),
        ctx: Context,
        alloc: Allocator,

        const Self = @This();

        fn init(allocator: Allocator) Self {
            return .{ .b3 = .{}, .ctx = .{}, .alloc = allocator };
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

fn randomize(comptime T: type, random: *std.rand.Random, output: *T) !void {
    if (T == i32) {
        output.* = random.int(i32);
    } else if (T == [32]u8) {
        random.bytes(output);
    } else {
        @compileError("randomize not implemented for type " ++ @typeName(T));
    }
}

fn benchmark(repetitions: u32, context: anytype) !Accumulator {
    var acc = Accumulator{};
    for (0..repetitions) |_| {
        var begin: i128 = undefined;
        var end: i128 = undefined;
        try context.code(&begin, &end);
        const elapsed_micros = @as(f64, @floatFromInt(end - begin)) / 1e3;
        acc.add(elapsed_micros);
    }
    return acc;
}

fn setBenchmarks(comptime Set: type, comptime Element: type, allocator: Allocator, n: u64) !void {
    const b = Set.order;
    const element_name = @typeName(Element);

    const n2 = n * 2;
    const n_for_30_reps = 32_768;
    const repetitions: u32 = @intCast(@max(30, 30 * n_for_30_reps / n));

    const buffer = try allocator.alloc(Element, n2);
    defer allocator.free(buffer);

    const random_seed = 0;
    var prng = std.rand.DefaultPrng.init(random_seed);
    var random = prng.random();
    for (buffer) |*x| try randomize(Element, &random, x);
    const xs = buffer[0..n];
    const xs2 = buffer[0..n2];

    var arena = Arena.init(allocator);
    defer arena.deinit();

    const upsert = try benchmark(
        repetitions,
        struct {
            xs: []const Element,
            arena: *Arena,
            fn code(self: @This(), begin: *i128, end: *i128) !void {
                var set = Set.init(self.arena.allocator());
                defer _ = self.arena.reset(.{ .retain_capacity = {} });

                begin.* = std.time.nanoTimestamp();
                for (self.xs) |x| try set.upsert(x);
                end.* = std.time.nanoTimestamp();
            }
        }{ .xs = xs, .arena = &arena },
    );
    try printBenchmark(b, element_name, "upsert", n, repetitions, upsert);

    const lookup_find = try benchmark(
        repetitions,
        struct {
            xs: []const Element,
            arena: *Arena,
            fn code(self: @This(), begin: *i128, end: *i128) !void {
                var set = Set.init(self.arena.allocator());
                defer _ = self.arena.reset(.{ .retain_capacity = {} });

                for (self.xs) |x| try set.upsert(x);

                begin.* = std.time.nanoTimestamp();
                for (self.xs) |x| {
                    if (!set.contains(x)) return error.NotFound;
                }
                end.* = std.time.nanoTimestamp();
            }
        }{ .xs = xs, .arena = &arena },
    );
    try printBenchmark(b, element_name, "lookupFind", n, repetitions, lookup_find);

    const remove = try benchmark(
        repetitions,
        struct {
            xs: []const Element,
            arena: *Arena,
            fn code(self: @This(), begin: *i128, end: *i128) !void {
                var set = Set.init(self.arena.allocator());
                defer _ = self.arena.reset(.{ .retain_capacity = {} });

                for (self.xs) |x| try set.upsert(x);

                begin.* = std.time.nanoTimestamp();
                for (self.xs) |x| set.remove(x);
                end.* = std.time.nanoTimestamp();
            }
        }{ .xs = xs, .arena = &arena },
    );
    try printBenchmark(b, element_name, "remove", n, repetitions, remove);

    const lookup_fail = try benchmark(repetitions, struct {
        xs: []const Element,
        xs2: []const Element,
        arena: *Arena,
        fn code(self: @This(), begin: *i128, end: *i128) !void {
            var set = Set.init(self.arena.allocator());
            defer _ = self.arena.reset(.{ .retain_capacity = {} });

            for (self.xs2) |x| try set.upsert(x);

            for (self.xs) |x| set.remove(x);

            begin.* = std.time.nanoTimestamp();
            for (self.xs) |x| {
                if (set.contains(x)) return error.Found;
            }
            end.* = std.time.nanoTimestamp();
        }
    }{ .xs = xs, .xs2 = xs2, .arena = &arena });
    try printBenchmark(b, element_name, "lookupFail", n, repetitions, lookup_fail);
}

fn printBenchmark(
    comptime b: usize,
    comptime element: []const u8,
    comptime operation: []const u8,
    n: u64,
    reps: u64,
    op: Accumulator,
) !void {
    const stdout = std.io.getStdOut().writer();
    const fmt = comptime "B={d}" ++
        "\tT=" ++ element ++
        "\tOP=" ++ operation ++
        "\tn={d}" ++
        "\treps={d}" ++
        "\tzmin={d:.3}" ++
        "\tzmax={d:.3}" ++
        "\ttime={d:.3}\n";
    const avg = op.mean();
    const stddev = @sqrt(op.variance());
    const z_min = (op.min() - avg) / stddev;
    const z_max = (op.max() - avg) / stddev;
    const nanos_per_element = op.min() * 1e3 / @as(f64, @floatFromInt(n));
    try stdout.print(fmt, .{ b, n, reps, z_min, z_max, nanos_per_element });
}
