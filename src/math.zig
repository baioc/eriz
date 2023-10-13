//! Miscellaneous helpers for numerical math.

const std = @import("std");
const math = std.math;
const testing = std.testing;
const assert = std.debug.assert;

/// Tracks a stream of numbers in order to compute some statistics in constant space.
pub fn Accumulator(comptime Number: type) type {
    if (!std.meta.trait.isFloat(Number)) {
        @compileError(@typeName(Number) ++ " is not a floating-point type");
    }
    return struct {
        count_: usize = 0,
        min_: Number = math.inf(Number),
        max_: Number = -math.inf(Number),
        mean_: Number = 0.0,
        m2: Number = 0.0,
        // ^ mean and M2 accumulators used in Welford's online variance calculation algorithm
        // ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

        const Self = @This();

        /// Accumulate a given (non-`NaN`) value.
        pub fn add(acc: *Self, x: Number) void {
            assert(!math.isNan(x));
            acc.count_ += 1;
            acc.min_ = if (x < acc.min_) x else acc.min_;
            acc.max_ = if (x > acc.max_) x else acc.max_;
            const delta = x - acc.mean_;
            acc.mean_ += delta / @as(Number, @floatFromInt(acc.count_));
            const delta2 = x - acc.mean_;
            acc.m2 += delta * delta2;
        }

        /// Number of accumulated values.
        pub fn count(acc: Self) usize {
            return acc.count_;
        }

        /// Smallest accumulated value.
        pub fn min(acc: Self) Number {
            assert(acc.count_ > 0);
            return acc.min_;
        }

        /// Biggest accumulated value.
        pub fn max(acc: Self) Number {
            assert(acc.count_ > 0);
            return acc.max_;
        }

        /// Average of accumulated values.
        pub fn mean(acc: Self) Number {
            assert(acc.count_ > 0);
            return acc.mean_;
        }

        /// Sample variance.
        pub fn variance(acc: Self) Number {
            assert(acc.count_ > 1);
            const n: Number = @floatFromInt(acc.count_ - 1);
            return acc.m2 / n;
        }

        /// Population variance.
        pub fn populationVariance(acc: Self) Number {
            assert(acc.count_ > 0);
            const n: Number = @floatFromInt(acc.count_);
            return acc.m2 / n;
        }
    };
}

test "Accumulator: correct statistics with both small and big values" {
    const big_numbers = [_]f64{ 1.000000004e9, 1.000000007e9, 1.000000013e9, 1.000000016e9 };
    const small_numbers = [_]f64{ 4, 7, 13, 16 }; // => same as big_numbers - 1e9

    var big = Accumulator(f64){};
    for (big_numbers) |x| big.add(x);

    var small = Accumulator(f64){};
    for (small_numbers) |x| small.add(x);

    const tol = @sqrt(math.floatEps(f64));
    try testing.expectEqual(small.count(), big.count());
    try testing.expectApproxEqRel(small.min() + 1e9, big.min(), tol);
    try testing.expectApproxEqRel(small.max() + 1e9, big.max(), tol);
    try testing.expectApproxEqRel(small.mean() + 1e9, big.mean(), tol);
    try testing.expectApproxEqRel(small.populationVariance(), big.populationVariance(), tol);
    try testing.expectApproxEqRel(small.variance(), big.variance(), tol);
}
