//! Generic B-list data structure: a mutable list implemented with a B-tree.
//!
//! Zig's `std.ArrayList` is a dynamically-sized array, where insertions and deletions at the
//! beginning or middle of the list require moving most of its elements in memory.
//! This data structure should provide better performance when modifying large lists,
//! since the B-tree's hybrid array/tree structure requires less moves on such operations.

const std = @import("std");
const mem = std.mem;
const testing = std.testing;
const Allocator = mem.Allocator;

const btree = @import("btree.zig");

/// Static parameters used to generate a custom B-list. Refer to the `btree` documentation.
pub const Config = struct {
    Element: type,
    slots_per_node: ?usize = null,
    bytes_per_node: ?usize = null,
};

/// Generates a custom B-list with the given configuration.
pub fn BList(comptime config: Config) type {
    const Element = config.Element;
    const BTree = btree.BTree(.{
        .Element = Element,
        .order_statistics = true,
        .slots_per_node = config.slots_per_node,
        .bytes_per_node = config.bytes_per_node,
    });

    return struct {
        const Cursor = BTree.Cursor;
        const InsertError = BTree.InsertError;

        b3: BTree = .{},

        const List = @This();

        /// Returns the number of elements currently stored in the list, in constant time.
        pub fn len(list: List) usize {
            return list.b3.len();
        }

        /// Looks up the N-th element in the list, or `null` if out of bounds.
        pub fn select(list: List, index: usize) ?Cursor {
            return list.b3.select(index);
        }

        /// Inserts an element at the position specified by `select(i)` operation.
        pub fn insert(
            list: *List,
            allocator: Allocator,
            position: ?Cursor,
            element: Element,
        ) InsertError!Cursor {
            return list.b3.insertUnsorted(allocator, position, element);
        }

        /// Appends an element to the end of the list.
        pub fn append(list: *List, allocator: Allocator, element: Element) InsertError!Cursor {
            return list.insert(allocator, null, element);
        }

        /// Clears the list using stack space logarithmic on the number of elements.
        pub fn clear(list: *List, allocator: Allocator) void {
            return list.b3.clear(allocator);
        }

        /// Deletes the element referenced by the given cursor from the list.
        pub fn delete(list: *List, allocator: Allocator, cursor: Cursor) void {
            return list.b3.delete(allocator, cursor);
        }
    };
}

test "BList: mutable list operations" {
    const List = BList(.{ .Element = i32 });
    const alloc = testing.allocator;
    var numbers = List{};
    defer numbers.clear(alloc);

    const payload = [_]i32{ -6, 0, 2, 3, 6, 7, 11 };
    const n = payload.len;

    // initially, size should be 0
    try testing.expectEqual(@as(usize, 0), numbers.len());

    // push numbers in order
    for (payload) |element| {
        const cursor = try numbers.append(alloc, element);
        try testing.expectEqual(element, cursor.get().*);
    }

    // len should be the same as in payload
    try testing.expectEqual(n, numbers.len());

    // check if the list's front is the expected value, then pop it and repeat
    for (0..n) |j| {
        const front = numbers.select(0);
        try testing.expect(front != null);
        try testing.expectEqual(payload[j], front.?.get().*);
        numbers.delete(alloc, front.?);
    }

    // after removals, size should be zero
    try testing.expectEqual(@as(usize, 0), numbers.len());
}
