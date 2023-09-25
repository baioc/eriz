// TODO: top-level //! docs

const std = @import("std");
const assert = std.debug.assert;

/// Static parameters used to generate a custom B-tree.
pub const Config = struct {
    /// Type of elements being stored in the tree.
    Element: type,

    /// Whether or not to allow duplicate elements.
    allow_duplicates: bool = false, // TODO: actually apply this

    /// Whether to use a binary search inside each node. Leave `null` in order
    /// to have the implementation choose based on the max number of elements per node.
    use_binary_search: ?bool = null,

    /// Number of elements in a full node. Leave `null` in order to have the implementation
    /// choose a default based on either `desired_node_bytes` or `@sizeOf(Element)`.
    slots_per_node: ?usize = null,

    /// Ensures every allocation (i.e., node) fits inside a block of this many bytes.
    ///
    /// If this config is set, and `slots_per_node` is not, then `slots_per_node` will be set to
    /// the maximum number allowed by this config in order to reduce internal fragmentation.
    /// Otherwise, if both are set, this config just adds a comptime constraint to `slots_per_node`.
    desired_node_bytes: ?usize = null,

    /// Overrides the default alignment of the array field containing each node's elements.
    /// Does not change the alignment of elements themselves.
    slots_alignment: ?usize = null,
};

// TODO: document context "trait" (including pseudo-key type)
// TODO: check that all methods use constant space, document time complexity
/// Generates a custom B-tree with the given parameters.
pub fn BTree(comptime config: Config) type {
    const Element = config.Element;
    const slots_alignment = config.slots_alignment orelse @alignOf(Element);
    const n = blk: {
        // TODO: test compile-time errors
        const slots_per_node = xxx: {
            if (config.slots_per_node) |requested_slots| break :xxx requested_slots;
            if (config.desired_node_bytes) |bytes| break :xxx maxSlotsPerNode(Element, bytes, slots_alignment);
            break :xxx defaultSlotsPerNode(Element);
        };
        assert(slots_per_node >= 2); // B-tree nodes need at least 2 slots
        assert(slots_per_node <= std.math.maxInt(u31)); // implementation-defined limit
        if (config.desired_node_bytes) |desired_node_bytes| {
            const node_size = @sizeOf(InternalNodeTemplate(Element, slots_per_node, slots_alignment));
            assert(node_size <= desired_node_bytes);
        }
        break :blk slots_per_node;
    };
    const bsearch_threshold = 2 * defaultSlotsPerNode(u64);

    return struct {
        /// Number of element slots in each node.
        pub const slots_per_node: u31 = n;

        /// Whether or not lookups will use a binary search within each node.
        pub const use_binary_search: bool = config.use_binary_search orelse n > bsearch_threshold;

        // the B-tree itself is basically a fat pointer to the node root
        root_: ?NodeHandle = null,
        total_in_use: usize = 0,

        // every node is either an external (leaf) or an internal (branch) node
        const ExternalNode = extern struct {
            header: NodeHeader = .{ .is_internal = false },
            slots: [n]Element align(slots_alignment) = undefined,

            fn handle(self: *ExternalNode) NodeHandle {
                return &self.header;
            }
        };
        const InternalNode = extern struct {
            header: NodeHeader = .{ .is_internal = true },
            slots: [n]Element align(slots_alignment) = undefined,
            children: [n + 1]?NodeHandle = [_]?NodeHandle{null} ** (n + 1),

            fn handle(self: *InternalNode) NodeHandle {
                return &self.header;
            }
        };
        comptime {
            const Template = InternalNodeTemplate(Element, n, slots_alignment);
            assert(@sizeOf(InternalNode) == @sizeOf(Template));
        }

        // handle is a type-punned reference to any node in the tree
        // NOTE: as long as the header is only instantiated at offset 0 of BTree nodes
        const NodeHandle = *NodeHeader;
        const NodeHeader = packed struct {
            is_internal: bool,
            slots_in_use: u31 = 0,
            index_in_parent: u32 = undefined,
            parent: ?*InternalNode = null,

            fn asInternal(node: NodeHandle) ?*InternalNode {
                if (!node.is_internal) return null;
                comptime assert(@offsetOf(InternalNode, "header") == 0);
                return @alignCast(@ptrCast(node));
            }

            fn asExternal(node: NodeHandle) ?*ExternalNode {
                if (node.is_internal) return null;
                comptime assert(@offsetOf(ExternalNode, "header") == 0);
                return @alignCast(@ptrCast(node));
            }
        };
        comptime {
            assert(@sizeOf(NodeHeader) == @sizeOf(NodeHeaderTemplate));
        }

        // TODO: document cursor invalidation
        /// Reference to a specific slot in a node; used to navigate the B-tree.
        pub const Cursor = struct {
            node: NodeHandle,
            index: u31,

            /// Gets the address of the element being referenced by this cursor.
            pub fn get(cursor: Cursor) *Element {
                assert(cursor.node.slots_in_use > 0);
                assert(cursor.index < cursor.node.slots_in_use);
                if (cursor.node.asExternal()) |node| {
                    return &node.slots[cursor.index];
                } else if (cursor.node.asInternal()) |node| {
                    return &node.slots[cursor.index];
                }
                unreachable;
            }

            /// Tries to move the cursor to the next element in the tree and get its address.
            /// When it can't, this returns `null` and does NOT mutate the cursor.
            pub fn next(cursor: *Cursor) ?*Element {
                assert(cursor.index < cursor.node.slots_in_use);
                defer assert(cursor.index < cursor.node.slots_in_use);

                // at branch nodes, the next slot is always first in the >child
                if (cursor.node.asInternal()) |node| {
                    const rchild = node.children[cursor.index + 1].?;
                    cursor.node = rchild;
                    cursor.index = 0;
                    return cursor.get();
                }

                // at leaves, we'll first try to just move to the next in-bound slot
                assert(!cursor.node.is_internal);
                if (cursor.index < cursor.node.slots_in_use - 1) {
                    cursor.index += 1;
                    return cursor.get();
                }

                // otherwise, climb upwards until we find a branch with more >children
                var node = cursor.node;
                while (node.parent) |parent| {
                    const index = node.index_in_parent;
                    assert(parent.children[index] == node);
                    if (index < parent.header.slots_in_use) {
                        cursor.node = parent.handle();
                        cursor.index = @intCast(index);
                        return cursor.get();
                    }
                    node = parent.handle();
                }
                return null; // we're already at the biggest element in the tree
            }

            /// Tries to move the cursor to the previous element in the tree and get its address.
            /// When it can't, this returns `null` and does NOT mutate the cursor.
            pub fn prev(cursor: *Cursor) ?*Element {
                assert(cursor.index < cursor.node.slots_in_use);
                defer assert(cursor.index < cursor.node.slots_in_use);

                // at branch nodes, the previous slot is always last in the <child
                if (cursor.node.asInternal()) |node| {
                    const lchild = node.children[cursor.index].?;
                    cursor.node = lchild;
                    cursor.index = lchild.slots_in_use - 1;
                    return cursor.get();
                }

                // at leaves, we'll first try to just move to the previous in-bound slot
                assert(!cursor.node.is_internal);
                if (cursor.index > 0) {
                    cursor.index -= 1;
                    return cursor.get();
                }

                // otherwise, climb upwards until we find a branch with more <children
                var node = cursor.node;
                while (node.parent) |parent| {
                    const index = node.index_in_parent;
                    assert(parent.children[index] == node);
                    if (index > 0) {
                        cursor.node = parent.handle();
                        cursor.index = @intCast(index - 1);
                        return cursor.get();
                    }
                    node = parent.handle();
                }
                return null; // we're already at the smallest element in the tree
            }
        };

        const Tree = @This();

        /// Returns the number of elements currently stored in the tree.
        pub fn len(tree: Tree) usize {
            return tree.total_in_use;
        }

        /// Creates a cursor at the root node of the tree, pointing to any occupied slot.
        /// Returns `null` if the tree is empty.
        pub fn root(tree: Tree) ?Cursor {
            return if (tree.root_) |r| .{ .node = r, .index = 0 } else null;
        }

        /// Returns a cursor to the smallest element in the tree, or `null` if empty.
        pub fn min(tree: Tree) ?Cursor {
            var cursor = tree.root() orelse return null;
            while (cursor.node.asInternal()) |branch| {
                cursor.node = branch.children[0].?;
            }
            cursor.index = 0;
            return cursor;
        }

        /// Returns a cursor to the biggest element in the tree, or `null` if empty.
        pub fn max(tree: Tree) ?Cursor {
            var cursor = tree.root() orelse return null;
            while (cursor.node.asInternal()) |branch| {
                cursor.node = branch.children[branch.header.slots_in_use].?;
            }
            cursor.index = cursor.node.slots_in_use - 1;
            return cursor;
        }

        /// Either a cursor at the element being looked up (if `found`), or an insertion hint.
        /// Both of these results are subject to invalidation, same as all `Cursor`s.
        pub const LookupResult = union(enum) { found: Cursor, not_found: ?InsertionHint };
        const InsertionHint = struct { node: NodeHandle, index: u31 };

        /// Searches for an element in the tree. See also: `LookupResult`.
        ///
        /// This function accepts keys of any type `K` such that `K` can be used with the total
        /// order implemented by `context`. In order words, the following should compile:
        /// `var ord: std.math.Order = context.cmp(key, x)`, where `x` is an element in the tree.
        pub fn lookup(tree: Tree, key: anytype, context: anytype) LookupResult {
            // go down the tree (if not empty) while looking for the requested key
            var node = tree.root_ orelse return .{ .not_found = null };
            while (node.asInternal()) |branch| {
                const search = bisect(key, &branch.slots, node.slots_in_use, context);
                if (search.found) {
                    const cursor = Cursor{ .node = node, .index = search.index };
                    return .{ .found = cursor };
                }
            }

            // if we reach a leaf, we either find the key here or declare not found
            const leaf = node.asExternal().?;
            const search = bisect(key, &leaf.slots, node.slots_in_use, context);
            if (search.found) {
                const cursor = Cursor{ .node = node, .index = search.index };
                return .{ .found = cursor };
            } else {
                const hint = InsertionHint{ .node = node, .index = search.index };
                return .{ .not_found = hint };
            }
        }

        // computes the leftmost index such that, by the context's total order,
        // `slots[0 .. idx) < key <= slots[idx .. slots_in_use)`; the returned struct
        // also indicates whether the key was actually found (`slots[idx] == key`).
        fn bisect(
            key: anytype,
            slots: *align(slots_alignment) const [n]Element,
            slots_in_use: u31,
            context: anytype,
        ) struct { index: u31, found: bool } {
            assert(slots_in_use <= n);
            if (comptime use_binary_search) {
                var begin: u31 = 0;
                var end: u31 = slots_in_use;
                while (begin < end) {
                    const mid = begin + (end - begin) / 2;
                    const cmp: std.math.Order = context.cmp(key, slots[mid]);
                    switch (cmp) {
                        .lt => end = mid,
                        .eq => return .{ .found = true, .index = mid },
                        .gt => begin = mid + 1,
                    }
                }
                return .{ .found = false, .index = begin };
            } else { // use linear search
                var i: u31 = 0;
                while (i < slots_per_node) {
                    const cmp: std.math.Order = context.cmp(key, slots[i]);
                    switch (cmp) {
                        .gt => i += 1,
                        .eq => return .{ .found = true, .index = i },
                        .lt => break,
                    }
                }
                return .{ .found = false, .index = i };
            }
            // TODO: check assembly and implement @Vector'ized version if needed
        }

        // TODO: tree methods
        // ?insert
        // ?remove
        // ?clear
    };
}

// TODO: test implemented stuff, including node with max possible slots_per_node
test "BTree" {
    const BT = BTree(.{ .Element = i32 });
    var bt = BT{};
    var s = bt.min();
    _ = s.?.prev();
    var b = bt.max();
    _ = b.?.next();
    var r = bt.lookup(5, AutoContext(i32){});
    _ = r;
}

// same structure as an actual BTree, but with the wrong type of parent ptr
const NodeHeaderTemplate = packed struct {
    is_internal: bool,
    slots_in_use: u31 = 0,
    index_in_parent: u32 = undefined,
    parent: ?*anyopaque = null,
};
comptime {
    assert(@sizeOf(NodeHeaderTemplate) == 8 + @sizeOf(*anyopaque));
}

fn defaultSlotsPerNode(comptime Element: type) usize {
    const leaf_bytes = 256; // <- chosen after a few experiments in the original D implementation
    return @max(2, (leaf_bytes - @sizeOf(NodeHeaderTemplate)) / @sizeOf(Element));
}

fn maxSlotsPerNode(comptime Element: type, comptime max_bytes: usize, comptime slots_alignment: usize) usize {
    var begin: usize = 2;
    var end: usize = max_bytes;
    while (begin < end) {
        const n = begin + (end - begin) / 2;
        const bytes = @sizeOf(InternalNodeTemplate(Element, n, slots_alignment));
        switch (std.math.order(bytes, max_bytes)) {
            .gt => end = n,
            .lt => begin = n,
            .eq => return n,
        }
    }
    return begin;
}

fn InternalNodeTemplate(comptime Element: type, comptime n: usize, comptime slots_alignment: usize) type {
    return extern struct {
        header: NodeHeaderTemplate,
        slots: [n]Element align(slots_alignment),
        children: [n + 1]?*NodeHeaderTemplate,
    };
}

/// Creates a closure which can be used as a total order context in a `BTree`.
///
/// Implemented in terms of `std.math.order`.
/// NOTE: `std.math.order` does NOT implement a total order for floating-point types,
/// so it should be avoided whenever `NaN`s are going to be stored in an ordered container.
pub fn AutoContext(comptime T: type) type {
    return struct {
        pub fn cmp(self: @This(), lhs: T, rhs: T) std.math.Order {
            _ = self;
            return std.math.order(lhs, rhs);
        }
    };
}

test "AutoContext" {
    const Order = std.math.Order;

    inline for ([_]type{ u8, i32, usize, f32, f64 }) |Number| {
        const two: Number = 2;
        const three: Number = 3;
        const ctx = AutoContext(Number){};
        try std.testing.expectEqual(Order.eq, ctx.cmp(two, two));
        try std.testing.expectEqual(Order.eq, ctx.cmp(three, three));
        try std.testing.expectEqual(Order.lt, ctx.cmp(two, three));
        try std.testing.expectEqual(Order.gt, ctx.cmp(three, two));
    }

    try std.testing.expectEqual(Order.gt, AutoContext(u64).cmp(.{}, std.math.maxInt(u64), 0));
    try std.testing.expectEqual(Order.lt, AutoContext(i32).cmp(.{}, std.math.minInt(i32), 0));
}
