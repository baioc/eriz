// TODO: top-level //! docs, including Guides

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
    use_binary_search: ?bool = null, // TODO: actually apply this

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

// NOTE: only supposed to be instantiated at offset 0 of `BTree` nodes;
// this ensures we can use `NodeHandle` as a type-punned reference to any node.
const NodeHeader = packed struct {
    is_internal: bool,
    slots_in_use: u31 = 0,
    index_in_parent: u32 = undefined,
    parent: ?NodeHandle = null,
};
pub const NodeHandle = *NodeHeader;
comptime {
    assert(@sizeOf(NodeHeader) == 8 + @sizeOf(*anyopaque));
}

/// Generates a custom B-tree with the given parameters.
pub fn BTree(comptime config: Config) type {
    const Element = config.Element;
    const slots_alignment = config.slots_alignment orelse @alignOf(Element);
    const n: u31 = blk: {
        // TODO: test compile-time errors
        const slots_per_node = xxx: {
            if (config.slots_per_node) |requested_slots| {
                break :xxx requested_slots;
            } else if (config.desired_node_bytes) |max_bytes| {
                break :xxx maxSlotsPerNode(Element, max_bytes, slots_alignment);
            } else {
                break :xxx defaultSlotsPerNode(Element);
            }
        };
        assert(slots_per_node >= 2); // B-tree nodes need at least 2 slots
        assert(slots_per_node < std.math.maxInt(u31)); // implementation-defined limit
        if (config.desired_node_bytes) |desired_node_bytes| {
            const node_size = @sizeOf(InternalNodeTemplate(Element, slots_per_node, slots_alignment));
            assert(node_size <= desired_node_bytes);
        }
        break :blk slots_per_node;
    };

    return struct {
        const Self = @This();

        /// Number of element slots in each node.
        pub const slots_per_node: u31 = n;

        // the B-tree itself is basically a fat pointer to the node root
        root: ?NodeHandle = null,
        total_in_use: usize = 0,

        // and every node is either an external (leaf) or an internal (branch) node
        const ExternalNode = extern struct {
            header: NodeHeader = .{ .is_internal = false },
            slots: [n]Element align(slots_alignment) = undefined,

            fn handle(self: *ExternalNode) NodeHandle {
                return &self.header;
            }

            fn fromHandle(anynode: NodeHandle) ?*ExternalNode {
                if (anynode.is_internal) return null;
                comptime assert(@offsetOf(ExternalNode, "header") == 0);
                return @ptrCast(anynode);
            }
        };
        const InternalNode = extern struct {
            header: NodeHeader = .{ .is_internal = true },
            slots: [n]Element align(slots_alignment) = undefined,
            children: [n + 1]?NodeHandle = [_]?NodeHandle{null} ** (n + 1),

            fn handle(self: *InternalNode) NodeHandle {
                return &self.header;
            }

            fn fromHandle(anynode: NodeHandle) ?*InternalNode {
                if (!anynode.is_internal) return null;
                comptime assert(@offsetOf(InternalNode, "header") == 0);
                return @ptrCast(anynode);
            }
        };
        comptime { // check that our helper template actually represents this node's structure
            assert(@sizeOf(InternalNode) == @sizeOf(InternalNodeTemplate(Element, n, slots_alignment)));
        }

        /// Reference to a specific slot in a node; used to navigate the B-tree.
        ///
        /// TODO: document cursor invalidation
        pub const Cursor = struct {
            node: NodeHandle,
            index: u32,

            // TODO: pub cursor methods
            // ?get
            // ?right
            // ?left
            // ?down
            // ?up
        };

        /// Returns the number of elements currently stored in the tree.
        pub fn len(self: *Self) usize {
            return self.total_in_use;
        }

        // TODO: pub tree methods
        // ?iterate (get cursor to root)
        // ?lookup
        // ?insert
        // ?remove
        // ?clear
    };
}

test { // TODO: name test
    const BT = BTree(.{ .Element = i32 });
    var bt = BT{};
    _ = bt;
}

fn defaultSlotsPerNode(comptime Element: type) usize {
    const leaf_bytes = 256; // <- chosen after a few experiments in the original D implementation
    return @max(2, (leaf_bytes - @sizeOf(NodeHeader)) / @sizeOf(Element));
}

fn maxSlotsPerNode(comptime Element: type, comptime max_bytes: usize, comptime slots_alignment: usize) usize {
    var begin: usize = 2;
    var end: usize = max_bytes;
    while (begin < end) {
        const n = begin + (end - begin) / 2;
        const bytes = @sizeOf(InternalNodeTemplate(Element, n, slots_alignment));
        if (bytes > max_bytes) end = n;
        if (bytes < max_bytes) begin = n;
        if (bytes == max_bytes) return n;
    }
    unreachable;
}

fn InternalNodeTemplate(comptime Element: type, comptime n: usize, comptime slots_alignment: usize) type {
    return extern struct {
        header: NodeHeader,
        slots: [n]Element align(slots_alignment),
        children: [n + 1]NodeHandle,
    };
}