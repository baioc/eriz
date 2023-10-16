//! Generic B-tree data structure (not a B+ tree).
//!
//! In general, [B-trees](https://en.wikipedia.org/wiki/B-tree) are optimized for "cache
//! friendliness" and low overhead per stored element.
//! Their main tradeoff w.r.t. other self-balancing trees is that **any insertion or
//! deletion may move multiple elements in memory, invalidating references and iterators**.
//! See also: `BTree.Cursor`.
//!
//! **When elements are big, consider keeping them elsewhere and storing only "handles" in the
//! tree**; a custom comparator can then be provided in order to sort handles according to the
//! order of the elements themselves.
//! See also: `BTree`.
//!
//! Modern usage of the name "B-tree" for a data structure is usually (and especially in the
//! context of databases) referring to a [B+ tree](https://en.wikipedia.org/wiki/B%2B_tree);
//! in general, iterating over all elements in a B+ tree is faster than in a B-tree.
//! However, B+ trees store multiple copies of keys, and may keep using these copies as part of an
//! index after the corresponding element has been deleted.
//! In an unmanaged environment, storing extra keys may lead to issues with memory (de)allocation,
//! making it hard to sort elements residing outside the tree (as in the "handles" example above).
//! **Not needing extra key copies** is the main advantage of an in-memory B-tree over a B+ tree
//! (and if you're always storing value types as keys, a modern
//! [radix tree](https://en.wikipedia.org/wiki/Radix_tree) is probably faster anyway).

const std = @import("std");
const math = std.math;
const mem = std.mem;
const testing = std.testing;
const Allocator = mem.Allocator;
const assert = std.debug.assert;

/// Static parameters used to generate a custom B-tree.
pub const Config = struct {
    /// Type of elements being stored in the tree.
    ///
    /// XXX: If a custom struct is used here, it must be either an `extern struct`
    /// or an ABI-sized `packed struct`.
    Element: type,

    /// Whether to use a binary search inside each node. Leave `null` in order
    /// to have the implementation choose based on the max number of elements per node.
    use_binary_search: ?bool = null,

    /// Desired number of elements in a full node. Leave `null` in order to have the implementation
    /// choose a default based on either `bytes_per_node` or `@sizeOf(Element)`.
    slots_per_node: ?usize = null,

    /// Ensures every allocation (i.e; node) fits inside a block of this many bytes.
    ///
    /// If this config is set, and `slots_per_node` is not, then `slots_per_node` will be set to
    /// the maximum number allowed by this config in order to reduce memory waste.
    /// Otherwise, if both are set, this config just adds an upper bound to `slots_per_node`.
    bytes_per_node: ?usize = null,

    /// Overrides the default alignment of the array field containing each node's elements.
    /// Does not change the alignment of elements themselves.
    slots_alignment: ?usize = null,
};

/// Generates a custom B-tree with the given configuration.
///
/// The resulting data structure stores values of type `Element`, sorted according to
/// the total order defined by a "context" parameter used in lookups and insertions.
/// This context can be any struct type with at least one member function:
/// - `cmp(Self, Element, Element) std.math.Order`
///
/// The `lookup` method allows "adapated contexts" to be used in order to search for
/// any type `Key` without constructing an `Element`:
/// - `cmp(Self, Key, Element) std.math.Order`
///
/// Correctness of the B-tree requires these contexts to provide a consistent total order.
/// See also: `AutoContext`
///
/// Unless otherwise stated, methods take time proportional to the height of the tree, which
/// is logarithmic on the number of elements with a base equal to the order of the B-tree
/// (i.e; the number of slots per node). Space complexity is O(1) by default.
pub fn BTree(comptime config: Config) type {
    const Element = config.Element;
    const slots_alignment = config.slots_alignment orelse @alignOf(Element);
    const n = blk: {
        const slots_per_node = xxx: {
            if (config.slots_per_node) |requested_slots| break :xxx requested_slots;
            if (config.bytes_per_node) |bytes| break :xxx maxSlotsPerNode(Element, bytes, slots_alignment);
            break :xxx defaultSlotsPerNode(Element);
        };
        if (slots_per_node < 2) @compileError("B-tree nodes need at least 2 slots");
        if (slots_per_node > math.maxInt(u31)) @compileError("too many slots per node");
        if (config.bytes_per_node) |bytes_per_node| {
            const node_size = @sizeOf(InternalNodeTemplate(Element, slots_per_node, slots_alignment));
            if (node_size > bytes_per_node) @compileError("incompatible number of slots and max bytes per node");
        }
        break :blk slots_per_node;
    };
    const bsearch_threshold = 2 * defaultSlotsPerNode(u64);

    return struct {
        /// Whether or not lookups will use a binary intra-node search.
        pub const use_binary_search: bool = config.use_binary_search orelse (n > bsearch_threshold);

        // TODO: fill padding space in leaf nodes with extra slots
        /// Number of element slots in each branch node.
        pub const slots_per_node: u32 = n;

        /// Maximum block size the tree will ever request per allocation.
        pub const bytes_per_node: usize = @max(@sizeOf(ExternalNode), @sizeOf(InternalNode));

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
            children: [n + 1]?NodeHandle = [_]?NodeHandle{null} ** (n + 1),
            slots: [n]Element align(slots_alignment) = undefined,

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

        /// Reference to a specific slot in a node; used to navigate the B-tree.
        ///
        /// When a cursor is first created by a B-tree, it is always **valid** (i.e; it points to
        /// valid memory containing the expected element). Any mutating operation on the B-tree
        /// may make the cursor **invalid**, so it still points to the same memory location, but
        /// that memory might have been freed or may now hold an entirely different element.
        /// Therefore, **dereferencing an invalid cursor leads to U.B.**
        ///
        /// Methods have O(1) space and time complexity, unless stated otherwise.
        pub const Cursor = struct {
            node: NodeHandle,
            index: u32,

            /// Gets the address of the element being referenced by this cursor
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
            ///
            /// This method takes constant time when the cursor stays in the same leaf node,
            /// but may need to climb the entire height of the tree in the worst case.
            /// Still, **iterating over all `N` elements has time complexity in O(N)**.
            pub fn next(cursor: *Cursor) ?*Element {
                assert(cursor.index < cursor.node.slots_in_use);
                defer assert(cursor.index < cursor.node.slots_in_use);

                // at branch nodes, the next slot is the leftmost of the >subtree
                if (cursor.node.asInternal()) |branch| {
                    cursor.node = branch.children[cursor.index + 1].?;
                    subtreeMin(cursor);
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
                while (node.parent) |parent| : (node = parent.handle()) {
                    const index = node.index_in_parent;
                    assert(parent.children[index] == node);
                    if (index < parent.header.slots_in_use) {
                        cursor.node = parent.handle();
                        cursor.index = index;
                        return cursor.get();
                    }
                }
                return null; // already at max
            }

            /// Tries to move the cursor to the previous element in the tree and get its address.
            /// When it can't, this returns `null` and does NOT mutate the cursor.
            ///
            /// This method takes constant time when the cursor stays in the same leaf node,
            /// but may need to climb the entire height of the tree in the worst case.
            /// Still, **iterating over all `N` elements has time complexity in O(N)**.
            pub fn prev(cursor: *Cursor) ?*Element {
                assert(cursor.index < cursor.node.slots_in_use);
                defer assert(cursor.index < cursor.node.slots_in_use);

                // at branch nodes, the previous slot is the rightmost of the <subtree
                if (cursor.node.asInternal()) |branch| {
                    cursor.node = branch.children[cursor.index].?;
                    subtreeMax(cursor);
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
                while (node.parent) |parent| : (node = parent.handle()) {
                    const index = node.index_in_parent;
                    assert(parent.children[index] == node);
                    if (index > 0) {
                        cursor.node = parent.handle();
                        cursor.index = index - 1;
                        return cursor.get();
                    }
                }
                return null; // already at min
            }
        };

        const Tree = @This();

        /// Returns the number of elements currently stored in the tree, in constant time.
        pub fn len(tree: Tree) usize {
            return tree.total_in_use;
        }

        /// Creates a cursor at the root node of the tree, pointing to any occupied slot.
        /// Returns `null` if the tree is empty.
        /// Takes constant time.
        pub fn root(tree: Tree) ?Cursor {
            return if (tree.root_) |r| .{ .node = r, .index = 0 } else null;
        }

        /// Returns a cursor to the smallest element in the tree, or `null` if empty.
        pub fn min(tree: Tree) ?Cursor {
            var cursor = tree.root() orelse return null;
            subtreeMin(&cursor);
            return cursor;
        }

        fn subtreeMin(cursor: *Cursor) void {
            while (cursor.node.asInternal()) |branch| {
                cursor.node = branch.children[0].?;
            }
            cursor.index = 0;
        }

        /// Returns a cursor to the biggest element in the tree, or `null` if empty.
        pub fn max(tree: Tree) ?Cursor {
            var cursor = tree.root() orelse return null;
            subtreeMax(&cursor);
            return cursor;
        }

        fn subtreeMax(cursor: *Cursor) void {
            while (cursor.node.asInternal()) |branch| {
                cursor.node = branch.children[branch.header.slots_in_use].?;
            }
            cursor.index = cursor.node.slots_in_use - 1;
        }

        // computes the leftmost index such that, by the context's total order,
        // `slots[0 .. idx) < key <= slots[idx .. slots_in_use)`; the returned struct
        // also indicates whether the key was actually found (`slots[idx] == key`).
        fn bisect(
            key: anytype,
            slots: *align(slots_alignment) const [n]Element,
            slots_in_use: u32,
            context: anytype,
        ) struct { index: u32, found: bool } {
            assert(slots_in_use <= n);
            if (comptime use_binary_search) {
                var begin: u32 = 0;
                var end: u32 = slots_in_use;
                while (begin < end) {
                    const mid = begin + (end - begin) / 2;
                    const cmp: math.Order = context.cmp(key, slots[mid]);
                    switch (cmp) {
                        .lt => end = mid,
                        .eq => return .{ .found = true, .index = mid },
                        .gt => begin = mid + 1,
                    }
                }
                return .{ .found = false, .index = begin };
            } else { // use linear search
                var i: u32 = 0;
                while (i < slots_in_use) : (i += 1) {
                    const cmp: math.Order = context.cmp(key, slots[i]);
                    switch (cmp) {
                        .gt => continue,
                        .eq => return .{ .found = true, .index = i },
                        .lt => break,
                    }
                }
                return .{ .found = false, .index = i };
            }
            // TODO: check disassembly and implement @Vector'ized version if needed
        }

        /// Either a valid cursor at the element being looked up (if found), or an insertion hint.
        ///
        /// While the not-found case should NEVER be dereferenced, it can be used in `insert` as
        /// long as it is not invalidated by other updates (same as all `Cursor`s).
        pub const LookupResult = union(enum) { found: Cursor, not_found: InsertionHint };
        const InsertionHint = ?Cursor;

        /// Searches for an element in the tree. See also: `LookupResult`.
        ///
        /// This function accepts keys of any type `K` such that `K` can be used with the total
        /// order implemented by `context`. In order words, the following should compile:
        /// `var ord: std.math.Order = context.cmp(key, x);` where `x` is an element in the tree.
        ///
        /// If duplicates are present and an element was found, the resulting cursor is placed
        /// at the "first" (leftmost) element in the tree which compares equal to the search key.
        pub fn lookup(tree: Tree, key: anytype, context: anytype) LookupResult {
            var node = tree.root_ orelse return .{ .not_found = null };
            while (node.asInternal()) |branch| {
                const search = bisect(key, &branch.slots, branch.header.slots_in_use, context);
                if (search.found) {
                    const cursor = Cursor{ .node = node, .index = search.index };
                    return .{ .found = cursor };
                }
                node = branch.children[search.index].?;
            }
            const leaf = node.asExternal().?;
            const search = bisect(key, &leaf.slots, leaf.header.slots_in_use, context);
            const cursor = Cursor{ .node = node, .index = search.index };
            return if (search.found) .{ .found = cursor } else .{ .not_found = cursor };
        }

        fn shiftElement(slots: *[n]Element, begin: u32, end: u32, x: Element) void {
            mem.copyBackwards(Element, slots[begin + 1 .. end + 1], slots[begin..end]);
            slots[begin] = x;
        }

        fn unshiftElement(slots: *[n]Element, begin: u32, end: u32) void {
            mem.copyForwards(Element, slots[begin .. end - 1], slots[begin + 1 .. end]);
        }

        fn reparent(child: NodeHandle, parent: *InternalNode, index: u32) void {
            parent.children[index] = child;
            child.parent = parent;
            child.index_in_parent = index;
        }

        fn shiftChild(parent: *InternalNode, index: u32, child: NodeHandle) void {
            var i = parent.header.slots_in_use;
            while (i >= index + 1) : (i -= 1) {
                reparent(parent.children[i - 1].?, parent, i);
            }
            reparent(child, parent, index);
        }

        fn unshiftChild(parent: *InternalNode, index: u32) void {
            for (index..parent.header.slots_in_use) |i| {
                reparent(parent.children[i + 1].?, parent, @intCast(i));
            }
        }

        /// Error set of `insert` operations.
        pub const InsertError = error{UnrecoverableOom} || Allocator.Error;

        /// Inserts an element in the tree using the result of a previous lookup.
        ///
        /// If the lookup result indicates that the element was already in the tree,
        /// **this procedure will insert a duplicate**, otherwise it is a simple insertion.
        ///
        /// On success, returns a cursor placed over the just-inserted element.
        /// **This cursor is the only one guaranteed to be valid after this operation**.
        ///
        /// NOTE: Some allocation failures are unrecoverable errors (`UnrecoverableOom`),
        /// meaning that the B-tree may be left in an inconsistent state, which can lead to
        /// further undefined behavior, crashes and/or memory leaks.
        /// If this happens, the best course of action is to stop using the tree immediately.
        /// Even better: make sure the allocator can provide for the worst case scenario of
        /// `h + 1` new node allocations, where `h` is the current height of the B-tree, and
        /// unrecoverable errors will never happen.
        pub fn insert(
            tree: *Tree,
            allocator: Allocator,
            position: LookupResult,
            element: Element,
            context: anytype,
        ) InsertError!Cursor {
            // validate given hint and extract a precise insertion location from it
            const start: struct { node: *ExternalNode, index: u32 } = switch (position) {
                .not_found => |nf| blk: {
                    if (nf) |cursor| { // if not found, we should be at a leaf node
                        break :blk .{ .node = cursor.node.asExternal().?, .index = cursor.index };
                    } else if (tree.root_ == null) { // (which could be an empty root)
                        const new_root = try allocator.create(ExternalNode);
                        new_root.* = .{};
                        tree.root_ = new_root.handle();
                    }
                    break :blk .{ .node = tree.root_.?.asExternal().?, .index = 0 };
                },
                .found => |cursor| blk: { // someone wants duplicates ...
                    if (cursor.node.asExternal()) |leaf| {
                        break :blk .{ .node = leaf, .index = cursor.index };
                    } else if (cursor.node.asInternal()) |parent| {
                        // at branch nodes, pretend the element wasn't found and finish the search
                        var node = parent.children[cursor.index].?;
                        while (node.asInternal()) |branch| {
                            const search = bisect(element, &branch.slots, branch.header.slots_in_use, context);
                            node = branch.children[search.index].?;
                        }
                        const leaf = node.asExternal().?;
                        const search = bisect(element, &leaf.slots, leaf.header.slots_in_use, context);
                        break :blk .{ .node = leaf, .index = search.index };
                    }
                    unreachable;
                },
            };

            const leaf = start.node;
            assert(start.index <= leaf.header.slots_in_use);

            // easy case: insert without split
            if (leaf.header.slots_in_use < slots_per_node) {
                shiftElement(&leaf.slots, start.index, leaf.header.slots_in_use, element);
                leaf.header.slots_in_use += 1;
                tree.total_in_use += 1;
                return .{ .node = leaf.handle(), .index = start.index };
            }

            // hard case: insert with a split at the leaf node ...
            const new_leaf = try allocator.create(ExternalNode);
            new_leaf.* = .{};
            tree.total_in_use += 1;
            const leaf_cursor = splitLeaf(leaf, new_leaf, element, start.index);
            // NOTE: after this point, OOM is an irrecoverable error

            // then, move the median element of the just-split subtree up to its parent
            // in order to add the newly-created child pointer. this may trigger splits
            // recursively as we go up the tree. it stops when there are no overflows or
            // after splitting the root. also note that this process may move the element
            // we're inserting, and we want to keep track of it for the return cursor
            var node = leaf.handle();
            var new_sibling = new_leaf.handle();
            const median_index = leaf.header.slots_in_use - 1;
            var median = leaf.slots[median_index];
            leaf.header.slots_in_use -= 1;
            var child_cursor = leaf_cursor;
            const inserted_at_median = (child_cursor.node == node and child_cursor.index == median_index);
            var return_child_cursor = !inserted_at_median;
            while (node.parent) |parent| {
                const index = node.index_in_parent;
                // NOTE: ^ no need to search the parent in order to choose the lifted key's position

                if (parent.header.slots_in_use < slots_per_node) { // no-overflow case
                    shiftElement(&parent.slots, index, parent.header.slots_in_use, median);
                    parent.header.slots_in_use += 1;
                    shiftChild(parent, index + 1, new_sibling);
                    const parent_cursor = Cursor{ .node = parent.handle(), .index = index };
                    return if (return_child_cursor) child_cursor else parent_cursor;
                }

                // recursive split case
                const new_branch = allocator.create(InternalNode) catch return error.UnrecoverableOom;
                new_branch.* = .{};
                const parent_cursor = splitBranch(parent, new_branch, median, new_sibling, index);

                // update loop variables
                node = parent.handle();
                new_sibling = new_branch.handle();
                const new_median_index = parent.header.slots_in_use - 1;
                median = parent.slots[new_median_index];
                parent.header.slots_in_use -= 1;
                child_cursor = if (return_child_cursor) child_cursor else parent_cursor;
                const inserted_at_new_median = (parent_cursor.node == node and parent_cursor.index == new_median_index);
                return_child_cursor = (return_child_cursor or !inserted_at_new_median);
            }

            // if we got here, it means that we've just split our old tree root, so
            // we need to add an upper level to the tree with a new root. the new root
            // contains two children (the split-up old root) and one element (median of the split)
            assert(node == tree.root_.?);
            const new_root = allocator.create(InternalNode) catch return error.UnrecoverableOom;
            new_root.* = .{};
            new_root.header.slots_in_use = 1;
            new_root.slots[0] = median;
            reparent(node, new_root, 0);
            reparent(new_sibling, new_root, 1);
            tree.root_ = new_root.handle();
            const root_cursor = Cursor{ .node = new_root.handle(), .index = 0 };
            return if (return_child_cursor) child_cursor else root_cursor;
        }

        fn splitLeaf(left: *ExternalNode, right: *ExternalNode, key: Element, pos: u32) Cursor {
            return splitImpl(ExternalNode, left, right, key, {}, pos);
        }

        fn splitBranch(left: *InternalNode, right: *InternalNode, key: Element, child: NodeHandle, pos: u32) Cursor {
            return splitImpl(InternalNode, left, right, key, child, pos);
        }

        fn splitImpl(comptime Node: type, left: *Node, right: *Node, key: Element, child: anytype, pos: u32) Cursor {
            assert(left.header.slots_in_use == slots_per_node);
            assert(right.header.slots_in_use == 0);
            assert(pos <= slots_per_node);
            const is_branch = (Node == InternalNode);

            var cursor: Cursor = undefined;
            const mid = slots_per_node / 2;

            if (pos <= mid) {
                // for an insertion in the left node, set up the right node first
                for (mid..slots_per_node) |i| {
                    right.slots[i - mid] = left.slots[i];
                    if (is_branch) reparent(left.children[i + 1].?, right, @intCast(i - mid + 1));
                }
                right.header.slots_in_use = slots_per_node - mid;
                // then do a normal insert on the left node
                shiftElement(&left.slots, pos, mid, key);
                left.header.slots_in_use = mid - 0 + 1;
                if (is_branch) shiftChild(left, pos + 1, child);
                cursor = .{ .node = left.handle(), .index = pos };
            } else {
                // for an insertion in the right node,
                // we need to add the pending element while we do the split
                var to: u32 = 0;
                for (mid + 1..pos) |i| {
                    right.slots[to] = left.slots[i];
                    if (is_branch) reparent(left.children[i + 1].?, right, to + 1);
                    to += 1;
                }
                right.slots[to] = key;
                if (is_branch) reparent(child, right, to + 1);
                cursor = .{ .node = right.handle(), .index = to };
                to += 1;
                for (pos..slots_per_node) |i| {
                    right.slots[to] = left.slots[i];
                    if (is_branch) reparent(left.children[i + 1].?, right, to + 1);
                    to += 1;
                }
                // then simply adjust slot use counts
                right.header.slots_in_use = @intCast(to);
                left.header.slots_in_use = mid + 1 - 0;
            }

            // there's one last detail when splitting internal nodes: since this was
            // a split, we know that the last slot of the left node will be pushed
            // upwards to its parent, but the child pointer to the right of that node
            // needs to be put somewhere; that somewhere is precisely the first
            // pointer slot of the just-created right node, which should be null now
            if (is_branch) {
                assert(right.children[0] == null);
                reparent(left.children[left.header.slots_in_use].?, right, 0);
            }

            return cursor;
        }

        /// Deallocates all nodes in the tree, using stack space proportional to its height.
        pub fn clear(tree: *Tree, allocator: Allocator) void {
            if (tree.root_) |r| deallocate(allocator, r);
            tree.* = .{};
        }

        fn deallocate(allocator: Allocator, node: NodeHandle) void {
            if (node.asExternal()) |leaf| {
                allocator.destroy(leaf);
            } else if (node.asInternal()) |branch| {
                for (branch.children[0 .. @as(u32, branch.header.slots_in_use) + 1]) |child| {
                    deallocate(allocator, child.?);
                }
                allocator.destroy(branch);
            } else {
                unreachable;
            }
        }

        const min_load = slots_per_node / 2;

        /// Deletes the element referenced by the given cursor.
        ///
        /// **This operation may invalidate any existing `Cursor`s, including the given one**.
        pub fn delete(tree: *Tree, allocator: Allocator, cursor: Cursor) void {
            // steal biggest element of left subtree when deleting at branch nodes
            if (cursor.node.asInternal()) |branch| {
                var prevCursor = cursor;
                const biggestOfLeftSubtree = prevCursor.prev().?;
                branch.slots[cursor.index] = biggestOfLeftSubtree.*;
                tree.delete(allocator, prevCursor);
                return;
            }

            // at leafs, do the unshift, then check for underflow
            var leaf = cursor.node.asExternal().?;
            unshiftElement(&leaf.slots, cursor.index, leaf.header.slots_in_use);
            leaf.header.slots_in_use -= 1;
            tree.total_in_use -= 1;
            if (leaf.header.slots_in_use >= min_load) return;

            // on underflow, start rebalancing at the leaf
            leaf = rebalanceLeaf(tree, allocator, leaf);
            var node = leaf.header.parent;
            while (node) |branch| {
                if (branch.header.slots_in_use >= min_load) break;
                const possiblyMerged = rebalanceBranch(tree, allocator, branch);
                node = possiblyMerged.header.parent;
            }
        }

        fn rebalanceLeaf(tree: *Tree, allocator: Allocator, leaf: *ExternalNode) *ExternalNode {
            return rebalanceImpl(ExternalNode, tree, allocator, leaf);
        }

        fn rebalanceBranch(tree: *Tree, allocator: Allocator, branch: *InternalNode) *InternalNode {
            return rebalanceImpl(InternalNode, tree, allocator, branch);
        }

        fn rebalanceImpl(comptime Node: type, tree: *Tree, allocator: Allocator, node: *Node) *Node {
            assert(node.header.slots_in_use < min_load);
            if (node.header.parent == null) {
                // this early return means that we never deallocate the root, even if it is
                // empty after a deletion. we keep it pre-allocated and only free in clear()
                assert(node.handle() == tree.root_);
                return node;
            }

            const is_branch = (Node == InternalNode);

            const parent = node.header.parent.?;
            const child_index = node.header.index_in_parent;
            assert(parent.children[child_index].? == node.handle());

            const left_sibling: ?*Node = if (child_index > 0)
                @alignCast(@ptrCast(parent.children[child_index - 1].?))
            else
                null;

            const right_sibling: ?*Node = if (child_index < parent.header.slots_in_use)
                @alignCast(@ptrCast(parent.children[child_index + 1].?))
            else
                null;

            if (existsAndHasSpare(Node, right_sibling)) |right| { // counter-clockwise rotation
                const separator_index = child_index;
                node.slots[node.header.slots_in_use] = parent.slots[separator_index];
                node.header.slots_in_use += 1;
                parent.slots[separator_index] = right.slots[0];
                unshiftElement(&right.slots, 0, right.header.slots_in_use);
                if (is_branch) {
                    const sparePointer = right.children[0].?;
                    unshiftChild(right, 0);
                    shiftChild(node, node.header.slots_in_use, sparePointer);
                }
                right.header.slots_in_use -= 1;
                return node;
            } else if (existsAndHasSpare(Node, left_sibling)) |left| { // clockwise rotation
                const separator_index = child_index - 1;
                shiftElement(&node.slots, 0, node.header.slots_in_use, parent.slots[separator_index]);
                node.header.slots_in_use += 1;
                parent.slots[separator_index] = left.slots[left.header.slots_in_use - 1];
                if (is_branch) {
                    shiftChild(node, 0, left.children[left.header.slots_in_use].?);
                }
                left.header.slots_in_use -= 1;
                return node;
            } else { // merge two sibling nodes and delete the rightmost
                var left: *Node = undefined;
                var right: *Node = undefined;
                var separator_index: u32 = undefined;
                if (right_sibling != null) {
                    left = node;
                    right = right_sibling.?;
                    separator_index = child_index;
                } else {
                    assert(left_sibling != null);
                    left = left_sibling.?;
                    right = node;
                    separator_index = child_index - 1;
                }
                assert(left.header.slots_in_use + right.header.slots_in_use < slots_per_node);

                // separator element in parent moves down
                left.slots[left.header.slots_in_use] = parent.slots[separator_index];
                left.header.slots_in_use += 1;
                unshiftElement(&parent.slots, separator_index, parent.header.slots_in_use);
                unshiftChild(parent, separator_index + 1);
                parent.header.slots_in_use -= 1;

                // merge left <- right
                for (0..right.header.slots_in_use) |i| {
                    left.slots[left.header.slots_in_use + i] = right.slots[i];
                }
                if (is_branch) {
                    for (0..right.header.slots_in_use + 1) |i| {
                        reparent(right.children[i].?, left, @intCast(left.header.slots_in_use + i));
                    }
                }
                left.header.slots_in_use += right.header.slots_in_use;
                allocator.destroy(right);

                // if the root just became empty, delete it and make the tree
                // shallower by using the merged node as the new root
                if (parent.handle() == tree.root_ and parent.header.slots_in_use == 0) {
                    allocator.destroy(parent);
                    tree.root_ = left.handle();
                    tree.root_.?.parent = null;
                }

                return left;
            }
        }

        fn existsAndHasSpare(comptime Node: type, node: ?*Node) ?*Node {
            if (node == null) return null;
            return if (node.?.header.slots_in_use > min_load) node.? else null;
        }
    };
}

test "BTree: ordered set using AutoContext" {
    // tip: debug w/ visualizer at https://www.cs.usfca.edu/~galles/visualization/BTree.html
    inline for ([_]bool{ true, false }) |bsearch| {
        const B3 = BTree(.{ .Element = i32, .slots_per_node = 3, .use_binary_search = bsearch });

        const alloc = testing.allocator;
        const ctx = AutoContext(i32){};
        var set = B3{};
        const payload = [_]i32{
            34, 33, 38,
            28, 27, 22,
            30, 21, 24,
            18, 19, 20,
            26, 32, 42,
            23,
        };

        // deferred cleaning + empty test
        defer {
            set.clear(alloc);
            assert(set.len() == 0);
            for (payload) |x| {
                assert(switch (set.lookup(x, ctx)) {
                    .not_found => true,
                    .found => false,
                });
            }
        }

        // lookup / insert test
        try testing.expectEqual(@as(usize, 0), set.len());
        for (payload) |element| {
            const pre_insert_lookup = set.lookup(element, ctx);
            switch (pre_insert_lookup) {
                .found => return error.FoundElementNotInserted,
                .not_found => {},
            }

            const cursor = try set.insert(alloc, pre_insert_lookup, element, ctx);
            try testing.expectEqual(element, cursor.get().*);

            const post_insert_lookup = set.lookup(element, ctx);
            switch (post_insert_lookup) {
                .found => |c| try testing.expectEqual(cursor, c),
                .not_found => return error.InsertedElementNotFound,
            }
        }

        // testing both ordered iterators
        // also an insertion sanity check: b-tree can't come up with values not in the payload
        try testing.expectEqual(payload.len, set.len());
        var asc = set.min().?;
        var desc = set.max().?;
        var iterated_length: usize = 0;
        while (true) {
            const asc_count = mem.count(i32, &payload, &[_]i32{asc.get().*});
            const desc_count = mem.count(i32, &payload, &[_]i32{desc.get().*});
            try testing.expectEqual(@as(usize, 1), asc_count);
            try testing.expectEqual(@as(usize, 1), desc_count);
            iterated_length += 1;
            const asc_next = asc.next();
            const desc_prev = desc.prev();
            try testing.expectEqual(asc_next == null, desc_prev == null);
            if (asc_next == null) break;
        }
        try testing.expectEqual(set.len(), iterated_length);

        // deletion test removes all but the last 3 elements, in reverse order
        const n = payload.len;
        for (1..n - 2) |i| {
            const element = payload[n - i];

            const pre_delete_lookup = set.lookup(element, ctx);
            switch (pre_delete_lookup) {
                .found => |cursor| {
                    try testing.expectEqual(element, cursor.get().*);
                    set.delete(alloc, cursor);
                },
                .not_found => return error.InsertedElementNotFound,
            }

            const post_delete_lookup = set.lookup(element, ctx);
            switch (post_delete_lookup) {
                .found => return error.FoundDeletedElement,
                .not_found => {},
            }
        }
    }
}

test "BTree: ordered multiset with adapted lookup context" {
    const S = extern struct { key: i32, discriminator: u32 };
    const DefaultContext = struct {
        fn cmp(_: @This(), lhs: S, rhs: S) math.Order {
            return math.order(lhs.key, rhs.key);
        }
    };
    const AdaptedKeyContext = struct {
        fn cmp(_: @This(), lhs: i32, rhs: S) math.Order {
            return math.order(lhs, rhs.key);
        }
    };

    const alloc = testing.allocator;
    var multiset = BTree(.{ .Element = S }){};
    defer multiset.clear(alloc);
    const payload = [_]i32{ 1, 2, 4, 5, 6, 4 };

    try testing.expectEqual(@as(usize, 0), multiset.len());
    var count: u32 = 0;
    for (payload) |key| {
        // notice that we perform the lookup by key only, BEFORE creating the element
        const hint = multiset.lookup(key, AdaptedKeyContext{});
        const element = S{ .key = key, .discriminator = count };
        // the ordering context provided to insert needs to deal with elements, not keys
        const cursor = try multiset.insert(alloc, hint, element, DefaultContext{});
        count += 1;
        try testing.expectEqual(element, cursor.get().*);
    }
    try testing.expectEqual(payload.len, multiset.len());

    // delete both duplicates with key = 4
    switch (multiset.lookup(4, AdaptedKeyContext{})) {
        .found => |cursor| multiset.delete(alloc, cursor),
        .not_found => return error.InsertedElementNotFound,
    }
    try testing.expectEqual(payload.len - 1, multiset.len());
    switch (multiset.lookup(4, AdaptedKeyContext{})) {
        .found => |cursor| multiset.delete(alloc, cursor),
        .not_found => return error.InsertedElementNotFound,
    }
    switch (multiset.lookup(4, AdaptedKeyContext{})) {
        .found => return error.FoundDeletedElement,
        .not_found => {},
    }
    try testing.expectEqual(payload.len - 2, multiset.len());
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
        switch (math.order(bytes, max_bytes)) {
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
        children: [n + 1]?*NodeHeaderTemplate,
        slots: [n]Element align(slots_alignment),
    };
}

/// Creates a total ordering context for built-in types.
///
/// NOTE: This does NOT implement a total order for floating-point types, so unless
/// `NaN`s are impossible, avoid using this with floats.
pub fn AutoContext(comptime T: type) type {
    return struct {
        const Self = @This();
        pub fn cmp(_: Self, lhs: T, rhs: T) math.Order {
            return math.order(lhs, rhs);
        }
    };
}

test "AutoContext: built-in types are sorted correctly" {
    const Order = math.Order;

    inline for ([_]type{ u8, i32, usize, f32, f64 }) |Number| {
        const two: Number = 2;
        const three: Number = 3;
        const ctx = AutoContext(Number){};
        try testing.expectEqual(Order.eq, ctx.cmp(two, two));
        try testing.expectEqual(Order.eq, ctx.cmp(three, three));
        try testing.expectEqual(Order.lt, ctx.cmp(two, three));
        try testing.expectEqual(Order.gt, ctx.cmp(three, two));
    }

    try testing.expectEqual(Order.gt, AutoContext(u64).cmp(.{}, math.maxInt(u64), 0));
    try testing.expectEqual(Order.lt, AutoContext(i32).cmp(.{}, math.minInt(i32), 0));
}
