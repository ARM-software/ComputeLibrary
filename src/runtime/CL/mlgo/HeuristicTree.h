/*
 * Copyright (c) 2021 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef SRC_RUNTIME_CL_MLGO_HEURISTIC_TREE_H
#define SRC_RUNTIME_CL_MLGO_HEURISTIC_TREE_H

#include "arm_compute/core/Types.h"
#include "src/runtime/CL/mlgo/Common.h"

#include <map>
#include <memory>
#include <string>
#include <utility>

namespace arm_compute
{
namespace mlgo
{
/** Conditional ops */
enum class ConditionalOp
{
    EQ, /**< Equal */
    LT, /**< Less than */
    LE, /**< Less than or equal to */
    GT, /**< Greater than */
    GE, /**< Greater than or equal to */
};

/** A branch condition expression evaluating: feature op threshold */
struct Condition
{
    std::string   feature;   /**< Feature name */
    ConditionalOp op;        /**< Condtional op */
    float         threshold; /**< Threshold value */
};

/** GEMM Shape used for query */
struct GEMMShape
{
    unsigned int m; /**< Number of rows for the lhs matrix. Lhs matrix NOT transposed */
    unsigned int n; /**< Number of columns for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int k; /**< Number of rows for the rhs matrix. Rhs matrix NOT transposed */
    unsigned int b; /**< Batch size */
};

/** A binary decision tree based heuristic */
class HeuristicTree
{
public:
    using NodeID = size_t;
    using TreeID = size_t;
    using Index  = std::tuple<HeuristicType, std::string, DataType>;
    enum class NodeType
    {
        Branch,
        Leaf
    };
    struct Node
    {
        virtual NodeType type() const = 0;
        virtual ~Node()               = default;
    };

    struct BranchNode : public Node
    {
        BranchNode(NodeID id, Condition cond, NodeID t_node, NodeID f_node)
            : id{ id }, condition{ cond }, true_node{ t_node }, false_node{ f_node }
        {
        }
        NodeType type() const override
        {
            return NodeType::Branch;
        }
        NodeID    id;
        Condition condition;
        NodeID    true_node;
        NodeID    false_node;
    };

    template <typename T>
    struct LeafNode : public Node
    {
        LeafNode(NodeID id, T val)
            : id{ id }, value{ val }
        {
        }
        NodeType type() const override
        {
            return NodeType::Leaf;
        }
        NodeID id;
        T      value;
    };

public:
    /** Constructor */
    HeuristicTree();
    /** Constructor */
    HeuristicTree(TreeID id, HeuristicType h_type, const std::string &ip_target, DataType data_type);
    // Since the HeuristicTree is a handle that owns the the nodes, it is move-only
    /** Prevent copy construction */
    HeuristicTree(const HeuristicTree &) = delete;
    /** Prevent copy assignment */
    HeuristicTree &operator=(const HeuristicTree &) = delete;
    /** Move constructor */
    HeuristicTree(HeuristicTree &&other) noexcept = default;
    /** Move assignment */
    HeuristicTree &operator=(HeuristicTree &&other) = default;

    /** Query a leaf value given a gemm shape
     *
     * @tparam T           Leaf value type
     * @param shape        A @ref GEMMShape for the query
     * @return std::pair<bool, T> Outcome contains bool, signalling if the query succeeded or not
     */
    template <typename T>
    std::pair<bool, T> query(GEMMShape shape) const;

    /** Add a leaf node
     *
     * @tparam T  Leaf value type
     * @param id  Leaf node ID
     * @param leaf_value Leaf node value
     * @return bool  If the addition succeeded or not
     */
    template <typename T>
    bool add_leaf(NodeID id, T leaf_value);
    /** Add a branch node
     *
     * @param id  Branch node ID
     * @param cond Branch node @ref Condition
     * @param true_node  True node's ID
     * @param false_node  False node's ID
     * @return bool  If the addition succeeded or not
     */
    bool add_branch(NodeID id, Condition cond, NodeID true_node, NodeID false_node);

    /** Get tree ID
     * @return TreeID
     */
    TreeID id() const
    {
        return _id;
    }

    /** Get tree index
     * @return Index
     */
    Index index() const
    {
        return std::make_tuple(_heuristic_type, _ip_target, _data_type);
    }

    /** Check if tree is valid
     * @return bool
     */
    bool check();

private:
    static constexpr size_t _max_query_depth{ 1000 }; // Maximum depth of query
    static constexpr size_t _max_num_nodes{ 100000 }; // Maximum number of nodes contained by the tree
    static constexpr NodeID _root{ 0 };               // Root tree ID

private:
    bool check_if_structurally_correct() const;

private:
    TreeID        _id;                             /**< Heuristic tree ID */
    HeuristicType _heuristic_type;                 /**< Heuristic type */
    std::string   _ip_target;                      /**< IP target associated with the tree */
    DataType      _data_type;                      /**< Data type associated with the tree */
    std::map<NodeID, std::unique_ptr<Node>> _tree; /**< Tree representation */
};
} // namespace mlgo

} // namespace arm_compute

#endif //SRC_RUNTIME_CL_MLGO_HEURISTIC_TREE_H