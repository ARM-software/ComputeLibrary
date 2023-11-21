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
#include "src/runtime/CL/mlgo/HeuristicTree.h"

#include "arm_compute/core/Log.h"

#include "support/Cast.h"

#include <algorithm>
#include <deque>
#include <set>
namespace arm_compute
{
namespace mlgo
{
namespace
{
bool evaluate(GEMMShape shape, Condition cond)
{
    // PRE: all features and ConditionalOps are valid
    constexpr float eps = 0.0001f;
    // Calculate all secondary features
    std::vector<std::pair<std::string, float>> cond_values{
        {"m", static_cast<float>(shape.m)},
        {"n", static_cast<float>(shape.n)},
        {"k", static_cast<float>(shape.k)},
        {"b", static_cast<float>(shape.b)},
        {"r_mn", static_cast<float>(shape.m) / shape.n},
        {"r_mk", static_cast<float>(shape.m) / shape.k},
        {"r_nk", static_cast<float>(shape.n) / shape.k},
        {"r_mnk", static_cast<float>(shape.m) / (static_cast<float>(shape.n) / shape.k)},
        {"workload", (static_cast<float>(shape.m) * shape.n * shape.b) / 20.0}};
    auto cond_value_pair_it =
        std::find_if(cond_values.begin(), cond_values.end(),
                     [&cond](decltype(*cond_values.begin()) it) { return it.first == cond.feature; });

    ARM_COMPUTE_ERROR_ON(cond_value_pair_it == cond_values.end());
    const float cond_value = cond_value_pair_it->second;
    switch (cond.op)
    {
        case ConditionalOp::LT:
        {
            return cond_value < cond.threshold;
        }
        case ConditionalOp::LE:
        {
            return cond_value <= cond.threshold;
        }
        case ConditionalOp::GT:
        {
            return cond_value > cond.threshold;
        }
        case ConditionalOp::GE:
        {
            return cond_value >= cond.threshold;
        }
        case ConditionalOp::EQ:
        default:
        {
            return std::abs(cond_value - cond.threshold) < eps;
        }
    }
}

} // namespace

constexpr size_t                HeuristicTree::_max_num_nodes;
constexpr size_t                HeuristicTree::_max_query_depth;
constexpr HeuristicTree::NodeID HeuristicTree::_root;

HeuristicTree::HeuristicTree() : HeuristicTree(0, HeuristicType::GEMM_Type, "", DataType::F32)
{
}

HeuristicTree::HeuristicTree(TreeID id, HeuristicType h_type, const std::string &ip_target, DataType data_type)
    : _id{id}, _heuristic_type{h_type}, _ip_target{ip_target}, _data_type{data_type}, _tree{}
{
}

template <typename T>
std::pair<bool, T> HeuristicTree::query(GEMMShape shape) const
{
    // Root ID = 0;
    auto   cur_node = _tree.at(_root).get();
    size_t depth    = 0;
    while (cur_node->type() != NodeType::Leaf)
    {
        if (depth > _max_query_depth)
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Exceeding max query depth: %zu. Is the tree too deep?",
                                                      _max_query_depth);
            return std::make_pair(false, T{});
        }
        ARM_COMPUTE_ERROR_ON_MSG(cur_node->type() != NodeType::Branch, "Unexpected NodeType");
        auto br_node = utils::cast::polymorphic_downcast<BranchNode *>(cur_node);
        if (evaluate(shape, br_node->condition))
        {
            cur_node = _tree.at(br_node->true_node).get();
        }
        else
        {
            cur_node = _tree.at(br_node->false_node).get();
        }
        ++depth;
    }
    ARM_COMPUTE_ERROR_ON_MSG(cur_node->type() != NodeType::Leaf, "Unexpected NodeType");
    auto l_node = utils::cast::polymorphic_downcast<LeafNode<T> *>(cur_node);
    return std::make_pair(true, l_node->value);
}

template <typename T>
bool HeuristicTree::add_leaf(NodeID id, T val)
{
    if (_tree.size() >= _max_num_nodes)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Exceeding the maximum number of nodes allowed %zu", _max_num_nodes);
        return false;
    }
    if (_tree.find(id) != _tree.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Cannot add node; node id %zu already exists", id);
        return false;
    }
    _tree[id] = std::make_unique<LeafNode<T>>(id, val);
    return true;
}

bool HeuristicTree::add_branch(NodeID id, Condition cond, NodeID t_node, NodeID f_node)
{
    if (_tree.size() >= _max_num_nodes)
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Exceeding the maximum number of nodes allowed %zu", _max_num_nodes);
        return false;
    }

    const std::set<std::string> supported_features = {"m", "n", "k", "b", "r_mn", "r_mk", "r_nk", "r_mnk", "workload"};
    const auto                  orig_feature       = cond.feature;
    std::transform(cond.feature.begin(), cond.feature.end(), cond.feature.begin(),
                   [](char c) { return std::tolower(c); });
    if (supported_features.find(cond.feature) == supported_features.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Unsupported feature %s", orig_feature.c_str());
        return false;
    }

    if (_tree.find(id) != _tree.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Cannot add node; node id %zu already exists", id);
        return false;
    }
    _tree[id] = std::make_unique<BranchNode>(id, cond, t_node, f_node);
    return true;
}

bool HeuristicTree::check_if_structurally_correct() const
{
    std::set<NodeID>   visited;
    std::deque<NodeID> to_visit{_root};

    while (!to_visit.empty())
    {
        auto id = to_visit.front();
        to_visit.pop_front();
        if (_tree.find(id) == _tree.end())
        {
            ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Missing node %zu", id);
            return false;
        }
        auto not_seen_before = visited.insert(id);
        if (!not_seen_before.second)
        {
            ARM_COMPUTE_LOG_INFO_MSG_CORE("Not a tree; contains cycles or loops");
            return false;
        }
        auto cur_node = _tree.at(id).get();
        if (cur_node->type() == NodeType::Branch)
        {
            auto br_node = utils::cast::polymorphic_downcast<BranchNode *>(cur_node);
            to_visit.push_back(br_node->true_node);
            to_visit.push_back(br_node->false_node);
        }
    }
    if (visited.size() != _tree.size())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Contains disjoint nodes");
        return false;
    }
    return true;
}

bool HeuristicTree::check()
{
    if (_tree.empty())
    {
        ARM_COMPUTE_LOG_INFO_MSG_CORE("Empty tree encountered");
        return false;
    }
    if (_tree.find(_root) == _tree.end())
    {
        ARM_COMPUTE_LOG_INFO_MSG_WITH_FORMAT_CORE("Missing root. Root must have a Node ID of %zu", _root);
        return false;
    }
    return check_if_structurally_correct();
}

/** Explicit template instantiation @relates HeuristicTree */
template std::pair<bool, GEMMType> HeuristicTree::query<GEMMType>(GEMMShape shape) const;
/** Explicit template instantiation @relates HeuristicTree */
template std::pair<bool, GEMMConfigNative> HeuristicTree::query<GEMMConfigNative>(GEMMShape shape) const;
/** Explicit template instantiation @relates HeuristicTree */
template std::pair<bool, GEMMConfigReshapedOnlyRHS>
HeuristicTree::query<GEMMConfigReshapedOnlyRHS>(GEMMShape shape) const;
/** Explicit template instantiation @relates HeuristicTree */
template std::pair<bool, GEMMConfigReshaped> HeuristicTree::query<GEMMConfigReshaped>(GEMMShape shape) const;

/** Explicit template instantiation @relates HeuristicTree */
template bool HeuristicTree::add_leaf(NodeID id, GEMMType val);
/** Explicit template instantiation @relates HeuristicTree */
template bool HeuristicTree::add_leaf(NodeID id, GEMMConfigNative val);
/** Explicit template instantiation @relates HeuristicTree */
template bool HeuristicTree::add_leaf(NodeID id, GEMMConfigReshapedOnlyRHS val);
/** Explicit template instantiation @relates HeuristicTree */
template bool HeuristicTree::add_leaf(NodeID id, GEMMConfigReshaped val);

} // namespace mlgo

} // namespace arm_compute
