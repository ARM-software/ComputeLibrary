/*
 * Copyright (c) 2022 Arm Limited.
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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION
#include "arm_compute/core/experimental/DependencyGraph.h"

#include <algorithm>
#include <deque>
#include <set>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
DependencyGraph::DependencyGraph(const AdjList &adj_src_tensors, const AdjList &adj_dst_tensors, const AdjList &adj_src_ops, const AdjList &adj_dst_ops, std::map<Id, Id> merge_points)
    : _adj_src_tensors{ adj_src_tensors }, _adj_dst_tensors{ adj_dst_tensors }, _adj_src_ops{ adj_src_ops }, _adj_dst_ops{ adj_dst_ops }, _merge_to_internal{ merge_points }, _operator_id{}, _tensor_id{}
{
}
DependencyGraph::DependencyGraph(const std::vector<Id> &imported_tensors)
    : _adj_src_tensors{}, _adj_dst_tensors{}, _adj_src_ops{}, _adj_dst_ops{}, _merge_to_internal{}, _operator_id{}, _tensor_id{}
{
    for(auto t : imported_tensors)
    {
        _adj_src_ops[t] = {};
        _adj_dst_ops[t] = {};
    }
}

Status DependencyGraph::update_merge_point(Id t_id, Id merge_point)
{
    if(_merge_to_internal.find(merge_point) == _merge_to_internal.end())
    {
        return Status{ ErrorCode::RUNTIME_ERROR, "Merge point does not exist" };
    }
    _merge_to_internal[merge_point] = t_id;
    return Status{};
}

DependencyGraph::Id DependencyGraph::add_tensor(Id merge_tensor)
{
    Id new_tensor{ empty_id() };
    if(merge_tensor != empty_id())
    {
        if(_merge_to_internal.find(merge_tensor) != _merge_to_internal.end())
        {
            new_tensor = _merge_to_internal[merge_tensor];
        }
        else
        {
            new_tensor                       = insert_new_tensor();
            _merge_to_internal[merge_tensor] = new_tensor;
        }
    }
    else
    {
        new_tensor = insert_new_tensor();
    }
    return new_tensor;
}

void DependencyGraph::remove_tensor(Id tensor)
{
    for(auto src_op : _adj_src_ops.at(tensor))
    {
        auto &dst_tensors = _adj_dst_tensors.at(src_op);
        dst_tensors.erase(
            std::remove(std::begin(dst_tensors), std::end(dst_tensors), tensor),
            std::end(dst_tensors));
    }
    for(auto dst_op : _adj_dst_ops.at(tensor))
    {
        auto &src_tensors = _adj_src_tensors.at(dst_op);
        src_tensors.erase(
            std::remove(std::begin(src_tensors), std::end(src_tensors), tensor),
            std::end(src_tensors));
    }
    _adj_src_ops.erase(tensor);
    _adj_dst_ops.erase(tensor);
}

std::pair<Status, DependencyGraph::Id> DependencyGraph::add_operator(const std::vector<Id> &inputs, const std::vector<Id> &outputs)
{
    Id new_op = insert_new_op();
    for(Id tensor : inputs)
    {
        link_input(new_op, tensor);
    }
    for(Id tensor : outputs)
    {
        link_output(new_op, tensor);
    }

    // Use topological sort in order to detect possible loops / cycles.
    // NOTE: This is unscalable. We'll need to have a better way of detecting loops or relax this invariant during operation, and add a validate method instead
    return std::pair<Status, DependencyGraph::Id>(topological_sort().first, new_op);
}

void DependencyGraph::remove_operator(Id op)
{
    for(auto src_tensor : _adj_src_tensors.at(op))
    {
        auto &dst_ops = _adj_dst_ops.at(src_tensor);
        dst_ops.erase(
            std::remove(std::begin(dst_ops), std::end(dst_ops), op),
            std::end(dst_ops));
    }
    for(auto dst_tensor : _adj_dst_tensors.at(op))
    {
        auto &src_ops = _adj_src_ops.at(dst_tensor);
        src_ops.erase(
            std::remove(std::begin(src_ops), std::end(src_ops), op),
            std::end(src_ops));
    }
    _adj_src_tensors.erase(op);
    _adj_dst_tensors.erase(op);
}

std::map<DependencyGraph::Id, DependencyGraph::Id> DependencyGraph::get_merge_points() const
{
    return _merge_to_internal;
}

std::vector<DependencyGraph::Id> DependencyGraph::get_root_ops() const
{
    std::vector<Id> ops{};
    const auto      op_list = all_ops();

    for(auto op : op_list)
    {
        if(src_ops(op).empty())
        {
            ops.emplace_back(op);
        }
    }
    return ops;
}

std::vector<DependencyGraph::Id> DependencyGraph::get_dst_ops() const
{
    std::vector<Id> ops{};
    const auto      op_list = all_ops();

    for(auto op : op_list)
    {
        if(dst_ops(op).empty())
        {
            ops.emplace_back(op);
        }
    }
    return ops;
}

std::vector<DependencyGraph::Id> DependencyGraph::src_tensors(Id op) const
{
    ARM_COMPUTE_ERROR_ON(!operator_exists(op));
    return _adj_src_tensors.at(op);
}

std::vector<DependencyGraph::Id> DependencyGraph::dst_tensors(Id op) const
{
    ARM_COMPUTE_ERROR_ON(!operator_exists(op));
    return _adj_dst_tensors.at(op);
}

std::vector<DependencyGraph::Id> DependencyGraph::src_tensors() const
{
    std::vector<Id> tensors;
    for(auto tensor_src_ops : _adj_src_ops)
    {
        if(tensor_src_ops.second.empty())
            tensors.push_back(tensor_src_ops.first);
    }
    return tensors;
}

std::vector<DependencyGraph::Id> DependencyGraph::dst_tensors() const
{
    std::vector<Id> tensors;
    for(auto tensor_dst_ops : _adj_dst_ops)
    {
        if(tensor_dst_ops.second.empty())
            tensors.push_back(tensor_dst_ops.first);
    }
    return tensors;
}

std::vector<DependencyGraph::Id> DependencyGraph::src_ops_from_tensor(Id tensor) const
{
    return _adj_src_ops.at(tensor);
}
std::vector<DependencyGraph::Id> DependencyGraph::dst_ops_from_tensor(Id tensor) const
{
    return _adj_dst_ops.at(tensor);
}

std::vector<DependencyGraph::Id> DependencyGraph::all_ops() const
{
    std::vector<Id> ops{};
    std::transform(std::begin(_adj_src_tensors), std::end(_adj_src_tensors), std::back_inserter(ops), [](const auto & it)
    {
        return it.first;
    });
    return ops;
}

bool DependencyGraph::path_exists_from_tensor_to_op(Id src_tensor, Id dst_op) const
{
    for(auto child_op : dst_ops_from_tensor(src_tensor))
    {
        if(path_exists_from_op_to_op(child_op, dst_op))
        {
            return true;
        }
    }
    return false;
}

bool DependencyGraph::path_exists_from_op_to_op(Id src_op, Id dst_op) const
{
    if(src_op == dst_op)
    {
        return true;
    }
    if(is_in(src_op, get_dst_ops()))
    {
        return false;
    }
    for(auto child_tensor : dst_tensors(src_op))
    {
        if(path_exists_from_tensor_to_op(child_tensor, dst_op))
        {
            return true;
        }
    }
    return false;
}

std::vector<DependencyGraph::Id> DependencyGraph::all_tensors() const
{
    std::vector<Id> tensors{};
    std::transform(std::begin(_adj_src_ops), std::end(_adj_src_ops), std::back_inserter(tensors), [](const auto & it)
    {
        return it.first;
    });
    return tensors;
}

unsigned int DependencyGraph::number_of_ops() const
{
    return _adj_src_tensors.size();
}

unsigned int DependencyGraph::number_of_tensors() const
{
    return _adj_src_ops.size();
}

DependencyGraph::Id DependencyGraph::insert_new_tensor()
{
    Id new_tensor            = _tensor_id.alloc();
    _adj_src_ops[new_tensor] = {};
    _adj_dst_ops[new_tensor] = {};
    return new_tensor;
}
DependencyGraph::Id DependencyGraph::insert_new_op()
{
    Id new_op                = _operator_id.alloc();
    _adj_src_tensors[new_op] = {};
    _adj_dst_tensors[new_op] = {};
    return new_op;
}
void DependencyGraph::link_input(Id op, Id in_tensor)
{
    ARM_COMPUTE_ERROR_ON(!operator_exists(op));
    ARM_COMPUTE_ERROR_ON(!tensor_exists(in_tensor));
    ARM_COMPUTE_ERROR_ON(are_connected(op, in_tensor));
    _adj_src_tensors[op].push_back(in_tensor);
    _adj_dst_ops[in_tensor].push_back(op);
}
void DependencyGraph::link_output(Id op, Id out_tensor)
{
    ARM_COMPUTE_ERROR_ON(!operator_exists(op));
    ARM_COMPUTE_ERROR_ON(!tensor_exists(out_tensor));
    ARM_COMPUTE_ERROR_ON(are_connected(op, out_tensor));
    _adj_dst_tensors[op].push_back(out_tensor);
    _adj_src_ops[out_tensor].push_back(op);
}
bool DependencyGraph::tensor_exists(Id tensor) const
{
    return _adj_src_ops.find(tensor) != _adj_src_ops.end() && _adj_dst_ops.find(tensor) != _adj_dst_ops.end();
}
bool DependencyGraph::operator_exists(Id op) const
{
    return _adj_src_tensors.find(op) != _adj_src_tensors.end() && _adj_dst_tensors.find(op) != _adj_dst_tensors.end();
}

bool DependencyGraph::is_src_tensor(Id tensor) const
{
    if(!tensor_exists(tensor))
    {
        return false;
    }
    return _adj_src_ops.at(tensor).empty();
}

bool DependencyGraph::is_dst_tensor(Id tensor) const
{
    if(!tensor_exists(tensor))
    {
        return false;
    }
    return _adj_dst_ops.at(tensor).empty();
}
bool DependencyGraph::is_src_tensor_of(Id op, Id tensor) const
{
    if(!operator_exists(op) || !tensor_exists(tensor))
    {
        return false;
    }
    const auto op_inputs = src_tensors(op);
    return std::find(op_inputs.begin(), op_inputs.end(), tensor) != op_inputs.end();
}
bool DependencyGraph::is_dst_tensor_of(Id op, Id tensor) const
{
    if(!operator_exists(op) || !tensor_exists(tensor))
    {
        return false;
    }
    const auto op_outputs = dst_tensors(op);
    return std::find(op_outputs.begin(), op_outputs.end(), tensor) != op_outputs.end();
}
bool DependencyGraph::are_connected(Id op, Id tensor) const
{
    return is_src_tensor_of(op, tensor) || is_dst_tensor_of(op, tensor);
}
std::vector<DependencyGraph::Id> DependencyGraph::src_ops(Id op) const
{
    ARM_COMPUTE_ERROR_ON(!operator_exists(op));
    std::vector<Id> ops{};
    for(Id src_tensor : src_tensors(op))
    {
        ops.insert(ops.end(), std::begin(_adj_src_ops.at(src_tensor)), std::end(_adj_src_ops.at(src_tensor)));
    }
    return ops;
}

std::vector<DependencyGraph::Id> DependencyGraph::dst_ops(Id op) const
{
    ARM_COMPUTE_ERROR_ON(!operator_exists(op));
    std::vector<Id> ops{};
    for(Id dst_tensor : _adj_dst_tensors.at(op))
    {
        ops.insert(ops.end(), std::begin(_adj_dst_ops.at(dst_tensor)), std::end(_adj_dst_ops.at(dst_tensor)));
    }
    return ops;
}

std::pair<Status, std::vector<DependencyGraph::OpPack>> DependencyGraph::topological_sort() const
{
    // Incident degree (number of source operators to an op)
    std::map<Id, unsigned int> in_degree{};
    std::set<Id>        visited_ops{};
    std::deque<Id>      zero_in_degree_ops{};
    std::vector<OpPack> sorted_op_packs{};
    for(auto op : all_ops())
    {
        const auto degree = src_ops(op).size();
        in_degree[op]     = degree;
        if(degree == 0)
        {
            zero_in_degree_ops.push_back(op);
            visited_ops.insert(op);
        }
    }

    while(!zero_in_degree_ops.empty())
    {
        const Id op = zero_in_degree_ops.front();
        zero_in_degree_ops.pop_front();
        sorted_op_packs.push_back(OpPack{ op, src_tensors(op), dst_tensors(op) });

        for(const auto next_op : dst_ops(op))
        {
            if(in_degree[next_op] > 0)
            {
                in_degree[next_op]--;
            }
            if(in_degree[next_op] == 0 && visited_ops.find(next_op) == visited_ops.end())
            {
                zero_in_degree_ops.push_back(next_op);
                visited_ops.insert(op);
            }
        }
    }

    // If there are remaining ops with in_degree > 0, then it's indication that there are cycles in the graph
    Status st{};
    if(sorted_op_packs.size() != number_of_ops())
    {
        st = Status{ ErrorCode::RUNTIME_ERROR, "Cycles or loops are not allowed in a DependencyGraph" };
    }
    return std::make_pair(st, sorted_op_packs);
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */