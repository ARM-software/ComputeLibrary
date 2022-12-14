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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_UTILS_DEPENDENCYGRAPH
#define SRC_DYNAMIC_FUSION_SKETCH_UTILS_DEPENDENCYGRAPH

#include "arm_compute/core/Error.h"
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
template <typename T>
bool is_in(const T &v, const std::vector<T> &vec)
{
    return std::find(std::begin(vec), std::end(vec), v) != std::end(vec);
}
} // namespace

/** A multi-input (tensors), multi-output (tensors) acyclic directed graph
 *  Represented as a doubly-linked adjacency list with the differentiation between source and destination
 */
class DependencyGraph
{
public:
    using Id         = int32_t;
    using TensorId   = Id;
    using OperatorId = Id;
    /** Adjacency list
     *
     */
    using AdjList = std::map<Id, std::vector<Id>>;

    /** A pack of operator including its input and output tensors, used by traversing through the graph in topological order
     *
     */
    struct OpPack
    {
        OperatorId            op{};
        std::vector<TensorId> inputs{};
        std::vector<TensorId> outputs{};
        friend bool operator==(const OpPack &opp0, const OpPack &opp1)
        {
            return std::make_tuple(
                       opp0.op, opp0.inputs, opp0.outputs)
                   == std::make_tuple(
                       opp1.op, opp1.inputs, opp1.outputs);
        }
    };

public:
    DependencyGraph() = default;
    friend std::ostream &operator<<(std::ostream &os, const DependencyGraph &);

    /** Try adding an operator (without actually adding it), while keeping the graph as a "linear sequence" / list
     *
     * Rule: If the new operator is not the first operator, at least one input tensor must be
     *       the output tensor of the last non-output operator. All other input tensors must be
     *       the global input of the graph (i.e. not the output of any operator).
     *
     * Rule: The output tensor of the new operator must not be the input tensor of any previously
     *       added operator.
     *
     * PRECONDITION: The current graph is already linear
     *
     * @return true  If the operator can be added while keeping the graph as a linear sequence
     * @return false  Otherwise
     */
    bool try_add_operator_as_linear(OperatorId op, const std::vector<TensorId> &inputs, const std::vector<TensorId> &outputs, bool is_output = false) const
    {
        ARM_COMPUTE_UNUSED(op, is_output);
        if(all_ops().empty())
        {
            return true;
        }

        // If the new operator is not the first operator, at least one input tensor must be
        // the output tensor of the last non-output operator. All other input tensors must be
        // the global input of the graph (i.e. not the output of any operator).
        if(_last_op_available)
        {
            auto use_input_from_last_op = false;

            for(auto src_tensor : inputs)
            {
                const auto src_ops = _adj_src_ops.find(src_tensor);

                if(src_ops != _adj_src_ops.end())
                {
                    ARM_COMPUTE_ERROR_ON(src_ops->second.size() > 1);

                    if(!src_ops->second.empty())
                    {
                        const auto src_op = src_ops->second[0];

                        if(src_op == _last_op)
                        {
                            if(use_input_from_last_op)
                            {
                                // To be safe, we also forbid using the output tensor
                                // of the last operator twice.
                                return false;
                            }

                            use_input_from_last_op = true;
                        }
                        else
                        {
                            // The input tensor of this operator must not be the output tensor
                            // of any other operator except the last non-output operator.
                            return false;
                        }
                    }
                }
            }

            if(!use_input_from_last_op)
            {
                // At least one input tensor must be the output tensor of the last non-output operator.
                return false;
            }
        }

        // The output tensor of the new operator must not be the input tensor of any previously
        // added operator.
        for(auto dst_tensor : outputs)
        {
            if(_adj_dst_ops.find(dst_tensor) != _adj_dst_ops.end())
            {
                return false;
            }
        }

        return true;
    }
    /** Add an operator, while keeping the graph as a "linear sequence"
     *
     * PRECONDITION: The current graph is already linear
     * INVARIANT: The list can only grow from head to tail
     * INVARIANT: POSTCONDITION: The graph is linear
     */
    void add_operator_as_linear(OperatorId op, const std::vector<TensorId> &inputs, const std::vector<TensorId> &outputs, bool is_output = false)
    {
        const auto success = add_operator(op, inputs, outputs, is_output);
        ARM_COMPUTE_UNUSED(success);
        ARM_COMPUTE_ERROR_ON(!success);
    }
    /** Add a new operator
     *  Return invalid if it violates the DAG invariant
     *  Invalid operation will not change the graph
     *
     * @param[in] op        Operator to add
     * @param[in] inputs    Input tensors to the operator
     * @param[in] outputs   Output tensors to the operator
     * @param[in] is_output Whether this is an output operator
     */
    bool add_operator(OperatorId op, const std::vector<TensorId> &inputs, const std::vector<TensorId> &outputs, bool is_output = false)
    {
        if(operator_exists(op))
        {
            return false;
        }
        _adj_src_tensors[op] = {};
        _adj_dst_tensors[op] = {};
        for(auto in_tensor : inputs)
        {
            // Linking input tensor to operator node will never create a cycle / loop because we guarantee
            // each op is newly created, so every <input, op> pair / edge is new
            link_input(op, in_tensor);
        }
        for(auto out_tensor : outputs)
        {
            // If there exists a back path from op's output tensor to op already, then linking the two will create a loop / cycle
            if(path_exists_from_tensor_to_op(out_tensor, op))
            {
                remove_operator(op);
                return false;
            }
            else
            {
                link_output(op, out_tensor);
            }
        }

        if(!is_output)
        {
            _last_op_available = true;
            _last_op = op;
        }

        return true;
    }

    /** Build a sequence of operators from the acyclic graph of operators.
     *
     * The graph will be visited in depth-first strategy. The operator can only be added to
     * the sequence when all operators that supply the input tensors have been added. Otherwise,
     * the operator will be ignored and later visited again. In other words, the dependency between
     * operators will be preserved in the sequence.
     */
    std::vector<OpPack> build_operators_sequence() const
    {
        std::vector<OpPack> ops_seq;
        std::set<Id> done_ops;
        std::set<Id> done_tensors;

        const auto input_tensors = global_src_tensors();

        for(auto tensor : input_tensors)
        {
            done_tensors.insert(tensor);

            for(auto op : _adj_dst_ops.at(tensor))
            {
                build_operators_sequence_from_op(op, ops_seq, done_ops, done_tensors);
            }
        }

        return ops_seq;
    }

    /** Strict equality comparison (all internal ids and order of insertion matter).
     *        In the future this may be replaced with a topological comparison, allowing equivalent graphs with different internal ids to be equal
     *
     *
     * @param[in] g0
     * @param[in] g1
     * @return true  If the same
     * @return false Otherwise
     */
    friend bool operator==(const DependencyGraph &g0, const DependencyGraph &g1)
    {
        // Do not compare id allocators
        return std::make_tuple(
                   g0._adj_src_tensors, g0._adj_dst_tensors, g0._adj_src_ops, g0._adj_dst_ops)
               == std::make_tuple(
                   g1._adj_src_tensors, g1._adj_dst_tensors, g1._adj_src_ops, g1._adj_dst_ops);
    }
    std::vector<OperatorId> src_ops_from_tensor(TensorId tensor) const
    {
        return _adj_src_ops.at(tensor);
    }
    std::vector<OperatorId> dst_ops_from_tensor(TensorId tensor) const
    {
        return _adj_dst_ops.at(tensor);
    }
    /** Get all tensors
     *
     * @return std::vector<TensorId>
     */
    std::vector<TensorId> all_tensors() const
    {
        std::vector<TensorId> tensors{};
        std::transform(std::begin(_adj_src_ops), std::end(_adj_src_ops), std::back_inserter(tensors), [](const auto & it)
        {
            return it.first;
        });
        return tensors;
    }
    /** Get source tensors of the whole graph
     *
     * @return std::vector<TensorId>
     */
    std::vector<TensorId> global_src_tensors() const
    {
        std::vector<TensorId> tensors;
        for(auto tensor_src_ops : _adj_src_ops)
        {
            if(tensor_src_ops.second.empty())
            {
                tensors.push_back(tensor_src_ops.first);
            }
        }
        return tensors;
    }
    /** Get destination tensors of the whole graph
     *
     * @return std::vector<TensorId>
     */
    std::vector<TensorId> global_dst_tensors() const
    {
        std::vector<TensorId> tensors;
        for(auto tensor_dst_ops : _adj_dst_ops)
        {
            if(tensor_dst_ops.second.empty())
            {
                tensors.push_back(tensor_dst_ops.first);
            }
        }
        return tensors;
    }
    /** Get intermediate tensors of the whole graph.
     *
     * @return std::vector<TensorId>
     */
    std::vector<TensorId> intermediate_tensors() const
    {
        std::vector<TensorId> tensors;

        // If a tensor is used to connect the input of an operator and the output of another operator,
        // it is not allocated in the memory. The tensor exists as a temporary variable only.
        for(auto src_tensor : _adj_src_ops)
        {
            if(!src_tensor.second.empty())
            {
                const auto dst_tensor = _adj_dst_ops.find(src_tensor.first);
                if(dst_tensor != _adj_dst_ops.end())
                {
                    if(!dst_tensor->second.empty())
                    {
                        tensors.push_back(src_tensor.first);
                    }
                }
            }
        }

        return tensors;
    }
    /** Get all root ops. Root ops can also be referred to as "src ops" of the whole graph
     *
     * @return std::vector<OperatorId>
     */
    std::vector<OperatorId> get_root_ops() const
    {
        std::vector<OperatorId> ops{};
        const auto              op_list = all_ops();

        for(auto op : op_list)
        {
            if(src_ops(op).empty())
            {
                ops.emplace_back(op);
            }
        }
        return ops;
    }

private:
    void link_input(OperatorId op, TensorId in_tensor)
    {
        ARM_COMPUTE_ERROR_ON(!operator_exists(op));
        if(!tensor_exists(in_tensor))
        {
            insert_new_tensor(in_tensor);
        }
        ARM_COMPUTE_ERROR_ON(are_connected(op, in_tensor)); // Prevent repetitive linking
        _adj_src_tensors[op].push_back(in_tensor);
        _adj_dst_ops[in_tensor].push_back(op);
    }
    void link_output(OperatorId op, TensorId out_tensor)
    {
        ARM_COMPUTE_ERROR_ON(!operator_exists(op));
        if(!tensor_exists(out_tensor))
        {
            insert_new_tensor(out_tensor);
        }
        ARM_COMPUTE_ERROR_ON(are_connected(op, out_tensor)); // Prevent repetitive linking
        _adj_dst_tensors[op].push_back(out_tensor);
        _adj_src_ops[out_tensor].push_back(op);
    }

    std::vector<OperatorId> src_ops(OperatorId op) const
    {
        ARM_COMPUTE_ERROR_ON(!operator_exists(op));
        std::vector<OperatorId> ops{};
        for(TensorId src_tensor : src_tensors(op))
        {
            ops.insert(ops.end(), std::begin(_adj_src_ops.at(src_tensor)), std::end(_adj_src_ops.at(src_tensor)));
        }
        return ops;
    }
    std::vector<OperatorId> dst_ops(OperatorId op) const
    {
        ARM_COMPUTE_ERROR_ON(!operator_exists(op));
        std::vector<OperatorId> ops{};
        for(TensorId dst_tensor : _adj_dst_tensors.at(op))
        {
            ops.insert(ops.end(), std::begin(_adj_dst_ops.at(dst_tensor)), std::end(_adj_dst_ops.at(dst_tensor)));
        }
        return ops;
    }

    /** Get source tensors to an operator
     *
     * @param[in] op
     * @return std::vector<TensorId>
     */
    std::vector<TensorId> src_tensors(OperatorId op) const
    {
        ARM_COMPUTE_ERROR_ON(!operator_exists(op));
        return _adj_src_tensors.at(op);
    }
    /** Get destination tensors to an operator
     *
     * @param[in] op
     * @return std::vector<TensorId>
     */
    std::vector<TensorId> dst_tensors(OperatorId op) const
    {
        ARM_COMPUTE_ERROR_ON(!operator_exists(op));
        return _adj_dst_tensors.at(op);
    }
    /** Get all operators
     *
     * @return std::vector<OperatorId>
     */
    std::vector<OperatorId> all_ops() const
    {
        std::vector<OperatorId> ops{};
        std::transform(std::begin(_adj_src_tensors), std::end(_adj_src_tensors), std::back_inserter(ops), [](const auto & it)
        {
            return it.first;
        });
        return ops;
    }
    /** Remove an operator from graph.
     *
     * @param[in] op
     */
    void remove_operator(OperatorId op)
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
        // Remove any isolated tensors
        // An isolated tensor is one where both its _adj_src_ops and _adj_dst_ops are empty
        for(auto t : all_tensors())
        {
            if(_adj_src_ops.at(t).empty() && _adj_dst_ops.at(t).empty())
            {
                _adj_src_ops.erase(t);
                _adj_dst_ops.erase(t);
            }
        }
        _adj_src_tensors.erase(op);
        _adj_dst_tensors.erase(op);
    }
    void insert_new_tensor(TensorId tensor)
    {
        _adj_src_ops[tensor] = {};
        _adj_dst_ops[tensor] = {};
    }
    bool tensor_exists(TensorId tensor) const
    {
        return _adj_src_ops.find(tensor) != _adj_src_ops.end() && _adj_dst_ops.find(tensor) != _adj_dst_ops.end();
    }
    bool operator_exists(OperatorId op) const
    {
        return _adj_src_tensors.find(op) != _adj_src_tensors.end() && _adj_dst_tensors.find(op) != _adj_dst_tensors.end();
    }
    bool is_src_tensor_of(OperatorId op, TensorId tensor) const
    {
        if(!operator_exists(op) || !tensor_exists(tensor))
        {
            return false;
        }
        const auto op_inputs = src_tensors(op);
        return std::find(op_inputs.begin(), op_inputs.end(), tensor) != op_inputs.end();
    }
    bool is_dst_tensor_of(OperatorId op, TensorId tensor) const
    {
        if(!operator_exists(op) || !tensor_exists(tensor))
        {
            return false;
        }
        const auto op_outputs = dst_tensors(op);
        return std::find(op_outputs.begin(), op_outputs.end(), tensor) != op_outputs.end();
    }
    bool are_connected(OperatorId op, TensorId tensor) const
    {
        return is_src_tensor_of(op, tensor) || is_dst_tensor_of(op, tensor);
    }
    /** If op is the destination / leaf operator of the whole graph
     *
     * @param[in] op
     * @return true
     * @return false
     */
    bool is_dst_op(OperatorId op) const
    {
        return dst_ops(op).empty();
    }
    std::vector<OperatorId> get_dst_ops() const
    {
        std::vector<OperatorId> ops{};
        const auto              op_list = all_ops();

        for(auto op : op_list)
        {
            if(is_dst_op(op))
            {
                ops.emplace_back(op);
            }
        }
        return ops;
    }
    bool path_exists_from_tensor_to_op(TensorId src_tensor, OperatorId dst_op) const
    {
        if(!tensor_exists(src_tensor) || !operator_exists(dst_op))
        {
            return false;
        }
        for(auto child_op : dst_ops_from_tensor(src_tensor))
        {
            if(path_exists_from_op_to_op(child_op, dst_op))
            {
                return true;
            }
        }
        return false;
    }

    bool path_exists_from_op_to_op(OperatorId src_op, OperatorId dst_op) const
    {
        if(!operator_exists(src_op) || !operator_exists(dst_op))
        {
            return false;
        }
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

    void build_operators_sequence_from_op(
        Id op,
        std::vector<OpPack> &ops_seq,
        std::set<Id> &done_ops,
        std::set<Id> &done_tensors) const
    {
        while(true)
        {
            // If the operator has been added to the sequence, ignore it.
            if(done_ops.find(op) != done_ops.end())
            {
                return;
            }

            // If not all the input tensors of the operator are available, this operator cannot be
            // added to the sequence for now. It will be visited again after the source operator
            // is added to the sequence.
            const auto src_tensors = _adj_src_tensors.at(op);

            for(auto src : src_tensors)
            {
                if(done_tensors.find(src) == done_tensors.end())
                {
                    return;
                }
            }

            // This operator is ready to be added to the sequence.
            const auto dst_tensors = _adj_dst_tensors.at(op);

            done_ops.insert(op);

            OpPack pack{ op, src_tensors, dst_tensors };
            ops_seq.push_back(pack);

            done_tensors.insert(dst_tensors.begin(), dst_tensors.end());

            // Visit all the sink operators.
            // Call this function recursively unless there is only one sink.
            if(dst_tensors.size() == 1 && _adj_dst_ops.at(dst_tensors[0]).size() == 1)
            {
                op = _adj_dst_ops.at(dst_tensors[0])[0];
            }
            else
            {
                for(auto dst_tensor : dst_tensors)
                {
                    const auto dst_ops = _adj_dst_ops.at(dst_tensor);

                    for(auto dst_op : dst_ops)
                    {
                        build_operators_sequence_from_op(dst_op, ops_seq, done_ops, done_tensors);
                    }
                }

                return;
            }
        }
    }

private:
    AdjList _adj_src_tensors{};
    AdjList _adj_dst_tensors{};
    AdjList _adj_src_ops{};
    AdjList _adj_dst_ops{};

    bool _last_op_available{ false };
    OperatorId _last_op{ 0 };
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_UTILS_DEPENDENCYGRAPH */
