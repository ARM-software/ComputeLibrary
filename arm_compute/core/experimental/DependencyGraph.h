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
#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_DEPENDENCYGRAPH_H
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_DEPENDENCYGRAPH_H

#include "arm_compute/core/Error.h"

#include <algorithm>
#include <map>
#include <vector>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
template <typename T>
bool is_in(const T &v, const std::vector<T> &vec)
{
    return std::find(std::begin(vec), std::end(vec), v) != std::end(vec);
}

/** The dependency graph of a workload, where the nodes are of 2 types: Tensor or Operator
 *  Represented as a doubly-linked adjacency list with the differentiation between source and destination
 *
 * A "Merge Tensor" is an external tensor associated with the tensor within the graph, and serve as a merge point
 */
class DependencyGraph
{
public:
    /** A serial Id allocator
     *
     */
    class SerialIdAllocator
    {
    public:
        using Id = int;
        Id alloc()
        {
            return _counter++;
        }
        constexpr static Id empty()
        {
            return -1;
        }

    private:
        Id _counter{ 0 };
    };
    using Id = SerialIdAllocator::Id;
    /** Adjacency list
     *
     */
    using AdjList = std::map<Id, std::vector<Id>>;

    /** A pack of operator including its input and output tensors, used by traversing through the graph in topological order
     *
     */
    struct OpPack
    {
        Id              op{};
        std::vector<Id> inputs{};
        std::vector<Id> outputs{};
        friend bool operator==(const OpPack &opp0, const OpPack &opp1)
        {
            return std::make_tuple(
                       opp0.op, opp0.inputs, opp0.outputs)
                   == std::make_tuple(
                       opp1.op, opp1.inputs, opp1.outputs);
        }
    };

public:
    constexpr static Id empty_id()
    {
        return SerialIdAllocator::empty();
    }

    DependencyGraph() = default;
    // Used in cases where two DependencyGraphs may want to share the same configuration of tensors
    explicit DependencyGraph(const std::vector<Id> &imported_tensors);
    // Testing only
    DependencyGraph(const AdjList &adj_src_tensors, const AdjList &adj_dst_tensors, const AdjList &adj_src_ops, const AdjList &adj_dst_ops, std::map<Id, Id> merge_points = {});

    /** Add a new tensor
     *
     * @param merge_tensor The external merge point associated with the tensor. Leave empty if not needed.
     * @return Id  The newly allocated tensor, or a previously added tensor associated with @p merge_tensor
     */
    Id add_tensor(Id merge_tensor = empty_id());

    void remove_tensor(Id tensor);

    /** Add a new operator
     *
     * @param inputs  Input tensors to the operator
     * @param outputs  Output tensors to the operator
     * @return std::pair<Status, DependencyGraph::Id> where id is the newly allocated operator
     */
    std::pair<Status, DependencyGraph::Id> add_operator(const std::vector<Id> &inputs, const std::vector<Id> &outputs);

    void remove_operator(Id op);
    /** Sort the graph in a topological order
     *
     * @return std::pair<Status, std::vector<OpPack>>
     */
    std::pair<Status, std::vector<OpPack>> topological_sort() const;

    std::vector<Id> src_ops(Id op) const;
    std::vector<Id> dst_ops(Id op) const;

    std::vector<Id> src_ops_from_tensor(Id tensor) const;
    std::vector<Id> dst_ops_from_tensor(Id tensor) const;
    /** Get the merge points object
     *
     * @return std::map<Id, Id>
     */
    std::map<Id, Id> get_merge_points() const;
    /** Get all root ops. Root ops can also be referred to as "src ops" of the whole graph
     *
     * @return std::vector<Id>
     */
    std::vector<Id> get_root_ops() const;
    /** Get all dst ops of the whole graph
     *
     * @return std::vector<Id>
     */
    std::vector<Id> get_dst_ops() const;

    /** Get source tensors to an operator
     *
     * @param op
     * @return std::vector<Id>
     */
    std::vector<Id> src_tensors(Id op) const;
    /** Get destination tensors to an operator
     *
     * @param op
     * @return std::vector<Id>
     */
    std::vector<Id> dst_tensors(Id op) const;
    /** Get source tensors of the whole graph
     *
     * @return std::vector<Id>
     */
    std::vector<Id> src_tensors() const;
    /** Get destination tensors of the whole graph
     *
     * @return std::vector<Id>
     */
    std::vector<Id> dst_tensors() const;
    /** Get all operators
     *
     * @return std::vector<Id>
     */
    std::vector<Id> all_ops() const;
    /** Get all tensors
     *
     * @return std::vector<Id>
     */
    std::vector<Id> all_tensors() const;
    /** Number of operators
     *
     * @return unsigned int
     */
    unsigned int number_of_ops() const;
    /** Number of tensors
     *
     * @return unsigned int
     */
    unsigned int number_of_tensors() const;

    /** Update @p merge_point to point to @p t_id
     *
     * @param t_id
     * @param merge_point
     */
    Status update_merge_point(Id t_id, Id merge_point);

    /** Strict equality comparison (all internal ids and order of insertion matter).
     *        In the future this may be replaced with a topological comparison, allowing equivalent graphs with different internal ids to be equal
     *
     *
     * @param g0
     * @param g1
     * @return true
     * @return false
     */
    friend bool operator==(const DependencyGraph &g0, const DependencyGraph &g1)
    {
        // Do not compare id allocators
        return std::make_tuple(
                   g0._adj_src_tensors, g0._adj_dst_tensors, g0._adj_src_ops, g0._adj_dst_ops, g0._merge_to_internal)
               == std::make_tuple(
                   g1._adj_src_tensors, g1._adj_dst_tensors, g1._adj_src_ops, g1._adj_dst_ops, g1._merge_to_internal);
    }
    void link_input(Id op, Id in_tensor);
    void link_output(Id op, Id out_tensor);
    /** Check if there's a path from @p src_tensor to @p dst_op
     *
     * @param src_tensor
     * @param dst_op
     * @return true
     * @return false
     */
    bool path_exists_from_tensor_to_op(Id src_tensor, Id dst_op) const;
    /** Check if there's a path from @p src_op to @p dst_op
     *
     * @param src_op
     * @param dst_op
     * @return true
     * @return false
     */
    bool path_exists_from_op_to_op(Id src_op, Id dst_op) const;
    /** Check if tensor is the src tensor of the entire graph
     *
     * @param tensor
     * @return true
     * @return false
     */
    bool is_src_tensor(Id tensor) const;
    /** Check if tensor is the dst tensor of the entire graph
     *
     * @param tensor
     * @return true
     * @return false
     */
    bool is_dst_tensor(Id tensor) const;

private:
    Id   insert_new_tensor();
    Id   insert_new_op();
    bool tensor_exists(Id tensor) const;
    bool operator_exists(Id op) const;
    bool is_src_tensor_of(Id op, Id tensor) const;
    bool is_dst_tensor_of(Id op, Id tensor) const;
    bool are_connected(Id op, Id tensor) const;

private:
    AdjList _adj_src_tensors{};
    AdjList _adj_dst_tensors{};
    AdjList _adj_src_ops{};
    AdjList _adj_dst_ops{};
    std::map<Id, Id> _merge_to_internal{}; // From merge tensor to internal tensor
    SerialIdAllocator _operator_id{};
    SerialIdAllocator _tensor_id{};
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_DEPENDENCYGRAPH_H
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */