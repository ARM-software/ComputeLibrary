/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_GRAPH_GRAPH_H__
#define __ARM_COMPUTE_GRAPH_GRAPH_H__

#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/SubTensor.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "support/ToolchainSupport.h"

#include <memory>

namespace arm_compute
{
class IFunction;

namespace graph
{
/** Graph class */
class Graph final
{
public:
    /** Constructor */
    Graph();
    /** Destructor */
    ~Graph();
    /** Prevent instances from being copy constructed */
    Graph(const Graph &) = delete;
    /** Prevent instances from being copy assigned */
    const Graph &operator=(const Graph &) = delete;
    /** Prevent instances from being move constructed */
    Graph(Graph &&) = delete;
    /** Prevent instances from being move assigned */
    Graph &operator=(Graph &&) = delete;
    /** Executes the graph */
    void run();
    /** Adds a node to the graph
     *
     * @param[in] node Node to add
     */
    void add_node(std::unique_ptr<INode> node);
    /** Adds a tensor to the graph
     *
     * @param[in] tensor Tensor to add
     */
    void add_tensor_object(std::unique_ptr<ITensorObject> tensor);
    /** Finalizes the current node's configuration
     */
    static bool opencl_is_available();
    /** Manually sets the output of the current node
     *
     * @param[in] tmp Output info to set
     */
    void set_temp(TensorInfo &&tmp);

    /** Returns the graph hints that are currently used
     *
     * @return Graph hints
     */
    GraphHints &hints();

private:
    class Private;
    std::unique_ptr<Private> _pimpl; /**< Internal implementation class */
};

/** Overloaded stream operator to add a tensor through its tensor info to the graph
 *
 * @param[in, out] graph Graph to add the tensor
 * @param[in]      info  Tensor information of the tensor to be added
 *
 * @return Updated graph
 */
Graph &operator<<(Graph &graph, TensorInfo &&info);
/** Overloaded stream operator to add a tensor to the graph
 *
 * @param[in, out] graph  Graph to add the tensor
 * @param[in]      tensor Tensor to be added
 *
 * @return Updated graph
 */
Graph &operator<<(Graph &graph, Tensor &&tensor);
/** Overloaded stream operator to add a sub-tensor to the graph
 *
 * @param[in, out] graph      Graph to add the tensor
 * @param[in]      sub_tensor Sub-tensor to be added
 *
 * @return Updated graph
 */
Graph &operator<<(Graph &graph, SubTensor &&sub_tensor);
/** Overloaded stream operator to provide a target hint to the graph
 *
 * @param[in, out] graph       Graph to provide the hint to
 * @param[in]      target_hint Target hint to be considered
 *
 * @return Updated graph
 */
Graph &operator<<(Graph &graph, TargetHint target_hint);
/** Overloaded stream operator to provide a convolution method hint to the graph
 *
 * @param[in, out] graph            Graph to provide the hint to
 * @param[in]      conv_method_hint Convolution method hint to be considered
 *
 * @return Updated graph
 */
Graph &operator<<(Graph &graph, ConvolutionMethodHint conv_method_hint);
/** Overloaded stream operator to add a node to the graph
 *
 * @param[in, out] graph Graph to add the tensor
 * @param[in]      node  Node to be added
 *
 * @return Updated graph
 */
template <typename Node>
Graph &operator<<(Graph &graph, Node node)
{
    graph.add_node(arm_compute::support::cpp14::make_unique<Node>(std::move(node)));
    return graph;
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_GRAPH_H__ */
