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
#ifndef __ARM_COMPUTE_GRAPH_SUBGRAPH_H__
#define __ARM_COMPUTE_GRAPH_SUBGRAPH_H__

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/ITensorObject.h"
#include "arm_compute/graph/SubTensor.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"
#include "arm_compute/runtime/IFunction.h"

#include <memory>

namespace arm_compute
{
namespace graph
{
/** SubGraph class */
class SubGraph
{
public:
    /** Constructor */
    SubGraph();
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
    /** Constructs a graph from a subgraph
     *
     * @param[in] ctx    Parent graph context
     * @param[in] input  Input to the graph
     * @param[in] output Output to the graph
     *
     * @return A graph
     */
    std::unique_ptr<Graph> construct(const GraphContext &ctx, std::unique_ptr<ITensorObject> input, std::unique_ptr<ITensorObject> output);
    /** Checks if the subgraph has an input
     *
     * @return True if the sub-graph has an input else false
     */
    bool has_input() const;
    /** Checks if the subgraph has an output
     *
     * @return True if the sub-graph has an output else false
     */
    bool has_output() const;

private:
    std::vector<std::unique_ptr<INode>> _nodes;
    std::unique_ptr<ITensorObject>      _input;
    std::unique_ptr<ITensorObject>      _output;
};

SubGraph &operator<<(SubGraph &graph, Tensor &&tensor);
SubGraph &operator<<(SubGraph &graph, SubTensor &&sub_tensor);

template <typename Node>
SubGraph &operator<<(SubGraph &sub_graph, Node node)
{
    sub_graph.add_node(arm_compute::support::cpp14::make_unique<Node>(std::move(node)));
    return sub_graph;
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_INODE_H__ */
