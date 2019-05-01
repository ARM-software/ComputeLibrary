/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#include "arm_compute/graph/Edge.h"
#include "arm_compute/graph/INode.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/Types.h"

#include "support/Mutex.h"
#include "support/ToolchainSupport.h"

#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace arm_compute
{
namespace graph
{
/** Graph class
 *
 * Represents a multiple source - multiple sink directed graph
 */
class Graph final
{
public:
    Graph() = default;
    /** Constructor
     *
     * @param[in] id   Graph identification number. Can be used to differentiate between graphs. Default value 0
     * @param[in] name Graph name. Default value empty string
     */
    Graph(GraphID id, std::string name);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    Graph(const Graph &) = delete;
    /** Prevent instances of this class from being copy assigned (As this class contains pointers) */
    Graph &operator=(const Graph &) = delete;
    /** Allow instances of this class to be moved */
    Graph(Graph &&) = default;
    /** Allow instances of this class to be move assigned */
    Graph &operator=(Graph &&) = default;
    /** Adds a node to the graph
     *
     * @note Models a single output node
     *
     * @tparam NT Node operation
     * @tparam Ts Arguments to operation
     *
     * @param[in] args Node arguments
     *
     * @return ID of the node
     */
    template <typename NT, typename... Ts>
    NodeID add_node(Ts &&... args);
    /** Remove the node with the given ID
     *
     * @param[in] nid ID of the node to remove
     *
     * @return True if the removal took place else false
     */
    bool remove_node(NodeID nid);
    /** Adds a connection between two nodes
     *
     * @param[in] source     ID of the source node
     * @param[in] source_idx Output index of the source node
     * @param[in] sink       ID of the sink node
     * @param[in] sink_idx   Input index of the sink node
     *
     * @return ID of this connection
     */
    EdgeID add_connection(NodeID source, size_t source_idx, NodeID sink, size_t sink_idx);
    /** Removes an edge (connection)
     *
     * @param[in] eid Connection to remove
     *
     * @return True if the removal took place else false
     */
    bool remove_connection(EdgeID eid);
    /** Returns graph name
     *
     * @return Graph name
     */
    std::string name() const;
    /** Returns graph id
     *
     * @return Graph id
     */
    GraphID id() const;
    /** Returns graph input nodes
     *
     * @param[in] type Type of nodes to return
     *
     * @return vector containing the graph node of given type
     */
    const std::vector<NodeID> &nodes(NodeType type);
    /** Returns nodes of graph
     *
     * @warning Nodes can be nullptr if they have been removed during the mutation steps of the graph
     *
     * @return Nodes of graph
     */
    std::vector<std::unique_ptr<INode>> &nodes();
    /** Returns nodes of graph
     *
     * @warning Nodes can be nullptr if they have been removed during the mutation steps of the graph
     *
     * @return Nodes of graph
     */
    const std::vector<std::unique_ptr<INode>> &nodes() const;
    /** Returns edges of graph
     *
     * @warning Edges can be nullptr if they have been removed during the mutation steps of the graph
     *
     * @return Edges of graph
     */
    const std::vector<std::unique_ptr<Edge>> &edges() const;
    /** Returns tensors of graph
     *
     * @warning Tensor can be nullptr if they have been removed during the mutation steps of the graph
     *
     * @return Tensors of graph
     */
    std::vector<std::unique_ptr<Tensor>> &tensors();
    /** Returns tensors of graph
     *
     * @warning Tensor can be nullptr if they have been removed during the mutation steps of the graph
     *
     * @return Tensors of graph
     */
    const std::vector<std::unique_ptr<Tensor>> &tensors() const;
    /** Get node object given its id
     *
     * @warning Can be nullptr if node was removed during the mutation steps of the graph
     *
     * @param[in] id Node ID
     *
     * @return The actual node object
     */
    const INode *node(NodeID id) const;
    /** Get node object given its id
     *
     * @warning Can be nullptr if node was removed during the mutation steps of the graph
     *
     * @param[in] id Node ID
     *
     * @return The actual node object
     */
    INode *node(NodeID id);
    /** Get edge object given its id
     *
     * @warning Can be nullptr if node was removed during the mutation steps of the graph
     *
     * @param[in] id Edge ID
     *
     * @return The actual edge object
     */
    const Edge *edge(EdgeID id) const;
    /** Get edge object given its id
     *
     * @warning Can be nullptr if node was removed during the mutation steps of the graph
     *
     * @param[in] id Edge ID
     *
     * @return The actual edge object
     */
    Edge *edge(EdgeID id);
    /** Get tensor object given its id
     *
     * @warning Can be nullptr if tensor was removed during the mutation steps of the graph
     *
     * @param[in] id Tensor ID
     *
     * @return The actual tensor object
     */
    const Tensor *tensor(TensorID id) const;
    /** Get tensor object given its id
     *
     * @warning Can be nullptr if tensor was removed during the mutation steps of the graph
     *
     * @param[in] id Tensor ID
     *
     * @return The actual tensor object
     */
    Tensor *tensor(TensorID id);

private:
    /** Creates a tensor object
     *
     * @param[in] desc Tensor descriptor
     *
     * @return Tensor ID
     */
    TensorID create_tensor(const TensorDescriptor &desc = TensorDescriptor());

private:
    GraphID                              _id      = GraphID(0); /**< Graph id */
    std::string                          _name    = {};         /**< Graph name */
    std::vector<std::unique_ptr<INode>>  _nodes   = {};         /**< Graph nodes */
    std::vector<std::unique_ptr<Edge>>   _edges   = {};         /**< Graph edges */
    std::vector<std::unique_ptr<Tensor>> _tensors = {};         /**< Graph tensors */
    std::map<NodeType, std::vector<NodeID>> _tagged_nodes = {}; /**< Graph nodes map with the node type as key */
    arm_compute::Mutex _mtx = {};                               /**< Mutex used for graph construction */
};

template <typename NT, typename... Ts>
inline NodeID Graph::add_node(Ts &&... args)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);

    // Create node
    NodeID nid  = _nodes.size();
    auto   node = support::cpp14::make_unique<NT>(std::forward<Ts>(args)...);
    node->set_graph(this);
    node->set_id(nid);

    // Keep track of input nodes
    _tagged_nodes[node->type()].push_back(nid);

    // Associate a new tensor with each output
    for(auto &output : node->_outputs)
    {
        output = create_tensor();
    }

    // Propagate node shape if possible
    node->forward_descriptors();

    // Add node to the graph nodes
    _nodes.push_back(std::move(node));

    return nid;
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_GRAPH_GRAPH_H__ */
