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
#include "arm_compute/graph/Graph.h"

namespace arm_compute
{
namespace graph
{
Graph::Graph(GraphID id, std::string name)
    : _id(id), _name(std::move(name)), _nodes(), _edges(), _tensors(), _tagged_nodes(), _mtx()
{
}

bool Graph::remove_node(NodeID nid)
{
    if(nid >= _nodes.size())
    {
        return false;
    }

    std::unique_ptr<INode> &node = _nodes[nid];

    if(node)
    {
        // Remove input connections
        for(auto &input_eid : node->_input_edges)
        {
            remove_connection(input_eid);
        }

        // Remove output connections
        std::set<EdgeID> output_edges_copy = node->output_edges();
        for(auto &outpud_eid : output_edges_copy)
        {
            remove_connection(outpud_eid);
        }

        // Remove nid from tagged nodes
        std::vector<NodeID> &tnodes = _tagged_nodes.at(node->type());
        tnodes.erase(std::remove(tnodes.begin(), tnodes.end(), nid), tnodes.end());
    }

    node = nullptr;

    return true;
}

EdgeID Graph::add_connection(NodeID source, size_t source_idx, NodeID sink, size_t sink_idx)
{
    std::lock_guard<arm_compute::Mutex> lock(_mtx);

    // Check if node index is valid, if node exists and finally if the connection index is valid
    ARM_COMPUTE_ERROR_ON((source >= _nodes.size()) || (_nodes[source] == nullptr) || (source_idx >= _nodes[source]->num_outputs()));
    ARM_COMPUTE_ERROR_ON((sink >= _nodes.size()) || (_nodes[sink] == nullptr) || (sink_idx >= _nodes[sink]->num_inputs()));

    // Get nodes
    std::unique_ptr<INode> &source_node = _nodes[source];
    std::unique_ptr<INode> &sink_node   = _nodes[sink];

    // Check for duplicate connections (Check only sink node)
    Edge *sink_node_edge = sink_node->input_edge(sink_idx);
    if((sink_node_edge != nullptr) && (sink_node_edge->producer_id() == source) && (sink_node_edge->producer_idx() == source_idx)
       && (sink_node_edge->consumer_id() == sink) && (sink_node_edge->consumer_idx() == sink_idx))
    {
        return sink_node_edge->id();
    }

    // Check if there is already a tensor associated with output if not create one
    TensorID tid = source_node->output_id(source_idx);
    if(tid == NullTensorID)
    {
        tid = create_tensor();
    }
    std::unique_ptr<Tensor> &tensor = _tensors[tid];

    // Create connections
    EdgeID eid        = _edges.size();
    auto   connection = arm_compute::support::cpp14::make_unique<Edge>(eid, source_node.get(), source_idx, sink_node.get(), sink_idx, tensor.get());
    _edges.push_back(std::move(connection));

    // Add connections to source and sink nodes
    source_node->_output_edges.insert(eid);
    sink_node->_input_edges[sink_idx] = eid;

    // Set tensor output node
    source_node->_outputs[source_idx] = tid;

    // Bind tensor to the edge
    tensor->bind_edge(eid);

    // Try and propagate shapes in sink node
    sink_node->forward_descriptors();

    return eid;
}

bool Graph::remove_connection(EdgeID eid)
{
    if(eid >= _edges.size())
    {
        return false;
    }

    std::unique_ptr<Edge> &edge = _edges[eid];

    // Remove node connections
    if(edge != nullptr)
    {
        // Get tensor bound to the edge
        if(edge->tensor() != nullptr)
        {
            edge->tensor()->unbind_edge(eid);
        }

        // Remove edges from source node
        if(edge->producer() != nullptr)
        {
            edge->producer()->_output_edges.erase(eid);
        }

        // Remove edges from sink node
        if((edge->consumer() != nullptr) && (edge->consumer_idx() < edge->consumer()->_input_edges.size()))
        {
            edge->consumer()->_input_edges[edge->consumer_idx()] = EmptyEdgeID;
        }
    }

    // Clear edge
    edge = nullptr;

    return true;
}

TensorID Graph::create_tensor(const TensorDescriptor &desc)
{
    TensorID tid    = _tensors.size();
    auto     tensor = support::cpp14::make_unique<Tensor>(tid, desc);
    _tensors.push_back(std::move(tensor));

    return tid;
}

std::string Graph::name() const
{
    return _name;
}

GraphID Graph::id() const
{
    return _id;
}

const std::vector<NodeID> &Graph::nodes(NodeType type)
{
    return _tagged_nodes[type];
}

std::vector<std::unique_ptr<INode>> &Graph::nodes()
{
    return _nodes;
}

const std::vector<std::unique_ptr<INode>> &Graph::nodes() const
{
    return _nodes;
}

const std::vector<std::unique_ptr<Edge>> &Graph::edges() const
{
    return _edges;
}

std::vector<std::unique_ptr<Tensor>> &Graph::tensors()
{
    return _tensors;
}

const std::vector<std::unique_ptr<Tensor>> &Graph::tensors() const
{
    return _tensors;
}

const INode *Graph::node(NodeID id) const
{
    return (id >= _nodes.size()) ? nullptr : _nodes[id].get();
}

INode *Graph::node(NodeID id)
{
    return (id >= _nodes.size()) ? nullptr : _nodes[id].get();
}

const Edge *Graph::edge(EdgeID id) const
{
    return (id >= _edges.size()) ? nullptr : _edges[id].get();
}

Edge *Graph::edge(EdgeID id)
{
    return (id >= _edges.size()) ? nullptr : _edges[id].get();
}

const Tensor *Graph::tensor(TensorID id) const
{
    return (id >= _tensors.size()) ? nullptr : _tensors[id].get();
}

Tensor *Graph::tensor(TensorID id)
{
    return (id >= _tensors.size()) ? nullptr : _tensors[id].get();
}
} // namespace graph
} // namespace arm_compute