/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph/INode.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/graph/Edge.h"
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Tensor.h"

namespace arm_compute
{
namespace graph
{
// *INDENT-OFF*
// clang-format off
INode::INode()
    : _graph(nullptr), _id(EmptyNodeID), _common_params({ "", Target::UNSPECIFIED}),
      _outputs(), _input_edges(), _output_edges(), _assigned_target(Target::UNSPECIFIED)
{
}
// clang-format on
// *INDENT-ON*

Status INode::validate() const
{
    return Status{};
}

void INode::set_graph(Graph *g)
{
    ARM_COMPUTE_ERROR_ON(g == nullptr);
    _graph = g;
}

void INode::set_id(NodeID id)
{
    _id = id;
}

void INode::set_common_node_parameters(NodeParams common_params)
{
    _common_params = std::move(common_params);
}

void INode::set_requested_target(Target target)
{
    _common_params.target = target;
}

void INode::set_assigned_target(Target target)
{
    _assigned_target = target;
}

void INode::set_output_tensor(TensorID tid, size_t idx)
{
    if(tid != NullTensorID && (idx < _outputs.size()) && (_graph->tensor(tid) != nullptr))
    {
        ARM_COMPUTE_ERROR_ON(_graph == nullptr);
        Tensor *updated_tensor = _graph->tensor(tid);
        _outputs[idx]          = tid;

        // Set tensor to all output edges of the node
        for(auto &output_edge_id : _output_edges)
        {
            auto output_edge = _graph->edge(output_edge_id);
            if(output_edge != nullptr)
            {
                // Unbind edge from current tensor
                auto current_output_tensor = output_edge->tensor();
                current_output_tensor->unbind_edge(output_edge->id());

                // Update tensor to edge and rebind tensor
                output_edge->update_bound_tensor(updated_tensor);
                updated_tensor->bind_edge(output_edge->id());
            }
        }
    }
}

NodeID INode::id() const
{
    return _id;
}

std::string INode::name() const
{
    return _common_params.name;
}

const Graph *INode::graph() const
{
    return _graph;
}

Graph *INode::graph()
{
    return _graph;
}

const std::vector<TensorID> &INode::outputs() const
{
    return _outputs;
}

const std::vector<EdgeID> &INode::input_edges() const
{
    return _input_edges;
}

const std::set<EdgeID> &INode::output_edges() const
{
    return _output_edges;
}

TensorID INode::input_id(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(idx >= _input_edges.size());
    Edge *e = _graph->edge(_input_edges[idx]);
    return (e != nullptr) ? e->tensor_id() : NullTensorID;
}

TensorID INode::output_id(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());
    return _outputs[idx];
}

Tensor *INode::input(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(_graph == nullptr);
    ARM_COMPUTE_ERROR_ON(idx >= _input_edges.size());
    Edge *e = _graph->edge(_input_edges[idx]);
    return (e != nullptr) ? e->tensor() : nullptr;
}

Tensor *INode::output(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(_graph == nullptr);
    ARM_COMPUTE_ERROR_ON(idx >= _outputs.size());
    return _graph->tensor(_outputs[idx]);
}

EdgeID INode::input_edge_id(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(idx >= _input_edges.size());
    return _input_edges[idx];
}

Edge *INode::input_edge(size_t idx) const
{
    ARM_COMPUTE_ERROR_ON(_graph == nullptr);
    ARM_COMPUTE_ERROR_ON(idx >= _input_edges.size());
    return _graph->edge(_input_edges[idx]);
}

size_t INode::num_inputs() const
{
    return _input_edges.size();
}

size_t INode::num_outputs() const
{
    return _outputs.size();
}

NodeParams INode::common_node_params() const
{
    return _common_params;
}

Target INode::requested_target() const
{
    return _common_params.target;
}

Target INode::assigned_target() const
{
    return _assigned_target;
}
} // namespace graph
} // namespace arm_compute