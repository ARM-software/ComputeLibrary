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
#include "arm_compute/graph/mutators/DepthConcatSubTensorMutator.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/algorithms/TopologicalSort.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/nodes/ConcatenateLayerNode.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/core/utils/misc/Iterable.h"

namespace arm_compute
{
namespace graph
{
const char *DepthConcatSubTensorMutator::name()
{
    return "DepthConcatSubTensorMutator";
}

void DepthConcatSubTensorMutator::mutate(Graph &g)
{
    // Early exit if no Concatenation layers exist in graph
    if(g.nodes(NodeType::ConcatenateLayer).empty())
    {
        return;
    }

    // Perform topological sort
    std::vector<NodeID> topological_sorted_node_ids = dfs(g);

    // Should be in reverse order of execution
    for(auto &node_id : arm_compute::utils::iterable::reverse_iterate(topological_sorted_node_ids))
    {
        INode *node = g.node(node_id);
        if(node != nullptr && node->type() == NodeType::ConcatenateLayer && node->output(0) != nullptr)
        {
            // Get output tensor
            auto output_tensor = node->output(0);

            // Check concatenation axis (Sub-tensor optimization is supported for concatenation axis >=2)
            auto *concat_node = arm_compute::utils::cast::polymorphic_downcast<ConcatenateLayerNode *>(node);
            if(output_tensor == nullptr || get_dimension_idx(output_tensor->desc().layout, concat_node->concatenation_axis()) < 2)
            {
                continue;
            }

            // Check that all tensor have the same target, valid inputs and same quantization info
            bool is_valid = std::all_of(node->input_edges().cbegin(), node->input_edges().cend(),
                                        [&](const EdgeID & eid)
            {
                return (g.edge(eid) != nullptr) && (g.edge(eid)->tensor() != nullptr) && (g.edge(eid)->tensor()->desc().target == output_tensor->desc().target)
                       && (g.edge(eid)->tensor()->desc().quant_info == output_tensor->desc().quant_info);
            });

            // Create subtensors
            if(is_valid && is_target_supported(output_tensor->desc().target))
            {
                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Using sub-tensors for the node with ID : "
                                              << node->id() << " and name : " << node->name() << std::endl);
                // Create sub-tensor handles
                unsigned depth = 0;
                for(unsigned int i = 0; i < node->input_edges().size(); ++i)
                {
                    auto       input_tensor = node->input(i);
                    const auto input_shape  = input_tensor->desc().shape;

                    backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(input_tensor->desc().target);
                    std::unique_ptr<ITensorHandle> handle  = backend.create_subtensor(output_tensor->handle(), input_shape, Coordinates(0, 0, depth), false);
                    input_tensor->set_handle(std::move(handle));

                    depth += input_shape.z();
                }

                auto *dc_node = arm_compute::utils::cast::polymorphic_downcast<ConcatenateLayerNode *>(node);
                dc_node->set_enabled(false);
            }
        }
    }
}
} // namespace graph
} // namespace arm_compute
