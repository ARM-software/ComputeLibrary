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
#include "arm_compute/graph/mutators/SplitLayerSubTensorMutator.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/nodes/SplitLayerNode.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/core/utils/misc/Iterable.h"

namespace arm_compute
{
namespace graph
{
const char *SplitLayerSubTensorMutator::name()
{
    return "SplitLayerSubTensorMutator";
}

void SplitLayerSubTensorMutator::mutate(Graph &g)
{
    // Should be in reverse order of execution
    for(auto &node : arm_compute::utils::iterable::reverse_iterate(g.nodes()))
    {
        if(node && node->type() == NodeType::SplitLayer && node->input(0) != nullptr)
        {
            // Get output tensor
            Tensor *input_tensor = node->input(0);

            // Check that all tensor have the same target and are valid
            bool is_valid = std::all_of(node->outputs().cbegin(), node->outputs().cend(),
                                        [&](const TensorID & tid)
            {
                return (g.tensor(tid) != nullptr) && (g.tensor(tid)->desc().target == input_tensor->desc().target);
            });

            // Create subtensors
            if(is_valid && backends::BackendRegistry::get().find_backend(input_tensor->desc().target) != nullptr)
            {
                ARM_COMPUTE_LOG_GRAPH_VERBOSE("Using sub-tensors for the node with ID : "
                                              << node->id() << " and name : " << node->name() << std::endl);

                auto *split_node = arm_compute::utils::cast::polymorphic_downcast<SplitLayerNode *>(node.get());

                const unsigned int axis          = split_node->axis();
                const unsigned int num_splits    = split_node->num_splits();
                const bool         extend_parent = (axis < 2);

                // Create sub-tensor handles
                for(unsigned int i = 0; i < node->outputs().size(); ++i)
                {
                    Tensor           *output_tensor = node->output(i);
                    const TensorShape output_shape  = output_tensor->desc().shape;
                    Coordinates       coords;
                    std::tie(std::ignore, coords) = SplitLayerNode::compute_output_shape(input_tensor->desc().shape, num_splits, axis, i);

                    backends::IDeviceBackend      *backend = backends::BackendRegistry::get().find_backend(output_tensor->desc().target);
                    std::unique_ptr<ITensorHandle> handle  = backend->create_subtensor(input_tensor->handle(), output_shape, coords, extend_parent);
                    output_tensor->set_handle(std::move(handle));
                }
            }
        }
    }
}
} // namespace graph
} // namespace arm_compute
