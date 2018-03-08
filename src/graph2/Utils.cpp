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
#include "arm_compute/graph2/Utils.h"

#include "arm_compute/graph2/GraphContext.h"
#include "arm_compute/graph2/backends/BackendRegistry.h"
#include "arm_compute/graph2/mutators/GraphMutators.h"

namespace arm_compute
{
namespace graph2
{
bool is_target_supported(Target target)
{
    return backends::BackendRegistry::get().contains(target);
}

Target get_default_target()
{
    if(is_target_supported(Target::NEON))
    {
        return Target::NEON;
    }
    if(is_target_supported(Target::CL))
    {
        return Target::CL;
    }

    ARM_COMPUTE_ERROR("No backend exists!");
}

void force_target_to_graph(Graph &g, Target target)
{
    auto &nodes = g.nodes();
    for(auto &node : nodes)
    {
        if(node)
        {
            node->set_assigned_target(target);
        }
    }

    auto &tensors = g.tensors();
    for(auto &tensor : tensors)
    {
        if(tensor)
        {
            tensor->desc().target = target;
        }
    }
}

PassManager create_default_pass_manager()
{
    PassManager pm;

    pm.append(support::cpp14::make_unique<InPlaceOperationMutator>());
    pm.append(support::cpp14::make_unique<NodeFusionMutator>());
    pm.append(support::cpp14::make_unique<SplitLayerSubTensorMutator>());
    pm.append(support::cpp14::make_unique<DepthConcatSubTensorMutator>());

    return pm;
}

/** Default setups a graph Context
 *
 * @param[in] ctx Context to default initialize
 */
void setup_default_graph_context(GraphContext &ctx)
{
    for(const auto &backend : backends::BackendRegistry::get().backends())
    {
        backend.second->setup_backend_context(ctx);
    }
}
} // namespace graph2
} // namespace arm_compute