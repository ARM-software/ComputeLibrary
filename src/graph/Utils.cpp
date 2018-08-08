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
#include "arm_compute/graph/Utils.h"

#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include "arm_compute/graph/mutators/GraphMutators.h"

namespace arm_compute
{
namespace graph
{
bool is_target_supported(Target target)
{
    return backends::BackendRegistry::get().contains(target) && backends::BackendRegistry::get().find_backend(target)->is_backend_supported();
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
    if(is_target_supported(Target::GC))
    {
        return Target::GC;
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

PassManager create_default_pass_manager(Target target)
{
    PassManager pm;

    if(target != Target::GC)
    {
        pm.append(support::cpp14::make_unique<NodeFusionMutator>());
        pm.append(support::cpp14::make_unique<InPlaceOperationMutator>());
        pm.append(support::cpp14::make_unique<DepthConcatSubTensorMutator>());
        pm.append(support::cpp14::make_unique<SplitLayerSubTensorMutator>());
    }

    return pm;
}

void release_default_graph_context(GraphContext &ctx)
{
    for(const auto &backend : backends::BackendRegistry::get().backends())
    {
        backend.second->release_backend_context(ctx);
    }
}

void setup_default_graph_context(GraphContext &ctx)
{
    for(const auto &backend : backends::BackendRegistry::get().backends())
    {
        backend.second->setup_backend_context(ctx);
    }
}

size_t get_dimension_size(const TensorDescriptor &descriptor, const DataLayoutDimension data_layout_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(descriptor.layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");
    return descriptor.shape[get_dimension_idx(descriptor, data_layout_dimension)];
}

size_t get_dimension_idx(const TensorDescriptor &descriptor, const DataLayoutDimension data_layout_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(descriptor.layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");

    /* Return the index based on the data layout
     * [N C H W]
     * [3 2 1 0]
     * [N H W C]
     */
    switch(data_layout_dimension)
    {
        case DataLayoutDimension::CHANNEL:
            return (descriptor.layout == DataLayout::NCHW) ? 2 : 0;
            break;
        case DataLayoutDimension::HEIGHT:
            return (descriptor.layout == DataLayout::NCHW) ? 1 : 2;
            break;
        case DataLayoutDimension::WIDTH:
            return (descriptor.layout == DataLayout::NCHW) ? 0 : 1;
            break;
        case DataLayoutDimension::BATCHES:
            return 3;
            break;
        default:
            ARM_COMPUTE_ERROR("Data layout index not supported!");
            break;
    }
}
} // namespace graph
} // namespace arm_compute
