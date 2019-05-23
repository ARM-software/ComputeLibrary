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

    const bool is_target_gc = target == Target::GC;

    // Passes that mutate graph IR
    pm.append(support::cpp14::make_unique<NodeFusionMutator>(), !is_target_gc);
    pm.append(support::cpp14::make_unique<GroupedConvolutionMutator>());
    pm.append(support::cpp14::make_unique<InPlaceOperationMutator>(), !is_target_gc);

    // Passes that mutate backend information
    pm.append(support::cpp14::make_unique<DepthConcatSubTensorMutator>(), !is_target_gc);
    pm.append(support::cpp14::make_unique<SplitLayerSubTensorMutator>(), !is_target_gc);
    pm.append(support::cpp14::make_unique<NodeExecutionMethodMutator>());

    return pm;
}

void release_default_graph_context(GraphContext &ctx)
{
    for(const auto &backend : backends::BackendRegistry::get().backends())
    {
        if(backend.second->is_backend_supported())
        {
            backend.second->release_backend_context(ctx);
        }
    }
}

void setup_requested_backend_context(GraphContext &ctx, Target target)
{
    if(backends::BackendRegistry::get().contains(target))
    {
        const auto &backend = backends::BackendRegistry::get().find_backend(target);
        if(backend->is_backend_supported())
        {
            backend->setup_backend_context(ctx);
        }
    }
}

size_t get_dimension_size(const TensorDescriptor &descriptor, const DataLayoutDimension data_layout_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(descriptor.layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");
    return descriptor.shape[get_dimension_idx(descriptor.layout, data_layout_dimension)];
}

size_t get_dimension_idx(DataLayout data_layout, const DataLayoutDimension data_layout_dimension)
{
    ARM_COMPUTE_ERROR_ON_MSG(data_layout == DataLayout::UNKNOWN, "Cannot retrieve the dimension index for an unknown layout!");

    /* Return the index based on the data layout
     * [N C H W]
     * [3 2 1 0]
     * [N H W C]
     */
    switch(data_layout_dimension)
    {
        case DataLayoutDimension::CHANNEL:
            return (data_layout == DataLayout::NCHW) ? 2 : 0;
            break;
        case DataLayoutDimension::HEIGHT:
            return (data_layout == DataLayout::NCHW) ? 1 : 2;
            break;
        case DataLayoutDimension::WIDTH:
            return (data_layout == DataLayout::NCHW) ? 0 : 1;
            break;
        case DataLayoutDimension::BATCHES:
            return 3;
            break;
        default:
            ARM_COMPUTE_ERROR("Data layout index not supported!");
            break;
    }
}

std::vector<NodeIdxPair> get_driving_nodes(const INode &node)
{
    std::vector<NodeIdxPair> driving_nodes;

    const Graph *g = node.graph();
    ARM_COMPUTE_ERROR_ON(g == nullptr);

    for(auto &output_edge_id : node.output_edges())
    {
        auto output_edge = g->edge(output_edge_id);
        if(output_edge != nullptr)
        {
            ARM_COMPUTE_ERROR_ON(output_edge->consumer() == nullptr);
            driving_nodes.push_back({ output_edge->consumer_id(), output_edge->consumer_idx() });
        }
    }

    return driving_nodes;
}

void configure_tensor(Tensor *tensor)
{
    if(tensor != nullptr && tensor->handle() == nullptr)
    {
        Target                         target  = tensor->desc().target;
        backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(target);
        std::unique_ptr<ITensorHandle> handle  = backend.create_tensor(*tensor);
        ARM_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
        tensor->set_handle(std::move(handle));
    }
}
} // namespace graph
} // namespace arm_compute
