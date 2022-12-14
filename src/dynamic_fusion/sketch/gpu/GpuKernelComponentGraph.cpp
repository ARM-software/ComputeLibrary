/*
 * Copyright (c) 2022 Arm Limited.
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
#include "GpuKernelComponentGraph.h"

#include "arm_compute/dynamic_fusion/sketch/MemoryDescriptor.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
namespace
{
/** Automatically create memory descriptors for all tensors in the graph
 *
 * @param[in] tensors @ref ITensorInfo map
 * @param[in] graph   @ref DependencyGraph of which the @p tensors are a part
 *
 * @return MemoryDescriptorMap  An assignment map of @ref MemoryDescriptors for each ITensorInfo in the graph
 */
MemoryDescriptorMap assign_memory_descriptors(const std::map<ITensorInfo::Id, const ITensorInfo *> tensors, const DependencyGraph &graph)
{
    const auto all_tensors = graph.all_tensors();
    const auto src_tensors = graph.global_src_tensors();
    const auto dst_tensors = graph.global_dst_tensors();
    const auto interm_tensors = graph.intermediate_tensors();

    MemoryDescriptorMap mem_map{};
    for(auto t_id : all_tensors)
    {
        const auto &tensor = tensors.at(t_id);
        // Only global src and dst tensors to the entire component graph are "User" tensors, which are user-specified memories
        if(is_in(t_id, src_tensors) || is_in(t_id, dst_tensors))
        {
            mem_map[t_id] = MemoryDescriptor{ MemoryType::User };
        }
        else if(is_in(t_id, interm_tensors))
        {
            mem_map[t_id] = MemoryDescriptor { MemoryType::NoAlloc };
        }
        else
        {
            AuxMemoryInfo aux_mem_info{ tensor->total_size() };
            mem_map[t_id] = MemoryDescriptor{ MemoryType::Auxiliary, aux_mem_info };
        }
    }
    return mem_map;
}

} // namespace

std::vector<DependencyGraph::TensorId> GpuKernelComponentGraph::get_tensor_ids(const std::vector<const ITensorInfo *> tensors)
{
    std::vector<DependencyGraph::TensorId> tensor_ids{};
    std::transform(
        std::begin(tensors), std::end(tensors),
        std::back_inserter(tensor_ids),
        [](const auto & t)
    {
        return t->id();
    });
    return tensor_ids;
}

GpuKernelComponentGraph::GpuKernelComponentGraph(GpuComponentServices *services)
    : _services{ services }, _components{}, _tensors{}, _dependency_graph{}
{
}

GpuKernelComponentStream GpuKernelComponentGraph::fuse() const
{
    // Obtain memory descriptor map
    const auto mem_map = assign_memory_descriptors(_tensors, _dependency_graph);

    GpuKernelComponentStream stream{ _services, mem_map };
    const auto op_seq = _dependency_graph.build_operators_sequence();

    stream.new_component_group();
    for(auto op : op_seq)
    {
        const auto component = _components.at(op.op).get();
        const auto success = stream.add_component(component);
        ARM_COMPUTE_ERROR_ON(!success);
        ARM_COMPUTE_UNUSED(success);
    }

    return stream;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
