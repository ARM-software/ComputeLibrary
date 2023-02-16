/*
 * Copyright (c) 2022-2023 Arm Limited.
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

GpuKernelComponentStream GpuKernelComponentGraph::fuse(const MemoryDescriptorMap &mem_map) const
{
    GpuKernelComponentStream stream{ _services, mem_map };
    const auto               op_seq = _dependency_graph.build_operators_sequence();

    stream.new_component_group();
    for(auto op : op_seq)
    {
        const auto component = _components.at(op.op).get();
        const auto success   = stream.add_component(component);
        if(!success) // Assume first failure was because the root component is unfusable
        {
            stream.new_component_group();
            const auto success = stream.add_component(component);
            ARM_COMPUTE_ERROR_ON(!success);
            ARM_COMPUTE_UNUSED(success);
        }
    }

    return stream;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
