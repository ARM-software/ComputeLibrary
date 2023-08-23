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
#include "GpuKernelComponentStream.h"

#include "src/dynamic_fusion/sketch/gpu/GpuLogicalKernel.h"
#include "src/dynamic_fusion/sketch/gpu/GpuWorkloadSourceCode.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuKernelComponentStream::GpuKernelComponentStream(GpuWorkloadContext *context, GpuComponentServices *services, const MemoryDescriptorMap &mem_map)
    : _context{ context }, _services{ services }, _component_groups{}, _mem_map{ mem_map }
{
}

GpuWorkloadSourceCode GpuKernelComponentStream::write_workload_code()
{
    GpuWorkloadSourceCode source_code;
    // Traverse through component groups and assemble workload together
    for(auto && group : _component_groups)
    {
        group.finalize();

        // Write kernel code
        GpuLogicalKernel          logical_kernel(_services, group);
        const GpuKernelSourceCode kernel_code = logical_kernel.write_kernel_code();
        // The whole unit workload stage is determined by the root component
        const auto unit_workload_stage = group.get_root_component()->properties().stage();
        source_code.add_unit_workload(kernel_code, unit_workload_stage, _mem_map, _context);
    }
    return source_code;
}

void GpuKernelComponentStream::new_component_group()
{
    _component_groups.emplace_back();
}

bool GpuKernelComponentStream::add_component(IGpuKernelComponent *component)
{
    ARM_COMPUTE_ERROR_ON(_component_groups.empty());
    return _component_groups.back().add_component(component);
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
