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
#include "GpuLogicalKernel.h"

#include "arm_compute/core/experimental/Types.h"

#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuComponentServices.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentStore.h"
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateWriter.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuLogicalKernel::GpuLogicalKernel(GpuComponentServices *services, const GpuKernelComponentGroup &components)
    : _services{ services }, _comp_group{ components }, _store_components{}
{
    add_load_store();
}

GpuKernelSourceCode GpuLogicalKernel::write_kernel_code()
{
    GpuKernelSourceCode code;
    ClTemplateWriter    writer{ _comp_group };

    code.name(writer.get_name());
    code.code(writer.get_code());
    code.arguments(writer.get_tensors());
    code.build_options(writer.get_build_options());
    code.config_id(writer.get_config_id());
    code.window(writer.get_window());

    return code;
}

void GpuLogicalKernel::add_load_store()
{
    const auto dst_tensors = _comp_group.get_dst_tensors();
    // Each dst tensor from the component group requires exactly one store component
    for(const auto &dst_tensor : dst_tensors)
    {
        ArgumentPack<ITensorInfo> tensors;
        // Pass same destination tensor to both source and destination of the store component
        // In other words, the addition of a store component does not create a new dst tensor
        // This way we avoid the issue of the dst tensor of the component group differs from that of a logical kernel
        // This may seem to violate the acyclic-ness of the component graph. But it is fine because at the point of
        // the construction of the logical kernel, we do not need a graph representation of components anymore
        // (the graph has been serialized)
        tensors.add_const_tensor(ACL_SRC_0, dst_tensor);
        tensors.add_const_tensor(ACL_DST_0, dst_tensor);

        auto store = _services->component_factory().create<ClComponentStore>(
                         _comp_group.get_root_component()->properties(), // Store component share the same properties as that of the root component
                         tensors);
        _store_components.push_back(std::move(store));
        auto success = _comp_group.add_component(_store_components.back().get());
        ARM_COMPUTE_ERROR_ON(!success); // It's guaranteed that any load store insertion should be successful
    }
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
