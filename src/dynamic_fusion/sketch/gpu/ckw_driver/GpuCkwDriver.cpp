/*
 * Copyright (c) 2023 Arm Limited.
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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwDriver.h"

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/IGpuCkwComponentDriver.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Window.h"
#include "src/common/utils/Log.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwVariableTable.h"

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwKernelWriter.h"
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"

using namespace ckw;
namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuCkwDriver::GpuCkwDriver(const GpuKernelComponentGroup &components)
    : _components{ components }
{
}

std::string GpuCkwDriver::get_name()
{
    ARM_COMPUTE_LOG_PARAMS(std::string("[V1] TODO"));
    return "todo_get_name";
}

std::string GpuCkwDriver::get_code()
{
    ARM_COMPUTE_LOG_PARAMS(std::string("[V1] TODO"));
    ckw::Kernel              kernel(get_name().c_str(), GpuTargetLanguage::OpenCL);
    GpuCkwKernelWriter       root_writer(kernel);
    GpuCkwScopedKernelWriter writer(&root_writer);
    GpuCkwVariableTable      vtable{};

    // Global Kernel Writer Driver code

    // The following is just an incomplete example of using the kernel writer

    // Iterate over component specific Ckw Driver; generate component code and concatenate them
    for(auto &comp : _components)
    {
        auto ckw_driver = comp->ckw_component_driver();
        ARM_COMPUTE_ERROR_ON(ckw_driver == nullptr);
        ckw_driver->write_component_code(_components, vtable, writer);
    }

    std::string code = root_writer.generate_code();

    return code;
}

CLBuildOptions GpuCkwDriver::get_build_options()
{
    ARM_COMPUTE_LOG_PARAMS(std::string("[V1] TO REMOVE"));
    return CLBuildOptions{};
}

std::string GpuCkwDriver::get_config_id()
{
    ARM_COMPUTE_LOG_PARAMS(std::string("[V1] TODO"));
    return "";
}

Window GpuCkwDriver::get_window() const
{
    const auto root_comp = _components.get_root_component();
    ARM_COMPUTE_ERROR_ON_MSG(root_comp == nullptr, "No root component found");
    return root_comp->ckw_component_driver()->get_window();
}

std::map<ITensorInfo::Id, GpuKernelArgument> GpuCkwDriver::get_tensors()
{
    ARM_COMPUTE_LOG_PARAMS(std::string("[V1] TODO"));
    // Assemble GpuKernelArguments
    std::map<ITensorInfo::Id, GpuKernelArgument> tensors;
    for(const auto t : _components.get_argument_tensors())
    {
        tensors.emplace(
            t->id(),
            GpuKernelArgument{ *t, { GpuKernelArgumentInfo::Type::Tensor_Special_0 } });
    }
    return tensors;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
