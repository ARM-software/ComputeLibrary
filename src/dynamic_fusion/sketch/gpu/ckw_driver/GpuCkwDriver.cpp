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
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/utils/type_converter/Common.h"

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
    : _components{ components }, _kernel{ GpuTargetLanguage::OpenCL }, _code{}
{
    // Generate kernel name
    std::string name = "";
    for(auto &comp : _components)
    {
        auto ckw_driver = comp->ckw_component_driver();
        ARM_COMPUTE_ERROR_ON(ckw_driver == nullptr);
        name += ckw_driver->get_name(_components) + "__";
    }

    // Generate kernel code
    _kernel.name(name);
    GpuCkwKernelWriter       root_writer(_kernel);
    GpuCkwScopedKernelWriter writer(&root_writer);
    GpuCkwVariableTable      vtable{};

    for(auto &comp : _components)
    {
        auto ckw_driver = comp->ckw_component_driver();
        ARM_COMPUTE_ERROR_ON(ckw_driver == nullptr);
        ckw_driver->write_component_code(_components, vtable, writer);
    }
    _code = root_writer.generate_code();
}

std::string GpuCkwDriver::get_name()
{
    return _kernel.name();
}

std::string GpuCkwDriver::get_code()
{
    return _code;
}

std::string GpuCkwDriver::get_config_id()
{
    std::string id = "";
    for(auto &comp : _components)
    {
        auto ckw_driver = comp->ckw_component_driver();
        ARM_COMPUTE_ERROR_ON(ckw_driver == nullptr);
        id = ckw_driver->get_tuner_id(_components) + "__";
    }
    return id;
}

Window GpuCkwDriver::get_window() const
{
    const auto root_comp = _components.get_root_component();
    ARM_COMPUTE_ERROR_ON_MSG(root_comp == nullptr, "No root component found");
    return root_comp->ckw_component_driver()->get_window();
}

GpuKernelArgumentList GpuCkwDriver::get_kernel_arguments()
{
    GpuKernelArgumentList args{};
    for(const auto &arg : _kernel.arguments())
    {
        switch(arg.type())
        {
            case KernelArgument::Type::TensorStorage:
            {
                args.emplace_back(static_cast<ITensorInfo::Id>(arg.id()), from_ckw(arg.tensor_storage_type()));
                break;
            }
            case KernelArgument::Type::TensorComponent:
            {
                args.emplace_back(static_cast<ITensorInfo::Id>(arg.id()), from_ckw(arg.tensor_component_type()));
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Unsupported KernelArgument Type");
                break;
            }
        }
    }
    return args;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
