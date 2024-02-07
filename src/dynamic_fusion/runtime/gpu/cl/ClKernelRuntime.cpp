/*
 * Copyright (c) 2022-2024 Arm Limited.
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
#include "ClKernelRuntime.h"

#include "arm_compute/core/CL/ICLTensor.h"

#include "src/core/CL/CLUtils.h"
#include "src/dynamic_fusion/runtime/gpu/cl/ckw_driver/GpuCkwKernelArgumentsHelpers.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelSourceCode.h"
#include "src/gpu/cl/ClKernelLibrary.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
using namespace arm_compute::opencl;

void ClKernelRuntime::configure(const ClCompileContext &compile_ctx, const GpuKernelSourceCode &code)
{
    // Create kernel from kernel source string
    opencl::ClKernelLibrary &klib = opencl::ClKernelLibrary::get();
    _kernel                       = static_cast<cl::Kernel>(compile_ctx.create_kernel(
                              code.name(),
                              code.name(), // program name has to be provided to differentiate between different unfusable components' kernels.
                              // Each program contains exactly one kernel
                              code.code(), klib.kernel_path() /* Kernel path: Used in cases of embedded kernels */,
                              code.build_options().options(), false /* Is source binary */));

    // Configure execution window
    IClKernel::configure_internal(code.window());

    // Set config id for lws tuning
    _config_id = code.config_id();

    // Set kernel arguments
    _arguments = code.arguments();
}

inline void ClKernelRuntime::add_kernel_argument(unsigned int                   &idx,
                                                 const GpuKernelArgumentBinding &arg,
                                                 const ICLTensor                *tensor,
                                                 std::vector<cl::Image2D>       &cl_images)
{
    switch (arg.type())
    {
        case GpuKernelArgumentBinding::Type::TensorStorage:
        {
            switch (arg.tensor_storage_type())
            {
                case TensorStorageType::ClBufferUint8Ptr:
                {
                    cl_add_buffer_argument(_kernel, idx, tensor->cl_buffer());
                    break;
                }
                case TensorStorageType::ClImage2dReadOnly:
                {
                    cl::Image2D tensor_image2d = create_image2d_from_tensor(tensor, CLImage2DType::ReadOnly);
                    cl_images.push_back(tensor_image2d);
                    cl_add_texture_argument(_kernel, idx, tensor_image2d);
                    break;
                }
                case TensorStorageType::ClImage2dWriteOnly:
                {
                    cl::Image2D tensor_image2d = create_image2d_from_tensor(tensor, CLImage2DType::WriteOnly);
                    cl_images.push_back(tensor_image2d);
                    cl_add_texture_argument(_kernel, idx, tensor_image2d);
                    break;
                }
                default:
                {
                    ARM_COMPUTE_ERROR("Do not accept other TensorStorageType");
                    break;
                }
            }
            break;
        }
        case GpuKernelArgumentBinding::Type::TensorComponent:
        {
            cl_add_tensor_component_argument(_kernel, idx, tensor, arg.tensor_component_type());
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Do not accept other types of kernel arguments");
            break;
        }
    }
}

void ClKernelRuntime::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();

    /// NOTE: Parameters extracted from old kernels. So far they seem to be constant
    /// but we may need to make them into another configuration passed from GpuWorkloadSourceCode if needed in the future
    constexpr bool skip_sliding_window  = false;
    constexpr bool use_dummy_work_items = false;

    unsigned int idx = 0;
    do
    {
        // Set kernel arguments
        // CLImages created from tensor arguments. Need to be retained until enqueue
        std::vector<cl::Image2D> cl_images;

        for (const auto &arg : _arguments)
        {
            auto tensor = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(arg.id()));
            ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
            ARM_COMPUTE_ERROR_ON_NULLPTR(tensor->info());
            add_kernel_argument(idx, arg, tensor, cl_images);
        }

        // Dispatch kernel
        enqueue(queue, *this, slice, lws_hint(), use_dummy_work_items);
    } while (skip_sliding_window && window.slide_window_slice_3D(slice));
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
