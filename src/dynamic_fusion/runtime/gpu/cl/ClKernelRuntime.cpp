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
#include "ClKernelRuntime.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "src/core/CL/CLUtils.h"
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
    _kernel                       = static_cast<cl::Kernel>(compile_ctx.create_kernel(code.name(),
                                                                                      code.name(), // program name has to be provided to differentiate between different unfusable components' kernels.
                                                                                      code.code(),
                                                                                      klib.kernel_path() /* Kernel path: Used in cases of embedded kernels */,
                                                                                      code.build_options().options(),
                                                                                      false /* Is source binary */));

    // Configure execution window
    IClKernel::configure_internal(code.window());

    // Set config id for lws tuning
    _config_id = code.config_id();

    // Set kernel arguments
    _arguments = code.arguments();
}

inline void ClKernelRuntime::add_tensor_argument(unsigned int &idx, const GpuKernelArgumentInfo &arg, const ICLTensor *tensor, const Window &arg_slice, std::vector<cl::Image2D> &cl_images)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);

    switch(arg.type)
    {
        case GpuKernelArgumentInfo::Type::Scalar:
        {
            ARM_COMPUTE_ERROR("Unsupported yet");
            break;
        }

        case GpuKernelArgumentInfo::Type::Vector:
        {
            add_1D_tensor_argument(idx, tensor, arg_slice);
            break;
        }

        case GpuKernelArgumentInfo::Type::Image:
        {
            add_2D_tensor_argument(idx, tensor, arg_slice);
            break;
        }
        case GpuKernelArgumentInfo::Type::Image_Reinterpret_As_3D:
        {
            add_2D_tensor_argument(idx, tensor, arg_slice);
            const unsigned int total_cross_plane_pad = tensor->info()->padding().top + tensor->info()->padding().bottom;
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(total_cross_plane_pad));
            break;
        }
        case GpuKernelArgumentInfo::Type::Image_Export_To_ClImage2D:
        {
            const TensorShape shape2d(tensor->info()->dimension(0) / 4, tensor->info()->dimension(1) * tensor->info()->dimension(2) * tensor->info()->dimension(3));
            const size_t      image_row_pitch = tensor->info()->strides_in_bytes()[1];
            cl::Image2D       tensor_image2d  = create_image2d_from_buffer(CLKernelLibrary::get().context(), tensor->cl_buffer(), shape2d, tensor->info()->data_type(), image_row_pitch, CLImage2DType::ReadOnly);
            cl_images.push_back(tensor_image2d);
            _kernel.setArg(idx++, tensor_image2d);
            break;
        }

        case GpuKernelArgumentInfo::Type::Image_3D:
        {
            add_2D_tensor_argument(idx, tensor, arg_slice);
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(tensor->info()->strides_in_bytes()[2]));
            break;
        }
        case GpuKernelArgumentInfo::Type::Image_3D_Export_To_ClImage2D:
        {
            const TensorShape shape2d(tensor->info()->dimension(0) / 4, tensor->info()->dimension(1) * tensor->info()->dimension(2) * tensor->info()->dimension(3));
            const size_t      image_row_pitch = tensor->info()->strides_in_bytes()[1];
            cl::Image2D       tensor_image2d  = create_image2d_from_buffer(CLKernelLibrary::get().context(), tensor->cl_buffer(), shape2d, tensor->info()->data_type(), image_row_pitch, CLImage2DType::ReadOnly);
            cl_images.push_back(tensor_image2d);
            _kernel.setArg(idx++, tensor_image2d);
            _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(tensor->info()->strides_in_bytes()[2]));
            break;
        }

        case GpuKernelArgumentInfo::Type::Tensor_3D:
        {
            add_3D_tensor_argument(idx, tensor, arg_slice);
            break;
        }

        case GpuKernelArgumentInfo::Type::Tensor_4D:
        {
            add_4D_tensor_argument(idx, tensor, arg_slice);
            break;
        }
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Buffer:
        {
            add_4d_tensor_nhwc_argument(idx, tensor);
            break;
        }
        case GpuKernelArgumentInfo::Type::Tensor_4D_t_Image:
        {
            const size_t image_w        = tensor->info()->dimension(0) / 4;
            const size_t image_h        = tensor->info()->tensor_shape().total_size_upper(1);
            const size_t image_stride_y = tensor->info()->strides_in_bytes()[1];

            cl::Image2D tensor_image2d = create_image2d_from_buffer(CLKernelLibrary::get().context(), tensor->cl_buffer(),
                                                                    TensorShape(image_w, image_h), tensor->info()->data_type(), image_stride_y, CLImage2DType::ReadOnly);
            cl_images.push_back(tensor_image2d);

            _kernel.setArg(idx++, tensor_image2d);
            add_4d_tensor_nhwc_argument(idx, tensor);
            break;
        }
        case GpuKernelArgumentInfo::Type::Tensor_Special_0:
        {
            const ITensorInfo *info    = tensor->info();
            const Strides     &strides = info->strides_in_bytes();

            _kernel.setArg(idx++, tensor->cl_buffer());
            const size_t dim1xdim2 = info->tensor_shape()[1] * info->tensor_shape()[2];
            _kernel.setArg<cl_int>(idx++, static_cast<int32_t>(dim1xdim2));
            const size_t stride1 = strides[1];
            _kernel.setArg<cl_int>(idx++, static_cast<int32_t>(stride1));
            break;
        }
        default:
        {
            ARM_COMPUTE_ERROR("Unsupported");
        }
    }
}

void ClKernelRuntime::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice = window.first_slice_window_3D();
    // Don't slice matrix along the z dimension if matrix has just 2 dimensions and matrix A more than 2
    // This scenario can happen when the matrix multiplication is used to perform a convolution operation
    Window slice_fixed_z = slice;
    slice_fixed_z.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice_fixed_z.set(Window::DimY, Window::Dimension(0, 1, 1));

    /// NOTE: Parameters extracted from old kernels. So far they seem to be constant
    /// but we may need to make them into another configuration passed from GpuWorkloadSourceCode if needed in the future
    constexpr bool slide_along_dimz     = true;
    constexpr bool skip_sliding_window  = false;
    constexpr bool use_dummy_work_items = false;

    unsigned int idx = 0;
    do
    {
        // Set kernel arguments
        Window arg_slice = slice;
        // CLImages created from tensor arguments. Need to be retained until enqueue
        std::vector<cl::Image2D> cl_images;
        for(auto id_arg : _arguments)
        {
            const auto arg    = id_arg.second;
            auto       tensor = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(id_arg.first));
            ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
            ARM_COMPUTE_ERROR_ON_NULLPTR(tensor->info());
            if(!slide_along_dimz)
            {
                // The stride_z for matrix must be zero if we do not slice
                ARM_COMPUTE_ERROR_ON(tensor->info()->strides_in_bytes()[3] != 0);
                arg_slice = slice_fixed_z;
            }
            add_tensor_argument(idx, *arg.kernel_argument_info(), tensor, arg_slice, cl_images);
        }

        // Dispatch kernel
        enqueue(queue, *this, slice, lws_hint(), use_dummy_work_items);
    }
    while(skip_sliding_window && window.slide_window_slice_3D(slice));
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
