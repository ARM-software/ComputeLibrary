/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClCopyKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/Validate.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, Window *dst_window = nullptr)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, dst);

    // Validate dst if initialized
    if (dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(src, dst);
        if (dst_window == nullptr)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(src->tensor_shape(), dst->tensor_shape());
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(src->tensor_shape(), dst_window->shape());
        }
    }

    return Status{};
}

} // namespace

ClCopyKernel::ClCopyKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClCopyKernel::configure(const CLCompileContext &compile_context,
                             const ITensorInfo      *src,
                             ITensorInfo            *dst,
                             Window                 *dst_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, dst_window));

    auto padding_info = get_padding_info({src, dst});

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, *src);

    // Configure window
    const unsigned int vec_size_x = adjust_vec_size(16 / src->element_size(), src->dimension(0));

    const Window win_config = calculate_max_window(*src, Steps(vec_size_x));

    if (dst_window != nullptr)
    {
        _has_dst_window                = true;
        _dst_window                    = Window(*dst_window);
        const int  width_x             = dst_window->num_iterations(0);
        const int  vec_size_x_leftover = width_x % vec_size_x;
        const bool multi_access_x      = width_x >= static_cast<int32_t>(vec_size_x);

        if (multi_access_x)
        {
            _dst_window.set(Window::DimX,
                            Window::Dimension(dst_window->x().start(),
                                              ceil_to_multiple(dst_window->x().end(), vec_size_x), vec_size_x));
        }

        build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftover));
    }
    else
    {
        const int width_x             = src->tensor_shape().x();
        const int vec_size_x_leftover = width_x % vec_size_x;

        build_opts.add_option("-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftover));
    }

    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));

    // Build kernel
    _kernel = create_kernel(compile_context, "copy_tensor", build_opts.options());

    // Validate and set the window
    ICLKernel::configure_internal(win_config);

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status
ClCopyKernel::validate(const arm_compute::ITensorInfo *src, const arm_compute::ITensorInfo *dst, Window *dst_window)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, dst_window));

    return Status{};
}

void ClCopyKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window slice;

    if (_has_dst_window)
    {
        slice            = window.first_slice_window_3D();
        Window out_slice = _dst_window.first_slice_window_3D();
        do
        {
            unsigned int idx = 0;
            add_3D_tensor_argument(idx, src, slice);
            add_3D_tensor_argument(idx, dst, out_slice);
            enqueue(queue, *this, slice, lws_hint());
        } while (window.slide_window_slice_3D(slice) && _dst_window.slide_window_slice_3D(out_slice));
    }
    else
    {
        Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
        slice            = collapsed.first_slice_window_3D();
        do
        {
            unsigned int idx = 0;
            add_3D_tensor_argument(idx, src, slice);
            add_3D_tensor_argument(idx, dst, slice);
            enqueue(queue, *this, slice, lws_hint());
        } while (collapsed.slide_window_slice_3D(slice));
    }
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
