/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClCropKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"

#include <map>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
ClCropKernel::ClCropKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClCropKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src, ITensorInfo *dst, Coordinates2D start, Coordinates2D end, uint32_t batch_index,
                             float extrapolation_value, Window *dst_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate(src, dst, start, end, batch_index, extrapolation_value, dst_window));

    _start               = start;
    _batch_index         = batch_index;
    _extrapolation_value = extrapolation_value;

    const uint32_t vec_size_x = 4;
    // Create and update the window (if needed)
    Window win = calculate_max_window(*dst);

    if(dst_window != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(win, *dst_window);
        win = *dst_window;
    }

    const uint32_t dst_width_x    = win.num_iterations(0);
    const bool     multi_access_x = dst_width_x >= vec_size_x;
    const bool     remainder_x    = dst_width_x % vec_size_x > 0;

    if(multi_access_x)
    {
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option_if(multi_access_x && remainder_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(dst_width_x - vec_size_x, 0)));
    build_opts.add_option_if(start.x > end.x, "-DWIDTH_FLIPPED=");
    build_opts.add_option_if(start.y > end.y, "-DHEIGHT_FLIPPED=");
    _kernel = create_kernel(compile_context, "crop_tensor", build_opts.options());
}

Status ClCropKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value, Window *dst_window)
{
    ARM_COMPUTE_UNUSED(extrapolation_value, dst_window);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON(src->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(src, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON(src->tensor_shape().num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(start.x < 0 || start.y < 0 || end.x < 0 || end.y < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(start.x >= static_cast<int32_t>(src->dimension(1)) || start.y >= static_cast<int32_t>(src->dimension(2))
                                || end.x >= static_cast<int32_t>(src->dimension(1)) || end.y >= static_cast<int32_t>(src->dimension(2)));
    ARM_COMPUTE_RETURN_ERROR_ON(batch_index >= src->dimension(3));
    if(dst_window != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(dst_window->x().step() != 1);
    }
    if(dst->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(dst, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON(dst->num_dimensions() > 3);
    }
    return Status{};
}

void ClCropKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));
    auto       dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    Window in_slice = Window();
    in_slice.use_tensor_dimensions(src->info()->tensor_shape());
    in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start(), ceil_to_multiple(in_slice.x().end(), window.x().step()), window.x().step()));
    in_slice.set(3, Window::Dimension(_batch_index, _batch_index + 1, 1));

    unsigned int idx = 0;
    add_3D_tensor_argument(idx, src, in_slice);
    add_3D_tensor_argument(idx, dst, window);
    add_argument(idx, _start.x);
    add_argument(idx, _start.y);
    enqueue(queue, *this, window, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
