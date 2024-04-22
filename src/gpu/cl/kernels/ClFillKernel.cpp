/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/gpu/cl/kernels/ClFillKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
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
ClFillKernel::ClFillKernel()
{
    _type = CLKernelType::ELEMENTWISE;
}

void ClFillKernel::configure(const CLCompileContext &compile_context,
                             ITensorInfo            *tensor,
                             const PixelValue       &constant_value,
                             Window                 *window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(tensor);
    ARM_COMPUTE_ERROR_THROW_ON(validate(tensor, constant_value, window));

    const DataType data_type  = tensor->data_type();
    const int      vec_size_x = 16 / tensor->element_size();

    // Create and update the window (if needed)
    _full_window = calculate_max_window(*tensor);
    Window win   = _full_window;
    if (window != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(win, *window);
        win = *window;
    }

    const int  output_width_x = win.num_iterations(0);
    const bool multi_access_x = output_width_x >= vec_size_x;
    const bool remainder_x    = output_width_x % vec_size_x > 0;

    if (multi_access_x)
    {
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type));
    build_opts.add_option("-DCONSTANT_VALUE=" + string_from_pixel_value(constant_value, data_type));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option_if(multi_access_x && remainder_x,
                             "-DLAST_ACCESSED_X=" +
                                 support::cpp11::to_string(std::max<int>(output_width_x - vec_size_x, 0)));
    _kernel = create_kernel(compile_context, "memset", build_opts.options());
}

Status ClFillKernel::validate(const ITensorInfo *tensor, const PixelValue &constant_value, Window *window)
{
    ARM_COMPUTE_UNUSED(tensor);
    ARM_COMPUTE_UNUSED(constant_value);
    if (window != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(window->x().step() != 1);
    }
    return Status{};
}

void ClFillKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto tensor =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC));

    // Collapse all the batches on the third
    Window collapsed = window.collapse_if_possible(_full_window, Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, tensor, slice);
        enqueue(queue, *this, slice, lws_hint());
    } while (collapsed.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
