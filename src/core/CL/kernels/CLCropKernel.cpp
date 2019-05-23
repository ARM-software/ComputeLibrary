/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLCropKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Window.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/helpers/bit_ops.h"
#include "arm_compute/core/utils/helpers/tensor_transform.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <map>

namespace arm_compute
{
CLCropKernel::CLCropKernel()
    : _input(nullptr), _output(nullptr), _start(), _batch_index(0), _extrapolation_value(0)
{
}

void CLCropKernel::configure(const ICLTensor *input, ICLTensor *output, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value, Window *output_window)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), start, end, batch_index, extrapolation_value, output_window));

    _input               = input;
    _output              = output;
    _start               = start;
    _batch_index         = batch_index;
    _extrapolation_value = extrapolation_value;

    const int vec_size_x = 4;
    // Create and update the window (if needed)
    Window win = calculate_max_window(*output->info());

    if(output_window != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(win, *output_window);
        win = *output_window;
    }

    const int  output_width_x = win.num_iterations(0);
    const bool multi_access_x = output_width_x >= vec_size_x;
    const bool remainder_x    = output_width_x % vec_size_x > 0;

    if(multi_access_x)
    {
        win.set(Window::DimX,
                Window::Dimension(win.x().start(), ceil_to_multiple(win.x().end(), vec_size_x), vec_size_x));
    }
    ICLKernel::configure_internal(win);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option_if(multi_access_x, "-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option_if(multi_access_x && remainder_x, "-DLAST_ACCESSED_X=" + support::cpp11::to_string(std::max<int>(output_width_x - vec_size_x, 0)));
    build_opts.add_option_if(start.x > end.x, "-DWIDTH_FLIPPED=");
    build_opts.add_option_if(start.y > end.y, "-DHEIGHT_FLIPPED=");
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("crop_tensor", build_opts.options()));
}

Status CLCropKernel::validate(const ITensorInfo *input, const ITensorInfo *output, Coordinates2D start, Coordinates2D end, uint32_t batch_index, float extrapolation_value, Window *output_window)
{
    ARM_COMPUTE_UNUSED(extrapolation_value, output_window);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U16, DataType::S16, DataType::F16, DataType::U32, DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON(input->tensor_shape().num_dimensions() > 4);
    ARM_COMPUTE_RETURN_ERROR_ON(start.x < 0 || start.y < 0 || end.x < 0 || end.y < 0);
    ARM_COMPUTE_RETURN_ERROR_ON(start.x >= static_cast<int32_t>(input->dimension(1)) || start.y >= static_cast<int32_t>(input->dimension(2))
                                || end.x >= static_cast<int32_t>(input->dimension(1)) || end.y >= static_cast<int32_t>(input->dimension(2)));
    ARM_COMPUTE_RETURN_ERROR_ON(batch_index >= input->dimension(3));
    if(output_window != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(output_window->x().step() != 1);
    }
    if(output->total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(output, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 3);
    }
    return Status{};
}

void CLCropKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window in_slice = Window();
    in_slice.use_tensor_dimensions(_input->info()->tensor_shape());
    in_slice.set(Window::DimX, Window::Dimension(in_slice.x().start(), ceil_to_multiple(in_slice.x().end(), window.x().step()), window.x().step()));
    in_slice.set(3, Window::Dimension(_batch_index, _batch_index + 1, 1));

    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input, in_slice);
    add_3D_tensor_argument(idx, _output, window);
    add_argument(idx, _start.x);
    add_argument(idx, _start.y);
    enqueue(queue, *this, window);
}
} // namespace arm_compute
