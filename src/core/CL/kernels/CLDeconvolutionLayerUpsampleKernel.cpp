/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDeconvolutionLayerUpsampleKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

CLDeconvolutionLayerUpsampleKernel::CLDeconvolutionLayerUpsampleKernel()
    : _input(nullptr), _output(nullptr), _inner_border(), _info()
{
}

Status CLDeconvolutionLayerUpsampleKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const BorderSize &inner_border,
                                                    const PadStrideInfo &info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_UNUSED(info);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(0) == 0);
    ARM_COMPUTE_RETURN_ERROR_ON(output->dimension(1) == 0);

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(i) != output->dimension(i));
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(inner_border.right > info.stride().first - 1, "inner_border_right must be smaller that stride_x");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(inner_border.top > info.stride().second - 1, "inner_border_top must be smaller that stride_y");

    return Status{};
}

void CLDeconvolutionLayerUpsampleKernel::configure(const ICLTensor *input, ICLTensor *output, const BorderSize &inner_border,
                                                   const PadStrideInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _input        = input;
    _output       = output;
    _inner_border = inner_border;
    _info         = info;

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(CLDeconvolutionLayerUpsampleKernel::validate(input->info(), output->info(), inner_border, info));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("deconvolution_upsample"));

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal output_access(output->info(), 0, 0, num_elems_processed_per_iteration);
    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLDeconvolutionLayerUpsampleKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const int out_start_x = _info.pad().first;
    const int out_end_x   = _output->info()->dimension(0) - _inner_border.right - _info.pad().first + _info.stride().first - 1;
    const int out_step_x  = _info.stride().first;

    const int out_start_y = _inner_border.top + _info.pad().second;
    const int out_end_y   = _output->info()->dimension(1) - _info.pad().second + _info.stride().second - 1;
    const int out_step_y  = _info.stride().second;

    Window slice_out = window.first_slice_window_2D();
    slice_out.set(Window::DimX, Window::Dimension(out_start_x, out_end_x, out_step_x));
    slice_out.set(Window::DimY, Window::Dimension(out_start_y, out_end_y, out_step_y));

    Window slice_in = window.first_slice_window_2D();

    do
    {
        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out);
    }
    while(window.slide_window_slice_2D(slice_in) && window.slide_window_slice_2D(slice_out));
}
