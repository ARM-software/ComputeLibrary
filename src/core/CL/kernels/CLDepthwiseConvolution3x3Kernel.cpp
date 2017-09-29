/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolution3x3Kernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"

using namespace arm_compute;

CLDepthwiseConvolution3x3Kernel::CLDepthwiseConvolution3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _conv_stride_x(0), _conv_stride_y(0), _conv_pad_x(0), _conv_pad_y(0)
{
}

BorderSize CLDepthwiseConvolution3x3Kernel::border_size() const
{
    return _border_size;
}

void CLDepthwiseConvolution3x3Kernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *weights, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != 3 || weights->info()->dimension(1) != 3);

    std::pair<unsigned int, unsigned int> expected_output = scaled_dimensions(input->info()->tensor_shape().x(), input->info()->tensor_shape().y(),
                                                                              weights->info()->tensor_shape().x(), weights->info()->tensor_shape().y(),
                                                                              conv_info);

    ARM_COMPUTE_UNUSED(expected_output);
    ARM_COMPUTE_ERROR_ON(expected_output.first != output->info()->tensor_shape().x());
    ARM_COMPUTE_ERROR_ON(expected_output.second != output->info()->tensor_shape().y());

    _input         = input;
    _output        = output;
    _weights       = weights;
    _conv_stride_x = conv_info.stride().first;
    _conv_stride_y = conv_info.stride().second;
    _conv_pad_x    = conv_info.pad().first;
    _conv_pad_y    = conv_info.pad().second;
    _border_size   = BorderSize(_conv_pad_y, _conv_pad_x);

    // Set build options
    ARM_COMPUTE_ERROR_ON(_conv_stride_x < 1 || _conv_stride_x > 3);
    std::set<std::string> options{ "-DCONV_STRIDE_X=" + support::cpp11::to_string(_conv_stride_x) };

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("depthwise_convolution_3x3", options));

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = 2;
    const unsigned int num_elems_written_per_iteration   = 2;
    const unsigned int num_elems_read_per_iteration      = 3 + _conv_stride_x;
    const unsigned int num_rows_read_per_iteration       = 3;

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration, _conv_stride_x, _conv_stride_y);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);
    AccessWindowStatic     weights_access(weights->info(), 0, 0, weights->info()->dimension(0), weights->info()->dimension(1));

    update_window_and_padding(win, input_access, weights_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLDepthwiseConvolution3x3Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window slice_in      = window.first_slice_window_3D();
    Window slice_out     = window.first_slice_window_3D();
    Window slice_weights = window.first_slice_window_3D();

    slice_in.adjust(Window::DimX, -_conv_pad_x, true);
    slice_in.adjust(Window::DimY, -_conv_pad_y, true);
    slice_in.set_dimension_step(Window::DimX, window.x().step() * _conv_stride_x);
    slice_in.set_dimension_step(Window::DimY, window.y().step() * _conv_stride_y);
    slice_weights.set_dimension_step(Window::DimX, 0);
    slice_weights.set_dimension_step(Window::DimY, 0);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice_out);
        add_3D_tensor_argument(idx, _weights, slice_weights);

        enqueue(queue, *this, slice_out);
    }
    while(window.slide_window_slice_3D(slice_out));
}
