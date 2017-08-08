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
#include "arm_compute/core/CL/kernels/CLDirectConvolutionLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

CLDirectConvolutionLayerKernel::CLDirectConvolutionLayerKernel()
    : _input(nullptr), _biases(nullptr), _weights(nullptr), _output(nullptr), _border_size(0), _conv_pad_x(0), _conv_pad_y(0), _conv_stride_x(0), _conv_stride_y(0)
{
}

BorderSize CLDirectConvolutionLayerKernel::border_size() const
{
    return _border_size;
}

void CLDirectConvolutionLayerKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON_MSG(weights->info()->dimension(0) != weights->info()->dimension(1),
                             "Only kernel sizes 1x1 and 3x3 are supported");
    ARM_COMPUTE_ERROR_ON_MSG(weights->info()->dimension(0) != 1 && weights->info()->dimension(0) != 3,
                             "Only kernel sizes 1x1 and 3x3 are supported");
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != weights->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON_MSG((weights->info()->dimension(0) == 1) && std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported for 1x1 convolution.");
    ARM_COMPUTE_ERROR_ON_MSG((weights->info()->dimension(0) == 3) && std::get<0>(conv_info.stride()) > 2, "Strides larger than 2 not supported for 3x3 convolution.");

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    const unsigned int kernel_size = weights->info()->dimension(0);

    // Get convolved dimensions
    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input->info()->dimension(0), input->info()->dimension(1), kernel_size, kernel_size, conv_info);

    TensorShape output_shape = input->info()->tensor_shape();
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);
    output_shape.set(2, weights->info()->dimension(3));

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _conv_stride_x = std::get<0>(conv_info.stride());
    _conv_stride_y = std::get<1>(conv_info.stride());
    _conv_pad_x    = std::min(std::get<0>(conv_info.pad()), kernel_size / 2);
    _conv_pad_y    = std::min(std::get<1>(conv_info.pad()), kernel_size / 2);

    _input       = input;
    _weights     = weights;
    _output      = output;
    _biases      = biases;
    _border_size = BorderSize(_conv_pad_y, _conv_pad_x);

    std::stringstream     kernel_name;
    std::set<std::string> options;
    kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;

    options.emplace("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    options.emplace("-DDATA_SIZE=" + get_data_size_from_data_type(input->info()->data_type()));
    options.emplace("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(2)));
    options.emplace("-DSTRIDE_X=" + support::cpp11::to_string(_conv_stride_x));

    if(_biases != nullptr)
    {
        options.emplace("-DHAS_BIAS");
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str(), options));

    // Configure kernel window
    Window win = calculate_max_window(*output->info());

    bool is_kernel3x3_stride2 = ((kernel_size == 3) && (_conv_stride_x == 2));

    const unsigned int num_elems_read_per_iteration_x    = 8 + 2 * (kernel_size / 2) + (is_kernel3x3_stride2 ? 7 : 0);
    const unsigned int num_elems_read_per_iteration_y    = kernel_size;
    const unsigned int num_elems_written_per_iteration_x = 8;
    const unsigned int num_elems_written_per_iteration_y = 1;

    // Calculate right and bottom border
    const int input_width  = input->info()->dimension(0) - kernel_size / 2 + _conv_pad_x;
    const int input_height = input->info()->dimension(1) - kernel_size / 2 + _conv_pad_y;

    // Create window and update padding
    win = calculate_max_window(*output->info(), Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y));

    AccessWindowStatic    input_access(input->info(), -_conv_pad_x, -_conv_pad_y, input_width + num_elems_read_per_iteration_x, input_height + num_elems_read_per_iteration_y);
    AccessWindowStatic    weights_access(weights->info(), 0, 0, kernel_size, kernel_size);
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_y);

    update_window_and_padding(win, input_access, weights_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLDirectConvolutionLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    Window slice  = window.first_slice_window_3D();
    Window win_in = window;

    win_in.adjust(Window::DimX, -_conv_pad_x, true);
    win_in.adjust(Window::DimY, -_conv_pad_y, true);
    win_in.set_dimension_step(Window::DimX, window.x().step() * _conv_stride_x);
    win_in.set_dimension_step(Window::DimY, window.y().step() * _conv_stride_y);

    Window slice_in = win_in.first_slice_window_3D();

    unsigned int idx1 = 2 * num_arguments_per_3D_tensor();
    add_3D_tensor_argument(idx1, _weights, slice);

    if(_biases != nullptr)
    {
        Window slice_biases;
        slice_biases.use_tensor_dimensions(_biases->info());
        add_1D_tensor_argument(idx1, _biases, slice_biases);
    }

    _kernel.setArg(idx1++, static_cast<unsigned int>(_weights->info()->strides_in_bytes()[3]));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice) && win_in.slide_window_slice_3D(slice_in));
}
