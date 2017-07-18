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
#include "arm_compute/core/Validate.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

template <unsigned int kernel_size>
CLDirectConvolutionLayerKernel<kernel_size>::CLDirectConvolutionLayerKernel()
    : _input(nullptr), _biases(nullptr), _weights(nullptr), _output(nullptr), _border_size(0), _conv_pad_x(0), _conv_pad_y(0), _conv_stride_x(0), _conv_stride_y(0)
{
}

template <unsigned int kernel_size>
BorderSize             CLDirectConvolutionLayerKernel<kernel_size>::border_size() const
{
    return _border_size;
}

template <unsigned int kernel_size>
void CLDirectConvolutionLayerKernel<kernel_size>::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    static_assert(kernel_size == 3, "Currently only 3x3 direct convolution is supported!");

    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights, output);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(2) != input->info()->dimension(2));
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != weights->info()->dimension(1));
    ARM_COMPUTE_ERROR_ON(weights->info()->num_dimensions() > 4);
    ARM_COMPUTE_ERROR_ON_MSG((kernel_size == 3 && std::get<0>(conv_info.stride()) > 2), "Strides larger than 2 not supported in 3x3 direct convolution!");

    ARM_COMPUTE_ERROR_ON(kernel_size != weights->info()->dimension(0));

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(3));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    _conv_stride_x = std::get<0>(conv_info.stride());
    _conv_stride_y = std::get<1>(conv_info.stride());
    _conv_pad_x    = std::get<0>(conv_info.pad());
    _conv_pad_y    = std::get<1>(conv_info.pad());

    _input       = input;
    _weights     = weights;
    _output      = output;
    _biases      = biases;
    _border_size = BorderSize(_conv_pad_y, _conv_pad_x);

    std::stringstream     kernel_name;
    std::set<std::string> options;
    kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;

    options.insert("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));

    options.emplace("-DSTRIDE_X=" + support::cpp11::to_string(_conv_stride_x));

    if(_biases != nullptr)
    {
        options.emplace("-DHAS_BIAS");
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str(), options));

    unsigned int idx = (_biases == nullptr) ? 3 * num_arguments_per_3D_tensor() : (num_arguments_per_1D_tensor() + 3 * num_arguments_per_3D_tensor());
    _kernel.setArg<cl_uint>(idx++, _weights->info()->strides_in_bytes()[3]); // weights_stride_w
    _kernel.setArg<cl_uint>(idx++, _weights->info()->dimension(2));          // filter depth

    // Using this local workgroup size gives better performance over others that have been tried.
    _lws_hint = cl::NDRange(4, 1, 8);

    // Configure kernel window
    Window win = calculate_max_window(*output->info());

    unsigned int num_elems_read_per_iteration    = 16 * _conv_stride_x;
    unsigned int num_elems_written_per_iteration = 8;

    // Calculate right and bottom border
    const int input_width    = input->info()->dimension(0);
    const int input_height   = input->info()->dimension(1);
    const int upper_bound_w  = ceil_to_multiple(((output->info()->dimension(0) - 1) * _conv_stride_x + kernel_size), num_elems_read_per_iteration) - _conv_pad_x - input_width;
    const int upper_bound_h  = ((output->info()->dimension(1) - 1) * _conv_stride_y - _conv_pad_y + kernel_size) - input_height;
    const int padding_right  = std::max(upper_bound_w, static_cast<int>(kernel_size));
    const int padding_bottom = std::max(upper_bound_h, static_cast<int>(kernel_size));

    // Create window and update padding
    win = calculate_max_window(*output->info(), Steps(num_elems_written_per_iteration));
    AccessWindowStatic input_access(input->info(), -_conv_pad_x, -_conv_pad_y, input_width + padding_right, input_height + padding_bottom);

    AccessWindowStatic     weights_access(weights->info(), 0, 0, kernel_size, kernel_size);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);
    update_window_and_padding(win, input_access, weights_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

template <unsigned int kernel_size>
void CLDirectConvolutionLayerKernel<kernel_size>::run(const Window &window, cl::CommandQueue &queue)
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

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice);

        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice) && win_in.slide_window_slice_3D(slice_in));
}

template class arm_compute::CLDirectConvolutionLayerKernel<3>;
