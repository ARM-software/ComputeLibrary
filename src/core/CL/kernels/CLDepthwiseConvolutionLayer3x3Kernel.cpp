/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3Kernel.h"

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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLDepthwiseConvolutionLayer3x3Kernel::CLDepthwiseConvolutionLayer3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _biases(), _conv_stride_x(0), _conv_stride_y(0), _conv_pad_left(0), _conv_pad_top(0)
{
}

BorderSize CLDepthwiseConvolutionLayer3x3Kernel::border_size() const
{
    return _border_size;
}

void CLDepthwiseConvolutionLayer3x3Kernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F32);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != 3 || weights->info()->dimension(1) != 3);

    if(biases != nullptr)
    {
        if(is_data_type_quantized_asymmetric(weights->info()->data_type()))
        {
            ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(2));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    // Get convolved dimensions
    const TensorShape output_shape = compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(),
                       output_shape,
                       1,
                       input->info()->data_type(),
                       input->info()->fixed_point_position(),
                       input->info()->quantization_info());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);

    _input         = input;
    _output        = output;
    _weights       = weights;
    _biases        = biases;
    _conv_stride_x = conv_info.stride().first;
    _conv_stride_y = conv_info.stride().second;
    _conv_pad_left = conv_info.pad_left();
    _conv_pad_top  = conv_info.pad_top();
    _border_size   = BorderSize(_conv_pad_top, conv_info.pad_right(), conv_info.pad_bottom(), _conv_pad_left);

    // Set build options
    ARM_COMPUTE_ERROR_ON(_conv_stride_x < 1 || _conv_stride_x > 3);
    CLBuildOptions build_opts;
    build_opts.add_option("-DCONV_STRIDE_X=" + support::cpp11::to_string(_conv_stride_x));
    build_opts.add_option_if(_biases != nullptr, "-DHAS_BIAS");

    // Create kernel
    std::string kernel_name = is_data_type_quantized_asymmetric(_input->info()->data_type()) ? "depthwise_convolution_3x3_quantized" : "depthwise_convolution_3x3";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set static arguments
    if(is_data_type_quantized_asymmetric(_input->info()->data_type()))
    {
        float multiplier        = _input->info()->quantization_info().scale * _weights->info()->quantization_info().scale / _output->info()->quantization_info().scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        unsigned int idx = 3 * num_arguments_per_3D_tensor() + ((_biases != nullptr) ? num_arguments_per_1D_tensor() : 0);

        _kernel.setArg(idx++, -_input->info()->quantization_info().offset);
        _kernel.setArg(idx++, -_weights->info()->quantization_info().offset);
        _kernel.setArg(idx++, _output->info()->quantization_info().offset);
        _kernel.setArg(idx++, output_multiplier);
        _kernel.setArg(idx++, output_shift);
    }

    // Configure the local work size for Bifrost with a value obtained
    // via exhaustive autotuning for the MobileNets tensor shapes.
    const GPUTarget gpu_target = get_arch_from_target(get_target());
    if(gpu_target == GPUTarget::BIFROST)
    {
        // Assume uniform padding and striding.
        const size_t pad    = _conv_pad_left;
        const size_t stride = _conv_stride_x;
        const size_t width  = input->info()->dimension(0);
        if(pad == 1)
        {
            const size_t width_by_stride = width / stride;
            if(width_by_stride == 28) // 56/2 or 28/1
            {
                _lws_hint = cl::NDRange(7, 4, 3);
            }
            else if(width_by_stride == 14) // 28/2 or 14/1
            {
                _lws_hint = cl::NDRange(7, 7, 4);
            }
        }
        else if(pad == 0)
        {
            if(width >= 56) // 56 or 112
            {
                _lws_hint = cl::NDRange(8, 5, 2);
            }
            else if(width >= 14) // 14 or 28
            {
                _lws_hint = cl::NDRange(1, 5, 2);
            }
            else // 7
            {
                _lws_hint = cl::NDRange(1, 1, 2);
            }
        }
    }

    // Configure kernel window
    const unsigned int num_elems_processed_per_iteration = 8 / data_size_from_type(input->info()->data_type());
    const unsigned int num_elems_written_per_iteration   = num_elems_processed_per_iteration;
    const unsigned int num_elems_read_per_iteration      = 3 + (num_elems_processed_per_iteration - 1) * _conv_stride_x;
    const unsigned int num_rows_read_per_iteration       = 3;

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration, _conv_stride_x, _conv_stride_y);
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);
    AccessWindowStatic     weights_access(weights->info(), 0, 0, weights->info()->dimension(0), weights->info()->dimension(1));

    update_window_and_padding(win, input_access, weights_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    ICLKernel::configure(win);
}

void CLDepthwiseConvolutionLayer3x3Kernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Create input window and adjust
    Window win_in = window;
    win_in.adjust(Window::DimX, -_conv_pad_left, true);
    win_in.adjust(Window::DimY, -_conv_pad_top, true);
    win_in.set_dimension_step(Window::DimX, window.x().step() * _conv_stride_x);
    win_in.set_dimension_step(Window::DimY, window.y().step() * _conv_stride_y);

    Window slice_in      = win_in.first_slice_window_3D();
    Window slice_out     = window.first_slice_window_3D();
    Window slice_weights = window.first_slice_window_3D();
    slice_weights.set_dimension_step(Window::DimX, 0);
    slice_weights.set_dimension_step(Window::DimY, 0);

    // Set biases
    if(_biases != nullptr)
    {
        unsigned int idx = 3 * num_arguments_per_3D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(_biases->info()->tensor_shape());
        add_1D_tensor_argument(idx, _biases, slice_biases);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice_out);
        add_3D_tensor_argument(idx, _weights, slice_weights);

        enqueue(queue, *this, slice_out, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice_out) && win_in.slide_window_slice_3D(slice_in));
}
