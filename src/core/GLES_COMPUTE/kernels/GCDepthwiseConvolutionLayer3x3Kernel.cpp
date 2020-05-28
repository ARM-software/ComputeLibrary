/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/GLES_COMPUTE/kernels/GCDepthwiseConvolutionLayer3x3Kernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/GLES_COMPUTE/GCHelpers.h"
#include "arm_compute/core/GLES_COMPUTE/GCKernelLibrary.h"
#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"
#include "arm_compute/core/GLES_COMPUTE/IGCTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "support/StringSupport.h"

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

GCDepthwiseConvolutionLayer3x3Kernel::GCDepthwiseConvolutionLayer3x3Kernel()
    : _border_size(0), _input(), _output(), _weights(), _biases(), _conv_stride_x(0), _conv_stride_y(0), _conv_pad_left(0), _conv_pad_top(0), _lws(gles::NDRange(1U, 1U, 1U))
{
}

BorderSize GCDepthwiseConvolutionLayer3x3Kernel::border_size() const
{
    return _border_size;
}

void GCDepthwiseConvolutionLayer3x3Kernel::configure(const IGCTensor *input, const IGCTensor *weights, const IGCTensor *biases, IGCTensor *output, const PadStrideInfo &conv_info,
                                                     unsigned int depth_multiplier)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_ERROR_ON(weights->info()->dimension(0) != 3 || weights->info()->dimension(1) != 3);

    if(biases != nullptr)
    {
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        ARM_COMPUTE_ERROR_ON(biases->info()->dimension(0) != weights->info()->dimension(2));
        ARM_COMPUTE_ERROR_ON(biases->info()->num_dimensions() > 1);
    }

    // Get convolved dimensions
    const TensorShape output_shape = compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info, depth_multiplier);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(),
                       output_shape,
                       1,
                       input->info()->data_type());

    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON(output->info()->dimension(2) != weights->info()->dimension(2));

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
    std::set<std::string> options;

    options.emplace("#define DEPTH_MULTIPLIER " + support::cpp11::to_string(depth_multiplier));
    options.emplace("#define LOCAL_SIZE_X " + support::cpp11::to_string(_lws[0]));
    options.emplace("#define LOCAL_SIZE_Y " + support::cpp11::to_string(_lws[1]));
    options.emplace("#define LOCAL_SIZE_Z " + support::cpp11::to_string(_lws[2]));
    options.emplace("#define STRIDE_X " + support::cpp11::to_string(_conv_stride_x));
    options.emplace("#define STRIDE_Y " + support::cpp11::to_string(_conv_stride_y));

    std::string dt_name = (input->info()->data_type() == DataType::F32) ? "DATA_TYPE_FP32" : "DATA_TYPE_FP16";
    options.emplace(("#define " + dt_name));

    unsigned int num_elems_read_per_iteration_x    = 8;
    unsigned int num_elems_read_per_iteration_y    = 1;
    unsigned int num_elems_written_per_iteration_x = 4;
    unsigned int num_elems_written_per_iteration_y = 1;
    unsigned int num_elems_written_per_iteration_z = 1;

    if((_conv_stride_x == 1) && (_conv_stride_y == 1))
    {
        switch(input->info()->data_type())
        {
#define PROCESS_4X_3Y_1Z

            case DataType::F16:
#if defined(PROCESS_4X_3Y_1Z)
                options.emplace("#define PROCESS_4X_3Y_1Z");
                num_elems_read_per_iteration_y    = 5;
                num_elems_written_per_iteration_y = 3;
#endif /* PROCESS_4X_3Y_1Z */
#undef PROCESS_4X_3Y_1Z
                break;

            default:
                ARM_COMPUTE_ERROR("Current data type is not supported");
                break;
        }
    }
    else
    {
        switch(input->info()->data_type())
        {
            case DataType::F16:
                options.emplace("#define PROCESS_4X_1Y_1Z");
                break;

            default:
                ARM_COMPUTE_ERROR("Current data type is not supported");
                break;
        }
    }

    if(_biases != nullptr)
    {
        options.emplace("#define BIAS");
    }

    // Create kernel
    std::string kernel_name = "depthwise_convolution_3x3";
    _kernel                 = static_cast<GCKernel>(GCKernelLibrary::get().create_kernel(kernel_name, options));

    // Calculate output right and bottom border
    const int output_width          = output->info()->dimension(0);
    const int output_height         = output->info()->dimension(1);
    const int output_padding_right  = ceil_to_multiple(output_width, num_elems_written_per_iteration_x * _lws[0]) - output_width;
    const int output_padding_bottom = ceil_to_multiple(output_height, num_elems_written_per_iteration_y * _lws[1]) - output_height;

    // Calculate input right and bottom border
    const int input_width  = input->info()->dimension(0);
    const int input_height = input->info()->dimension(1);

    const int input_total_width  = std::max(int(input->info()->padding().left), int(_conv_pad_left)) + input_width + std::max(int(input->info()->padding().right), int(_conv_pad_left));
    const int input_total_height = std::max(int(input->info()->padding().top), int(_conv_pad_top)) + input_height + std::max(int(input->info()->padding().bottom), int(_conv_pad_top));

    const int input_padding_right  = ceil_to_multiple(input_total_width, num_elems_read_per_iteration_x * _lws[0]) - input_width - _conv_pad_left;
    const int input_padding_bottom = ceil_to_multiple(input_total_height, num_elems_read_per_iteration_y * _lws[1]) - input_height - _conv_pad_top;

    BorderSize border = BorderSize(0, output_padding_right, output_padding_bottom, 0);

    Window win = calculate_max_enlarged_window(*output->info(), Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y, num_elems_written_per_iteration_z), border);

    AccessWindowStatic input_access(input->info(), -_conv_pad_left, -_conv_pad_top, input_width + input_padding_right, input_height + input_padding_bottom);
    AccessWindowStatic weights_access = AccessWindowStatic(nullptr, 0, 0, 0, 0);
    AccessWindowStatic bias_access    = AccessWindowStatic(nullptr, 0, 0, 0, 1);

    switch(weights->info()->data_type())
    {
        case DataType::F16:
            weights_access = AccessWindowStatic(weights->info(), 0, 0, 4, 3);
            if(_biases != nullptr)
            {
                bias_access = AccessWindowStatic(_biases->info(), 0, 0, _biases->info()->dimension(0) + 1, 1);
            }
            break;

        default:
            ARM_COMPUTE_ERROR("Current data type is not supported");
            break;
    }

    AccessWindowStatic output_access(output->info(), 0, 0, output_width + output_padding_right, output_height + output_padding_bottom);

    if(_biases != nullptr)
    {
        update_window_and_padding(win, input_access, weights_access, bias_access, output_access);
    }
    else
    {
        update_window_and_padding(win, input_access, weights_access, output_access);
    }

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->info()->tensor_shape()));

    IGCKernel::configure(win);
}

void GCDepthwiseConvolutionLayer3x3Kernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    _kernel.use();

    _output->set_needs_shifting(true);

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
        add_1D_tensor_argument(idx, _biases, 4, slice_biases);
    }

    slice_out.shift(Window::DimX, -(_output->info()->padding()).left);

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, 1, slice_in);
        add_3D_tensor_argument(idx, _output, 2, slice_out);
        add_3D_tensor_argument(idx, _weights, 3, slice_weights);

        _kernel.update_shader_params();
        enqueue(*this, slice_out, _lws);
    }
    while(window.slide_window_slice_3D(slice_out) && win_in.slide_window_slice_3D(slice_in));
}
