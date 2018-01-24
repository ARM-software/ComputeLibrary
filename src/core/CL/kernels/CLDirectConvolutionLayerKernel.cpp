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
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
/** Calculates expected output shape dimension
 *
 * @param[in] Input shape
 *
 * @return Expected output shape
 */
TensorShape get_output_shape(TensorShape input_shape, TensorShape weights_shape, PadStrideInfo conv_info)
{
    unsigned int output_width  = 0;
    unsigned int output_height = 0;
    std::tie(output_width, output_height) = scaled_dimensions(input_shape.x(), input_shape.y(), weights_shape.x(), weights_shape.y(), conv_info);

    TensorShape output_shape = input_shape;
    output_shape.set(0, output_width);
    output_shape.set(1, output_height);
    output_shape.set(2, weights_shape[3]);

    return output_shape;
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QS8, DataType::QASYMM8, DataType::QS16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) != weights->dimension(1),
                                    "Weights should have same width as length");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) != 1 && weights->dimension(0) != 3 && weights->dimension(0) != 5,
                                    "Kernel sizes other than 1x1, 3x3 or 5x5 are not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(2) != input->dimension(2),
                                    "Weights feature map dimension should match the respective input's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(0) != weights->dimension(1),
                                    "Only rectangular weights are supported!");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4,
                                    "Weights can be at most 4 dimensional");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(0) == 1) && std::get<0>(conv_info.stride()) > 3,
                                    "Strides larger than 3 not supported for 1x1 convolution.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(0) == 3 || weights->dimension(0) == 5) && std::get<0>(conv_info.stride()) > 2,
                                    "Strides larger than 2 not supported for 3x3 convolution.");

    if(biases != nullptr)
    {
        if(is_data_type_quantized_asymmetric(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(0) != weights->dimension(3),
                                        "Biases size and number of input feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1,
                                        "Biases should be one dimensional");
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(),
                                                           get_output_shape(input->tensor_shape(), weights->tensor_shape(), conv_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, const GPUTarget target)
{
    const unsigned int kernel_size = weights->dimension(0);
    const DataType     data_type   = input->data_type();

    // Get convolved dimensions
    TensorShape output_shape = get_output_shape(input->tensor_shape(), weights->tensor_shape(), conv_info);

    // Output auto inizialitation if not yet initialized
    //input->clone()->set_tensor_shape(output_shape) doesn't work with subtensors for grouped direct convolutions (AlexNet).
    auto_init_if_empty(*output, output_shape,
                       1,
                       input->data_type(),
                       input->fixed_point_position(),
                       input->quantization_info());

    unsigned int conv_stride_x   = std::get<0>(conv_info.stride());
    unsigned int conv_stride_y   = std::get<1>(conv_info.stride());
    unsigned int conv_pad_left   = std::max(conv_info.pad_left(), kernel_size / 2);
    unsigned int conv_pad_top    = std::max(conv_info.pad_top(), kernel_size / 2);
    unsigned int conv_pad_right  = std::max(conv_info.pad_right(), kernel_size / 2);
    unsigned int conv_pad_bottom = std::max(conv_info.pad_bottom(), kernel_size / 2);

    unsigned int num_elems_read_per_iteration_x    = 0;
    unsigned int num_elems_read_per_iteration_y    = 0;
    unsigned int num_elems_written_per_iteration_x = 0;
    unsigned int num_elems_written_per_iteration_y = 0;

    if((target == GPUTarget::BIFROST) && (kernel_size <= 5) && (conv_stride_x == 1) && (conv_stride_y == 1) && (data_type == DataType::F32))
    {
        // Configure kernel window

        switch(kernel_size)
        {
            case 1:
            {
                num_elems_read_per_iteration_x    = 4;
                num_elems_read_per_iteration_y    = 4;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 4;
                break;
            }
            case 3:
            {
                num_elems_read_per_iteration_x    = 6;
                num_elems_read_per_iteration_y    = 5;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 3;
                break;
            }
            case 5:
            {
                num_elems_read_per_iteration_x    = 8;
                num_elems_read_per_iteration_y    = 6;
                num_elems_written_per_iteration_x = 4;
                num_elems_written_per_iteration_y = 2;
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Kernel size not optimized for Bifrost");
            }
        }
    }
    else
    {
        num_elems_read_per_iteration_y    = kernel_size;
        num_elems_written_per_iteration_x = 8;
        num_elems_written_per_iteration_y = 1;
        switch(kernel_size)
        {
            case 1:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_x = 8;
                        break;
                    case 2:
                        num_elems_read_per_iteration_x = 16;
                        break;
                    case 3:
                        switch(input->element_size())
                        {
                            case 1:
                                num_elems_read_per_iteration_x = 28;
                                break;
                            case 2:
                                num_elems_read_per_iteration_x = 24;
                                break;
                            case 4:
                                num_elems_read_per_iteration_x = 22;
                                break;
                            default:
                                ARM_COMPUTE_ERROR("Invalid data size");
                        }
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            case 3:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_x = 10;
                        break;
                    case 2:
                        num_elems_read_per_iteration_x = 17;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            case 5:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_x = 12;
                        break;
                    case 2:
                        num_elems_read_per_iteration_x = 20;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid direct convolution size");
        }
    }

    // Calculate right and bottom border
    int input_width  = input->dimension(0) + conv_pad_left + conv_pad_right;
    int input_height = input->dimension(1) + conv_pad_top + conv_pad_bottom;

    // Add padding only if necessary or it would always result in a window_changed
    input_width  = ceil_to_multiple(input_width, num_elems_read_per_iteration_x);
    input_height = ceil_to_multiple(input_height, num_elems_read_per_iteration_y);

    // Create window and update padding
    bool   window_changed = false;
    Window win            = calculate_max_window(*output, Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y));

    AccessWindowStatic    input_access(input, -conv_pad_left, -conv_pad_top, input_width, input_height);
    AccessWindowStatic    weights_access(weights, 0, 0, kernel_size, kernel_size);
    AccessWindowRectangle output_access(output, 0, 0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_y);

    window_changed = update_window_and_padding(win, input_access, weights_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLDirectConvolutionLayerKernel::CLDirectConvolutionLayerKernel()
    : _input(nullptr), _biases(nullptr), _weights(nullptr), _output(nullptr), _border_size(0), _conv_stride_x(0), _conv_stride_y(0)
{
}

BorderSize CLDirectConvolutionLayerKernel::border_size() const
{
    return _border_size;
}

void CLDirectConvolutionLayerKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    const unsigned int kernel_size = weights->info()->dimension(0);
    const DataType     data_type   = input->info()->data_type();

    // Get convolved dimensions
    TensorShape output_shape = get_output_shape(input->info()->tensor_shape(), weights->info()->tensor_shape(), conv_info);

    // Output auto inizialitation if not yet initialized
    //input->clone()->set_tensor_shape(output_shape) doesn't work with subtensors for grouped direct convolutions (AlexNet).
    auto_init_if_empty(*output->info(),
                       output_shape,
                       1,
                       input->info()->data_type(),
                       input->info()->fixed_point_position(),
                       input->info()->quantization_info());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  weights->info(),
                                                  (biases != nullptr) ? biases->info() : nullptr,
                                                  output->info(),
                                                  conv_info));

    _conv_stride_x = std::get<0>(conv_info.stride());
    _conv_stride_y = std::get<1>(conv_info.stride());

    _input   = input;
    _weights = weights;
    _output  = output;
    _biases  = biases;

    int conv_pad_left   = std::min(conv_info.pad_left(), kernel_size / 2);
    int conv_pad_top    = std::min(conv_info.pad_top(), kernel_size / 2);
    int conv_pad_right  = std::min(conv_info.pad_right(), kernel_size / 2);
    int conv_pad_bottom = std::min(conv_info.pad_bottom(), kernel_size / 2);
    _border_size        = BorderSize(conv_pad_top, conv_pad_right, conv_pad_bottom, conv_pad_left);

    const GPUTarget gpu_target = get_arch_from_target(get_target());

    std::stringstream kernel_name;
    kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;

    CLBuildOptions build_options;
    build_options.add_option_if(_biases != nullptr, std::string("-DHAS_BIAS"));

    if((gpu_target == GPUTarget::BIFROST) && (kernel_size <= 5) && (_conv_stride_x == 1) && (_conv_stride_y == 1) && (data_type == DataType::F32))
    {
        build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(2))));

        kernel_name << "_f32_bifrost";
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str(), build_options.options()));

        // Through extensive experimentation with over 30 representative tensor
        // shapes, we found a small number of local work size configurations
        // that result in nearly optimal execution times. Selecting the right
        // lws for a given shape, however, required a complex decision tree,
        // until we constructed a simple feature as described below.
        //
        // We started from the number of multiply-accumulate operations for a
        // convolution layer, which is equal to the product of the input
        // dimensions 0..2 and the weights dimensions 0..2.  Unfortunately,
        // this resulted in ties between distinct shapes that required distinct
        // lws configurations. Replacing the width of the input with the kernel
        // size, however, resulted in nearly optimal predictions. We use underscores
        // in variable names to indicate when they are intentionally misleading.
        const size_t product_of_weights_dimensions = weights->info()->dimension(0) * weights->info()->dimension(1) * weights->info()->dimension(2);
        const size_t product_of_input_dimensions_  = input->info()->dimension(0) * weights->info()->dimension(1) * input->info()->dimension(2);
        const float  mega_ops_                     = 1e-6 * product_of_weights_dimensions * product_of_input_dimensions_;

        switch(kernel_size)
        {
            case 1:
            {
                if(mega_ops_ < 1.f)
                {
                    _lws_hint = cl::NDRange(1, 1, 8);
                }
                else if(mega_ops_ < 7.f)
                {
                    _lws_hint = cl::NDRange(1, 1, 4);
                }
                else
                {
                    _lws_hint = cl::NDRange(1, 1, 2);
                }
                break;
            }
            case 3:
            {
                if(mega_ops_ < 1.f)
                {
                    _lws_hint = cl::NDRange(1, 1, 8);
                }
                else if(mega_ops_ < 13.f)
                {
                    _lws_hint = cl::NDRange(2, 1, 4);
                }
                else if(mega_ops_ < 50.f)
                {
                    _lws_hint = cl::NDRange(3, 1, 4);
                }
                else
                {
                    _lws_hint = cl::NDRange(2, 1, 6);
                }
                break;
            }
            case 5:
            {
                if(mega_ops_ < 2.f || mega_ops_ > 80.f)
                {
                    _lws_hint = cl::NDRange(2, 1, 4);
                }
                else
                {
                    _lws_hint = cl::NDRange(2, 1, 8);
                }
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Kernel size not optimized for Bifrost");
            }
        }
    }
    else
    {
        bool     is_quantized_fixed_point = is_data_type_fixed_point(data_type);
        bool     is_quantized_asymm       = is_data_type_quantized_asymmetric(data_type);
        DataType promoted_type            = (is_quantized_fixed_point) ? get_promoted_data_type(data_type) : data_type;

        build_options.add_option_if(is_quantized_asymm, std::string("-DKERNEL_SIZE=" + support::cpp11::to_string(kernel_size)));
        build_options.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type)));
        build_options.add_option(std::string("-DDATA_SIZE=" + get_data_size_from_data_type(data_type)));
        build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(2))));
        build_options.add_option(std::string("-DSTRIDE_X=" + support::cpp11::to_string(_conv_stride_x)));
        build_options.add_option_if(is_quantized_fixed_point,
                                    std::string("-DFIXED_POINT_POSITION=" + support::cpp11::to_string(input->info()->fixed_point_position())));
        build_options.add_option(std::string("-DDATA_TYPE_PROMOTED=" + get_cl_type_from_data_type(promoted_type)));

        // Create kernel
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(is_quantized_asymm ? "direct_convolution_1x1_3x3_5x5_quantized" : kernel_name.str(),
                                                                               build_options.options()));
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), output->info(), conv_info, gpu_target);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    // Set static kernel arguments
    if(is_data_type_quantized_asymmetric(data_type))
    {
        int output_multiplier = 0;
        int output_shift      = 0;

        float multiplier = _input->info()->quantization_info().scale * _weights->info()->quantization_info().scale / _output->info()->quantization_info().scale;
        ARM_COMPUTE_THROW_ON_ERROR(quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift));

        unsigned int idx = 3 * num_arguments_per_3D_tensor() + ((_biases != nullptr) ? num_arguments_per_1D_tensor() : 0) + 1;
        _kernel.setArg(idx++, -_input->info()->quantization_info().offset);
        _kernel.setArg(idx++, -_weights->info()->quantization_info().offset);
        _kernel.setArg(idx++, _output->info()->quantization_info().offset);
        _kernel.setArg(idx++, output_multiplier);
        _kernel.setArg(idx++, output_shift);
    }

    // Set config_id for enabling LWS tuning
    _config_id = "direct_convolution_";
    _config_id += lower_string(string_from_data_type(data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(kernel_size);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_pad_left);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_pad_top);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_pad_right);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_pad_bottom);
    _config_id += "_";
    _config_id += support::cpp11::to_string(_conv_stride_x);
    _config_id += "_";
    _config_id += support::cpp11::to_string(_conv_stride_y);
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLDirectConvolutionLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                const GPUTarget target)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), output->clone().get(), conv_info, target).first);

    return Status{};
}

void CLDirectConvolutionLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    Window slice  = window.first_slice_window_3D();
    Window win_in = window;

    win_in.adjust(Window::DimX, -_border_size.left, true);
    win_in.adjust(Window::DimY, -_border_size.top, true);
    win_in.set_dimension_step(Window::DimX, window.x().step() * _conv_stride_x);
    win_in.set_dimension_step(Window::DimY, window.y().step() * _conv_stride_y);

    Window slice_in = win_in.first_slice_window_3D();

    unsigned int idx1 = 2 * num_arguments_per_3D_tensor();
    add_3D_tensor_argument(idx1, _weights, slice);

    if(_biases != nullptr)
    {
        Window slice_biases;
        slice_biases.use_tensor_dimensions(_biases->info()->tensor_shape());
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
