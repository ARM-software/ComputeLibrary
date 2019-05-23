/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    const DataLayout data_layout = input->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != weights->dimension(height_idx), "Weights should have same width and height");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != 1 && weights->dimension(width_idx) != 3 && weights->dimension(width_idx) != 5,
                                    "Kernel sizes other than 1x1, 3x3 or 5x5 are not supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != input->dimension(channel_idx),
                                    "Weights feature map dimension should match the respective input's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 1) && std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported for 1x1 convolution.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 3 || weights->dimension(width_idx) == 5) && std::get<0>(conv_info.stride()) > 2,
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
                                                           misc::shape_calculator::compute_deep_convolution_shape(*input, *weights, conv_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

inline bool can_run_optimized_kernel_for_bifrost(GPUTarget gpu_target, unsigned int conv_stride_x, unsigned int conv_stride_y, unsigned int kernel_size,
                                                 DataType data_type, DataLayout data_layout)
{
    return gpu_target_is_in(gpu_target,
                            GPUTarget::G71, GPUTarget::G72, GPUTarget::G76,
                            GPUTarget::G51, GPUTarget::G51BIG, GPUTarget::G51LIT,
                            GPUTarget::G52, GPUTarget::G52LIT)
           && (kernel_size <= 5)
           && (conv_stride_x == 1) && (conv_stride_y == 1)
           && (data_type == DataType::F32)
           && (data_layout == DataLayout::NCHW);
}

inline void setup_num_elems(unsigned int &num_elems_read_per_iteration_x, unsigned int &num_elems_read_per_iteration_y,
                            unsigned int &num_elems_written_per_iteration_x, unsigned int &num_elems_written_per_iteration_y,
                            unsigned int kernel_size, const PadStrideInfo &conv_info, const GPUTarget target, ITensorInfo *input)
{
    const DataType   data_type     = input->data_type();
    const DataLayout data_layout   = input->data_layout();
    unsigned int     conv_stride_x = std::get<0>(conv_info.stride());
    unsigned int     conv_stride_y = std::get<1>(conv_info.stride());

    const bool run_optimized_bifrost = can_run_optimized_kernel_for_bifrost(target, conv_stride_x, conv_stride_y, kernel_size, data_type, data_layout);

    if(run_optimized_bifrost)
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

    if(data_layout == DataLayout::NHWC)
    {
        num_elems_written_per_iteration_x = 1;
        num_elems_read_per_iteration_x    = 1;
        switch(kernel_size)
        {
            case 1:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_y    = 8;
                        num_elems_written_per_iteration_y = 8;
                        break;
                    case 2:
                        num_elems_read_per_iteration_y    = 16;
                        num_elems_written_per_iteration_y = 8;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            case 3:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_y    = 10;
                        num_elems_written_per_iteration_y = 8;
                        break;
                    case 2:
                        num_elems_read_per_iteration_y    = 17;
                        num_elems_written_per_iteration_y = 8;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            case 5:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_y    = 12;
                        num_elems_written_per_iteration_y = 8;
                        break;
                    case 2:
                        num_elems_read_per_iteration_y    = 20;
                        num_elems_written_per_iteration_y = 8;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented.");
                break;
        }
    }
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, const GPUTarget target)
{
    const DataLayout   data_layout = input->data_layout();
    const int          width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const unsigned int kernel_size = weights->dimension(width_idx);

    // Get convolved dimensions
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*input, *weights, conv_info);

    // Output auto inizialitation if not yet initialized
    // TODO(COMPMID-2078): input->clone()->set_tensor_shape(output_shape) doesn't work with subtensors for grouped direct convolutions (AlexNet).
    auto_init_if_empty(*output, output_shape,
                       1,
                       input->data_type(),
                       input->quantization_info());

    unsigned int num_elems_read_per_iteration_x    = 0;
    unsigned int num_elems_read_per_iteration_y    = 0;
    unsigned int num_elems_written_per_iteration_x = 0;
    unsigned int num_elems_written_per_iteration_y = 0;

    unsigned int conv_pad_left = conv_info.pad_left();
    unsigned int conv_pad_top  = conv_info.pad_top();
    unsigned int conv_stride_x = std::get<0>(conv_info.stride());
    unsigned int conv_stride_y = std::get<1>(conv_info.stride());

    setup_num_elems(num_elems_read_per_iteration_x, num_elems_read_per_iteration_y,
                    num_elems_written_per_iteration_x, num_elems_written_per_iteration_y,
                    kernel_size, conv_info, target, input);

    // Create window and update padding
    bool   window_changed = false;
    Window win            = calculate_max_window(*output, Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y));

    if(data_layout == DataLayout::NHWC)
    {
        AccessWindowStatic input_access(input, 0, -conv_pad_left,
                                        num_elems_read_per_iteration_x,
                                        ceil_to_multiple(input->dimension(1) + conv_info.pad_right(), num_elems_read_per_iteration_y));
        AccessWindowStatic    weights_access(weights, 0, 0, weights->dimension(0), weights->dimension(1));
        AccessWindowRectangle output_access(output, 0, 0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_y);
        window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
        return std::make_pair(err, win);
    }
    else if(data_layout == DataLayout::NCHW)
    {
        AccessWindowRectangle input_access(input, -conv_pad_left, -conv_pad_top, num_elems_read_per_iteration_x, num_elems_read_per_iteration_y, conv_stride_x, conv_stride_y);
        AccessWindowStatic    weights_access(weights, 0, 0, kernel_size, kernel_size);
        AccessWindowRectangle output_access(output, 0, 0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_y);
        window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
        return std::make_pair(err, win);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }
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

    const DataLayout data_layout = input->info()->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    const unsigned int kernel_size = weights->info()->dimension(width_idx);
    const DataType     data_type   = input->info()->data_type();

    // Get convolved dimensions
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*input->info(), *weights->info(), conv_info);

    // Output auto inizialitation if not yet initialized
    // TODO(COMPMID-2078): input->clone()->set_tensor_shape(output_shape) doesn't work with subtensors for grouped direct convolutions (AlexNet).
    auto_init_if_empty(*output->info(),
                       output_shape,
                       1,
                       input->info()->data_type(),
                       input->info()->quantization_info());

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  weights->info(),
                                                  (biases != nullptr) ? biases->info() : nullptr,
                                                  output->info(),
                                                  conv_info));

    _conv_stride_x = std::get<0>(conv_info.stride());
    _conv_stride_y = std::get<1>(conv_info.stride());

    if(data_layout == DataLayout::NHWC)
    {
        _border_size = BorderSize(conv_info.pad_left(), 0, conv_info.pad_right(), 0);
    }
    else if(data_layout == DataLayout::NCHW)
    {
        _border_size = BorderSize(conv_info.pad_top(), conv_info.pad_right(), conv_info.pad_bottom(), conv_info.pad_left());
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }

    _input   = input;
    _weights = weights;
    _output  = output;
    _biases  = biases;

    const GPUTarget gpu_target = get_target();

    std::stringstream kernel_name;
    kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;
    if(data_layout == DataLayout::NHWC)
    {
        kernel_name << "_" << lower_string(string_from_data_layout(data_layout));
    }

    CLBuildOptions build_options;
    build_options.add_option_if(_biases != nullptr, std::string("-DHAS_BIAS"));

    const bool run_optimized_for_bifrost = can_run_optimized_kernel_for_bifrost(gpu_target, _conv_stride_x, _conv_stride_y, kernel_size, data_type, data_layout);

    if(run_optimized_for_bifrost)
    {
        build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(channel_idx))));

        kernel_name << "_f32_bifrost";
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name.str(), build_options.options()));
    }
    else
    {
        const bool is_quantized_asymm = is_data_type_quantized_asymmetric(data_type);
        build_options.add_option_if(is_quantized_asymm, std::string("-DKERNEL_SIZE=" + support::cpp11::to_string(kernel_size)));
        build_options.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type)));
        build_options.add_option(std::string("-DDATA_SIZE=" + get_data_size_from_data_type(data_type)));
        build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(channel_idx))));
        build_options.add_option(std::string("-DSTRIDE_X=" + support::cpp11::to_string(_conv_stride_x)));
        if(data_layout == DataLayout::NHWC)
        {
            build_options.add_option(std::string("-DDATA_LAYOUT_NHWC=1"));
            build_options.add_option(std::string("-DDST_HEIGHT=" + support::cpp11::to_string(_output->info()->dimension(height_idx))));
            build_options.add_option(std::string("-DDST_WIDTH=" + support::cpp11::to_string(_output->info()->dimension(width_idx))));
            build_options.add_option(std::string("-DSRC_HEIGHT=" + support::cpp11::to_string(_input->info()->dimension(height_idx))));
            build_options.add_option(std::string("-DSRC_WIDTH=" + support::cpp11::to_string(_input->info()->dimension(width_idx))));
            build_options.add_option(std::string("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left())));
            build_options.add_option(std::string("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_top())));
            build_options.add_option(std::string("-DSTRIDE_Y=" + support::cpp11::to_string(_conv_stride_y)));
        }
        build_options.add_option(std::string("-DDATA_TYPE_PROMOTED=" + get_cl_type_from_data_type(data_type)));
        // Create kernel
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(is_quantized_asymm ? "direct_convolution_1x1_3x3_5x5_quantized" : kernel_name.str(),
                                                                               build_options.options()));
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), output->info(), conv_info, gpu_target);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

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
    _config_id += support::cpp11::to_string(border_size().left);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().top);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().right);
    _config_id += "_";
    _config_id += support::cpp11::to_string(border_size().bottom);
    _config_id += "_";
    _config_id += support::cpp11::to_string(_conv_stride_x);
    _config_id += "_";
    _config_id += support::cpp11::to_string(_conv_stride_y);
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(width_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(height_idx));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(data_layout));
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

    const DataLayout data_layout = _input->info()->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    win_in.set_dimension_step(width_idx, window[width_idx].step() * _conv_stride_x);
    win_in.set_dimension_step(height_idx, window[height_idx].step() * _conv_stride_y);

    Window       slice_in = win_in.first_slice_window_3D();
    unsigned int idx1     = 2 * num_arguments_per_3D_tensor();
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
        enqueue(queue, *this, slice, lws_hint());
    }
    while(window.slide_window_slice_3D(slice) && win_in.slide_window_slice_3D(slice_in));
}
