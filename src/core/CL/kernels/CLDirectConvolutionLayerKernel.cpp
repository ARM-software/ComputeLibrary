/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/CL/kernels/CLDirectConvolutionLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/PixelValue.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);

    const DataLayout data_layout = input->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != weights->dimension(height_idx), "Weights should have same width and height");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != input->dimension(channel_idx),
                                    "Weights feature map dimension should match the respective input's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 1) && std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported for 1x1 convolution.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 3 || weights->dimension(width_idx) == 5 || weights->dimension(width_idx) == 9)
                                    && std::get<0>(conv_info.stride()) > 2,
                                    "Strides larger than 2 not supported for 3x3, 5x5, 9x9 convolution.");

    if(data_layout == DataLayout::NCHW)
    {
        if(is_data_type_quantized(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != 1 && weights->dimension(width_idx) != 3 && weights->dimension(width_idx) != 5 && weights->dimension(width_idx) != 9,
                                            "Kernel sizes other than 1x1, 3x3, 5x5 or 9x9 are not supported with quantized data types");
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != 1 && weights->dimension(width_idx) != 3 && weights->dimension(width_idx) != 5,
                                            "Kernel sizes other than 1x1, 3x3 or 5x5 are not supported with float data types");
        }
    }

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

    const auto data_type = input->data_type();
    if(is_data_type_quantized(data_type))
    {
        const UniformQuantizationInfo iqinfo = input->quantization_info().uniform();
        const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = output->quantization_info().uniform();

        float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));
    }
    return Status{};
}

inline bool can_run_optimized_kernel_for_bifrost_nchw(GPUTarget gpu_target, unsigned int conv_stride_x, unsigned int conv_stride_y, unsigned int kernel_size,
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

inline void setup_num_elems_nchw(unsigned int &num_elems_read_per_iteration_x, unsigned int &num_elems_read_per_iteration_y,
                                 unsigned int &num_elems_written_per_iteration_x, unsigned int &num_elems_written_per_iteration_y,
                                 unsigned int kernel_size, const PadStrideInfo &conv_info, const GPUTarget target, ITensorInfo *input)
{
    const DataType   data_type     = input->data_type();
    const DataLayout data_layout   = input->data_layout();
    unsigned int     conv_stride_x = std::get<0>(conv_info.stride());
    unsigned int     conv_stride_y = std::get<1>(conv_info.stride());

    const bool run_optimized_bifrost = can_run_optimized_kernel_for_bifrost_nchw(target, conv_stride_x, conv_stride_y, kernel_size, data_type, data_layout);

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
            case 9:
                switch(conv_stride_x)
                {
                    case 1:
                        num_elems_read_per_iteration_x = 16;
                        break;
                    case 2:
                        num_elems_read_per_iteration_x = 24;
                        break;
                    default:
                        ARM_COMPUTE_ERROR("Invalid convolution stride X");
                }
                break;
            default:
                ARM_COMPUTE_ERROR("Invalid direct convolution size");
        }
    }
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, const GPUTarget target)
{
    const DataLayout data_layout = input->data_layout();

    // Get output shape
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*input, *weights, conv_info);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output, output_shape,
                       1,
                       input->data_type(),
                       input->quantization_info());

    if(data_layout == DataLayout::NHWC)
    {
        const unsigned int vec_size = std::min(static_cast<unsigned int>(output->tensor_shape()[0]), 4u);

        // Create window and update padding
        Window win = calculate_max_window(*output, Steps(vec_size, 1U));
        output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));
        Status err = Status{};
        return std::make_pair(err, win);
    }
    else if(data_layout == DataLayout::NCHW)
    {
        const int          width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
        const unsigned int kernel_size = weights->dimension(width_idx);

        unsigned int num_elems_read_per_iteration_x    = 0;
        unsigned int num_elems_read_per_iteration_y    = 0;
        unsigned int num_elems_written_per_iteration_x = 0;
        unsigned int num_elems_written_per_iteration_y = 0;

        unsigned int conv_pad_left = conv_info.pad_left();
        unsigned int conv_pad_top  = conv_info.pad_top();
        unsigned int conv_stride_x = std::get<0>(conv_info.stride());
        unsigned int conv_stride_y = std::get<1>(conv_info.stride());

        setup_num_elems_nchw(num_elems_read_per_iteration_x, num_elems_read_per_iteration_y,
                             num_elems_written_per_iteration_x, num_elems_written_per_iteration_y,
                             kernel_size, conv_info, target, input);

        // Create window and update padding
        bool   window_changed = false;
        Window win            = calculate_max_window(*output, Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y));

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
    : _input(nullptr), _biases(nullptr), _weights(nullptr), _output(nullptr), _data_layout(DataLayout::UNKNOWN), _border_size(0), _conv_stride_x(0), _conv_stride_y(0), _conv_info()
{
}

BorderSize CLDirectConvolutionLayerKernel::border_size() const
{
    return _border_size;
}

void CLDirectConvolutionLayerKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info);
}

void CLDirectConvolutionLayerKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                               const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    // Perform validation
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(),
                                                  weights->info(),
                                                  (biases != nullptr) ? biases->info() : nullptr,
                                                  output->info(),
                                                  conv_info));

    _conv_stride_x = std::get<0>(conv_info.stride());
    _conv_stride_y = std::get<1>(conv_info.stride());
    _data_layout   = input->info()->data_layout();
    _input         = input;
    _weights       = weights;
    _output        = output;
    _biases        = biases;
    _conv_info     = conv_info;

    const unsigned int width_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const unsigned int kernel_size = weights->info()->dimension(width_idx);
    const DataType     data_type   = input->info()->data_type();

    const GPUTarget gpu_target = get_target();

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), output->info(), conv_info, gpu_target);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    std::stringstream kernel_name;
    CLBuildOptions    build_options;

    if(_data_layout == DataLayout::NHWC)
    {
        _border_size = BorderSize();

        kernel_name << "direct_convolution_nhwc";

        const unsigned int n0               = win_config.second.x().step();
        const unsigned int m0               = win_config.second.y().step();
        const unsigned int k0               = adjust_vec_size(16u, _input->info()->dimension(channel_idx));
        const unsigned int partial_store_n0 = _output->info()->dimension(channel_idx) % n0;
        const unsigned int partial_store_m0 = (_output->info()->dimension(width_idx) * _output->info()->dimension(height_idx)) % m0;
        const unsigned int pad_left         = conv_info.pad_left();
        const unsigned int pad_top          = conv_info.pad_top();

        if(_biases != nullptr)
        {
            build_options.add_option(std::string("-DHAS_BIAS"));
            build_options.add_option(std::string("-DBIA_DATA_TYPE=" + get_cl_type_from_data_type(_biases->info()->data_type())));
        }
        build_options.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(_input->info()->dimension(width_idx)));
        build_options.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(_input->info()->dimension(height_idx)));
        build_options.add_option("-DSRC_CHANNELS=" + support::cpp11::to_string(_input->info()->dimension(channel_idx)));
        build_options.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(_input->info()->data_type()));
        build_options.add_option("-DDST_WIDTH=" + support::cpp11::to_string(_output->info()->dimension(width_idx)));
        build_options.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(_output->info()->dimension(height_idx)));
        build_options.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(_output->info()->dimension(channel_idx)));
        build_options.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(_output->info()->data_type()));
        build_options.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(_weights->info()->dimension(width_idx)));
        build_options.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(_weights->info()->dimension(height_idx)));
        build_options.add_option("-DWEI_DATA_TYPE=" + get_cl_type_from_data_type(_weights->info()->data_type()));
        build_options.add_option("-DSTRIDE_X=" + support::cpp11::to_string(_conv_stride_x));
        build_options.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(_conv_stride_y));
        build_options.add_option("-DPAD_LEFT=" + support::cpp11::to_string(pad_left));
        build_options.add_option("-DPAD_TOP=" + support::cpp11::to_string(pad_top));
        build_options.add_option("-DN0=" + support::cpp11::to_string(n0));
        build_options.add_option("-DM0=" + support::cpp11::to_string(m0));
        build_options.add_option("-DK0=" + support::cpp11::to_string(k0));
        build_options.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));
        build_options.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));

        if(is_data_type_quantized(data_type))
        {
            const UniformQuantizationInfo iqinfo = _input->info()->quantization_info().uniform();
            const UniformQuantizationInfo wqinfo = _weights->info()->quantization_info().uniform();
            const UniformQuantizationInfo oqinfo = _output->info()->quantization_info().uniform();

            PixelValue zero_value = PixelValue(0, input->info()->data_type(), input->info()->quantization_info());
            int        zero_value_s32;
            zero_value.get(zero_value_s32);

            float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
            int   output_multiplier = 0;
            int   output_shift      = 0;
            quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
            build_options.add_option("-DIS_QUANTIZED");
            build_options.add_option("-DDST_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
            build_options.add_option("-DDST_SHIFT=" + support::cpp11::to_string(output_shift));
            build_options.add_option("-DSRC_OFFSET=" + support::cpp11::to_string(-iqinfo.offset));
            build_options.add_option("-DWEI_OFFSET=" + support::cpp11::to_string(-wqinfo.offset));
            build_options.add_option("-DDST_OFFSET=" + support::cpp11::to_string(oqinfo.offset));
            build_options.add_option("-DZERO_VALUE=" + support::cpp11::to_string(zero_value_s32));
            build_options.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(DataType::S32));
        }
        else
        {
            build_options.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(data_type));
            build_options.add_option("-DSRC_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option("-DWEI_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option("-DDST_OFFSET=" + support::cpp11::to_string(0));
        }
    }
    else
    {
        _border_size = BorderSize(_input->info()->padding());

        kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;

        build_options.add_option_if(_biases != nullptr, std::string("-DHAS_BIAS"));

        const bool run_optimized_for_bifrost = can_run_optimized_kernel_for_bifrost_nchw(gpu_target, _conv_stride_x, _conv_stride_y, kernel_size, data_type, _data_layout);

        if(run_optimized_for_bifrost)
        {
            build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(channel_idx))));

            kernel_name << "_f32_bifrost";
        }
        else
        {
            build_options.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type)));
            build_options.add_option(std::string("-DDATA_SIZE=" + get_data_size_from_data_type(data_type)));
            build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(_weights->info()->dimension(channel_idx))));
            build_options.add_option(std::string("-DSTRIDE_X=" + support::cpp11::to_string(_conv_stride_x)));
            build_options.add_option(std::string("-DDATA_TYPE_PROMOTED=" + get_cl_type_from_data_type(data_type)));

            if(is_data_type_quantized(data_type))
            {
                const UniformQuantizationInfo iqinfo = _input->info()->quantization_info().uniform();
                const UniformQuantizationInfo wqinfo = _weights->info()->quantization_info().uniform();
                const UniformQuantizationInfo oqinfo = _output->info()->quantization_info().uniform();

                float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
                int   output_multiplier = 0;
                int   output_shift      = 0;
                quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
                build_options.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
                build_options.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));
                build_options.add_option("-DKERNEL_SIZE=" + support::cpp11::to_string(kernel_size));
                build_options.add_option("-DINPUT_OFFSET=" + support::cpp11::to_string(-iqinfo.offset));
                build_options.add_option("-DWEIGHTS_OFFSET=" + support::cpp11::to_string(-wqinfo.offset));
                build_options.add_option("-DOUTPUT_OFFSET=" + support::cpp11::to_string(oqinfo.offset));

                kernel_name.str("direct_convolution_quantized");
            }
        }
    }

    _kernel = create_kernel(compile_context, kernel_name.str(), build_options.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name.str();
    _config_id += "_";
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
    _config_id += lower_string(string_from_data_layout(_data_layout));
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
    Window slice = window.first_slice_window_3D();

    if(_data_layout == DataLayout::NHWC)
    {
        slice.set(Window::DimY, Window::Dimension(0, _output->info()->dimension(1) * _output->info()->dimension(2), 1));
        slice.set(Window::DimZ, Window::Dimension(0, _output->info()->dimension(3), 1));

        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_3D_tensor_argument(idx, _weights, slice);
        if(_biases != nullptr)
        {
            add_1D_tensor_argument(idx, _biases, slice);
        }
        _kernel.setArg(idx++, static_cast<unsigned int>(_weights->info()->strides_in_bytes()[3]));
        enqueue(queue, *this, slice, lws_hint());
    }
    else
    {
        Window win_in = window;

        win_in.adjust(Window::DimX, -_conv_info.pad_left(), true);
        win_in.adjust(Window::DimY, -_conv_info.pad_top(), true);

        const int width_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
        const int height_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

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
}
} // namespace arm_compute
