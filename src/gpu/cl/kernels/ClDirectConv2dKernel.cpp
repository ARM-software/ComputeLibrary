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
#include "src/gpu/cl/kernels/ClDirectConv2dKernel.h"

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
#include "src/core/CL/CLUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "support/Cast.h"
#include "support/StringSupport.h"
namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                          const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);

    const DataLayout data_layout = src->data_layout();
    const int        width_idx   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        height_idx  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        channel_idx = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(width_idx) != weights->dimension(height_idx), "Weights should have same width and height");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != src->dimension(channel_idx),
                                    "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 1) && std::get<0>(conv_info.stride()) > 3, "Strides larger than 3 not supported for 1x1 convolution.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((weights->dimension(width_idx) == 3 || weights->dimension(width_idx) == 5 || weights->dimension(width_idx) == 9)
                                    && std::get<0>(conv_info.stride()) > 2,
                                    "Strides larger than 2 not supported for 3x3, 5x5, 9x9 convolution.");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(data_layout != DataLayout::NHWC && !is_data_type_float(src->data_type()) && act_info.enabled(),
                                    "Activation supported only for floating point and NHWC.");

    if(data_layout == DataLayout::NCHW)
    {
        if(is_data_type_quantized(src->data_type()))
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
        if(is_data_type_quantized_asymmetric(src->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(0) != weights->dimension(3),
                                        "Biases size and number of src feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1,
                                        "Biases should be one dimensional");
    }

    // Checks performed when dst is configured
    if(dst->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(dst->tensor_shape(),
                                                           misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info));
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    const auto data_type = src->data_type();
    if(is_data_type_quantized(data_type))
    {
        const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
        const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

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
                                 unsigned int kernel_size, const PadStrideInfo &conv_info, const GPUTarget target, ITensorInfo *src)
{
    const DataType   data_type     = src->data_type();
    const DataLayout data_layout   = src->data_layout();
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
                        switch(src->element_size())
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

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *src, ITensorInfo *weights, ITensorInfo *dst, const PadStrideInfo &conv_info, const GPUTarget target)
{
    const DataLayout data_layout = src->data_layout();

    // Get dst shape
    TensorShape output_shape = misc::shape_calculator::compute_deep_convolution_shape(*src, *weights, conv_info);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*dst, output_shape,
                       1,
                       src->data_type(),
                       src->quantization_info());

    if(data_layout == DataLayout::NHWC)
    {
        const unsigned int vec_size = std::min(static_cast<unsigned int>(dst->tensor_shape()[0]), 4u);
        unsigned int       num_rows = 1U;
        if(dst->tensor_shape()[0] > 16)
        {
            num_rows = src->data_type() == DataType::F32 ? 2U : 4U;
        }

        // Create window and update padding
        Window win = calculate_max_window(output_shape, Steps(vec_size, num_rows));
        return std::make_pair(Status{}, win);
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
                             kernel_size, conv_info, target, src);

        // Create window and update padding
        bool   window_changed = false;
        Window win            = calculate_max_window(*dst, Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y));

        AccessWindowRectangle input_access(src, -conv_pad_left, -conv_pad_top, num_elems_read_per_iteration_x, num_elems_read_per_iteration_y, conv_stride_x, conv_stride_y);
        AccessWindowStatic    weights_access(weights, 0, 0, kernel_size, kernel_size);
        AccessWindowRectangle output_access(dst, 0, 0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_y);
        window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
        output_access.set_valid_region(win, ValidRegion(Coordinates(), dst->tensor_shape()));
        Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
        return std::make_pair(err, win);
    }
    else
    {
        ARM_COMPUTE_ERROR("Not supported");
    }
}

bool export_to_cl_image_support(ITensorInfo *tensor, GPUTarget gpu_target, DataLayout data_layout)
{
    if(tensor->tensor_shape()[0] % 4 || (data_layout != DataLayout::NHWC))
    {
        return false;
    }

    // If not floating point
    if(!is_data_type_float(tensor->data_type()))
    {
        return false;
    }

    if(gpu_target == GPUTarget::G71 || get_arch_from_target(gpu_target) == GPUTarget::MIDGARD)
    {
        return false;
    }

    // Check if the cl_khr_image2d_from_buffer extension is supported on the target platform
    if(!image2d_from_buffer_supported(CLKernelLibrary::get().get_device()))
    {
        return false;
    }

    // Check cl image pitch alignment
    if(get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) == 0)
    {
        return false;
    }

    const size_t image_w     = tensor->tensor_shape()[0] / 4;
    const size_t image_h     = tensor->tensor_shape()[1] * tensor->tensor_shape()[2] * tensor->tensor_shape()[3];
    const size_t max_image_w = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
    const size_t max_image_h = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

    if(image_w > max_image_w || image_h > max_image_h)
    {
        return false;
    }

    return true;
}

} // namespace

BorderSize ClDirectConv2dKernel::border_size() const
{
    return _border_size;
}

ClDirectConv2dKernel::ClDirectConv2dKernel()
{
    _type = CLKernelType::DIRECT;
}

void ClDirectConv2dKernel::configure(const CLCompileContext &compile_context, ITensorInfo *src, ITensorInfo *weights, ITensorInfo *biases, ITensorInfo *dst,
                                     const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);

    // Perform validation
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, dst, conv_info, act_info));

    const int conv_stride_x = std::get<0>(conv_info.stride());
    const int conv_stride_y = std::get<1>(conv_info.stride());

    _data_layout = src->data_layout();
    _conv_info   = conv_info;

    const unsigned int width_idx   = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const unsigned int height_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const unsigned int channel_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::CHANNEL);
    const unsigned int kernel_size = weights->dimension(width_idx);
    const DataType     data_type   = src->data_type();

    const GPUTarget gpu_target = get_target();

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, weights, dst, conv_info, gpu_target);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    std::stringstream kernel_name;
    CLBuildOptions    build_options;

    if(_data_layout == DataLayout::NHWC)
    {
        _border_size = BorderSize();

        kernel_name << "direct_convolution_nhwc";

        const unsigned int n0                 = win_config.second.x().step();
        const unsigned int m0                 = win_config.second.y().step();
        const unsigned int k0                 = adjust_vec_size(is_data_type_quantized(data_type) ? 16u : 8u, src->dimension(channel_idx));
        const unsigned int partial_store_n0   = dst->dimension(channel_idx) % n0;
        const unsigned int pad_left           = conv_info.pad_left();
        const unsigned int pad_top            = conv_info.pad_top();
        const bool         export_to_cl_image = export_to_cl_image_support(weights, gpu_target, _data_layout);

        // Update the padding for the weights tensor if we can export to cl_image
        if(export_to_cl_image)
        {
            gemm::update_padding_for_cl_image(weights);
        }

        if(biases != nullptr)
        {
            build_options.add_option(std::string("-DHAS_BIAS"));
            build_options.add_option(std::string("-DBIA_DATA_TYPE=" + get_cl_type_from_data_type(biases->data_type())));
        }

        build_options.add_option("-cl-fast-relaxed-math");
        build_options.add_option("-DSRC_TENSOR_TYPE=BUFFER");
        build_options.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(src->dimension(width_idx)));
        build_options.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(src->dimension(height_idx)));
        build_options.add_option("-DSRC_CHANNELS=" + support::cpp11::to_string(src->dimension(channel_idx)));
        build_options.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(src->data_type()));
        build_options.add_option("-DDST_TENSOR_TYPE=BUFFER");
        build_options.add_option("-DDST_WIDTH=" + support::cpp11::to_string(dst->dimension(width_idx)));
        build_options.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(dst->dimension(height_idx)));
        build_options.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(dst->dimension(channel_idx)));
        build_options.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(dst->data_type()));
        build_options.add_option_if_else(export_to_cl_image, "-DWEI_TENSOR_TYPE=IMAGE", "-DWEI_TENSOR_TYPE=BUFFER");
        build_options.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(weights->dimension(width_idx)));
        build_options.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(weights->dimension(height_idx)));
        build_options.add_option("-DWEI_DATA_TYPE=" + get_cl_type_from_data_type(weights->data_type()));
        build_options.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_stride_x));
        build_options.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_stride_y));
        build_options.add_option("-DPAD_LEFT=" + support::cpp11::to_string(pad_left));
        build_options.add_option("-DPAD_TOP=" + support::cpp11::to_string(pad_top));
        build_options.add_option("-DN0=" + support::cpp11::to_string(n0));
        build_options.add_option("-DM0=" + support::cpp11::to_string(m0));
        build_options.add_option("-DK0=" + support::cpp11::to_string(k0));
        build_options.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));
        build_options.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_info.activation())));

        if(is_data_type_quantized(data_type))
        {
            const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
            const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
            const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

            PixelValue zero_value = PixelValue(0, src->data_type(), src->quantization_info());
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
            build_options.add_option("-DZERO_VALUE=" + support::cpp11::to_string(0));
            build_options.add_option("-DSRC_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option("-DWEI_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option("-DDST_OFFSET=" + support::cpp11::to_string(0));
            build_options.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
            build_options.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
        }
    }
    else
    {
        _border_size = BorderSize(src->padding());

        kernel_name << "direct_convolution" << kernel_size << "x" << kernel_size;

        build_options.add_option_if(biases != nullptr, std::string("-DHAS_BIAS"));

        const bool run_optimized_for_bifrost = can_run_optimized_kernel_for_bifrost_nchw(gpu_target, conv_stride_x, conv_stride_y, kernel_size, data_type, _data_layout);

        if(run_optimized_for_bifrost)
        {
            build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(weights->dimension(channel_idx))));

            kernel_name << "_f32_bifrost";
        }
        else
        {
            build_options.add_option(std::string("-DDATA_TYPE=" + get_cl_type_from_data_type(data_type)));
            build_options.add_option(std::string("-DDATA_SIZE=" + get_data_size_from_data_type(data_type)));
            build_options.add_option(std::string("-DWEIGHTS_DEPTH=" + support::cpp11::to_string(weights->dimension(channel_idx))));
            build_options.add_option(std::string("-DSTRIDE_X=" + support::cpp11::to_string(conv_stride_x)));
            build_options.add_option(std::string("-DDATA_TYPE_PROMOTED=" + get_cl_type_from_data_type(data_type)));

            if(is_data_type_quantized(data_type))
            {
                const UniformQuantizationInfo iqinfo = src->quantization_info().uniform();
                const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
                const UniformQuantizationInfo oqinfo = dst->quantization_info().uniform();

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
    _config_id += support::cpp11::to_string(conv_stride_x);
    _config_id += "_";
    _config_id += support::cpp11::to_string(conv_stride_y);
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(width_idx));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(height_idx));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(_data_layout));
}

Status ClDirectConv2dKernel::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                      const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, const GPUTarget target)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, conv_info, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), weights->clone().get(), dst->clone().get(), conv_info, target).first);

    return Status{};
}

void ClDirectConv2dKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    Window slice = window.first_slice_window_3D();

    const auto src     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto weights = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto biases  = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    auto       dst     = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    if(_data_layout == DataLayout::NHWC)
    {
        cl::Image2D weights_cl_image;

        const size_t dim_y_collapsed    = ceil_to_multiple(dst->info()->dimension(1) * dst->info()->dimension(2), slice.y().step());
        const bool   export_to_cl_image = export_to_cl_image_support(weights->info(), get_target(), _data_layout);

        slice.set(Window::DimY, Window::Dimension(0, dim_y_collapsed, slice.y().step()));
        slice.set(Window::DimZ, Window::Dimension(0, dst->info()->dimension(3), 1));

        if(export_to_cl_image)
        {
            const size_t      image_w = weights->info()->dimension(0) / 4;
            const size_t      image_h = weights->info()->dimension(1) * weights->info()->dimension(2) * weights->info()->dimension(3);
            const TensorShape shape2d(image_w, image_h);
            const size_t      image_row_pitch = weights->info()->strides_in_bytes()[1];

            // Export cl_buffer to cl_image
            weights_cl_image = create_image2d_from_buffer(CLKernelLibrary::get().context(), weights->cl_buffer(), shape2d, weights->info()->data_type(), image_row_pitch);
        }

        unsigned int idx = 0;
        add_4D_tensor_argument(idx, src, slice);
        add_4D_tensor_argument(idx, dst, slice);
        if(export_to_cl_image)
        {
            _kernel.setArg(idx++, weights_cl_image);
        }
        add_4D_tensor_argument(idx, weights, slice);
        if(biases != nullptr)
        {
            add_1D_tensor_argument(idx, biases, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    else
    {
        Window win_in = window;

        win_in.adjust(Window::DimX, -_conv_info.pad_left(), true);
        win_in.adjust(Window::DimY, -_conv_info.pad_top(), true);

        const int width_idx  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
        const int height_idx = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

        const int conv_stride_x = std::get<0>(_conv_info.stride());
        const int conv_stride_y = std::get<1>(_conv_info.stride());

        win_in.set_dimension_step(width_idx, window[width_idx].step() * conv_stride_x);
        win_in.set_dimension_step(height_idx, window[height_idx].step() * conv_stride_y);

        Window       slice_in = win_in.first_slice_window_3D();
        unsigned int idx1     = 2 * num_arguments_per_3D_tensor();
        add_3D_tensor_argument(idx1, weights, slice);

        if(biases != nullptr)
        {
            Window slice_biases;
            slice_biases.use_tensor_dimensions(biases->info()->tensor_shape());
            add_1D_tensor_argument(idx1, biases, slice_biases);
        }

        _kernel.setArg(idx1++, static_cast<unsigned int>(weights->info()->strides_in_bytes()[3]));

        do
        {
            unsigned int idx = 0;
            add_3D_tensor_argument(idx, src, slice_in);
            add_3D_tensor_argument(idx, dst, slice);
            enqueue(queue, *this, slice, lws_hint());
        }
        while(window.slide_window_slice_3D(slice) && win_in.slide_window_slice_3D(slice_in));
    }
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
