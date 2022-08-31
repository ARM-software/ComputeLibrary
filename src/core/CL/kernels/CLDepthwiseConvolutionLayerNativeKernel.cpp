/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/CL/CLUtils.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const DWCComputeKernelInfo &dwc_info,
                          const ConvolutionInfo &conv_info, const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_UNUSED(dwc_info);
    bool in_place = false;
    if(output == nullptr || output == input)
    {
        in_place = true;
        output   = input;
    }
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride().first > 1 && dwc_info.m0 != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.dilation.x() > 1 && dwc_info.m0 != 1);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((dwc_info.export_weights_to_cl_image == true) && (export_weights_to_cl_image(weights) == false), "Export to cl_image not supported!");
    ARM_COMPUTE_RETURN_ERROR_ON((dwc_info.export_weights_to_cl_image == true) && ((dwc_info.n0 % 4) != 0));
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride().first < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride().second < 1);
    ARM_COMPUTE_RETURN_ERROR_ON((conv_info.dilation.x() < 1) || (conv_info.dilation.y() < 1));
    const size_t idx_c = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_UNUSED(idx_c);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_c) != (input->dimension(idx_c) * conv_info.depth_multiplier));

    // In place restrictions
    if(in_place)
    {
        const int weights_width_idx  = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::WIDTH);
        const int weights_height_idx = get_data_layout_dimension_index(weights->data_layout(), DataLayoutDimension::HEIGHT);
        ARM_COMPUTE_RETURN_ERROR_ON(weights->tensor_shape()[weights_width_idx] != 1U || weights->tensor_shape()[weights_height_idx] != 1U);
        ARM_COMPUTE_RETURN_ERROR_ON(conv_info.depth_multiplier != 1U);
        ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.stride() != std::make_pair(1U, 1U));
        ARM_COMPUTE_RETURN_ERROR_ON(conv_info.dilation != Size2D(1U, 1U));
        ARM_COMPUTE_RETURN_ERROR_ON(conv_info.pad_stride_info.has_padding()); // Note that in princple padding can be supported with in_place but we choose not to support it
    }

    const ConvolutionInfo info{ conv_info.pad_stride_info, conv_info.depth_multiplier, ActivationLayerInfo(), conv_info.dilation };
    const TensorShape     output_shape = arm_compute::misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info);

    if(conv_info.depth_multiplier > 1 && dwc_info.n0 > 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON((conv_info.depth_multiplier % dwc_info.n0) != 0);
    }

    const bool is_quantized = is_data_type_quantized(input->data_type());

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != output_shape[idx_c]);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);

        if(is_quantized)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
        }
    }

    if(is_quantized)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output_multipliers, output_shifts);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_multipliers, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_shifts, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(output_multipliers->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shifts->num_dimensions() > 1);

        if(is_data_type_quantized_per_channel(weights->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QSYMM8_PER_CHANNEL);
            ARM_COMPUTE_RETURN_ERROR_ON(output_shape[idx_c] != output_multipliers->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(output_shape[idx_c] != output_shifts->dimension(0));
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
            ARM_COMPUTE_RETURN_ERROR_ON(1 != output_multipliers->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(1 != output_shifts->dimension(0));
        }
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    if(is_data_type_quantized(input->data_type()))
    {
        const UniformQuantizationInfo iq_info = input->quantization_info().uniform();
        const UniformQuantizationInfo wq_info = weights->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = (output->total_size() != 0) ? output->quantization_info().uniform() : iq_info;

        float multiplier        = iq_info.scale * wq_info.scale / oq_info.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        ARM_COMPUTE_RETURN_ON_ERROR(quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift));
    }

    return Status{};
}
} // namespace

CLDepthwiseConvolutionLayerNativeKernel::CLDepthwiseConvolutionLayerNativeKernel()
    : _input(nullptr),
      _weights(nullptr),
      _biases(nullptr),
      _output(nullptr),
      _depth_multiplier(1),
      _output_multipliers(nullptr),
      _output_shifts(nullptr),
      _export_to_cl_image(false),
      _is_quantized(false)
{
    _type = CLKernelType::DEPTHWISE;
}

void CLDepthwiseConvolutionLayerNativeKernel::configure(ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                                        const DWCComputeKernelInfo &dwc_info, const ConvolutionInfo &conv_info,
                                                        const ICLTensor *output_multipliers, const ICLTensor *output_shifts)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, dwc_info, conv_info, output_multipliers, output_shifts);
}

void CLDepthwiseConvolutionLayerNativeKernel::configure(const CLCompileContext &compile_context, ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                                        const DWCComputeKernelInfo &dwc_info, const ConvolutionInfo &conv_info,
                                                        const ICLTensor *output_multipliers, const ICLTensor *output_shifts)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    if(output == nullptr)
    {
        // In-place
        output = input;
    }
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(),
                                                  dwc_info, conv_info, (output_multipliers != nullptr) ? output_multipliers->info() : nullptr, (output_shifts != nullptr) ? output_shifts->info() : nullptr));

    auto padding_info = get_padding_info({ input, output });

    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_depthwise_convolution_shape(*(input->info()), *(weights->info()), conv_info);
    auto_init_if_empty(*(output->info()), input->info()->clone()->set_tensor_shape(output_shape).set_quantization_info(output->info()->quantization_info()));

    _input              = input;
    _output             = output;
    _weights            = weights;
    _biases             = biases;
    _depth_multiplier   = conv_info.depth_multiplier;
    _output_multipliers = output_multipliers;
    _output_shifts      = output_shifts;
    _export_to_cl_image = dwc_info.export_weights_to_cl_image;
    _is_quantized       = is_data_type_quantized(input->info()->data_type());

    const unsigned int n0          = adjust_vec_size(dwc_info.n0, output->info()->dimension(0));
    const unsigned int m0          = std::min(dwc_info.m0, (unsigned int)output->info()->dimension(1));
    std::string        kernel_name = "";

    CLBuildOptions build_opts;

    // Update the padding for the weights tensor if we can export to cl_image
    if(_export_to_cl_image)
    {
        arm_compute::opencl::kernels::gemm::update_padding_for_cl_image(weights->info());
    }

    // Conditions of -cl-fast-relaxed-math causing accuracy issues can be traced from COMPMID-5324
    const GPUTarget gpu_target    = get_target();
    const auto      act_function  = conv_info.act_info.activation();
    const auto      dst_data_type = _output->info()->data_type();

    if((gpu_target != GPUTarget::G71 && (gpu_target & GPUTarget::GPU_ARCH_MASK) == GPUTarget::BIFROST)
       && (act_function == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU || act_function == ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
       && (dst_data_type == DataType::F32 || dst_data_type == DataType::F16))
    {
        // -cl-fast-relaxed-math also sets -cl-finite-math-only and -cl-unsafe-math-optimizations
        // to disable -cl-finite-math-only, we only include -cl-unsafe-math-optimizations
        build_opts.add_option("-cl-unsafe-math-optimizations");
    }
    else
    {
        build_opts.add_option("-cl-fast-relaxed-math");
    }

    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_function)));
    build_opts.add_option("-DDEPTH_MULTIPLIER=" + support::cpp11::to_string(conv_info.depth_multiplier));
    build_opts.add_option("-DSRC_TENSOR_TYPE=BUFFER");
    // Note: SRC_DATA_TYPE must have the same data type of WEI_DATA_TYPE. In quantized, we could
    // have a case where the data types for the activation and weights are different. However, since the implementation
    // only works when both have same data type, we have to change the offset to take into account this aspect
    build_opts.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(_input->info()->data_type()));
    build_opts.add_option("-DDST_TENSOR_TYPE=BUFFER");
    build_opts.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(dst_data_type));
    build_opts.add_option_if_else(_export_to_cl_image, "-DWEI_TENSOR_TYPE=IMAGE", "-DWEI_TENSOR_TYPE=BUFFER");
    build_opts.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(_weights->info()->dimension(1)));
    build_opts.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(_weights->info()->dimension(2)));
    build_opts.add_option("-DWEI_DATA_TYPE=" + get_cl_type_from_data_type(_weights->info()->data_type()));
    build_opts.add_option("-DPAD_TOP=" + support::cpp11::to_string(conv_info.pad_stride_info.pad_top()));
    build_opts.add_option("-DPAD_LEFT=" + support::cpp11::to_string(conv_info.pad_stride_info.pad_left()));
    build_opts.add_option("-DSTRIDE_X=" + support::cpp11::to_string(conv_info.pad_stride_info.stride().first));
    build_opts.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(conv_info.pad_stride_info.stride().second));
    build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(conv_info.dilation.x()));
    build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(conv_info.dilation.y()));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_opts.add_option("-DM0_A=" + support::cpp11::to_string(_weights->info()->dimension(1) + m0 - 1));
    build_opts.add_option_if_else(conv_info.depth_multiplier > 1, "-DN0_A=1", "-DN0_A=" + support::cpp11::to_string(n0));
    build_opts.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(_output->info()->dimension(0) % n0));
    build_opts.add_option_if(_input->info()->num_dimensions() > 3, "-DBATCHED_EXECUTION");

    // Force unroll with pragma when any of the following values exceed the maximum number of manual unroll
    set_unroll_with_pragma(build_opts, { static_cast<int>(_weights->info()->dimension(1) + m0 - 1),
                                         static_cast<int>(_weights->info()->dimension(1)),
                                         static_cast<int>(_weights->info()->dimension(2))
                                       });

    if(biases != nullptr)
    {
        build_opts.add_option(std::string("-DHAS_BIAS"));
        build_opts.add_option(std::string("-DBIA_DATA_TYPE=" + get_cl_type_from_data_type(biases->info()->data_type())));
    }

    if(_is_quantized)
    {
        kernel_name                          = "dwc_native_quantized_nhwc";
        const UniformQuantizationInfo iqinfo = input->info()->quantization_info().uniform();
        const UniformQuantizationInfo wqinfo = weights->info()->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = output->info()->quantization_info().uniform();

        PixelValue zero_value = PixelValue(0, input->info()->data_type(), input->info()->quantization_info());
        int        zero_value_s32;
        zero_value.get(zero_value_s32);

        float multiplier        = iqinfo.scale * wqinfo.scale / oqinfo.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
        build_opts.add_option("-DDST_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
        build_opts.add_option("-DDST_SHIFT=" + support::cpp11::to_string(output_shift));
        build_opts.add_option("-DSRC_OFFSET=" + support::cpp11::to_string(-iqinfo.offset));
        build_opts.add_option("-DWEI_OFFSET=" + support::cpp11::to_string(-wqinfo.offset));
        build_opts.add_option("-DDST_OFFSET=" + support::cpp11::to_string(oqinfo.offset));
        build_opts.add_option("-DZERO_VALUE=" + support::cpp11::to_string(zero_value_s32));
        build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(DataType::S32));
        build_opts.add_option("-DDST_MULTIPLIERS_DATA_TYPE=" + get_cl_type_from_data_type(_output_multipliers->info()->data_type()));
        build_opts.add_option("-DDST_SHIFTS_DATA_TYPE=" + get_cl_type_from_data_type(_output_shifts->info()->data_type()));
        build_opts.add_option_if_else(weights->info()->data_type() == DataType::QSYMM8_PER_CHANNEL, "-DQUANTIZATION_TYPE=PER_CHANNEL", "-DQUANTIZATION_TYPE=PER_TENSOR");
        // Note: We expect the input and output tensors to always adopt a per-tensor quantization approach
        int a_val{};
        int b_val{};
        std::tie(b_val, a_val) = get_quantized_activation_min_max(conv_info.act_info, input->info()->data_type(), oqinfo);

        build_opts.add_option_if(conv_info.act_info.enabled(), "-DA_VAL=" + support::cpp11::to_string(a_val));
        build_opts.add_option_if(conv_info.act_info.enabled(), "-DB_VAL=" + support::cpp11::to_string(b_val));
    }
    else
    {
        kernel_name = "dwc_native_fp_nhwc";
        build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
        build_opts.add_option_if(conv_info.act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(conv_info.act_info.a()));
        build_opts.add_option_if(conv_info.act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(conv_info.act_info.b()));
    }

    Window win = calculate_max_window(*(output->info()), Steps(n0, m0));
    ICLKernel::configure_internal(win);

    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(2));
    _config_id += "_";
    _config_id += string_from_data_type(input->info()->data_type());
}

Status CLDepthwiseConvolutionLayerNativeKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                         const DWCComputeKernelInfo &dwc_info, const ConvolutionInfo &conv_info, const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, dwc_info, conv_info, output_multipliers, output_shifts));
    return Status{};
}

void CLDepthwiseConvolutionLayerNativeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Collapse window
    Window window_collapsed = window.collapse(ICLKernel::window(), Window::DimZ);

    Window slice = window_collapsed.first_slice_window_4D();

    cl::Image2D weights_cl_image;

    if(_export_to_cl_image)
    {
        const size_t      image_w = _weights->info()->dimension(0) / 4;
        const size_t      image_h = _weights->info()->dimension(1) * _weights->info()->dimension(2) * _weights->info()->dimension(3);
        const TensorShape shape2d(image_w, image_h);
        const size_t      image_row_pitch = _weights->info()->strides_in_bytes()[1];

        // Export cl_buffer to cl_image
        weights_cl_image = create_image2d_from_buffer(CLKernelLibrary::get().context(), _weights->cl_buffer(), shape2d, _weights->info()->data_type(), image_row_pitch);
    }

    unsigned int idx = 0;
    add_4d_tensor_nhwc_argument(idx, _input);
    add_4d_tensor_nhwc_argument(idx, _output);

    if(_export_to_cl_image)
    {
        _kernel.setArg(idx++, weights_cl_image);
    }
    add_4D_tensor_argument(idx, _weights, slice);
    if(_is_quantized)
    {
        add_1D_tensor_argument(idx, _output_multipliers, slice);
        add_1D_tensor_argument(idx, _output_shifts, slice);
    }
    if(_biases != nullptr)
    {
        add_1D_tensor_argument(idx, _biases, slice);
    }
    enqueue(queue, *this, slice, lws_hint());
}
} // namespace arm_compute
