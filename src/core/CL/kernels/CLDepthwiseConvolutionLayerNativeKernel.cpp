/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayerNativeKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const DWCWeightsKernelInfo &dwc_weights_info,
                          const DWCKernelInfo &dwc_info, const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation,
                          const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_UNUSED(dwc_info);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(depth_multiplier > 1 && dwc_weights_info.n0 != 1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.stride().first < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.stride().second < 1);
    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));
    const size_t idx_c = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL);
    ARM_COMPUTE_UNUSED(idx_c);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_c) != (input->dimension(idx_c) * depth_multiplier));

    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);

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

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *output, const DWCWeightsKernelInfo &dwc_weights_info,
                                                        const DWCKernelInfo &dwc_info, const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation,
                                                        ITensorInfo *output_multipliers, ITensorInfo *output_shifts)
{
    ARM_COMPUTE_UNUSED(dwc_info);

    // Get convolved dimensions
    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape).set_quantization_info(output->quantization_info()));

    const unsigned int n0 = dwc_weights_info.n0;

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(n0));

    // The following access windows are only valid in case of NHWC and because n0 must unit in case depth_multiplier > 1
    AccessWindowHorizontal input_access(input, 0, n0);
    AccessWindowHorizontal weights_access(weights, 0, n0);
    AccessWindowHorizontal output_access(output, 0, n0);

    bool window_changed = false;

    if(bias != nullptr)
    {
        AccessWindowHorizontal bias_access(bias, 0, n0);
        window_changed = update_window_and_padding(win, input_access, weights_access, bias_access, output_access);
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access, weights_access, output_access);
    }

    if(is_data_type_quantized(input->data_type()))
    {
        if((output_multipliers != nullptr) && (output_shifts != nullptr))
        {
            AccessWindowHorizontal output_multipliers_access(output_multipliers, 0, n0);
            AccessWindowHorizontal output_shifts_access(output_shifts, 0, n0);
            window_changed = window_changed || update_window_and_padding(win, output_multipliers_access, output_shifts_access);
        }
        else
        {
            Status err = ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "output_multipliers and output_shifts must be non-nullptr for quantized input");
            return std::make_pair(err, win);
        }
    }

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
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
      _is_quantized(false)
{
}

void CLDepthwiseConvolutionLayerNativeKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const DWCWeightsKernelInfo &dwc_weights_info,
                                                        const DWCKernelInfo &dwc_info, const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation,
                                                        const ICLTensor *output_multipliers, const ICLTensor *output_shifts)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(),
                                                  dwc_weights_info, dwc_info, conv_info, depth_multiplier, dilation,
                                                  (output_multipliers != nullptr) ? output_multipliers->info() : nullptr, (output_shifts != nullptr) ? output_shifts->info() : nullptr));

    auto win_config = validate_and_configure_window(input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(),
                                                    dwc_weights_info, dwc_info, conv_info, depth_multiplier, dilation,
                                                    (output_multipliers != nullptr) ? output_multipliers->info() : nullptr, (output_shifts != nullptr) ? output_shifts->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    _input              = input;
    _output             = output;
    _weights            = weights;
    _biases             = biases;
    _depth_multiplier   = depth_multiplier;
    _output_multipliers = output_multipliers;
    _output_shifts      = output_shifts;
    _is_quantized       = is_data_type_quantized(input->info()->data_type());

    const size_t idx_w          = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h          = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::HEIGHT);
    const size_t weights_width  = weights->info()->dimension(idx_w);
    const size_t weights_height = weights->info()->dimension(idx_h);

    CLBuildOptions build_opts;
    build_opts.add_option_if(_biases != nullptr, "-DHAS_BIAS");
    build_opts.add_option_if(_input->info()->tensor_shape().total_size_upper(3) > 1, "-DDST_DEPTH=" + support::cpp11::to_string(static_cast<int>(_output->info()->dimension(2))));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(_input->info()->data_type()));
    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(dwc_info.activation_info.activation())));
    build_opts.add_option("-DDEPTH_MULTIPLIER=" + support::cpp11::to_string(depth_multiplier));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(dwc_weights_info.n0));
    build_opts.add_option("-DSRC_DIM1=" + support::cpp11::to_string(_input->info()->dimension(1)));
    build_opts.add_option("-DSRC_DIM2=" + support::cpp11::to_string(_input->info()->dimension(2)));
    build_opts.add_option("-DKERNEL_WIDTH=" + support::cpp11::to_string(weights_width));
    build_opts.add_option("-DKERNEL_HEIGHT=" + support::cpp11::to_string(weights_height));
    build_opts.add_option("-DCONV_PAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.add_option("-DCONV_PAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.add_option("-DCONV_STRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
    build_opts.add_option("-DCONV_STRIDE_Y=" + support::cpp11::to_string(conv_info.stride().second));
    build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
    build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));

    std::string kernel_name = (_is_quantized) ? "dwc_MxN_native_quantized8_nhwc" : "dwc_MxN_native_fp_nhwc";

    if(_is_quantized)
    {
        const UniformQuantizationInfo iq_info = _input->info()->quantization_info().uniform();
        const UniformQuantizationInfo wq_info = _weights->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = _output->info()->quantization_info().uniform();

        build_opts.add_option("-DINPUT_OFFSET=" + support::cpp11::to_string(-iq_info.offset));
        build_opts.add_option("-DWEIGHTS_OFFSET=" + support::cpp11::to_string(-wq_info.offset));
        build_opts.add_option("-DOUTPUT_OFFSET=" + support::cpp11::to_string(oq_info.offset));
        build_opts.add_option_if(is_data_type_quantized_per_channel(weights->info()->data_type()), "-DPER_CHANNEL_QUANTIZATION");

        // Compute non-per-channel multiplier and shift anyway to make OpenCL kernel simpler
        float multiplier        = iq_info.scale * wq_info.scale / oq_info.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
        build_opts.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
        build_opts.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));

        if(dwc_info.activation_info.enabled())
        {
            const int a_val = quantize_qasymm8(dwc_info.activation_info.a(), oq_info);
            const int b_val = quantize_qasymm8(dwc_info.activation_info.b(), oq_info);
            const int o1    = oq_info.offset;

            build_opts.add_option("-DA_VAL=" + support::cpp11::to_string(a_val));
            build_opts.add_option("-DB_VAL=" + support::cpp11::to_string(b_val));
            build_opts.add_option("-DCONST_0=" + support::cpp11::to_string(o1));

            const float s1 = iq_info.scale;
            build_opts.add_option("-DS1_VAL=" + float_to_string_with_full_precision(s1));
            build_opts.add_option("-DO1_VAL=" + support::cpp11::to_string(o1));
        }

        build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
        build_opts.add_option("-DWEIGHTS_TYPE=" + get_cl_type_from_data_type(weights->info()->data_type()));
    }
    else
    {
        build_opts.add_option_if(dwc_info.activation_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(dwc_info.activation_info.a()));
        build_opts.add_option_if(dwc_info.activation_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(dwc_info.activation_info.b()));
    }

    ICLKernel::configure_internal(win_config.second);
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

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
                                                         const DWCWeightsKernelInfo &dwc_weights_info, const DWCKernelInfo &dwc_info, const PadStrideInfo &conv_info,
                                                         unsigned int depth_multiplier, const Size2D &dilation, const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, dwc_weights_info, dwc_info, conv_info, depth_multiplier, dilation, output_multipliers, output_shifts));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(),
                                                              biases != nullptr ? biases->clone().get() : nullptr,
                                                              output->clone().get(), dwc_weights_info, dwc_info, conv_info, depth_multiplier, dilation,
                                                              output_multipliers != nullptr ? output_multipliers->clone().get() : nullptr,
                                                              output_shifts != nullptr ? output_shifts->clone().get() : nullptr)
                                .first);

    return Status{};
}

void CLDepthwiseConvolutionLayerNativeKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Collapse window
    Window window_collapsed = window.collapse(ICLKernel::window(), Window::DimZ);
    Window slice_in         = window.first_slice_window_4D();
    Window slice_out        = window_collapsed.first_slice_window_4D();

    if(_depth_multiplier != 1)
    {
        ARM_COMPUTE_ERROR_ON(slice_out.x().step() != 1);
        slice_out.set(Window::DimX, Window::Dimension(0, _input->info()->tensor_shape()[0], 1));
    }

    unsigned int idx = 2 * num_arguments_per_4D_tensor() + num_arguments_per_3D_tensor();

    // Set output multipliers in case of quantized data type
    if(_is_quantized)
    {
        add_1D_tensor_argument(idx, _output_multipliers, slice_in);
        add_1D_tensor_argument(idx, _output_shifts, slice_in);
    }

    if(_biases != nullptr)
    {
        add_1D_tensor_argument(idx, _biases, slice_in);
    }

    do
    {
        idx = 0;
        add_4D_tensor_argument(idx, _input, slice_in);
        add_4D_tensor_argument(idx, _output, slice_out);
        add_3D_tensor_argument(idx, _weights, slice_out);
        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(window_collapsed.slide_window_slice_4D(slice_out) && window.slide_window_slice_4D(slice_in));
}
} // namespace arm_compute
