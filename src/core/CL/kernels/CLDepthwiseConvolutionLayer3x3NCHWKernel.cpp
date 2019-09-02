/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NCHWKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
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

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                          const ActivationLayerInfo &act_info, const Size2D dilation)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((act_info.enabled()) && (input->data_type() == DataType::QASYMM8) && (act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::RELU)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                    "For QASYMM8 only logistic, relu, lower bounded relu and lower-upper bounded relu are supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(0) != 3 || weights->dimension(1) != 3);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.stride().first < 1 || conv_info.stride().first > 3);

    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));

    const bool is_qasymm = is_data_type_quantized_asymmetric(input->data_type());

    if(biases != nullptr)
    {
        if(is_qasymm)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }
        ARM_COMPUTE_RETURN_ERROR_ON((biases->dimension(0) != weights->dimension(2)) && (weights->dimension(2) != 1 || biases->dimension(0) != weights->dimension(3)));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if(output->total_size() != 0)
    {
        const TensorShape output_shape = compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    if(is_qasymm)
    {
        const UniformQuantizationInfo iq_info = input->quantization_info().uniform();
        const UniformQuantizationInfo wq_info = weights->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = (output->total_size() != 0) ? output->quantization_info().uniform() : iq_info;

        float multiplier = iq_info.scale * wq_info.scale / oq_info.scale;
        ARM_COMPUTE_UNUSED(multiplier);
        ARM_COMPUTE_RETURN_ERROR_ON(multiplier > 1.0f);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                                                        GPUTarget gpu_target, std::string &kernel_name, const Size2D dilation)
{
    // Output auto inizialitation if not yet initialized
    const TensorShape output_shape = compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier, dilation);
    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape).set_quantization_info(output->quantization_info()));

    const unsigned int conv_stride_x = conv_info.stride().first;
    const unsigned int conv_stride_y = conv_info.stride().second;
    const bool         is_qasymm     = is_data_type_quantized_asymmetric(input->data_type());
    const bool         is_bifrost    = get_arch_from_target(gpu_target) == GPUTarget::BIFROST;

    // Configure kernel window
    unsigned int num_elems_read_per_iteration_x    = 0;
    unsigned int num_elems_read_per_iteration_y    = 0;
    unsigned int num_elems_written_per_iteration_x = 0;
    unsigned int num_elems_written_per_iteration_y = 0;

    if(input->data_type() == DataType::F16)
    {
        kernel_name                       = "depthwise_convolution_3x3_f16";
        num_elems_written_per_iteration_x = 8 / data_size_from_type(input->data_type());
        num_elems_written_per_iteration_y = 1;
        num_elems_read_per_iteration_y    = 3;
        switch(conv_stride_x)
        {
            case 1:
                num_elems_read_per_iteration_x = 8;
                break;
            case 2:
                num_elems_read_per_iteration_x = 9;
                break;
            case 3:
                num_elems_read_per_iteration_x = 16;
                break;
            default:
                num_elems_read_per_iteration_x = 3 + (num_elems_written_per_iteration_x - 1) * conv_stride_x;
                break;
        }
        if(is_bifrost)
        {
            if(conv_stride_x == 1 && conv_stride_y == 1)
            {
                kernel_name                       = "depthwise_convolution_3x3_stridex1_stridey1_bifrost_f16";
                num_elems_read_per_iteration_x    = 8;
                num_elems_written_per_iteration_x = 4;
                num_elems_read_per_iteration_y    = 6;
                num_elems_written_per_iteration_y = 4;
            }
            else if(conv_stride_x == 2 && conv_stride_y == 2)
            {
                kernel_name                       = "depthwise_convolution_3x3_stridex2_stridey2_bifrost_f16";
                num_elems_read_per_iteration_x    = 10;
                num_elems_written_per_iteration_x = 4;
                num_elems_read_per_iteration_y    = 5;
                num_elems_written_per_iteration_y = 2;
            }
        }
    }
    else if(input->data_type() == DataType::F32 && is_bifrost)
    {
        if(conv_stride_x == 1 && conv_stride_y == 1)
        {
            kernel_name                       = "depthwise_convolution_3x3_stridex1_stridey1_bifrost_f32";
            num_elems_read_per_iteration_x    = 4;
            num_elems_read_per_iteration_y    = 6;
            num_elems_written_per_iteration_x = 2;
            num_elems_written_per_iteration_y = 4;
        }
        else if(conv_stride_x == 2 && conv_stride_y == 2)
        {
            kernel_name                       = "depthwise_convolution_3x3_stridex2_stridey2_bifrost_f32";
            num_elems_read_per_iteration_x    = 6;
            num_elems_read_per_iteration_y    = 5;
            num_elems_written_per_iteration_x = 2;
            num_elems_written_per_iteration_y = 2;
        }
        else
        {
            kernel_name                       = "depthwise_convolution_3x3";
            num_elems_written_per_iteration_x = 8 / data_size_from_type(input->data_type());
            num_elems_written_per_iteration_y = 1;
            num_elems_read_per_iteration_x    = 3 + (num_elems_written_per_iteration_x - 1) * conv_stride_x;
            num_elems_read_per_iteration_y    = 3;
        }
    }
    else
    {
        const bool is_dot8_supported = dot8_supported(CLKernelLibrary::get().get_device());

        kernel_name = is_qasymm ? "dwc_3x3_native_qasymm8" : "depthwise_convolution_3x3";
        kernel_name += (is_qasymm && is_dot8_supported ? "_dot8" : "");
        kernel_name += (is_qasymm ? "_nchw" : "");

        num_elems_written_per_iteration_x = 8 / data_size_from_type(input->data_type());
        num_elems_written_per_iteration_y = (is_qasymm && conv_stride_y == 1 && dilation.y() == 1) ? 2 : 1;
        num_elems_read_per_iteration_x    = 3 + (num_elems_written_per_iteration_x - 1) * conv_stride_x + (conv_stride_x > 1 ? 1 : 0);
        num_elems_read_per_iteration_y    = num_elems_written_per_iteration_y + 2;
    }
    num_elems_read_per_iteration_x += (num_elems_read_per_iteration_x - 1) * (dilation.x() - 1);
    num_elems_read_per_iteration_y += (num_elems_read_per_iteration_y - 1) * (dilation.y() - 1);

    // Create window and update padding
    Window win = calculate_max_window(*output, Steps(num_elems_written_per_iteration_x, num_elems_written_per_iteration_y));

    AccessWindowRectangle input_access(input, -conv_info.pad_left(), -conv_info.pad_top(),
                                       num_elems_read_per_iteration_x, num_elems_read_per_iteration_y,
                                       conv_stride_x, conv_stride_y);
    AccessWindowStatic    weights_access(weights, 0, 0, 3, 3);
    AccessWindowRectangle output_access(output, 0, 0, num_elems_written_per_iteration_x, num_elems_written_per_iteration_y);

    bool window_changed = update_window_and_padding(win, input_access, weights_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLDepthwiseConvolutionLayer3x3NCHWKernel::CLDepthwiseConvolutionLayer3x3NCHWKernel()
    : _conv_stride_x(0), _conv_pad_top(0), _conv_pad_left(0)
{
}

BorderSize CLDepthwiseConvolutionLayer3x3NCHWKernel::border_size() const
{
    return _border_size;
}

void CLDepthwiseConvolutionLayer3x3NCHWKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                                         unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info, depth_multiplier, act_info, dilation));

    bool is_qasymm = is_data_type_quantized_asymmetric(input->info()->data_type());

    _input         = input;
    _output        = output;
    _weights       = weights;
    _biases        = biases;
    _conv_stride_x = conv_info.stride().first;
    _conv_stride_y = conv_info.stride().second;
    _conv_pad_left = conv_info.pad_left();
    _conv_pad_top  = conv_info.pad_top();
    _border_size   = BorderSize(_conv_pad_top, conv_info.pad_right(), conv_info.pad_bottom(), _conv_pad_left);

    // Configure kernel window
    std::string     kernel_name;
    const GPUTarget gpu_target = get_target();

    auto win_config = validate_and_configure_window(input->info(), weights->info(), output->info(), conv_info, depth_multiplier, gpu_target, kernel_name, dilation);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_info.activation())));
    build_opts.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(_output->info()->tensor_shape().z()));
    build_opts.add_option("-DDEPTH_MULTIPLIER=" + support::cpp11::to_string(depth_multiplier));
    build_opts.add_option("-DCONV_STRIDE_X=" + support::cpp11::to_string(_conv_stride_x));
    build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
    build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));
    build_opts.add_option_if(_biases != nullptr, "-DHAS_BIAS");

    if(is_qasymm)
    {
        const UniformQuantizationInfo iq_info = _input->info()->quantization_info().uniform();
        const UniformQuantizationInfo wq_info = _weights->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = _output->info()->quantization_info().uniform();

        float multiplier        = iq_info.scale * wq_info.scale / oq_info.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        build_opts.add_option("-DCONV_STRIDE_Y=" + support::cpp11::to_string(_conv_stride_y));
        build_opts.add_option("-DINPUT_OFFSET=" + support::cpp11::to_string(-iq_info.offset));
        build_opts.add_option("-DWEIGHTS_OFFSET=" + support::cpp11::to_string(-wq_info.offset));
        build_opts.add_option("-DOUTPUT_OFFSET=" + support::cpp11::to_string(oq_info.offset));
        build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(9 * iq_info.offset * wq_info.offset));
        build_opts.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
        build_opts.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));

        if(act_info.enabled())
        {
            const int a_val = quantize_qasymm8(act_info.a(), oq_info);
            const int b_val = quantize_qasymm8(act_info.b(), oq_info);
            const int o1    = oq_info.offset;

            build_opts.add_option("-DA_VAL=" + support::cpp11::to_string(a_val));
            build_opts.add_option("-DB_VAL=" + support::cpp11::to_string(b_val));
            build_opts.add_option("-DCONST_0=" + support::cpp11::to_string(o1));

            const float s1 = iq_info.scale;
            build_opts.add_option("-DS1_VAL=" + float_to_string_with_full_precision(s1));
            build_opts.add_option("-DO1_VAL=" + support::cpp11::to_string(o1));
        }
    }
    else
    {
        build_opts.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
        build_opts.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
        build_opts.add_option_if(act_info.enabled(), "-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
        build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(win_config.second.x().step()));
    }

    build_opts.add_option_if(input->info()->data_type() == DataType::F16, "-DIS_F16");
    build_opts.add_option_if(input->info()->data_type() == DataType::F32, "-DIS_F32");

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
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
}

Status CLDepthwiseConvolutionLayer3x3NCHWKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                          unsigned int depth_multiplier, ActivationLayerInfo act_info, GPUTarget gpu_target, const Size2D &dilation)
{
    std::string kernel_name;
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(), output->clone().get(), conv_info, depth_multiplier, gpu_target, kernel_name, dilation).first);

    return Status{};
}

void CLDepthwiseConvolutionLayer3x3NCHWKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    // Create input window and adjust
    Window collapsed_in = collapsed;
    collapsed_in.adjust(Window::DimX, -_conv_pad_left, true);
    collapsed_in.adjust(Window::DimY, -_conv_pad_top, true);
    collapsed_in.set_dimension_step(Window::DimX, collapsed_in.x().step() * _conv_stride_x);
    collapsed_in.set_dimension_step(Window::DimY, collapsed_in.y().step() * _conv_stride_y);

    Window slice_in      = collapsed_in.first_slice_window_3D();
    Window slice_out     = collapsed.first_slice_window_3D();
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

        enqueue(queue, *this, slice_out, lws_hint());
    }
    while(collapsed.slide_window_slice_3D(slice_out) && collapsed_in.slide_window_slice_3D(slice_in));
}
