/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NHWCKernel.h"

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

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info, unsigned int depth_multiplier,
                          const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((act_info.enabled()) && (input->data_type() == DataType::F32 || ((act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
                                                                                                     && (act_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
                                                                                                     && (act_info.activation() != ActivationLayerInfo::ActivationFunction::RELU))),
                                    "For QASYMM8 only relu, lower bounded relu and lower-upper bounded relu are supported"); //COMPMID-1317 add fused activation for F32
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON(depth_multiplier > 1); // COMPMID-1071 Add depth multiplier support for NHWC
    ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(1) != 3 || weights->dimension(2) != 3);

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
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != weights->dimension(0));
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if(output->total_size() != 0)
    {
        const TensorShape output_shape = compute_depthwise_convolution_shape(*input, *weights, conv_info, depth_multiplier);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *output,
                                                        const PadStrideInfo &conv_info)
{
    const bool is_qasymm   = is_data_type_quantized_asymmetric(input->data_type());
    const bool is_stride_1 = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));

    const unsigned int num_rows_processed_per_iteration = is_qasymm ? 4 : (is_stride_1 ? 2 : 1);
    const unsigned int num_elems_accessed_per_iteration = is_qasymm ? 4 : 2;
    const unsigned int num_rows_read_per_iteration      = num_rows_processed_per_iteration + 2;
    const unsigned int num_rows_written_per_iteration   = std::ceil(num_rows_processed_per_iteration / static_cast<float>(conv_info.stride().first));

    BorderSize border_size;
    if(is_qasymm)
    {
        border_size = BorderSize(std::max(conv_info.pad_left(), conv_info.pad_top()), 0, std::max(conv_info.pad_right(), conv_info.pad_bottom()), 0);
    }
    else
    {
        border_size = BorderSize(conv_info.pad_left(), 0, std::max(std::max(conv_info.pad_right(), conv_info.pad_bottom()), conv_info.pad_top()), 0);
    }

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_accessed_per_iteration, num_rows_written_per_iteration));

    AccessWindowStatic input_access(input, 0, -border_size.top, ceil_to_multiple(input->dimension(0), num_elems_accessed_per_iteration),
                                    ceil_to_multiple(input->dimension(1) + border_size.bottom, num_rows_read_per_iteration));
    AccessWindowRectangle  output_access(output, 0, 0, num_elems_accessed_per_iteration, num_rows_written_per_iteration);
    AccessWindowHorizontal weights_access(weights, 0, num_elems_accessed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, weights_access, output_access);

    if(bias != nullptr)
    {
        AccessWindowHorizontal bias_access(bias, 0, num_elems_accessed_per_iteration);
        window_changed = window_changed || update_window_and_padding(win, bias_access);
    }

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLDepthwiseConvolutionLayer3x3NHWCKernel::CLDepthwiseConvolutionLayer3x3NHWCKernel()
    : _num_rows_processed_per_iteration(1), _num_planes_processed_per_iteration(1)
{
}

BorderSize CLDepthwiseConvolutionLayer3x3NHWCKernel::border_size() const
{
    return _border_size;
}

void CLDepthwiseConvolutionLayer3x3NHWCKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output, const PadStrideInfo &conv_info,
                                                         unsigned int        depth_multiplier,
                                                         ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    // Get convolved dimensions
    const TensorShape output_shape = compute_depthwise_convolution_shape(*input->info(), *weights->info(), conv_info, depth_multiplier);

    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(),
                       output_shape,
                       1,
                       input->info()->data_type(),
                       input->info()->quantization_info());

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(), conv_info, depth_multiplier, act_info));

    const unsigned int conv_stride_x = conv_info.stride().first;
    ARM_COMPUTE_ERROR_ON(conv_stride_x < 1 || conv_stride_x > 2);
    ARM_COMPUTE_ERROR_ON(std::max(conv_info.pad_top(), conv_info.pad_bottom()) > 1);

    const bool is_qasymm   = is_data_type_quantized_asymmetric(input->info()->data_type());
    const bool is_stride_1 = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));

    _input                              = input;
    _output                             = output;
    _weights                            = weights;
    _biases                             = biases;
    _conv_stride_y                      = conv_info.stride().second;
    _num_rows_processed_per_iteration   = is_qasymm ? 4 : (is_stride_1 ? 2 : 1);
    _num_planes_processed_per_iteration = (is_stride_1 && !is_qasymm) ? 2 : 1;

    const unsigned int num_elems_accessed_per_iteration = is_qasymm ? 4 : 2;

    if(is_qasymm)
    {
        _border_size = BorderSize(std::max(conv_info.pad_left(), conv_info.pad_top()), 0, std::max(conv_info.pad_right(), conv_info.pad_bottom()), 0);
    }
    else
    {
        _border_size = BorderSize(conv_info.pad_left(), 0, std::max(std::max(conv_info.pad_right(), conv_info.pad_bottom()), conv_info.pad_top()), 0);
    }

    CLBuildOptions build_opts;
    build_opts.add_option_if(_biases != nullptr, "-DHAS_BIAS");
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_accessed_per_iteration));
    build_opts.add_option("-DSRC_DIM_2=" + support::cpp11::to_string(_input->info()->dimension(2)));
    build_opts.add_option("-DCONV_PAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.add_option("-DCONV_PAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));

    if(is_qasymm)
    {
        float multiplier        = _input->info()->quantization_info().scale * _weights->info()->quantization_info().scale / _output->info()->quantization_info().scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        quantization::calculate_quantized_multiplier_less_than_one(multiplier, &output_multiplier, &output_shift);

        build_opts.add_option("-DSRC_DIM_1=" + support::cpp11::to_string(_input->info()->dimension(1)));
        build_opts.add_option("-DINPUT_OFFSET=" + support::cpp11::to_string(-_input->info()->quantization_info().offset));
        build_opts.add_option("-DWEIGHTS_OFFSET=" + support::cpp11::to_string(-_weights->info()->quantization_info().offset));
        build_opts.add_option("-DOUTPUT_OFFSET=" + support::cpp11::to_string(_output->info()->quantization_info().offset));
        build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(9 * input->info()->quantization_info().offset * weights->info()->quantization_info().offset));
        build_opts.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
        build_opts.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));

        if(act_info.enabled())
        {
            const int a_val = input->info()->quantization_info().quantize(act_info.a(), RoundingPolicy::TO_NEAREST_UP);
            const int b_val = input->info()->quantization_info().quantize(act_info.b(), RoundingPolicy::TO_NEAREST_UP);
            const int o1    = input->info()->quantization_info().offset;

            build_opts.add_option("-DFUSED_ACTIVATION=" + lower_string(string_from_activation_func(act_info.activation())));
            build_opts.add_option("-DA_VAL=" + support::cpp11::to_string(a_val));
            build_opts.add_option("-DB_VAL=" + support::cpp11::to_string(b_val));
            build_opts.add_option("-DCONST_0=" + support::cpp11::to_string(o1));

            if(output != nullptr)
            {
                const float s1 = input->info()->quantization_info().scale;
                const float s2 = output->info()->quantization_info().scale;
                const int   o2 = output->info()->quantization_info().offset;

                if(o1 != o2 || s1 != s2)
                {
                    build_opts.add_option("-DS1_VAL=" + float_to_string_with_full_precision(s1));
                    build_opts.add_option("-DS2_VAL=" + float_to_string_with_full_precision(s2));
                    build_opts.add_option("-DO1_VAL=" + support::cpp11::to_string(o1));
                    build_opts.add_option("-DO2_VAL=" + support::cpp11::to_string(o2));
                }
            }
        }
    }
    else if(is_stride_1)
    {
        build_opts.add_option("-DNUM_ROWS_PROCESSED=" + support::cpp11::to_string(_num_rows_processed_per_iteration));
        build_opts.add_option("-DNUM_PLANES_PROCESSED=" + support::cpp11::to_string(_num_planes_processed_per_iteration));
        build_opts.add_option("-DDST_DIM_2=" + support::cpp11::to_string(_output->info()->dimension(2)));
    }
    else
    {
        build_opts.add_option("-DCONV_STRIDE_X=" + support::cpp11::to_string(conv_stride_x));
        build_opts.add_option("-DCONV_STRIDE_Y=" + support::cpp11::to_string(_conv_stride_y));
    }

    // Create kernel
    std::string kernel_name = std::string("depthwise_convolution_3x3") + (is_qasymm ? std::string("_quantized") : std::string()) + std::string("_nhwc");
    if(is_qasymm || is_stride_1)
    {
        kernel_name += std::string("_stride") + support::cpp11::to_string(conv_stride_x);
    }

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(), conv_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

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
    _config_id += string_from_data_type(input->info()->data_type());
}

Status CLDepthwiseConvolutionLayer3x3NHWCKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,
                                                          unsigned int        depth_multiplier,
                                                          ActivationLayerInfo act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info, depth_multiplier, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(),
                                                              biases != nullptr ? biases->clone().get() : nullptr,
                                                              output->clone().get(), conv_info)
                                .first);

    return Status{};
}

void CLDepthwiseConvolutionLayer3x3NHWCKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window win = window;
    win.set(Window::DimZ, Window::Dimension(0, std::ceil(_output->info()->dimension(2) / static_cast<float>(_num_planes_processed_per_iteration)), 1));

    // Create input window and adjust
    Window win_in = win;
    win_in.set_dimension_step(Window::DimY, _num_rows_processed_per_iteration);
    win_in.set_dimension_step(Window::DimZ, _conv_stride_y);

    ARM_COMPUTE_ERROR_ON((win_in.y().step() < window.y().step()) || (win_in.z().step() < window.z().step()));

    Window slice_in  = win_in.first_slice_window_3D();
    Window slice_out = win.first_slice_window_3D();

    if(_biases != nullptr)
    {
        unsigned int idx = 3 * num_arguments_per_3D_tensor();
        Window       win_biases;
        win_biases.use_tensor_dimensions(_biases->info()->tensor_shape());
        win_biases.set_dimension_step(Window::DimX, window.x().step());
        add_1D_tensor_argument(idx, _biases, win_biases);
    }

    if(!(is_data_type_quantized_asymmetric(_input->info()->data_type())))
    {
        unsigned int idx        = 3 * num_arguments_per_3D_tensor() + ((_biases != nullptr) ? num_arguments_per_1D_tensor() : 0);
        const int    max_offset = _input->info()->strides_in_bytes().z() * _input->info()->dimension(2) - (_input->info()->padding().bottom + _input->info()->padding().top) *
                                  _input->info()->strides_in_bytes().y();

        _kernel.setArg(idx, max_offset);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_3D_tensor_argument(idx, _output, slice_out);
        add_3D_tensor_argument(idx, _weights, slice_out);

        enqueue(queue, *this, slice_out, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice_out) && win_in.slide_window_slice_3D(slice_in));
}
