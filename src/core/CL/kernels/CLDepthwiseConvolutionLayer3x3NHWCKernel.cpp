/*
 * Copyright (c) 2018-2020 Arm Limited.
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
#include "src/core/CL/kernels/CLDepthwiseConvolutionLayer3x3NHWCKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "src/core/AccessWindowStatic.h"
#include "src/core/CL/CLValidate.h"
#include "src/core/CL/ICLKernel.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                          const PadStrideInfo &conv_info, unsigned int depth_multiplier, const ActivationLayerInfo &act_info, const Size2D &dilation,
                          const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((act_info.enabled()) && (input->data_type() == DataType::QASYMM8 || input->data_type() == DataType::QASYMM8_SIGNED)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::BOUNDED_RELU)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::RELU)
                                    && (act_info.activation() != ActivationLayerInfo::ActivationFunction::LOGISTIC),
                                    "For QASYMM8 only logistic, relu, lower bounded relu and lower-upper bounded relu are supported");
    ARM_COMPUTE_RETURN_ERROR_ON(depth_multiplier > 1); // COMPMID-1071 Add depth multiplier support for NHWC

    ARM_COMPUTE_RETURN_ERROR_ON(conv_info.stride().first < 1);
    ARM_COMPUTE_RETURN_ERROR_ON(std::max(conv_info.pad_top(), conv_info.pad_bottom()) > 4);

    ARM_COMPUTE_RETURN_ERROR_ON((dilation.x() < 1) || (dilation.y() < 1));

    const bool   is_qasymm      = is_data_type_quantized_asymmetric(input->data_type());
    const size_t weights_width  = 3;
    const size_t weights_height = 3;

    const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_depthwise_convolution_shape(
                                         *input, TensorInfo(TensorShape(weights_width, weights_height), 1, weights->data_type()).set_data_layout(DataLayout::NCHW), conv_info, depth_multiplier, dilation);
    if(is_qasymm)
    {
        DepthwiseConvolutionReshapeInfo info;
        info.c0 = 4;
        ARM_COMPUTE_RETURN_ERROR_ON((weights->dimension(0) / info.c0) != weights_width * weights_height);

        ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output_multipliers, output_shifts);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_multipliers, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_shifts, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(output_multipliers->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(output_shifts->num_dimensions() > 1);

        if(is_data_type_quantized_per_channel(weights->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON(output_shape[0] != output_multipliers->dimension(0));
            ARM_COMPUTE_RETURN_ERROR_ON(output_shape[0] != output_shifts->dimension(0));
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
        ARM_COMPUTE_RETURN_ERROR_ON((weights->dimension(1) != weights_width) || (weights->dimension(2) != weights_height));
    }

    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(biases->dimension(0) != output_shape[0]);
        if(is_qasymm)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }

        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }

    if(output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *weights, ITensorInfo *bias, ITensorInfo *output,
                                                        const PadStrideInfo &conv_info, unsigned int depth_multiplier, const Size2D &dilation,
                                                        ITensorInfo *output_multipliers, ITensorInfo *output_shifts)
{
    ARM_COMPUTE_UNUSED(weights);
    ARM_COMPUTE_UNUSED(depth_multiplier);

    const bool   is_stride_1_dilation_1           = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1) && dilation.x() == 1 && dilation.y() == 1);
    unsigned int num_rows_processed_per_iteration = is_stride_1_dilation_1 ? 2 : 1;

    Window win{};
    Status err{};

    if(is_data_type_quantized_asymmetric(input->data_type()))
    {
        const unsigned int num_elems_accessed_per_iteration = 4;
        const unsigned int num_rows_read_per_iteration      = num_rows_processed_per_iteration + 2;
        const unsigned int num_rows_written_per_iteration   = std::ceil(num_rows_processed_per_iteration / static_cast<float>(conv_info.stride().first));

        BorderSize border_size;
        border_size = BorderSize(conv_info.pad_left(), 0, std::max(std::max(conv_info.pad_right(), conv_info.pad_bottom()), conv_info.pad_top()), 0);

        // Configure kernel window
        win = calculate_max_window(*output, Steps(num_elems_accessed_per_iteration, num_rows_written_per_iteration));

        AccessWindowStatic input_access(input, 0, -border_size.top, ceil_to_multiple(input->dimension(0), num_elems_accessed_per_iteration),
                                        ceil_to_multiple(input->dimension(1) + border_size.bottom, num_rows_read_per_iteration));
        AccessWindowRectangle output_access(output, 0, 0, num_elems_accessed_per_iteration, num_rows_written_per_iteration);

        bool window_changed = false;

        if((output_multipliers != nullptr) && (output_shifts != nullptr))
        {
            AccessWindowHorizontal output_multipliers_access(output_multipliers, 0, num_elems_accessed_per_iteration);
            AccessWindowHorizontal output_shifts_access(output_shifts, 0, num_elems_accessed_per_iteration);
            window_changed = window_changed || update_window_and_padding(win, input_access, output_access, output_multipliers_access, output_shifts_access);
        }
        else
        {
            Status err = ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "output_multipliers and output_shifts must be non-nullptr for quantized input");
            return std::make_pair(err, win);
        }

        if(bias != nullptr)
        {
            AccessWindowHorizontal bias_access(bias, 0, num_elems_accessed_per_iteration);
            window_changed = window_changed || update_window_and_padding(win, bias_access);
        }
        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

        err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    }
    else
    {
        unsigned int num_elems_accessed_per_iteration = adjust_vec_size(4 / input->element_size(), input->dimension(0));
        win                                           = calculate_max_window(*output, Steps(num_elems_accessed_per_iteration, num_rows_processed_per_iteration));
    }

    return std::make_pair(err, win);
}
} // namespace

CLDepthwiseConvolutionLayer3x3NHWCKernel::CLDepthwiseConvolutionLayer3x3NHWCKernel()
    : _num_planes_processed_per_iteration(1)
{
}

BorderSize CLDepthwiseConvolutionLayer3x3NHWCKernel::border_size() const
{
    return _border_size;
}

void CLDepthwiseConvolutionLayer3x3NHWCKernel::configure(const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                                         const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation,
                                                         const ICLTensor *output_multipliers, const ICLTensor *output_shifts)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation, output_multipliers, output_shifts);
}

void CLDepthwiseConvolutionLayer3x3NHWCKernel::configure(const CLCompileContext &compile_context, const ICLTensor *input, const ICLTensor *weights, const ICLTensor *biases, ICLTensor *output,
                                                         const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation,
                                                         const ICLTensor *output_multipliers, const ICLTensor *output_shifts)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), weights->info(), (biases != nullptr) ? biases->info() : nullptr, output->info(),
                                                  conv_info, depth_multiplier, act_info, dilation,
                                                  (output_multipliers != nullptr) ? output_multipliers->info() : nullptr,
                                                  (output_shifts != nullptr) ? output_shifts->info() : nullptr));

    auto padding_info = get_padding_info({ input, weights, biases, output });

    auto win_config = validate_and_configure_window(input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(),
                                                    conv_info, depth_multiplier, dilation,
                                                    (output_multipliers != nullptr) ? output_multipliers->info() : nullptr,
                                                    (output_shifts != nullptr) ? output_shifts->info() : nullptr);

    const bool is_stride_1              = ((conv_info.stride().first == conv_info.stride().second) && (conv_info.stride().first == 1));
    const bool is_stride_1_dilation_1   = (is_stride_1 && dilation.x() == 1 && dilation.y() == 1);
    const bool is_quantized_per_channel = is_data_type_quantized_per_channel(weights->info()->data_type());
    const bool is_dot8_supported        = dot8_supported(CLKernelLibrary::get().get_device()) && !is_quantized_per_channel;

    _input                              = input;
    _output                             = output;
    _weights                            = weights;
    _biases                             = biases;
    _conv_stride_y                      = conv_info.stride().second;
    _num_planes_processed_per_iteration = is_stride_1_dilation_1 ? 2 : 1;
    _output_multipliers                 = output_multipliers;
    _output_shifts                      = output_shifts;
    _is_quantized                       = is_data_type_quantized_asymmetric(input->info()->data_type());

    if(_is_quantized)
    {
        _border_size = BorderSize(is_stride_1 ? 0 : conv_info.pad_left(), 0, std::max(std::max(conv_info.pad_right(), conv_info.pad_bottom()), conv_info.pad_top()), 0);

        // If QASYMM8 and the 8 bit dot product is available, force _num_planes_processed_per_iteration to 1
        if(is_dot8_supported)
        {
            _num_planes_processed_per_iteration = 1;
        }
    }

    unsigned int num_elems_accessed_per_iteration = _is_quantized ? 4 : adjust_vec_size(4 / input->info()->element_size(), input->info()->dimension(0));
    unsigned int num_rows_processed_per_iteration = is_stride_1_dilation_1 ? 2 : 1;

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(_input->info()->data_type()));
    build_opts.add_option("-DACTIVATION_TYPE=" + lower_string(string_from_activation_func(act_info.activation())));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(num_elems_accessed_per_iteration));
    build_opts.add_option("-DSRC_DIM_1=" + support::cpp11::to_string(_input->info()->dimension(1)));
    build_opts.add_option("-DSRC_DIM_2=" + support::cpp11::to_string(_input->info()->dimension(2)));
    build_opts.add_option("-DCONV_PAD_TOP=" + support::cpp11::to_string(conv_info.pad_top()));
    build_opts.add_option("-DCONV_PAD_LEFT=" + support::cpp11::to_string(conv_info.pad_left()));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(input->info()->dimension(0) % num_elems_accessed_per_iteration));
    build_opts.add_option_if(_biases != nullptr, "-DHAS_BIAS");
    build_opts.add_option_if(_input->info()->tensor_shape().total_size_upper(3) > 1,
                             "-DDST_DEPTH=" + support::cpp11::to_string(static_cast<int>(std::ceil(_output->info()->dimension(2) / static_cast<float>(_num_planes_processed_per_iteration)))));

    if(_is_quantized)
    {
        const UniformQuantizationInfo iq_info = _input->info()->quantization_info().uniform();
        const UniformQuantizationInfo wq_info = _weights->info()->quantization_info().uniform();
        const UniformQuantizationInfo oq_info = _output->info()->quantization_info().uniform();

        build_opts.add_option("-DSRC_DIM_1=" + support::cpp11::to_string(_input->info()->dimension(1)));
        build_opts.add_option("-DINPUT_OFFSET=" + support::cpp11::to_string(-iq_info.offset));
        build_opts.add_option("-DWEIGHTS_OFFSET=" + support::cpp11::to_string(-wq_info.offset));
        build_opts.add_option("-DOUTPUT_OFFSET=" + support::cpp11::to_string(oq_info.offset));
        build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(9 * iq_info.offset * wq_info.offset));
        build_opts.add_option_if(is_quantized_per_channel, "-DPER_CHANNEL_QUANTIZATION");
        build_opts.add_option_if(is_dot8_supported, "-DIS_DOT8");

        // Compute non-per-channel multiplier and shift anyway to make OpenCL kernel simpler
        float multiplier        = iq_info.scale * wq_info.scale / oq_info.scale;
        int   output_multiplier = 0;
        int   output_shift      = 0;
        quantization::calculate_quantized_multiplier(multiplier, &output_multiplier, &output_shift);
        build_opts.add_option("-DOUTPUT_MULTIPLIER=" + support::cpp11::to_string(output_multiplier));
        build_opts.add_option("-DOUTPUT_SHIFT=" + support::cpp11::to_string(output_shift));

        if(act_info.enabled())
        {
            int a_val{};
            int b_val{};
            std::tie(b_val, a_val) = get_quantized_activation_min_max(act_info, input->info()->data_type(), oq_info);

            const int o1 = oq_info.offset;

            build_opts.add_option("-DA_VAL=" + support::cpp11::to_string(a_val));
            build_opts.add_option("-DB_VAL=" + support::cpp11::to_string(b_val));
            build_opts.add_option("-DCONST_0=" + support::cpp11::to_string(o1));

            const float s1 = iq_info.scale;
            build_opts.add_option("-DS1_VAL=" + float_to_string_with_full_precision(s1));
            build_opts.add_option("-DO1_VAL=" + support::cpp11::to_string(o1));
        }

        build_opts.add_option("-DWEIGHTS_TYPE=" + get_cl_type_from_data_type(weights->info()->data_type()));
        build_opts.add_option("-DWEIGHTS_PROMOTED_TYPE=" + get_cl_promoted_type_from_data_type(weights->info()->data_type()));
    }
    else
    {
        build_opts.add_option_if(act_info.enabled(), "-DA_VAL=" + float_to_string_with_full_precision(act_info.a()));
        build_opts.add_option_if(act_info.enabled(), "-DB_VAL=" + float_to_string_with_full_precision(act_info.b()));
    }

    if(is_stride_1_dilation_1)
    {
        build_opts.add_option("-DNUM_ROWS_PROCESSED=" + support::cpp11::to_string(num_rows_processed_per_iteration));
        build_opts.add_option("-DNUM_PLANES_PROCESSED=" + support::cpp11::to_string(_num_planes_processed_per_iteration));
        build_opts.add_option("-DDST_DIM_1=" + support::cpp11::to_string(_output->info()->dimension(1)));
        build_opts.add_option("-DDST_DIM_2=" + support::cpp11::to_string(_output->info()->dimension(2)));
        build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string((input->info()->dimension(1) + conv_info.pad_left() + conv_info.pad_right()) % num_rows_processed_per_iteration));
    }
    else
    {
        build_opts.add_option("-DCONV_STRIDE_X=" + support::cpp11::to_string(conv_info.stride().first));
        build_opts.add_option("-DCONV_STRIDE_Y=" + support::cpp11::to_string(_conv_stride_y));
        build_opts.add_option("-DDILATION_X=" + support::cpp11::to_string(dilation.x()));
        build_opts.add_option("-DDILATION_Y=" + support::cpp11::to_string(dilation.y()));
    }

    std::string kernel_name;
    // Create kernel
    if(_is_quantized)
    {
        kernel_name = std::string("dwc_3x3_reshaped_quantized8");
        kernel_name += (is_dot8_supported && is_stride_1_dilation_1 ? "_dot8" : "");
        kernel_name += (is_stride_1_dilation_1 ? "_stride1" : "");
        kernel_name += "_nhwc";
    }
    else
    {
        kernel_name = std::string("depthwise_convolution_3x3_nhwc");
        kernel_name += (is_stride_1_dilation_1 ? "_stride1" : "");
    }

    ICLKernel::configure_internal(win_config.second);
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    ARM_COMPUTE_ERROR_ON(!_is_quantized && has_padding_changed(padding_info));

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

Status CLDepthwiseConvolutionLayer3x3NHWCKernel::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output,
                                                          const PadStrideInfo &conv_info, unsigned int depth_multiplier, ActivationLayerInfo act_info, const Size2D &dilation,
                                                          const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, weights, biases, output, conv_info, depth_multiplier, act_info, dilation, output_multipliers, output_shifts));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), weights->clone().get(),
                                                              biases != nullptr ? biases->clone().get() : nullptr,
                                                              output->clone().get(), conv_info, depth_multiplier, dilation,
                                                              (output_multipliers != nullptr) ? output_multipliers->clone().get() : nullptr,
                                                              (output_shifts != nullptr) ? output_shifts->clone().get() : nullptr)
                                .first);
    return Status{};
}

void CLDepthwiseConvolutionLayer3x3NHWCKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    const size_t total_batches = _input->info()->tensor_shape().total_size_upper(3);

    Window win = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    win.set(Window::DimZ, Window::Dimension(0, std::ceil(_output->info()->dimension(2) / static_cast<float>(_num_planes_processed_per_iteration)) * total_batches, 1));

    unsigned int idx = 2 * num_arguments_per_4D_tensor() + (_is_quantized ? num_arguments_per_2D_tensor() : num_arguments_per_3D_tensor());

    if(_is_quantized)
    {
        Window slice;
        slice.use_tensor_dimensions(_output_multipliers->info()->tensor_shape());
        slice.set_dimension_step(Window::DimX, window.x().step());
        add_1D_tensor_argument(idx, _output_multipliers, slice);
        add_1D_tensor_argument(idx, _output_shifts, slice);
    }

    if(_biases != nullptr)
    {
        Window win_biases;
        win_biases.use_tensor_dimensions(_biases->info()->tensor_shape());
        win_biases.set_dimension_step(Window::DimX, window.x().step());
        add_1D_tensor_argument(idx, _biases, win_biases);
    }

    if(_is_quantized)
    {
        // Calculate the max_offset.
        // max_offset is the offset for the last NOT valid value in the Z dimension (spatial dimension Y for NHWC)
        //  |******************|
        //  |     pad_top      |
        //  |******************|
        //  |                  |
        //  |      plane0      |
        //  |      batch0      |
        //  |__________________|
        //  |******************|       Batch 0
        //  |    pad_bottom    |
        //  |     pad_top      |
        //  |******************|
        //  |                  |
        //  |      plane1      |
        //  |      batch0      |
        //  |__________________|-----> max_offset
        //  |******************|
        //  |    pad_bottom    |
        //  |     pad_top      |
        //  |******************|
        //  |                  |
        //  |      plane0      |
        //  |      batch1      |
        //  |__________________|
        //  |******************|       Batch 1
        //  |    pad_bottom    |
        //  |     pad_top      |
        //  |******************|
        //  |                  |
        //  |      plane1      |
        //  |      batch1      |
        //  |__________________|
        //  |     pad_bottom   |
        //  |******************|
        const int max_offset = _input->info()->strides_in_bytes().z() * _input->info()->dimension(2) - (_input->info()->padding().bottom + _input->info()->padding().top) *
                               _input->info()->strides_in_bytes().y();
        _kernel.setArg(idx, max_offset);
    }

    Window slice = win.first_slice_window_4D();
    do
    {
        unsigned int idx = 0;
        add_4D_tensor_argument(idx, _input, slice);
        add_4D_tensor_argument(idx, _output, slice);
        if(_is_quantized)
        {
            add_2D_tensor_argument(idx, _weights, slice);
        }
        else
        {
            add_3D_tensor_argument(idx, _weights, slice);
        }
        enqueue(queue, *this, slice, lws_hint());
    }
    while(win.slide_window_slice_4D(slice));
}
} // namespace arm_compute
