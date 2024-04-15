/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/ClTransposedConvolutionKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/core/utils/StringUtils.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace
{
Status validate_arguments(const ITensorInfo   *input,
                          const ITensorInfo   *weights,
                          const ITensorInfo   *biases,
                          const ITensorInfo   *output,
                          const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32,
                                                         DataType::QASYMM8_SIGNED, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(weights, DataLayout::NHWC);

    constexpr unsigned int channel_idx = 0;
    constexpr unsigned int width_idx   = 1;
    constexpr unsigned int height_idx  = 2;
    constexpr unsigned int batch_idx   = 3;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->dimension(channel_idx) != input->dimension(channel_idx),
                                    "Weights feature map dimension should match the respective src's one");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(weights->num_dimensions() > 4, "Weights can be at most 4 dimensional");

    if (biases != nullptr)
    {
        if (is_data_type_quantized_asymmetric(input->data_type()))
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(biases, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(weights, biases);
        }

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->dimension(channel_idx) != weights->dimension(batch_idx),
                                        "Biases size and number of dst feature maps should match");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(biases->num_dimensions() > 1, "Biases should be one dimensional");
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(input, DataLayout::NHWC);
    }

    // Checks performed when output is configured
    if (output->total_size() != 0)
    {
        const size_t input_width    = input->dimension(width_idx);
        const size_t input_height   = input->dimension(height_idx);
        const size_t weights_width  = weights->dimension(width_idx);
        const size_t weights_height = weights->dimension(height_idx);

        auto out_dims =
            deconvolution_output_dimensions(input_width, input_height, weights_width, weights_height, deconv_info);
        TensorShape output_shape =
            misc::shape_calculator::compute_deconvolution_output_shape(out_dims, *input, *weights);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_LAYOUT_NOT_IN(output, DataLayout::NHWC);
    }

    return Status{};
}
} // namespace

void ClTransposedConvolutionKernel::configure(const CLCompileContext &compile_context,
                                              const ITensorInfo      *input,
                                              const ITensorInfo      *weights,
                                              const ITensorInfo      *biases,
                                              ITensorInfo            *output,
                                              const PadStrideInfo    &deconv_info)
{
    ARM_COMPUTE_UNUSED(biases, deconv_info);
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

    // Perform validation
    ARM_COMPUTE_ERROR_THROW_ON(validate(input, weights, biases, output, deconv_info));

    constexpr unsigned int channel_idx = 0;
    constexpr unsigned int width_idx   = 1;
    constexpr unsigned int height_idx  = 2;

    const size_t input_channels  = input->dimension(channel_idx); // same as weight channels
    const size_t input_width     = input->dimension(width_idx);
    const size_t input_height    = input->dimension(height_idx);
    const size_t weights_width   = weights->dimension(width_idx);
    const size_t weights_height  = weights->dimension(height_idx);
    const size_t output_width    = output->dimension(width_idx);
    const size_t output_height   = output->dimension(height_idx);
    const size_t output_channels = output->dimension(channel_idx);

    // Calculate output shape
    auto out_dims =
        deconvolution_output_dimensions(input_width, input_height, weights_width, weights_height, deconv_info);
    TensorShape output_shape = misc::shape_calculator::compute_deconvolution_output_shape(out_dims, *input, *weights);
    auto_init_if_empty(*output, output_shape, 1, input->data_type(), input->quantization_info());

    // Calculate updated paddings
    // p' = k - p - 1 (k: kernel dimensions)
    const uint32_t pad_left = weights_width - deconv_info.pad_left() - 1;
    const uint32_t pad_top  = weights_height - deconv_info.pad_top() - 1;

    // Configure kernel window
    Window win;
    output_shape.collapse(2U, 1U); // Collapse width and height into single dimension

    const unsigned int n0               = adjust_vec_size(16 / output->element_size(), output_channels);
    const unsigned int m0               = 1;
    const unsigned int k0               = adjust_vec_size(16 / input->element_size(), input_channels);
    const unsigned int partial_store_n0 = output_channels % n0;

    // Create window and update padding
    win = calculate_max_window(output_shape, Steps(n0, m0));
    ICLKernel::configure_internal(win);

    const std::string kernel_name = "transposed_convolution_nhwc";
    CLBuildOptions    build_options;

    const DataType    input_data_type = input->data_type();
    const PaddingInfo strides         = deconv_info.stride();

    if (biases != nullptr)
    {
        build_options.add_option(std::string("-DHAS_BIAS"));
        build_options.add_option(std::string("-DBIA_DATA_TYPE=" + get_cl_type_from_data_type(biases->data_type())));
    }

    const auto output_data_type = output->data_type();

    build_options.add_option("-cl-fast-relaxed-math");
    build_options.add_option("-DSRC_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DSRC_DATA_TYPE=" + get_cl_type_from_data_type(input_data_type));
    build_options.add_option("-DSRC_CHANNELS=" + support::cpp11::to_string(input_channels));
    build_options.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(input_width));
    build_options.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(input_height));
    build_options.add_option("-DDST_CHANNELS=" + support::cpp11::to_string(output_channels));
    build_options.add_option("-DDST_WIDTH=" + support::cpp11::to_string(output_width));
    build_options.add_option("-DDST_HEIGHT=" + support::cpp11::to_string(output_height));
    build_options.add_option("-DDST_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DDST_DATA_TYPE=" + get_cl_type_from_data_type(output_data_type));
    build_options.add_option("-DWEI_TENSOR_TYPE=BUFFER");
    build_options.add_option("-DWEI_WIDTH=" + support::cpp11::to_string(weights_width));
    build_options.add_option("-DWEI_HEIGHT=" + support::cpp11::to_string(weights_height));
    build_options.add_option("-DWEI_DATA_TYPE=" + get_cl_type_from_data_type(weights->data_type()));
    build_options.add_option("-DSTRIDE_X=" + support::cpp11::to_string(strides.first));
    build_options.add_option("-DSTRIDE_Y=" + support::cpp11::to_string(strides.second));
    build_options.add_option("-DPAD_LEFT=" + support::cpp11::to_string(pad_left));
    build_options.add_option("-DPAD_TOP=" + support::cpp11::to_string(pad_top));
    build_options.add_option("-DN0=" + support::cpp11::to_string(n0));
    build_options.add_option("-DM0=" + support::cpp11::to_string(m0));
    build_options.add_option("-DK0=" + support::cpp11::to_string(k0));
    build_options.add_option("-DPARTIAL_N0=" + support::cpp11::to_string(partial_store_n0));
    build_options.add_option_if((input_channels % k0) != 0, "-DLEFTOVER_LOOP");

    if (is_data_type_quantized(output_data_type))
    {
        const UniformQuantizationInfo iqinfo = input->quantization_info().uniform();
        const UniformQuantizationInfo wqinfo = weights->quantization_info().uniform();
        const UniformQuantizationInfo oqinfo = output->quantization_info().uniform();

        PixelValue zero_value = PixelValue(0, input->data_type(), input->quantization_info());
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
        build_options.add_option("-DACC_DATA_TYPE=" + get_cl_type_from_data_type(input_data_type));
        build_options.add_option("-DZERO_VALUE=" + support::cpp11::to_string(0));
    }

    if (compile_context.get_ddk_version() >= 30)
    {
        build_options.add_option("-fregister-allocation=64");
    }

    _kernel = create_kernel(compile_context, kernel_name, build_options.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input_data_type));
    _config_id += "_";
    _config_id += support::cpp11::to_string(weights_width);
    _config_id += "_";
    _config_id += support::cpp11::to_string(strides.first);
    _config_id += "_";
    _config_id += support::cpp11::to_string(strides.second);
    _config_id += "_";
    _config_id += support::cpp11::to_string(output_width);
    _config_id += "_";
    _config_id += support::cpp11::to_string(m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(n0);
}

Status ClTransposedConvolutionKernel::validate(const ITensorInfo   *src,
                                               const ITensorInfo   *weights,
                                               const ITensorInfo   *biases,
                                               const ITensorInfo   *dst,
                                               const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, deconv_info));
    return Status{};
}

void ClTransposedConvolutionKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    // Get initial windows
    Window slice = window.first_slice_window_3D();

    const auto src =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto weights =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto biases =
        utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_2));
    auto dst = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    unsigned int idx = 0;
    add_4d_tensor_nhwc_argument(idx, src);
    add_4d_tensor_nhwc_argument(idx, dst);

    add_4d_tensor_nhwc_argument(idx, weights);
    if (biases != nullptr)
    {
        add_1D_tensor_argument(idx, biases, slice);
    }

    enqueue(queue, *this, slice, lws_hint());
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
