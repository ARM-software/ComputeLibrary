/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDeconvolutionReshapeOutputKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const ITensorInfo *input_info, const ITensorInfo *weights_info,
                          const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, input_info, weights_info);
    const DataLayout   data_layout = input_info->data_layout();
    const unsigned int stride_x    = deconv_info.stride().first;
    const unsigned int stride_y    = deconv_info.stride().second;

    const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t idx_b = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    const bool is_qasymm = is_data_type_quantized_asymmetric(input_info->data_type());

    ARM_COMPUTE_RETURN_ERROR_ON(weights_info->dimension(idx_w) != deconv_info.stride().first);
    ARM_COMPUTE_RETURN_ERROR_ON(weights_info->dimension(idx_h) != deconv_info.stride().second);

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32, DataType::F16, DataType::QASYMM8, DataType::S32);
    if(!is_qasymm)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, input_info, weights_info);
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != weights_info->dimension(idx_w) * weights_info->dimension(idx_h) * weights_info->dimension(idx_b));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != input_info->dimension(idx_w));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(2) != input_info->dimension(idx_h));
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(3) != input_info->dimension(idx_b));

    if(bias != nullptr)
    {
        if(is_qasymm)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(bias, input);
        }
        ARM_COMPUTE_RETURN_ERROR_ON(bias->dimension(0) != weights_info->dimension(idx_b));
    }

    if(output->total_size() != 0)
    {
        auto out_dims = deconvolution_output_dimensions(input_info->dimension(idx_w), input_info->dimension(idx_h), weights_info->dimension(idx_w), weights_info->dimension(idx_h),
                                                        0, 0, stride_x, stride_y);

        const TensorShape output_shape = misc::shape_calculator::compute_deconvolution_output_shape(out_dims, *input_info, *weights_info);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *input, ITensorInfo *output, const ITensorInfo *input_info, const ITensorInfo *weights_info, const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    const DataLayout data_layout = input_info->data_layout();

    const unsigned int stride_x = deconv_info.stride().first;
    const unsigned int stride_y = deconv_info.stride().second;
    const size_t       idx_w    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t       idx_h    = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    auto out_dims = deconvolution_output_dimensions(input_info->dimension(idx_w), input_info->dimension(idx_h), weights_info->dimension(idx_w), weights_info->dimension(idx_h),
                                                    0, 0, stride_x, stride_y);

    const TensorShape output_shape = misc::shape_calculator::compute_deconvolution_output_shape(out_dims, *input_info, *weights_info);

    auto_init_if_empty(*output, input->clone()->set_tensor_shape(output_shape).set_data_layout(data_layout).set_quantization_info(input->quantization_info()));

    Window win = calculate_max_window(*input);

    return std::make_pair(Status{}, win);
}
} // namespace

CLDeconvolutionReshapeOutputKernel::CLDeconvolutionReshapeOutputKernel()
    : _add_bias(false),
      _bias(nullptr)
{
}

void CLDeconvolutionReshapeOutputKernel::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const ITensorInfo *input_info, const ITensorInfo *weights_info,
                                                   const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, input_info, weights_info);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias != nullptr ? bias->info() : nullptr), output->info(), input_info, weights_info, deconv_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), input_info, weights_info, deconv_info);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);

    const DataLayout data_layout = input_info->data_layout();
    const size_t     idx_w       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const size_t     idx_h       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const size_t     idx_b       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);

    _input    = input;
    _output   = output;
    _add_bias = (bias != nullptr);
    _bias     = bias;

    const int filter_w = weights_info->dimension(idx_w);
    const int filter_h = weights_info->dimension(idx_h);
    const int filter_b = weights_info->dimension(idx_b);
    const int img_w    = input_info->dimension(idx_w);
    const int img_h    = input_info->dimension(idx_h);

    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DFILTER_WIDTH=" + support::cpp11::to_string(filter_w));
    build_opts.add_option("-DFILTER_HEIGHT=" + support::cpp11::to_string(filter_h));
    build_opts.add_option("-DSRC_WIDTH=" + support::cpp11::to_string(img_w));
    build_opts.add_option("-DSRC_HEIGHT=" + support::cpp11::to_string(img_h));
    build_opts.add_option_if(data_layout == DataLayout::NCHW, "-DNUM_FILTERS=" + support::cpp11::to_string(filter_b));
    build_opts.add_option_if(_add_bias, "-DADD_BIAS");

    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("deconvolution_reshape", build_opts.options()));
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = "deconvolution_reshape_output_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += lower_string(string_from_data_layout(input->info()->data_layout()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLDeconvolutionReshapeOutputKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const ITensorInfo *input_info, const ITensorInfo *weights_info,
                                                    const PadStrideInfo &deconv_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, bias, output, input_info, weights_info, deconv_info));
    return Status{};
}

void CLDeconvolutionReshapeOutputKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);
    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);

    unsigned int idx = 0;
    add_3D_tensor_argument(idx, _input, collapsed);
    add_3D_tensor_argument(idx, _output, collapsed);
    if(_add_bias)
    {
        add_1D_tensor_argument(idx, _bias, collapsed);
    }
    enqueue(queue, *this, collapsed, lws_hint());
}
} // namespace arm_compute
