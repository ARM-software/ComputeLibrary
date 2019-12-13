/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLScaleKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/CLValidate.h"
#include "arm_compute/core/CL/ICLKernel.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"

#include <set>
#include <string>

using namespace arm_compute;

namespace
{
inline std::pair<float, float> calculate_scale_factors(const ITensorInfo &input, const ITensorInfo &output)
{
    DataLayout data_layout = input.data_layout();
    const int  idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

    // Compute the ratio between source width/height and destination width/height
    const unsigned int input_width   = input.dimension(idx_width);
    const unsigned int input_height  = input.dimension(idx_height);
    const unsigned int output_width  = output.dimension(idx_width);
    const unsigned int output_height = output.dimension(idx_height);

    float wr = static_cast<float>(input_width) / static_cast<float>(output_width);
    float hr = static_cast<float>(input_height) / static_cast<float>(output_height);

    return std::make_pair(wr, hr);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::U8, DataType::S16, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON(output == input);

    float wr = 0.f;
    float hr = 0.f;
    std::tie(wr, hr) = calculate_scale_factors(*input, *output);

    ARM_COMPUTE_RETURN_ERROR_ON(policy == InterpolationPolicy::AREA && (wr > 1.f || hr > 1.f));

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy, BorderSize &border)
{
    Window       win{};
    bool         window_changed{};
    unsigned int num_elems_processed_per_iteration = 0;
    DataLayout   data_layout                       = input->data_layout();

    switch(data_layout)
    {
        case DataLayout::NCHW:
        {
            if(border_mode == BorderMode::UNDEFINED)
            {
                border = BorderSize(0);
            }

            num_elems_processed_per_iteration = 4;
            // Configure kernel window
            win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
            AccessWindowStatic input_access(input,
                                            -border.left, -border.top,
                                            input->dimension(0) + border.right,
                                            input->dimension(1) + border.bottom);
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

            output_access.set_valid_region(win, calculate_valid_region_scale(*(input),
                                                                             output->tensor_shape(),
                                                                             policy,
                                                                             sampling_policy,
                                                                             border_mode == BorderMode::UNDEFINED));

            window_changed = update_window_and_padding(win, input_access, output_access);
        }
        break;
        case DataLayout::NHWC:
        {
            num_elems_processed_per_iteration = 1;
            // Configure kernel window
            win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));
            AccessWindowStatic input_access(input, -border.left, -border.top,
                                            input->dimension(0) + border.right,
                                            input->dimension(1) + border.bottom);
            AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
            window_changed = update_window_and_padding(win, input_access, output_access);
            output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
        }
        break;
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

BorderSize CLScaleKernel::border_size() const
{
    return BorderSize(1);
}

Status CLScaleKernel::validate(const ITensorInfo *input, const ITensorInfo *output, InterpolationPolicy policy,
                               BorderMode border_mode, SamplingPolicy sampling_policy)
{
    BorderSize border = BorderSize(1);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get(), policy, border_mode, sampling_policy, border).first);

    return Status{};
}

const ICLTensor *CLScaleKernel::input() const
{
    return _input;
}

const ICLTensor *CLScaleKernel::output() const
{
    return _output;
}

void CLScaleKernel::configure(const ICLTensor *input, ICLTensor *output, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), policy));

    _input               = input;
    _output              = output;
    _interpolationPolicy = policy;
    _data_layout         = input->info()->data_layout();

    float wr = 0.f;
    float hr = 0.f;
    std::tie(wr, hr) = calculate_scale_factors(*input->info(), *output->info());

    const bool call_quantized_kernel = is_data_type_quantized_asymmetric(input->info()->data_type()) && policy == InterpolationPolicy::BILINEAR;

    const int  idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
    const int  idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
    const bool is_nhwc    = _data_layout == DataLayout::NHWC;

    // Compute the ratio between source width/height and destination width/height
    const unsigned int input_width   = input->info()->dimension(idx_width);
    const unsigned int input_height  = input->info()->dimension(idx_height);
    const unsigned int output_width  = output->info()->dimension(idx_width);
    const unsigned int output_height = output->info()->dimension(idx_height);

    // Compute actual border size
    BorderSize border = border_size();

    // Area interpolation behaves as Nearest Neighbour in case of up-sampling
    if(policy == InterpolationPolicy::AREA && wr <= 1.f && hr <= 1.f)
    {
        policy = InterpolationPolicy::NEAREST_NEIGHBOR;
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), policy, border_mode, sampling_policy, border);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Create kernel
    CLBuildOptions build_opts;
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(input->info()->data_type()));
    build_opts.add_option("-DBORDER_SIZE=" + support::cpp11::to_string(border.right));
    build_opts.add_option_if(border_mode == BorderMode::REPLICATE, "-DBORDER_MODE_REPLICATE");
    build_opts.add_option_if(is_nhwc, "-DDEPTH_OUT=" + support::cpp11::to_string(output->info()->dimension(2)));
    build_opts.add_option_if_else(sampling_policy == SamplingPolicy::CENTER, "-DSAMPLING_POLICY_CENTER", "-DSAMPLING_POLICY_TOP_LEFT");
    if(call_quantized_kernel)
    {
        const UniformQuantizationInfo qinfo = input->info()->quantization_info().uniform();
        build_opts.add_option("-DSCALE=" + support::cpp11::to_string(qinfo.scale));
        build_opts.add_option("-DOFFSET=" + support::cpp11::to_string(qinfo.offset));
    }

    std::string interpolation_name = string_from_interpolation_policy(policy);
    std::transform(interpolation_name.begin(), interpolation_name.end(), interpolation_name.begin(), ::tolower);
    std::string kernel_name = "scale_" + interpolation_name;
    kernel_name += call_quantized_kernel ? "_quantized_" : "_";
    kernel_name += lower_string(string_from_data_layout(_data_layout));
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    unsigned int idx = is_nhwc ? 2 * num_arguments_per_4D_tensor() : 2 * num_arguments_per_2D_tensor(); //Skip the input and output parameters

    // Set static kernel arguments
    const float scale_x = static_cast<float>(input_width) / output_width;
    const float scale_y = static_cast<float>(input_height) / output_height;

    _kernel.setArg<float>(idx++, input_width);
    _kernel.setArg<float>(idx++, input_height);
    _kernel.setArg<float>(idx++, scale_x);
    _kernel.setArg<float>(idx++, scale_y);

    // Set config_id for enabling LWS tuning
    _config_id = "scale_";
    _config_id += (border_mode == BorderMode::REPLICATE ? "Bord_rep" : "");
    _config_id += (sampling_policy == SamplingPolicy::CENTER ? "center" : "topleft");
    _config_id += (is_nhwc ? "nhwc" : "nchw");
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(3));
}

void CLScaleKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    switch(_data_layout)
    {
        case DataLayout::NCHW:
        {
            Window slice = window.first_slice_window_2D();

            do
            {
                unsigned int idx = 0;
                add_2D_tensor_argument(idx, _input, slice);
                add_2D_tensor_argument(idx, _output, slice);
                enqueue(queue, *this, slice, lws_hint());
            }
            while(window.slide_window_slice_2D(slice));
            break;
        }
        case DataLayout::NHWC:
        {
            Window collapsed = window.collapse(ICLKernel::window(), Window::DimZ);
            Window slice     = collapsed.first_slice_window_4D();

            unsigned int idx = 0;
            add_4D_tensor_argument(idx, _input, slice);
            add_4D_tensor_argument(idx, _output, slice);
            enqueue(queue, *this, slice, lws_hint());
            break;
        }
        default:
            ARM_COMPUTE_ERROR("Data layout not supported");
    }
}
