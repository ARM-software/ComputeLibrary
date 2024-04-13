/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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
#include "src/core/CL/kernels/CLChannelShuffleLayerKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/helpers/AdjustVecSize.h"
#include "arm_compute/core/utils/StringUtils.h"

#include "src/core/CL/CLValidate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/StringSupport.h"

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ERROR_ON_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_type() == DataType::UNKNOWN);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_groups < 2, "Channel shuffling with less than 2 groups would be inefficient");

    const unsigned int channels =
        input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL));

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        num_groups == channels,
        "Channel shuffling with same number of groups as number of channels would be inefficient");
    // There cannot be more groups than channels
    ARM_COMPUTE_RETURN_ERROR_ON(num_groups > channels);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((channels % num_groups) != 0,
                                    "The number of channels must be a multiple of the number of groups");

    // Checks performed when output is configured
    if (output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, *input->clone());

    const bool is_nhwc = input->data_layout() == DataLayout::NHWC;
    if (is_nhwc)
    {
        unsigned int num_elems_processed_per_iteration_x =
            adjust_vec_size(max_cl_vector_width / input->element_size(), input->dimension(0));
        Window win           = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x));
        Window win_collapsed = win.collapse(win, Window::DimZ);
        return std::make_pair(Status{}, win_collapsed);
    }
    else
    {
        const unsigned int     num_elems_processed_per_iteration_x = max_cl_vector_width / input->element_size();
        constexpr unsigned int num_elems_processed_per_iteration_y = 2;

        // Configure kernel window
        Window win = calculate_max_window(
            *input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
        AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration_x,
                                           num_elems_processed_per_iteration_y);
        AccessWindowRectangle output_access(output, 0, 0, num_elems_processed_per_iteration_x,
                                            num_elems_processed_per_iteration_y);

        const bool window_changed = update_window_and_padding(win, input_access, output_access);

        Window win_collapsed = win.collapse(win, Window::DimZ);

        Status err =
            (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
        return std::make_pair(err, win_collapsed);
    }
}
} // namespace

CLChannelShuffleLayerKernel::CLChannelShuffleLayerKernel() : _input(nullptr), _output(nullptr)
{
    _type = CLKernelType::ELEMENTWISE;
}

void CLChannelShuffleLayerKernel::configure(const ICLTensor *input, ICLTensor *output, unsigned int num_groups)
{
    configure(CLKernelLibrary::get().get_compile_context(), input, output, num_groups);
}

void CLChannelShuffleLayerKernel::configure(const CLCompileContext &compile_context,
                                            const ICLTensor        *input,
                                            ICLTensor              *output,
                                            unsigned int            num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), num_groups));
    auto padding_info = get_padding_info({input, output});

    _input  = input;
    _output = output;

    const DataLayout   data_layout = input->info()->data_layout();
    const bool         is_nhwc     = data_layout == DataLayout::NHWC;
    const unsigned int channels =
        input->info()->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL));
    unsigned int vec_size_x           = 0;
    unsigned int vec_size_x_leftovers = 0;
    if (is_nhwc)
    {
        vec_size_x = adjust_vec_size(max_cl_vector_width / input->info()->element_size(), input->info()->dimension(0));
        vec_size_x_leftovers = input->info()->dimension(0) % vec_size_x;
    }
    else
    {
        vec_size_x = max_cl_vector_width / input->info()->element_size();
    }

    // Set kernel build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DNUM_GROUPS=" + support::cpp11::to_string(num_groups));
    build_opts.add_option("-DK=" + support::cpp11::to_string(channels / num_groups));
    build_opts.add_option("-DVEC_SIZE=" + support::cpp11::to_string(vec_size_x));
    build_opts.add_option_if(is_nhwc, "-DVEC_SIZE_LEFTOVER=" + support::cpp11::to_string(vec_size_x_leftovers));
    build_opts.add_option_if(is_nhwc, "-DSRC_DIM_X=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.add_option("-DSRC_DIM_Z=" + support::cpp11::to_string(input->info()->dimension(2)));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_unsigned_type_from_element_size(input->info()->element_size()));

    // Create kernel
    std::string kernel_name = "channel_shuffle_" + lower_string(string_from_data_layout(data_layout));

    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(num_groups);
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
    if (data_layout == DataLayout::NHWC)
    {
        ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
    }
}

Status
CLChannelShuffleLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, unsigned int num_groups)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, num_groups));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void CLChannelShuffleLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, window);
    add_4D_tensor_argument(idx, _output, window);
    enqueue(queue, *this, window, lws_hint());
}
} // namespace arm_compute
