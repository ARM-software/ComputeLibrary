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
#include "arm_compute/core/CL/kernels/CLWinogradOutputTransformKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "support/ToolchainSupport.h"

#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(winograd_info.output_data_layout != DataLayout::NCHW);

    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        input_dimensions = winograd_info.input_dimensions;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size != Size2D(3U, 3U) && kernel_size != Size2D(5U, 5U), "Only 3x3 and 5x5 kernels are supported");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size == Size2D(3U, 3U) && output_tile_size == Size2D(2U, 2U) && input->dimension(2) != 16, "Wrong number of batches");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size == Size2D(5U, 5U) && output_tile_size == Size2D(4U, 4U) && input->dimension(2) != 64, "Wrong number of batches");

    // Compute number of elements to process in the X and Y direction
    const int num_elements_x = input_dimensions.width - (kernel_size.width - 1) + conv_info.pad_left() + conv_info.pad_right();
    const int num_elements_y = input_dimensions.height - (kernel_size.height - 1) + conv_info.pad_top() + conv_info.pad_bottom();
    const int num_tiles_x    = std::ceil(num_elements_x / static_cast<float>(output_tile_size.width));
    const int num_tiles_y    = std::ceil(num_elements_y / static_cast<float>(output_tile_size.height));

    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(1) != static_cast<unsigned int>((num_tiles_x * num_tiles_y)));
    ARM_COMPUTE_UNUSED(output_tile_size);

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
        ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != bias->dimension(0));
    }

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(compute_winograd_output_transform_shape(*input, winograd_info));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *bias, ITensorInfo *output, const Size2D &output_tile_size)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    constexpr unsigned int num_elems_processed_per_iteration = 1;

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    bool   window_changed = false;

    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration, num_elems_processed_per_iteration);
    AccessWindowStatic    output_access(output, 0, 0, ceil_to_multiple(output->dimension(0), output_tile_size.width), ceil_to_multiple(output->dimension(1), output_tile_size.height));

    if(bias != nullptr)
    {
        AccessWindowStatic bias_access(bias, 0, 0, bias->dimension(0), bias->dimension(1));
        window_changed = update_window_and_padding(win, input_access, bias_access, output_access);
    }
    else
    {
        window_changed = update_window_and_padding(win, input_access, output_access);
    }
    output->set_valid_region(ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

CLWinogradOutputTransformKernel::CLWinogradOutputTransformKernel()
    : _input(nullptr), _bias(nullptr), _output(nullptr)
{
}

void CLWinogradOutputTransformKernel::configure(const ICLTensor *input, const ICLTensor *bias, ICLTensor *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_winograd_output_transform_shape(*input->info(), winograd_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (bias != nullptr ? bias->info() : nullptr), output->info(), winograd_info));

    _input  = input;
    _bias   = bias;
    _output = output;

    // Compute num_tiles_x
    const Size2D        input_dimensions = winograd_info.input_dimensions;
    const Size2D        kernel_size      = winograd_info.kernel_size;
    const Size2D        output_tile_size = winograd_info.output_tile_size;
    const PadStrideInfo conv_info        = winograd_info.convolution_info;
    const int           num_elements_x   = input_dimensions.width - (kernel_size.width - 1) + conv_info.pad_left() + conv_info.pad_right();
    const int           num_tiles_x      = std::ceil(num_elements_x / static_cast<float>(output_tile_size.width));

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(_bias != nullptr, std::string("-DHAS_BIAS"));
    build_opts.add_option("-DNUM_TILES_X=" + support::cpp11::to_string(num_tiles_x));

    // Create kernel
    std::string kernel_name = "winograd_output_transform_" + output_tile_size.to_string() + "_" + kernel_size.to_string() + "_nchw";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (bias != nullptr ? bias->info() : nullptr), output->info(), winograd_info.output_tile_size);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += lower_string(string_from_data_type(input->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(input->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
}

Status CLWinogradOutputTransformKernel::validate(const ITensorInfo *input, const ITensorInfo *bias, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, (bias != nullptr ? bias->clone().get() : nullptr), output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (bias != nullptr ? bias->clone().get() : nullptr), output->clone().get(), winograd_info.output_tile_size).first);

    return Status{};
}

void CLWinogradOutputTransformKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Get initial windows
    Window slice = window.first_slice_window_3D();
    slice.set(Window::DimZ, Window::Dimension(0, 1, 1));

    // Setup output slice
    Window slice_out(slice);
    slice_out.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_out.set(Window::DimY, Window::Dimension(0, 0, 0));

    if(_bias != nullptr)
    {
        unsigned int idx1 = 2 * num_arguments_per_3D_tensor();
        Window       slice_biases;
        slice_biases.use_tensor_dimensions(_bias->info()->tensor_shape());
        add_1D_tensor_argument(idx1, _bias, slice_biases);
    }

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_3D(slice) && window.slide_window_slice_3D(slice_out));
}