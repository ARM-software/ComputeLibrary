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
#include "arm_compute/core/CL/kernels/CLWinogradFilterTransformKernel.h"

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

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->data_layout() != DataLayout::NCHW);

    const Size2D kernel_size      = winograd_info.kernel_size;
    const Size2D output_tile_size = winograd_info.output_tile_size;

    const size_t idx_w = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH);
    const size_t idx_h = get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size != Size2D(3U, 3U) && kernel_size != Size2D(5U, 5U), "Winograd filter transform only supports 3x3 and 5x5 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size == Size2D(3U, 3U) && output_tile_size != Size2D(2U, 2U)
                                    && output_tile_size != Size2D(4U, 4U),
                                    "Winograd filter transform only supports 2x2 or 4x4 output tile for 3x3 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(kernel_size == Size2D(5U, 5U) && output_tile_size != Size2D(4U, 4U), "Winograd filter transform only supports 4x4 output tile for 5x5 kernels");
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(idx_w) != kernel_size.width || input->dimension(idx_h) != kernel_size.height);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() > 4);

    // Checks performed when output is configured
    if(output->total_size() != 0)
    {
        const TensorInfo tensor_info_output = input->clone()->set_tensor_shape(compute_winograd_filter_transform_shape(*input, winograd_info));

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(output, &tensor_info_output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    const unsigned int num_elems_processed_per_iteration_x = input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::WIDTH));
    const unsigned int num_elems_processed_per_iteration_y = input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::HEIGHT));

    Window win            = calculate_max_window(*input, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    bool   window_changed = false;

    AccessWindowRectangle input_access(input, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    AccessWindowStatic    output_access(output, 0, 0, output->dimension(0), output->dimension(1));
    window_changed = update_window_and_padding(win, input_access, output_access);
    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));

    Window win_collapsed = win.collapse(win, Window::DimZ);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win_collapsed);
}
} // namespace

CLWinogradFilterTransformKernel::CLWinogradFilterTransformKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLWinogradFilterTransformKernel::configure(const ICLTensor *input, ICLTensor *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output->info(), input->info()->clone()->set_tensor_shape(compute_winograd_filter_transform_shape(*input->info(), winograd_info)));

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), winograd_info));

    const size_t idx_c = get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);

    // Set build options
    CLBuildOptions build_opts;
    build_opts.add_option("-DNUM_CHANNELS=" + support::cpp11::to_string(input->info()->dimension(idx_c)));

    const Size2D kernel_size      = winograd_info.kernel_size;
    const Size2D output_tile_size = winograd_info.output_tile_size;

    // Create kernel
    std::string kernel_name = "winograd_filter_transform_" + output_tile_size.to_string() + "_" + kernel_size.to_string() + "_nchw";
    _kernel                 = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel(kernel_name, build_opts.options()));

    _input  = input;
    _output = output;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLWinogradFilterTransformKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const WinogradInfo &winograd_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, winograd_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);

    return Status{};
}

void CLWinogradFilterTransformKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    // Setup output window
    Window window_out;
    window_out.use_tensor_dimensions(_output->info()->tensor_shape(), 0);

    unsigned int idx = 0;
    add_4D_tensor_argument(idx, _input, window);
    add_3D_tensor_argument(idx, _output, window_out);
    enqueue(queue, *this, window);
}
