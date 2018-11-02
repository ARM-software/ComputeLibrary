/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLDequantizationLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

using namespace arm_compute;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *min_max)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output, min_max);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() < 3);

    if(output->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *min_max)
{
    // Output tensor auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, DataType::F32);

    constexpr unsigned int num_elems_processed_per_iteration = 4;

    // Configure window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
    AccessWindowStatic     min_max_access(min_max, 0, 0, 2, min_max->dimension(1));

    // Update window and padding
    bool window_changed = update_window_and_padding(win, input_access, output_access, min_max_access);

    output_access.set_valid_region(win, input->valid_region());

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_tuple(err, win);
}
} // namespace

CLDequantizationLayerKernel::CLDequantizationLayerKernel()
    : _input(nullptr), _output(nullptr), _min_max(nullptr)
{
}

void CLDequantizationLayerKernel::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *min_max)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output, min_max);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), min_max->info()));

    _input   = input;
    _output  = output;
    _min_max = min_max;

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("dequantization_layer"));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), output->info(), min_max->info());

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config));
}

Status CLDequantizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *min_max)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, min_max));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get(), min_max->clone().get())));

    return Status{};
}

void CLDequantizationLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), 3);
    Window slice            = window_collapsed.first_slice_window_3D();

    Window min_max_window = window;
    min_max_window.set(Window::DimX, Window::Dimension(0, 0, 0));
    min_max_window.set(Window::DimY, Window::Dimension(0, _min_max->info()->dimension(1), 1));
    min_max_window.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window min_max_slice = min_max_window.first_slice_window_1D();

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice);
        add_3D_tensor_argument(idx, _output, slice);
        add_1D_tensor_argument(idx, _min_max, min_max_slice);
        enqueue(queue, *this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice) && min_max_window.slide_window_slice_1D(min_max_slice));
}
