/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLMinMaxLayerKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <climits>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON(input->num_dimensions() < 3);

    if(output->tensor_shape().total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

        TensorShape output_shape = compute_min_max_shape(input);

        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output->tensor_shape(), output_shape);
    }

    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    TensorShape output_shape = compute_min_max_shape(input);

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, output_shape, 1, input->data_type());

    const unsigned int num_elems_processed_per_iteration = 1;

    // Configure kernel window
    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    AccessWindowStatic     output_access(output, 0, 0, 2, output->dimension(1));

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_tuple(err, win);
}
} // namespace

CLMinMaxLayerKernel::CLMinMaxLayerKernel()
    : _input(nullptr), _output(nullptr)
{
}

void CLMinMaxLayerKernel::configure(const ICLTensor *input, ICLTensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info()));

    _input  = input;
    _output = output;

    std::set<std::string> build_opts;
    build_opts.emplace("-DWIDTH=" + support::cpp11::to_string(input->info()->dimension(0)));
    build_opts.emplace("-DHEIGHT=" + support::cpp11::to_string(input->info()->dimension(1)));
    build_opts.emplace("-DDEPTH=" + support::cpp11::to_string(input->info()->dimension(2)));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("minmax_layer", build_opts));

    auto win_config = validate_and_configure_window(input->info(), output->info());

    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    ICLKernel::configure_internal(std::get<1>(win_config));
}

Status CLMinMaxLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));

    return Status{};
}

void CLMinMaxLayerKernel::reset(cl::CommandQueue &queue)
{
    _output->map(queue, true);

    Window window_output;
    window_output.use_tensor_dimensions(_output->info()->tensor_shape());
    window_output.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator output(_output, window_output);

    // Reset output
    execute_window_loop(window_output, [&](const Coordinates &)
    {
        auto *ptr = reinterpret_cast<float *>(output.ptr());
        ptr[0]    = std::numeric_limits<float>::max();
        ptr[1]    = std::numeric_limits<float>::min();
    },
    output);

    _output->unmap(queue);
}

void CLMinMaxLayerKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    Window window_collapsed = window.collapse_if_possible(ICLKernel::window(), 3);
    Window slice            = window_collapsed.first_slice_window_3D();
    slice.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice.set(Window::DimY, Window::Dimension(0, 1, 1));
    slice.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        Window output_slice = slice.shift_dimensions(2);

        unsigned int idx = 0;
        // Set inputs
        add_3D_tensor_argument(idx, _input, slice);
        add_1D_tensor_argument(idx, _output, output_slice);
        enqueue(queue, *this, slice);
    }
    while(window_collapsed.slide_window_slice_3D(slice));
}
