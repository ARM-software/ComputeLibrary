/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLGEMMLowpReductionKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
Status validate_arguments_matrix_a_reduction(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    return Status{};
}
std::pair<Status, Window> validate_and_configure_window_matrix_a_reduction(ITensorInfo *input, ITensorInfo *output)
{
    const unsigned int num_elems_processed_per_iteration = 1;

    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input, 0, 0, ceil_to_multiple(input->dimension(0), 16), input->dimension(1));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}

Status validate_arguments_matrix_b_reduction(const ITensorInfo *input, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window_matrix_b_reduction(ITensorInfo *input, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration));

    AccessWindowStatic     input_access(input, 0, 0, ceil_to_multiple(input->dimension(0), num_elems_processed_per_iteration), input->dimension(1));
    AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);

    bool window_changed = update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->tensor_shape()));

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

ICLGEMMLowpReductionKernel::ICLGEMMLowpReductionKernel()
    : _input(), _output()
{
}

void CLGEMMLowpMatrixAReductionKernel::configure(const ICLTensor *mtx_a, ICLTensor *vector_sum_row)
{
    // Perform validate step
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_a, vector_sum_row);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_a_reduction(mtx_a->info(), vector_sum_row->info()));

    _input  = mtx_a;
    _output = vector_sum_row;

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DCOLS_A=" + support::cpp11::to_string(mtx_a->info()->dimension(0)));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemmlowp_matrix_a_reduction", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window_matrix_a_reduction(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLGEMMLowpMatrixAReductionKernel::validate(const ITensorInfo *mtx_a, const ITensorInfo *vector_sum_row)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_a_reduction(mtx_a, vector_sum_row));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_matrix_a_reduction(mtx_a->clone().get(), vector_sum_row->clone().get()).first);

    return Status{};
}

void CLGEMMLowpMatrixAReductionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimY);
    Window slice_in  = collapsed.first_slice_window_2D();
    Window slice_out = collapsed.first_slice_window_2D();

    // Setup input slice. Its dimensions are increased in the cl kernel.
    slice_in.set(Window::DimX, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimY, Window::Dimension(0, 0, 0));
    slice_in.set(Window::DimZ, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out);
    }
    while(collapsed.slide_window_slice_2D(slice_out));
}

void CLGEMMLowpMatrixBReductionKernel::configure(const ICLTensor *mtx_b, ICLTensor *vector_sum_col)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(mtx_b, vector_sum_col);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_matrix_b_reduction(mtx_b->info(), vector_sum_col->info()));

    _input  = mtx_b;
    _output = vector_sum_col;

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;
    build_opts.add_option("-DCOLS_B=" + support::cpp11::to_string(mtx_b->info()->dimension(0)));
    build_opts.add_option("-DROWS_B=" + support::cpp11::to_string(mtx_b->info()->dimension(1)));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemmlowp_matrix_b_reduction", build_opts.options()));

    // Configure kernel window
    auto win_config = validate_and_configure_window_matrix_b_reduction(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure(win_config.second);
}

Status CLGEMMLowpMatrixBReductionKernel::validate(const ITensorInfo *mtx_b, const ITensorInfo *vector_sum_col)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_matrix_b_reduction(mtx_b, vector_sum_col));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window_matrix_b_reduction(mtx_b->clone().get(), vector_sum_col->clone().get()).first);

    return Status{};
}

void CLGEMMLowpMatrixBReductionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(IKernel::window(), Window::DimY);

    Window slice_out = collapsed.first_slice_window_2D();
    Window slice_in  = slice_out;

    slice_in.set(Window::DimY, Window::Dimension(0, 1, 1));
    slice_in.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _input, slice_in);
        add_2D_tensor_argument(idx, _output, slice_out);
        enqueue(queue, *this, slice_out);
    }
    while(collapsed.slide_window_slice_2D(slice_out));
}
