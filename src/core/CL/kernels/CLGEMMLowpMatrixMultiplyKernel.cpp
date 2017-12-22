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
#include "arm_compute/core/CL/kernels/CLGEMMLowpMatrixMultiplyKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

CLGEMMLowpMatrixMultiplyKernel::CLGEMMLowpMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr)
{
}

void CLGEMMLowpMatrixMultiplyKernel::configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, bool is_interleaved_transposed)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::QASYMM8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    if(!is_interleaved_transposed)
    {
        ARM_COMPUTE_ERROR_ON(input0->info()->dimension(0) != input1->info()->dimension(1));
    }

    _input0 = input0;
    _input1 = input1;
    _output = output;

    CLBuildOptions build_opts;

    if(is_interleaved_transposed)
    {
        // Create kernel and set static arguments
        build_opts.add_option("-DCOLS_B=" + support::cpp11::to_string(input1->info()->dimension(0)));
        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemmlowp_mm_interleaved_transposed", build_opts.options()));

        // Configure window
        constexpr unsigned int num_elems_processed_per_iteration_x = 16;
        constexpr unsigned int num_elems_processed_per_iteration_y = 4;
        constexpr unsigned int num_elems_read_per_iteration_input0 = 4;
        constexpr unsigned int num_elems_read_per_iteration_input1 = 16;

        Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowRectangle input0_access(input0->info(), 0, 0, num_elems_read_per_iteration_input0, 1);
        AccessWindowRectangle input1_access(input1->info(), 0, 0, num_elems_read_per_iteration_input1, 1);
        AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        update_window_and_padding(win, input0_access, input1_access, output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

        ICLKernel::configure(win);
    }
    else
    {
        // Special case for 1xN, 2xN, 3xN and 4xN input0 tensor. num_elems_processed_per_iteration_x
        constexpr unsigned int num_elems_processed_per_iteration_x = 16;
        const unsigned int     num_elems_processed_per_iteration_y = std::min(static_cast<int>(output->info()->dimension(1)), 4);

        build_opts.add_option("-DCOLS_A=" + support::cpp11::to_string(input0->info()->dimension(0)));
        build_opts.add_option("-DNUM_ELEMS_PROCESSED_PER_THREAD_X=" + support::cpp11::to_string(num_elems_processed_per_iteration_x));
        build_opts.add_option("-DNUM_ELEMS_PROCESSED_PER_THREAD_Y=" + support::cpp11::to_string(num_elems_processed_per_iteration_y));

        _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemmlowp_mm", build_opts.options()));

        // Configure window
        Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        AccessWindowStatic    input0_access(input0->info(), 0, 0, input0->info()->dimension(0), ceil_to_multiple(input0->info()->dimension(1), num_elems_processed_per_iteration_y));
        AccessWindowStatic    input1_access(input1->info(), 0, 0, ceil_to_multiple(input1->info()->dimension(0), num_elems_processed_per_iteration_x), input1->info()->dimension(1));
        AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        update_window_and_padding(win, input0_access, input1_access, output_access);

        Coordinates coord;
        coord.set_num_dimensions(output->info()->num_dimensions());
        output_access.set_valid_region(win, ValidRegion(coord, output->info()->tensor_shape()));

        ICLKernel::configure(win);
    }

    // Set config_id for enabling LWS tuning
    _config_id = "gemmlowp_";
    _config_id += (is_interleaved_transposed ? "reshaped_" : "");
    _config_id += lower_string(string_from_data_type(input0->info()->data_type()));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(output->info()->dimension(0));
    _config_id += "_";
    _config_id += (is_interleaved_transposed ? support::cpp11::to_string(input1->info()->dimension(0)) : support::cpp11::to_string(input1->info()->dimension(1)));
}

void CLGEMMLowpMatrixMultiplyKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window slice          = window.first_slice_window_2D();
    Window slice_matrix_b = slice;
    slice_matrix_b.set(Window::DimX, Window::Dimension(0, _input1->info()->dimension(0), 1));
    slice_matrix_b.set(Window::DimY, Window::Dimension(0, _input1->info()->dimension(1), 1));
    slice_matrix_b.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        Window slice_b = slice;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        if(_input1->info()->num_dimensions() < 3)
        {
            slice_b = slice_matrix_b;
        }

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, _input0, slice);
        add_2D_tensor_argument(idx, _input1, slice_b);
        add_2D_tensor_argument(idx, _output, slice);
        enqueue(queue, *this, slice, _lws_hint);
    }
    while(window.slide_window_slice_2D(slice));
}
