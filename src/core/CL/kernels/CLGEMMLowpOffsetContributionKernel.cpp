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
#include "arm_compute/core/CL/kernels/CLGEMMLowpOffsetContributionKernel.h"

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

CLGEMMLowpOffsetContributionKernel::CLGEMMLowpOffsetContributionKernel()
    : _vector_sum_col(nullptr), _vector_sum_row(nullptr), _mm_result(nullptr)
{
}

void CLGEMMLowpOffsetContributionKernel::configure(ICLTensor *mm_result, const ICLTensor *vector_sum_col, const ICLTensor *vector_sum_row, int32_t k, int32_t a_offset, int32_t b_offset)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(mm_result, 1, DataType::S32);

    // Set the arguments to pass at compile time
    CLBuildOptions build_opts;

    // If a_offset == 0, vector_sum_col can be a nullptr
    if(a_offset != 0)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
        ARM_COMPUTE_ERROR_ON(vector_sum_col->info()->dimension(0) != mm_result->info()->dimension(0));

        build_opts.add_option("-DA_OFFSET=" + support::cpp11::to_string(a_offset));
        build_opts.add_option_if(vector_sum_col->info()->tensor_shape().num_dimensions() > 1, "-DSUM_COL_HAS_BATCHES");
    }

    // If b_offset == 0, vector_sum_row can be a nullptr
    if(b_offset != 0)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);
        ARM_COMPUTE_ERROR_ON(vector_sum_row->info()->dimension(0) != mm_result->info()->dimension(1));

        // Validate batches
        TensorShape output_shape = mm_result->info()->tensor_shape();
        if(output_shape.num_dimensions() > 1)
        {
            TensorShape vector_sum_row_shape = vector_sum_row->info()->tensor_shape();
            vector_sum_row_shape.collapse_from(1);
            output_shape.collapse_from(2);

            ARM_COMPUTE_ERROR_ON_MSG(vector_sum_row_shape[1] != output_shape[2], "mm_result tensor must have the same number of batches of output tensor");

            if(a_offset != 0)
            {
                TensorShape vector_sum_col_shape = vector_sum_col->info()->tensor_shape();
                vector_sum_col_shape.collapse_from(1);

                ARM_COMPUTE_ERROR_ON_MSG(vector_sum_col_shape[1] != 1
                                         && vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                         "vector_sum_col tensor must have the same number of batches of vector_sum_row_shape or the number of batches must be set to 1");
            }
        }

        build_opts.add_option("-DB_OFFSET=" + support::cpp11::to_string(b_offset));
    }

    build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(a_offset * b_offset * k));

    // Create kernel
    _kernel = static_cast<cl::Kernel>(CLKernelLibrary::get().create_kernel("gemmlowp_offset_contribution", build_opts.options()));

    _vector_sum_col = vector_sum_col;
    _vector_sum_row = vector_sum_row;
    _mm_result      = mm_result;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*mm_result->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal mm_result_access(mm_result->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win, mm_result_access);

    if(a_offset != 0)
    {
        AccessWindowHorizontal vector_sum_col_access(vector_sum_col->info(), 0, num_elems_processed_per_iteration);
        update_window_and_padding(win, vector_sum_col_access);
    }

    if(b_offset != 0)
    {
        AccessWindowStatic vector_sum_row_access(vector_sum_row->info(), 0, 0, vector_sum_row->info()->dimension(0), 0);
        update_window_and_padding(win, vector_sum_row_access);
    }

    ICLKernel::configure(win);
}

void CLGEMMLowpOffsetContributionKernel::run(const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    Window collapsed = window.collapse_if_possible(ICLKernel::window(), Window::DimZ);
    Window slice     = collapsed.first_slice_window_3D();

    // Set window for vector_sum_col
    Window win_vector_sum_col = slice;
    win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));

    // Set window for vector_sum_row
    Window win_vector_sum_row = slice;
    win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));

    do
    {
        unsigned int idx = 0;
        add_3D_tensor_argument(idx, _mm_result, slice);
        if(_vector_sum_col != nullptr)
        {
            add_2D_tensor_argument(idx, _vector_sum_col, win_vector_sum_col);
        }
        if(_vector_sum_row != nullptr)
        {
            add_2D_tensor_argument(idx, _vector_sum_row, win_vector_sum_row);
        }
        enqueue(queue, *this, slice);
    }
    while(collapsed.slide_window_slice_3D(slice));
}
