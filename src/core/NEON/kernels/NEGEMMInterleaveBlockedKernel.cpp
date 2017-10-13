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
#include "arm_compute/core/NEON/kernels/NEGEMMInterleaveBlockedKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

namespace
{
inline void gemm_interleave_blocked_transposed_8bit(const ITensor *input, ITensor *output, const Window &window, unsigned int block_width, unsigned int block_height)
{
    const size_t in_stride = input->info()->strides_in_bytes()[1];

    const unsigned int in_height = input->info()->dimension(1);
    const unsigned int in_width  = input->info()->dimension(0);

    const float scale_y_factor = 1.f / float(block_height);

    // Set window for output tensor
    Window win_out(window);
    win_out.scale(Window::DimY, scale_y_factor);
    Iterator in(input, window);

    win_out.set_dimension_step(Window::DimX, block_width * block_height);
    Iterator out(output, win_out);

    execute_window_loop(window, [&](const Coordinates &)
    {
        std::fill_n(out.ptr(), block_width * block_height, 0);
    },
    out);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        for(unsigned int z = id.y(); (z < in_width) && z < (id.y() + block_height); ++z)
        {
            int j = (z - id.y()) * block_width;
            for(unsigned int b = id.x(); (b < in_height) && (b < (id.x() + block_width)); ++b)
            {
                *(out.ptr() + j++) = *(input->buffer() + b * in_stride + z);
            }
        }
    },
    in, out);
}

inline void gemm_interleave_blocked_8bit(const ITensor *input, ITensor *output, const Window &window, unsigned int block_width, unsigned int block_height)
{
    const size_t in_stride = input->info()->strides_in_bytes()[1];

    const unsigned int in_height = input->info()->dimension(1);
    const unsigned int in_width  = input->info()->dimension(0);

    const float scale_y_factor = 1.f / float(block_height);

    // Set window for output tensor
    Window win_out(window);
    win_out.scale(Window::DimY, scale_y_factor);
    Iterator in(input, window);

    win_out.set_dimension_step(Window::DimX, block_width * block_height);
    Iterator out(output, win_out);

    execute_window_loop(window, [&](const Coordinates &)
    {
        std::fill_n(out.ptr(), block_width * block_height, 0);
    },
    out);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        for(unsigned int z = id.y(); (z < in_height) && z < (id.y() + block_height); ++z)
        {
            int j = (z - id.y()) * block_width;
            for(unsigned int b = id.x(); (b < in_width) && (b < (id.x() + block_width)); ++b)
            {
                *(out.ptr() + j++) = *(input->buffer() + z * in_stride + b);
            }
        }
    },
    in, out);
}
} // namespace

NEGEMMInterleaveBlockedKernel::NEGEMMInterleaveBlockedKernel()
    : _block_height(0), _block_width(0), _transpose(false)
{
}

void NEGEMMInterleaveBlockedKernel::configure(const ITensor *input, ITensor *output, unsigned int block_height, unsigned int block_width, bool transpose)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    ARM_COMPUTE_ERROR_ON_MSG(block_height < 1, "Block height must be greater than 0");
    ARM_COMPUTE_ERROR_ON_MSG(block_width < 1, "Block window must be greater than 0");

    TensorShape output_shape      = input->info()->tensor_shape();
    const float interleave_by_f32 = block_height;
    output_shape.set(0, input->info()->dimension(0) * interleave_by_f32);
    output_shape.set(1, std::ceil(static_cast<float>(input->info()->dimension(1)) / interleave_by_f32));
    // Output auto inizialitation if not yet initialized
    auto_init_if_empty(*output->info(), output_shape, 1, input->info()->data_type(), input->info()->fixed_point_position());
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(output->info()->tensor_shape(), output_shape);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_FIXED_POINT(input, output);

    _input        = input;
    _output       = output;
    _block_height = block_height;
    _block_width  = block_width;
    _transpose    = transpose;

    const unsigned int num_elems_processed_per_iteration_x = block_width;
    const unsigned int num_elems_processed_per_iteration_y = block_height;

    // Configure kernel window
    Window      win           = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    const float scaley_factor = 1.f / interleave_by_f32;

    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x * num_elems_processed_per_iteration_y, 1, num_elems_processed_per_iteration_y, scaley_factor);
    AccessWindowRectangle input_access(input->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    update_window_and_padding(win, output_access, input_access);

    output_access.set_valid_region(win, input->info()->valid_region());

    INEKernel::configure(win);
}

void NEGEMMInterleaveBlockedKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    if(_transpose)
    {
        gemm_interleave_blocked_transposed_8bit(_input, _output, window, _block_width, _block_height);
    }
    else
    {
        gemm_interleave_blocked_8bit(_input, _output, window, _block_width, _block_height);
    }
}
