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
#include "arm_compute/core/NEON/kernels/NEDilateKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

BorderSize NEDilateKernel::border_size() const
{
    return BorderSize(1);
}

void NEDilateKernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);
    AccessWindowRectangle  input_access(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration);

    update_window_and_padding(win, input_access, output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEDilateKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator in(_input, window);
    Iterator out(_output, window);

    const size_t in_stride = _input->info()->strides_in_bytes()[1];

    execute_window_loop(window, [&](const Coordinates &)
    {
        uint8_t         *in_ptr   = in.ptr() - 1;
        const uint8x16_t top_data = vld1q_u8(in_ptr - in_stride);
        const uint8x16_t mid_data = vld1q_u8(in_ptr);
        const uint8x16_t bot_data = vld1q_u8(in_ptr + in_stride);

        uint8x8_t top_high_data = vget_high_u8(top_data);
        uint8x8_t top_low_data  = vget_low_u8(top_data);

        uint8x8_t mid_high_data = vget_high_u8(mid_data);
        uint8x8_t mid_low_data  = vget_low_u8(mid_data);

        uint8x8_t bot_high_data = vget_high_u8(bot_data);
        uint8x8_t bot_low_data  = vget_low_u8(bot_data);

        uint8x8_t p0;
        uint8x8_t p1;

        p0 = top_low_data;
        p1 = vext_u8(top_low_data, top_high_data, 1);
        p0 = vmax_u8(p0, p1);

        p1 = vext_u8(top_low_data, top_high_data, 2);
        p0 = vmax_u8(p0, p1);

        p1 = mid_low_data;
        p0 = vmax_u8(p0, p1);

        p1 = vext_u8(mid_low_data, mid_high_data, 1);
        p0 = vmax_u8(p0, p1);

        p1 = vext_u8(mid_low_data, mid_high_data, 2);
        p0 = vmax_u8(p0, p1);

        p1 = bot_low_data;
        p0 = vmax_u8(p0, p1);

        p1 = vext_u8(bot_low_data, bot_high_data, 1);
        p0 = vmax_u8(p0, p1);

        p1 = vext_u8(bot_low_data, bot_high_data, 2);
        p0 = vmax_u8(p0, p1);

        vst1_u8(out.ptr(), p0);
    },
    in, out);
}
