/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEMedian3x3Kernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <utility>

using namespace arm_compute;

namespace
{
inline void sort(uint8x8_t &a, uint8x8_t &b)
{
    const uint8x8_t min = vmin_u8(a, b);
    const uint8x8_t max = vmax_u8(a, b);
    a                   = min;
    b                   = max;
}
} // namespace

BorderSize NEMedian3x3Kernel::border_size() const
{
    return BorderSize(1);
}

void NEMedian3x3Kernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    _input  = input;
    _output = output;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEMedian3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    const unsigned char *input_bot_ptr = _input->ptr_to_element(Coordinates(-1, -1));
    const unsigned char *input_mid_ptr = _input->ptr_to_element(Coordinates(-1, 0));
    const unsigned char *input_top_ptr = _input->ptr_to_element(Coordinates(-1, +1));

    Iterator input(_input, window);
    Iterator output(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

        uint8x8_t p0 = vget_low_u8(top_data);
        uint8x8_t p1 = vext_u8(vget_low_u8(top_data), vget_high_u8(top_data), 1);
        uint8x8_t p2 = vext_u8(vget_low_u8(top_data), vget_high_u8(top_data), 2);
        uint8x8_t p3 = vget_low_u8(mid_data);
        uint8x8_t p4 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 1);
        uint8x8_t p5 = vext_u8(vget_low_u8(mid_data), vget_high_u8(mid_data), 2);
        uint8x8_t p6 = vget_low_u8(bot_data);
        uint8x8_t p7 = vext_u8(vget_low_u8(bot_data), vget_high_u8(bot_data), 1);
        uint8x8_t p8 = vext_u8(vget_low_u8(bot_data), vget_high_u8(bot_data), 2);

        sort(p1, p2);
        sort(p4, p5);
        sort(p7, p8);

        sort(p0, p1);
        sort(p3, p4);
        sort(p6, p7);

        sort(p1, p2);
        sort(p4, p5);
        sort(p7, p8);

        sort(p0, p3);
        sort(p5, p8);
        sort(p4, p7);

        sort(p3, p6);
        sort(p1, p4);
        sort(p2, p5);

        sort(p4, p7);
        sort(p4, p2);
        sort(p6, p4);

        sort(p4, p2);

        vst1_u8(output.ptr(), p4);
    },
    input, output);
}
