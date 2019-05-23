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
#include "arm_compute/core/NEON/kernels/NEGaussian3x3Kernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>

using namespace arm_compute;

BorderSize NEGaussian3x3Kernel::border_size() const
{
    return BorderSize(1);
}

void NEGaussian3x3Kernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

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

void NEGaussian3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator output(_output, window);

    const uint8_t *input_bot_ptr = _input->ptr_to_element(Coordinates(-1, -1));
    const uint8_t *input_mid_ptr = _input->ptr_to_element(Coordinates(-1, 0));
    const uint8_t *input_top_ptr = _input->ptr_to_element(Coordinates(-1, +1));

    static const int16x8_t two  = vdupq_n_s16(2);
    static const int16x8_t four = vdupq_n_s16(4);

    execute_window_loop(window, [&](const Coordinates &)
    {
        uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
        uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
        uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

        const int16x8x2_t top_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(top_data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(top_data)))
            }
        };
        const int16x8x2_t mid_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mid_data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mid_data)))
            }
        };
        const int16x8x2_t bot_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bot_data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bot_data)))
            }
        };

        //top left
        int16x8_t out = top_s16.val[0];
        //top mid
        out = vmlaq_s16(out, vextq_s16(top_s16.val[0], top_s16.val[1], 1), two);
        //top right
        out = vaddq_s16(out, vextq_s16(top_s16.val[0], top_s16.val[1], 2));
        //mid left
        out = vmlaq_s16(out, mid_s16.val[0], two);
        //mid mid
        out = vmlaq_s16(out, vextq_s16(mid_s16.val[0], mid_s16.val[1], 1), four);
        //mid right
        out = vmlaq_s16(out, vextq_s16(mid_s16.val[0], mid_s16.val[1], 2), two);
        //bot left
        out = vaddq_s16(out, bot_s16.val[0]);
        //bot mid
        out = vmlaq_s16(out, vextq_s16(bot_s16.val[0], bot_s16.val[1], 1), two);
        //bot right
        out = vaddq_s16(out, vextq_s16(bot_s16.val[0], bot_s16.val[1], 2));

        vst1_u8(output.ptr(), vqshrun_n_s16(out, 4));
    },
    input, output);
}
