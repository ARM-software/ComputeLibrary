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
#include "arm_compute/core/NEON/kernels/NEScharr3x3Kernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace
{
const int16x8_t three       = vdupq_n_s16(3);
const int16x8_t minus_three = vdupq_n_s16(-3);
const int16x8_t ten         = vdupq_n_s16(10);
const int16x8_t minus_ten   = vdupq_n_s16(-10);

inline int16x8_t scharr_y(const int16x8x2_t &top, const int16x8x2_t &bottom)
{
    // Top left
    int16x8_t out = vmulq_s16(top.val[0], minus_three);
    // Top center
    out = vmlaq_s16(out, vextq_s16(top.val[0], top.val[1], 1), minus_ten);
    // Top right
    out = vmlaq_s16(out, vextq_s16(top.val[0], top.val[1], 2), minus_three);

    // Bottom left
    out = vmlaq_s16(out, bottom.val[0], three);
    // Bottom center
    out = vmlaq_s16(out, vextq_s16(bottom.val[0], bottom.val[1], 1), ten);
    // Bottom right
    out = vmlaq_s16(out, vextq_s16(bottom.val[0], bottom.val[1], 2), three);

    return out;
}

inline int16x8_t scharr_x(const int16x8x2_t &top, const int16x8x2_t &middle, const int16x8x2_t &bottom)
{
    // Top left
    int16x8_t out = vmulq_s16(top.val[0], minus_three);
    // Top right
    out = vmlaq_s16(out, vextq_s16(top.val[0], top.val[1], 2), three);

    // Middle left
    out = vmlaq_s16(out, middle.val[0], minus_ten);
    // Middle right
    out = vmlaq_s16(out, vextq_s16(middle.val[0], middle.val[1], 2), ten);

    // Bottom left
    out = vmlaq_s16(out, bottom.val[0], minus_three);
    // Bottom right
    out = vmlaq_s16(out, vextq_s16(bottom.val[0], bottom.val[1], 2), three);

    return out;
}
} // namespace

NEScharr3x3Kernel::NEScharr3x3Kernel()
    : _run_scharr_x(false), _run_scharr_y(false), _input(nullptr), _output_x(nullptr), _output_y(nullptr)
{
}

void NEScharr3x3Kernel::configure(const ITensor *input, ITensor *output_x, ITensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _run_scharr_x = output_x != nullptr;
    _run_scharr_y = output_y != nullptr;

    if(_run_scharr_x)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_x, 1, DataType::S16);
    }

    if(_run_scharr_y)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_y, 1, DataType::S16);
    }

    _input    = input;
    _output_x = output_x;
    _output_y = output_y;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_x_access(output_x == nullptr ? nullptr : output_x->info(), 0, num_elems_written_per_iteration);
    AccessWindowHorizontal output_y_access(output_y == nullptr ? nullptr : output_y->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), -border_size().left, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_x_access,
                              output_y_access);

    output_x_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
    output_y_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

BorderSize NEScharr3x3Kernel::border_size() const
{
    return BorderSize(1);
}

void NEScharr3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const unsigned char *const input_top_ptr = _input->ptr_to_element(Coordinates(-1, -1));
    const unsigned char *const input_mid_ptr = _input->ptr_to_element(Coordinates(-1, 0));
    const unsigned char *const input_bot_ptr = _input->ptr_to_element(Coordinates(-1, +1));

    Iterator input(_input, window);
    Iterator output_y;
    Iterator output_x;

    if(_run_scharr_y)
    {
        output_y = Iterator(_output_y, window);
    }

    if(_run_scharr_x)
    {
        output_x = Iterator(_output_x, window);
    }

    if(_run_scharr_x && _run_scharr_y)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {

            const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
            const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
            const uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

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

            vst1q_s16(reinterpret_cast<int16_t *>(output_x.ptr()), scharr_x(top_s16, mid_s16, bot_s16));
            vst1q_s16(reinterpret_cast<int16_t *>(output_y.ptr()), scharr_y(top_s16, bot_s16));
        },
        input, output_x, output_y);
    }
    else if(_run_scharr_x)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {

            const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
            const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
            const uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

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

            vst1q_s16(reinterpret_cast<int16_t *>(output_x.ptr()), scharr_x(top_s16, mid_s16, bot_s16));
        },
        input, output_x);
    }
    else if(_run_scharr_y)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {

            const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
            const uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

            const int16x8x2_t top_s16 =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(top_data))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(top_data)))
                }
            };
            const int16x8x2_t bot_s16 =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(bot_data))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(bot_data)))
                }
            };

            vst1q_s16(reinterpret_cast<int16_t *>(output_y.ptr()), scharr_y(top_s16, bot_s16));
        },
        input, output_y);
    }
}
