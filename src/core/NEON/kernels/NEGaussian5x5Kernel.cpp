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
#include "arm_compute/core/NEON/kernels/NEGaussian5x5Kernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

NEGaussian5x5HorKernel::NEGaussian5x5HorKernel()
    : _border_size(0)
{
}

BorderSize NEGaussian5x5HorKernel::border_size() const
{
    return _border_size;
}

void NEGaussian5x5HorKernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S16);

    _input       = input;
    _output      = output;
    _border_size = BorderSize(border_undefined ? 0 : 2, 2);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;

    Window                 win = calculate_max_window_horizontal(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), -border_size().left, num_elems_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEGaussian5x5HorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window win_in(window);
    win_in.shift(Window::DimX, -2);

    Iterator input(_input, win_in);
    Iterator output(_output, window);

    static const int16x8_t six  = vdupq_n_s16(6);
    static const int16x8_t four = vdupq_n_s16(4);

    execute_window_loop(window, [&](const Coordinates &)
    {
        uint8x16_t data = vld1q_u8(input.ptr());

        const int16x8x2_t data_s16 =
        {
            {
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
            }
        };

        int16x8_t out = vaddq_s16(data_s16.val[0], vextq_s16(data_s16.val[0], data_s16.val[1], 4));
        out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 1), four);
        out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 2), six);
        out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 3), four);

        vst1q_s16(reinterpret_cast<int16_t *>(output.ptr()), out);
    },
    input, output);
}

BorderSize NEGaussian5x5VertKernel::border_size() const
{
    return BorderSize{ 2, 0 };
}

void NEGaussian5x5VertKernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    _input  = input;
    _output = output;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_elems_read_per_iteration      = 32;
    constexpr unsigned int num_elems_written_per_iteration   = 16;
    constexpr unsigned int num_rows_read_per_iteration       = 5;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEGaussian5x5VertKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator output(_output, window);

    const uint8_t *input_top2_ptr = _input->ptr_to_element(Coordinates(0, -2));
    const uint8_t *input_top_ptr  = _input->ptr_to_element(Coordinates(0, -1));
    const uint8_t *input_mid_ptr  = _input->ptr_to_element(Coordinates(0, 0));
    const uint8_t *input_low_ptr  = _input->ptr_to_element(Coordinates(0, 1));
    const uint8_t *input_low2_ptr = _input->ptr_to_element(Coordinates(0, 2));

    const uint16x8_t six  = vdupq_n_u16(6);
    const uint16x8_t four = vdupq_n_u16(4);

    execute_window_loop(window, [&](const Coordinates &)
    {
        const size_t input_offset_high_s16 = input.offset();
        const size_t input_offset_low_s16  = input.offset() + 16;

        //HIGH DATA
        //top2
        uint16x8_t data_high = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top2_ptr + input_offset_high_s16)));
        uint16x8_t out_high  = data_high;
        //top
        data_high = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top_ptr + input_offset_high_s16)));
        out_high  = vmlaq_u16(out_high, data_high, four);
        //mid
        data_high = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_mid_ptr + input_offset_high_s16)));
        out_high  = vmlaq_u16(out_high, data_high, six);
        //low
        data_high = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low_ptr + input_offset_high_s16)));
        out_high  = vmlaq_u16(out_high, data_high, four);
        //low2
        data_high = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low2_ptr + input_offset_high_s16)));
        out_high  = vaddq_u16(out_high, data_high);

        //LOW DATA
        //top2
        uint16x8_t data_low = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top2_ptr + input_offset_low_s16)));
        uint16x8_t out_low  = data_low;
        //top
        data_low = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top_ptr + input_offset_low_s16)));
        out_low  = vmlaq_u16(out_low, data_low, four);
        //mid
        data_low = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_mid_ptr + input_offset_low_s16)));
        out_low  = vmlaq_u16(out_low, data_low, six);
        //low
        data_low = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low_ptr + input_offset_low_s16)));
        out_low  = vmlaq_u16(out_low, data_low, four);
        //low2
        data_low = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low2_ptr + input_offset_low_s16)));
        out_low  = vaddq_u16(out_low, data_low);

        vst1q_u8(output.ptr(), vcombine_u8(vqshrn_n_u16(out_high, 8),
                                           vqshrn_n_u16(out_low, 8)));
    },
    input, output);
}
