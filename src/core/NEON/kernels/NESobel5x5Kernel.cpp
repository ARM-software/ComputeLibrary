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
#include "arm_compute/core/NEON/kernels/NESobel5x5Kernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

NESobel5x5HorKernel::NESobel5x5HorKernel()
    : _input(nullptr), _output_x(nullptr), _output_y(nullptr), _run_sobel_x(false), _run_sobel_y(false), _border_size(0)
{
}

BorderSize NESobel5x5HorKernel::border_size() const
{
    return _border_size;
}

void NESobel5x5HorKernel::configure(const ITensor *input, ITensor *output_x, ITensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _run_sobel_x = output_x != nullptr;
    _run_sobel_y = output_y != nullptr;

    if(_run_sobel_x)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_x, 1, DataType::S16);
    }

    if(_run_sobel_y)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_y, 1, DataType::S16);
    }

    _input       = input;
    _output_x    = output_x;
    _output_y    = output_y;
    _border_size = BorderSize(border_undefined ? 0 : 2, 2);

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;

    Window                 win = calculate_max_window_horizontal(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_x_access(output_x == nullptr ? nullptr : output_x->info(), 0, num_elems_written_per_iteration);
    AccessWindowHorizontal output_y_access(output_y == nullptr ? nullptr : output_y->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), -border_size().left, num_elems_read_per_iteration),
                              output_x_access,
                              output_y_access);

    output_x_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
    output_y_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NESobel5x5HorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Window win_in(window);
    win_in.shift(Window::DimX, -2);

    Iterator input(_input, win_in);
    Iterator output_x;
    Iterator output_y;

    if(_run_sobel_x)
    {
        output_x = Iterator(_output_x, window);
    }

    if(_run_sobel_y)
    {
        output_y = Iterator(_output_y, window);
    }

    if(_run_sobel_y && _run_sobel_x)
    {
        static const int16x8_t six      = vdupq_n_s16(6);
        static const int16x8_t four     = vdupq_n_s16(4);
        static const int16x8_t two      = vdupq_n_s16(2);
        static const int16x8_t minustwo = vdupq_n_s16(-2);

        execute_window_loop(window, [&](const Coordinates &)
        {
            const uint8x16_t data = vld1q_u8(input.ptr());

            const int16x8x2_t data_s16 =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
                }
            };

            int16x8_t out_y = data_s16.val[0];
            out_y           = vmlaq_s16(out_y, vextq_s16(data_s16.val[0], data_s16.val[1], 1), four);
            out_y           = vmlaq_s16(out_y, vextq_s16(data_s16.val[0], data_s16.val[1], 2), six);
            out_y           = vmlaq_s16(out_y, vextq_s16(data_s16.val[0], data_s16.val[1], 3), four);
            out_y           = vaddq_s16(out_y, vextq_s16(data_s16.val[0], data_s16.val[1], 4));

            vst1q_s16(reinterpret_cast<int16_t *>(output_y.ptr()), out_y);

            int16x8_t out_x = vnegq_s16(data_s16.val[0]);
            out_x           = vmlaq_s16(out_x, vextq_s16(data_s16.val[0], data_s16.val[1], 1), minustwo);
            out_x           = vmlaq_s16(out_x, vextq_s16(data_s16.val[0], data_s16.val[1], 3), two);
            out_x           = vaddq_s16(out_x, vextq_s16(data_s16.val[0], data_s16.val[1], 4));

            vst1q_s16(reinterpret_cast<int16_t *>(output_x.ptr()), out_x);
        },
        input, output_x, output_y);
    }
    else if(_run_sobel_x)
    {
        static const int16x8_t two      = vdupq_n_s16(2);
        static const int16x8_t minustwo = vdupq_n_s16(-2);

        execute_window_loop(window, [&](const Coordinates &)
        {
            const uint8x16_t data = vld1q_u8(input.ptr());

            const int16x8x2_t data_s16 =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
                }
            };

            int16x8_t out = vnegq_s16(data_s16.val[0]);
            out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 1), minustwo);
            out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 3), two);
            out           = vaddq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 4));

            vst1q_s16(reinterpret_cast<int16_t *>(output_x.ptr()), out);
        },
        input, output_x);
    }
    else if(_run_sobel_y)
    {
        static const int16x8_t six  = vdupq_n_s16(6);
        static const int16x8_t four = vdupq_n_s16(4);

        execute_window_loop(window, [&](const Coordinates &)
        {
            const uint8x16_t data = vld1q_u8(input.ptr());

            const int16x8x2_t data_s16 =
            {
                {
                    vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data))),
                    vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(data)))
                }
            };

            int16x8_t out = data_s16.val[0];
            out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 1), four);
            out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 2), six);
            out           = vmlaq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 3), four);
            out           = vaddq_s16(out, vextq_s16(data_s16.val[0], data_s16.val[1], 4));

            vst1q_s16(reinterpret_cast<int16_t *>(output_y.ptr()), out);
        },
        input, output_y);
    }
}

NESobel5x5VertKernel::NESobel5x5VertKernel()
    : _input_x(nullptr), _input_y(nullptr), _output_x(nullptr), _output_y(nullptr), _run_sobel_x(false), _run_sobel_y(false)
{
}

BorderSize NESobel5x5VertKernel::border_size() const
{
    return BorderSize{ 2, 0 };
}

void NESobel5x5VertKernel::configure(ITensor *input_x, ITensor *input_y, ITensor *output_x, ITensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _run_sobel_x = output_x != nullptr;
    _run_sobel_y = output_y != nullptr;

    if(_run_sobel_x)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input_x, Format::S16);
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output_x, Format::S16);
    }

    if(_run_sobel_y)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input_y, Format::S16);
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output_y, Format::S16);
    }

    _input_x  = input_x;
    _input_y  = input_y;
    _output_x = output_x;
    _output_y = output_y;

    const ITensor *const input = _run_sobel_x ? input_x : input_y;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 16;
    constexpr unsigned int num_rows_read_per_iteration       = 5;

    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_x_access(output_x == nullptr ? nullptr : output_x->info(), 0, num_elems_written_per_iteration);
    AccessWindowHorizontal output_y_access(output_y == nullptr ? nullptr : output_y->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win,
                              AccessWindowRectangle(input_x == nullptr ? nullptr : input_x->info(), 0, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              AccessWindowRectangle(input_y == nullptr ? nullptr : input_y->info(), 0, -border_size().top, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_x_access,
                              output_y_access);

    output_x_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
    output_y_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NESobel5x5VertKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Iterator input_x;
    Iterator input_y;
    Iterator output_x;
    Iterator output_y;

    const int16_t *input_x_low2_ptr = nullptr;
    const int16_t *input_x_low_ptr  = nullptr;
    const int16_t *input_x_mid_ptr  = nullptr;
    const int16_t *input_x_top_ptr  = nullptr;
    const int16_t *input_x_top2_ptr = nullptr;

    const int16_t *input_y_low2_ptr = nullptr;
    const int16_t *input_y_low_ptr  = nullptr;
    const int16_t *input_y_top_ptr  = nullptr;
    const int16_t *input_y_top2_ptr = nullptr;

    if(_run_sobel_x)
    {
        input_x          = Iterator(_input_x, window);
        output_x         = Iterator(_output_x, window);
        input_x_top2_ptr = reinterpret_cast<const int16_t *>(_input_x->ptr_to_element(Coordinates(0, -2)));
        input_x_top_ptr  = reinterpret_cast<const int16_t *>(_input_x->ptr_to_element(Coordinates(0, -1)));
        input_x_mid_ptr  = reinterpret_cast<const int16_t *>(_input_x->ptr_to_element(Coordinates(0, 0)));
        input_x_low_ptr  = reinterpret_cast<const int16_t *>(_input_x->ptr_to_element(Coordinates(0, 1)));
        input_x_low2_ptr = reinterpret_cast<const int16_t *>(_input_x->ptr_to_element(Coordinates(0, 2)));
    }

    if(_run_sobel_y)
    {
        input_y          = Iterator(_input_y, window);
        output_y         = Iterator(_output_y, window);
        input_y_top2_ptr = reinterpret_cast<const int16_t *>(_input_y->ptr_to_element(Coordinates(0, -2)));
        input_y_top_ptr  = reinterpret_cast<const int16_t *>(_input_y->ptr_to_element(Coordinates(0, -1)));
        input_y_low_ptr  = reinterpret_cast<const int16_t *>(_input_y->ptr_to_element(Coordinates(0, 1)));
        input_y_low2_ptr = reinterpret_cast<const int16_t *>(_input_y->ptr_to_element(Coordinates(0, 2)));
    }

    static const int16x8_t six      = vdupq_n_s16(6);
    static const int16x8_t four     = vdupq_n_s16(4);
    static const int16x8_t two      = vdupq_n_s16(2);
    static const int16x8_t minustwo = vdupq_n_s16(-2);

    if(_run_sobel_x)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {
            // Convert offset from uint8_t* to uint16_t*
            const size_t input_offset_high_s16 = input_x.offset() / 2;
            const size_t input_offset_low_s16  = input_offset_high_s16 + 8;

            //HIGH DATA
            //top2
            int16x8_t data_high = vld1q_s16(input_x_top2_ptr + input_offset_high_s16);
            int16x8_t out_high  = data_high;
            //top
            data_high = vld1q_s16(input_x_top_ptr + input_offset_high_s16);
            out_high  = vmlaq_s16(out_high, data_high, four);
            //mid
            data_high = vld1q_s16(input_x_mid_ptr + input_offset_high_s16);
            out_high  = vmlaq_s16(out_high, data_high, six);
            //low
            data_high = vld1q_s16(input_x_low_ptr + input_offset_high_s16);
            out_high  = vmlaq_s16(out_high, data_high, four);
            //low2
            data_high = vld1q_s16(input_x_low2_ptr + input_offset_high_s16);
            out_high  = vaddq_s16(out_high, data_high);

            vst1q_s16((reinterpret_cast<int16_t *>(output_x.ptr())), out_high);

            //LOW DATA
            //top2
            int16x8_t data_low = vld1q_s16(input_x_top2_ptr + input_offset_low_s16);
            int16x8_t out_low  = data_low;
            //top
            data_low = vld1q_s16(input_x_top_ptr + input_offset_low_s16);
            out_low  = vmlaq_s16(out_low, data_low, four);
            //mid
            data_low = vld1q_s16(input_x_mid_ptr + input_offset_low_s16);
            out_low  = vmlaq_s16(out_low, data_low, six);
            //low
            data_low = vld1q_s16(input_x_low_ptr + input_offset_low_s16);
            out_low  = vmlaq_s16(out_low, data_low, four);
            //low2
            data_low = vld1q_s16(input_x_low2_ptr + input_offset_low_s16);
            out_low  = vaddq_s16(out_low, data_low);

            vst1q_s16((reinterpret_cast<int16_t *>(output_x.ptr())) + 8, out_low);
        },
        input_x, output_x);
    }

    if(_run_sobel_y)
    {
        execute_window_loop(window, [&](const Coordinates &)
        {
            // Convert offset from uint8_t* to uint16_t*
            const size_t input_offset_high_s16 = input_y.offset() / 2;
            const size_t input_offset_low_s16  = input_offset_high_s16 + 8;

            //HIGH DATA
            //top2
            int16x8_t data_high = vld1q_s16(input_y_top2_ptr + input_offset_high_s16);
            int16x8_t out_high  = vnegq_s16(data_high);
            //top
            data_high = vld1q_s16(input_y_top_ptr + input_offset_high_s16);
            out_high  = vmlaq_s16(out_high, data_high, minustwo);
            //low
            data_high = vld1q_s16(input_y_low_ptr + input_offset_high_s16);
            out_high  = vmlaq_s16(out_high, data_high, two);
            //low2
            data_high = vld1q_s16(input_y_low2_ptr + input_offset_high_s16);
            out_high  = vaddq_s16(out_high, data_high);

            vst1q_s16((reinterpret_cast<int16_t *>(output_y.ptr())), out_high);

            //LOW DATA
            //top2
            int16x8_t data_low = vld1q_s16(input_y_top2_ptr + input_offset_low_s16);
            int16x8_t out_low  = vnegq_s16(data_low);
            //top
            data_low = vld1q_s16(input_y_top_ptr + input_offset_low_s16);
            out_low  = vmlaq_s16(out_low, data_low, minustwo);
            //low
            data_low = vld1q_s16(input_y_low_ptr + input_offset_low_s16);
            out_low  = vmlaq_s16(out_low, data_low, two);
            //low2
            data_low = vld1q_s16(input_y_low2_ptr + input_offset_low_s16);
            out_low  = vaddq_s16(out_low, data_low);

            vst1q_s16((reinterpret_cast<int16_t *>(output_y.ptr())) + 8, out_low);
        },
        input_y, output_y);
    }
}
