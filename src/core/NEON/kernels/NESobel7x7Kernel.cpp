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
#include "arm_compute/core/NEON/kernels/NESobel7x7Kernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

namespace
{
const int32x4_t minusfour = vdupq_n_s32(-4);
const int32x4_t minusfive = vdupq_n_s32(-5);
const int32x4_t four      = vdupq_n_s32(4);
const int32x4_t five      = vdupq_n_s32(5);
const int32x4_t six       = vdupq_n_s32(6);
const int32x4_t fifteen   = vdupq_n_s32(15);
const int32x4_t twenty    = vdupq_n_s32(20);

inline int32x4x2_t compute_hor_sobel_x(const int32x4x4_t &data)
{
    int32x4x2_t out =
    {
        {
            vnegq_s32(data.val[0]),
            vnegq_s32(data.val[1])
        }
    };

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[0], data.val[1], 1), minusfour);

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[0], data.val[1], 2), minusfive);

    out.val[0] = vmlaq_s32(out.val[0], data.val[1], five);

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[1], data.val[2], 1), four);

    out.val[0] = vaddq_s32(out.val[0],
                           vextq_s32(data.val[1], data.val[2], 2));

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[1], data.val[2], 1), minusfour);

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[1], data.val[2], 2), minusfive);

    out.val[1] = vmlaq_s32(out.val[1], data.val[2], five);

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[2], data.val[3], 1), four);

    out.val[1] = vaddq_s32(out.val[1],
                           vextq_s32(data.val[2], data.val[3], 2));

    return out;
}

inline int32x4x2_t compute_hor_sobel_y(const int32x4x4_t &data)
{
    int32x4x2_t out =
    {
        {
            data.val[0],
            data.val[1]
        }
    };

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[0], data.val[1], 1), six);

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[0], data.val[1], 2), fifteen);

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[0], data.val[1], 3), twenty);

    out.val[0] = vmlaq_s32(out.val[0], data.val[1], fifteen);

    out.val[0] = vmlaq_s32(out.val[0],
                           vextq_s32(data.val[1], data.val[2], 1), six);

    out.val[0] = vaddq_s32(out.val[0],
                           vextq_s32(data.val[1], data.val[2], 2));

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[1], data.val[2], 1), six);

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[1], data.val[2], 2), fifteen);

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[1], data.val[2], 3), twenty);

    out.val[1] = vmlaq_s32(out.val[1], data.val[2], fifteen);

    out.val[1] = vmlaq_s32(out.val[1],
                           vextq_s32(data.val[2], data.val[3], 1), six);

    out.val[1] = vaddq_s32(out.val[1],
                           vextq_s32(data.val[2], data.val[3], 2));

    return out;
}
} // namespace

NESobel7x7HorKernel::NESobel7x7HorKernel()
    : _input(nullptr), _output_x(nullptr), _output_y(nullptr), _run_sobel_x(false), _run_sobel_y(false), _border_size(0)
{
}

BorderSize NESobel7x7HorKernel::border_size() const
{
    return _border_size;
}

void NESobel7x7HorKernel::configure(const ITensor *input, ITensor *output_x, ITensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input, Format::U8);
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _run_sobel_x = output_x != nullptr;
    _run_sobel_y = output_y != nullptr;

    if(_run_sobel_x)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output_x, Format::S32);
    }

    if(_run_sobel_y)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output_y, Format::S32);
    }

    _input       = input;
    _output_x    = output_x;
    _output_y    = output_y;
    _border_size = BorderSize(border_undefined ? 0 : 3, 3);

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

void NESobel7x7HorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Iterator input(_input, window);
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
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const uint8x16_t data = vld1q_u8(input.ptr() - 3);

            const uint16x8_t tmp_low_u16  = vmovl_u8(vget_low_u8(data));
            const uint16x8_t tmp_high_u16 = vmovl_u8(vget_high_u8(data));

            const int32x4x4_t data_s32 =
            {
                {
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tmp_low_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tmp_low_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tmp_high_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tmp_high_u16)))
                }
            };

            const int32x4x2_t out_y = compute_hor_sobel_y(data_s32);
            vst1q_s32(reinterpret_cast<int32_t *>(output_y.ptr()), out_y.val[0]);
            vst1q_s32(reinterpret_cast<int32_t *>(output_y.ptr()) + 4, out_y.val[1]);

            const int32x4x2_t out_x = compute_hor_sobel_x(data_s32);
            vst1q_s32(reinterpret_cast<int32_t *>(output_x.ptr()), out_x.val[0]);
            vst1q_s32(reinterpret_cast<int32_t *>(output_x.ptr()) + 4, out_x.val[1]);
        },
        input, output_x, output_y);
    }
    else if(_run_sobel_x)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const uint8x16_t data = vld1q_u8(input.ptr() - 3);

            const uint16x8_t tmp_low_u16  = vmovl_u8(vget_low_u8(data));
            const uint16x8_t tmp_high_u16 = vmovl_u8(vget_high_u8(data));

            const int32x4x4_t data_s32 =
            {
                {
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tmp_low_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tmp_low_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tmp_high_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tmp_high_u16)))
                }
            };

            const int32x4x2_t out = compute_hor_sobel_x(data_s32);
            vst1q_s32(reinterpret_cast<int32_t *>(output_x.ptr()), out.val[0]);
            vst1q_s32(reinterpret_cast<int32_t *>(output_x.ptr()) + 4, out.val[1]);
        },
        input, output_x);
    }
    else if(_run_sobel_y)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            const uint8x16_t data = vld1q_u8(input.ptr() - 3);

            const uint16x8_t tmp_low_u16  = vmovl_u8(vget_low_u8(data));
            const uint16x8_t tmp_high_u16 = vmovl_u8(vget_high_u8(data));

            const int32x4x4_t data_s32 =
            {
                {
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tmp_low_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tmp_low_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(tmp_high_u16))),
                    vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(tmp_high_u16)))
                }
            };

            const int32x4x2_t out = compute_hor_sobel_y(data_s32);
            vst1q_s32(reinterpret_cast<int32_t *>(output_y.ptr()), out.val[0]);
            vst1q_s32(reinterpret_cast<int32_t *>(output_y.ptr()) + 4, out.val[1]);
        },
        input, output_y);
    }
}

NESobel7x7VertKernel::NESobel7x7VertKernel()
    : _input_x(nullptr), _input_y(nullptr), _output_x(nullptr), _output_y(nullptr), _run_sobel_x(false), _run_sobel_y(false)
{
}

BorderSize NESobel7x7VertKernel::border_size() const
{
    return BorderSize(3, 0);
}

void NESobel7x7VertKernel::configure(const ITensor *input_x, const ITensor *input_y, ITensor *output_x, ITensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    _run_sobel_x = (output_x != nullptr);
    _run_sobel_y = (output_y != nullptr);

    if(_run_sobel_x)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input_x, Format::S32);
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output_x, Format::S32);
    }

    if(_run_sobel_y)
    {
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(input_y, Format::S32);
        ARM_COMPUTE_ERROR_ON_FORMAT_NOT_IN(output_y, Format::S32);
    }

    _input_x  = input_x;
    _input_y  = input_y;
    _output_x = output_x;
    _output_y = output_y;

    const ITensor *const input = _run_sobel_x ? input_x : input_y;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 8;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_rows_read_per_iteration       = 7;

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

void NESobel7x7VertKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    Iterator input_x;
    Iterator input_y;
    Iterator output_x;
    Iterator output_y;

    int32_t in_x_stride = 0;
    int32_t in_y_stride = 0;

    if(_run_sobel_x)
    {
        input_x     = Iterator(_input_x, window);
        output_x    = Iterator(_output_x, window);
        in_x_stride = _input_x->info()->strides_in_bytes()[1] / pixel_size_from_format(_input_x->info()->format());
    }

    if(_run_sobel_y)
    {
        input_y     = Iterator(_input_y, window);
        output_y    = Iterator(_output_y, window);
        in_y_stride = _input_y->info()->strides_in_bytes()[1] / pixel_size_from_format(_input_y->info()->format());
    }

    if(_run_sobel_x)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            auto in_ptr = reinterpret_cast<int32_t *>(input_x.ptr()) - 3 * in_x_stride;

            //top3
            int32x4x2_t data =
            {
                {
                    vld1q_s32(in_ptr),
                    vld1q_s32(in_ptr + 4)
                }
            };

            int32x4x2_t out = data;

            //top2
            in_ptr += in_x_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], six);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], six);

            //top
            in_ptr += in_x_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], fifteen);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], fifteen);

            //mid
            in_ptr += in_x_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], twenty);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], twenty);

            //low
            in_ptr += in_x_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], fifteen);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], fifteen);

            //low2
            in_ptr += in_x_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], six);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], six);

            //low3
            in_ptr += in_x_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vaddq_s32(out.val[0], data.val[0]);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vaddq_s32(out.val[1], data.val[1]);

            vst1q_s32(reinterpret_cast<int32_t *>(output_x.ptr()) + 0, out.val[0]);
            vst1q_s32(reinterpret_cast<int32_t *>(output_x.ptr()) + 4, out.val[1]);
        },
        input_x, output_x);
    }

    if(_run_sobel_y)
    {
        execute_window_loop(window, [&](const Coordinates & id)
        {
            auto in_ptr = reinterpret_cast<int32_t *>(input_y.ptr()) - 3 * in_y_stride;

            //top3
            int32x4x2_t data =
            {
                {
                    vld1q_s32(in_ptr),
                    vld1q_s32(in_ptr + 4)
                }
            };

            int32x4x2_t out =
            {
                {
                    vnegq_s32(data.val[0]),
                    vnegq_s32(data.val[1])
                }
            };

            //top2
            in_ptr += in_y_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], minusfour);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], minusfour);

            //top
            in_ptr += in_y_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], minusfive);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], minusfive);

            //low
            in_ptr += (2 * in_y_stride);
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], five);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], five);

            //low2
            in_ptr += in_y_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vmlaq_s32(out.val[0], data.val[0], four);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vmlaq_s32(out.val[1], data.val[1], four);

            //low3
            in_ptr += in_y_stride;
            data.val[0] = vld1q_s32(in_ptr);
            out.val[0]  = vaddq_s32(out.val[0], data.val[0]);

            data.val[1] = vld1q_s32(in_ptr + 4);
            out.val[1]  = vaddq_s32(out.val[1], data.val[1]);

            vst1q_s32(reinterpret_cast<int32_t *>(output_y.ptr()) + 0, out.val[0]);
            vst1q_s32(reinterpret_cast<int32_t *>(output_y.ptr()) + 4, out.val[1]);
        },
        input_y, output_y);
    }
}
