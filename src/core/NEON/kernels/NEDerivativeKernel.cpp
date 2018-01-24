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
#include "arm_compute/core/NEON/kernels/NEDerivativeKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEDerivativeKernel::NEDerivativeKernel()
    : _func(nullptr), _input(nullptr), _output_x(nullptr), _output_y(nullptr)
{
}

BorderSize NEDerivativeKernel::border_size() const
{
    return BorderSize(1);
}

void NEDerivativeKernel::configure(const ITensor *input, ITensor *output_x, ITensor *output_y, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON((output_x == nullptr) && (output_y == nullptr));

    const bool run_der_x = output_x != nullptr;
    const bool run_der_y = output_y != nullptr;

    if(run_der_x)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_x, 1, DataType::S16);
    }

    if(run_der_y)
    {
        ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_y, 1, DataType::S16);
    }

    _input    = input;
    _output_x = output_x;
    _output_y = output_y;

    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_rows_read_per_iteration       = 3;

    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());

    AccessWindowHorizontal out_x_access(output_x == nullptr ? nullptr : output_x->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal out_y_access(output_y == nullptr ? nullptr : output_y->info(), 0, num_elems_processed_per_iteration);

    AccessWindowHorizontal in_x_access(input->info(), -border_size().left, num_elems_processed_per_iteration + 2);
    AccessWindowRectangle  in_y_access(input->info(), 0, -border_size().left, num_elems_processed_per_iteration, num_rows_read_per_iteration);

    AccessWindowRectangle in_xy_access(input->info(), -border_size().left, -border_size().top, num_elems_processed_per_iteration + 2, num_rows_read_per_iteration);

    if(run_der_x && run_der_y)
    {
        _func = &NEDerivativeKernel::derivative_xy;
        update_window_and_padding(win, in_xy_access, out_x_access, out_y_access);
        out_y_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
        out_x_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
    }
    else
    {
        if(run_der_x)
        {
            _func = &NEDerivativeKernel::derivative_x;
            update_window_and_padding(win, in_x_access, out_x_access);
            out_x_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
        }
        else if(run_der_y)
        {
            _func = &NEDerivativeKernel::derivative_y;
            update_window_and_padding(win, in_y_access, out_y_access);
            out_y_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());
        }
        else
        {
            ARM_COMPUTE_ERROR("At least one output must be NOT NULL");
        }
    }

    INEKernel::configure(win);
}

void NEDerivativeKernel::derivative_x(const Window &window)
{
    Iterator in(_input, window);
    Iterator out_x(_output_x, window);

    /* Apply 1-D centered point discrete derivative mask ([-1 0 1]) along the X direction */
    execute_window_loop(window, [&](const Coordinates & id)
    {
        /* Load left and right data */
        const uint8x16_t l_data = vld1q_u8(in.ptr() - 1);
        const uint8x16_t r_data = vld1q_u8(in.ptr() + 1);

        /* Cast to int16 and perform the subtraction between the right and left data */
        const int16x8_t out0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(l_data))));

        /* Cast to int16 and perform the subtraction between the right and left data */
        const int16x8_t out1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(r_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(l_data))));

        /* Store result of derivative along the X direction */
        vst1q_s16(reinterpret_cast<int16_t *>(out_x.ptr()), out0);
        vst1q_s16(reinterpret_cast<int16_t *>(out_x.ptr()) + 8, out1);
    },
    in, out_x);
}

void NEDerivativeKernel::derivative_y(const Window &window)
{
    Iterator in(_input, window);
    Iterator out_y(_output_y, window);

    const size_t stride = _input->info()->strides_in_bytes()[1];

    /* Apply 1-D centered point discrete derivative mask ([-1 0 1]^T) along the Y direction */
    execute_window_loop(window, [&](const Coordinates & id)
    {
        /* Load top and bottom data */
        const uint8x16_t t_data = vld1q_u8(in.ptr() - stride);
        const uint8x16_t b_data = vld1q_u8(in.ptr() + stride);

        /* Cast to int16 and perform the subtraction between the bottom and top data */
        const int16x8_t out0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_data))));

        /* Cast to int16 and perform the subtraction between the bottom and top data */
        const int16x8_t out1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_data))));

        /* Store result of derivative along the Y direction */
        vst1q_s16(reinterpret_cast<int16_t *>(out_y.ptr()), out0);
        vst1q_s16(reinterpret_cast<int16_t *>(out_y.ptr()) + 8, out1);
    },
    in, out_y);
}

void NEDerivativeKernel::derivative_xy(const Window &window)
{
    Iterator in(_input, window);
    Iterator out_x(_output_x, window);
    Iterator out_y(_output_y, window);

    const size_t stride = _input->info()->strides_in_bytes()[1];

    /* Apply 1-D centered point discrete derivative masks ([-1 0 1] and [-1 0 1]^T) along the X and Y directions */
    execute_window_loop(window, [&](const Coordinates & id)
    {
        /* Load top, bottom, left and right data */
        const uint8x16_t t_data = vld1q_u8(in.ptr() - stride);
        const uint8x16_t b_data = vld1q_u8(in.ptr() + stride);
        const uint8x16_t l_data = vld1q_u8(in.ptr() - 1);
        const uint8x16_t r_data = vld1q_u8(in.ptr() + 1);

        /* Cast to int16 and perform the subtraction between the bottom and top data */
        const int16x8_t out0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(t_data))));

        /* Cast to int16 and perform the subtraction between the bottom and top data */
        const int16x8_t out1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(b_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(t_data))));

        /* Cast to int16 and perform the subtraction between the right and left data */
        const int16x8_t out2 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(l_data))));

        /* Cast to int16 and perform the subtraction between the right and left data */
        const int16x8_t out3 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(r_data))),
                                         vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(l_data))));

        /* Store result of derivative along the Y direction */
        vst1q_s16(reinterpret_cast<int16_t *>(out_y.ptr()), out0);
        vst1q_s16(reinterpret_cast<int16_t *>(out_y.ptr()) + 8, out1);

        /* Store result of derivative along the X direction */
        vst1q_s16(reinterpret_cast<int16_t *>(out_x.ptr()), out2);
        vst1q_s16(reinterpret_cast<int16_t *>(out_x.ptr()) + 8, out3);
    },
    in, out_x, out_y);
}

void NEDerivativeKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
