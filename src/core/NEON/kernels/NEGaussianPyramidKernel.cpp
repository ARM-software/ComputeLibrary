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
#include "arm_compute/core/NEON/kernels/NEGaussianPyramidKernel.h"

#include "arm_compute/core/AccessWindowAutoPadding.h"
#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>

using namespace arm_compute;

NEGaussianPyramidHorKernel::NEGaussianPyramidHorKernel()
    : _input(nullptr), _output(nullptr)
{
}

NEGaussianPyramidVertKernel::NEGaussianPyramidVertKernel()
    : _input(nullptr), _output(nullptr)
{
}

void NEGaussianPyramidHorKernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != 2 * output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != output->info()->dimension(1));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    const unsigned int processed_elements = 8;

    // Configure kernel window
    Window                  win = calculate_max_window_horizontal(*input->info(), Steps(processed_elements), border_undefined, border_size());
    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    INEKernel::configure(win);
}

BorderSize NEGaussianPyramidHorKernel::border_size() const
{
    return BorderSize(2);
}

void NEGaussianPyramidHorKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(window.x().step() % 2);

    const int16x8_t six  = vdupq_n_s16(6);
    const int16x8_t four = vdupq_n_s16(4);

    //The output is half the width of the input:
    Window win_out(window);
    win_out.set(Window::DimX, Window::Dimension(window.x().start() / 2, window.x().end() / 2, window.x().step() / 2));

    Iterator out(_output, win_out);

    const int even_width = 1 - (_input->info()->dimension(0) % 2);
    Window    win_in(window);
    win_in.shift(Window::DimX, -2 + even_width);

    Iterator in(_input, win_in);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16x2_t data_2q   = vld2q_u8(in.ptr());
        const uint8x16_t &data_even = data_2q.val[0];
        const uint8x16_t &data_odd  = data_2q.val[1];

        const int16x8_t data_l2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data_even)));
        const int16x8_t data_l1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(data_odd)));
        const int16x8_t data_m  = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vextq_u8(data_even, data_even, 1))));
        const int16x8_t data_r1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vextq_u8(data_odd, data_odd, 1))));
        const int16x8_t data_r2 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vextq_u8(data_even, data_even, 2))));

        int16x8_t out_val = vaddq_s16(data_l2, data_r2);
        out_val           = vmlaq_s16(out_val, data_l1, four);
        out_val           = vmlaq_s16(out_val, data_m, six);
        out_val           = vmlaq_s16(out_val, data_r1, four);

        vst1q_s16(reinterpret_cast<int16_t *>(out.ptr()), out_val);
    },
    in, out);
}

void NEGaussianPyramidVertKernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);

    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != output->info()->dimension(0));
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != 2 * output->info()->dimension(1));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    const int          even_height        = 1 - (_input->info()->dimension(1) % 2);
    const unsigned int processed_elements = 16;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(processed_elements), border_undefined, border_size());
    // Use all elements in X direction
    win.set(Window::DimY, Window::Dimension(win.y().start() + even_height, win.y().end() + even_height, 2));

    AccessWindowAutoPadding output_access(output->info());

    update_window_and_padding(win,
                              AccessWindowAutoPadding(input->info()),
                              output_access);

    output_access.set_valid_region();

    INEKernel::configure(win);
}

BorderSize NEGaussianPyramidVertKernel::border_size() const
{
    return BorderSize(2, 0);
}

void NEGaussianPyramidVertKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(window.x().step() != 16);
    ARM_COMPUTE_ERROR_ON(window.y().step() % 2);
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    const uint16x8_t six  = vdupq_n_u16(6);
    const uint16x8_t four = vdupq_n_u16(4);

    Window win_in(window);
    win_in.set_dimension_step(Window::DimX, 8);

    Iterator in(_input, win_in);

    Window win_out(window);
    win_out.set(Window::DimY, Window::Dimension(window.y().start() / 2, window.y().end() / 2, 1));

    Iterator out(_output, win_out);

    const uint8_t *input_top2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(win_in.x().start(), 2));
    const uint8_t *input_top_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(win_in.x().start(), 1));
    const uint8_t *input_mid_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(win_in.x().start(), 0));
    const uint8_t *input_low_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(win_in.x().start(), -1));
    const uint8_t *input_low2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(win_in.x().start(), -2));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Low data
        const uint16x8_t data_low_t2 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top2_ptr + in.offset())));
        const uint16x8_t data_low_t1 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top_ptr + in.offset())));
        const uint16x8_t data_low_m  = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_mid_ptr + in.offset())));
        const uint16x8_t data_low_b1 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low_ptr + in.offset())));
        const uint16x8_t data_low_b2 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low2_ptr + in.offset())));

        uint16x8_t out_low = vaddq_u16(data_low_t2, data_low_b2);
        out_low            = vmlaq_u16(out_low, data_low_t1, four);
        out_low            = vmlaq_u16(out_low, data_low_m, six);
        out_low            = vmlaq_u16(out_low, data_low_b1, four);

        in.increment(Window::DimX);

        // High data
        const uint16x8_t data_high_t2 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top2_ptr + in.offset())));
        const uint16x8_t data_high_t1 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_top_ptr + in.offset())));
        const uint16x8_t data_high_m  = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_mid_ptr + in.offset())));
        const uint16x8_t data_high_b1 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low_ptr + in.offset())));
        const uint16x8_t data_high_b2 = vreinterpretq_u16_s16(vld1q_s16(reinterpret_cast<const int16_t *>(input_low2_ptr + in.offset())));

        uint16x8_t out_high = vaddq_u16(data_high_t2, data_high_b2);
        out_high            = vmlaq_u16(out_high, data_high_t1, four);
        out_high            = vmlaq_u16(out_high, data_high_m, six);
        out_high            = vmlaq_u16(out_high, data_high_b1, four);

        vst1q_u8(out.ptr(), vcombine_u8(vqshrn_n_u16(out_low, 8), vqshrn_n_u16(out_high, 8)));
    },
    in, out);
}
