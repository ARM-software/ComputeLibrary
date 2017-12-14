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

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
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
    : _l2_load_offset(0)
{
}

BorderSize NEGaussianPyramidHorKernel::border_size() const
{
    return BorderSize(0, 2);
}

void NEGaussianPyramidHorKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(1) != output->info()->dimension(1));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_elems_read_per_iteration      = 32;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    const float            scale_x                           = static_cast<float>(output->info()->dimension(0)) / input->info()->dimension(0);

    Window                 win = calculate_max_window_horizontal(*input->info(), Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration, scale_x);

    // Sub sampling selects odd pixels (1, 3, 5, ...) for images with even
    // width and even pixels (0, 2, 4, ...) for images with odd width. (Whether
    // a pixel is even or odd is determined based on the tensor shape not the
    // valid region!)
    // Thus the offset from which the first pixel (L2) for the convolution is
    // loaded depends on the anchor and shape of the valid region.
    // In the case of an even shape (= even image width) we need to load L2
    // from -2 if the anchor is odd and from -1 if the anchor is even. That
    // makes sure that L2 is always loaded from an odd pixel.
    // On the other hand, for an odd shape (= odd image width) we need to load
    // L2 from -1 if the anchor is odd and from -2 if the anchor is even to
    // achieve the opposite effect.
    // The condition can be simplified to checking whether anchor + shape is
    // odd (-2) or even (-1) as only adding an odd and an even number will have
    // an odd result.
    _l2_load_offset = -border_size().left;

    if((_input->info()->valid_region().anchor[0] + _input->info()->valid_region().shape[0]) % 2 == 0)
    {
        _l2_load_offset += 1;
    }

    // Replace input access with static window
    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), _l2_load_offset, num_elems_read_per_iteration),
                              output_access);

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGaussianPyramidHorKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(window.x().step() % 2);

    static const int16x8_t six  = vdupq_n_s16(6);
    static const int16x8_t four = vdupq_n_s16(4);

    Window win_in(window);
    win_in.shift(Window::DimX, _l2_load_offset);

    Iterator in(_input, win_in);

    // The output is half the width of the input
    Window win_out(window);
    win_out.scale(Window::DimX, 0.5f);

    Iterator out(_output, win_out);

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

NEGaussianPyramidVertKernel::NEGaussianPyramidVertKernel()
    : _t2_load_offset(0)
{
}

BorderSize NEGaussianPyramidVertKernel::border_size() const
{
    return BorderSize(2, 0);
}

void NEGaussianPyramidVertKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S16);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != output->info()->dimension(0));

    for(size_t i = 2; i < Coordinates::num_max_dimensions; ++i)
    {
        ARM_COMPUTE_ERROR_ON(input->info()->dimension(i) != output->info()->dimension(i));
    }

    _input  = input;
    _output = output;

    // Configure kernel window
    constexpr unsigned int num_elems_processed_per_iteration = 16;
    constexpr unsigned int num_rows_processed_per_iteration  = 2;

    constexpr unsigned int num_elems_written_per_iteration = 16;
    constexpr unsigned int num_rows_written_per_iteration  = 1;

    constexpr unsigned int num_elems_read_per_iteration = 16;
    constexpr unsigned int num_rows_read_per_iteration  = 5;

    const float scale_y = static_cast<float>(output->info()->dimension(1)) / input->info()->dimension(1);

    Window                win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration, num_rows_processed_per_iteration));
    AccessWindowRectangle output_access(output->info(), 0, 0, num_elems_written_per_iteration, num_rows_written_per_iteration, 1.f, scale_y);

    // Determine whether we need to load even or odd rows. See above for a
    // detailed explanation.
    _t2_load_offset = -border_size().top;

    if((_input->info()->valid_region().anchor[1] + _input->info()->valid_region().shape[1]) % 2 == 0)
    {
        _t2_load_offset += 1;
    }

    update_window_and_padding(win,
                              AccessWindowRectangle(input->info(), 0, _t2_load_offset, num_elems_read_per_iteration, num_rows_read_per_iteration),
                              output_access);

    output->info()->set_valid_region(ValidRegion(Coordinates(), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGaussianPyramidVertKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(window.x().step() != 16);
    ARM_COMPUTE_ERROR_ON(window.y().step() % 2);
    ARM_COMPUTE_ERROR_ON(_input->buffer() == nullptr);

    static const uint16x8_t six  = vdupq_n_u16(6);
    static const uint16x8_t four = vdupq_n_u16(4);

    Window win_in(window);
    // Need to load two times 8 values instead of 16 values once
    win_in.set_dimension_step(Window::DimX, 8);
    win_in.shift(Window::DimY, _t2_load_offset);

    Iterator in(_input, win_in);

    // Output's height is half of input's
    Window win_out(window);
    win_out.scale(Window::DimY, 0.5f);

    Iterator out(_output, win_out);

    const uint8_t *input_top2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, 0));
    const uint8_t *input_top_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, 1));
    const uint8_t *input_mid_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, 2));
    const uint8_t *input_low_ptr  = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, 3));
    const uint8_t *input_low2_ptr = _input->buffer() + _input->info()->offset_element_in_bytes(Coordinates(0, 4));

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
