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
#include "arm_compute/core/NEON/kernels/NEIntegralImageKernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>

using namespace arm_compute;

void NEIntegralImageKernel::configure(const ITensor *input, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U32);

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration));
    // The kernel is effectively reading 17 values from -1 as it loads 16
    // starting at -1 and also 16 starting at 0
    AccessWindowRectangle  output_read_access(output->info(), -1, -1, num_elems_processed_per_iteration + 1, 1);
    AccessWindowHorizontal output_write_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              AccessWindowHorizontal(input->info(), 0, num_elems_processed_per_iteration),
                              output_read_access, output_write_access);

    output_write_access.set_valid_region(win, input->info()->valid_region());

    IKernel::configure(win);
}

BorderSize NEIntegralImageKernel::border_size() const
{
    return BorderSize(1, 0, 0, 1);
}

bool NEIntegralImageKernel::is_parallelisable() const
{
    return false;
}

void NEIntegralImageKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator output(_output, window);

    const auto output_top_left = reinterpret_cast<const uint32_t *>(_output->ptr_to_element(Coordinates(-1, -1)));
    const auto output_top_mid  = reinterpret_cast<const uint32_t *>(_output->ptr_to_element(Coordinates(0, -1)));

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t input_pixels = vld1q_u8(input.ptr());

        const uint16x8x2_t tmp =
        {
            {
                vmovl_u8(vget_low_u8(input_pixels)),
                vmovl_u8(vget_high_u8(input_pixels))
            }
        };

        uint32x4x4_t pixels =
        {
            {
                vmovl_u16(vget_low_u16(tmp.val[0])),
                vmovl_u16(vget_high_u16(tmp.val[0])),
                vmovl_u16(vget_low_u16(tmp.val[1])),
                vmovl_u16(vget_high_u16(tmp.val[1]))
            }
        };

        // Divide by four as pointer is now uint32 instead of uint8!
        const size_t off = output.offset() / 4;

        // Add top mid pixel values
        const uint32_t *const top_mid_ptr = output_top_mid + off;

        pixels.val[0] = vaddq_u32(vld1q_u32(top_mid_ptr), pixels.val[0]);
        pixels.val[1] = vaddq_u32(vld1q_u32(top_mid_ptr + 4), pixels.val[1]);
        pixels.val[2] = vaddq_u32(vld1q_u32(top_mid_ptr + 8), pixels.val[2]);
        pixels.val[3] = vaddq_u32(vld1q_u32(top_mid_ptr + 12), pixels.val[3]);

        // Subtract top left diagonal values
        const auto            outptr       = reinterpret_cast<uint32_t *>(output.ptr());
        const uint32_t *const top_left_ptr = output_top_left + off;

        pixels.val[0] = vsubq_u32(pixels.val[0], vld1q_u32(top_left_ptr));
        vst1q_u32(outptr, pixels.val[0]);

        pixels.val[1] = vsubq_u32(pixels.val[1], vld1q_u32(top_left_ptr + 4));
        vst1q_u32(outptr + 4, pixels.val[1]);

        pixels.val[2] = vsubq_u32(pixels.val[2], vld1q_u32(top_left_ptr + 8));
        vst1q_u32(outptr + 8, pixels.val[2]);

        pixels.val[3] = vsubq_u32(pixels.val[3], vld1q_u32(top_left_ptr + 12));
        vst1q_u32(outptr + 12, pixels.val[3]);

        // Perform prefix summation
        for(auto i = 0; i < 16; ++i)
        {
            outptr[i] += outptr[i - 1];
        }
    },
    input, output);
}
