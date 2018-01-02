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
#include "arm_compute/core/NEON/kernels/NEBox3x3Kernel.h"

#include "arm_compute/core/Coordinates.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/INEKernel.h"
#include "arm_compute/core/Validate.h"
#include <arm_neon.h>

using namespace arm_compute;

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void NEBox3x3FP16Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator output(_output, window);

    unsigned char *const input_top_ptr = _input->ptr_to_element(Coordinates(-1, -1));
    unsigned char *const input_mid_ptr = _input->ptr_to_element(Coordinates(-1, 0));
    unsigned char *const input_bot_ptr = _input->ptr_to_element(Coordinates(-1, +1));

    const float16x8_t oneovernine = vdupq_n_f16(1.0f / 9.0f);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        const uint8x16_t top_data = vld1q_u8(input_top_ptr + input.offset());
        const uint8x16_t mid_data = vld1q_u8(input_mid_ptr + input.offset());
        const uint8x16_t bot_data = vld1q_u8(input_bot_ptr + input.offset());

        const float16x8x2_t top_f16 =
        {
            {
                vcvtq_f16_u16(vmovl_u8(vget_low_u8(top_data))),
                vcvtq_f16_u16(vmovl_u8(vget_high_u8(top_data)))
            }
        };

        const float16x8x2_t mid_f16 =
        {
            {
                vcvtq_f16_u16(vmovl_u8(vget_low_u8(mid_data))),
                vcvtq_f16_u16(vmovl_u8(vget_high_u8(mid_data)))
            }
        };

        const float16x8x2_t bot_f16 =
        {
            {
                vcvtq_f16_u16(vmovl_u8(vget_low_u8(bot_data))),
                vcvtq_f16_u16(vmovl_u8(vget_high_u8(bot_data)))
            }
        };

        //top left
        float16x8_t out = top_f16.val[0];
        //top mid
        out = vaddq_f16(out, vextq_f16(top_f16.val[0], top_f16.val[1], 1));
        //top right
        out = vaddq_f16(out, vextq_f16(top_f16.val[0], top_f16.val[1], 2));
        //mid left
        out = vaddq_f16(out, mid_f16.val[0]);
        //mid mid
        out = vaddq_f16(out, vextq_f16(mid_f16.val[0], mid_f16.val[1], 1));
        //mid right
        out = vaddq_f16(out, vextq_f16(mid_f16.val[0], mid_f16.val[1], 2));
        //bot left
        out = vaddq_f16(out, bot_f16.val[0]);
        //bot mid
        out = vaddq_f16(out, vextq_f16(bot_f16.val[0], bot_f16.val[1], 1));
        //bot right
        out = vaddq_f16(out, vextq_f16(bot_f16.val[0], bot_f16.val[1], 2));

        out = vmulq_f16(out, oneovernine);

        vst1_u8(output.ptr(), vqmovun_s16(vcvtq_s16_f16(out)));
    },
    input, output);
}
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */

BorderSize NEBox3x3Kernel::border_size() const
{
    return BorderSize(1);
}

void NEBox3x3Kernel::configure(const ITensor *input, ITensor *output, bool border_undefined)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    set_shape_if_empty(*output->info(), input->info()->tensor_shape());

    set_format_if_unknown(*input->info(), Format::U8);
    set_format_if_unknown(*output->info(), Format::U8);

    ARM_COMPUTE_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);

    _input  = input;
    _output = output;

    constexpr unsigned int num_elems_processed_per_iteration = 8;
    constexpr unsigned int num_elems_read_per_iteration      = 16;
    constexpr unsigned int num_elems_written_per_iteration   = 8;
    constexpr unsigned int num_rows_read_per_iteration       = 3;
    constexpr int          rect_offset_xy                    = -1;

    // Configure kernel window
    Window                 win = calculate_max_window(*input->info(), Steps(num_elems_processed_per_iteration), border_undefined, border_size());
    AccessWindowHorizontal output_access(output->info(), 0, num_elems_written_per_iteration);

    update_window_and_padding(win, AccessWindowRectangle(input->info(), rect_offset_xy, rect_offset_xy, num_elems_read_per_iteration, num_rows_read_per_iteration), output_access);

    output_access.set_valid_region(win, input->info()->valid_region(), border_undefined, border_size());

    INEKernel::configure(win);
}

void NEBox3x3Kernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INESimpleKernel::window(), window);

    Iterator input(_input, window);
    Iterator output(_output, window);

    unsigned char *const input_top_ptr = _input->ptr_to_element(Coordinates(-1, -1));
    unsigned char *const input_mid_ptr = _input->ptr_to_element(Coordinates(-1, 0));
    unsigned char *const input_bot_ptr = _input->ptr_to_element(Coordinates(-1, +1));

    const float32x4_t oneovernine = vdupq_n_f32(1.0f / 9.0f);

    execute_window_loop(window, [&](const Coordinates & id)
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

        //top left
        int16x8_t out = top_s16.val[0];
        //top mid
        out = vaddq_s16(out, vextq_s16(top_s16.val[0], top_s16.val[1], 1));
        //top right
        out = vaddq_s16(out, vextq_s16(top_s16.val[0], top_s16.val[1], 2));
        //mid left
        out = vaddq_s16(out, mid_s16.val[0]);
        //mid mid
        out = vaddq_s16(out, vextq_s16(mid_s16.val[0], mid_s16.val[1], 1));
        //mid right
        out = vaddq_s16(out, vextq_s16(mid_s16.val[0], mid_s16.val[1], 2));
        //bot left
        out = vaddq_s16(out, bot_s16.val[0]);
        //bot mid
        out = vaddq_s16(out, vextq_s16(bot_s16.val[0], bot_s16.val[1], 1));
        //bot right
        out = vaddq_s16(out, vextq_s16(bot_s16.val[0], bot_s16.val[1], 2));

        float32x4_t outfloathigh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(out)));
        float32x4_t outfloatlow  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(out)));

        outfloathigh = vmulq_f32(outfloathigh, oneovernine);
        outfloatlow  = vmulq_f32(outfloatlow, oneovernine);

        out = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(outfloatlow)),
                           vqmovn_s32(vcvtq_s32_f32(outfloathigh)));

        vst1_u8(output.ptr(), vqmovun_s16(out));
    },
    input, output);
}
