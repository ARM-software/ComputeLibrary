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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpMatrixMultiplyKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <tuple>

using namespace arm_compute;

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEGEMMLowpMatrixMultiplyKernel::NEGEMMLowpMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr), _a_offset(0), _b_offset(0), _output_offset(0), _output_mult_int(0), _shift(0)
{
}

void NEGEMMLowpMatrixMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output,
                                               int32_t a_offset, int32_t b_offset, int32_t output_offset, int32_t output_mult_int, int32_t shift)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input1, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8);
    ARM_COMPUTE_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1, output);

    _input0          = input0;
    _input1          = input1;
    _output          = output;
    _a_offset        = a_offset;
    _b_offset        = b_offset;
    _output_offset   = output_offset;
    _output_mult_int = output_mult_int;
    _shift           = shift;

    constexpr unsigned int num_elems_processed_per_iteration_x = 4;
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;

    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    AccessWindowRectangle  output_access(output->info(), 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);
    AccessWindowHorizontal in0_access(input0->info(), 0, num_elems_processed_per_iteration_x);
    AccessWindowHorizontal in1_access(input1->info(), 0, num_elems_processed_per_iteration_x);

    update_window_and_padding(win, in0_access, in1_access, output_access);

    output_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));
    INEKernel::configure(win);
}

void NEGEMMLowpMatrixMultiplyKernel::run(const Window &window)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const size_t in_b_stride = _input1->info()->strides_in_bytes()[1];
    const size_t out_stride  = _output->info()->strides_in_bytes()[1];

    /* Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the output matrix */
    Window win_a(window);
    win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_a.set(Window::DimY, Window::Dimension(window.y().start() >> 2, window.y().end() >> 2, 1));

    /* Set step_x and step_y for matrix B. Scale by a factor of 4 the X range as the input transposed matrix A has 4 times less the cols of the output matrix */
    Window win_b(window);
    win_b.set(Window::DimX, Window::Dimension(window.x().start() >> 2, window.x().end() >> 2, in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

    /* The step x and step y for the output matrix has been already set using in configure() */
    Iterator ina(_input0, win_a);
    Iterator inb(_input1, win_b);
    Iterator out(_output, window);

    const int32x4_t voffset_a   = vdupq_n_s32(_a_offset);
    const int32x4_t voffset_b   = vdupq_n_s32(_b_offset);
    const int32x4_t voffset_out = vdupq_n_s32(_output_offset);
    const int32x4_t vshiftr     = vdupq_n_s32(-_shift);

    const int width_b   = _input1->info()->dimension(0);
    const int max_it_16 = width_b - 16;

    execute_window_loop(window, [&](const Coordinates &)
    {
        auto *mtx_a0 = reinterpret_cast<const int8_t *>(ina.ptr());
        auto *mtx_b0 = reinterpret_cast<const int8_t *>(inb.ptr());

        int32x4x4_t c =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };
        int k = 0;
        // if max_it_16 < 0 we skip the for block and fall back to process just 4 elements
        for(; k <= max_it_16; k += 16, mtx_a0 += 16, mtx_b0 += 16)
        {
            const int8x16_t   p00 = vld1q_s8(mtx_a0);
            const int8x16_t   q00 = vld1q_s8(mtx_b0);
            const int32x4x4_t ia0 =
            {
                {
                    vaddw_s16(voffset_a, vget_low_s16(vmovl_s8(vget_low_s8(p00)))),
                    vaddw_s16(voffset_a, vget_high_s16(vmovl_s8(vget_low_s8(p00)))),
                    vaddw_s16(voffset_a, vget_low_s16(vmovl_s8(vget_high_s8(p00)))),
                    vaddw_s16(voffset_a, vget_high_s16(vmovl_s8(vget_high_s8(p00))))
                }
            };
            const int32x4x4_t ib0 =
            {
                {
                    vaddw_s16(voffset_b, vget_low_s16(vmovl_s8(vget_low_s8(q00)))),
                    vaddw_s16(voffset_b, vget_high_s16(vmovl_s8(vget_low_s8(q00)))),
                    vaddw_s16(voffset_b, vget_low_s16(vmovl_s8(vget_high_s8(q00)))),
                    vaddw_s16(voffset_b, vget_high_s16(vmovl_s8(vget_high_s8(q00))))
                }
            };
            /* Accumulation 0 */
            c.val[0] = vmlaq_lane_s32(c.val[0], ib0.val[0], vget_low_s32(ia0.val[0]), 0);
            c.val[1] = vmlaq_lane_s32(c.val[1], ib0.val[0], vget_low_s32(ia0.val[0]), 1);
            c.val[2] = vmlaq_lane_s32(c.val[2], ib0.val[0], vget_high_s32(ia0.val[0]), 0);
            c.val[3] = vmlaq_lane_s32(c.val[3], ib0.val[0], vget_high_s32(ia0.val[0]), 1);
            /* Accumulation 1 */
            c.val[0] = vmlaq_lane_s32(c.val[0], ib0.val[1], vget_low_s32(ia0.val[1]), 0);
            c.val[1] = vmlaq_lane_s32(c.val[1], ib0.val[1], vget_low_s32(ia0.val[1]), 1);
            c.val[2] = vmlaq_lane_s32(c.val[2], ib0.val[1], vget_high_s32(ia0.val[1]), 0);
            c.val[3] = vmlaq_lane_s32(c.val[3], ib0.val[1], vget_high_s32(ia0.val[1]), 1);
            /* Accumulation 2 */
            c.val[0] = vmlaq_lane_s32(c.val[0], ib0.val[2], vget_low_s32(ia0.val[2]), 0);
            c.val[1] = vmlaq_lane_s32(c.val[1], ib0.val[2], vget_low_s32(ia0.val[2]), 1);
            c.val[2] = vmlaq_lane_s32(c.val[2], ib0.val[2], vget_high_s32(ia0.val[2]), 0);
            c.val[3] = vmlaq_lane_s32(c.val[3], ib0.val[2], vget_high_s32(ia0.val[2]), 1);
            /* Accumulation 3 */
            c.val[0] = vmlaq_lane_s32(c.val[0], ib0.val[3], vget_low_s32(ia0.val[3]), 0);
            c.val[1] = vmlaq_lane_s32(c.val[1], ib0.val[3], vget_low_s32(ia0.val[3]), 1);
            c.val[2] = vmlaq_lane_s32(c.val[2], ib0.val[3], vget_high_s32(ia0.val[3]), 0);
            c.val[3] = vmlaq_lane_s32(c.val[3], ib0.val[3], vget_high_s32(ia0.val[3]), 1);
        }
        for(; k < width_b; k += 4, mtx_a0 += 4, mtx_b0 += 4)
        {
            const int8x8_t  p00 = vld1_s8(mtx_a0);
            const int8x8_t  q00 = vld1_s8(mtx_b0);
            const int32x4_t ia0 = vaddw_s16(voffset_a, vget_low_s16(vmovl_s8(p00)));
            const int32x4_t ib0 = vaddw_s16(voffset_b, vget_low_s16(vmovl_s8(q00)));

            c.val[0] = vmlaq_lane_s32(c.val[0], ib0, vget_low_s32(ia0), 0);
            c.val[1] = vmlaq_lane_s32(c.val[1], ib0, vget_low_s32(ia0), 1);
            c.val[2] = vmlaq_lane_s32(c.val[2], ib0, vget_high_s32(ia0), 0);
            c.val[3] = vmlaq_lane_s32(c.val[3], ib0, vget_high_s32(ia0), 1);
        }
        c.val[0] = vshlq_s32(vmulq_n_s32(vaddq_s32(voffset_out, c.val[0]), _output_mult_int), vshiftr);
        c.val[1] = vshlq_s32(vmulq_n_s32(vaddq_s32(voffset_out, c.val[1]), _output_mult_int), vshiftr);
        c.val[2] = vshlq_s32(vmulq_n_s32(vaddq_s32(voffset_out, c.val[2]), _output_mult_int), vshiftr);
        c.val[3] = vshlq_s32(vmulq_n_s32(vaddq_s32(voffset_out, c.val[3]), _output_mult_int), vshiftr);
        const uint8x8x2_t r =
        {
            {
                vqmovun_s16(vcombine_s16(vqmovn_s32(c.val[0]), vqmovn_s32(c.val[1]))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(c.val[2]), vqmovn_s32(c.val[3])))
            }
        };
        const auto mtx_out = reinterpret_cast<uint8_t *>(out.ptr());
        vst1_lane_u8(mtx_out + 0 * out_stride + 0, r.val[0], 0);
        vst1_lane_u8(mtx_out + 0 * out_stride + 1, r.val[0], 1);
        vst1_lane_u8(mtx_out + 0 * out_stride + 2, r.val[0], 2);
        vst1_lane_u8(mtx_out + 0 * out_stride + 3, r.val[0], 3);
        vst1_lane_u8(mtx_out + 1 * out_stride + 0, r.val[0], 4);
        vst1_lane_u8(mtx_out + 1 * out_stride + 1, r.val[0], 5);
        vst1_lane_u8(mtx_out + 1 * out_stride + 2, r.val[0], 6);
        vst1_lane_u8(mtx_out + 1 * out_stride + 3, r.val[0], 7);
        vst1_lane_u8(mtx_out + 2 * out_stride + 0, r.val[1], 0);
        vst1_lane_u8(mtx_out + 2 * out_stride + 1, r.val[1], 1);
        vst1_lane_u8(mtx_out + 2 * out_stride + 2, r.val[1], 2);
        vst1_lane_u8(mtx_out + 2 * out_stride + 3, r.val[1], 3);
        vst1_lane_u8(mtx_out + 3 * out_stride + 0, r.val[1], 4);
        vst1_lane_u8(mtx_out + 3 * out_stride + 1, r.val[1], 5);
        vst1_lane_u8(mtx_out + 3 * out_stride + 2, r.val[1], 6);
        vst1_lane_u8(mtx_out + 3 * out_stride + 3, r.val[1], 7);
    },
    ina, inb, out);
}
