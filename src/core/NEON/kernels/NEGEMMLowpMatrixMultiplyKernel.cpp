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

    constexpr unsigned int num_elems_processed_per_iteration_x = 16;
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

    /* Set step_x and step_y for matrix B. Scale by a factor of 16 the X range as the input transposed matrix A has 16 times less the cols of the output matrix */
    Window win_b(window);
    win_b.set(Window::DimX, Window::Dimension(window.x().start() >> 4, window.x().end() >> 4, in_b_stride));
    win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

    /* The step x and step y for the output matrix has been already set using in configure() */
    Iterator ina(_input0, win_a);
    Iterator inb(_input1, win_b);
    Iterator out(_output, window);

    const int32x4_t voffset_a = vdupq_n_s32(_a_offset);
    const int32x4_t voffset_b = vdupq_n_s32(_b_offset);
    const int32x4_t vshiftr   = vdupq_n_s32(-_shift);

    const int width_b = _input1->info()->dimension(0);

    // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with NEGEMMInterleave4x4 and NEGEMMTranspose1xW
    // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
    // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8_t *mtx_a0 = ina.ptr();
        const uint8_t *mtx_b0 = inb.ptr();

        // Accumulators for the block 0
        int32x4x4_t c0 =
        {
            {
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset)
            }
        };

        // Accumulators for the block 1
        int32x4x4_t c1 =
        {
            {
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset)
            }
        };

        // Accumulators for the block 2
        int32x4x4_t c2 =
        {
            {
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset)
            }
        };

        // Accumulators for the block 3
        int32x4x4_t c3 =
        {
            {
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset),
                vdupq_n_s32(_output_offset)
            }
        };

        int k = 0;
        // This for loop performs 4 accumulations per iteration
        for(; k <= (width_b - 64); k += 64, mtx_a0 += 16, mtx_b0 += 64)
        {
            const uint8x8_t p00  = vld1_u8(mtx_a0 + 0);
            const uint8x8_t p01  = vld1_u8(mtx_a0 + 8);
            const uint8x8_t q00l = vld1_u8(mtx_b0 + 0);
            const uint8x8_t q00h = vld1_u8(mtx_b0 + 8);
            const uint8x8_t q01l = vld1_u8(mtx_b0 + 16);
            const uint8x8_t q01h = vld1_u8(mtx_b0 + 24);
            const uint8x8_t q02l = vld1_u8(mtx_b0 + 32);
            const uint8x8_t q02h = vld1_u8(mtx_b0 + 40);
            const uint8x8_t q03l = vld1_u8(mtx_b0 + 48);
            const uint8x8_t q03h = vld1_u8(mtx_b0 + 56);

            const int32x4_t ia0l = vaddw_s16(voffset_a, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(p00))));
            const int32x4_t ia0h = vaddw_s16(voffset_a, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(p00))));
            const int32x4_t ia1l = vaddw_s16(voffset_a, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(p01))));
            const int32x4_t ia1h = vaddw_s16(voffset_a, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(p01))));

            const int32x2x4_t ia0 =
            {
                {
                    vget_low_s32(ia0l),
                    vget_high_s32(ia0l),
                    vget_low_s32(ia0h),
                    vget_high_s32(ia0h)
                }
            };

            const int32x2x4_t ia1 =
            {
                {
                    vget_low_s32(ia1l),
                    vget_high_s32(ia1l),
                    vget_low_s32(ia1h),
                    vget_high_s32(ia1h)
                }
            };

            const int32x4x4_t ib0 =
            {
                {
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q00l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q00l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q00h)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q00h))))
                }
            };

            const int32x4x4_t ib1 =
            {
                {
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q01l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q01l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q01h)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q01h))))
                }
            };

            const int32x4x4_t ib2 =
            {
                {
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q02l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q02l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q02h)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q02h))))
                }
            };

            const int32x4x4_t ib3 =
            {
                {
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q03l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q03l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q03h)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q03h))))
                }
            };

            // 4x4 block 0 - Accumulation 0
            c0.val[0] = vmlaq_lane_s32(c0.val[0], ib0.val[0], ia0.val[0], 0);
            c0.val[1] = vmlaq_lane_s32(c0.val[1], ib0.val[0], ia0.val[0], 1);
            c0.val[2] = vmlaq_lane_s32(c0.val[2], ib0.val[0], ia0.val[1], 0);
            c0.val[3] = vmlaq_lane_s32(c0.val[3], ib0.val[0], ia0.val[1], 1);
            // 4x4 block 0 - Accumulation 1
            c0.val[0] = vmlaq_lane_s32(c0.val[0], ib1.val[0], ia0.val[2], 0);
            c0.val[1] = vmlaq_lane_s32(c0.val[1], ib1.val[0], ia0.val[2], 1);
            c0.val[2] = vmlaq_lane_s32(c0.val[2], ib1.val[0], ia0.val[3], 0);
            c0.val[3] = vmlaq_lane_s32(c0.val[3], ib1.val[0], ia0.val[3], 1);
            // 4x4 block 0 - Accumulation 2
            c0.val[0] = vmlaq_lane_s32(c0.val[0], ib2.val[0], ia1.val[0], 0);
            c0.val[1] = vmlaq_lane_s32(c0.val[1], ib2.val[0], ia1.val[0], 1);
            c0.val[2] = vmlaq_lane_s32(c0.val[2], ib2.val[0], ia1.val[1], 0);
            c0.val[3] = vmlaq_lane_s32(c0.val[3], ib2.val[0], ia1.val[1], 1);
            // 4x4 block 0 - Accumulation 3
            c0.val[0] = vmlaq_lane_s32(c0.val[0], ib3.val[0], ia1.val[2], 0);
            c0.val[1] = vmlaq_lane_s32(c0.val[1], ib3.val[0], ia1.val[2], 1);
            c0.val[2] = vmlaq_lane_s32(c0.val[2], ib3.val[0], ia1.val[3], 0);
            c0.val[3] = vmlaq_lane_s32(c0.val[3], ib3.val[0], ia1.val[3], 1);

            // 4x4 block 1 - Accumulation 0
            c1.val[0] = vmlaq_lane_s32(c1.val[0], ib0.val[1], ia0.val[0], 0);
            c1.val[1] = vmlaq_lane_s32(c1.val[1], ib0.val[1], ia0.val[0], 1);
            c1.val[2] = vmlaq_lane_s32(c1.val[2], ib0.val[1], ia0.val[1], 0);
            c1.val[3] = vmlaq_lane_s32(c1.val[3], ib0.val[1], ia0.val[1], 1);
            // 4x4 block 1 - Accumulation 1
            c1.val[0] = vmlaq_lane_s32(c1.val[0], ib1.val[1], ia0.val[2], 0);
            c1.val[1] = vmlaq_lane_s32(c1.val[1], ib1.val[1], ia0.val[2], 1);
            c1.val[2] = vmlaq_lane_s32(c1.val[2], ib1.val[1], ia0.val[3], 0);
            c1.val[3] = vmlaq_lane_s32(c1.val[3], ib1.val[1], ia0.val[3], 1);
            // 4x4 block 1 - Accumulation 2
            c1.val[0] = vmlaq_lane_s32(c1.val[0], ib2.val[1], ia1.val[0], 0);
            c1.val[1] = vmlaq_lane_s32(c1.val[1], ib2.val[1], ia1.val[0], 1);
            c1.val[2] = vmlaq_lane_s32(c1.val[2], ib2.val[1], ia1.val[1], 0);
            c1.val[3] = vmlaq_lane_s32(c1.val[3], ib2.val[1], ia1.val[1], 1);
            // 4x4 block 1 - Accumulation 3
            c1.val[0] = vmlaq_lane_s32(c1.val[0], ib3.val[1], ia1.val[2], 0);
            c1.val[1] = vmlaq_lane_s32(c1.val[1], ib3.val[1], ia1.val[2], 1);
            c1.val[2] = vmlaq_lane_s32(c1.val[2], ib3.val[1], ia1.val[3], 0);
            c1.val[3] = vmlaq_lane_s32(c1.val[3], ib3.val[1], ia1.val[3], 1);

            // 4x4 block 2 - Accumulation 0
            c2.val[0] = vmlaq_lane_s32(c2.val[0], ib0.val[2], ia0.val[0], 0);
            c2.val[1] = vmlaq_lane_s32(c2.val[1], ib0.val[2], ia0.val[0], 1);
            c2.val[2] = vmlaq_lane_s32(c2.val[2], ib0.val[2], ia0.val[1], 0);
            c2.val[3] = vmlaq_lane_s32(c2.val[3], ib0.val[2], ia0.val[1], 1);
            // 4x4 block 2 - Accumulation 1
            c2.val[0] = vmlaq_lane_s32(c2.val[0], ib1.val[2], ia0.val[2], 0);
            c2.val[1] = vmlaq_lane_s32(c2.val[1], ib1.val[2], ia0.val[2], 1);
            c2.val[2] = vmlaq_lane_s32(c2.val[2], ib1.val[2], ia0.val[3], 0);
            c2.val[3] = vmlaq_lane_s32(c2.val[3], ib1.val[2], ia0.val[3], 1);
            // 4x4 block 2 - Accumulation 2
            c2.val[0] = vmlaq_lane_s32(c2.val[0], ib2.val[2], ia1.val[0], 0);
            c2.val[1] = vmlaq_lane_s32(c2.val[1], ib2.val[2], ia1.val[0], 1);
            c2.val[2] = vmlaq_lane_s32(c2.val[2], ib2.val[2], ia1.val[1], 0);
            c2.val[3] = vmlaq_lane_s32(c2.val[3], ib2.val[2], ia1.val[1], 1);
            // 4x4 block 2 - Accumulation 3
            c2.val[0] = vmlaq_lane_s32(c2.val[0], ib3.val[2], ia1.val[2], 0);
            c2.val[1] = vmlaq_lane_s32(c2.val[1], ib3.val[2], ia1.val[2], 1);
            c2.val[2] = vmlaq_lane_s32(c2.val[2], ib3.val[2], ia1.val[3], 0);
            c2.val[3] = vmlaq_lane_s32(c2.val[3], ib3.val[2], ia1.val[3], 1);

            // 4x4 block 3 - Accumulation 0
            c3.val[0] = vmlaq_lane_s32(c3.val[0], ib0.val[3], ia0.val[0], 0);
            c3.val[1] = vmlaq_lane_s32(c3.val[1], ib0.val[3], ia0.val[0], 1);
            c3.val[2] = vmlaq_lane_s32(c3.val[2], ib0.val[3], ia0.val[1], 0);
            c3.val[3] = vmlaq_lane_s32(c3.val[3], ib0.val[3], ia0.val[1], 1);
            // 4x4 block 3 - Accumulation 1
            c3.val[0] = vmlaq_lane_s32(c3.val[0], ib1.val[3], ia0.val[2], 0);
            c3.val[1] = vmlaq_lane_s32(c3.val[1], ib1.val[3], ia0.val[2], 1);
            c3.val[2] = vmlaq_lane_s32(c3.val[2], ib1.val[3], ia0.val[3], 0);
            c3.val[3] = vmlaq_lane_s32(c3.val[3], ib1.val[3], ia0.val[3], 1);
            // 4x4 block 3 - Accumulation 2
            c3.val[0] = vmlaq_lane_s32(c3.val[0], ib2.val[3], ia1.val[0], 0);
            c3.val[1] = vmlaq_lane_s32(c3.val[1], ib2.val[3], ia1.val[0], 1);
            c3.val[2] = vmlaq_lane_s32(c3.val[2], ib2.val[3], ia1.val[1], 0);
            c3.val[3] = vmlaq_lane_s32(c3.val[3], ib2.val[3], ia1.val[1], 1);
            // 4x4 block 3 - Accumulation 3
            c3.val[0] = vmlaq_lane_s32(c3.val[0], ib3.val[3], ia1.val[2], 0);
            c3.val[1] = vmlaq_lane_s32(c3.val[1], ib3.val[3], ia1.val[2], 1);
            c3.val[2] = vmlaq_lane_s32(c3.val[2], ib3.val[3], ia1.val[3], 0);
            c3.val[3] = vmlaq_lane_s32(c3.val[3], ib3.val[3], ia1.val[3], 1);
        }

        // This for loop handles the left-over accumulations
        for(; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16)
        {
            const uint8x8_t p00  = vld1_u8(mtx_a0);
            const uint8x8_t q00l = vld1_u8(mtx_b0);
            const uint8x8_t q00h = vld1_u8(mtx_b0 + 8);

            const int32x4_t ia0 = vaddw_s16(voffset_a, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(p00))));

            const int32x2x2_t ia =
            {
                {
                    vget_low_s32(ia0),
                    vget_high_s32(ia0)
                }
            };

            const int32x4x4_t ib0 =
            {
                {
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q00l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q00l)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_low_u16(vmovl_u8(q00h)))),
                    vaddw_s16(voffset_b, vreinterpret_s16_u16(vget_high_u16(vmovl_u8(q00h))))
                }
            };

            // 4x4 block 0
            c0.val[0] = vmlaq_lane_s32(c0.val[0], ib0.val[0], ia.val[0], 0);
            c0.val[1] = vmlaq_lane_s32(c0.val[1], ib0.val[0], ia.val[0], 1);
            c0.val[2] = vmlaq_lane_s32(c0.val[2], ib0.val[0], ia.val[1], 0);
            c0.val[3] = vmlaq_lane_s32(c0.val[3], ib0.val[0], ia.val[1], 1);

            // 4x4 block 1
            c1.val[0] = vmlaq_lane_s32(c1.val[0], ib0.val[1], ia.val[0], 0);
            c1.val[1] = vmlaq_lane_s32(c1.val[1], ib0.val[1], ia.val[0], 1);
            c1.val[2] = vmlaq_lane_s32(c1.val[2], ib0.val[1], ia.val[1], 0);
            c1.val[3] = vmlaq_lane_s32(c1.val[3], ib0.val[1], ia.val[1], 1);

            // 4x4 block 2
            c2.val[0] = vmlaq_lane_s32(c2.val[0], ib0.val[2], ia.val[0], 0);
            c2.val[1] = vmlaq_lane_s32(c2.val[1], ib0.val[2], ia.val[0], 1);
            c2.val[2] = vmlaq_lane_s32(c2.val[2], ib0.val[2], ia.val[1], 0);
            c2.val[3] = vmlaq_lane_s32(c2.val[3], ib0.val[2], ia.val[1], 1);

            // 4x4 block 3
            c3.val[0] = vmlaq_lane_s32(c3.val[0], ib0.val[3], ia.val[0], 0);
            c3.val[1] = vmlaq_lane_s32(c3.val[1], ib0.val[3], ia.val[0], 1);
            c3.val[2] = vmlaq_lane_s32(c3.val[2], ib0.val[3], ia.val[1], 0);
            c3.val[3] = vmlaq_lane_s32(c3.val[3], ib0.val[3], ia.val[1], 1);
        }

        c0.val[0] = vshlq_s32(vmulq_n_s32(c0.val[0], _output_mult_int), vshiftr);
        c0.val[1] = vshlq_s32(vmulq_n_s32(c0.val[1], _output_mult_int), vshiftr);
        c0.val[2] = vshlq_s32(vmulq_n_s32(c0.val[2], _output_mult_int), vshiftr);
        c0.val[3] = vshlq_s32(vmulq_n_s32(c0.val[3], _output_mult_int), vshiftr);

        c1.val[0] = vshlq_s32(vmulq_n_s32(c1.val[0], _output_mult_int), vshiftr);
        c1.val[1] = vshlq_s32(vmulq_n_s32(c1.val[1], _output_mult_int), vshiftr);
        c1.val[2] = vshlq_s32(vmulq_n_s32(c1.val[2], _output_mult_int), vshiftr);
        c1.val[3] = vshlq_s32(vmulq_n_s32(c1.val[3], _output_mult_int), vshiftr);

        c2.val[0] = vshlq_s32(vmulq_n_s32(c2.val[0], _output_mult_int), vshiftr);
        c2.val[1] = vshlq_s32(vmulq_n_s32(c2.val[1], _output_mult_int), vshiftr);
        c2.val[2] = vshlq_s32(vmulq_n_s32(c2.val[2], _output_mult_int), vshiftr);
        c2.val[3] = vshlq_s32(vmulq_n_s32(c2.val[3], _output_mult_int), vshiftr);

        c3.val[0] = vshlq_s32(vmulq_n_s32(c3.val[0], _output_mult_int), vshiftr);
        c3.val[1] = vshlq_s32(vmulq_n_s32(c3.val[1], _output_mult_int), vshiftr);
        c3.val[2] = vshlq_s32(vmulq_n_s32(c3.val[2], _output_mult_int), vshiftr);
        c3.val[3] = vshlq_s32(vmulq_n_s32(c3.val[3], _output_mult_int), vshiftr);

        const uint8x16x4_t r =
        {
            {
                vcombine_u8(vqmovun_s16(vcombine_s16(vqmovn_s32(c0.val[0]), vqmovn_s32(c1.val[0]))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(c2.val[0]), vqmovn_s32(c3.val[0])))),
                vcombine_u8(vqmovun_s16(vcombine_s16(vqmovn_s32(c0.val[1]), vqmovn_s32(c1.val[1]))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(c2.val[1]), vqmovn_s32(c3.val[1])))),
                vcombine_u8(vqmovun_s16(vcombine_s16(vqmovn_s32(c0.val[2]), vqmovn_s32(c1.val[2]))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(c2.val[2]), vqmovn_s32(c3.val[2])))),
                vcombine_u8(vqmovun_s16(vcombine_s16(vqmovn_s32(c0.val[3]), vqmovn_s32(c1.val[3]))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(c2.val[3]), vqmovn_s32(c3.val[3]))))
            }
        };

        uint8_t *const mtx_out = out.ptr();
        vst1q_u8(mtx_out + 0 * out_stride, r.val[0]);
        vst1q_u8(mtx_out + 1 * out_stride, r.val[1]);
        vst1q_u8(mtx_out + 2 * out_stride, r.val[2]);
        vst1q_u8(mtx_out + 3 * out_stride, r.val[3]);
    },
    ina, inb, out);
}
