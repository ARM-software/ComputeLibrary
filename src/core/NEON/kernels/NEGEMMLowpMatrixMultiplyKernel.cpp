/*
 * Copyright (c) 2017-2019 ARM Limited.
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

#include "arm_compute/core/AccessWindowStatic.h"
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
namespace
{
void inline vector_matrix_multiply_u8(Iterator &ina, Iterator &inb, Iterator &out, int width_a, int width_b, size_t stride_b, const Window &window)
{
    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(id.x() > width_b)
        {
            return;
        }

        // Note: Since the input are all positives, we can use uint32_t
        // Accumulators for the block 0
        uint32x4x4_t c0 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        auto vec_a          = reinterpret_cast<const uint8_t *>(ina.ptr());
        auto matrix_b       = reinterpret_cast<const uint8_t *>(inb.ptr());
        auto vec_a_end_addr = vec_a + width_a;

        // This for loop performs 8 accumulations
        for(; vec_a <= (vec_a_end_addr - 8);)
        {
            const uint8x8_t  a00_u8 = vld1_u8(vec_a);
            const uint8x16_t b00_u8 = vld1q_u8(matrix_b + 0 * stride_b);
            const uint8x16_t b10_u8 = vld1q_u8(matrix_b + 1 * stride_b);
            const uint8x16_t b20_u8 = vld1q_u8(matrix_b + 2 * stride_b);
            const uint8x16_t b30_u8 = vld1q_u8(matrix_b + 3 * stride_b);
            const uint8x16_t b40_u8 = vld1q_u8(matrix_b + 4 * stride_b);
            const uint8x16_t b50_u8 = vld1q_u8(matrix_b + 5 * stride_b);
            const uint8x16_t b60_u8 = vld1q_u8(matrix_b + 6 * stride_b);
            const uint8x16_t b70_u8 = vld1q_u8(matrix_b + 7 * stride_b);

            // Convert a00_u8 to uint16_t and get the lower part
            const uint16x4x2_t a00_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(a00_u8)),
                    vget_high_u16(vmovl_u8(a00_u8))
                }
            };

            const uint16x4x4_t b00_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))
                }
            };

            const uint16x4x4_t b10_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b10_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b10_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b10_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b10_u8)))
                }
            };

            const uint16x4x4_t b20_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b20_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b20_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b20_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b20_u8)))
                }
            };

            const uint16x4x4_t b30_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b30_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b30_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b30_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b30_u8)))
                }
            };

            const uint16x4x4_t b40_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b40_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b40_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b40_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b40_u8)))
                }
            };

            const uint16x4x4_t b50_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b50_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b50_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b50_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b50_u8)))
                }
            };

            const uint16x4x4_t b60_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b60_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b60_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b60_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b60_u8)))
                }
            };

            const uint16x4x4_t b70_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b70_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b70_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b70_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b70_u8)))
                }
            };

            // Accumulate 0:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16.val[0], 0);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16.val[0], 0);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16.val[0], 0);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16.val[0], 0);

            // Accumulate 1:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b10_u16.val[0], a00_u16.val[0], 1);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b10_u16.val[1], a00_u16.val[0], 1);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b10_u16.val[2], a00_u16.val[0], 1);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b10_u16.val[3], a00_u16.val[0], 1);

            // Accumulate 2:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b20_u16.val[0], a00_u16.val[0], 2);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b20_u16.val[1], a00_u16.val[0], 2);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b20_u16.val[2], a00_u16.val[0], 2);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b20_u16.val[3], a00_u16.val[0], 2);

            // Accumulate 3:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b30_u16.val[0], a00_u16.val[0], 3);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b30_u16.val[1], a00_u16.val[0], 3);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b30_u16.val[2], a00_u16.val[0], 3);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b30_u16.val[3], a00_u16.val[0], 3);

            // Accumulate 4:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b40_u16.val[0], a00_u16.val[1], 0);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b40_u16.val[1], a00_u16.val[1], 0);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b40_u16.val[2], a00_u16.val[1], 0);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b40_u16.val[3], a00_u16.val[1], 0);

            // Accumulate 5:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b50_u16.val[0], a00_u16.val[1], 1);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b50_u16.val[1], a00_u16.val[1], 1);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b50_u16.val[2], a00_u16.val[1], 1);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b50_u16.val[3], a00_u16.val[1], 1);

            // Accumulate 6:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b60_u16.val[0], a00_u16.val[1], 2);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b60_u16.val[1], a00_u16.val[1], 2);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b60_u16.val[2], a00_u16.val[1], 2);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b60_u16.val[3], a00_u16.val[1], 2);

            // Accumulate 7:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b70_u16.val[0], a00_u16.val[1], 3);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b70_u16.val[1], a00_u16.val[1], 3);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b70_u16.val[2], a00_u16.val[1], 3);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b70_u16.val[3], a00_u16.val[1], 3);

            vec_a += 8;
            matrix_b += 8 * stride_b;
        }

        // This for loop performs the left-over accumulations
        for(; vec_a < vec_a_end_addr;)
        {
            const uint8x8_t  a00_u8 = vld1_dup_u8(vec_a);
            const uint8x16_t b00_u8 = vld1q_u8(matrix_b);

            const uint16x4x4_t b00_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))
                }
            };

            // Convert a00_u8 to uint16_t and get the lower part
            const uint16x4_t a00_u16 = vget_low_u16(vmovl_u8(a00_u8));

            // Accumulate 0:
            c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16, 0);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16, 0);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16, 0);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16, 0);

            vec_a += 1;
            matrix_b += stride_b;
        }

        auto vec_out = reinterpret_cast<int32_t *>(out.ptr());
        vst1q_s32(vec_out + 0, vreinterpretq_s32_u32(c0.val[0]));
        vst1q_s32(vec_out + 4, vreinterpretq_s32_u32(c0.val[1]));
        vst1q_s32(vec_out + 8, vreinterpretq_s32_u32(c0.val[2]));
        vst1q_s32(vec_out + 12, vreinterpretq_s32_u32(c0.val[3]));
    },
    ina, inb, out);
}

void inline vector_matrix_multiply_s8(Iterator &ina, Iterator &inb, Iterator &out, int width_a, int width_b, size_t stride_b, const Window &window)
{
    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(id.x() > width_b)
        {
            return;
        }

        // Accumulators for the block 0
        int32x4x4_t c0 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        auto vec_a          = reinterpret_cast<const int8_t *>(ina.ptr());
        auto matrix_b       = reinterpret_cast<const int8_t *>(inb.ptr());
        auto vec_a_end_addr = vec_a + width_a;

        // This for loop performs 8 accumulations
        for(; vec_a <= (vec_a_end_addr - 8);)
        {
            const int8x8_t  a00_s8 = vld1_s8(vec_a);
            const int8x16_t b00_s8 = vld1q_s8(matrix_b + 0 * stride_b);
            const int8x16_t b10_s8 = vld1q_s8(matrix_b + 1 * stride_b);
            const int8x16_t b20_s8 = vld1q_s8(matrix_b + 2 * stride_b);
            const int8x16_t b30_s8 = vld1q_s8(matrix_b + 3 * stride_b);
            const int8x16_t b40_s8 = vld1q_s8(matrix_b + 4 * stride_b);
            const int8x16_t b50_s8 = vld1q_s8(matrix_b + 5 * stride_b);
            const int8x16_t b60_s8 = vld1q_s8(matrix_b + 6 * stride_b);
            const int8x16_t b70_s8 = vld1q_s8(matrix_b + 7 * stride_b);

            // Convert a00_s8 to int16_t and get the lower part
            const int16x4x2_t a00_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(a00_s8)),
                    vget_high_s16(vmovl_s8(a00_s8))
                }
            };

            const int16x4x4_t b00_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))
                }
            };

            const int16x4x4_t b10_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b10_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b10_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b10_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b10_s8)))
                }
            };

            const int16x4x4_t b20_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b20_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b20_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b20_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b20_s8)))
                }
            };

            const int16x4x4_t b30_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b30_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b30_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b30_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b30_s8)))
                }
            };

            const int16x4x4_t b40_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b40_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b40_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b40_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b40_s8)))
                }
            };

            const int16x4x4_t b50_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b50_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b50_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b50_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b50_s8)))
                }
            };

            const int16x4x4_t b60_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b60_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b60_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b60_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b60_s8)))
                }
            };

            const int16x4x4_t b70_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b70_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b70_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b70_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b70_s8)))
                }
            };

            // Accumulate 0:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16.val[0], 0);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16.val[0], 0);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16.val[0], 0);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16.val[0], 0);

            // Accumulate 1:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b10_s16.val[0], a00_s16.val[0], 1);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b10_s16.val[1], a00_s16.val[0], 1);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b10_s16.val[2], a00_s16.val[0], 1);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b10_s16.val[3], a00_s16.val[0], 1);

            // Accumulate 2:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b20_s16.val[0], a00_s16.val[0], 2);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b20_s16.val[1], a00_s16.val[0], 2);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b20_s16.val[2], a00_s16.val[0], 2);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b20_s16.val[3], a00_s16.val[0], 2);

            // Accumulate 3:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b30_s16.val[0], a00_s16.val[0], 3);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b30_s16.val[1], a00_s16.val[0], 3);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b30_s16.val[2], a00_s16.val[0], 3);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b30_s16.val[3], a00_s16.val[0], 3);

            // Accumulate 4:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b40_s16.val[0], a00_s16.val[1], 0);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b40_s16.val[1], a00_s16.val[1], 0);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b40_s16.val[2], a00_s16.val[1], 0);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b40_s16.val[3], a00_s16.val[1], 0);

            // Accumulate 5:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b50_s16.val[0], a00_s16.val[1], 1);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b50_s16.val[1], a00_s16.val[1], 1);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b50_s16.val[2], a00_s16.val[1], 1);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b50_s16.val[3], a00_s16.val[1], 1);

            // Accumulate 6:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b60_s16.val[0], a00_s16.val[1], 2);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b60_s16.val[1], a00_s16.val[1], 2);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b60_s16.val[2], a00_s16.val[1], 2);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b60_s16.val[3], a00_s16.val[1], 2);

            // Accumulate 7:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b70_s16.val[0], a00_s16.val[1], 3);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b70_s16.val[1], a00_s16.val[1], 3);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b70_s16.val[2], a00_s16.val[1], 3);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b70_s16.val[3], a00_s16.val[1], 3);

            vec_a += 8;
            matrix_b += 8 * stride_b;
        }

        // This for loop performs the left-over accumulations
        for(; vec_a < vec_a_end_addr;)
        {
            const int8x8_t  a00_s8 = vld1_dup_s8(vec_a);
            const int8x16_t b00_s8 = vld1q_s8(matrix_b);

            const int16x4x4_t b00_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))
                }
            };

            // Convert a00_s8 to uint16_t and get the lower part
            const int16x4_t a00_s16 = vget_low_s16(vmovl_s8(a00_s8));

            // Accumulate 0:
            c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16, 0);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16, 0);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16, 0);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16, 0);

            vec_a += 1;
            matrix_b += stride_b;
        }

        auto vec_out = reinterpret_cast<int32_t *>(out.ptr());
        vst1q_s32(vec_out + 0, c0.val[0]);
        vst1q_s32(vec_out + 4, c0.val[1]);
        vst1q_s32(vec_out + 8, c0.val[2]);
        vst1q_s32(vec_out + 12, c0.val[3]);
    },
    ina, inb, out);
}

void inline matrix_multiply_u8(Iterator &ina, Iterator &inb, Iterator &out, int width_b, size_t out_stride, const Window &window)
{
    execute_window_loop(window, [&](const Coordinates &)
    {
        const uint8_t *mtx_a0 = ina.ptr();
        const uint8_t *mtx_b0 = inb.ptr();

        // Note: Since the input are all positives, we can use uint32_t
        // Accumulators for the block 0
        uint32x4x4_t c0 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        // Accumulators for the block 1
        uint32x4x4_t c1 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        // Accumulators for the block 2
        uint32x4x4_t c2 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        // Accumulators for the block 3
        uint32x4x4_t c3 =
        {
            {
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0),
                vdupq_n_u32(0)
            }
        };

        for(int k = 0; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16)
        {
            const uint8x8_t  a00_u8 = vld1_u8(mtx_a0);
            const uint8x16_t b00_u8 = vld1q_u8(mtx_b0);

            // Convert a00_u8 to uint16_t and get the lower part
            const uint16x4_t a00_u16 = vget_low_u16(vmovl_u8(a00_u8));

            // Convert b00_s8 to uint16_t
            const uint16x4x4_t b00_u16 =
            {
                {
                    vget_low_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_low_u8(b00_u8))),
                    vget_low_u16(vmovl_u8(vget_high_u8(b00_u8))),
                    vget_high_u16(vmovl_u8(vget_high_u8(b00_u8)))
                }
            };

            // 4x4 block 0
            c0.val[0] = vmlal_lane_u16(c0.val[0], b00_u16.val[0], a00_u16, 0);
            c0.val[1] = vmlal_lane_u16(c0.val[1], b00_u16.val[1], a00_u16, 0);
            c0.val[2] = vmlal_lane_u16(c0.val[2], b00_u16.val[2], a00_u16, 0);
            c0.val[3] = vmlal_lane_u16(c0.val[3], b00_u16.val[3], a00_u16, 0);

            // 4x4 block 1
            c1.val[0] = vmlal_lane_u16(c1.val[0], b00_u16.val[0], a00_u16, 1);
            c1.val[1] = vmlal_lane_u16(c1.val[1], b00_u16.val[1], a00_u16, 1);
            c1.val[2] = vmlal_lane_u16(c1.val[2], b00_u16.val[2], a00_u16, 1);
            c1.val[3] = vmlal_lane_u16(c1.val[3], b00_u16.val[3], a00_u16, 1);

            // 4x4 block 2
            c2.val[0] = vmlal_lane_u16(c2.val[0], b00_u16.val[0], a00_u16, 2);
            c2.val[1] = vmlal_lane_u16(c2.val[1], b00_u16.val[1], a00_u16, 2);
            c2.val[2] = vmlal_lane_u16(c2.val[2], b00_u16.val[2], a00_u16, 2);
            c2.val[3] = vmlal_lane_u16(c2.val[3], b00_u16.val[3], a00_u16, 2);

            // 4x4 block 3
            c3.val[0] = vmlal_lane_u16(c3.val[0], b00_u16.val[0], a00_u16, 3);
            c3.val[1] = vmlal_lane_u16(c3.val[1], b00_u16.val[1], a00_u16, 3);
            c3.val[2] = vmlal_lane_u16(c3.val[2], b00_u16.val[2], a00_u16, 3);
            c3.val[3] = vmlal_lane_u16(c3.val[3], b00_u16.val[3], a00_u16, 3);
        }

        auto mtx_out = reinterpret_cast<int32_t *>(out.ptr());
        vst1q_s32(mtx_out + 0 * out_stride + 0, vreinterpretq_s32_u32(c0.val[0]));
        vst1q_s32(mtx_out + 0 * out_stride + 4, vreinterpretq_s32_u32(c0.val[1]));
        vst1q_s32(mtx_out + 0 * out_stride + 8, vreinterpretq_s32_u32(c0.val[2]));
        vst1q_s32(mtx_out + 0 * out_stride + 12, vreinterpretq_s32_u32(c0.val[3]));
        vst1q_s32(mtx_out + 1 * out_stride + 0, vreinterpretq_s32_u32(c1.val[0]));
        vst1q_s32(mtx_out + 1 * out_stride + 4, vreinterpretq_s32_u32(c1.val[1]));
        vst1q_s32(mtx_out + 1 * out_stride + 8, vreinterpretq_s32_u32(c1.val[2]));
        vst1q_s32(mtx_out + 1 * out_stride + 12, vreinterpretq_s32_u32(c1.val[3]));
        vst1q_s32(mtx_out + 2 * out_stride + 0, vreinterpretq_s32_u32(c2.val[0]));
        vst1q_s32(mtx_out + 2 * out_stride + 4, vreinterpretq_s32_u32(c2.val[1]));
        vst1q_s32(mtx_out + 2 * out_stride + 8, vreinterpretq_s32_u32(c2.val[2]));
        vst1q_s32(mtx_out + 2 * out_stride + 12, vreinterpretq_s32_u32(c2.val[3]));
        vst1q_s32(mtx_out + 3 * out_stride + 0, vreinterpretq_s32_u32(c3.val[0]));
        vst1q_s32(mtx_out + 3 * out_stride + 4, vreinterpretq_s32_u32(c3.val[1]));
        vst1q_s32(mtx_out + 3 * out_stride + 8, vreinterpretq_s32_u32(c3.val[2]));
        vst1q_s32(mtx_out + 3 * out_stride + 12, vreinterpretq_s32_u32(c3.val[3]));
    },
    ina, inb, out);
}

void inline matrix_multiply_s8(Iterator &ina, Iterator &inb, Iterator &out, int width_b, size_t out_stride, const Window &window)
{
    // The implementation assumes that the matrix A and Matrix B have been reshaped respectively with NEGEMMInterleave4x4 and NEGEMMTranspose1xW
    // The reshaping of the matrices helps to have a cache friendly implementation and helps to avoid the data re-arrangements needed for computing 16x4 elements per iteration
    // All the values needed for computing a single 4x4 block will be read from consecutive memory positions
    execute_window_loop(window, [&](const Coordinates &)
    {
        auto *mtx_a0 = reinterpret_cast<const int8_t *>(ina.ptr());
        auto *mtx_b0 = reinterpret_cast<const int8_t *>(inb.ptr());

        // Note: Since the input are all positives, we can use uint32_t
        // Accumulators for the block 0
        int32x4x4_t c0 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        // Accumulators for the block 1
        int32x4x4_t c1 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        // Accumulators for the block 2
        int32x4x4_t c2 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        // Accumulators for the block 3
        int32x4x4_t c3 =
        {
            {
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0),
                vdupq_n_s32(0)
            }
        };

        for(int k = 0; k < width_b; k += 16, mtx_a0 += 4, mtx_b0 += 16)
        {
            const int8x8_t  a00_s8 = vld1_s8(mtx_a0);
            const int8x16_t b00_s8 = vld1q_s8(mtx_b0);

            // Convert a00_s8 to uint16_t and get the lower part
            const int16x4_t a00_s16 = vget_low_s16(vmovl_s8(a00_s8));

            // Convert b00_s8 to int16_t
            const int16x4x4_t b00_s16 =
            {
                {
                    vget_low_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_low_s8(b00_s8))),
                    vget_low_s16(vmovl_s8(vget_high_s8(b00_s8))),
                    vget_high_s16(vmovl_s8(vget_high_s8(b00_s8)))
                }
            };

            // 4x4 block 0
            c0.val[0] = vmlal_lane_s16(c0.val[0], b00_s16.val[0], a00_s16, 0);
            c0.val[1] = vmlal_lane_s16(c0.val[1], b00_s16.val[1], a00_s16, 0);
            c0.val[2] = vmlal_lane_s16(c0.val[2], b00_s16.val[2], a00_s16, 0);
            c0.val[3] = vmlal_lane_s16(c0.val[3], b00_s16.val[3], a00_s16, 0);

            // 4x4 block 1
            c1.val[0] = vmlal_lane_s16(c1.val[0], b00_s16.val[0], a00_s16, 1);
            c1.val[1] = vmlal_lane_s16(c1.val[1], b00_s16.val[1], a00_s16, 1);
            c1.val[2] = vmlal_lane_s16(c1.val[2], b00_s16.val[2], a00_s16, 1);
            c1.val[3] = vmlal_lane_s16(c1.val[3], b00_s16.val[3], a00_s16, 1);

            // 4x4 block 2
            c2.val[0] = vmlal_lane_s16(c2.val[0], b00_s16.val[0], a00_s16, 2);
            c2.val[1] = vmlal_lane_s16(c2.val[1], b00_s16.val[1], a00_s16, 2);
            c2.val[2] = vmlal_lane_s16(c2.val[2], b00_s16.val[2], a00_s16, 2);
            c2.val[3] = vmlal_lane_s16(c2.val[3], b00_s16.val[3], a00_s16, 2);

            // 4x4 block 3
            c3.val[0] = vmlal_lane_s16(c3.val[0], b00_s16.val[0], a00_s16, 3);
            c3.val[1] = vmlal_lane_s16(c3.val[1], b00_s16.val[1], a00_s16, 3);
            c3.val[2] = vmlal_lane_s16(c3.val[2], b00_s16.val[2], a00_s16, 3);
            c3.val[3] = vmlal_lane_s16(c3.val[3], b00_s16.val[3], a00_s16, 3);
        }

        auto mtx_out = reinterpret_cast<int32_t *>(out.ptr());
        vst1q_s32(mtx_out + 0 * out_stride + 0, c0.val[0]);
        vst1q_s32(mtx_out + 0 * out_stride + 4, c0.val[1]);
        vst1q_s32(mtx_out + 0 * out_stride + 8, c0.val[2]);
        vst1q_s32(mtx_out + 0 * out_stride + 12, c0.val[3]);
        vst1q_s32(mtx_out + 1 * out_stride + 0, c1.val[0]);
        vst1q_s32(mtx_out + 1 * out_stride + 4, c1.val[1]);
        vst1q_s32(mtx_out + 1 * out_stride + 8, c1.val[2]);
        vst1q_s32(mtx_out + 1 * out_stride + 12, c1.val[3]);
        vst1q_s32(mtx_out + 2 * out_stride + 0, c2.val[0]);
        vst1q_s32(mtx_out + 2 * out_stride + 4, c2.val[1]);
        vst1q_s32(mtx_out + 2 * out_stride + 8, c2.val[2]);
        vst1q_s32(mtx_out + 2 * out_stride + 12, c2.val[3]);
        vst1q_s32(mtx_out + 3 * out_stride + 0, c3.val[0]);
        vst1q_s32(mtx_out + 3 * out_stride + 4, c3.val[1]);
        vst1q_s32(mtx_out + 3 * out_stride + 8, c3.val[2]);
        vst1q_s32(mtx_out + 3 * out_stride + 12, c3.val[3]);
    },
    ina, inb, out);
}
} // namespace

class Coordinates;
} // namespace arm_compute

namespace
{
Status validate_arguments(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input0, 1, DataType::QASYMM8, DataType::S8, DataType::U8);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input0, input1);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::S32);

    TensorShape in0_shape = input0->tensor_shape();
    TensorShape in1_shape = input1->tensor_shape();
    TensorShape out_shape = output->tensor_shape();

    // Check vector-by-matrix case
    if(out_shape[1] == 1)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in0_shape[0] != in1_shape[1], "The number of input0's columns must be equal to input1's rows");
    }
    else
    {
        in0_shape.collapse(2);
        in1_shape.collapse(2);
        out_shape.collapse(2);

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in0_shape[2] != out_shape[2], "Output tensor must have the same number of batches of input0 tensor");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in1_shape[2] != 1 && in0_shape[2] != in1_shape[2], "Input1 tensor must have the same number of batches of input0 or the number of batches must be set to 1");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(in1_shape[0] % 16, "Input1's width must be a multiple of 16");
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input0, ITensorInfo *input1, ITensorInfo *output)
{
    constexpr unsigned int num_elems_processed_per_iteration_x = 16;
    constexpr unsigned int num_elems_processed_per_iteration_y = 4;

    Window win;
    bool   window_changed = false;

    // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication
    if((output->dimension(1) == 1))
    {
        // Configure kernel window
        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x));

        // We cannot read out-of-bound elements from matrix A as we use the left-over for loop
        AccessWindowStatic     in0_access(input0, 0, 0, input0->tensor_shape().x(), 1);
        AccessWindowHorizontal in1_access(input1, 0, num_elems_processed_per_iteration_x);
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration_x);

        window_changed = update_window_and_padding(win, in0_access, in1_access, output_access);

        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output_access.set_valid_region(win, ValidRegion(coord, output->tensor_shape()));
    }
    else
    {
        win = calculate_max_window(*output, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

        unsigned int num_k_iterations = ceil_to_multiple(input1->dimension(0), num_elems_processed_per_iteration_x) / 16;
        // For each iteration of "k" we increment the input pointer by 4, and we load 8 elements a the time:
        AccessWindowStatic     in0_access(input0, 0, 0, (num_k_iterations - 1) * 4 + 8, input0->dimension(1));
        AccessWindowHorizontal in1_access(input1, 0, input1->dimension(0));
        AccessWindowRectangle  output_access(output, 0, 0, num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y);

        window_changed = update_window_and_padding(win, in0_access, in1_access, output_access);

        output_access.set_valid_region(win, ValidRegion(Coordinates(), output->tensor_shape()));
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} // namespace

NEGEMMLowpMatrixMultiplyKernel::NEGEMMLowpMatrixMultiplyKernel()
    : _input0(nullptr), _input1(nullptr), _output(nullptr), _slide_matrix_b(true)
{
}

void NEGEMMLowpMatrixMultiplyKernel::configure(const ITensor *input0, const ITensor *input1, ITensor *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input0, input1, output);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input0->info(), input1->info(), output->info()));

    TensorShape in1_shape = input1->info()->tensor_shape();
    in1_shape.collapse(2);

    _input0         = input0;
    _input1         = input1;
    _output         = output;
    _slide_matrix_b = in1_shape[2] != 1;

    // Configure kernel window
    auto win_config = validate_and_configure_window(input0->info(), input1->info(), output->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEGEMMLowpMatrixMultiplyKernel::validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input0, input1, output));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input0->clone().get(), input1->clone().get(), output->clone().get()).first);

    return Status{};
}

void NEGEMMLowpMatrixMultiplyKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    // Check if the output tensor is a vector. If so,the kernel runs the vector-matrix multiplication path
    if((_output->info()->dimension(1) == 1))
    {
        const auto width_matrix_a = static_cast<int>(_input0->info()->dimension(0));
        const auto width_matrix_b = static_cast<int>(_input1->info()->dimension(0));
        const auto in_b_stride    = static_cast<int>(_input1->info()->strides_in_bytes()[1] / data_size_from_type(_input1->info()->data_type()));

        // The implementation computes 16 elements per iteration
        const int window_start_x = 16 * info.thread_id;
        const int window_step_x  = 16 * info.num_threads;
        // Make sure (window_end_x - window_start_x) is a multiple of window_step_x
        const int window_end_x = ceil_to_multiple(width_matrix_b - window_start_x, window_step_x) + window_start_x;

        Window win_out(window);
        win_out.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));
        win_out.set(Window::DimY, Window::Dimension(0, 1, 1));

        Window win_a(window);
        win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_a.set(Window::DimY, Window::Dimension(0, 0, 0));

        Window win_b;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        if(_input1->info()->num_dimensions() >= 3)
        {
            win_b = window;
        }
        win_b.set(Window::DimX, Window::Dimension(window_start_x, window_end_x, window_step_x));
        win_b.set(Window::DimY, Window::Dimension(0, 1, 1));

        Iterator ina(_input0, win_a);
        Iterator inb(_input1, win_b);
        Iterator out(_output, win_out);

        switch(_input0->info()->data_type())
        {
            case DataType::S8:
            {
                vector_matrix_multiply_s8(ina, inb, out, width_matrix_a, width_matrix_b, in_b_stride, window);
                break;
            }
            case DataType::U8:
            case DataType::QASYMM8:
            {
                vector_matrix_multiply_u8(ina, inb, out, width_matrix_a, width_matrix_b, in_b_stride, window);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                break;
            }
        }
    }
    else
    {
        const size_t in_b_stride = _input1->info()->strides_in_bytes()[1];
        const size_t out_stride  = _output->info()->strides_in_bytes()[1] / _output->info()->element_size();

        // Set step_x and step_y for matrix A. Scale by a factor of 4 the Y range as the input interleaved matrix A has 4 times less the rows of the output matrix
        Window win_a(window);
        win_a.set(Window::DimX, Window::Dimension(0, 0, 0));
        win_a.set(Window::DimY, Window::Dimension(window.y().start() / 4, window.y().end() / 4, 1));

        // Set step_x and step_y for matrix B. Scale by a factor of 16 the X range as the input transposed matrix A has 16 times less the columns of the output matrix
        Window win_b;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the the matrix multiplication is used to perform a convolution operation
        if(_slide_matrix_b)
        {
            win_b = window;
        }
        win_b.set(Window::DimX, Window::Dimension(window.x().start() / 16, window.x().end() / 16, in_b_stride));
        win_b.set(Window::DimY, Window::Dimension(0, 0, 0));

        // The step x and step y for the output matrix has been already set using in configure()
        Iterator ina(_input0, win_a);
        Iterator inb(_input1, win_b);
        Iterator out(_output, window);

        const int width_b = _input1->info()->dimension(0);
        switch(_input0->info()->data_type())
        {
            case DataType::S8:
            {
                matrix_multiply_s8(ina, inb, out, width_b, out_stride, window);
                break;
            }
            case DataType::U8:
            case DataType::QASYMM8:
            {
                matrix_multiply_u8(ina, inb, out, width_b, out_stride, window);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                break;
            }
        }
    }
}
