/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel.h"

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

namespace arm_compute
{
class Coordinates;
} // namespace arm_compute

NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel()
    : _input(nullptr), _output(nullptr), _result_offset(0), _result_mult_int(0), _result_shift(0)
{
}

void NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::configure(const ITensor *input, ITensor *output, int result_offset, int result_mult_int, int result_shift)
{
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::S32);
    ARM_COMPUTE_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::QASYMM8);

    _input           = input;
    _output          = output;
    _result_offset   = result_offset;
    _result_mult_int = result_mult_int;
    _result_shift    = result_shift;

    constexpr unsigned int num_elems_processed_per_iteration = 16;

    // Configure kernel window
    Window win = calculate_max_window(*output->info(), Steps(num_elems_processed_per_iteration));

    AccessWindowHorizontal input_access(input->info(), 0, num_elems_processed_per_iteration);
    AccessWindowHorizontal output_result_access(output->info(), 0, num_elems_processed_per_iteration);

    update_window_and_padding(win,
                              input_access,
                              output_result_access);

    output_result_access.set_valid_region(win, ValidRegion(Coordinates(0, 0), output->info()->tensor_shape()));

    INEKernel::configure(win);
}

void NEGEMMLowpQuantizeDownInt32ToUint8ScaleKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    const int32x4_t result_offset_s32 = vdupq_n_s32(_result_offset);
    const int32x4_t result_shift_s32  = vdupq_n_s32(-_result_shift);
    const int32x4_t zero_s32          = vdupq_n_s32(0);

    Iterator in(_input, window);
    Iterator out(_output, window);

    execute_window_loop(window, [&](const Coordinates & id)
    {
        int32x4x4_t in_s32 =
        {
            {
                vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + 0),
                vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + 4),
                vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + 8),
                vld1q_s32(reinterpret_cast<const int32_t *>(in.ptr()) + 12)
            }
        };

        // Add the offset terms to GEMM's result
        in_s32.val[0] = vaddq_s32(in_s32.val[0], result_offset_s32);
        in_s32.val[1] = vaddq_s32(in_s32.val[1], result_offset_s32);
        in_s32.val[2] = vaddq_s32(in_s32.val[2], result_offset_s32);
        in_s32.val[3] = vaddq_s32(in_s32.val[3], result_offset_s32);

        // Multiply by c_mult_int
        in_s32.val[0] = vmulq_n_s32(in_s32.val[0], _result_mult_int);
        in_s32.val[1] = vmulq_n_s32(in_s32.val[1], _result_mult_int);
        in_s32.val[2] = vmulq_n_s32(in_s32.val[2], _result_mult_int);
        in_s32.val[3] = vmulq_n_s32(in_s32.val[3], _result_mult_int);

        // Shift final result (negative value shift right)
        in_s32.val[0] = vshlq_s32(in_s32.val[0], result_shift_s32);
        in_s32.val[1] = vshlq_s32(in_s32.val[1], result_shift_s32);
        in_s32.val[2] = vshlq_s32(in_s32.val[2], result_shift_s32);
        in_s32.val[3] = vshlq_s32(in_s32.val[3], result_shift_s32);

        // Saturate negative values
        in_s32.val[0] = vmaxq_s32(in_s32.val[0], zero_s32);
        in_s32.val[1] = vmaxq_s32(in_s32.val[1], zero_s32);
        in_s32.val[2] = vmaxq_s32(in_s32.val[2], zero_s32);
        in_s32.val[3] = vmaxq_s32(in_s32.val[3], zero_s32);

        // Convert S32 to S16
        const int16x8x2_t in_s16 =
        {
            {
                vcombine_s16(vqmovn_s32(in_s32.val[0]), vqmovn_s32(in_s32.val[1])),
                vcombine_s16(vqmovn_s32(in_s32.val[2]), vqmovn_s32(in_s32.val[3]))
            }
        };

        // Convert S16 to U8
        const uint8x16_t out_u8 = vcombine_u8(vqmovun_s16(in_s16.val[0]), vqmovun_s16(in_s16.val[1]));

        vst1q_u8(out.ptr(), out_u8);
    },
    in, out);
}