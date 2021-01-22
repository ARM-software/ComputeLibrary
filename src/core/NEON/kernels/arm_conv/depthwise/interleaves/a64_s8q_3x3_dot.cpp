/*
 * Copyright (c) 2021 Arm Limited.
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

#if defined(__aarch64__)

#include "arm_gemm.hpp"
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "src/core/NEON/kernels/assembly/depthwise.hpp"
#include <cstdint>

namespace arm_conv {
namespace depthwise {

struct interleave_a64_s8q_3x3_dot
{
  static size_t get_packed_size(const DepthwiseArgs &);
  static void pack_parameters(unsigned int n_channels, void *outptr, const int32_t *bias, const int8_t *weights, const arm_gemm::Requantize32 &qp, size_t ld_weight_col, size_t ld_weight_row);
};

size_t interleave_a64_s8q_3x3_dot::get_packed_size(const DepthwiseArgs &args)
{
  // We store 7 vectors for every <vector_of_ints> of channels.
  const unsigned int n = arm_gemm::roundup(
    arm_gemm::iceildiv((long unsigned int) args.input_channels,
                       get_vector_length<int32_t>(arm_gemm::VLType::None)), 4lu
  );
  return n * 7 * get_vector_length<int8_t>(arm_gemm::VLType::None);
}

void interleave_a64_s8q_3x3_dot::pack_parameters(unsigned int n_channels, void *outptr, const int32_t *bias, const int8_t *weights, const arm_gemm::Requantize32 &qp, size_t ld_weight_col, size_t ld_weight_row)
{
  __asm__ __volatile__(
    "movi v0.16b, #0x0\n"
    "cmp %x[ld_weight_col], XZR\n"
    "movi v31.16b, #0x1\n"
    "csel %x[ld_weight_col], %x[ld_weight_col], %x[n_channels], NE\n"
    "movi v16.4s, #0x9\n"
    "mov x19, #0x3\n"
    "cmp %x[ld_weight_row], XZR\n"
    "mul x19, %x[ld_weight_col], x19\n"
    "csel %x[ld_weight_row], %x[ld_weight_row], x19, NE\n"
    "add x24, %x[weights], %x[ld_weight_row]\n"
    "add x23, x24, %x[ld_weight_row]\n"
    "add x22, %x[ld_weight_col], %x[ld_weight_col]\n"
    "lsr x20, %x[n_channels], #0x2\n"
    "mov x21, #0x0\n"
    "add x19, %x[qp], %[offsetof_input_offset]\n"
    "ld1r { v30.4s }, [x19]\n"
    "add x19, %x[qp], %[offsetof_weights_offset]\n"
    "ld1r { v29.4s }, [x19]\n"
    "mul v29.4s, v29.4s, v30.4s\n"
    "add x19, %x[qp], %[offsetof_per_layer_mul]\n"
    "ld1r { v28.4s }, [x19]\n"
    "mul v29.4s, v29.4s, v16.4s\n"
    "add x19, %x[qp], %[offsetof_per_layer_right_shift]\n"
    "ld1r { v27.4s }, [x19]\n"
    "cbz x20, 4f\n"
    "1:"  // Loop
    "movi v26.4s, #0x0\n"
    "cbz %x[bias], 2f\n"
    "ldr q26, [%x[bias], x21]\n"
    "2:"  // Loop: Skip bias load
    "movi v25.4s, #0x0\n"
    "ldr s24, [%x[weights], #0x0]\n"
    "ldr s23, [%x[weights], %x[ld_weight_col]]\n"
    "zip1 v23.16b, v23.16b, v0.16b\n"
    "ldr s21, [%x[weights], x22]\n"
    "add %x[weights], %x[weights], #0x4\n"
    "zip1 v21.16b, v24.16b, v21.16b\n"
    "ldr s22, [x24, #0x0]\n"
    "ldr s20, [x24, %x[ld_weight_col]]\n"
    "zip1 v21.16b, v21.16b, v23.16b\n"
    "ldr s18, [x24, x22]\n"
    ".inst 0x4e9597f9  // sdot v25.4s, v31.16b, v21.16b\n"
    "add x24, x24, #0x4\n"
    "zip1 v20.16b, v20.16b, v0.16b\n"
    "ldr s19, [x23, #0x0]\n"
    "ldr s17, [x23, %x[ld_weight_col]]\n"
    "zip1 v18.16b, v22.16b, v18.16b\n"
    "ldr s16, [x23, x22]\n"
    "zip1 v18.16b, v18.16b, v20.16b\n"
    "add x23, x23, #0x4\n"
    ".inst 0x4e9297f9  // sdot v25.4s, v31.16b, v18.16b\n"
    "zip1 v17.16b, v17.16b, v0.16b\n"
    "zip1 v16.16b, v19.16b, v16.16b\n"
    "zip1 v16.16b, v16.16b, v17.16b\n"
    ".inst 0x4e9097f9  // sdot v25.4s, v31.16b, v16.16b\n"
    "mls v26.4s, v25.4s, v30.4s\n"
    "add v26.4s, v26.4s, v29.4s\n"
    "str q26, [%x[outptr], #0x0]\n"
    "str q21, [%x[outptr], #0x10]\n"
    "str q18, [%x[outptr], #0x20]\n"
    "str q16, [%x[outptr], #0x30]\n"
    "add %x[outptr], %x[outptr], #0x40\n"
    "cbz %x[rq_mul_perchannel], 3f\n"
    "ldr q28, [%x[rq_mul_perchannel], x21]\n"
    "ldr q27, [%x[rq_shift_perchannel], x21]\n"
    "3:"  // Loop: Quantisation parameters: Store
    "str q28, [%x[outptr], #0x0]\n"
    "add x21, x21, #0x10\n"
    "str q27, [%x[outptr], #0x10]\n"
    "subs x20, x20, #0x1\n"
    "add %x[outptr], %x[outptr], #0x20\n"
    "bgt 1b\n"
    "tst %x[n_channels], #0x3\n"
    "beq 13f\n"
    "4:"  // Oddments
    "movi v26.4s, #0x0\n"
    "cbz %x[bias], 7f\n"
    "add %x[bias], %x[bias], x21\n"
    "tbz %x[n_channels], #1, 5f\n"
    "ld1 { v26.d }[0], [%x[bias]], #0x8\n"
    "tbz %x[n_channels], #0, 6f\n"
    "ld1 { v26.s }[2], [%x[bias]], #0x4\n"
    "b 6f\n"
    "5:"  // Oddments: Load bias: Bit 1: Unset
    "tbz %x[n_channels], #0, 6f\n"
    "ld1 { v26.s }[0], [%x[bias]], #0x4\n"
    "6:"  // Oddments: Load bias: Bit 1: End

    "7:"  // Oddments: Skip bias load
    "tbz %x[n_channels], #1, 8f\n"
    "ld1 { v24.h }[0], [%x[weights]]\n"
    "ld1 { v22.h }[0], [x24]\n"
    "add x20, %x[weights], %x[ld_weight_col]\n"
    "ld1 { v19.h }[0], [x23]\n"
    "add x19, %x[weights], x22\n"
    "ld1 { v23.h }[0], [x20]\n"
    "add %x[weights], %x[weights], #0x2\n"
    "ld1 { v21.h }[0], [x19]\n"
    "add x20, x24, %x[ld_weight_col]\n"
    "add x19, x24, x22\n"
    "ld1 { v20.h }[0], [x20]\n"
    "ld1 { v18.h }[0], [x19]\n"
    "add x24, x24, #0x2\n"
    "add x19, x23, %x[ld_weight_col]\n"
    "ld1 { v17.h }[0], [x19]\n"
    "add x19, x23, x22\n"
    "ld1 { v16.h }[0], [x19]\n"
    "add x23, x23, #0x2\n"
    "tbz %x[n_channels], #0, 9f\n"
    "ld1 { v24.b }[2], [%x[weights]]\n"
    "ld1 { v22.b }[2], [x24]\n"
    "add x20, %x[weights], %x[ld_weight_col]\n"
    "ld1 { v19.b }[2], [x23]\n"
    "add x19, %x[weights], x22\n"
    "ld1 { v23.b }[2], [x20]\n"
    "add %x[weights], %x[weights], #0x1\n"
    "ld1 { v21.b }[2], [x19]\n"
    "add x20, x24, %x[ld_weight_col]\n"
    "add x19, x24, x22\n"
    "ld1 { v20.b }[2], [x20]\n"
    "ld1 { v18.b }[2], [x19]\n"
    "add x20, x23, %x[ld_weight_col]\n"
    "add x19, x23, x22\n"
    "ld1 { v17.b }[2], [x20]\n"
    "ld1 { v16.b }[2], [x19]\n"
    "b 9f\n"
    "8:"  // Oddments: Load weights: Bit 1: Unset
    "tbz %x[n_channels], #0, 9f\n"
    "ld1 { v24.b }[0], [%x[weights]]\n"
    "ld1 { v22.b }[0], [x24]\n"
    "add x20, %x[weights], %x[ld_weight_col]\n"
    "ld1 { v19.b }[0], [x23]\n"
    "add x19, %x[weights], x22\n"
    "ld1 { v23.b }[0], [x20]\n"
    "add %x[weights], %x[weights], #0x1\n"
    "ld1 { v21.b }[0], [x19]\n"
    "add x20, x24, %x[ld_weight_col]\n"
    "add x19, x24, x22\n"
    "ld1 { v20.b }[0], [x20]\n"
    "ld1 { v18.b }[0], [x19]\n"
    "add x20, x23, %x[ld_weight_col]\n"
    "add x19, x23, x22\n"
    "ld1 { v17.b }[0], [x20]\n"
    "ld1 { v16.b }[0], [x19]\n"
    "9:"  // Oddments: Load weights: Bit 1: End
    "zip1 v21.16b, v24.16b, v21.16b\n"
    "zip1 v23.16b, v23.16b, v0.16b\n"
    "zip1 v18.16b, v22.16b, v18.16b\n"
    "zip1 v20.16b, v20.16b, v0.16b\n"
    "zip1 v16.16b, v19.16b, v16.16b\n"
    "zip1 v17.16b, v17.16b, v0.16b\n"
    "zip1 v21.16b, v21.16b, v23.16b\n"
    "zip1 v18.16b, v18.16b, v20.16b\n"
    "zip1 v16.16b, v16.16b, v17.16b\n"
    "movi v25.4s, #0x0\n"
    ".inst 0x4e9597f9  // sdot v25.4s, v31.16b, v21.16b\n"
    ".inst 0x4e9297f9  // sdot v25.4s, v31.16b, v18.16b\n"
    ".inst 0x4e9097f9  // sdot v25.4s, v31.16b, v16.16b\n"
    "mls v26.4s, v25.4s, v30.4s\n"
    "add v26.4s, v26.4s, v29.4s\n"
    "str q26, [%x[outptr], #0x0]\n"
    "str q21, [%x[outptr], #0x10]\n"
    "str q18, [%x[outptr], #0x20]\n"
    "str q16, [%x[outptr], #0x30]\n"
    "add %x[outptr], %x[outptr], #0x40\n"
    "cbz %x[rq_mul_perchannel], 12f\n"
    "add x20, %x[rq_mul_perchannel], x21\n"
    "add x19, %x[rq_shift_perchannel], x21\n"
    "tbz %x[n_channels], #1, 10f\n"
    "ld1 { v28.d }[0], [x20], #0x8\n"
    "ld1 { v27.d }[0], [x19], #0x8\n"
    "tbz %x[n_channels], #0, 11f\n"
    "ld1 { v28.s }[2], [x20], #0x4\n"
    "ld1 { v27.s }[2], [x19], #0x4\n"
    "b 11f\n"
    "10:"  // Oddments: Quantisation parameters: Load quant params: Bit 1: Unset
    "tbz %x[n_channels], #0, 11f\n"
    "ld1 { v28.s }[0], [x20], #0x4\n"
    "ld1 { v27.s }[0], [x19], #0x4\n"
    "11:"  // Oddments: Quantisation parameters: Load quant params: Bit 1: End

    "12:"  // Oddments: Quantisation parameters: Store
    "str q28, [%x[outptr], #0x0]\n"
    "str q27, [%x[outptr], #0x10]\n"
    "add %x[outptr], %x[outptr], #0x20\n"
    "13:"  // End

    : [bias] "+&r" (bias), [ld_weight_col] "+&r" (ld_weight_col), [ld_weight_row] "+&r" (ld_weight_row), [outptr] "+&r" (outptr), [weights] "+&r" (weights)
    : [n_channels] "r" (n_channels), [offsetof_input_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_per_layer_mul] "I" (offsetof(arm_gemm::Requantize32, per_layer_mul)), [offsetof_per_layer_right_shift] "I" (offsetof(arm_gemm::Requantize32, per_layer_right_shift)), [offsetof_weights_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [qp] "r" (&qp), [rq_mul_perchannel] "r" (qp.per_channel_muls), [rq_shift_perchannel] "r" (qp.per_channel_right_shifts)
    : "cc", "memory", "v0", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x23", "x24"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__)
