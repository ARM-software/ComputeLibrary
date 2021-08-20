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

#if defined(ARM_COMPUTE_ENABLE_SVE)

#include "arm_gemm.hpp"
#include "src/core/NEON/kernels/arm_gemm/utils.hpp"
#include "src/core/NEON/kernels/assembly/depthwise.hpp"
#include <cstdint>

namespace arm_conv {
namespace depthwise {

struct interleave_sve_u8q_3x3_dot
{
  static size_t get_packed_size(const DepthwiseArgs &);
  static void pack_parameters(unsigned int n_channels, void *outptr, const int32_t *bias, const uint8_t *weights, const arm_gemm::Requantize32 &qp, size_t ld_weight_col, size_t ld_weight_row);
};

size_t interleave_sve_u8q_3x3_dot::get_packed_size(const DepthwiseArgs &args)
{
  // We store 7 vectors for every <vector_of_ints> of channels.
  const unsigned int n = arm_gemm::roundup(
    arm_gemm::iceildiv((long unsigned int) args.input_channels,
                       get_vector_length<int32_t>(arm_gemm::VLType::SVE)), 4lu
  );
  return n * 7 * get_vector_length<uint8_t>(arm_gemm::VLType::SVE);
}

void interleave_sve_u8q_3x3_dot::pack_parameters(unsigned int n_channels, void *outptr, const int32_t *bias, const uint8_t *weights, const arm_gemm::Requantize32 &qp, size_t ld_weight_col, size_t ld_weight_row)
{
  __asm__ __volatile__(
    "mov z30.b, #0x0\n"
    "ptrue p2.b\n"
    "ld1rw { z29.s }, p2/Z, [%x[qp], %[offsetof_input_offset]]\n"
    "mov z28.b, #0x1\n"
    "cmp %x[ld_weight_col], XZR\n"
    "mov z16.s, #0x9\n"
    "ld1rw { z27.s }, p2/Z, [%x[qp], %[offsetof_weights_offset]]\n"
    "csel %x[ld_weight_col], %x[ld_weight_col], %x[n_channels], NE\n"
    "mul z27.s, p2/M, z27.s, z29.s\n"
    "ld1rw { z26.s }, p2/Z, [%x[qp], %[offsetof_per_layer_mul]]\n"
    "mov x19, #0x3\n"
    "mul z27.s, p2/M, z27.s, z16.s\n"
    "ld1rw { z25.s }, p2/Z, [%x[qp], %[offsetof_per_layer_right_shift]]\n"
    "mul x19, %x[ld_weight_col], x19\n"
    "cmp %x[ld_weight_row], XZR\n"
    "add x23, %x[ld_weight_col], %x[ld_weight_col]\n"
    "csel %x[ld_weight_row], %x[ld_weight_row], x19, NE\n"
    "add x22, %x[weights], %x[ld_weight_row]\n"
    "add x21, x22, %x[ld_weight_row]\n"
    "whilelt p1.s, XZR, %x[n_channels]\n"
    "mov x20, #0x0\n"
    "pfalse p8.b\n"
    "cbz %x[bias], 1f\n"
    "ptrue p8.s\n"
    "1:"  // No bias

    "2:"  // Loop
    "mov z24.s, #0x0\n"
    "cntp x19, p2, p1.s\n"
    "and p0.b, p2/Z, p8.b, p1.b\n"
    "ld1w { z23.s }, p0/Z, [%x[bias], x20, LSL #2]\n"
    "whilelt p0.b, XZR, x19\n"
    "ld1b { z17.b }, p0/Z, [%x[weights]]\n"
    "ld1b { z16.b }, p0/Z, [%x[weights], %x[ld_weight_col]]\n"
    "zip1 z18.b, z16.b, z30.b\n"
    "ld1b { z16.b }, p0/Z, [%x[weights], x23]\n"
    "add %x[weights], %x[weights], x19\n"
    "zip1 z16.b, z17.b, z16.b\n"
    "ld1b { z22.b }, p0/Z, [x22]\n"
    "ld1b { z17.b }, p0/Z, [x22, %x[ld_weight_col]]\n"
    "zip1 z21.b, z16.b, z18.b\n"
    "ld1b { z16.b }, p0/Z, [x22, x23]\n"
    "udot z24.s, z28.b, z21.b\n"
    "add x22, x22, x19\n"
    "zip1 z18.b, z17.b, z30.b\n"
    "ld1b { z20.b }, p0/Z, [x21]\n"
    "ld1b { z19.b }, p0/Z, [x21, %x[ld_weight_col]]\n"
    "zip1 z17.b, z22.b, z16.b\n"
    "ld1b { z16.b }, p0/Z, [x21, x23]\n"
    "zip1 z18.b, z17.b, z18.b\n"
    "add x21, x21, x19\n"
    "zip1 z17.b, z19.b, z30.b\n"
    "udot z24.s, z28.b, z18.b\n"
    "zip1 z16.b, z20.b, z16.b\n"
    "zip1 z16.b, z16.b, z17.b\n"
    "udot z24.s, z28.b, z16.b\n"
    "mls z23.s, p2/M, z24.s, z29.s\n"
    "add z23.s, z23.s, z27.s\n"
    "st1w { z23.s }, p2, [%x[outptr]]\n"
    "st1b { z21.b }, p2, [%x[outptr], #1, MUL VL]\n"
    "st1b { z18.b }, p2, [%x[outptr], #2, MUL VL]\n"
    "st1b { z16.b }, p2, [%x[outptr], #3, MUL VL]\n"
    "addvl %x[outptr], %x[outptr], #4\n"
    "cbz %x[rq_mul_perchannel], 3f\n"
    "ld1w { z26.s }, p1/Z, [%x[rq_mul_perchannel], x20, LSL #2]\n"
    "ld1w { z25.s }, p1/Z, [%x[rq_shift_perchannel], x20, LSL #2]\n"
    "3:"  // Loop: Quantisation parameters: Store
    "st1w { z26.s }, p2, [%x[outptr]]\n"
    "incw x20\n"
    "st1w { z25.s }, p2, [%x[outptr], #1, MUL VL]\n"
    "whilelt p1.s, x20, %x[n_channels]\n"
    "addvl %x[outptr], %x[outptr], #2\n"
    "b.any 2b\n"
    : [ld_weight_col] "+&r" (ld_weight_col), [ld_weight_row] "+&r" (ld_weight_row), [outptr] "+&r" (outptr), [weights] "+&r" (weights)
    : [bias] "r" (bias), [n_channels] "r" (n_channels), [offsetof_input_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_per_layer_mul] "I" (offsetof(arm_gemm::Requantize32, per_layer_mul)), [offsetof_per_layer_right_shift] "I" (offsetof(arm_gemm::Requantize32, per_layer_right_shift)), [offsetof_weights_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [qp] "r" (&qp), [rq_mul_perchannel] "r" (qp.per_channel_muls), [rq_shift_perchannel] "r" (qp.per_channel_right_shifts)
    : "cc", "memory", "p0", "p1", "p2", "p8", "x19", "x20", "x21", "x22", "x23", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
