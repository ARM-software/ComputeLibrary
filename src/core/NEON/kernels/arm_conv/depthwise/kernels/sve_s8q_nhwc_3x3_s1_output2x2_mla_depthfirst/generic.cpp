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

#include "arm_gemm.hpp"

#include <cstddef>
#include <cstdint>

#if defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)

namespace arm_conv {
namespace depthwise {

void sve_s8q_nhwc_3x3_s1_output2x2_mla_depthfirst_impl(
  const unsigned int n_channels,
  const int8_t *const *const inptrs,
  const int8_t *const weights,
  const int32_t *const bias,
  const arm_gemm::Requantize32 &qp,
  const int32_t *const requant_muls,
  const int32_t *const requant_shifts,
  int8_t *const *const outptrs
)
{
  struct Params
  {
    long unsigned int n_channels;
    const int8_t *weights;
    const int32_t *bias;
    const arm_gemm::Requantize32 *requant;
    const int32_t *const requant_muls;
    const int32_t *const requant_shifts;
    int8_t *const *const outptrs;
    const int8_t *inptrs[16];

    Params(
      long unsigned int n_channels,
      const int8_t *const *inptrs_raw,
      const int8_t *const weights,
      const int32_t *const bias,
      const arm_gemm::Requantize32 &qp,
      const int32_t *const requant_muls,
      const int32_t *const requant_shifts,
      int8_t *const *outptrs
    ) : n_channels(n_channels), weights(weights), bias(bias),
        requant(&qp), requant_muls(requant_muls),
        requant_shifts(requant_shifts), outptrs(outptrs)
    {
      inptrs[0] = inptrs_raw[5];
      inptrs[1] = inptrs_raw[0];
      inptrs[2] = inptrs_raw[3];
      inptrs[3] = inptrs_raw[6];
      inptrs[4] = inptrs_raw[9];
      inptrs[5] = inptrs_raw[12];
      inptrs[6] = inptrs_raw[15];
      inptrs[7] = inptrs_raw[1];
      inptrs[8] = inptrs_raw[2];
      inptrs[9] = inptrs_raw[10];
      inptrs[10] = inptrs_raw[4];
      inptrs[11] = inptrs_raw[7];
      inptrs[12] = inptrs_raw[8];
      inptrs[13] = inptrs_raw[11];
      inptrs[14] = inptrs_raw[13];
      inptrs[15] = inptrs_raw[14];

    }
  };

  const Params params(n_channels, inptrs, weights, bias, qp,
                      requant_muls, requant_shifts, outptrs);

  __asm__ __volatile__(
    "ldr x17, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ptrue p4.b\n"
    "ldr x16, [%x[params], %[offsetof_Params_weights]]\n"
    "mov x15, #0x0\n"
    "ldr x22, [%x[params], %[offsetof_Params_requant]]\n"
    "mov x14, #0x0\n"
    "ldr x13, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "add x12, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x11, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x19, x22, %[offsetof_Requantize32_a_offset]\n"
    "ldr x21, [%x[params], %[offsetof_Params_outptrs]]\n"
    "add x20, x22, %[offsetof_Requantize32_b_offset]\n"
    "ld1rb { z12.b }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z18.b }, p4/Z, [x20]\n"
    "add x20, x22, %[offsetof_Requantize32_minval]\n"
    "ld1rw { z15.s }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_maxval]\n"
    "ld1rw { z13.s }, p4/Z, [x20]\n"
    "whilelt p3.h, x15, x17\n"
    "ld1rw { z14.s }, p4/Z, [x19]\n"
    "whilelt p2.s, x15, x17\n"
    "ldp x10, x9, [x21, #0x0]\n"
    "mov x19, x15\n"
    "incw x19\n"
    "ldp x28, x27, [x21, #0x10]\n"
    "whilelt p1.s, x19, x17\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z17.s }, p2/Z, [x19]\n"
    "ld1w { z16.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z11.s, z17.s, z16.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z17.s, z17.s, z16.s\n"
    "mov z9.d, z11.d\n"
    "ld1sb { z0.h }, p4/Z, [x16]\n"
    ".inst 0x45521000  // ssublb z0.h, z0.b, z18.b\n"
    "mov z20.d, z17.d\n"
    "ld1sb { z1.h }, p4/Z, [x16, #1, MUL VL]\n"
    "mov z24.d, z11.d\n"
    "ld1sb { z2.h }, p4/Z, [x16, #2, MUL VL]\n"
    ".inst 0x45521021  // ssublb z1.h, z1.b, z18.b\n"
    "mov z19.d, z17.d\n"
    "ld1sb { z3.h }, p4/Z, [x16, #3, MUL VL]\n"
    "mov z26.d, z11.d\n"
    "ld1sb { z4.h }, p4/Z, [x16, #4, MUL VL]\n"
    ".inst 0x45521042  // ssublb z2.h, z2.b, z18.b\n"
    "mov z23.d, z17.d\n"
    "ld1sb { z5.h }, p4/Z, [x16, #5, MUL VL]\n"
    ".inst 0x45521063  // ssublb z3.h, z3.b, z18.b\n"
    "ld1sb { z6.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x16, #7, MUL VL]\n"
    ".inst 0x45521084  // ssublb z4.h, z4.b, z18.b\n"
    "inch x16, ALL, MUL #8\n"
    "ld1sb { z8.h }, p4/Z, [x16]\n"
    "ldp x23, x22, [x12, #0x0]\n"
    ".inst 0x455210a5  // ssublb z5.h, z5.b, z18.b\n"
    ".inst 0x455210c6  // ssublb z6.h, z6.b, z18.b\n"
    "ldp x21, x20, [x12, #0x10]\n"
    ".inst 0x455210e7  // ssublb z7.h, z7.b, z18.b\n"
    ".inst 0x45521108  // ssublb z8.h, z8.b, z18.b\n"
    "ldr x19, [x12, #0x20]\n"
    "ld1sb { z31.h }, p3/Z, [x23, x15]\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    "ld1sb { z30.h }, p3/Z, [x22, x15]\n"
    "ld1sb { z29.h }, p3/Z, [x21, x15]\n"
    ".inst 0x454c13de  // ssublb z30.h, z30.b, z12.b\n"
    "ld1sb { z28.h }, p3/Z, [x20, x15]\n"
    "ld1sb { z27.h }, p3/Z, [x19, x15]\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    ".inst 0x454c137b  // ssublb z27.h, z27.b, z12.b\n"
    "1:"  // Loop
    ".inst 0x448443eb  // smlalb z11.s, p4/M, z31.h, z4.h\n"
    "ldr x21, [x12, #0x28]\n"
    "whilelt p0.h, x14, x17\n"
    ".inst 0x448447f1  // smlalt z17.s, p4/M, z31.h, z4.h\n"
    "ldr x20, [x12, #0x30]\n"
    "inch x16\n"
    ".inst 0x448343e9  // smlalb z9.s, p4/M, z31.h, z3.h\n"
    "ldr x26, [x12, #0x38]\n"
    ".inst 0x448347f4  // smlalt z20.s, p4/M, z31.h, z3.h\n"
    "ldr x25, [x12, #0x40]\n"
    ".inst 0x448143f8  // smlalb z24.s, p4/M, z31.h, z1.h\n"
    "ldr x19, [x12, #0x48]\n"
    ".inst 0x448147f3  // smlalt z19.s, p4/M, z31.h, z1.h\n"
    "ldr x24, [x12, #0x50]\n"
    ".inst 0x448043fa  // smlalb z26.s, p4/M, z31.h, z0.h\n"
    "ldr x23, [x12, #0x58]\n"
    ".inst 0x448047f7  // smlalt z23.s, p4/M, z31.h, z0.h\n"
    "ld1sb { z31.h }, p3/Z, [x21, x15]\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    ".inst 0x448043cb  // smlalb z11.s, p4/M, z30.h, z0.h\n"
    "ldr x22, [x12, #0x60]\n"
    ".inst 0x448047d1  // smlalt z17.s, p4/M, z30.h, z0.h\n"
    "ld1sb { z30.h }, p3/Z, [x19, x15]\n"
    ".inst 0x454c13de  // ssublb z30.h, z30.b, z12.b\n"
    ".inst 0x448243a9  // smlalb z9.s, p4/M, z29.h, z2.h\n"
    "ldr x21, [x12, #0x68]\n"
    ".inst 0x448247b4  // smlalt z20.s, p4/M, z29.h, z2.h\n"
    "ld1sb { z29.h }, p3/Z, [x20, x15]\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x4485438b  // smlalb z11.s, p4/M, z28.h, z5.h\n"
    "ldr x20, [x12, #0x70]\n"
    ".inst 0x44854791  // smlalt z17.s, p4/M, z28.h, z5.h\n"
    "ldr x19, [x12, #0x78]\n"
    ".inst 0x44844389  // smlalb z9.s, p4/M, z28.h, z4.h\n"
    "ld1w { z25.s }, p2/Z, [x13]\n"
    ".inst 0x44844794  // smlalt z20.s, p4/M, z28.h, z4.h\n"
    "ld1w { z16.s }, p1/Z, [x13, #1, MUL VL]\n"
    "addvl x13, x13, #2\n"
    ".inst 0x44824398  // smlalb z24.s, p4/M, z28.h, z2.h\n"
    ".inst 0x44824793  // smlalt z19.s, p4/M, z28.h, z2.h\n"
    ".inst 0x4481439a  // smlalb z26.s, p4/M, z28.h, z1.h\n"
    "uzp1 z10.s, z25.s, z16.s\n"
    "uzp2 z22.s, z25.s, z16.s\n"
    "ld1w { z25.s }, p2/Z, [x11]\n"
    ".inst 0x44814797  // smlalt z23.s, p4/M, z28.h, z1.h\n"
    "ld1sb { z28.h }, p3/Z, [x26, x15]\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    ".inst 0x448643f8  // smlalb z24.s, p4/M, z31.h, z6.h\n"
    "ld1w { z16.s }, p1/Z, [x11, #1, MUL VL]\n"
    ".inst 0x448647f3  // smlalt z19.s, p4/M, z31.h, z6.h\n"
    "ld1sb { z31.h }, p3/Z, [x25, x15]\n"
    "addvl x11, x11, #2\n"
    ".inst 0x4487436b  // smlalb z11.s, p4/M, z27.h, z7.h\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    "uzp1 z21.s, z25.s, z16.s\n"
    "uzp2 z25.s, z25.s, z16.s\n"
    ".inst 0x44874771  // smlalt z17.s, p4/M, z27.h, z7.h\n"
    ".inst 0x44864369  // smlalb z9.s, p4/M, z27.h, z6.h\n"
    ".inst 0x44864774  // smlalt z20.s, p4/M, z27.h, z6.h\n"
    ".inst 0x44844378  // smlalb z24.s, p4/M, z27.h, z4.h\n"
    ".inst 0x44844773  // smlalt z19.s, p4/M, z27.h, z4.h\n"
    ".inst 0x4483437a  // smlalb z26.s, p4/M, z27.h, z3.h\n"
    ".inst 0x44834777  // smlalt z23.s, p4/M, z27.h, z3.h\n"
    ".inst 0x4481438b  // smlalb z11.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44814791  // smlalt z17.s, p4/M, z28.h, z1.h\n"
    ".inst 0x448843ba  // smlalb z26.s, p4/M, z29.h, z8.h\n"
    ".inst 0x448847b7  // smlalt z23.s, p4/M, z29.h, z8.h\n"
    "ld1sb { z29.h }, p3/Z, [x24, x15]\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x44804389  // smlalb z9.s, p4/M, z28.h, z0.h\n"
    ".inst 0x44804794  // smlalt z20.s, p4/M, z28.h, z0.h\n"
    "ld1sb { z28.h }, p3/Z, [x23, x15]\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    ".inst 0x448243eb  // smlalb z11.s, p4/M, z31.h, z2.h\n"
    ".inst 0x448247f1  // smlalt z17.s, p4/M, z31.h, z2.h\n"
    ".inst 0x448143e9  // smlalb z9.s, p4/M, z31.h, z1.h\n"
    ".inst 0x448147f4  // smlalt z20.s, p4/M, z31.h, z1.h\n"
    "ld1sb { z31.h }, p3/Z, [x22, x15]\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    ".inst 0x448843cb  // smlalb z11.s, p4/M, z30.h, z8.h\n"
    ".inst 0x448847d1  // smlalt z17.s, p4/M, z30.h, z8.h\n"
    ".inst 0x448743c9  // smlalb z9.s, p4/M, z30.h, z7.h\n"
    ".inst 0x448747d4  // smlalt z20.s, p4/M, z30.h, z7.h\n"
    ".inst 0x448543d8  // smlalb z24.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448547d3  // smlalt z19.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448443da  // smlalb z26.s, p4/M, z30.h, z4.h\n"
    ".inst 0x448447d7  // smlalt z23.s, p4/M, z30.h, z4.h\n"
    "ld1sb { z30.h }, p3/Z, [x21, x15]\n"
    ".inst 0x454c13de  // ssublb z30.h, z30.b, z12.b\n"
    ".inst 0x448343ab  // smlalb z11.s, p4/M, z29.h, z3.h\n"
    ".inst 0x448347b1  // smlalt z17.s, p4/M, z29.h, z3.h\n"
    ".inst 0x448043b8  // smlalb z24.s, p4/M, z29.h, z0.h\n"
    ".inst 0x448047b3  // smlalt z19.s, p4/M, z29.h, z0.h\n"
    "ld1sb { z29.h }, p3/Z, [x20, x15]\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x44854389  // smlalb z9.s, p4/M, z28.h, z5.h\n"
    ".inst 0x44854794  // smlalt z20.s, p4/M, z28.h, z5.h\n"
    ".inst 0x4482439a  // smlalb z26.s, p4/M, z28.h, z2.h\n"
    ".inst 0x44824797  // smlalt z23.s, p4/M, z28.h, z2.h\n"
    "ld1sb { z28.h }, p3/Z, [x19, x15]\n"
    "inch x15\n"
    ".inst 0x448643eb  // smlalb z11.s, p4/M, z31.h, z6.h\n"
    "whilelt p2.s, x15, x17\n"
    ".inst 0x448647f1  // smlalt z17.s, p4/M, z31.h, z6.h\n"
    "mov x19, x15\n"
    ".inst 0x448343f8  // smlalb z24.s, p4/M, z31.h, z3.h\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    ".inst 0x448347f3  // smlalt z19.s, p4/M, z31.h, z3.h\n"
    "incw x19\n"
    ".inst 0x448843c9  // smlalb z9.s, p4/M, z30.h, z8.h\n"
    "whilelt p1.s, x19, x17\n"
    ".inst 0x04aa756b  // sqrdmulh z11.s, z11.s, z10.s\n"
    "whilelt p3.h, x15, x17\n"
    ".inst 0x04b67631  // sqrdmulh z17.s, z17.s, z22.s\n"
    ".inst 0x448847d4  // smlalt z20.s, p4/M, z30.h, z8.h\n"
    ".inst 0x04aa7529  // sqrdmulh z9.s, z9.s, z10.s\n"
    "and z16.d, z11.d, z21.d\n"
    "asr z16.s, z16.s, #0x1f\n"
    "and z1.d, z17.d, z25.d\n"
    "and z27.d, z9.d, z21.d\n"
    "asr z1.s, z1.s, #0x1f\n"
    ".inst 0x04b67694  // sqrdmulh z20.s, z20.s, z22.s\n"
    ".inst 0x448543da  // smlalb z26.s, p4/M, z30.h, z5.h\n"
    "asr z27.s, z27.s, #0x1f\n"
    ".inst 0x448547d7  // smlalt z23.s, p4/M, z30.h, z5.h\n"
    "sqadd z11.s, z11.s, z16.s\n"
    ".inst 0x448743b8  // smlalb z24.s, p4/M, z29.h, z7.h\n"
    "and z16.d, z20.d, z25.d\n"
    "asr z16.s, z16.s, #0x1f\n"
    "sqadd z17.s, z17.s, z1.s\n"
    "sqadd z9.s, z9.s, z27.s\n"
    ".inst 0x448747b3  // smlalt z19.s, p4/M, z29.h, z7.h\n"
    ".inst 0x448643ba  // smlalb z26.s, p4/M, z29.h, z6.h\n"
    ".inst 0x448647b7  // smlalt z23.s, p4/M, z29.h, z6.h\n"
    ".inst 0x44884398  // smlalb z24.s, p4/M, z28.h, z8.h\n"
    "sqadd z20.s, z20.s, z16.s\n"
    ".inst 0x44884793  // smlalt z19.s, p4/M, z28.h, z8.h\n"
    ".inst 0x4487439a  // smlalb z26.s, p4/M, z28.h, z7.h\n"
    ".inst 0x04aa7718  // sqrdmulh z24.s, z24.s, z10.s\n"
    ".inst 0x44874797  // smlalt z23.s, p4/M, z28.h, z7.h\n"
    ".inst 0x04b67673  // sqrdmulh z19.s, z19.s, z22.s\n"
    ".inst 0x04aa775a  // sqrdmulh z26.s, z26.s, z10.s\n"
    "and z16.d, z24.d, z21.d\n"
    "asr z16.s, z16.s, #0x1f\n"
    "and z7.d, z19.d, z25.d\n"
    "and z3.d, z26.d, z21.d\n"
    "asr z7.s, z7.s, #0x1f\n"
    ".inst 0x04b676f7  // sqrdmulh z23.s, z23.s, z22.s\n"
    ".inst 0x448292ab  // srshl z11.s, p4/M, z11.s, z21.s\n"
    "asr z3.s, z3.s, #0x1f\n"
    ".inst 0x44829331  // srshl z17.s, p4/M, z17.s, z25.s\n"
    "sqadd z24.s, z24.s, z16.s\n"
    ".inst 0x448292a9  // srshl z9.s, p4/M, z9.s, z21.s\n"
    "add z11.s, z11.s, z15.s\n"
    "add z17.s, z17.s, z15.s\n"
    "sqadd z19.s, z19.s, z7.s\n"
    "add z9.s, z9.s, z15.s\n"
    "sqadd z26.s, z26.s, z3.s\n"
    "and z16.d, z23.d, z25.d\n"
    "asr z16.s, z16.s, #0x1f\n"
    "smin z11.s, p4/M, z11.s, z14.s\n"
    "smin z17.s, p4/M, z17.s, z14.s\n"
    "smin z9.s, p4/M, z9.s, z14.s\n"
    ".inst 0x44829334  // srshl z20.s, p4/M, z20.s, z25.s\n"
    ".inst 0x448292b8  // srshl z24.s, p4/M, z24.s, z21.s\n"
    "smax z11.s, p4/M, z11.s, z13.s\n"
    "sqadd z23.s, z23.s, z16.s\n"
    "add z20.s, z20.s, z15.s\n"
    "add z24.s, z24.s, z15.s\n"
    "smax z17.s, p4/M, z17.s, z13.s\n"
    "smax z9.s, p4/M, z9.s, z13.s\n"
    "smin z20.s, p4/M, z20.s, z14.s\n"
    "smin z24.s, p4/M, z24.s, z14.s\n"
    "trn1 z11.h, z11.h, z17.h\n"
    "st1b { z11.h }, p0, [x10, x14]\n"
    "smax z20.s, p4/M, z20.s, z13.s\n"
    ".inst 0x44829333  // srshl z19.s, p4/M, z19.s, z25.s\n"
    "smax z24.s, p4/M, z24.s, z13.s\n"
    ".inst 0x448292ba  // srshl z26.s, p4/M, z26.s, z21.s\n"
    ".inst 0x44829337  // srshl z23.s, p4/M, z23.s, z25.s\n"
    "trn1 z9.h, z9.h, z20.h\n"
    "st1b { z9.h }, p0, [x9, x14]\n"
    "add z19.s, z19.s, z15.s\n"
    "add z26.s, z26.s, z15.s\n"
    "add z23.s, z23.s, z15.s\n"
    "smin z19.s, p4/M, z19.s, z14.s\n"
    "smin z26.s, p4/M, z26.s, z14.s\n"
    "smin z23.s, p4/M, z23.s, z14.s\n"
    "smax z19.s, p4/M, z19.s, z13.s\n"
    "smax z26.s, p4/M, z26.s, z13.s\n"
    "smax z23.s, p4/M, z23.s, z13.s\n"
    "trn1 z24.h, z24.h, z19.h\n"
    "st1b { z24.h }, p0, [x28, x14]\n"
    "trn1 z26.h, z26.h, z23.h\n"
    "st1b { z26.h }, p0, [x27, x14]\n"
    "inch x14\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z17.s }, p2/Z, [x19]\n"
    "ld1w { z16.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z11.s, z17.s, z16.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z17.s, z17.s, z16.s\n"
    "mov z9.d, z11.d\n"
    "ld1sb { z0.h }, p4/Z, [x16]\n"
    ".inst 0x45521000  // ssublb z0.h, z0.b, z18.b\n"
    "mov z20.d, z17.d\n"
    "ld1sb { z1.h }, p4/Z, [x16, #1, MUL VL]\n"
    "mov z24.d, z11.d\n"
    "ld1sb { z2.h }, p4/Z, [x16, #2, MUL VL]\n"
    ".inst 0x45521021  // ssublb z1.h, z1.b, z18.b\n"
    "mov z19.d, z17.d\n"
    "ld1sb { z3.h }, p4/Z, [x16, #3, MUL VL]\n"
    "mov z26.d, z11.d\n"
    "ld1sb { z4.h }, p4/Z, [x16, #4, MUL VL]\n"
    ".inst 0x45521042  // ssublb z2.h, z2.b, z18.b\n"
    "mov z23.d, z17.d\n"
    "ld1sb { z5.h }, p4/Z, [x16, #5, MUL VL]\n"
    ".inst 0x45521063  // ssublb z3.h, z3.b, z18.b\n"
    "ld1sb { z6.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x16, #7, MUL VL]\n"
    ".inst 0x45521084  // ssublb z4.h, z4.b, z18.b\n"
    "inch x16, ALL, MUL #8\n"
    "ld1sb { z8.h }, p4/Z, [x16]\n"
    "ldp x23, x22, [x12, #0x0]\n"
    ".inst 0x455210a5  // ssublb z5.h, z5.b, z18.b\n"
    ".inst 0x455210c6  // ssublb z6.h, z6.b, z18.b\n"
    "ldp x21, x20, [x12, #0x10]\n"
    ".inst 0x455210e7  // ssublb z7.h, z7.b, z18.b\n"
    ".inst 0x45521108  // ssublb z8.h, z8.b, z18.b\n"
    "ldr x19, [x12, #0x20]\n"
    "ld1sb { z31.h }, p3/Z, [x23, x15]\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    "ld1sb { z30.h }, p3/Z, [x22, x15]\n"
    "ld1sb { z29.h }, p3/Z, [x21, x15]\n"
    ".inst 0x454c13de  // ssublb z30.h, z30.b, z12.b\n"
    "ld1sb { z28.h }, p3/Z, [x20, x15]\n"
    "ld1sb { z27.h }, p3/Z, [x19, x15]\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    ".inst 0x454c137b  // ssublb z27.h, z27.b, z12.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)
