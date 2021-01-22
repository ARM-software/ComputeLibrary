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

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(SVE2)

namespace arm_conv {
namespace depthwise {

void sve_u8s8u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_impl(
  const unsigned int n_channels,
  const uint8_t *const *const inptrs,
  const int8_t *const weights,
  const int32_t *const bias,
  const arm_gemm::Requantize32 &qp,
  const int32_t *const requant_muls,
  const int32_t *const requant_shifts,
  uint8_t *const *const outptrs
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
    uint8_t *const *const outptrs;
    const uint8_t *inptrs[16];

    Params(
      long unsigned int n_channels,
      const uint8_t *const *inptrs_raw,
      const int8_t *const weights,
      const int32_t *const bias,
      const arm_gemm::Requantize32 &qp,
      const int32_t *const requant_muls,
      const int32_t *const requant_shifts,
      uint8_t *const *outptrs
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
    "ldr x8, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ptrue p4.b\n"
    "ldr x17, [%x[params], %[offsetof_Params_weights]]\n"
    "mov x16, #0x0\n"
    "ldr x22, [%x[params], %[offsetof_Params_requant]]\n"
    "mov x15, #0x0\n"
    "ldr x14, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "add x13, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x12, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x19, x22, %[offsetof_Requantize32_a_offset]\n"
    "ldr x21, [%x[params], %[offsetof_Params_outptrs]]\n"
    "add x20, x22, %[offsetof_Requantize32_b_offset]\n"
    "ld1rb { z11.b }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z26.b }, p4/Z, [x20]\n"
    "add x20, x22, %[offsetof_Requantize32_minval]\n"
    "ld1rw { z12.s }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_maxval]\n"
    "ld1rw { z14.s }, p4/Z, [x20]\n"
    "whilelt p3.h, x16, x8\n"
    "ld1rw { z17.s }, p4/Z, [x19]\n"
    "whilelt p2.s, x16, x8\n"
    "ldp x11, x10, [x21, #0x0]\n"
    "mov x19, x16\n"
    "incw x19\n"
    "ldp x9, x28, [x21, #0x10]\n"
    "whilelt p1.s, x19, x8\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z10.s }, p2/Z, [x19]\n"
    "ld1w { z16.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z13.s, z10.s, z16.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z15.s, z10.s, z16.s\n"
    "mov z25.d, z13.d\n"
    "ld1sb { z0.h }, p4/Z, [x17]\n"
    "mov z23.d, z13.d\n"
    "ld1sb { z1.h }, p4/Z, [x17, #1, MUL VL]\n"
    "mov z9.d, z15.d\n"
    "ld1sb { z2.h }, p4/Z, [x17, #2, MUL VL]\n"
    "mov z22.d, z15.d\n"
    "ld1sb { z3.h }, p4/Z, [x17, #3, MUL VL]\n"
    "mov z10.d, z13.d\n"
    "ld1sb { z4.h }, p4/Z, [x17, #4, MUL VL]\n"
    "mov z24.d, z15.d\n"
    "ld1sb { z5.h }, p4/Z, [x17, #5, MUL VL]\n"
    ".inst 0x455a1000  // ssublb z0.h, z0.b, z26.b\n"
    "ld1sb { z6.h }, p4/Z, [x17, #6, MUL VL]\n"
    ".inst 0x455a1021  // ssublb z1.h, z1.b, z26.b\n"
    "ld1sb { z7.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x455a1042  // ssublb z2.h, z2.b, z26.b\n"
    "ld1sb { z8.h }, p4/Z, [x17]\n"
    ".inst 0x455a1063  // ssublb z3.h, z3.b, z26.b\n"
    "ldp x23, x22, [x13, #0x0]\n"
    ".inst 0x455a1084  // ssublb z4.h, z4.b, z26.b\n"
    "ldp x21, x20, [x13, #0x10]\n"
    ".inst 0x455a10a5  // ssublb z5.h, z5.b, z26.b\n"
    ".inst 0x455a10c6  // ssublb z6.h, z6.b, z26.b\n"
    "ldr x19, [x13, #0x20]\n"
    ".inst 0x455a10e7  // ssublb z7.h, z7.b, z26.b\n"
    ".inst 0x455a1108  // ssublb z8.h, z8.b, z26.b\n"
    "ld1b { z31.h }, p3/Z, [x23, x16]\n"
    "ld1b { z30.h }, p3/Z, [x22, x16]\n"
    ".inst 0x454b1bff  // usublb z31.h, z31.b, z11.b\n"
    "ld1b { z29.h }, p3/Z, [x21, x16]\n"
    ".inst 0x454b1bde  // usublb z30.h, z30.b, z11.b\n"
    "ld1b { z28.h }, p3/Z, [x20, x16]\n"
    "ld1b { z27.h }, p3/Z, [x19, x16]\n"
    ".inst 0x454b1bbd  // usublb z29.h, z29.b, z11.b\n"
    ".inst 0x454b1b9c  // usublb z28.h, z28.b, z11.b\n"
    ".inst 0x454b1b7b  // usublb z27.h, z27.b, z11.b\n"
    "1:"  // Loop
    ".inst 0x448443ed  // smlalb z13.s, p4/M, z31.h, z4.h\n"
    "ldr x20, [x13, #0x28]\n"
    "whilelt p0.h, x15, x8\n"
    ".inst 0x448447ef  // smlalt z15.s, p4/M, z31.h, z4.h\n"
    "ldr x27, [x13, #0x30]\n"
    "inch x17\n"
    ".inst 0x448343f9  // smlalb z25.s, p4/M, z31.h, z3.h\n"
    "ldr x26, [x13, #0x38]\n"
    ".inst 0x448347e9  // smlalt z9.s, p4/M, z31.h, z3.h\n"
    "ldr x25, [x13, #0x40]\n"
    ".inst 0x448143f7  // smlalb z23.s, p4/M, z31.h, z1.h\n"
    "ldr x19, [x13, #0x48]\n"
    ".inst 0x448147f6  // smlalt z22.s, p4/M, z31.h, z1.h\n"
    "ldr x24, [x13, #0x50]\n"
    ".inst 0x448043ea  // smlalb z10.s, p4/M, z31.h, z0.h\n"
    "ldr x23, [x13, #0x58]\n"
    ".inst 0x448047f8  // smlalt z24.s, p4/M, z31.h, z0.h\n"
    "ld1b { z31.h }, p3/Z, [x20, x16]\n"
    ".inst 0x448043cd  // smlalb z13.s, p4/M, z30.h, z0.h\n"
    "ldr x22, [x13, #0x60]\n"
    ".inst 0x448047cf  // smlalt z15.s, p4/M, z30.h, z0.h\n"
    "ld1b { z30.h }, p3/Z, [x19, x16]\n"
    ".inst 0x448243b9  // smlalb z25.s, p4/M, z29.h, z2.h\n"
    "ldr x21, [x13, #0x68]\n"
    ".inst 0x454b1bff  // usublb z31.h, z31.b, z11.b\n"
    "ldr x20, [x13, #0x70]\n"
    ".inst 0x448247a9  // smlalt z9.s, p4/M, z29.h, z2.h\n"
    "ld1b { z29.h }, p3/Z, [x27, x16]\n"
    ".inst 0x454b1bde  // usublb z30.h, z30.b, z11.b\n"
    "ldr x19, [x13, #0x78]\n"
    ".inst 0x4485438d  // smlalb z13.s, p4/M, z28.h, z5.h\n"
    "ld1w { z19.s }, p2/Z, [x14]\n"
    ".inst 0x4485478f  // smlalt z15.s, p4/M, z28.h, z5.h\n"
    "ld1w { z16.s }, p1/Z, [x14, #1, MUL VL]\n"
    "addvl x14, x14, #2\n"
    ".inst 0x454b1bbd  // usublb z29.h, z29.b, z11.b\n"
    ".inst 0x44844399  // smlalb z25.s, p4/M, z28.h, z4.h\n"
    ".inst 0x44844789  // smlalt z9.s, p4/M, z28.h, z4.h\n"
    "uzp1 z21.s, z19.s, z16.s\n"
    "uzp2 z18.s, z19.s, z16.s\n"
    "ld1w { z19.s }, p2/Z, [x12]\n"
    ".inst 0x44824397  // smlalb z23.s, p4/M, z28.h, z2.h\n"
    "ld1w { z16.s }, p1/Z, [x12, #1, MUL VL]\n"
    "addvl x12, x12, #2\n"
    ".inst 0x44824796  // smlalt z22.s, p4/M, z28.h, z2.h\n"
    ".inst 0x4481438a  // smlalb z10.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44814798  // smlalt z24.s, p4/M, z28.h, z1.h\n"
    "ld1b { z28.h }, p3/Z, [x26, x16]\n"
    "uzp1 z20.s, z19.s, z16.s\n"
    "uzp2 z19.s, z19.s, z16.s\n"
    ".inst 0x448643f7  // smlalb z23.s, p4/M, z31.h, z6.h\n"
    ".inst 0x454b1b9c  // usublb z28.h, z28.b, z11.b\n"
    ".inst 0x448647f6  // smlalt z22.s, p4/M, z31.h, z6.h\n"
    "ld1b { z31.h }, p3/Z, [x25, x16]\n"
    ".inst 0x4487436d  // smlalb z13.s, p4/M, z27.h, z7.h\n"
    ".inst 0x4487476f  // smlalt z15.s, p4/M, z27.h, z7.h\n"
    ".inst 0x44864379  // smlalb z25.s, p4/M, z27.h, z6.h\n"
    ".inst 0x454b1bff  // usublb z31.h, z31.b, z11.b\n"
    ".inst 0x44864769  // smlalt z9.s, p4/M, z27.h, z6.h\n"
    ".inst 0x44844377  // smlalb z23.s, p4/M, z27.h, z4.h\n"
    ".inst 0x44844776  // smlalt z22.s, p4/M, z27.h, z4.h\n"
    ".inst 0x4483436a  // smlalb z10.s, p4/M, z27.h, z3.h\n"
    ".inst 0x44834778  // smlalt z24.s, p4/M, z27.h, z3.h\n"
    ".inst 0x4481438d  // smlalb z13.s, p4/M, z28.h, z1.h\n"
    ".inst 0x4481478f  // smlalt z15.s, p4/M, z28.h, z1.h\n"
    ".inst 0x448843aa  // smlalb z10.s, p4/M, z29.h, z8.h\n"
    ".inst 0x448847b8  // smlalt z24.s, p4/M, z29.h, z8.h\n"
    "ld1b { z29.h }, p3/Z, [x24, x16]\n"
    ".inst 0x44804399  // smlalb z25.s, p4/M, z28.h, z0.h\n"
    ".inst 0x44804789  // smlalt z9.s, p4/M, z28.h, z0.h\n"
    "ld1b { z28.h }, p3/Z, [x23, x16]\n"
    ".inst 0x448243ed  // smlalb z13.s, p4/M, z31.h, z2.h\n"
    ".inst 0x454b1bbd  // usublb z29.h, z29.b, z11.b\n"
    ".inst 0x448247ef  // smlalt z15.s, p4/M, z31.h, z2.h\n"
    ".inst 0x454b1b9c  // usublb z28.h, z28.b, z11.b\n"
    ".inst 0x448143f9  // smlalb z25.s, p4/M, z31.h, z1.h\n"
    ".inst 0x448147e9  // smlalt z9.s, p4/M, z31.h, z1.h\n"
    "ld1b { z31.h }, p3/Z, [x22, x16]\n"
    ".inst 0x448843cd  // smlalb z13.s, p4/M, z30.h, z8.h\n"
    ".inst 0x448847cf  // smlalt z15.s, p4/M, z30.h, z8.h\n"
    ".inst 0x448743d9  // smlalb z25.s, p4/M, z30.h, z7.h\n"
    ".inst 0x454b1bff  // usublb z31.h, z31.b, z11.b\n"
    ".inst 0x448747c9  // smlalt z9.s, p4/M, z30.h, z7.h\n"
    ".inst 0x448543d7  // smlalb z23.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448547d6  // smlalt z22.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448443ca  // smlalb z10.s, p4/M, z30.h, z4.h\n"
    ".inst 0x448447d8  // smlalt z24.s, p4/M, z30.h, z4.h\n"
    "ld1b { z30.h }, p3/Z, [x21, x16]\n"
    ".inst 0x448343ad  // smlalb z13.s, p4/M, z29.h, z3.h\n"
    ".inst 0x448347af  // smlalt z15.s, p4/M, z29.h, z3.h\n"
    ".inst 0x448043b7  // smlalb z23.s, p4/M, z29.h, z0.h\n"
    ".inst 0x454b1bde  // usublb z30.h, z30.b, z11.b\n"
    ".inst 0x448047b6  // smlalt z22.s, p4/M, z29.h, z0.h\n"
    "ld1b { z29.h }, p3/Z, [x20, x16]\n"
    ".inst 0x44854399  // smlalb z25.s, p4/M, z28.h, z5.h\n"
    ".inst 0x44854789  // smlalt z9.s, p4/M, z28.h, z5.h\n"
    ".inst 0x4482438a  // smlalb z10.s, p4/M, z28.h, z2.h\n"
    ".inst 0x454b1bbd  // usublb z29.h, z29.b, z11.b\n"
    ".inst 0x44824798  // smlalt z24.s, p4/M, z28.h, z2.h\n"
    "ld1b { z28.h }, p3/Z, [x19, x16]\n"
    "inch x16\n"
    ".inst 0x448643ed  // smlalb z13.s, p4/M, z31.h, z6.h\n"
    "whilelt p2.s, x16, x8\n"
    ".inst 0x448647ef  // smlalt z15.s, p4/M, z31.h, z6.h\n"
    "mov x19, x16\n"
    ".inst 0x448343f7  // smlalb z23.s, p4/M, z31.h, z3.h\n"
    "incw x19\n"
    ".inst 0x454b1b9c  // usublb z28.h, z28.b, z11.b\n"
    "whilelt p1.s, x19, x8\n"
    ".inst 0x448347f6  // smlalt z22.s, p4/M, z31.h, z3.h\n"
    "whilelt p3.h, x16, x8\n"
    ".inst 0x04b575ad  // sqrdmulh z13.s, z13.s, z21.s\n"
    ".inst 0x04b275ef  // sqrdmulh z15.s, z15.s, z18.s\n"
    ".inst 0x448843d9  // smlalb z25.s, p4/M, z30.h, z8.h\n"
    ".inst 0x448847c9  // smlalt z9.s, p4/M, z30.h, z8.h\n"
    "and z4.d, z13.d, z20.d\n"
    "and z16.d, z15.d, z19.d\n"
    ".inst 0x04b57739  // sqrdmulh z25.s, z25.s, z21.s\n"
    "asr z4.s, z4.s, #0x1f\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x04b27529  // sqrdmulh z9.s, z9.s, z18.s\n"
    "sqadd z13.s, z13.s, z4.s\n"
    "sqadd z15.s, z15.s, z16.s\n"
    "and z2.d, z25.d, z20.d\n"
    "and z16.d, z9.d, z19.d\n"
    ".inst 0x448543ca  // smlalb z10.s, p4/M, z30.h, z5.h\n"
    "asr z2.s, z2.s, #0x1f\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x448547d8  // smlalt z24.s, p4/M, z30.h, z5.h\n"
    "sqadd z25.s, z25.s, z2.s\n"
    "sqadd z9.s, z9.s, z16.s\n"
    ".inst 0x448743b7  // smlalb z23.s, p4/M, z29.h, z7.h\n"
    ".inst 0x448747b6  // smlalt z22.s, p4/M, z29.h, z7.h\n"
    ".inst 0x448643aa  // smlalb z10.s, p4/M, z29.h, z6.h\n"
    ".inst 0x448647b8  // smlalt z24.s, p4/M, z29.h, z6.h\n"
    ".inst 0x44884397  // smlalb z23.s, p4/M, z28.h, z8.h\n"
    ".inst 0x44884796  // smlalt z22.s, p4/M, z28.h, z8.h\n"
    ".inst 0x4487438a  // smlalb z10.s, p4/M, z28.h, z7.h\n"
    ".inst 0x44874798  // smlalt z24.s, p4/M, z28.h, z7.h\n"
    ".inst 0x04b576f7  // sqrdmulh z23.s, z23.s, z21.s\n"
    ".inst 0x04b276d6  // sqrdmulh z22.s, z22.s, z18.s\n"
    ".inst 0x04b5754a  // sqrdmulh z10.s, z10.s, z21.s\n"
    ".inst 0x04b27718  // sqrdmulh z24.s, z24.s, z18.s\n"
    "and z18.d, z23.d, z20.d\n"
    "and z0.d, z22.d, z19.d\n"
    "and z16.d, z10.d, z20.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    "asr z0.s, z0.s, #0x1f\n"
    "asr z16.s, z16.s, #0x1f\n"
    "sqadd z23.s, z23.s, z18.s\n"
    "sqadd z22.s, z22.s, z0.s\n"
    "sqadd z10.s, z10.s, z16.s\n"
    "and z16.d, z24.d, z19.d\n"
    ".inst 0x4482928d  // srshl z13.s, p4/M, z13.s, z20.s\n"
    ".inst 0x4482926f  // srshl z15.s, p4/M, z15.s, z19.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x44829299  // srshl z25.s, p4/M, z25.s, z20.s\n"
    "add z13.s, z13.s, z12.s\n"
    "add z15.s, z15.s, z12.s\n"
    "sqadd z24.s, z24.s, z16.s\n"
    "add z25.s, z25.s, z12.s\n"
    "smin z13.s, p4/M, z13.s, z17.s\n"
    "smin z15.s, p4/M, z15.s, z17.s\n"
    "smin z25.s, p4/M, z25.s, z17.s\n"
    ".inst 0x44829269  // srshl z9.s, p4/M, z9.s, z19.s\n"
    "smax z13.s, p4/M, z13.s, z14.s\n"
    "smax z15.s, p4/M, z15.s, z14.s\n"
    "smax z25.s, p4/M, z25.s, z14.s\n"
    "add z9.s, z9.s, z12.s\n"
    ".inst 0x44829297  // srshl z23.s, p4/M, z23.s, z20.s\n"
    "trn1 z13.h, z13.h, z15.h\n"
    "st1b { z13.h }, p0, [x11, x15]\n"
    "smin z9.s, p4/M, z9.s, z17.s\n"
    ".inst 0x44829276  // srshl z22.s, p4/M, z22.s, z19.s\n"
    "add z23.s, z23.s, z12.s\n"
    ".inst 0x4482928a  // srshl z10.s, p4/M, z10.s, z20.s\n"
    ".inst 0x44829278  // srshl z24.s, p4/M, z24.s, z19.s\n"
    "add z22.s, z22.s, z12.s\n"
    "smax z9.s, p4/M, z9.s, z14.s\n"
    "add z10.s, z10.s, z12.s\n"
    "add z24.s, z24.s, z12.s\n"
    "smin z23.s, p4/M, z23.s, z17.s\n"
    "trn1 z25.h, z25.h, z9.h\n"
    "st1b { z25.h }, p0, [x10, x15]\n"
    "smin z22.s, p4/M, z22.s, z17.s\n"
    "smin z10.s, p4/M, z10.s, z17.s\n"
    "smax z23.s, p4/M, z23.s, z14.s\n"
    "smin z24.s, p4/M, z24.s, z17.s\n"
    "smax z22.s, p4/M, z22.s, z14.s\n"
    "smax z10.s, p4/M, z10.s, z14.s\n"
    "smax z24.s, p4/M, z24.s, z14.s\n"
    "trn1 z23.h, z23.h, z22.h\n"
    "st1b { z23.h }, p0, [x9, x15]\n"
    "trn1 z10.h, z10.h, z24.h\n"
    "st1b { z10.h }, p0, [x28, x15]\n"
    "inch x15\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z10.s }, p2/Z, [x19]\n"
    "ld1w { z16.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z13.s, z10.s, z16.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z15.s, z10.s, z16.s\n"
    "mov z25.d, z13.d\n"
    "ld1sb { z0.h }, p4/Z, [x17]\n"
    "mov z23.d, z13.d\n"
    "ld1sb { z1.h }, p4/Z, [x17, #1, MUL VL]\n"
    "mov z9.d, z15.d\n"
    "ld1sb { z2.h }, p4/Z, [x17, #2, MUL VL]\n"
    "mov z22.d, z15.d\n"
    "ld1sb { z3.h }, p4/Z, [x17, #3, MUL VL]\n"
    "mov z10.d, z13.d\n"
    "ld1sb { z4.h }, p4/Z, [x17, #4, MUL VL]\n"
    "mov z24.d, z15.d\n"
    "ld1sb { z5.h }, p4/Z, [x17, #5, MUL VL]\n"
    ".inst 0x455a1000  // ssublb z0.h, z0.b, z26.b\n"
    "ld1sb { z6.h }, p4/Z, [x17, #6, MUL VL]\n"
    ".inst 0x455a1021  // ssublb z1.h, z1.b, z26.b\n"
    "ld1sb { z7.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x455a1042  // ssublb z2.h, z2.b, z26.b\n"
    "ld1sb { z8.h }, p4/Z, [x17]\n"
    ".inst 0x455a1063  // ssublb z3.h, z3.b, z26.b\n"
    "ldp x23, x22, [x13, #0x0]\n"
    ".inst 0x455a1084  // ssublb z4.h, z4.b, z26.b\n"
    "ldp x21, x20, [x13, #0x10]\n"
    ".inst 0x455a10a5  // ssublb z5.h, z5.b, z26.b\n"
    ".inst 0x455a10c6  // ssublb z6.h, z6.b, z26.b\n"
    "ldr x19, [x13, #0x20]\n"
    ".inst 0x455a10e7  // ssublb z7.h, z7.b, z26.b\n"
    ".inst 0x455a1108  // ssublb z8.h, z8.b, z26.b\n"
    "ld1b { z31.h }, p3/Z, [x23, x16]\n"
    "ld1b { z30.h }, p3/Z, [x22, x16]\n"
    ".inst 0x454b1bff  // usublb z31.h, z31.b, z11.b\n"
    "ld1b { z29.h }, p3/Z, [x21, x16]\n"
    ".inst 0x454b1bde  // usublb z30.h, z30.b, z11.b\n"
    "ld1b { z28.h }, p3/Z, [x20, x16]\n"
    "ld1b { z27.h }, p3/Z, [x19, x16]\n"
    ".inst 0x454b1bbd  // usublb z29.h, z29.b, z11.b\n"
    ".inst 0x454b1b9c  // usublb z28.h, z28.b, z11.b\n"
    ".inst 0x454b1b7b  // usublb z27.h, z27.b, z11.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(SVE2)
