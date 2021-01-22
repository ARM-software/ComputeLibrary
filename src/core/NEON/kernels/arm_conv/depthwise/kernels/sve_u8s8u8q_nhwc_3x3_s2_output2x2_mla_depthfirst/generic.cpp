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

void sve_u8s8u8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(
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
    const uint8_t *inptrs[25];

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
      inptrs[0] = inptrs_raw[12];
      inptrs[1] = inptrs_raw[0];
      inptrs[2] = inptrs_raw[1];
      inptrs[3] = inptrs_raw[3];
      inptrs[4] = inptrs_raw[4];
      inptrs[5] = inptrs_raw[5];
      inptrs[6] = inptrs_raw[6];
      inptrs[7] = inptrs_raw[2];
      inptrs[8] = inptrs_raw[8];
      inptrs[9] = inptrs_raw[9];
      inptrs[10] = inptrs_raw[7];
      inptrs[11] = inptrs_raw[15];
      inptrs[12] = inptrs_raw[10];
      inptrs[13] = inptrs_raw[16];
      inptrs[14] = inptrs_raw[11];
      inptrs[15] = inptrs_raw[18];
      inptrs[16] = inptrs_raw[13];
      inptrs[17] = inptrs_raw[19];
      inptrs[18] = inptrs_raw[20];
      inptrs[19] = inptrs_raw[14];
      inptrs[20] = inptrs_raw[21];
      inptrs[21] = inptrs_raw[17];
      inptrs[22] = inptrs_raw[23];
      inptrs[23] = inptrs_raw[22];
      inptrs[24] = inptrs_raw[24];

    }
  };

  const Params params(n_channels, inptrs, weights, bias, qp,
                      requant_muls, requant_shifts, outptrs);

  __asm__ __volatile__(
    "ldr x4, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ptrue p4.b\n"
    "ldr x5, [%x[params], %[offsetof_Params_weights]]\n"
    "mov x6, #0x0\n"
    "ldr x22, [%x[params], %[offsetof_Params_requant]]\n"
    "mov x7, #0x0\n"
    "ldr x8, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "add x17, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x16, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x19, x22, %[offsetof_Requantize32_a_offset]\n"
    "ldr x21, [%x[params], %[offsetof_Params_outptrs]]\n"
    "add x20, x22, %[offsetof_Requantize32_b_offset]\n"
    "ld1rb { z16.b }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z12.b }, p4/Z, [x20]\n"
    "add x20, x22, %[offsetof_Requantize32_minval]\n"
    "ld1rw { z14.s }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_maxval]\n"
    "ld1rw { z17.s }, p4/Z, [x20]\n"
    "whilelt p3.h, x6, x4\n"
    "ld1rw { z15.s }, p4/Z, [x19]\n"
    "whilelt p2.s, x6, x4\n"
    "ldp x15, x14, [x21, #0x0]\n"
    "mov x19, x6\n"
    "incw x19\n"
    "ldp x13, x12, [x21, #0x10]\n"
    "whilelt p1.s, x19, x4\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z20.s }, p2/Z, [x19]\n"
    "ld1w { z10.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z13.s, z20.s, z10.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z20.s, z20.s, z10.s\n"
    "mov z11.d, z13.d\n"
    "ld1sb { z0.h }, p4/Z, [x5]\n"
    "mov z9.d, z13.d\n"
    "ld1sb { z1.h }, p4/Z, [x5, #1, MUL VL]\n"
    "mov z18.d, z20.d\n"
    "ld1sb { z2.h }, p4/Z, [x5, #2, MUL VL]\n"
    "mov z19.d, z20.d\n"
    "ld1sb { z3.h }, p4/Z, [x5, #3, MUL VL]\n"
    "mov z23.d, z13.d\n"
    "ld1sb { z4.h }, p4/Z, [x5, #4, MUL VL]\n"
    "mov z21.d, z20.d\n"
    "ld1sb { z5.h }, p4/Z, [x5, #5, MUL VL]\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    "ld1sb { z6.h }, p4/Z, [x5, #6, MUL VL]\n"
    ".inst 0x454c1021  // ssublb z1.h, z1.b, z12.b\n"
    "ld1sb { z7.h }, p4/Z, [x5, #7, MUL VL]\n"
    "inch x5, ALL, MUL #8\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    "ld1sb { z8.h }, p4/Z, [x5]\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ldp x26, x25, [x17, #0x0]\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    "ldp x24, x23, [x17, #0x10]\n"
    ".inst 0x454c10a5  // ssublb z5.h, z5.b, z12.b\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    "ldp x22, x21, [x17, #0x20]\n"
    ".inst 0x454c10e7  // ssublb z7.h, z7.b, z12.b\n"
    ".inst 0x454c1108  // ssublb z8.h, z8.b, z12.b\n"
    "ldp x20, x19, [x17, #0x30]\n"
    "ld1b { z31.h }, p3/Z, [x26, x6]\n"
    ".inst 0x45501bff  // usublb z31.h, z31.b, z16.b\n"
    "ld1b { z30.h }, p3/Z, [x25, x6]\n"
    "ld1b { z29.h }, p3/Z, [x24, x6]\n"
    ".inst 0x45501bde  // usublb z30.h, z30.b, z16.b\n"
    "ld1b { z28.h }, p3/Z, [x23, x6]\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    "ld1b { z27.h }, p3/Z, [x22, x6]\n"
    "ld1b { z26.h }, p3/Z, [x21, x6]\n"
    ".inst 0x45501b9c  // usublb z28.h, z28.b, z16.b\n"
    "ld1b { z25.h }, p3/Z, [x20, x6]\n"
    "ld1b { z24.h }, p3/Z, [x19, x6]\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    ".inst 0x45501b5a  // usublb z26.h, z26.b, z16.b\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    ".inst 0x45501b18  // usublb z24.h, z24.b, z16.b\n"
    "1:"  // Loop
    ".inst 0x448843ed  // smlalb z13.s, p4/M, z31.h, z8.h\n"
    "ldr x22, [x17, #0x40]\n"
    "whilelt p0.h, x7, x4\n"
    ".inst 0x448847f4  // smlalt z20.s, p4/M, z31.h, z8.h\n"
    "ldr x21, [x17, #0x48]\n"
    "inch x5\n"
    ".inst 0x448643eb  // smlalb z11.s, p4/M, z31.h, z6.h\n"
    "ldr x20, [x17, #0x50]\n"
    ".inst 0x448647f2  // smlalt z18.s, p4/M, z31.h, z6.h\n"
    "ldr x19, [x17, #0x58]\n"
    ".inst 0x448243e9  // smlalb z9.s, p4/M, z31.h, z2.h\n"
    "ldr x11, [x17, #0x60]\n"
    ".inst 0x448247f3  // smlalt z19.s, p4/M, z31.h, z2.h\n"
    "ldr x10, [x17, #0x68]\n"
    ".inst 0x448043f7  // smlalb z23.s, p4/M, z31.h, z0.h\n"
    "ldr x9, [x17, #0x70]\n"
    ".inst 0x448047f5  // smlalt z21.s, p4/M, z31.h, z0.h\n"
    "ldr x28, [x17, #0x78]\n"
    ".inst 0x448043cd  // smlalb z13.s, p4/M, z30.h, z0.h\n"
    "ldr x27, [x17, #0x80]\n"
    ".inst 0x448047d4  // smlalt z20.s, p4/M, z30.h, z0.h\n"
    "ldr x26, [x17, #0x88]\n"
    ".inst 0x4481438b  // smlalb z11.s, p4/M, z28.h, z1.h\n"
    "ldr x25, [x17, #0x90]\n"
    ".inst 0x44814792  // smlalt z18.s, p4/M, z28.h, z1.h\n"
    "ld1b { z28.h }, p3/Z, [x21, x6]\n"
    ".inst 0x448143ad  // smlalb z13.s, p4/M, z29.h, z1.h\n"
    "ldr x24, [x17, #0x98]\n"
    ".inst 0x448147b4  // smlalt z20.s, p4/M, z29.h, z1.h\n"
    "ld1b { z29.h }, p3/Z, [x22, x6]\n"
    ".inst 0x4482436b  // smlalb z11.s, p4/M, z27.h, z2.h\n"
    "ldr x23, [x17, #0xa0]\n"
    ".inst 0x45501b9c  // usublb z28.h, z28.b, z16.b\n"
    "ldr x22, [x17, #0xa8]\n"
    ".inst 0x44824772  // smlalt z18.s, p4/M, z27.h, z2.h\n"
    "ld1b { z27.h }, p3/Z, [x20, x6]\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    "ldr x21, [x17, #0xb0]\n"
    ".inst 0x4483434d  // smlalb z13.s, p4/M, z26.h, z3.h\n"
    "ldr x20, [x17, #0xb8]\n"
    ".inst 0x44834754  // smlalt z20.s, p4/M, z26.h, z3.h\n"
    "ld1b { z26.h }, p3/Z, [x19, x6]\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    "ldr x19, [x17, #0xc0]\n"
    ".inst 0x4480430b  // smlalb z11.s, p4/M, z24.h, z0.h\n"
    "ld1w { z10.s }, p2/Z, [x8]\n"
    ".inst 0x4484432d  // smlalb z13.s, p4/M, z25.h, z4.h\n"
    "ld1w { z22.s }, p1/Z, [x8, #1, MUL VL]\n"
    "addvl x8, x8, #2\n"
    ".inst 0x45501b5a  // usublb z26.h, z26.b, z16.b\n"
    ".inst 0x44844734  // smlalt z20.s, p4/M, z25.h, z4.h\n"
    "ld1b { z25.h }, p3/Z, [x11, x6]\n"
    ".inst 0x44804712  // smlalt z18.s, p4/M, z24.h, z0.h\n"
    "uzp1 z31.s, z10.s, z22.s\n"
    "uzp2 z30.s, z10.s, z22.s\n"
    "ld1w { z10.s }, p2/Z, [x16]\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    "ld1w { z22.s }, p1/Z, [x16, #1, MUL VL]\n"
    "addvl x16, x16, #2\n"
    ".inst 0x4482430d  // smlalb z13.s, p4/M, z24.h, z2.h\n"
    ".inst 0x44824714  // smlalt z20.s, p4/M, z24.h, z2.h\n"
    "ld1b { z24.h }, p3/Z, [x9, x6]\n"
    ".inst 0x448443ab  // smlalb z11.s, p4/M, z29.h, z4.h\n"
    ".inst 0x448447b2  // smlalt z18.s, p4/M, z29.h, z4.h\n"
    "ld1b { z29.h }, p3/Z, [x10, x6]\n"
    ".inst 0x44834349  // smlalb z9.s, p4/M, z26.h, z3.h\n"
    ".inst 0x45501b18  // usublb z24.h, z24.b, z16.b\n"
    ".inst 0x4485438b  // smlalb z11.s, p4/M, z28.h, z5.h\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    ".inst 0x44854792  // smlalt z18.s, p4/M, z28.h, z5.h\n"
    "ld1b { z28.h }, p3/Z, [x27, x6]\n"
    ".inst 0x4485436d  // smlalb z13.s, p4/M, z27.h, z5.h\n"
    ".inst 0x44854774  // smlalt z20.s, p4/M, z27.h, z5.h\n"
    ".inst 0x4483436b  // smlalb z11.s, p4/M, z27.h, z3.h\n"
    ".inst 0x45501b9c  // usublb z28.h, z28.b, z16.b\n"
    ".inst 0x44834772  // smlalt z18.s, p4/M, z27.h, z3.h\n"
    "ld1b { z27.h }, p3/Z, [x28, x6]\n"
    ".inst 0x44834753  // smlalt z19.s, p4/M, z26.h, z3.h\n"
    "ld1b { z26.h }, p3/Z, [x26, x6]\n"
    ".inst 0x4486432d  // smlalb z13.s, p4/M, z25.h, z6.h\n"
    ".inst 0x44864734  // smlalt z20.s, p4/M, z25.h, z6.h\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    ".inst 0x45501b5a  // usublb z26.h, z26.b, z16.b\n"
    ".inst 0x44804329  // smlalb z9.s, p4/M, z25.h, z0.h\n"
    ".inst 0x44804733  // smlalt z19.s, p4/M, z25.h, z0.h\n"
    "ld1b { z25.h }, p3/Z, [x25, x6]\n"
    "uzp1 z0.s, z10.s, z22.s\n"
    "uzp2 z22.s, z10.s, z22.s\n"
    ".inst 0x448443a9  // smlalb z9.s, p4/M, z29.h, z4.h\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    ".inst 0x448447b3  // smlalt z19.s, p4/M, z29.h, z4.h\n"
    "ld1b { z29.h }, p3/Z, [x24, x6]\n"
    ".inst 0x4487430d  // smlalb z13.s, p4/M, z24.h, z7.h\n"
    ".inst 0x44874714  // smlalt z20.s, p4/M, z24.h, z7.h\n"
    ".inst 0x44814309  // smlalb z9.s, p4/M, z24.h, z1.h\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    ".inst 0x04bf75ad  // sqrdmulh z13.s, z13.s, z31.s\n"
    ".inst 0x04be7694  // sqrdmulh z20.s, z20.s, z30.s\n"
    ".inst 0x44814713  // smlalt z19.s, p4/M, z24.h, z1.h\n"
    "ld1b { z24.h }, p3/Z, [x22, x6]\n"
    ".inst 0x44844377  // smlalb z23.s, p4/M, z27.h, z4.h\n"
    "and z10.d, z13.d, z0.d\n"
    ".inst 0x44844775  // smlalt z21.s, p4/M, z27.h, z4.h\n"
    "ld1b { z27.h }, p3/Z, [x23, x6]\n"
    ".inst 0x45501b18  // usublb z24.h, z24.b, z16.b\n"
    "asr z10.s, z10.s, #0x1f\n"
    "and z4.d, z20.d, z22.d\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    "sqadd z13.s, z13.s, z10.s\n"
    "asr z4.s, z4.s, #0x1f\n"
    ".inst 0x4487438b  // smlalb z11.s, p4/M, z28.h, z7.h\n"
    ".inst 0x44874792  // smlalt z18.s, p4/M, z28.h, z7.h\n"
    "sqadd z20.s, z20.s, z4.s\n"
    ".inst 0x44814397  // smlalb z23.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44814795  // smlalt z21.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44864329  // smlalb z9.s, p4/M, z25.h, z6.h\n"
    ".inst 0x44864733  // smlalt z19.s, p4/M, z25.h, z6.h\n"
    "ld1b { z25.h }, p3/Z, [x20, x6]\n"
    ".inst 0x44854357  // smlalb z23.s, p4/M, z26.h, z5.h\n"
    ".inst 0x44854755  // smlalt z21.s, p4/M, z26.h, z5.h\n"
    "ld1b { z26.h }, p3/Z, [x21, x6]\n"
    ".inst 0x448843ab  // smlalb z11.s, p4/M, z29.h, z8.h\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    ".inst 0x448847b2  // smlalt z18.s, p4/M, z29.h, z8.h\n"
    ".inst 0x45501b5a  // usublb z26.h, z26.b, z16.b\n"
    ".inst 0x04bf756b  // sqrdmulh z11.s, z11.s, z31.s\n"
    ".inst 0x448243b7  // smlalb z23.s, p4/M, z29.h, z2.h\n"
    ".inst 0x04be7652  // sqrdmulh z18.s, z18.s, z30.s\n"
    ".inst 0x448247b5  // smlalt z21.s, p4/M, z29.h, z2.h\n"
    "ld1b { z29.h }, p3/Z, [x19, x6]\n"
    "inch x6\n"
    "and z2.d, z11.d, z0.d\n"
    "whilelt p2.s, x6, x4\n"
    ".inst 0x44874369  // smlalb z9.s, p4/M, z27.h, z7.h\n"
    "mov x19, x6\n"
    "and z10.d, z18.d, z22.d\n"
    "incw x19\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    "whilelt p1.s, x19, x4\n"
    "asr z2.s, z2.s, #0x1f\n"
    "whilelt p3.h, x6, x4\n"
    "asr z10.s, z10.s, #0x1f\n"
    ".inst 0x44874773  // smlalt z19.s, p4/M, z27.h, z7.h\n"
    "sqadd z11.s, z11.s, z2.s\n"
    "sqadd z18.s, z18.s, z10.s\n"
    ".inst 0x44854309  // smlalb z9.s, p4/M, z24.h, z5.h\n"
    ".inst 0x44854713  // smlalt z19.s, p4/M, z24.h, z5.h\n"
    ".inst 0x44834317  // smlalb z23.s, p4/M, z24.h, z3.h\n"
    ".inst 0x44834715  // smlalt z21.s, p4/M, z24.h, z3.h\n"
    ".inst 0x44884329  // smlalb z9.s, p4/M, z25.h, z8.h\n"
    ".inst 0x44884733  // smlalt z19.s, p4/M, z25.h, z8.h\n"
    ".inst 0x44874357  // smlalb z23.s, p4/M, z26.h, z7.h\n"
    ".inst 0x44874755  // smlalt z21.s, p4/M, z26.h, z7.h\n"
    ".inst 0x04bf7529  // sqrdmulh z9.s, z9.s, z31.s\n"
    ".inst 0x04be7673  // sqrdmulh z19.s, z19.s, z30.s\n"
    ".inst 0x44864337  // smlalb z23.s, p4/M, z25.h, z6.h\n"
    ".inst 0x44864735  // smlalt z21.s, p4/M, z25.h, z6.h\n"
    "and z10.d, z9.d, z0.d\n"
    "and z24.d, z19.d, z22.d\n"
    ".inst 0x448843b7  // smlalb z23.s, p4/M, z29.h, z8.h\n"
    "asr z10.s, z10.s, #0x1f\n"
    "asr z24.s, z24.s, #0x1f\n"
    ".inst 0x448847b5  // smlalt z21.s, p4/M, z29.h, z8.h\n"
    "sqadd z9.s, z9.s, z10.s\n"
    "sqadd z19.s, z19.s, z24.s\n"
    ".inst 0x04bf76f7  // sqrdmulh z23.s, z23.s, z31.s\n"
    ".inst 0x04be76b5  // sqrdmulh z21.s, z21.s, z30.s\n"
    ".inst 0x4482900d  // srshl z13.s, p4/M, z13.s, z0.s\n"
    ".inst 0x448292d4  // srshl z20.s, p4/M, z20.s, z22.s\n"
    "and z30.d, z23.d, z0.d\n"
    "and z28.d, z21.d, z22.d\n"
    "add z13.s, z13.s, z14.s\n"
    "add z20.s, z20.s, z14.s\n"
    "asr z30.s, z30.s, #0x1f\n"
    "asr z28.s, z28.s, #0x1f\n"
    "smin z13.s, p4/M, z13.s, z15.s\n"
    "sqadd z23.s, z23.s, z30.s\n"
    "sqadd z21.s, z21.s, z28.s\n"
    "smin z20.s, p4/M, z20.s, z15.s\n"
    "smax z13.s, p4/M, z13.s, z17.s\n"
    ".inst 0x4482900b  // srshl z11.s, p4/M, z11.s, z0.s\n"
    ".inst 0x448292d2  // srshl z18.s, p4/M, z18.s, z22.s\n"
    "smax z20.s, p4/M, z20.s, z17.s\n"
    ".inst 0x44829009  // srshl z9.s, p4/M, z9.s, z0.s\n"
    "add z11.s, z11.s, z14.s\n"
    "add z18.s, z18.s, z14.s\n"
    "trn1 z13.h, z13.h, z20.h\n"
    "st1b { z13.h }, p0, [x15, x7]\n"
    "add z9.s, z9.s, z14.s\n"
    "smin z11.s, p4/M, z11.s, z15.s\n"
    "smin z18.s, p4/M, z18.s, z15.s\n"
    ".inst 0x448292d3  // srshl z19.s, p4/M, z19.s, z22.s\n"
    "smin z9.s, p4/M, z9.s, z15.s\n"
    "smax z11.s, p4/M, z11.s, z17.s\n"
    "smax z18.s, p4/M, z18.s, z17.s\n"
    "add z19.s, z19.s, z14.s\n"
    "smax z9.s, p4/M, z9.s, z17.s\n"
    ".inst 0x44829017  // srshl z23.s, p4/M, z23.s, z0.s\n"
    "trn1 z11.h, z11.h, z18.h\n"
    "st1b { z11.h }, p0, [x14, x7]\n"
    "smin z19.s, p4/M, z19.s, z15.s\n"
    ".inst 0x448292d5  // srshl z21.s, p4/M, z21.s, z22.s\n"
    "add z23.s, z23.s, z14.s\n"
    "add z21.s, z21.s, z14.s\n"
    "smax z19.s, p4/M, z19.s, z17.s\n"
    "smin z23.s, p4/M, z23.s, z15.s\n"
    "smin z21.s, p4/M, z21.s, z15.s\n"
    "trn1 z9.h, z9.h, z19.h\n"
    "st1b { z9.h }, p0, [x13, x7]\n"
    "smax z23.s, p4/M, z23.s, z17.s\n"
    "smax z21.s, p4/M, z21.s, z17.s\n"
    "trn1 z23.h, z23.h, z21.h\n"
    "st1b { z23.h }, p0, [x12, x7]\n"
    "inch x7\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z20.s }, p2/Z, [x19]\n"
    "ld1w { z10.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z13.s, z20.s, z10.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z20.s, z20.s, z10.s\n"
    "mov z11.d, z13.d\n"
    "ld1sb { z0.h }, p4/Z, [x5]\n"
    "mov z9.d, z13.d\n"
    "ld1sb { z1.h }, p4/Z, [x5, #1, MUL VL]\n"
    "mov z18.d, z20.d\n"
    "ld1sb { z2.h }, p4/Z, [x5, #2, MUL VL]\n"
    "mov z19.d, z20.d\n"
    "ld1sb { z3.h }, p4/Z, [x5, #3, MUL VL]\n"
    "mov z23.d, z13.d\n"
    "ld1sb { z4.h }, p4/Z, [x5, #4, MUL VL]\n"
    "mov z21.d, z20.d\n"
    "ld1sb { z5.h }, p4/Z, [x5, #5, MUL VL]\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    "ld1sb { z6.h }, p4/Z, [x5, #6, MUL VL]\n"
    ".inst 0x454c1021  // ssublb z1.h, z1.b, z12.b\n"
    "ld1sb { z7.h }, p4/Z, [x5, #7, MUL VL]\n"
    "inch x5, ALL, MUL #8\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    "ld1sb { z8.h }, p4/Z, [x5]\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ldp x26, x25, [x17, #0x0]\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    "ldp x24, x23, [x17, #0x10]\n"
    ".inst 0x454c10a5  // ssublb z5.h, z5.b, z12.b\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    "ldp x22, x21, [x17, #0x20]\n"
    ".inst 0x454c10e7  // ssublb z7.h, z7.b, z12.b\n"
    ".inst 0x454c1108  // ssublb z8.h, z8.b, z12.b\n"
    "ldp x20, x19, [x17, #0x30]\n"
    "ld1b { z31.h }, p3/Z, [x26, x6]\n"
    ".inst 0x45501bff  // usublb z31.h, z31.b, z16.b\n"
    "ld1b { z30.h }, p3/Z, [x25, x6]\n"
    "ld1b { z29.h }, p3/Z, [x24, x6]\n"
    ".inst 0x45501bde  // usublb z30.h, z30.b, z16.b\n"
    "ld1b { z28.h }, p3/Z, [x23, x6]\n"
    ".inst 0x45501bbd  // usublb z29.h, z29.b, z16.b\n"
    "ld1b { z27.h }, p3/Z, [x22, x6]\n"
    "ld1b { z26.h }, p3/Z, [x21, x6]\n"
    ".inst 0x45501b9c  // usublb z28.h, z28.b, z16.b\n"
    "ld1b { z25.h }, p3/Z, [x20, x6]\n"
    "ld1b { z24.h }, p3/Z, [x19, x6]\n"
    ".inst 0x45501b7b  // usublb z27.h, z27.b, z16.b\n"
    ".inst 0x45501b5a  // usublb z26.h, z26.b, z16.b\n"
    ".inst 0x45501b39  // usublb z25.h, z25.b, z16.b\n"
    ".inst 0x45501b18  // usublb z24.h, z24.b, z16.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(SVE2)
