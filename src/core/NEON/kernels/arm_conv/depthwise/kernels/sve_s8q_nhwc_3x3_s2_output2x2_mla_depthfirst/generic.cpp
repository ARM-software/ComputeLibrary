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

void sve_s8q_nhwc_3x3_s2_output2x2_mla_depthfirst_impl(
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
    const int8_t *inptrs[25];

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
    "ldr x5, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ptrue p4.b\n"
    "ldr x6, [%x[params], %[offsetof_Params_weights]]\n"
    "mov x7, #0x0\n"
    "ldr x22, [%x[params], %[offsetof_Params_requant]]\n"
    "mov x8, #0x0\n"
    "ldr x17, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "add x16, %x[params], %[offsetof_Params_inptrs]\n"
    "ldr x15, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x19, x22, %[offsetof_Requantize32_a_offset]\n"
    "ldr x21, [%x[params], %[offsetof_Params_outptrs]]\n"
    "add x20, x22, %[offsetof_Requantize32_b_offset]\n"
    "ld1rb { z19.b }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z12.b }, p4/Z, [x20]\n"
    "add x20, x22, %[offsetof_Requantize32_minval]\n"
    "ld1rw { z14.s }, p4/Z, [x19]\n"
    "add x19, x22, %[offsetof_Requantize32_maxval]\n"
    "ld1rw { z20.s }, p4/Z, [x20]\n"
    "whilelt p3.h, x7, x5\n"
    "ld1rw { z15.s }, p4/Z, [x19]\n"
    "whilelt p2.s, x7, x5\n"
    "ldp x14, x13, [x21, #0x0]\n"
    "mov x19, x7\n"
    "incw x19\n"
    "ldp x12, x11, [x21, #0x10]\n"
    "whilelt p1.s, x19, x5\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z18.s }, p2/Z, [x19]\n"
    "ld1w { z16.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z13.s, z18.s, z16.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z16.s, z18.s, z16.s\n"
    "mov z11.d, z13.d\n"
    "ld1sb { z0.h }, p4/Z, [x6]\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    "mov z9.d, z16.d\n"
    "ld1sb { z1.h }, p4/Z, [x6, #1, MUL VL]\n"
    "mov z18.d, z13.d\n"
    "ld1sb { z2.h }, p4/Z, [x6, #2, MUL VL]\n"
    ".inst 0x454c1021  // ssublb z1.h, z1.b, z12.b\n"
    "mov z10.d, z16.d\n"
    "ld1sb { z3.h }, p4/Z, [x6, #3, MUL VL]\n"
    "mov z22.d, z13.d\n"
    "ld1sb { z4.h }, p4/Z, [x6, #4, MUL VL]\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    "mov z23.d, z16.d\n"
    "ld1sb { z5.h }, p4/Z, [x6, #5, MUL VL]\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ld1sb { z6.h }, p4/Z, [x6, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x6, #7, MUL VL]\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    "inch x6, ALL, MUL #8\n"
    "ld1sb { z8.h }, p4/Z, [x6]\n"
    "ldp x26, x25, [x16, #0x0]\n"
    ".inst 0x454c10a5  // ssublb z5.h, z5.b, z12.b\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    "ldp x24, x23, [x16, #0x10]\n"
    ".inst 0x454c10e7  // ssublb z7.h, z7.b, z12.b\n"
    ".inst 0x454c1108  // ssublb z8.h, z8.b, z12.b\n"
    "ldp x22, x21, [x16, #0x20]\n"
    "ldp x20, x19, [x16, #0x30]\n"
    "ld1sb { z31.h }, p3/Z, [x26, x7]\n"
    ".inst 0x455313ff  // ssublb z31.h, z31.b, z19.b\n"
    "ld1sb { z30.h }, p3/Z, [x25, x7]\n"
    "ld1sb { z29.h }, p3/Z, [x24, x7]\n"
    ".inst 0x455313de  // ssublb z30.h, z30.b, z19.b\n"
    "ld1sb { z28.h }, p3/Z, [x23, x7]\n"
    "ld1sb { z27.h }, p3/Z, [x22, x7]\n"
    ".inst 0x455313bd  // ssublb z29.h, z29.b, z19.b\n"
    "ld1sb { z26.h }, p3/Z, [x21, x7]\n"
    ".inst 0x4553139c  // ssublb z28.h, z28.b, z19.b\n"
    "ld1sb { z25.h }, p3/Z, [x20, x7]\n"
    "ld1sb { z24.h }, p3/Z, [x19, x7]\n"
    ".inst 0x4553137b  // ssublb z27.h, z27.b, z19.b\n"
    ".inst 0x4553135a  // ssublb z26.h, z26.b, z19.b\n"
    ".inst 0x45531339  // ssublb z25.h, z25.b, z19.b\n"
    ".inst 0x45531318  // ssublb z24.h, z24.b, z19.b\n"
    "1:"  // Loop
    ".inst 0x448843ed  // smlalb z13.s, p4/M, z31.h, z8.h\n"
    "ldr x23, [x16, #0x40]\n"
    "whilelt p0.h, x8, x5\n"
    ".inst 0x448847f0  // smlalt z16.s, p4/M, z31.h, z8.h\n"
    "ldr x22, [x16, #0x48]\n"
    "inch x6\n"
    ".inst 0x448643eb  // smlalb z11.s, p4/M, z31.h, z6.h\n"
    "ldr x21, [x16, #0x50]\n"
    ".inst 0x448647e9  // smlalt z9.s, p4/M, z31.h, z6.h\n"
    "ldr x20, [x16, #0x58]\n"
    ".inst 0x448243f2  // smlalb z18.s, p4/M, z31.h, z2.h\n"
    "ldr x19, [x16, #0x60]\n"
    ".inst 0x448247ea  // smlalt z10.s, p4/M, z31.h, z2.h\n"
    "ldr x10, [x16, #0x68]\n"
    ".inst 0x448043f6  // smlalb z22.s, p4/M, z31.h, z0.h\n"
    "ldr x9, [x16, #0x70]\n"
    ".inst 0x448047f7  // smlalt z23.s, p4/M, z31.h, z0.h\n"
    "ldr x28, [x16, #0x78]\n"
    ".inst 0x448043cd  // smlalb z13.s, p4/M, z30.h, z0.h\n"
    "ldr x27, [x16, #0x80]\n"
    ".inst 0x448047d0  // smlalt z16.s, p4/M, z30.h, z0.h\n"
    "ldr x26, [x16, #0x88]\n"
    ".inst 0x4481438b  // smlalb z11.s, p4/M, z28.h, z1.h\n"
    "ldr x25, [x16, #0x90]\n"
    ".inst 0x44814789  // smlalt z9.s, p4/M, z28.h, z1.h\n"
    "ld1sb { z28.h }, p3/Z, [x22, x7]\n"
    ".inst 0x4553139c  // ssublb z28.h, z28.b, z19.b\n"
    ".inst 0x448143ad  // smlalb z13.s, p4/M, z29.h, z1.h\n"
    "ldr x24, [x16, #0x98]\n"
    ".inst 0x448147b0  // smlalt z16.s, p4/M, z29.h, z1.h\n"
    "ld1sb { z29.h }, p3/Z, [x23, x7]\n"
    ".inst 0x455313bd  // ssublb z29.h, z29.b, z19.b\n"
    ".inst 0x4482436b  // smlalb z11.s, p4/M, z27.h, z2.h\n"
    "ldr x23, [x16, #0xa0]\n"
    ".inst 0x44824769  // smlalt z9.s, p4/M, z27.h, z2.h\n"
    "ld1sb { z27.h }, p3/Z, [x21, x7]\n"
    ".inst 0x4553137b  // ssublb z27.h, z27.b, z19.b\n"
    ".inst 0x4483434d  // smlalb z13.s, p4/M, z26.h, z3.h\n"
    "ldr x22, [x16, #0xa8]\n"
    ".inst 0x44834750  // smlalt z16.s, p4/M, z26.h, z3.h\n"
    "ld1sb { z26.h }, p3/Z, [x20, x7]\n"
    ".inst 0x4553135a  // ssublb z26.h, z26.b, z19.b\n"
    ".inst 0x4484432d  // smlalb z13.s, p4/M, z25.h, z4.h\n"
    "ldr x21, [x16, #0xb0]\n"
    ".inst 0x44844730  // smlalt z16.s, p4/M, z25.h, z4.h\n"
    "ld1sb { z25.h }, p3/Z, [x19, x7]\n"
    ".inst 0x45531339  // ssublb z25.h, z25.b, z19.b\n"
    ".inst 0x4482430d  // smlalb z13.s, p4/M, z24.h, z2.h\n"
    "ldr x20, [x16, #0xb8]\n"
    ".inst 0x44824710  // smlalt z16.s, p4/M, z24.h, z2.h\n"
    "ldr x19, [x16, #0xc0]\n"
    ".inst 0x4480430b  // smlalb z11.s, p4/M, z24.h, z0.h\n"
    "ld1w { z21.s }, p2/Z, [x17]\n"
    ".inst 0x44804709  // smlalt z9.s, p4/M, z24.h, z0.h\n"
    "ld1sb { z24.h }, p3/Z, [x9, x7]\n"
    ".inst 0x45531318  // ssublb z24.h, z24.b, z19.b\n"
    ".inst 0x448443ab  // smlalb z11.s, p4/M, z29.h, z4.h\n"
    "ld1w { z17.s }, p1/Z, [x17, #1, MUL VL]\n"
    ".inst 0x448447a9  // smlalt z9.s, p4/M, z29.h, z4.h\n"
    "ld1sb { z29.h }, p3/Z, [x10, x7]\n"
    "addvl x17, x17, #2\n"
    ".inst 0x4485436d  // smlalb z13.s, p4/M, z27.h, z5.h\n"
    ".inst 0x455313bd  // ssublb z29.h, z29.b, z19.b\n"
    "uzp1 z30.s, z21.s, z17.s\n"
    "uzp2 z31.s, z21.s, z17.s\n"
    "ld1w { z21.s }, p2/Z, [x15]\n"
    ".inst 0x4485438b  // smlalb z11.s, p4/M, z28.h, z5.h\n"
    "ld1w { z17.s }, p1/Z, [x15, #1, MUL VL]\n"
    "addvl x15, x15, #2\n"
    ".inst 0x44854789  // smlalt z9.s, p4/M, z28.h, z5.h\n"
    "ld1sb { z28.h }, p3/Z, [x27, x7]\n"
    ".inst 0x4553139c  // ssublb z28.h, z28.b, z19.b\n"
    ".inst 0x44854770  // smlalt z16.s, p4/M, z27.h, z5.h\n"
    ".inst 0x4483436b  // smlalb z11.s, p4/M, z27.h, z3.h\n"
    ".inst 0x44834769  // smlalt z9.s, p4/M, z27.h, z3.h\n"
    "ld1sb { z27.h }, p3/Z, [x28, x7]\n"
    ".inst 0x4553137b  // ssublb z27.h, z27.b, z19.b\n"
    ".inst 0x44834352  // smlalb z18.s, p4/M, z26.h, z3.h\n"
    ".inst 0x4483474a  // smlalt z10.s, p4/M, z26.h, z3.h\n"
    "ld1sb { z26.h }, p3/Z, [x26, x7]\n"
    ".inst 0x4553135a  // ssublb z26.h, z26.b, z19.b\n"
    ".inst 0x4486432d  // smlalb z13.s, p4/M, z25.h, z6.h\n"
    ".inst 0x44864730  // smlalt z16.s, p4/M, z25.h, z6.h\n"
    ".inst 0x44804332  // smlalb z18.s, p4/M, z25.h, z0.h\n"
    ".inst 0x4480472a  // smlalt z10.s, p4/M, z25.h, z0.h\n"
    "ld1sb { z25.h }, p3/Z, [x25, x7]\n"
    ".inst 0x45531339  // ssublb z25.h, z25.b, z19.b\n"
    "uzp1 z0.s, z21.s, z17.s\n"
    "uzp2 z21.s, z21.s, z17.s\n"
    ".inst 0x448443b2  // smlalb z18.s, p4/M, z29.h, z4.h\n"
    ".inst 0x448447aa  // smlalt z10.s, p4/M, z29.h, z4.h\n"
    "ld1sb { z29.h }, p3/Z, [x24, x7]\n"
    ".inst 0x455313bd  // ssublb z29.h, z29.b, z19.b\n"
    ".inst 0x4487430d  // smlalb z13.s, p4/M, z24.h, z7.h\n"
    ".inst 0x44874710  // smlalt z16.s, p4/M, z24.h, z7.h\n"
    ".inst 0x44814312  // smlalb z18.s, p4/M, z24.h, z1.h\n"
    ".inst 0x4481470a  // smlalt z10.s, p4/M, z24.h, z1.h\n"
    "ld1sb { z24.h }, p3/Z, [x22, x7]\n"
    ".inst 0x45531318  // ssublb z24.h, z24.b, z19.b\n"
    ".inst 0x04be75ad  // sqrdmulh z13.s, z13.s, z30.s\n"
    ".inst 0x04bf7610  // sqrdmulh z16.s, z16.s, z31.s\n"
    ".inst 0x44844376  // smlalb z22.s, p4/M, z27.h, z4.h\n"
    ".inst 0x44844777  // smlalt z23.s, p4/M, z27.h, z4.h\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x4553137b  // ssublb z27.h, z27.b, z19.b\n"
    "and z4.d, z13.d, z0.d\n"
    "and z17.d, z16.d, z21.d\n"
    "asr z4.s, z4.s, #0x1f\n"
    ".inst 0x4487438b  // smlalb z11.s, p4/M, z28.h, z7.h\n"
    ".inst 0x44874789  // smlalt z9.s, p4/M, z28.h, z7.h\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x44814396  // smlalb z22.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44814797  // smlalt z23.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44864332  // smlalb z18.s, p4/M, z25.h, z6.h\n"
    ".inst 0x4486472a  // smlalt z10.s, p4/M, z25.h, z6.h\n"
    "ld1sb { z25.h }, p3/Z, [x20, x7]\n"
    ".inst 0x45531339  // ssublb z25.h, z25.b, z19.b\n"
    "sqadd z13.s, z13.s, z4.s\n"
    "sqadd z16.s, z16.s, z17.s\n"
    ".inst 0x44854356  // smlalb z22.s, p4/M, z26.h, z5.h\n"
    ".inst 0x44854757  // smlalt z23.s, p4/M, z26.h, z5.h\n"
    "ld1sb { z26.h }, p3/Z, [x21, x7]\n"
    ".inst 0x4553135a  // ssublb z26.h, z26.b, z19.b\n"
    ".inst 0x448843ab  // smlalb z11.s, p4/M, z29.h, z8.h\n"
    ".inst 0x448847a9  // smlalt z9.s, p4/M, z29.h, z8.h\n"
    ".inst 0x448243b6  // smlalb z22.s, p4/M, z29.h, z2.h\n"
    ".inst 0x448247b7  // smlalt z23.s, p4/M, z29.h, z2.h\n"
    "ld1sb { z29.h }, p3/Z, [x19, x7]\n"
    "inch x7\n"
    ".inst 0x04be756b  // sqrdmulh z11.s, z11.s, z30.s\n"
    "whilelt p2.s, x7, x5\n"
    ".inst 0x04bf7529  // sqrdmulh z9.s, z9.s, z31.s\n"
    "mov x19, x7\n"
    ".inst 0x44874372  // smlalb z18.s, p4/M, z27.h, z7.h\n"
    ".inst 0x455313bd  // ssublb z29.h, z29.b, z19.b\n"
    ".inst 0x4487476a  // smlalt z10.s, p4/M, z27.h, z7.h\n"
    "incw x19\n"
    ".inst 0x44834316  // smlalb z22.s, p4/M, z24.h, z3.h\n"
    "whilelt p1.s, x19, x5\n"
    "and z1.d, z11.d, z0.d\n"
    "whilelt p3.h, x7, x5\n"
    "and z17.d, z9.d, z21.d\n"
    "asr z1.s, z1.s, #0x1f\n"
    ".inst 0x44854312  // smlalb z18.s, p4/M, z24.h, z5.h\n"
    ".inst 0x4485470a  // smlalt z10.s, p4/M, z24.h, z5.h\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x44834717  // smlalt z23.s, p4/M, z24.h, z3.h\n"
    ".inst 0x44874356  // smlalb z22.s, p4/M, z26.h, z7.h\n"
    ".inst 0x4482900d  // srshl z13.s, p4/M, z13.s, z0.s\n"
    ".inst 0x44884332  // smlalb z18.s, p4/M, z25.h, z8.h\n"
    "sqadd z11.s, z11.s, z1.s\n"
    "sqadd z9.s, z9.s, z17.s\n"
    "add z13.s, z13.s, z14.s\n"
    ".inst 0x04be7652  // sqrdmulh z18.s, z18.s, z30.s\n"
    ".inst 0x44874757  // smlalt z23.s, p4/M, z26.h, z7.h\n"
    ".inst 0x4488472a  // smlalt z10.s, p4/M, z25.h, z8.h\n"
    ".inst 0x44864336  // smlalb z22.s, p4/M, z25.h, z6.h\n"
    "and z17.d, z18.d, z0.d\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x04bf754a  // sqrdmulh z10.s, z10.s, z31.s\n"
    ".inst 0x44864737  // smlalt z23.s, p4/M, z25.h, z6.h\n"
    ".inst 0x448843b6  // smlalb z22.s, p4/M, z29.h, z8.h\n"
    "smin z13.s, p4/M, z13.s, z15.s\n"
    ".inst 0x448292b0  // srshl z16.s, p4/M, z16.s, z21.s\n"
    "and z1.d, z10.d, z21.d\n"
    "asr z1.s, z1.s, #0x1f\n"
    "add z16.s, z16.s, z14.s\n"
    "sqadd z18.s, z18.s, z17.s\n"
    ".inst 0x04be76d6  // sqrdmulh z22.s, z22.s, z30.s\n"
    ".inst 0x448847b7  // smlalt z23.s, p4/M, z29.h, z8.h\n"
    "smax z13.s, p4/M, z13.s, z20.s\n"
    "smin z16.s, p4/M, z16.s, z15.s\n"
    "sqadd z10.s, z10.s, z1.s\n"
    "and z2.d, z22.d, z0.d\n"
    "asr z2.s, z2.s, #0x1f\n"
    ".inst 0x04bf76f7  // sqrdmulh z23.s, z23.s, z31.s\n"
    "smax z16.s, p4/M, z16.s, z20.s\n"
    ".inst 0x4482900b  // srshl z11.s, p4/M, z11.s, z0.s\n"
    ".inst 0x448292a9  // srshl z9.s, p4/M, z9.s, z21.s\n"
    ".inst 0x44829012  // srshl z18.s, p4/M, z18.s, z0.s\n"
    "trn1 z13.h, z13.h, z16.h\n"
    "st1b { z13.h }, p0, [x14, x8]\n"
    "add z11.s, z11.s, z14.s\n"
    "add z9.s, z9.s, z14.s\n"
    "add z18.s, z18.s, z14.s\n"
    "sqadd z22.s, z22.s, z2.s\n"
    "and z16.d, z23.d, z21.d\n"
    "asr z16.s, z16.s, #0x1f\n"
    "smin z11.s, p4/M, z11.s, z15.s\n"
    "smin z9.s, p4/M, z9.s, z15.s\n"
    "smin z18.s, p4/M, z18.s, z15.s\n"
    ".inst 0x448292aa  // srshl z10.s, p4/M, z10.s, z21.s\n"
    ".inst 0x44829016  // srshl z22.s, p4/M, z22.s, z0.s\n"
    "smax z11.s, p4/M, z11.s, z20.s\n"
    "sqadd z23.s, z23.s, z16.s\n"
    "add z10.s, z10.s, z14.s\n"
    "add z22.s, z22.s, z14.s\n"
    "smax z9.s, p4/M, z9.s, z20.s\n"
    "smax z18.s, p4/M, z18.s, z20.s\n"
    "smin z10.s, p4/M, z10.s, z15.s\n"
    "smin z22.s, p4/M, z22.s, z15.s\n"
    "trn1 z11.h, z11.h, z9.h\n"
    "st1b { z11.h }, p0, [x13, x8]\n"
    "smax z10.s, p4/M, z10.s, z20.s\n"
    ".inst 0x448292b7  // srshl z23.s, p4/M, z23.s, z21.s\n"
    "smax z22.s, p4/M, z22.s, z20.s\n"
    "trn1 z18.h, z18.h, z10.h\n"
    "st1b { z18.h }, p0, [x12, x8]\n"
    "add z23.s, z23.s, z14.s\n"
    "smin z23.s, p4/M, z23.s, z15.s\n"
    "smax z23.s, p4/M, z23.s, z20.s\n"
    "trn1 z22.h, z22.h, z23.h\n"
    "st1b { z22.h }, p0, [x11, x8]\n"
    "inch x8\n"
    "ldr x19, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1w { z18.s }, p2/Z, [x19]\n"
    "ld1w { z16.s }, p1/Z, [x19, #1, MUL VL]\n"
    "uzp1 z13.s, z18.s, z16.s\n"
    "addvl x19, x19, #2\n"
    "str x19, [%x[params], %[offsetof_Params_bias]]\n"
    "uzp2 z16.s, z18.s, z16.s\n"
    "mov z11.d, z13.d\n"
    "ld1sb { z0.h }, p4/Z, [x6]\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    "mov z9.d, z16.d\n"
    "ld1sb { z1.h }, p4/Z, [x6, #1, MUL VL]\n"
    "mov z18.d, z13.d\n"
    "ld1sb { z2.h }, p4/Z, [x6, #2, MUL VL]\n"
    ".inst 0x454c1021  // ssublb z1.h, z1.b, z12.b\n"
    "mov z10.d, z16.d\n"
    "ld1sb { z3.h }, p4/Z, [x6, #3, MUL VL]\n"
    "mov z22.d, z13.d\n"
    "ld1sb { z4.h }, p4/Z, [x6, #4, MUL VL]\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    "mov z23.d, z16.d\n"
    "ld1sb { z5.h }, p4/Z, [x6, #5, MUL VL]\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ld1sb { z6.h }, p4/Z, [x6, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x6, #7, MUL VL]\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    "inch x6, ALL, MUL #8\n"
    "ld1sb { z8.h }, p4/Z, [x6]\n"
    "ldp x26, x25, [x16, #0x0]\n"
    ".inst 0x454c10a5  // ssublb z5.h, z5.b, z12.b\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    "ldp x24, x23, [x16, #0x10]\n"
    ".inst 0x454c10e7  // ssublb z7.h, z7.b, z12.b\n"
    ".inst 0x454c1108  // ssublb z8.h, z8.b, z12.b\n"
    "ldp x22, x21, [x16, #0x20]\n"
    "ldp x20, x19, [x16, #0x30]\n"
    "ld1sb { z31.h }, p3/Z, [x26, x7]\n"
    ".inst 0x455313ff  // ssublb z31.h, z31.b, z19.b\n"
    "ld1sb { z30.h }, p3/Z, [x25, x7]\n"
    "ld1sb { z29.h }, p3/Z, [x24, x7]\n"
    ".inst 0x455313de  // ssublb z30.h, z30.b, z19.b\n"
    "ld1sb { z28.h }, p3/Z, [x23, x7]\n"
    "ld1sb { z27.h }, p3/Z, [x22, x7]\n"
    ".inst 0x455313bd  // ssublb z29.h, z29.b, z19.b\n"
    "ld1sb { z26.h }, p3/Z, [x21, x7]\n"
    ".inst 0x4553139c  // ssublb z28.h, z28.b, z19.b\n"
    "ld1sb { z25.h }, p3/Z, [x20, x7]\n"
    "ld1sb { z24.h }, p3/Z, [x19, x7]\n"
    ".inst 0x4553137b  // ssublb z27.h, z27.b, z19.b\n"
    ".inst 0x4553135a  // ssublb z26.h, z26.b, z19.b\n"
    ".inst 0x45531339  // ssublb z25.h, z25.b, z19.b\n"
    ".inst 0x45531318  // ssublb z24.h, z24.b, z19.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE) && defined(ARM_COMPUTE_ENABLE_SVE2)
