/*
 * Copyright (c) 2021-2023 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE)

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
    const void *weights;
    const int32_t *bias;
    const arm_gemm::Requantize32 *requant;
    const int32_t *const requant_muls;
    const int32_t *const requant_shifts;
    int8_t *const *const outptrs;
    const int8_t *inptrs[25];

    Params(
      long unsigned int n_channels,
      const int8_t *const *inptrs_raw,
      const void *const weights,
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
    "mov x7, #0x0\n"
    "ldr x25, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x24, [%x[params], %[offsetof_Params_outptrs]]\n"
    "mov x23, x7\n"
    "add x21, x25, %[offsetof_Requantize32_a_offset]\n"
    "ldr x8, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x17, [%x[params], %[offsetof_Params_weights]]\n"
    "add x20, x25, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x25, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z26.b }, p4/Z, [x21]\n"
    "ld1rb { z13.b }, p4/Z, [x20]\n"
    "add x21, x25, %[offsetof_Requantize32_minval]\n"
    "add x20, x25, %[offsetof_Requantize32_maxval]\n"
    "ld1rh { z19.h }, p4/Z, [x22]\n"
    "ld1rh { z12.h }, p4/Z, [x21]\n"
    "ld1rh { z9.h }, p4/Z, [x20]\n"
    "ldp x16, x15, [x24, #0x0]\n"
    "incw x23\n"
    "whilelt p3.h, x7, x8\n"
    "ldp x14, x13, [x24, #0x10]\n"
    "whilelt p2.s, x7, x8\n"
    "whilelt p1.s, x23, x8\n"
    "ldr x12, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1sb { z25.h }, p4/Z, [x17]\n"
    "ld1sb { z30.h }, p4/Z, [x17, #1, MUL VL]\n"
    "add x11, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x10, #0x0\n"
    "ld1sb { z14.h }, p4/Z, [x17, #2, MUL VL]\n"
    "ld1sb { z4.h }, p4/Z, [x17, #3, MUL VL]\n"
    ".inst 0x454d1339  // ssublb z25.h, z25.b, z13.b\n"
    ".inst 0x454d13de  // ssublb z30.h, z30.b, z13.b\n"
    "ld1sb { z10.h }, p4/Z, [x17, #4, MUL VL]\n"
    "ld1sb { z3.h }, p4/Z, [x17, #5, MUL VL]\n"
    ".inst 0x454d11ce  // ssublb z14.h, z14.b, z13.b\n"
    ".inst 0x454d1084  // ssublb z4.h, z4.b, z13.b\n"
    "ld1sb { z23.h }, p4/Z, [x17, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x454d114a  // ssublb z10.h, z10.b, z13.b\n"
    "ld1w { z17.s }, p2/Z, [x12]\n"
    "ld1w { z16.s }, p1/Z, [x12, #1, MUL VL]\n"
    "uzp1 z8.s, z17.s, z16.s\n"
    "uzp2 z24.s, z17.s, z16.s\n"
    "ld1sb { z2.h }, p4/Z, [x17]\n"
    "ldp x27, x26, [x11, #0x0]\n"
    "addvl x12, x12, #2\n"
    "mov z18.d, z8.d\n"
    "ldp x25, x24, [x11, #0x10]\n"
    "ldp x23, x22, [x11, #0x20]\n"
    "mov z0.d, z24.d\n"
    "mov z15.d, z8.d\n"
    "ldp x21, x20, [x11, #0x30]\n"
    "ld1sb { z21.h }, p3/Z, [x27, x7]\n"
    "mov z1.d, z24.d\n"
    "mov z5.d, z8.d\n"
    "ld1sb { z22.h }, p3/Z, [x26, x7]\n"
    "ld1sb { z11.h }, p3/Z, [x25, x7]\n"
    "mov z6.d, z24.d\n"
    ".inst 0x454d1063  // ssublb z3.h, z3.b, z13.b\n"
    "ld1sb { z20.h }, p3/Z, [x24, x7]\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x454d12f7  // ssublb z23.h, z23.b, z13.b\n"
    ".inst 0x454d10e7  // ssublb z7.h, z7.b, z13.b\n"
    "ld1sb { z28.h }, p3/Z, [x22, x7]\n"
    "ld1sb { z16.h }, p3/Z, [x21, x7]\n"
    ".inst 0x454d1042  // ssublb z2.h, z2.b, z13.b\n"
    ".inst 0x455a12b5  // ssublb z21.h, z21.b, z26.b\n"
    "ld1sb { z31.h }, p3/Z, [x20, x7]\n"
    "ldr x9, [%x[params], %[offsetof_Params_requant_muls]]\n"
    ".inst 0x455a12d6  // ssublb z22.h, z22.b, z26.b\n"
    ".inst 0x455a116b  // ssublb z11.h, z11.b, z26.b\n"
    "ldr x28, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "str x12, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x455a1294  // ssublb z20.h, z20.b, z26.b\n"
    ".inst 0x455a137b  // ssublb z27.h, z27.b, z26.b\n"
    ".inst 0x455a139c  // ssublb z28.h, z28.b, z26.b\n"
    ".inst 0x455a1210  // ssublb z16.h, z16.b, z26.b\n"
    ".inst 0x455a13ff  // ssublb z31.h, z31.b, z26.b\n"
    "1:"  // Loop
    ".inst 0x448242a8  // smlalb z8.s, p4/M, z21.h, z2.h\n"
    "ldr x21, [x11, #0x58]\n"
    "ldr x20, [x11, #0x78]\n"
    ".inst 0x448246b8  // smlalt z24.s, p4/M, z21.h, z2.h\n"
    ".inst 0x449942c8  // smlalb z8.s, p4/M, z22.h, z25.h\n"
    "ld1sb { z17.h }, p3/Z, [x21, x7]\n"
    "ld1sb { z29.h }, p3/Z, [x20, x7]\n"
    ".inst 0x449742b2  // smlalb z18.s, p4/M, z21.h, z23.h\n"
    "ldr x21, [x11, #0x60]\n"
    "ldr x20, [x11, #0x80]\n"
    ".inst 0x448e42af  // smlalb z15.s, p4/M, z21.h, z14.h\n"
    ".inst 0x449942a5  // smlalb z5.s, p4/M, z21.h, z25.h\n"
    ".inst 0x449946d8  // smlalt z24.s, p4/M, z22.h, z25.h\n"
    ".inst 0x455a1231  // ssublb z17.h, z17.b, z26.b\n"
    ".inst 0x449e4168  // smlalb z8.s, p4/M, z11.h, z30.h\n"
    "ld1sb { z22.h }, p3/Z, [x21, x7]\n"
    ".inst 0x455a13bd  // ssublb z29.h, z29.b, z26.b\n"
    ".inst 0x449746a0  // smlalt z0.s, p4/M, z21.h, z23.h\n"
    ".inst 0x448e46a1  // smlalt z1.s, p4/M, z21.h, z14.h\n"
    "ldr x21, [x11, #0x68]\n"
    ".inst 0x449946a6  // smlalt z6.s, p4/M, z21.h, z25.h\n"
    "ld1sb { z21.h }, p3/Z, [x20, x7]\n"
    "ldr x20, [x11, #0x88]\n"
    ".inst 0x449e4292  // smlalb z18.s, p4/M, z20.h, z30.h\n"
    ".inst 0x4484422f  // smlalb z15.s, p4/M, z17.h, z4.h\n"
    ".inst 0x448a43a5  // smlalb z5.s, p4/M, z29.h, z10.h\n"
    ".inst 0x455a12d6  // ssublb z22.h, z22.b, z26.b\n"
    "ldr x22, [x11, #0x40]\n"
    ".inst 0x449e4578  // smlalt z24.s, p4/M, z11.h, z30.h\n"
    ".inst 0x455a12b5  // ssublb z21.h, z21.b, z26.b\n"
    ".inst 0x44844388  // smlalb z8.s, p4/M, z28.h, z4.h\n"
    "ld1sb { z11.h }, p3/Z, [x21, x7]\n"
    ".inst 0x449e4680  // smlalt z0.s, p4/M, z20.h, z30.h\n"
    "ld1sb { z20.h }, p3/Z, [x20, x7]\n"
    ".inst 0x44844621  // smlalt z1.s, p4/M, z17.h, z4.h\n"
    "ldr x21, [x11, #0x70]\n"
    ".inst 0x448a47a6  // smlalt z6.s, p4/M, z29.h, z10.h\n"
    "ldr x20, [x11, #0x98]\n"
    ".inst 0x448e4372  // smlalb z18.s, p4/M, z27.h, z14.h\n"
    "ldr x23, [x11, #0x50]\n"
    ".inst 0x449942cf  // smlalb z15.s, p4/M, z22.h, z25.h\n"
    ".inst 0x449e42a5  // smlalb z5.s, p4/M, z21.h, z30.h\n"
    ".inst 0x455a116b  // ssublb z11.h, z11.b, z26.b\n"
    "ld1sb { z17.h }, p3/Z, [x22, x7]\n"
    ".inst 0x44844798  // smlalt z24.s, p4/M, z28.h, z4.h\n"
    ".inst 0x455a1294  // ssublb z20.h, z20.b, z26.b\n"
    ".inst 0x448a4208  // smlalb z8.s, p4/M, z16.h, z10.h\n"
    "ld1sb { z29.h }, p3/Z, [x21, x7]\n"
    "ld1sb { z28.h }, p3/Z, [x20, x7]\n"
    ".inst 0x448e4760  // smlalt z0.s, p4/M, z27.h, z14.h\n"
    "ldr x22, [x11, #0x48]\n"
    ".inst 0x449946c1  // smlalt z1.s, p4/M, z22.h, z25.h\n"
    ".inst 0x449e46a6  // smlalt z6.s, p4/M, z21.h, z30.h\n"
    "ldr x21, [x11, #0x90]\n"
    "ldr x20, [x11, #0xa8]\n"
    ".inst 0x449943f2  // smlalb z18.s, p4/M, z31.h, z25.h\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x448a416f  // smlalb z15.s, p4/M, z11.h, z10.h\n"
    ".inst 0x44834285  // smlalb z5.s, p4/M, z20.h, z3.h\n"
    ".inst 0x455a1231  // ssublb z17.h, z17.b, z26.b\n"
    ".inst 0x448a4618  // smlalt z24.s, p4/M, z16.h, z10.h\n"
    ".inst 0x455a13bd  // ssublb z29.h, z29.b, z26.b\n"
    ".inst 0x448e43e8  // smlalb z8.s, p4/M, z31.h, z14.h\n"
    "ld1sb { z16.h }, p3/Z, [x22, x7]\n"
    ".inst 0x455a139c  // ssublb z28.h, z28.b, z26.b\n"
    ".inst 0x449947e0  // smlalt z0.s, p4/M, z31.h, z25.h\n"
    "ld1sb { z25.h }, p3/Z, [x21, x7]\n"
    ".inst 0x448a4561  // smlalt z1.s, p4/M, z11.h, z10.h\n"
    "ld1sb { z11.h }, p3/Z, [x20, x7]\n"
    ".inst 0x455a137b  // ssublb z27.h, z27.b, z26.b\n"
    ".inst 0x44834686  // smlalt z6.s, p4/M, z20.h, z3.h\n"
    "ldr x21, [x11, #0xa0]\n"
    "ldr x20, [x11, #0xb0]\n"
    ".inst 0x448a4232  // smlalb z18.s, p4/M, z17.h, z10.h\n"
    ".inst 0x449e43af  // smlalb z15.s, p4/M, z29.h, z30.h\n"
    ".inst 0x455a1210  // ssublb z16.h, z16.b, z26.b\n"
    ".inst 0x448e4385  // smlalb z5.s, p4/M, z28.h, z14.h\n"
    ".inst 0x448e47f8  // smlalt z24.s, p4/M, z31.h, z14.h\n"
    ".inst 0x455a1339  // ssublb z25.h, z25.b, z26.b\n"
    "ld1sb { z20.h }, p3/Z, [x21, x7]\n"
    ".inst 0x455a116b  // ssublb z11.h, z11.b, z26.b\n"
    ".inst 0x44834368  // smlalb z8.s, p4/M, z27.h, z3.h\n"
    "ld1sb { z31.h }, p3/Z, [x20, x7]\n"
    ".inst 0x448a4620  // smlalt z0.s, p4/M, z17.h, z10.h\n"
    ".inst 0x449e47a1  // smlalt z1.s, p4/M, z29.h, z30.h\n"
    ".inst 0x448e4786  // smlalt z6.s, p4/M, z28.h, z14.h\n"
    "ldr x20, [x11, #0xb8]\n"
    ".inst 0x455a1294  // ssublb z20.h, z20.b, z26.b\n"
    ".inst 0x44834212  // smlalb z18.s, p4/M, z16.h, z3.h\n"
    ".inst 0x4497432f  // smlalb z15.s, p4/M, z25.h, z23.h\n"
    ".inst 0x455a13ff  // ssublb z31.h, z31.b, z26.b\n"
    "ld1sb { z30.h }, p3/Z, [x20, x7]\n"
    ".inst 0x44844165  // smlalb z5.s, p4/M, z11.h, z4.h\n"
    ".inst 0x44834778  // smlalt z24.s, p4/M, z27.h, z3.h\n"
    "ldr x20, [x11, #0xc0]\n"
    "ld1w { z17.s }, p2/Z, [x9]\n"
    ".inst 0x449742c8  // smlalb z8.s, p4/M, z22.h, z23.h\n"
    ".inst 0x44834600  // smlalt z0.s, p4/M, z16.h, z3.h\n"
    "ld1w { z14.s }, p1/Z, [x9, #1, MUL VL]\n"
    ".inst 0x455a13de  // ssublb z30.h, z30.b, z26.b\n"
    ".inst 0x44974721  // smlalt z1.s, p4/M, z25.h, z23.h\n"
    ".inst 0x44844566  // smlalt z6.s, p4/M, z11.h, z4.h\n"
    "ld1sb { z25.h }, p3/Z, [x20, x7]\n"
    "uzp1 z10.s, z17.s, z14.s\n"
    ".inst 0x44844372  // smlalb z18.s, p4/M, z27.h, z4.h\n"
    ".inst 0x4487428f  // smlalb z15.s, p4/M, z20.h, z7.h\n"
    "uzp2 z14.s, z17.s, z14.s\n"
    "ld1w { z17.s }, p2/Z, [x28]\n"
    ".inst 0x448743e5  // smlalb z5.s, p4/M, z31.h, z7.h\n"
    ".inst 0x449746d8  // smlalt z24.s, p4/M, z22.h, z23.h\n"
    "ld1w { z16.s }, p1/Z, [x28, #1, MUL VL]\n"
    ".inst 0x455a1339  // ssublb z25.h, z25.b, z26.b\n"
    ".inst 0x448743a8  // smlalb z8.s, p4/M, z29.h, z7.h\n"
    ".inst 0x44844760  // smlalt z0.s, p4/M, z27.h, z4.h\n"
    "uzp1 z4.s, z17.s, z16.s\n"
    "inch x7\n"
    ".inst 0x44874681  // smlalt z1.s, p4/M, z20.h, z7.h\n"
    ".inst 0x448747e6  // smlalt z6.s, p4/M, z31.h, z7.h\n"
    ".inst 0x04aa7508  // sqrdmulh z8.s, z8.s, z10.s\n"
    "whilelt p0.h, x10, x8\n"
    ".inst 0x448742b2  // smlalb z18.s, p4/M, z21.h, z7.h\n"
    ".inst 0x4483416f  // smlalb z15.s, p4/M, z11.h, z3.h\n"
    "uzp2 z22.s, z17.s, z16.s\n"
    "mov x20, x7\n"
    ".inst 0x449743c5  // smlalb z5.s, p4/M, z30.h, z23.h\n"
    ".inst 0x448747b8  // smlalt z24.s, p4/M, z29.h, z7.h\n"
    "and z17.d, z8.d, z4.d\n"
    "inch x17\n"
    ".inst 0x448746a0  // smlalt z0.s, p4/M, z21.h, z7.h\n"
    ".inst 0x44834561  // smlalt z1.s, p4/M, z11.h, z3.h\n"
    ".inst 0x04ae7718  // sqrdmulh z24.s, z24.s, z14.s\n"
    "incw x20\n"
    ".inst 0x449747c6  // smlalt z6.s, p4/M, z30.h, z23.h\n"
    ".inst 0x44824392  // smlalb z18.s, p4/M, z28.h, z2.h\n"
    "asr z17.s, z17.s, #0x1f\n"
    "whilelt p2.s, x7, x8\n"
    ".inst 0x448243cf  // smlalb z15.s, p4/M, z30.h, z2.h\n"
    ".inst 0x44824325  // smlalb z5.s, p4/M, z25.h, z2.h\n"
    "and z16.d, z24.d, z22.d\n"
    "whilelt p1.s, x20, x8\n"
    ".inst 0x44824780  // smlalt z0.s, p4/M, z28.h, z2.h\n"
    ".inst 0x448247c1  // smlalt z1.s, p4/M, z30.h, z2.h\n"
    ".inst 0x04aa7652  // sqrdmulh z18.s, z18.s, z10.s\n"
    "ldr x20, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x44824726  // smlalt z6.s, p4/M, z25.h, z2.h\n"
    ".inst 0x04aa75ef  // sqrdmulh z15.s, z15.s, z10.s\n"
    "whilelt p3.h, x7, x8\n"
    "addvl x9, x9, #2\n"
    ".inst 0x04aa74a5  // sqrdmulh z5.s, z5.s, z10.s\n"
    "sqadd z8.s, z8.s, z17.s\n"
    ".inst 0x44829088  // srshl z8.s, p4/M, z8.s, z4.s\n"
    "addvl x28, x28, #2\n"
    "asr z16.s, z16.s, #0x1f\n"
    "and z21.d, z18.d, z4.d\n"
    ".inst 0x04ae7400  // sqrdmulh z0.s, z0.s, z14.s\n"
    "and z20.d, z15.d, z4.d\n"
    ".inst 0x04ae7421  // sqrdmulh z1.s, z1.s, z14.s\n"
    "and z28.d, z5.d, z4.d\n"
    ".inst 0x04ae74c6  // sqrdmulh z6.s, z6.s, z14.s\n"
    "sqadd z24.s, z24.s, z16.s\n"
    ".inst 0x448292d8  // srshl z24.s, p4/M, z24.s, z22.s\n"
    "asr z21.s, z21.s, #0x1f\n"
    "and z25.d, z0.d, z22.d\n"
    "asr z20.s, z20.s, #0x1f\n"
    "and z17.d, z1.d, z22.d\n"
    "asr z28.s, z28.s, #0x1f\n"
    "and z16.d, z6.d, z22.d\n"
    "sqadd z18.s, z18.s, z21.s\n"
    "asr z25.s, z25.s, #0x1f\n"
    ".inst 0x44829092  // srshl z18.s, p4/M, z18.s, z4.s\n"
    "sqadd z15.s, z15.s, z20.s\n"
    "asr z17.s, z17.s, #0x1f\n"
    ".inst 0x4482908f  // srshl z15.s, p4/M, z15.s, z4.s\n"
    "sqadd z5.s, z5.s, z28.s\n"
    "asr z16.s, z16.s, #0x1f\n"
    ".inst 0x44829085  // srshl z5.s, p4/M, z5.s, z4.s\n"
    "sqadd z0.s, z0.s, z25.s\n"
    "sqadd z1.s, z1.s, z17.s\n"
    ".inst 0x448292c0  // srshl z0.s, p4/M, z0.s, z22.s\n"
    ".inst 0x448292c1  // srshl z1.s, p4/M, z1.s, z22.s\n"
    "sqadd z6.s, z6.s, z16.s\n"
    ".inst 0x45304108  // sqxtnb z8.h, z8.s\n"
    ".inst 0x448292c6  // srshl z6.s, p4/M, z6.s, z22.s\n"
    ".inst 0x45304252  // sqxtnb z18.h, z18.s\n"
    ".inst 0x453041ef  // sqxtnb z15.h, z15.s\n"
    ".inst 0x453040a5  // sqxtnb z5.h, z5.s\n"
    ".inst 0x45304708  // sqxtnt z8.h, z24.s\n"
    ".inst 0x45304412  // sqxtnt z18.h, z0.s\n"
    ".inst 0x4530442f  // sqxtnt z15.h, z1.s\n"
    ".inst 0x453044c5  // sqxtnt z5.h, z6.s\n"
    "sqadd z8.h, z8.h, z19.h\n"
    "smax z8.h, p4/M, z8.h, z12.h\n"
    "smin z8.h, p4/M, z8.h, z9.h\n"
    "sqadd z18.h, z18.h, z19.h\n"
    "sqadd z15.h, z15.h, z19.h\n"
    "smax z18.h, p4/M, z18.h, z12.h\n"
    "smax z15.h, p4/M, z15.h, z12.h\n"
    "sqadd z5.h, z5.h, z19.h\n"
    "smax z5.h, p4/M, z5.h, z12.h\n"
    "smin z18.h, p4/M, z18.h, z9.h\n"
    "st1b { z8.h }, p0, [x16, x10]\n"
    "smin z15.h, p4/M, z15.h, z9.h\n"
    "smin z5.h, p4/M, z5.h, z9.h\n"
    "st1b { z18.h }, p0, [x15, x10]\n"
    "st1b { z15.h }, p0, [x14, x10]\n"
    "st1b { z5.h }, p0, [x13, x10]\n"
    "ld1sb { z25.h }, p4/Z, [x17]\n"
    "ld1sb { z30.h }, p4/Z, [x17, #1, MUL VL]\n"
    "inch x10\n"
    "ld1sb { z14.h }, p4/Z, [x17, #2, MUL VL]\n"
    "ld1sb { z4.h }, p4/Z, [x17, #3, MUL VL]\n"
    ".inst 0x454d1339  // ssublb z25.h, z25.b, z13.b\n"
    ".inst 0x454d13de  // ssublb z30.h, z30.b, z13.b\n"
    "ld1sb { z10.h }, p4/Z, [x17, #4, MUL VL]\n"
    "ld1sb { z3.h }, p4/Z, [x17, #5, MUL VL]\n"
    ".inst 0x454d11ce  // ssublb z14.h, z14.b, z13.b\n"
    ".inst 0x454d1084  // ssublb z4.h, z4.b, z13.b\n"
    "ld1sb { z23.h }, p4/Z, [x17, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x454d114a  // ssublb z10.h, z10.b, z13.b\n"
    "ld1w { z17.s }, p2/Z, [x20]\n"
    "ld1w { z16.s }, p1/Z, [x20, #1, MUL VL]\n"
    "uzp1 z8.s, z17.s, z16.s\n"
    "uzp2 z24.s, z17.s, z16.s\n"
    "ld1sb { z2.h }, p4/Z, [x17]\n"
    "ldp x27, x26, [x11, #0x0]\n"
    "addvl x20, x20, #2\n"
    "str x20, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x11, #0x10]\n"
    "ldp x23, x22, [x11, #0x20]\n"
    "mov z18.d, z8.d\n"
    "mov z0.d, z24.d\n"
    "ldp x21, x20, [x11, #0x30]\n"
    "ld1sb { z21.h }, p3/Z, [x27, x7]\n"
    "mov z15.d, z8.d\n"
    "mov z1.d, z24.d\n"
    "ld1sb { z22.h }, p3/Z, [x26, x7]\n"
    "ld1sb { z11.h }, p3/Z, [x25, x7]\n"
    "mov z5.d, z8.d\n"
    "mov z6.d, z24.d\n"
    "ld1sb { z20.h }, p3/Z, [x24, x7]\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x454d1063  // ssublb z3.h, z3.b, z13.b\n"
    ".inst 0x454d12f7  // ssublb z23.h, z23.b, z13.b\n"
    "ld1sb { z28.h }, p3/Z, [x22, x7]\n"
    "ld1sb { z16.h }, p3/Z, [x21, x7]\n"
    ".inst 0x454d10e7  // ssublb z7.h, z7.b, z13.b\n"
    ".inst 0x454d1042  // ssublb z2.h, z2.b, z13.b\n"
    "ld1sb { z31.h }, p3/Z, [x20, x7]\n"
    ".inst 0x455a12b5  // ssublb z21.h, z21.b, z26.b\n"
    ".inst 0x455a12d6  // ssublb z22.h, z22.b, z26.b\n"
    ".inst 0x455a116b  // ssublb z11.h, z11.b, z26.b\n"
    ".inst 0x455a1294  // ssublb z20.h, z20.b, z26.b\n"
    ".inst 0x455a137b  // ssublb z27.h, z27.b, z26.b\n"
    ".inst 0x455a139c  // ssublb z28.h, z28.b, z26.b\n"
    ".inst 0x455a1210  // ssublb z16.h, z16.b, z26.b\n"
    ".inst 0x455a13ff  // ssublb z31.h, z31.b, z26.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
