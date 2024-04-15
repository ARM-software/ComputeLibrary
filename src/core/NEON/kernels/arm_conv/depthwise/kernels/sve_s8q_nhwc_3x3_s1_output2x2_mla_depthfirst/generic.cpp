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
    const void *weights;
    const int32_t *bias;
    const arm_gemm::Requantize32 *requant;
    const int32_t *const requant_muls;
    const int32_t *const requant_shifts;
    int8_t *const *const outptrs;
    const int8_t *inptrs[16];

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
    "mov x16, #0x0\n"
    "ldr x25, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x24, [%x[params], %[offsetof_Params_outptrs]]\n"
    "mov x23, x16\n"
    "add x21, x25, %[offsetof_Requantize32_a_offset]\n"
    "ldr x15, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x14, [%x[params], %[offsetof_Params_weights]]\n"
    "add x20, x25, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x25, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z12.b }, p4/Z, [x21]\n"
    "ld1rb { z30.b }, p4/Z, [x20]\n"
    "add x21, x25, %[offsetof_Requantize32_minval]\n"
    "add x20, x25, %[offsetof_Requantize32_maxval]\n"
    "ld1rh { z24.h }, p4/Z, [x22]\n"
    "ld1rh { z11.h }, p4/Z, [x21]\n"
    "ld1rh { z26.h }, p4/Z, [x20]\n"
    "ldp x13, x12, [x24, #0x0]\n"
    "incw x23\n"
    "whilelt p3.h, x16, x15\n"
    "ldp x11, x10, [x24, #0x10]\n"
    "whilelt p2.s, x16, x15\n"
    "whilelt p1.s, x23, x15\n"
    "ldr x9, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1sb { z14.h }, p4/Z, [x14]\n"
    "ld1sb { z21.h }, p4/Z, [x14, #1, MUL VL]\n"
    "add x28, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x27, #0x0\n"
    "ld1sb { z1.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ld1sb { z6.h }, p4/Z, [x14, #3, MUL VL]\n"
    ".inst 0x455e11ce  // ssublb z14.h, z14.b, z30.b\n"
    ".inst 0x455e12b5  // ssublb z21.h, z21.b, z30.b\n"
    "ld1sb { z2.h }, p4/Z, [x14, #4, MUL VL]\n"
    "ld1sb { z18.h }, p4/Z, [x14, #5, MUL VL]\n"
    ".inst 0x455e1021  // ssublb z1.h, z1.b, z30.b\n"
    ".inst 0x455e10c6  // ssublb z6.h, z6.b, z30.b\n"
    "ld1sb { z7.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ld1sb { z10.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x455e1042  // ssublb z2.h, z2.b, z30.b\n"
    "ld1w { z17.s }, p2/Z, [x9]\n"
    "ld1w { z16.s }, p1/Z, [x9, #1, MUL VL]\n"
    "uzp1 z5.s, z17.s, z16.s\n"
    "uzp2 z9.s, z17.s, z16.s\n"
    "ld1sb { z8.h }, p4/Z, [x14]\n"
    "ldp x24, x23, [x28, #0x0]\n"
    "addvl x9, x9, #2\n"
    "mov z17.d, z5.d\n"
    "ldp x22, x21, [x28, #0x10]\n"
    "ldr x20, [x28, #0x20]\n"
    "mov z25.d, z9.d\n"
    "mov z16.d, z5.d\n"
    "ld1sb { z0.h }, p3/Z, [x24, x16]\n"
    "ld1sb { z29.h }, p3/Z, [x23, x16]\n"
    "mov z23.d, z9.d\n"
    "mov z22.d, z5.d\n"
    "ld1sb { z4.h }, p3/Z, [x22, x16]\n"
    "ld1sb { z13.h }, p3/Z, [x21, x16]\n"
    "mov z27.d, z9.d\n"
    ".inst 0x455e1252  // ssublb z18.h, z18.b, z30.b\n"
    "ld1sb { z20.h }, p3/Z, [x20, x16]\n"
    "ldr x26, [%x[params], %[offsetof_Params_requant_muls]]\n"
    ".inst 0x455e10e7  // ssublb z7.h, z7.b, z30.b\n"
    ".inst 0x455e114a  // ssublb z10.h, z10.b, z30.b\n"
    "ldr x25, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "str x9, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x455e1108  // ssublb z8.h, z8.b, z30.b\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    ".inst 0x454c11ad  // ssublb z13.h, z13.b, z12.b\n"
    ".inst 0x454c1294  // ssublb z20.h, z20.b, z12.b\n"
    "1:"  // Loop
    ".inst 0x44824005  // smlalb z5.s, p4/M, z0.h, z2.h\n"
    ".inst 0x44824409  // smlalt z9.s, p4/M, z0.h, z2.h\n"
    "ldr x20, [x28, #0x28]\n"
    "ldr x21, [x28, #0x38]\n"
    ".inst 0x448e43a5  // smlalb z5.s, p4/M, z29.h, z14.h\n"
    ".inst 0x44864011  // smlalb z17.s, p4/M, z0.h, z6.h\n"
    "ld1sb { z3.h }, p3/Z, [x20, x16]\n"
    "ldr x20, [x28, #0x30]\n"
    ".inst 0x44954010  // smlalb z16.s, p4/M, z0.h, z21.h\n"
    ".inst 0x448e4016  // smlalb z22.s, p4/M, z0.h, z14.h\n"
    "ld1sb { z31.h }, p3/Z, [x21, x16]\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    ".inst 0x448e47a9  // smlalt z9.s, p4/M, z29.h, z14.h\n"
    ".inst 0x449241a5  // smlalb z5.s, p4/M, z13.h, z18.h\n"
    "ldr x21, [x28, #0x40]\n"
    "ld1sb { z15.h }, p3/Z, [x20, x16]\n"
    ".inst 0x44864419  // smlalt z25.s, p4/M, z0.h, z6.h\n"
    ".inst 0x44954417  // smlalt z23.s, p4/M, z0.h, z21.h\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    "ldr x20, [x28, #0x48]\n"
    ".inst 0x448e441b  // smlalt z27.s, p4/M, z0.h, z14.h\n"
    ".inst 0x44814091  // smlalb z17.s, p4/M, z4.h, z1.h\n"
    "ld1sb { z19.h }, p3/Z, [x21, x16]\n"
    ".inst 0x454c11ef  // ssublb z15.h, z15.b, z12.b\n"
    ".inst 0x448141b0  // smlalb z16.s, p4/M, z13.h, z1.h\n"
    ".inst 0x449541b6  // smlalb z22.s, p4/M, z13.h, z21.h\n"
    "ld1sb { z28.h }, p3/Z, [x20, x16]\n"
    ".inst 0x454c1273  // ssublb z19.h, z19.b, z12.b\n"
    ".inst 0x449245a9  // smlalt z9.s, p4/M, z13.h, z18.h\n"
    ".inst 0x448a4285  // smlalb z5.s, p4/M, z20.h, z10.h\n"
    "ldr x21, [x28, #0x50]\n"
    "ldr x20, [x28, #0x58]\n"
    ".inst 0x44814499  // smlalt z25.s, p4/M, z4.h, z1.h\n"
    ".inst 0x448145b7  // smlalt z23.s, p4/M, z13.h, z1.h\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    "ld1sb { z4.h }, p3/Z, [x21, x16]\n"
    ".inst 0x449545bb  // smlalt z27.s, p4/M, z13.h, z21.h\n"
    ".inst 0x448241b1  // smlalb z17.s, p4/M, z13.h, z2.h\n"
    "ld1sb { z29.h }, p3/Z, [x20, x16]\n"
    "ldr x21, [x28, #0x60]\n"
    ".inst 0x44874070  // smlalb z16.s, p4/M, z3.h, z7.h\n"
    ".inst 0x44864296  // smlalb z22.s, p4/M, z20.h, z6.h\n"
    "ldr x20, [x28, #0x68]\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    ".inst 0x448a4689  // smlalt z9.s, p4/M, z20.h, z10.h\n"
    ".inst 0x449543e5  // smlalb z5.s, p4/M, z31.h, z21.h\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    "ld1sb { z0.h }, p3/Z, [x21, x16]\n"
    ".inst 0x448245b9  // smlalt z25.s, p4/M, z13.h, z2.h\n"
    ".inst 0x44874477  // smlalt z23.s, p4/M, z3.h, z7.h\n"
    "ld1sb { z3.h }, p3/Z, [x20, x16]\n"
    "ldr x20, [x28, #0x70]\n"
    ".inst 0x4486469b  // smlalt z27.s, p4/M, z20.h, z6.h\n"
    ".inst 0x44874291  // smlalb z17.s, p4/M, z20.h, z7.h\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    "ld1sb { z13.h }, p3/Z, [x20, x16]\n"
    ".inst 0x44824290  // smlalb z16.s, p4/M, z20.h, z2.h\n"
    ".inst 0x448841f6  // smlalb z22.s, p4/M, z15.h, z8.h\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ldr x20, [x28, #0x78]\n"
    ".inst 0x449547e9  // smlalt z9.s, p4/M, z31.h, z21.h\n"
    ".inst 0x44814265  // smlalb z5.s, p4/M, z19.h, z1.h\n"
    ".inst 0x454c11ad  // ssublb z13.h, z13.b, z12.b\n"
    "whilelt p0.h, x27, x15\n"
    ".inst 0x44874699  // smlalt z25.s, p4/M, z20.h, z7.h\n"
    ".inst 0x44824697  // smlalt z23.s, p4/M, z20.h, z2.h\n"
    "ld1w { z20.s }, p2/Z, [x26]\n"
    "inch x14\n"
    ".inst 0x448845fb  // smlalt z27.s, p4/M, z15.h, z8.h\n"
    ".inst 0x448e43f1  // smlalb z17.s, p4/M, z31.h, z14.h\n"
    "ld1w { z15.s }, p1/Z, [x26, #1, MUL VL]\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x44924390  // smlalb z16.s, p4/M, z28.h, z18.h\n"
    ".inst 0x44824396  // smlalb z22.s, p4/M, z28.h, z2.h\n"
    "addvl x26, x26, #2\n"
    ".inst 0x44814669  // smlalt z9.s, p4/M, z19.h, z1.h\n"
    ".inst 0x44884385  // smlalb z5.s, p4/M, z28.h, z8.h\n"
    ".inst 0x448e47f9  // smlalt z25.s, p4/M, z31.h, z14.h\n"
    ".inst 0x44924797  // smlalt z23.s, p4/M, z28.h, z18.h\n"
    "ld1sb { z31.h }, p3/Z, [x20, x16]\n"
    ".inst 0x454c13ff  // ssublb z31.h, z31.b, z12.b\n"
    ".inst 0x4482479b  // smlalt z27.s, p4/M, z28.h, z2.h\n"
    ".inst 0x44954271  // smlalb z17.s, p4/M, z19.h, z21.h\n"
    "uzp1 z2.s, z20.s, z15.s\n"
    "inch x16\n"
    ".inst 0x448e4090  // smlalb z16.s, p4/M, z4.h, z14.h\n"
    ".inst 0x448143b6  // smlalb z22.s, p4/M, z29.h, z1.h\n"
    "uzp2 z15.s, z20.s, z15.s\n"
    "ld1w { z20.s }, p2/Z, [x25]\n"
    ".inst 0x44884789  // smlalt z9.s, p4/M, z28.h, z8.h\n"
    ".inst 0x44864085  // smlalb z5.s, p4/M, z4.h, z6.h\n"
    "mov x20, x16\n"
    "incw x20\n"
    ".inst 0x44954679  // smlalt z25.s, p4/M, z19.h, z21.h\n"
    ".inst 0x448e4497  // smlalt z23.s, p4/M, z4.h, z14.h\n"
    "ld1w { z19.s }, p1/Z, [x25, #1, MUL VL]\n"
    "uzp1 z21.s, z20.s, z19.s\n"
    ".inst 0x448147bb  // smlalt z27.s, p4/M, z29.h, z1.h\n"
    ".inst 0x448a4391  // smlalb z17.s, p4/M, z28.h, z10.h\n"
    "uzp2 z1.s, z20.s, z19.s\n"
    "whilelt p2.s, x16, x15\n"
    ".inst 0x44864010  // smlalb z16.s, p4/M, z0.h, z6.h\n"
    ".inst 0x44924076  // smlalb z22.s, p4/M, z3.h, z18.h\n"
    "whilelt p1.s, x20, x15\n"
    "whilelt p3.h, x16, x15\n"
    ".inst 0x44864489  // smlalt z9.s, p4/M, z4.h, z6.h\n"
    ".inst 0x44874005  // smlalb z5.s, p4/M, z0.h, z7.h\n"
    ".inst 0x04a274a5  // sqrdmulh z5.s, z5.s, z2.s\n"
    "addvl x25, x25, #2\n"
    ".inst 0x448a4799  // smlalt z25.s, p4/M, z28.h, z10.h\n"
    ".inst 0x44864417  // smlalt z23.s, p4/M, z0.h, z6.h\n"
    "and z19.d, z5.d, z21.d\n"
    ".inst 0x4492447b  // smlalt z27.s, p4/M, z3.h, z18.h\n"
    ".inst 0x449243b1  // smlalb z17.s, p4/M, z29.h, z18.h\n"
    "asr z19.s, z19.s, #0x1f\n"
    ".inst 0x448a41b0  // smlalb z16.s, p4/M, z13.h, z10.h\n"
    ".inst 0x448741b6  // smlalb z22.s, p4/M, z13.h, z7.h\n"
    "sqadd z5.s, z5.s, z19.s\n"
    ".inst 0x448292a5  // srshl z5.s, p4/M, z5.s, z21.s\n"
    ".inst 0x44874409  // smlalt z9.s, p4/M, z0.h, z7.h\n"
    ".inst 0x449247b9  // smlalt z25.s, p4/M, z29.h, z18.h\n"
    ".inst 0x04af7529  // sqrdmulh z9.s, z9.s, z15.s\n"
    ".inst 0x448a45b7  // smlalt z23.s, p4/M, z13.h, z10.h\n"
    ".inst 0x448745bb  // smlalt z27.s, p4/M, z13.h, z7.h\n"
    "and z29.d, z9.d, z1.d\n"
    ".inst 0x44884071  // smlalb z17.s, p4/M, z3.h, z8.h\n"
    ".inst 0x448843f0  // smlalb z16.s, p4/M, z31.h, z8.h\n"
    ".inst 0x04a27631  // sqrdmulh z17.s, z17.s, z2.s\n"
    ".inst 0x448a43f6  // smlalb z22.s, p4/M, z31.h, z10.h\n"
    ".inst 0x44884479  // smlalt z25.s, p4/M, z3.h, z8.h\n"
    ".inst 0x04a27610  // sqrdmulh z16.s, z16.s, z2.s\n"
    ".inst 0x448847f7  // smlalt z23.s, p4/M, z31.h, z8.h\n"
    ".inst 0x448a47fb  // smlalt z27.s, p4/M, z31.h, z10.h\n"
    ".inst 0x04a276d6  // sqrdmulh z22.s, z22.s, z2.s\n"
    "asr z29.s, z29.s, #0x1f\n"
    "and z18.d, z17.d, z21.d\n"
    ".inst 0x04af7739  // sqrdmulh z25.s, z25.s, z15.s\n"
    "and z20.d, z16.d, z21.d\n"
    ".inst 0x04af76f7  // sqrdmulh z23.s, z23.s, z15.s\n"
    "and z19.d, z22.d, z21.d\n"
    ".inst 0x04af777b  // sqrdmulh z27.s, z27.s, z15.s\n"
    "sqadd z9.s, z9.s, z29.s\n"
    ".inst 0x44829029  // srshl z9.s, p4/M, z9.s, z1.s\n"
    "asr z18.s, z18.s, #0x1f\n"
    "and z7.d, z25.d, z1.d\n"
    "asr z20.s, z20.s, #0x1f\n"
    "and z6.d, z23.d, z1.d\n"
    "asr z19.s, z19.s, #0x1f\n"
    "and z2.d, z27.d, z1.d\n"
    "sqadd z17.s, z17.s, z18.s\n"
    "asr z7.s, z7.s, #0x1f\n"
    ".inst 0x448292b1  // srshl z17.s, p4/M, z17.s, z21.s\n"
    "sqadd z16.s, z16.s, z20.s\n"
    "asr z6.s, z6.s, #0x1f\n"
    ".inst 0x448292b0  // srshl z16.s, p4/M, z16.s, z21.s\n"
    "sqadd z22.s, z22.s, z19.s\n"
    "asr z2.s, z2.s, #0x1f\n"
    ".inst 0x448292b6  // srshl z22.s, p4/M, z22.s, z21.s\n"
    "sqadd z25.s, z25.s, z7.s\n"
    "sqadd z23.s, z23.s, z6.s\n"
    ".inst 0x44829039  // srshl z25.s, p4/M, z25.s, z1.s\n"
    ".inst 0x44829037  // srshl z23.s, p4/M, z23.s, z1.s\n"
    "sqadd z27.s, z27.s, z2.s\n"
    ".inst 0x453040a5  // sqxtnb z5.h, z5.s\n"
    ".inst 0x4482903b  // srshl z27.s, p4/M, z27.s, z1.s\n"
    ".inst 0x45304231  // sqxtnb z17.h, z17.s\n"
    ".inst 0x45304210  // sqxtnb z16.h, z16.s\n"
    ".inst 0x453042d6  // sqxtnb z22.h, z22.s\n"
    ".inst 0x45304525  // sqxtnt z5.h, z9.s\n"
    ".inst 0x45304731  // sqxtnt z17.h, z25.s\n"
    ".inst 0x453046f0  // sqxtnt z16.h, z23.s\n"
    ".inst 0x45304776  // sqxtnt z22.h, z27.s\n"
    "sqadd z5.h, z5.h, z24.h\n"
    "smax z5.h, p4/M, z5.h, z11.h\n"
    "smin z5.h, p4/M, z5.h, z26.h\n"
    "sqadd z17.h, z17.h, z24.h\n"
    "sqadd z16.h, z16.h, z24.h\n"
    "smax z17.h, p4/M, z17.h, z11.h\n"
    "smax z16.h, p4/M, z16.h, z11.h\n"
    "sqadd z22.h, z22.h, z24.h\n"
    "smax z22.h, p4/M, z22.h, z11.h\n"
    "smin z17.h, p4/M, z17.h, z26.h\n"
    "st1b { z5.h }, p0, [x13, x27]\n"
    "smin z16.h, p4/M, z16.h, z26.h\n"
    "smin z22.h, p4/M, z22.h, z26.h\n"
    "st1b { z17.h }, p0, [x12, x27]\n"
    "st1b { z16.h }, p0, [x11, x27]\n"
    "st1b { z22.h }, p0, [x10, x27]\n"
    "ld1sb { z14.h }, p4/Z, [x14]\n"
    "ld1sb { z21.h }, p4/Z, [x14, #1, MUL VL]\n"
    "inch x27\n"
    "ld1sb { z1.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ld1sb { z6.h }, p4/Z, [x14, #3, MUL VL]\n"
    ".inst 0x455e11ce  // ssublb z14.h, z14.b, z30.b\n"
    ".inst 0x455e12b5  // ssublb z21.h, z21.b, z30.b\n"
    "ld1sb { z2.h }, p4/Z, [x14, #4, MUL VL]\n"
    "ld1sb { z18.h }, p4/Z, [x14, #5, MUL VL]\n"
    ".inst 0x455e1021  // ssublb z1.h, z1.b, z30.b\n"
    ".inst 0x455e10c6  // ssublb z6.h, z6.b, z30.b\n"
    "ld1sb { z7.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ld1sb { z10.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x455e1042  // ssublb z2.h, z2.b, z30.b\n"
    "ld1w { z17.s }, p2/Z, [x21]\n"
    "ld1w { z16.s }, p1/Z, [x21, #1, MUL VL]\n"
    "uzp1 z5.s, z17.s, z16.s\n"
    "uzp2 z9.s, z17.s, z16.s\n"
    "ld1sb { z8.h }, p4/Z, [x14]\n"
    "ldp x24, x23, [x28, #0x0]\n"
    "addvl x21, x21, #2\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x28, #0x10]\n"
    "ldr x20, [x28, #0x20]\n"
    "mov z17.d, z5.d\n"
    "mov z25.d, z9.d\n"
    "ld1sb { z0.h }, p3/Z, [x24, x16]\n"
    "ld1sb { z29.h }, p3/Z, [x23, x16]\n"
    "mov z16.d, z5.d\n"
    "mov z23.d, z9.d\n"
    "ld1sb { z4.h }, p3/Z, [x22, x16]\n"
    "ld1sb { z13.h }, p3/Z, [x21, x16]\n"
    "mov z22.d, z5.d\n"
    "mov z27.d, z9.d\n"
    "ld1sb { z20.h }, p3/Z, [x20, x16]\n"
    ".inst 0x455e1252  // ssublb z18.h, z18.b, z30.b\n"
    ".inst 0x455e10e7  // ssublb z7.h, z7.b, z30.b\n"
    ".inst 0x455e114a  // ssublb z10.h, z10.b, z30.b\n"
    ".inst 0x455e1108  // ssublb z8.h, z8.b, z30.b\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    ".inst 0x454c13bd  // ssublb z29.h, z29.b, z12.b\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    ".inst 0x454c11ad  // ssublb z13.h, z13.b, z12.b\n"
    ".inst 0x454c1294  // ssublb z20.h, z20.b, z12.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
