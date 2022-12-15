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

#if defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE)

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
    "ld1rb { z23.b }, p4/Z, [x21]\n"
    "ld1rb { z12.b }, p4/Z, [x20]\n"
    "add x21, x25, %[offsetof_Requantize32_minval]\n"
    "add x20, x25, %[offsetof_Requantize32_maxval]\n"
    "ld1rh { z14.h }, p4/Z, [x22]\n"
    "ld1rh { z16.h }, p4/Z, [x21]\n"
    "ld1rh { z15.h }, p4/Z, [x20]\n"
    "ldp x16, x15, [x24, #0x0]\n"
    "incw x23\n"
    "whilelt p3.h, x7, x8\n"
    "ldp x14, x13, [x24, #0x10]\n"
    "whilelt p2.s, x7, x8\n"
    "whilelt p1.s, x23, x8\n"
    "ldr x12, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1sb { z0.h }, p4/Z, [x17]\n"
    "ld1sb { z1.h }, p4/Z, [x17, #1, MUL VL]\n"
    "add x11, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x10, #0x0\n"
    "ld1sb { z2.h }, p4/Z, [x17, #2, MUL VL]\n"
    "ld1sb { z3.h }, p4/Z, [x17, #3, MUL VL]\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    ".inst 0x454c1021  // ssublb z1.h, z1.b, z12.b\n"
    "ld1sb { z4.h }, p4/Z, [x17, #4, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x17, #5, MUL VL]\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ld1sb { z6.h }, p4/Z, [x17, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    "ld1w { z18.s }, p2/Z, [x12]\n"
    "ld1w { z8.s }, p1/Z, [x12, #1, MUL VL]\n"
    "uzp1 z13.s, z18.s, z8.s\n"
    "uzp2 z17.s, z18.s, z8.s\n"
    "ld1sb { z8.h }, p4/Z, [x17]\n"
    "ldp x9, x28, [x11, #0x0]\n"
    "addvl x12, x12, #2\n"
    "mov z9.d, z13.d\n"
    "ldp x25, x24, [x11, #0x10]\n"
    "ldp x23, x22, [x11, #0x20]\n"
    "mov z10.d, z17.d\n"
    "mov z11.d, z13.d\n"
    "ldp x21, x20, [x11, #0x30]\n"
    "ld1sb { z31.h }, p3/Z, [x9, x7]\n"
    "mov z22.d, z17.d\n"
    "mov z21.d, z13.d\n"
    "ld1sb { z30.h }, p3/Z, [x28, x7]\n"
    "ld1sb { z29.h }, p3/Z, [x25, x7]\n"
    "mov z18.d, z17.d\n"
    ".inst 0x454c10a5  // ssublb z5.h, z5.b, z12.b\n"
    "ld1sb { z28.h }, p3/Z, [x24, x7]\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    ".inst 0x454c10e7  // ssublb z7.h, z7.b, z12.b\n"
    "ld1sb { z26.h }, p3/Z, [x22, x7]\n"
    "ld1sb { z25.h }, p3/Z, [x21, x7]\n"
    ".inst 0x454c1108  // ssublb z8.h, z8.b, z12.b\n"
    ".inst 0x455713ff  // ssublb z31.h, z31.b, z23.b\n"
    "ld1sb { z24.h }, p3/Z, [x20, x7]\n"
    "ldr x27, [%x[params], %[offsetof_Params_requant_muls]]\n"
    ".inst 0x455713de  // ssublb z30.h, z30.b, z23.b\n"
    ".inst 0x455713bd  // ssublb z29.h, z29.b, z23.b\n"
    "ldr x26, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "str x12, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x4557139c  // ssublb z28.h, z28.b, z23.b\n"
    ".inst 0x4557137b  // ssublb z27.h, z27.b, z23.b\n"
    ".inst 0x4557135a  // ssublb z26.h, z26.b, z23.b\n"
    ".inst 0x45571339  // ssublb z25.h, z25.b, z23.b\n"
    ".inst 0x45571318  // ssublb z24.h, z24.b, z23.b\n"
    "1:"  // Loop
    ".inst 0x448843ed  // smlalb z13.s, p4/M, z31.h, z8.h\n"
    ".inst 0x448847f1  // smlalt z17.s, p4/M, z31.h, z8.h\n"
    "ldr x25, [x11, #0x40]\n"
    "ldr x24, [x11, #0x48]\n"
    ".inst 0x448643e9  // smlalb z9.s, p4/M, z31.h, z6.h\n"
    ".inst 0x448647ea  // smlalt z10.s, p4/M, z31.h, z6.h\n"
    "ldr x22, [x11, #0x50]\n"
    "ldr x20, [x11, #0x58]\n"
    ".inst 0x448043cd  // smlalb z13.s, p4/M, z30.h, z0.h\n"
    ".inst 0x448047d1  // smlalt z17.s, p4/M, z30.h, z0.h\n"
    "ldr x23, [x11, #0x78]\n"
    "ldr x21, [x11, #0x60]\n"
    ".inst 0x44814389  // smlalb z9.s, p4/M, z28.h, z1.h\n"
    ".inst 0x4481478a  // smlalt z10.s, p4/M, z28.h, z1.h\n"
    "ld1sb { z28.h }, p3/Z, [x24, x7]\n"
    ".inst 0x4557139c  // ssublb z28.h, z28.b, z23.b\n"
    ".inst 0x448143ad  // smlalb z13.s, p4/M, z29.h, z1.h\n"
    ".inst 0x448147b1  // smlalt z17.s, p4/M, z29.h, z1.h\n"
    "ld1sb { z29.h }, p3/Z, [x25, x7]\n"
    ".inst 0x455713bd  // ssublb z29.h, z29.b, z23.b\n"
    ".inst 0x44824369  // smlalb z9.s, p4/M, z27.h, z2.h\n"
    ".inst 0x4482476a  // smlalt z10.s, p4/M, z27.h, z2.h\n"
    "ld1sb { z27.h }, p3/Z, [x22, x7]\n"
    ".inst 0x4557137b  // ssublb z27.h, z27.b, z23.b\n"
    ".inst 0x4483434d  // smlalb z13.s, p4/M, z26.h, z3.h\n"
    ".inst 0x44834751  // smlalt z17.s, p4/M, z26.h, z3.h\n"
    "ld1sb { z26.h }, p3/Z, [x20, x7]\n"
    ".inst 0x4557135a  // ssublb z26.h, z26.b, z23.b\n"
    ".inst 0x44804309  // smlalb z9.s, p4/M, z24.h, z0.h\n"
    ".inst 0x4480470a  // smlalt z10.s, p4/M, z24.h, z0.h\n"
    "ldr x22, [x11, #0x80]\n"
    "ldr x20, [x11, #0x68]\n"
    ".inst 0x4484432d  // smlalb z13.s, p4/M, z25.h, z4.h\n"
    ".inst 0x44844731  // smlalt z17.s, p4/M, z25.h, z4.h\n"
    "ld1sb { z25.h }, p3/Z, [x21, x7]\n"
    ".inst 0x45571339  // ssublb z25.h, z25.b, z23.b\n"
    ".inst 0x448443a9  // smlalb z9.s, p4/M, z29.h, z4.h\n"
    ".inst 0x448447aa  // smlalt z10.s, p4/M, z29.h, z4.h\n"
    "ldr x21, [x11, #0x88]\n"
    "ld1sb { z29.h }, p3/Z, [x20, x7]\n"
    ".inst 0x4482430d  // smlalb z13.s, p4/M, z24.h, z2.h\n"
    ".inst 0x44824711  // smlalt z17.s, p4/M, z24.h, z2.h\n"
    "ldr x20, [x11, #0x70]\n"
    ".inst 0x455713bd  // ssublb z29.h, z29.b, z23.b\n"
    ".inst 0x44854389  // smlalb z9.s, p4/M, z28.h, z5.h\n"
    ".inst 0x4485478a  // smlalt z10.s, p4/M, z28.h, z5.h\n"
    "ld1sb { z28.h }, p3/Z, [x22, x7]\n"
    ".inst 0x4557139c  // ssublb z28.h, z28.b, z23.b\n"
    ".inst 0x448243eb  // smlalb z11.s, p4/M, z31.h, z2.h\n"
    ".inst 0x448247f6  // smlalt z22.s, p4/M, z31.h, z2.h\n"
    "ldr x25, [x11, #0x98]\n"
    "ld1sb { z24.h }, p3/Z, [x20, x7]\n"
    ".inst 0x4485436d  // smlalb z13.s, p4/M, z27.h, z5.h\n"
    ".inst 0x44854771  // smlalt z17.s, p4/M, z27.h, z5.h\n"
    ".inst 0x45571318  // ssublb z24.h, z24.b, z23.b\n"
    "ldr x24, [x11, #0x90]\n"
    ".inst 0x44834369  // smlalb z9.s, p4/M, z27.h, z3.h\n"
    ".inst 0x4483476a  // smlalt z10.s, p4/M, z27.h, z3.h\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x4557137b  // ssublb z27.h, z27.b, z23.b\n"
    ".inst 0x448043f5  // smlalb z21.s, p4/M, z31.h, z0.h\n"
    ".inst 0x4483434b  // smlalb z11.s, p4/M, z26.h, z3.h\n"
    "ldr x23, [x11, #0xa8]\n"
    "ldr x20, [x11, #0xa0]\n"
    ".inst 0x44834756  // smlalt z22.s, p4/M, z26.h, z3.h\n"
    ".inst 0x448047f2  // smlalt z18.s, p4/M, z31.h, z0.h\n"
    "ld1sb { z26.h }, p3/Z, [x21, x7]\n"
    ".inst 0x4557135a  // ssublb z26.h, z26.b, z23.b\n"
    ".inst 0x44844375  // smlalb z21.s, p4/M, z27.h, z4.h\n"
    ".inst 0x4480432b  // smlalb z11.s, p4/M, z25.h, z0.h\n"
    "ldr x22, [x11, #0xb0]\n"
    "ldr x21, [x11, #0xb8]\n"
    ".inst 0x44804736  // smlalt z22.s, p4/M, z25.h, z0.h\n"
    ".inst 0x44844772  // smlalt z18.s, p4/M, z27.h, z4.h\n"
    "ld1sb { z27.h }, p3/Z, [x20, x7]\n"
    ".inst 0x4557137b  // ssublb z27.h, z27.b, z23.b\n"
    ".inst 0x44814395  // smlalb z21.s, p4/M, z28.h, z1.h\n"
    ".inst 0x4486432d  // smlalb z13.s, p4/M, z25.h, z6.h\n"
    "ldr x20, [x11, #0xc0]\n"
    "ld1w { z31.s }, p2/Z, [x27]\n"
    ".inst 0x44864731  // smlalt z17.s, p4/M, z25.h, z6.h\n"
    ".inst 0x448443ab  // smlalb z11.s, p4/M, z29.h, z4.h\n"
    "ld1sb { z25.h }, p3/Z, [x24, x7]\n"
    ".inst 0x45571339  // ssublb z25.h, z25.b, z23.b\n"
    ".inst 0x448447b6  // smlalt z22.s, p4/M, z29.h, z4.h\n"
    "ld1sb { z29.h }, p3/Z, [x25, x7]\n"
    ".inst 0x44814792  // smlalt z18.s, p4/M, z28.h, z1.h\n"
    ".inst 0x455713bd  // ssublb z29.h, z29.b, z23.b\n"
    ".inst 0x44854355  // smlalb z21.s, p4/M, z26.h, z5.h\n"
    ".inst 0x4487430d  // smlalb z13.s, p4/M, z24.h, z7.h\n"
    "ld1w { z20.s }, p1/Z, [x27, #1, MUL VL]\n"
    "uzp1 z19.s, z31.s, z20.s\n"
    ".inst 0x44874711  // smlalt z17.s, p4/M, z24.h, z7.h\n"
    ".inst 0x4481430b  // smlalb z11.s, p4/M, z24.h, z1.h\n"
    "uzp2 z30.s, z31.s, z20.s\n"
    "ld1w { z31.s }, p2/Z, [x26]\n"
    ".inst 0x44814716  // smlalt z22.s, p4/M, z24.h, z1.h\n"
    "ld1sb { z24.h }, p3/Z, [x23, x7]\n"
    ".inst 0x44854752  // smlalt z18.s, p4/M, z26.h, z5.h\n"
    ".inst 0x45571318  // ssublb z24.h, z24.b, z23.b\n"
    ".inst 0x448243b5  // smlalb z21.s, p4/M, z29.h, z2.h\n"
    "ld1sb { z26.h }, p3/Z, [x22, x7]\n"
    ".inst 0x448247b2  // smlalt z18.s, p4/M, z29.h, z2.h\n"
    ".inst 0x4557135a  // ssublb z26.h, z26.b, z23.b\n"
    ".inst 0x4486432b  // smlalb z11.s, p4/M, z25.h, z6.h\n"
    ".inst 0x44834315  // smlalb z21.s, p4/M, z24.h, z3.h\n"
    "ld1w { z20.s }, p1/Z, [x26, #1, MUL VL]\n"
    "uzp1 z1.s, z31.s, z20.s\n"
    ".inst 0x44874389  // smlalb z9.s, p4/M, z28.h, z7.h\n"
    ".inst 0x4487478a  // smlalt z10.s, p4/M, z28.h, z7.h\n"
    ".inst 0x04b375ad  // sqrdmulh z13.s, z13.s, z19.s\n"
    "whilelt p0.h, x10, x8\n"
    ".inst 0x44864736  // smlalt z22.s, p4/M, z25.h, z6.h\n"
    "ld1sb { z25.h }, p3/Z, [x21, x7]\n"
    ".inst 0x44834712  // smlalt z18.s, p4/M, z24.h, z3.h\n"
    ".inst 0x45571339  // ssublb z25.h, z25.b, z23.b\n"
    ".inst 0x4487436b  // smlalb z11.s, p4/M, z27.h, z7.h\n"
    ".inst 0x44874355  // smlalb z21.s, p4/M, z26.h, z7.h\n"
    "uzp2 z31.s, z31.s, z20.s\n"
    "inch x17\n"
    ".inst 0x448843a9  // smlalb z9.s, p4/M, z29.h, z8.h\n"
    ".inst 0x448847aa  // smlalt z10.s, p4/M, z29.h, z8.h\n"
    "ld1sb { z29.h }, p3/Z, [x20, x7]\n"
    ".inst 0x455713bd  // ssublb z29.h, z29.b, z23.b\n"
    ".inst 0x44874776  // smlalt z22.s, p4/M, z27.h, z7.h\n"
    ".inst 0x44874752  // smlalt z18.s, p4/M, z26.h, z7.h\n"
    "and z0.d, z13.d, z1.d\n"
    "inch x7\n"
    ".inst 0x4485430b  // smlalb z11.s, p4/M, z24.h, z5.h\n"
    ".inst 0x44864335  // smlalb z21.s, p4/M, z25.h, z6.h\n"
    ".inst 0x04be7631  // sqrdmulh z17.s, z17.s, z30.s\n"
    "mov x20, x7\n"
    ".inst 0x44854716  // smlalt z22.s, p4/M, z24.h, z5.h\n"
    ".inst 0x44864732  // smlalt z18.s, p4/M, z25.h, z6.h\n"
    "asr z0.s, z0.s, #0x1f\n"
    "incw x20\n"
    ".inst 0x4488432b  // smlalb z11.s, p4/M, z25.h, z8.h\n"
    ".inst 0x448843b5  // smlalb z21.s, p4/M, z29.h, z8.h\n"
    "and z20.d, z17.d, z31.d\n"
    "whilelt p2.s, x7, x8\n"
    ".inst 0x44884736  // smlalt z22.s, p4/M, z25.h, z8.h\n"
    ".inst 0x448847b2  // smlalt z18.s, p4/M, z29.h, z8.h\n"
    ".inst 0x04b37529  // sqrdmulh z9.s, z9.s, z19.s\n"
    "whilelt p1.s, x20, x8\n"
    ".inst 0x04b3756b  // sqrdmulh z11.s, z11.s, z19.s\n"
    ".inst 0x04b376b5  // sqrdmulh z21.s, z21.s, z19.s\n"
    "ldr x12, [%x[params], %[offsetof_Params_bias]]\n"
    "whilelt p3.h, x7, x8\n"
    "sqadd z13.s, z13.s, z0.s\n"
    "asr z20.s, z20.s, #0x1f\n"
    ".inst 0x4482902d  // srshl z13.s, p4/M, z13.s, z1.s\n"
    "addvl x27, x27, #2\n"
    "and z19.d, z9.d, z1.d\n"
    ".inst 0x04be754a  // sqrdmulh z10.s, z10.s, z30.s\n"
    "addvl x26, x26, #2\n"
    "and z2.d, z11.d, z1.d\n"
    ".inst 0x04be76d6  // sqrdmulh z22.s, z22.s, z30.s\n"
    "and z0.d, z21.d, z1.d\n"
    ".inst 0x04be7652  // sqrdmulh z18.s, z18.s, z30.s\n"
    "sqadd z17.s, z17.s, z20.s\n"
    "asr z19.s, z19.s, #0x1f\n"
    ".inst 0x448293f1  // srshl z17.s, p4/M, z17.s, z31.s\n"
    "and z3.d, z10.d, z31.d\n"
    "asr z2.s, z2.s, #0x1f\n"
    "and z26.d, z22.d, z31.d\n"
    "asr z0.s, z0.s, #0x1f\n"
    "and z20.d, z18.d, z31.d\n"
    "sqadd z9.s, z9.s, z19.s\n"
    ".inst 0x44829029  // srshl z9.s, p4/M, z9.s, z1.s\n"
    "asr z3.s, z3.s, #0x1f\n"
    "sqadd z11.s, z11.s, z2.s\n"
    ".inst 0x4482902b  // srshl z11.s, p4/M, z11.s, z1.s\n"
    "asr z26.s, z26.s, #0x1f\n"
    "sqadd z21.s, z21.s, z0.s\n"
    ".inst 0x44829035  // srshl z21.s, p4/M, z21.s, z1.s\n"
    "asr z20.s, z20.s, #0x1f\n"
    "sqadd z10.s, z10.s, z3.s\n"
    ".inst 0x448293ea  // srshl z10.s, p4/M, z10.s, z31.s\n"
    "sqadd z22.s, z22.s, z26.s\n"
    "sqadd z18.s, z18.s, z20.s\n"
    ".inst 0x448293f6  // srshl z22.s, p4/M, z22.s, z31.s\n"
    ".inst 0x448293f2  // srshl z18.s, p4/M, z18.s, z31.s\n"
    ".inst 0x453041ad  // sqxtnb z13.h, z13.s\n"
    ".inst 0x45304129  // sqxtnb z9.h, z9.s\n"
    ".inst 0x4530416b  // sqxtnb z11.h, z11.s\n"
    ".inst 0x453042b5  // sqxtnb z21.h, z21.s\n"
    ".inst 0x4530462d  // sqxtnt z13.h, z17.s\n"
    ".inst 0x45304549  // sqxtnt z9.h, z10.s\n"
    ".inst 0x453046cb  // sqxtnt z11.h, z22.s\n"
    ".inst 0x45304655  // sqxtnt z21.h, z18.s\n"
    "sqadd z13.h, z13.h, z14.h\n"
    "sqadd z9.h, z9.h, z14.h\n"
    "smax z13.h, p4/M, z13.h, z16.h\n"
    "smax z9.h, p4/M, z9.h, z16.h\n"
    "sqadd z11.h, z11.h, z14.h\n"
    "sqadd z21.h, z21.h, z14.h\n"
    "smax z11.h, p4/M, z11.h, z16.h\n"
    "smax z21.h, p4/M, z21.h, z16.h\n"
    "smin z13.h, p4/M, z13.h, z15.h\n"
    "smin z9.h, p4/M, z9.h, z15.h\n"
    "st1b { z13.h }, p0, [x16, x10]\n"
    "smin z11.h, p4/M, z11.h, z15.h\n"
    "smin z21.h, p4/M, z21.h, z15.h\n"
    "st1b { z9.h }, p0, [x15, x10]\n"
    "st1b { z11.h }, p0, [x14, x10]\n"
    "st1b { z21.h }, p0, [x13, x10]\n"
    "ld1sb { z0.h }, p4/Z, [x17]\n"
    "ld1sb { z1.h }, p4/Z, [x17, #1, MUL VL]\n"
    "inch x10\n"
    "ld1sb { z2.h }, p4/Z, [x17, #2, MUL VL]\n"
    "ld1sb { z3.h }, p4/Z, [x17, #3, MUL VL]\n"
    ".inst 0x454c1000  // ssublb z0.h, z0.b, z12.b\n"
    ".inst 0x454c1021  // ssublb z1.h, z1.b, z12.b\n"
    "ld1sb { z4.h }, p4/Z, [x17, #4, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x17, #5, MUL VL]\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    ".inst 0x454c1063  // ssublb z3.h, z3.b, z12.b\n"
    "ld1sb { z6.h }, p4/Z, [x17, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x17, #7, MUL VL]\n"
    "inch x17, ALL, MUL #8\n"
    ".inst 0x454c1084  // ssublb z4.h, z4.b, z12.b\n"
    "ld1w { z18.s }, p2/Z, [x12]\n"
    "ld1w { z8.s }, p1/Z, [x12, #1, MUL VL]\n"
    "uzp1 z13.s, z18.s, z8.s\n"
    "uzp2 z17.s, z18.s, z8.s\n"
    "ld1sb { z8.h }, p4/Z, [x17]\n"
    "ldp x9, x28, [x11, #0x0]\n"
    "addvl x12, x12, #2\n"
    "str x12, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x11, #0x10]\n"
    "ldp x23, x22, [x11, #0x20]\n"
    "mov z9.d, z13.d\n"
    "mov z10.d, z17.d\n"
    "ldp x21, x20, [x11, #0x30]\n"
    "ld1sb { z31.h }, p3/Z, [x9, x7]\n"
    "mov z11.d, z13.d\n"
    "mov z22.d, z17.d\n"
    "ld1sb { z30.h }, p3/Z, [x28, x7]\n"
    "ld1sb { z29.h }, p3/Z, [x25, x7]\n"
    "mov z21.d, z13.d\n"
    "mov z18.d, z17.d\n"
    "ld1sb { z28.h }, p3/Z, [x24, x7]\n"
    "ld1sb { z27.h }, p3/Z, [x23, x7]\n"
    ".inst 0x454c10a5  // ssublb z5.h, z5.b, z12.b\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    "ld1sb { z26.h }, p3/Z, [x22, x7]\n"
    "ld1sb { z25.h }, p3/Z, [x21, x7]\n"
    ".inst 0x454c10e7  // ssublb z7.h, z7.b, z12.b\n"
    ".inst 0x454c1108  // ssublb z8.h, z8.b, z12.b\n"
    "ld1sb { z24.h }, p3/Z, [x20, x7]\n"
    ".inst 0x455713ff  // ssublb z31.h, z31.b, z23.b\n"
    ".inst 0x455713de  // ssublb z30.h, z30.b, z23.b\n"
    ".inst 0x455713bd  // ssublb z29.h, z29.b, z23.b\n"
    ".inst 0x4557139c  // ssublb z28.h, z28.b, z23.b\n"
    ".inst 0x4557137b  // ssublb z27.h, z27.b, z23.b\n"
    ".inst 0x4557135a  // ssublb z26.h, z26.b, z23.b\n"
    ".inst 0x45571339  // ssublb z25.h, z25.b, z23.b\n"
    ".inst 0x45571318  // ssublb z24.h, z24.b, z23.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE)
