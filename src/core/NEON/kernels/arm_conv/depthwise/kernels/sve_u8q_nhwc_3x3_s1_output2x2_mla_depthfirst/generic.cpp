/*
 * Copyright (c) 2021-2024, 2026 Arm Limited.
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

#include "arm_gemm/arm_gemm.hpp"

#include <cstddef>
#include <cstdint>

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace depthwise {

void sve_u8q_nhwc_3x3_s1_output2x2_mla_depthfirst_impl(
  const unsigned int n_channels,
  const uint8_t *const *const inptrs,
  const uint8_t *const weights,
  const int32_t *const bias,
  const arm_gemm::Requantize32 &qp,
  const int32_t *const requant_muls,
  const int32_t *const requant_shifts,
  uint8_t *const *const outptrs
)
{
  struct Params
  {
    uint64_t n_channels;
    const void *weights;
    const int32_t *bias;
    const arm_gemm::Requantize32 *requant;
    const int32_t *const requant_muls;
    const int32_t *const requant_shifts;
    uint8_t *const *const outptrs;
    const uint8_t *inptrs[16];

    Params(
      uint64_t n_channels,
      const uint8_t *const *inptrs_raw,
      const void *const weights,
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
    "mov x17, #0x0\n"
    "ldr x26, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x16, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ldr x15, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x14, [%x[params], %[offsetof_Params_weights]]\n"
    "add x13, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x12, #0x0\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldr x11, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "mov x24, x17\n"
    "add x20, x26, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x26, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x26, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z10.b }, p4/Z, [x20]\n"
    "ldr x10, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x21, x26, %[offsetof_Requantize32_minval]\n"
    "add x20, x26, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z15.b }, p4/Z, [x23]\n"
    "ld1rh { z26.h }, p4/Z, [x22]\n"
    "ld1rh { z2.h }, p4/Z, [x21]\n"
    "ld1rh { z14.h }, p4/Z, [x20]\n"
    "incw x24\n"
    "whilelt p3.h, x17, x15\n"
    "ldp x9, x28, [x16, #0x0]\n"
    "ldp x27, x26, [x16, #0x10]\n"
    "whilelt p2.s, x17, x15\n"
    "whilelt p1.s, x24, x15\n"
    "ld1b { z13.h }, p4/Z, [x14]\n"
    "ld1b { z11.h }, p4/Z, [x14, #1, MUL VL]\n"
    "ld1b { z18.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ld1b { z6.h }, p4/Z, [x14, #3, MUL VL]\n"
    "ld1b { z20.h }, p4/Z, [x14, #4, MUL VL]\n"
    "ld1b { z30.h }, p4/Z, [x14, #5, MUL VL]\n"
    "ld1b { z28.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ld1b { z17.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x454f19ad  // usublb z13.h, z13.b, z15.b\n"
    "ld1w { z19.s }, p2/Z, [x25]\n"
    "ld1w { z24.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    ".inst 0x454f196b  // usublb z11.h, z11.b, z15.b\n"
    ".inst 0x454f1a52  // usublb z18.h, z18.b, z15.b\n"
    ".inst 0x454f18c6  // usublb z6.h, z6.b, z15.b\n"
    "ld1b { z5.h }, p4/Z, [x14]\n"
    "ldp x24, x23, [x13, #0x0]\n"
    ".inst 0x454f1a94  // usublb z20.h, z20.b, z15.b\n"
    ".inst 0x454f1bde  // usublb z30.h, z30.b, z15.b\n"
    "uzp1 z3.s, z19.s, z24.s\n"
    "uzp2 z16.s, z19.s, z24.s\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x13, #0x10]\n"
    ".inst 0x454f1b9c  // usublb z28.h, z28.b, z15.b\n"
    ".inst 0x454f1a31  // usublb z17.h, z17.b, z15.b\n"
    ".inst 0x454f18a5  // usublb z5.h, z5.b, z15.b\n"
    "ldr x20, [x13, #0x20]\n"
    "ld1b { z7.h }, p3/Z, [x24, x17]\n"
    "ld1b { z24.h }, p3/Z, [x23, x17]\n"
    "ld1b { z4.h }, p3/Z, [x22, x17]\n"
    "mov z8.d, z3.d\n"
    "mov z21.d, z16.d\n"
    "ld1b { z1.h }, p3/Z, [x21, x17]\n"
    "mov z0.d, z3.d\n"
    "mov z29.d, z16.d\n"
    "ld1b { z27.h }, p3/Z, [x20, x17]\n"
    "mov z19.d, z3.d\n"
    "mov z9.d, z16.d\n"
    ".inst 0x454a18e7  // usublb z7.h, z7.b, z10.b\n"
    ".inst 0x454a1b18  // usublb z24.h, z24.b, z10.b\n"
    ".inst 0x454a1884  // usublb z4.h, z4.b, z10.b\n"
    ".inst 0x454a1821  // usublb z1.h, z1.b, z10.b\n"
    ".inst 0x454a1b7b  // usublb z27.h, z27.b, z10.b\n"
    "1:"  // Loop
    ".inst 0x449440e3  // smlalb z3.s, p4/M, z7.h, z20.h\n"
    ".inst 0x449444f0  // smlalt z16.s, p4/M, z7.h, z20.h\n"
    "ldr x25, [x13, #0x28]\n"
    "ldr x24, [x13, #0x38]\n"
    ".inst 0x448640e8  // smlalb z8.s, p4/M, z7.h, z6.h\n"
    ".inst 0x448b40e0  // smlalb z0.s, p4/M, z7.h, z11.h\n"
    "ldr x23, [x13, #0x30]\n"
    "ldr x22, [x13, #0x40]\n"
    ".inst 0x448d40f3  // smlalb z19.s, p4/M, z7.h, z13.h\n"
    ".inst 0x448644f5  // smlalt z21.s, p4/M, z7.h, z6.h\n"
    "ldr x20, [x13, #0x48]\n"
    "ldr x21, [x13, #0x50]\n"
    "ld1b { z22.h }, p3/Z, [x25, x17]\n"
    ".inst 0x448b44fd  // smlalt z29.s, p4/M, z7.h, z11.h\n"
    ".inst 0x448d44e9  // smlalt z9.s, p4/M, z7.h, z13.h\n"
    "ld1b { z31.h }, p3/Z, [x24, x17]\n"
    ".inst 0x448d4303  // smlalb z3.s, p4/M, z24.h, z13.h\n"
    ".inst 0x448d4710  // smlalt z16.s, p4/M, z24.h, z13.h\n"
    "ld1b { z24.h }, p3/Z, [x23, x17]\n"
    "ld1b { z25.h }, p3/Z, [x22, x17]\n"
    ".inst 0x44924088  // smlalb z8.s, p4/M, z4.h, z18.h\n"
    ".inst 0x44924020  // smlalb z0.s, p4/M, z1.h, z18.h\n"
    "ld1b { z23.h }, p3/Z, [x20, x17]\n"
    "ldr x20, [x13, #0x58]\n"
    ".inst 0x448b4033  // smlalb z19.s, p4/M, z1.h, z11.h\n"
    ".inst 0x454a1ad6  // usublb z22.h, z22.b, z10.b\n"
    ".inst 0x44924495  // smlalt z21.s, p4/M, z4.h, z18.h\n"
    "ld1b { z12.h }, p3/Z, [x21, x17]\n"
    ".inst 0x4492443d  // smlalt z29.s, p4/M, z1.h, z18.h\n"
    ".inst 0x448b4429  // smlalt z9.s, p4/M, z1.h, z11.h\n"
    ".inst 0x454a1bff  // usublb z31.h, z31.b, z10.b\n"
    "ldr x21, [x13, #0x60]\n"
    ".inst 0x449e4023  // smlalb z3.s, p4/M, z1.h, z30.h\n"
    ".inst 0x449e4430  // smlalt z16.s, p4/M, z1.h, z30.h\n"
    ".inst 0x454a1b18  // usublb z24.h, z24.b, z10.b\n"
    "ld1b { z4.h }, p3/Z, [x20, x17]\n"
    ".inst 0x44944028  // smlalb z8.s, p4/M, z1.h, z20.h\n"
    ".inst 0x449c42c0  // smlalb z0.s, p4/M, z22.h, z28.h\n"
    ".inst 0x454a1b39  // usublb z25.h, z25.b, z10.b\n"
    "ldr x20, [x13, #0x68]\n"
    ".inst 0x44864373  // smlalb z19.s, p4/M, z27.h, z6.h\n"
    ".inst 0x44944435  // smlalt z21.s, p4/M, z1.h, z20.h\n"
    ".inst 0x454a1af7  // usublb z23.h, z23.b, z10.b\n"
    "ld1b { z7.h }, p3/Z, [x21, x17]\n"
    ".inst 0x449c46dd  // smlalt z29.s, p4/M, z22.h, z28.h\n"
    ".inst 0x44864769  // smlalt z9.s, p4/M, z27.h, z6.h\n"
    ".inst 0x454a198c  // usublb z12.h, z12.b, z10.b\n"
    "ldr x21, [x13, #0x70]\n"
    ".inst 0x44914363  // smlalb z3.s, p4/M, z27.h, z17.h\n"
    ".inst 0x44914770  // smlalt z16.s, p4/M, z27.h, z17.h\n"
    ".inst 0x454a1884  // usublb z4.h, z4.b, z10.b\n"
    "ld1b { z22.h }, p3/Z, [x20, x17]\n"
    ".inst 0x449c4368  // smlalb z8.s, p4/M, z27.h, z28.h\n"
    ".inst 0x44944360  // smlalb z0.s, p4/M, z27.h, z20.h\n"
    ".inst 0x454a18e7  // usublb z7.h, z7.b, z10.b\n"
    "ldr x20, [x13, #0x78]\n"
    ".inst 0x44854313  // smlalb z19.s, p4/M, z24.h, z5.h\n"
    ".inst 0x449c4775  // smlalt z21.s, p4/M, z27.h, z28.h\n"
    "ld1b { z1.h }, p3/Z, [x21, x17]\n"
    "whilelt p0.h, x12, x15\n"
    ".inst 0x4494477d  // smlalt z29.s, p4/M, z27.h, z20.h\n"
    ".inst 0x44854709  // smlalt z9.s, p4/M, z24.h, z5.h\n"
    ".inst 0x454a1ad6  // usublb z22.h, z22.b, z10.b\n"
    "ld1w { z24.s }, p2/Z, [x11]\n"
    ".inst 0x448b43e3  // smlalb z3.s, p4/M, z31.h, z11.h\n"
    ".inst 0x448b47f0  // smlalt z16.s, p4/M, z31.h, z11.h\n"
    "ld1w { z27.s }, p1/Z, [x11, #1, MUL VL]\n"
    "inch x14\n"
    ".inst 0x448d43e8  // smlalb z8.s, p4/M, z31.h, z13.h\n"
    ".inst 0x449e42e0  // smlalb z0.s, p4/M, z23.h, z30.h\n"
    ".inst 0x454a1821  // usublb z1.h, z1.b, z10.b\n"
    "ldr x21, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x449442f3  // smlalb z19.s, p4/M, z23.h, z20.h\n"
    ".inst 0x448d47f5  // smlalt z21.s, p4/M, z31.h, z13.h\n"
    "ld1b { z31.h }, p3/Z, [x20, x17]\n"
    "inch x17\n"
    ".inst 0x449e46fd  // smlalt z29.s, p4/M, z23.h, z30.h\n"
    ".inst 0x449446e9  // smlalt z9.s, p4/M, z23.h, z20.h\n"
    "uzp1 z20.s, z24.s, z27.s\n"
    "addvl x11, x11, #2\n"
    ".inst 0x44924323  // smlalb z3.s, p4/M, z25.h, z18.h\n"
    ".inst 0x44924730  // smlalt z16.s, p4/M, z25.h, z18.h\n"
    "uzp2 z24.s, z24.s, z27.s\n"
    "ld1w { z27.s }, p2/Z, [x10]\n"
    ".inst 0x448b4328  // smlalb z8.s, p4/M, z25.h, z11.h\n"
    ".inst 0x448d4180  // smlalb z0.s, p4/M, z12.h, z13.h\n"
    ".inst 0x454a1bff  // usublb z31.h, z31.b, z10.b\n"
    "mov x20, x17\n"
    ".inst 0x44924093  // smlalb z19.s, p4/M, z4.h, z18.h\n"
    ".inst 0x448b4735  // smlalt z21.s, p4/M, z25.h, z11.h\n"
    "ld1w { z25.s }, p1/Z, [x10, #1, MUL VL]\n"
    "whilelt p2.s, x17, x15\n"
    ".inst 0x448d459d  // smlalt z29.s, p4/M, z12.h, z13.h\n"
    ".inst 0x44924489  // smlalt z9.s, p4/M, z4.h, z18.h\n"
    "addvl x10, x10, #2\n"
    ".inst 0x448542e3  // smlalb z3.s, p4/M, z23.h, z5.h\n"
    ".inst 0x448546f0  // smlalt z16.s, p4/M, z23.h, z5.h\n"
    "incw x20\n"
    ".inst 0x449142e8  // smlalb z8.s, p4/M, z23.h, z17.h\n"
    ".inst 0x448640e0  // smlalb z0.s, p4/M, z7.h, z6.h\n"
    "uzp1 z11.s, z27.s, z25.s\n"
    ".inst 0x449e42d3  // smlalb z19.s, p4/M, z22.h, z30.h\n"
    ".inst 0x449146f5  // smlalt z21.s, p4/M, z23.h, z17.h\n"
    "uzp2 z27.s, z27.s, z25.s\n"
    ".inst 0x448644fd  // smlalt z29.s, p4/M, z7.h, z6.h\n"
    ".inst 0x449e46c9  // smlalt z9.s, p4/M, z22.h, z30.h\n"
    "whilelt p1.s, x20, x15\n"
    "whilelt p3.h, x17, x15\n"
    ".inst 0x44864183  // smlalb z3.s, p4/M, z12.h, z6.h\n"
    ".inst 0x44864590  // smlalt z16.s, p4/M, z12.h, z6.h\n"
    ".inst 0x449e4088  // smlalb z8.s, p4/M, z4.h, z30.h\n"
    ".inst 0x44914020  // smlalb z0.s, p4/M, z1.h, z17.h\n"
    ".inst 0x449c4033  // smlalb z19.s, p4/M, z1.h, z28.h\n"
    ".inst 0x449e4495  // smlalt z21.s, p4/M, z4.h, z30.h\n"
    ".inst 0x4491443d  // smlalt z29.s, p4/M, z1.h, z17.h\n"
    ".inst 0x449c4429  // smlalt z9.s, p4/M, z1.h, z28.h\n"
    ".inst 0x449c40e3  // smlalb z3.s, p4/M, z7.h, z28.h\n"
    ".inst 0x449c44f0  // smlalt z16.s, p4/M, z7.h, z28.h\n"
    ".inst 0x448542c8  // smlalb z8.s, p4/M, z22.h, z5.h\n"
    ".inst 0x448543e0  // smlalb z0.s, p4/M, z31.h, z5.h\n"
    ".inst 0x449143f3  // smlalb z19.s, p4/M, z31.h, z17.h\n"
    ".inst 0x448546d5  // smlalt z21.s, p4/M, z22.h, z5.h\n"
    ".inst 0x448547fd  // smlalt z29.s, p4/M, z31.h, z5.h\n"
    ".inst 0x449147e9  // smlalt z9.s, p4/M, z31.h, z17.h\n"
    ".inst 0x04b47463  // sqrdmulh z3.s, z3.s, z20.s\n"
    ".inst 0x04b87610  // sqrdmulh z16.s, z16.s, z24.s\n"
    ".inst 0x04b47508  // sqrdmulh z8.s, z8.s, z20.s\n"
    ".inst 0x04b47400  // sqrdmulh z0.s, z0.s, z20.s\n"
    "and z4.d, z3.d, z11.d\n"
    ".inst 0x04b47673  // sqrdmulh z19.s, z19.s, z20.s\n"
    ".inst 0x04b876b5  // sqrdmulh z21.s, z21.s, z24.s\n"
    "and z13.d, z16.d, z27.d\n"
    "and z6.d, z8.d, z11.d\n"
    "asr z4.s, z4.s, #0x1f\n"
    "and z7.d, z0.d, z11.d\n"
    ".inst 0x04b877bd  // sqrdmulh z29.s, z29.s, z24.s\n"
    ".inst 0x04b87529  // sqrdmulh z9.s, z9.s, z24.s\n"
    "asr z13.s, z13.s, #0x1f\n"
    "asr z6.s, z6.s, #0x1f\n"
    "sqadd z3.s, z3.s, z4.s\n"
    "and z20.d, z19.d, z11.d\n"
    "and z18.d, z21.d, z27.d\n"
    "asr z7.s, z7.s, #0x1f\n"
    "sqadd z16.s, z16.s, z13.s\n"
    "and z13.d, z29.d, z27.d\n"
    "asr z20.s, z20.s, #0x1f\n"
    "and z23.d, z9.d, z27.d\n"
    ".inst 0x44829163  // srshl z3.s, p4/M, z3.s, z11.s\n"
    "sqadd z8.s, z8.s, z6.s\n"
    "asr z18.s, z18.s, #0x1f\n"
    "sqadd z0.s, z0.s, z7.s\n"
    "asr z13.s, z13.s, #0x1f\n"
    ".inst 0x44829370  // srshl z16.s, p4/M, z16.s, z27.s\n"
    "sqadd z19.s, z19.s, z20.s\n"
    "asr z23.s, z23.s, #0x1f\n"
    ".inst 0x44829168  // srshl z8.s, p4/M, z8.s, z11.s\n"
    "sqadd z21.s, z21.s, z18.s\n"
    ".inst 0x45304063  // sqxtnb z3.h, z3.s\n"
    ".inst 0x44829160  // srshl z0.s, p4/M, z0.s, z11.s\n"
    "sqadd z29.s, z29.s, z13.s\n"
    ".inst 0x44829173  // srshl z19.s, p4/M, z19.s, z11.s\n"
    "sqadd z9.s, z9.s, z23.s\n"
    ".inst 0x45304108  // sqxtnb z8.h, z8.s\n"
    ".inst 0x44829375  // srshl z21.s, p4/M, z21.s, z27.s\n"
    ".inst 0x45304000  // sqxtnb z0.h, z0.s\n"
    ".inst 0x45304603  // sqxtnt z3.h, z16.s\n"
    ".inst 0x4482937d  // srshl z29.s, p4/M, z29.s, z27.s\n"
    ".inst 0x44829369  // srshl z9.s, p4/M, z9.s, z27.s\n"
    ".inst 0x45304273  // sqxtnb z19.h, z19.s\n"
    ".inst 0x453046a8  // sqxtnt z8.h, z21.s\n"
    ".inst 0x453047a0  // sqxtnt z0.h, z29.s\n"
    ".inst 0x45304533  // sqxtnt z19.h, z9.s\n"
    "sqadd z3.h, z3.h, z26.h\n"
    "sqadd z8.h, z8.h, z26.h\n"
    "sqadd z0.h, z0.h, z26.h\n"
    "sqadd z19.h, z19.h, z26.h\n"
    "smax z3.h, p4/M, z3.h, z2.h\n"
    "smax z8.h, p4/M, z8.h, z2.h\n"
    "smax z0.h, p4/M, z0.h, z2.h\n"
    "smax z19.h, p4/M, z19.h, z2.h\n"
    "smin z3.h, p4/M, z3.h, z14.h\n"
    "smin z8.h, p4/M, z8.h, z14.h\n"
    "smin z0.h, p4/M, z0.h, z14.h\n"
    "smin z19.h, p4/M, z19.h, z14.h\n"
    "st1b { z3.h }, p0, [x9, x12]\n"
    "st1b { z8.h }, p0, [x28, x12]\n"
    "st1b { z0.h }, p0, [x27, x12]\n"
    "st1b { z19.h }, p0, [x26, x12]\n"
    "inch x12\n"
    "ld1b { z13.h }, p4/Z, [x14]\n"
    "ld1b { z11.h }, p4/Z, [x14, #1, MUL VL]\n"
    "ld1b { z18.h }, p4/Z, [x14, #2, MUL VL]\n"
    "ld1b { z6.h }, p4/Z, [x14, #3, MUL VL]\n"
    "ld1b { z20.h }, p4/Z, [x14, #4, MUL VL]\n"
    "ld1b { z30.h }, p4/Z, [x14, #5, MUL VL]\n"
    "ld1b { z28.h }, p4/Z, [x14, #6, MUL VL]\n"
    "ld1b { z17.h }, p4/Z, [x14, #7, MUL VL]\n"
    "inch x14, ALL, MUL #8\n"
    ".inst 0x454f19ad  // usublb z13.h, z13.b, z15.b\n"
    "ld1w { z1.s }, p2/Z, [x21]\n"
    "ld1w { z0.s }, p1/Z, [x21, #1, MUL VL]\n"
    "addvl x21, x21, #2\n"
    ".inst 0x454f196b  // usublb z11.h, z11.b, z15.b\n"
    ".inst 0x454f1a52  // usublb z18.h, z18.b, z15.b\n"
    ".inst 0x454f18c6  // usublb z6.h, z6.b, z15.b\n"
    "ld1b { z5.h }, p4/Z, [x14]\n"
    "ldp x24, x23, [x13, #0x0]\n"
    ".inst 0x454f1a94  // usublb z20.h, z20.b, z15.b\n"
    ".inst 0x454f1bde  // usublb z30.h, z30.b, z15.b\n"
    "uzp1 z3.s, z1.s, z0.s\n"
    "uzp2 z16.s, z1.s, z0.s\n"
    "str x21, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x13, #0x10]\n"
    ".inst 0x454f1b9c  // usublb z28.h, z28.b, z15.b\n"
    ".inst 0x454f1a31  // usublb z17.h, z17.b, z15.b\n"
    ".inst 0x454f18a5  // usublb z5.h, z5.b, z15.b\n"
    "ldr x20, [x13, #0x20]\n"
    "ld1b { z7.h }, p3/Z, [x24, x17]\n"
    "ld1b { z24.h }, p3/Z, [x23, x17]\n"
    "ld1b { z4.h }, p3/Z, [x22, x17]\n"
    "mov z8.d, z3.d\n"
    "mov z21.d, z16.d\n"
    "ld1b { z1.h }, p3/Z, [x21, x17]\n"
    "mov z0.d, z3.d\n"
    "mov z29.d, z16.d\n"
    "ld1b { z27.h }, p3/Z, [x20, x17]\n"
    "mov z19.d, z3.d\n"
    "mov z9.d, z16.d\n"
    ".inst 0x454a18e7  // usublb z7.h, z7.b, z10.b\n"
    ".inst 0x454a1b18  // usublb z24.h, z24.b, z10.b\n"
    ".inst 0x454a1884  // usublb z4.h, z4.b, z10.b\n"
    ".inst 0x454a1821  // usublb z1.h, z1.b, z10.b\n"
    ".inst 0x454a1b7b  // usublb z27.h, z27.b, z10.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
