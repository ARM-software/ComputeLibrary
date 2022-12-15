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
    "mov x8, #0x0\n"
    "ldr x25, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x24, [%x[params], %[offsetof_Params_outptrs]]\n"
    "mov x23, x8\n"
    "add x21, x25, %[offsetof_Requantize32_a_offset]\n"
    "ldr x17, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x16, [%x[params], %[offsetof_Params_weights]]\n"
    "add x20, x25, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x25, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z23.b }, p4/Z, [x21]\n"
    "ld1rb { z15.b }, p4/Z, [x20]\n"
    "add x21, x25, %[offsetof_Requantize32_minval]\n"
    "add x20, x25, %[offsetof_Requantize32_maxval]\n"
    "ld1rh { z14.h }, p4/Z, [x22]\n"
    "ld1rh { z12.h }, p4/Z, [x21]\n"
    "ld1rh { z11.h }, p4/Z, [x20]\n"
    "ldp x15, x14, [x24, #0x0]\n"
    "incw x23\n"
    "whilelt p3.h, x8, x17\n"
    "ldp x13, x12, [x24, #0x10]\n"
    "whilelt p2.s, x8, x17\n"
    "whilelt p1.s, x23, x17\n"
    "ldr x26, [%x[params], %[offsetof_Params_bias]]\n"
    "ld1sb { z0.h }, p4/Z, [x16]\n"
    "ld1sb { z1.h }, p4/Z, [x16, #1, MUL VL]\n"
    "add x11, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x10, #0x0\n"
    "ld1sb { z2.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1sb { z3.h }, p4/Z, [x16, #3, MUL VL]\n"
    ".inst 0x454f1000  // ssublb z0.h, z0.b, z15.b\n"
    ".inst 0x454f1021  // ssublb z1.h, z1.b, z15.b\n"
    "ld1sb { z4.h }, p4/Z, [x16, #4, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x16, #5, MUL VL]\n"
    ".inst 0x454f1042  // ssublb z2.h, z2.b, z15.b\n"
    ".inst 0x454f1063  // ssublb z3.h, z3.b, z15.b\n"
    "ld1sb { z6.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454f1084  // ssublb z4.h, z4.b, z15.b\n"
    "ld1w { z17.s }, p2/Z, [x26]\n"
    "ld1w { z16.s }, p1/Z, [x26, #1, MUL VL]\n"
    "uzp1 z13.s, z17.s, z16.s\n"
    "uzp2 z17.s, z17.s, z16.s\n"
    "ld1sb { z8.h }, p4/Z, [x16]\n"
    "ldp x24, x23, [x11, #0x0]\n"
    "addvl x26, x26, #2\n"
    "mov z26.d, z13.d\n"
    "ldp x22, x21, [x11, #0x10]\n"
    "ldr x20, [x11, #0x20]\n"
    "mov z10.d, z17.d\n"
    "mov z24.d, z13.d\n"
    "ld1b { z31.h }, p3/Z, [x24, x8]\n"
    "ld1b { z30.h }, p3/Z, [x23, x8]\n"
    "mov z16.d, z17.d\n"
    "mov z25.d, z13.d\n"
    "ld1b { z29.h }, p3/Z, [x22, x8]\n"
    "ld1b { z28.h }, p3/Z, [x21, x8]\n"
    "mov z9.d, z17.d\n"
    ".inst 0x454f10a5  // ssublb z5.h, z5.b, z15.b\n"
    "ld1b { z27.h }, p3/Z, [x20, x8]\n"
    "ldr x9, [%x[params], %[offsetof_Params_requant_muls]]\n"
    ".inst 0x454f10c6  // ssublb z6.h, z6.b, z15.b\n"
    ".inst 0x454f10e7  // ssublb z7.h, z7.b, z15.b\n"
    "ldr x28, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "str x26, [%x[params], %[offsetof_Params_bias]]\n"
    ".inst 0x454f1108  // ssublb z8.h, z8.b, z15.b\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    ".inst 0x45571bde  // usublb z30.h, z30.b, z23.b\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    ".inst 0x45571b7b  // usublb z27.h, z27.b, z23.b\n"
    "1:"  // Loop
    ".inst 0x448443ed  // smlalb z13.s, p4/M, z31.h, z4.h\n"
    ".inst 0x448447f1  // smlalt z17.s, p4/M, z31.h, z4.h\n"
    "ldr x22, [x11, #0x28]\n"
    "ldr x27, [x11, #0x38]\n"
    ".inst 0x448343fa  // smlalb z26.s, p4/M, z31.h, z3.h\n"
    ".inst 0x448347ea  // smlalt z10.s, p4/M, z31.h, z3.h\n"
    "ldr x21, [x11, #0x30]\n"
    "ldr x26, [x11, #0x40]\n"
    ".inst 0x448043cd  // smlalb z13.s, p4/M, z30.h, z0.h\n"
    ".inst 0x448047d1  // smlalt z17.s, p4/M, z30.h, z0.h\n"
    "ldr x20, [x11, #0x48]\n"
    "ld1b { z30.h }, p3/Z, [x20, x8]\n"
    ".inst 0x448243ba  // smlalb z26.s, p4/M, z29.h, z2.h\n"
    ".inst 0x448247aa  // smlalt z10.s, p4/M, z29.h, z2.h\n"
    "ld1b { z29.h }, p3/Z, [x21, x8]\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    ".inst 0x448143f8  // smlalb z24.s, p4/M, z31.h, z1.h\n"
    ".inst 0x448147f0  // smlalt z16.s, p4/M, z31.h, z1.h\n"
    "ldr x25, [x11, #0x50]\n"
    "ldr x24, [x11, #0x58]\n"
    ".inst 0x448043f9  // smlalb z25.s, p4/M, z31.h, z0.h\n"
    ".inst 0x448047e9  // smlalt z9.s, p4/M, z31.h, z0.h\n"
    "ld1b { z31.h }, p3/Z, [x22, x8]\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    ".inst 0x4485438d  // smlalb z13.s, p4/M, z28.h, z5.h\n"
    ".inst 0x44854791  // smlalt z17.s, p4/M, z28.h, z5.h\n"
    ".inst 0x45571bde  // usublb z30.h, z30.b, z23.b\n"
    "ldr x23, [x11, #0x60]\n"
    ".inst 0x4484439a  // smlalb z26.s, p4/M, z28.h, z4.h\n"
    ".inst 0x4484478a  // smlalt z10.s, p4/M, z28.h, z4.h\n"
    "ldr x22, [x11, #0x68]\n"
    "ldr x21, [x11, #0x70]\n"
    ".inst 0x44824398  // smlalb z24.s, p4/M, z28.h, z2.h\n"
    ".inst 0x44824790  // smlalt z16.s, p4/M, z28.h, z2.h\n"
    "ldr x20, [x11, #0x78]\n"
    "ld1w { z20.s }, p2/Z, [x9]\n"
    ".inst 0x44814399  // smlalb z25.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44814789  // smlalt z9.s, p4/M, z28.h, z1.h\n"
    "ld1b { z28.h }, p3/Z, [x27, x8]\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    ".inst 0x4487436d  // smlalb z13.s, p4/M, z27.h, z7.h\n"
    ".inst 0x44874771  // smlalt z17.s, p4/M, z27.h, z7.h\n"
    "ld1w { z18.s }, p1/Z, [x9, #1, MUL VL]\n"
    "uzp1 z19.s, z20.s, z18.s\n"
    ".inst 0x4486437a  // smlalb z26.s, p4/M, z27.h, z6.h\n"
    ".inst 0x4486476a  // smlalt z10.s, p4/M, z27.h, z6.h\n"
    "uzp2 z22.s, z20.s, z18.s\n"
    "ld1w { z20.s }, p2/Z, [x28]\n"
    ".inst 0x448643f8  // smlalb z24.s, p4/M, z31.h, z6.h\n"
    ".inst 0x448647f0  // smlalt z16.s, p4/M, z31.h, z6.h\n"
    "ld1b { z31.h }, p3/Z, [x26, x8]\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    ".inst 0x44834379  // smlalb z25.s, p4/M, z27.h, z3.h\n"
    ".inst 0x44834769  // smlalt z9.s, p4/M, z27.h, z3.h\n"
    "whilelt p0.h, x10, x17\n"
    "inch x16\n"
    ".inst 0x4481438d  // smlalb z13.s, p4/M, z28.h, z1.h\n"
    ".inst 0x44814791  // smlalt z17.s, p4/M, z28.h, z1.h\n"
    "ldr x26, [%x[params], %[offsetof_Params_bias]]\n"
    "addvl x9, x9, #2\n"
    ".inst 0x4480439a  // smlalb z26.s, p4/M, z28.h, z0.h\n"
    ".inst 0x4480478a  // smlalt z10.s, p4/M, z28.h, z0.h\n"
    "ld1b { z28.h }, p3/Z, [x24, x8]\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    ".inst 0x44844378  // smlalb z24.s, p4/M, z27.h, z4.h\n"
    ".inst 0x448843b9  // smlalb z25.s, p4/M, z29.h, z8.h\n"
    ".inst 0x44844770  // smlalt z16.s, p4/M, z27.h, z4.h\n"
    ".inst 0x448847a9  // smlalt z9.s, p4/M, z29.h, z8.h\n"
    "ld1b { z29.h }, p3/Z, [x25, x8]\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    ".inst 0x448243ed  // smlalb z13.s, p4/M, z31.h, z2.h\n"
    ".inst 0x448247f1  // smlalt z17.s, p4/M, z31.h, z2.h\n"
    "ld1w { z18.s }, p1/Z, [x28, #1, MUL VL]\n"
    "addvl x28, x28, #2\n"
    ".inst 0x448143fa  // smlalb z26.s, p4/M, z31.h, z1.h\n"
    ".inst 0x448147ea  // smlalt z10.s, p4/M, z31.h, z1.h\n"
    "ld1b { z31.h }, p3/Z, [x23, x8]\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    ".inst 0x448543d8  // smlalb z24.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448443d9  // smlalb z25.s, p4/M, z30.h, z4.h\n"
    "uzp1 z1.s, z20.s, z18.s\n"
    ".inst 0x448843cd  // smlalb z13.s, p4/M, z30.h, z8.h\n"
    ".inst 0x448847d1  // smlalt z17.s, p4/M, z30.h, z8.h\n"
    "uzp2 z27.s, z20.s, z18.s\n"
    ".inst 0x448743da  // smlalb z26.s, p4/M, z30.h, z7.h\n"
    ".inst 0x448747ca  // smlalt z10.s, p4/M, z30.h, z7.h\n"
    ".inst 0x448547d0  // smlalt z16.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448447c9  // smlalt z9.s, p4/M, z30.h, z4.h\n"
    "ld1b { z30.h }, p3/Z, [x22, x8]\n"
    ".inst 0x45571bde  // usublb z30.h, z30.b, z23.b\n"
    ".inst 0x448043b8  // smlalb z24.s, p4/M, z29.h, z0.h\n"
    ".inst 0x44824399  // smlalb z25.s, p4/M, z28.h, z2.h\n"
    ".inst 0x448343ad  // smlalb z13.s, p4/M, z29.h, z3.h\n"
    ".inst 0x448347b1  // smlalt z17.s, p4/M, z29.h, z3.h\n"
    ".inst 0x448047b0  // smlalt z16.s, p4/M, z29.h, z0.h\n"
    "ld1b { z29.h }, p3/Z, [x21, x8]\n"
    ".inst 0x44824789  // smlalt z9.s, p4/M, z28.h, z2.h\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    ".inst 0x448343f8  // smlalb z24.s, p4/M, z31.h, z3.h\n"
    ".inst 0x448543d9  // smlalb z25.s, p4/M, z30.h, z5.h\n"
    ".inst 0x4485439a  // smlalb z26.s, p4/M, z28.h, z5.h\n"
    ".inst 0x4485478a  // smlalt z10.s, p4/M, z28.h, z5.h\n"
    "ld1b { z28.h }, p3/Z, [x20, x8]\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    ".inst 0x448643ed  // smlalb z13.s, p4/M, z31.h, z6.h\n"
    ".inst 0x448347f0  // smlalt z16.s, p4/M, z31.h, z3.h\n"
    ".inst 0x04b375ad  // sqrdmulh z13.s, z13.s, z19.s\n"
    "inch x8\n"
    ".inst 0x448547c9  // smlalt z9.s, p4/M, z30.h, z5.h\n"
    ".inst 0x448743b8  // smlalb z24.s, p4/M, z29.h, z7.h\n"
    "and z21.d, z13.d, z1.d\n"
    "mov x20, x8\n"
    ".inst 0x448643b9  // smlalb z25.s, p4/M, z29.h, z6.h\n"
    ".inst 0x448647f1  // smlalt z17.s, p4/M, z31.h, z6.h\n"
    ".inst 0x04b67631  // sqrdmulh z17.s, z17.s, z22.s\n"
    "incw x20\n"
    ".inst 0x448747b0  // smlalt z16.s, p4/M, z29.h, z7.h\n"
    ".inst 0x448647a9  // smlalt z9.s, p4/M, z29.h, z6.h\n"
    "asr z21.s, z21.s, #0x1f\n"
    "whilelt p2.s, x8, x17\n"
    ".inst 0x448843da  // smlalb z26.s, p4/M, z30.h, z8.h\n"
    ".inst 0x44884398  // smlalb z24.s, p4/M, z28.h, z8.h\n"
    "and z20.d, z17.d, z27.d\n"
    "whilelt p1.s, x20, x17\n"
    ".inst 0x44874399  // smlalb z25.s, p4/M, z28.h, z7.h\n"
    ".inst 0x448847ca  // smlalt z10.s, p4/M, z30.h, z8.h\n"
    ".inst 0x04b3775a  // sqrdmulh z26.s, z26.s, z19.s\n"
    "whilelt p3.h, x8, x17\n"
    ".inst 0x44884790  // smlalt z16.s, p4/M, z28.h, z8.h\n"
    ".inst 0x44874789  // smlalt z9.s, p4/M, z28.h, z7.h\n"
    ".inst 0x04b37718  // sqrdmulh z24.s, z24.s, z19.s\n"
    ".inst 0x04b37739  // sqrdmulh z25.s, z25.s, z19.s\n"
    "sqadd z13.s, z13.s, z21.s\n"
    ".inst 0x4482902d  // srshl z13.s, p4/M, z13.s, z1.s\n"
    "asr z20.s, z20.s, #0x1f\n"
    "and z19.d, z26.d, z1.d\n"
    ".inst 0x04b6754a  // sqrdmulh z10.s, z10.s, z22.s\n"
    "and z18.d, z24.d, z1.d\n"
    ".inst 0x04b67610  // sqrdmulh z16.s, z16.s, z22.s\n"
    "and z21.d, z25.d, z1.d\n"
    ".inst 0x04b67529  // sqrdmulh z9.s, z9.s, z22.s\n"
    "sqadd z17.s, z17.s, z20.s\n"
    ".inst 0x44829371  // srshl z17.s, p4/M, z17.s, z27.s\n"
    "asr z19.s, z19.s, #0x1f\n"
    "and z2.d, z10.d, z27.d\n"
    "asr z18.s, z18.s, #0x1f\n"
    "and z22.d, z16.d, z27.d\n"
    "asr z21.s, z21.s, #0x1f\n"
    "and z20.d, z9.d, z27.d\n"
    "sqadd z26.s, z26.s, z19.s\n"
    "asr z2.s, z2.s, #0x1f\n"
    ".inst 0x4482903a  // srshl z26.s, p4/M, z26.s, z1.s\n"
    "sqadd z24.s, z24.s, z18.s\n"
    "asr z22.s, z22.s, #0x1f\n"
    ".inst 0x44829038  // srshl z24.s, p4/M, z24.s, z1.s\n"
    "sqadd z25.s, z25.s, z21.s\n"
    "asr z20.s, z20.s, #0x1f\n"
    ".inst 0x44829039  // srshl z25.s, p4/M, z25.s, z1.s\n"
    "sqadd z10.s, z10.s, z2.s\n"
    "sqadd z16.s, z16.s, z22.s\n"
    ".inst 0x4482936a  // srshl z10.s, p4/M, z10.s, z27.s\n"
    ".inst 0x44829370  // srshl z16.s, p4/M, z16.s, z27.s\n"
    "sqadd z9.s, z9.s, z20.s\n"
    ".inst 0x453041ad  // sqxtnb z13.h, z13.s\n"
    ".inst 0x44829369  // srshl z9.s, p4/M, z9.s, z27.s\n"
    ".inst 0x4530435a  // sqxtnb z26.h, z26.s\n"
    ".inst 0x45304318  // sqxtnb z24.h, z24.s\n"
    ".inst 0x45304339  // sqxtnb z25.h, z25.s\n"
    ".inst 0x4530462d  // sqxtnt z13.h, z17.s\n"
    ".inst 0x4530455a  // sqxtnt z26.h, z10.s\n"
    ".inst 0x45304618  // sqxtnt z24.h, z16.s\n"
    ".inst 0x45304539  // sqxtnt z25.h, z9.s\n"
    "sqadd z13.h, z13.h, z14.h\n"
    "smax z13.h, p4/M, z13.h, z12.h\n"
    "smin z13.h, p4/M, z13.h, z11.h\n"
    "sqadd z26.h, z26.h, z14.h\n"
    "sqadd z24.h, z24.h, z14.h\n"
    "smax z26.h, p4/M, z26.h, z12.h\n"
    "smax z24.h, p4/M, z24.h, z12.h\n"
    "sqadd z25.h, z25.h, z14.h\n"
    "smax z25.h, p4/M, z25.h, z12.h\n"
    "smin z26.h, p4/M, z26.h, z11.h\n"
    "st1b { z13.h }, p0, [x15, x10]\n"
    "smin z24.h, p4/M, z24.h, z11.h\n"
    "smin z25.h, p4/M, z25.h, z11.h\n"
    "st1b { z26.h }, p0, [x14, x10]\n"
    "st1b { z24.h }, p0, [x13, x10]\n"
    "st1b { z25.h }, p0, [x12, x10]\n"
    "ld1sb { z0.h }, p4/Z, [x16]\n"
    "ld1sb { z1.h }, p4/Z, [x16, #1, MUL VL]\n"
    "inch x10\n"
    "ld1sb { z2.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1sb { z3.h }, p4/Z, [x16, #3, MUL VL]\n"
    ".inst 0x454f1000  // ssublb z0.h, z0.b, z15.b\n"
    ".inst 0x454f1021  // ssublb z1.h, z1.b, z15.b\n"
    "ld1sb { z4.h }, p4/Z, [x16, #4, MUL VL]\n"
    "ld1sb { z5.h }, p4/Z, [x16, #5, MUL VL]\n"
    ".inst 0x454f1042  // ssublb z2.h, z2.b, z15.b\n"
    ".inst 0x454f1063  // ssublb z3.h, z3.b, z15.b\n"
    "ld1sb { z6.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1sb { z7.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454f1084  // ssublb z4.h, z4.b, z15.b\n"
    "ld1w { z17.s }, p2/Z, [x26]\n"
    "ld1w { z16.s }, p1/Z, [x26, #1, MUL VL]\n"
    "uzp1 z13.s, z17.s, z16.s\n"
    "uzp2 z17.s, z17.s, z16.s\n"
    "ld1sb { z8.h }, p4/Z, [x16]\n"
    "ldp x24, x23, [x11, #0x0]\n"
    "addvl x26, x26, #2\n"
    "str x26, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x22, x21, [x11, #0x10]\n"
    "ldr x20, [x11, #0x20]\n"
    "mov z26.d, z13.d\n"
    "mov z10.d, z17.d\n"
    "ld1b { z31.h }, p3/Z, [x24, x8]\n"
    "ld1b { z30.h }, p3/Z, [x23, x8]\n"
    "mov z24.d, z13.d\n"
    "mov z16.d, z17.d\n"
    "ld1b { z29.h }, p3/Z, [x22, x8]\n"
    "ld1b { z28.h }, p3/Z, [x21, x8]\n"
    "mov z25.d, z13.d\n"
    "mov z9.d, z17.d\n"
    "ld1b { z27.h }, p3/Z, [x20, x8]\n"
    ".inst 0x454f10a5  // ssublb z5.h, z5.b, z15.b\n"
    ".inst 0x454f10c6  // ssublb z6.h, z6.b, z15.b\n"
    ".inst 0x454f10e7  // ssublb z7.h, z7.b, z15.b\n"
    ".inst 0x454f1108  // ssublb z8.h, z8.b, z15.b\n"
    ".inst 0x45571bff  // usublb z31.h, z31.b, z23.b\n"
    ".inst 0x45571bde  // usublb z30.h, z30.b, z23.b\n"
    ".inst 0x45571bbd  // usublb z29.h, z29.b, z23.b\n"
    ".inst 0x45571b9c  // usublb z28.h, z28.b, z23.b\n"
    ".inst 0x45571b7b  // usublb z27.h, z27.b, z23.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(ARM_COMPUTE_ENABLE_SVE)
