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
    uint64_t n_channels;
    const void *weights;
    const int32_t *bias;
    const arm_gemm::Requantize32 *requant;
    const int32_t *const requant_muls;
    const int32_t *const requant_shifts;
    uint8_t *const *const outptrs;
    const uint8_t *inptrs[25];

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
    "mov x8, #0x0\n"
    "ldr x27, [%x[params], %[offsetof_Params_requant]]\n"
    "ptrue p4.b\n"
    "ldr x26, [%x[params], %[offsetof_Params_outptrs]]\n"
    "ldr x17, [%x[params], %[offsetof_Params_n_channels]]\n"
    "ldr x16, [%x[params], %[offsetof_Params_weights]]\n"
    "add x15, %x[params], %[offsetof_Params_inptrs]\n"
    "mov x14, #0x0\n"
    "ldr x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldr x13, [%x[params], %[offsetof_Params_requant_muls]]\n"
    "mov x24, x8\n"
    "add x20, x27, %[offsetof_Requantize32_a_offset]\n"
    "add x23, x27, %[offsetof_Requantize32_b_offset]\n"
    "add x22, x27, %[offsetof_Requantize32_c_offset]\n"
    "ld1rb { z17.b }, p4/Z, [x20]\n"
    "ldr x12, [%x[params], %[offsetof_Params_requant_shifts]]\n"
    "add x21, x27, %[offsetof_Requantize32_minval]\n"
    "add x20, x27, %[offsetof_Requantize32_maxval]\n"
    "ld1rb { z12.b }, p4/Z, [x23]\n"
    "ld1rh { z25.h }, p4/Z, [x22]\n"
    "ld1rh { z14.h }, p4/Z, [x21]\n"
    "ld1rh { z9.h }, p4/Z, [x20]\n"
    "incw x24\n"
    "whilelt p3.h, x8, x17\n"
    "ldp x11, x10, [x26, #0x0]\n"
    "ldp x9, x28, [x26, #0x10]\n"
    "whilelt p2.s, x8, x17\n"
    "whilelt p1.s, x24, x17\n"
    "ld1sb { z28.h }, p4/Z, [x16]\n"
    "ld1sb { z20.h }, p4/Z, [x16, #1, MUL VL]\n"
    "ld1sb { z13.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1sb { z18.h }, p4/Z, [x16, #3, MUL VL]\n"
    "ld1sb { z6.h }, p4/Z, [x16, #4, MUL VL]\n"
    "ld1sb { z2.h }, p4/Z, [x16, #5, MUL VL]\n"
    "ld1sb { z26.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1sb { z21.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    "ld1w { z11.s }, p2/Z, [x25]\n"
    "ld1w { z4.s }, p1/Z, [x25, #1, MUL VL]\n"
    "addvl x25, x25, #2\n"
    ".inst 0x454c1294  // ssublb z20.h, z20.b, z12.b\n"
    ".inst 0x454c11ad  // ssublb z13.h, z13.b, z12.b\n"
    ".inst 0x454c1252  // ssublb z18.h, z18.b, z12.b\n"
    "ld1sb { z15.h }, p4/Z, [x16]\n"
    "ldp x27, x26, [x15, #0x0]\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    "uzp1 z5.s, z11.s, z4.s\n"
    "uzp2 z11.s, z11.s, z4.s\n"
    "str x25, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x15, #0x10]\n"
    ".inst 0x454c135a  // ssublb z26.h, z26.b, z12.b\n"
    ".inst 0x454c12b5  // ssublb z21.h, z21.b, z12.b\n"
    ".inst 0x454c11ef  // ssublb z15.h, z15.b, z12.b\n"
    "ldp x23, x22, [x15, #0x20]\n"
    "mov z30.d, z5.d\n"
    "mov z16.d, z11.d\n"
    "mov z4.d, z5.d\n"
    "mov z8.d, z11.d\n"
    "mov z31.d, z5.d\n"
    "ldp x21, x20, [x15, #0x30]\n"
    "mov z10.d, z11.d\n"
    "ld1b { z3.h }, p3/Z, [x27, x8]\n"
    "ld1b { z29.h }, p3/Z, [x26, x8]\n"
    "ld1b { z23.h }, p3/Z, [x25, x8]\n"
    "ld1b { z0.h }, p3/Z, [x24, x8]\n"
    "ld1b { z24.h }, p3/Z, [x23, x8]\n"
    "ld1b { z22.h }, p3/Z, [x22, x8]\n"
    "ld1b { z27.h }, p3/Z, [x21, x8]\n"
    "ld1b { z19.h }, p3/Z, [x20, x8]\n"
    ".inst 0x45511863  // usublb z3.h, z3.b, z17.b\n"
    ".inst 0x45511bbd  // usublb z29.h, z29.b, z17.b\n"
    ".inst 0x45511af7  // usublb z23.h, z23.b, z17.b\n"
    ".inst 0x45511800  // usublb z0.h, z0.b, z17.b\n"
    ".inst 0x45511b18  // usublb z24.h, z24.b, z17.b\n"
    ".inst 0x45511ad6  // usublb z22.h, z22.b, z17.b\n"
    ".inst 0x45511b7b  // usublb z27.h, z27.b, z17.b\n"
    ".inst 0x45511a73  // usublb z19.h, z19.b, z17.b\n"
    "1:"  // Loop
    ".inst 0x448f4065  // smlalb z5.s, p4/M, z3.h, z15.h\n"
    "ldr x25, [x15, #0x58]\n"
    "ldr x24, [x15, #0x78]\n"
    ".inst 0x448f446b  // smlalt z11.s, p4/M, z3.h, z15.h\n"
    "ldr x23, [x15, #0x60]\n"
    "ldr x22, [x15, #0x80]\n"
    ".inst 0x449a407e  // smlalb z30.s, p4/M, z3.h, z26.h\n"
    ".inst 0x448d4064  // smlalb z4.s, p4/M, z3.h, z13.h\n"
    ".inst 0x449c407f  // smlalb z31.s, p4/M, z3.h, z28.h\n"
    ".inst 0x449a4470  // smlalt z16.s, p4/M, z3.h, z26.h\n"
    "ldr x21, [x15, #0x68]\n"
    "ldr x20, [x15, #0x88]\n"
    "ld1b { z1.h }, p3/Z, [x25, x8]\n"
    "ld1b { z7.h }, p3/Z, [x24, x8]\n"
    ".inst 0x448d4468  // smlalt z8.s, p4/M, z3.h, z13.h\n"
    ".inst 0x449c446a  // smlalt z10.s, p4/M, z3.h, z28.h\n"
    ".inst 0x449c43a5  // smlalb z5.s, p4/M, z29.h, z28.h\n"
    ".inst 0x449c47ab  // smlalt z11.s, p4/M, z29.h, z28.h\n"
    "ld1b { z29.h }, p3/Z, [x23, x8]\n"
    "ld1b { z3.h }, p3/Z, [x22, x8]\n"
    ".inst 0x4494401e  // smlalb z30.s, p4/M, z0.h, z20.h\n"
    "ldr x25, [x15, #0x40]\n"
    "ldr x24, [x15, #0x70]\n"
    "whilelt p0.h, x14, x17\n"
    ".inst 0x45511821  // usublb z1.h, z1.b, z17.b\n"
    ".inst 0x455118e7  // usublb z7.h, z7.b, z17.b\n"
    ".inst 0x44944410  // smlalt z16.s, p4/M, z0.h, z20.h\n"
    "ld1b { z0.h }, p3/Z, [x21, x8]\n"
    ".inst 0x45511bbd  // usublb z29.h, z29.b, z17.b\n"
    ".inst 0x45511863  // usublb z3.h, z3.b, z17.b\n"
    "ldr x23, [x15, #0x98]\n"
    "ldr x22, [x15, #0x50]\n"
    ".inst 0x449442e5  // smlalb z5.s, p4/M, z23.h, z20.h\n"
    ".inst 0x449446eb  // smlalt z11.s, p4/M, z23.h, z20.h\n"
    "ld1b { z23.h }, p3/Z, [x20, x8]\n"
    "ldr x21, [x15, #0x48]\n"
    ".inst 0x44924024  // smlalb z4.s, p4/M, z1.h, z18.h\n"
    ".inst 0x448640ff  // smlalb z31.s, p4/M, z7.h, z6.h\n"
    ".inst 0x45511800  // usublb z0.h, z0.b, z17.b\n"
    "ldr x20, [x15, #0x90]\n"
    ".inst 0x44924428  // smlalt z8.s, p4/M, z1.h, z18.h\n"
    ".inst 0x448644ea  // smlalt z10.s, p4/M, z7.h, z6.h\n"
    "ld1b { z1.h }, p3/Z, [x25, x8]\n"
    "ld1b { z7.h }, p3/Z, [x24, x8]\n"
    ".inst 0x448d431e  // smlalb z30.s, p4/M, z24.h, z13.h\n"
    ".inst 0x45511af7  // usublb z23.h, z23.b, z17.b\n"
    ".inst 0x448d4710  // smlalt z16.s, p4/M, z24.h, z13.h\n"
    "ld1b { z24.h }, p3/Z, [x23, x8]\n"
    ".inst 0x449242c5  // smlalb z5.s, p4/M, z22.h, z18.h\n"
    ".inst 0x449246cb  // smlalt z11.s, p4/M, z22.h, z18.h\n"
    "ldr x24, [x15, #0xa8]\n"
    "ld1b { z22.h }, p3/Z, [x22, x8]\n"
    ".inst 0x449c43a4  // smlalb z4.s, p4/M, z29.h, z28.h\n"
    ".inst 0x4494407f  // smlalb z31.s, p4/M, z3.h, z20.h\n"
    ".inst 0x45511821  // usublb z1.h, z1.b, z17.b\n"
    "ldr x23, [x15, #0xa0]\n"
    ".inst 0x449c47a8  // smlalt z8.s, p4/M, z29.h, z28.h\n"
    ".inst 0x4494446a  // smlalt z10.s, p4/M, z3.h, z20.h\n"
    ".inst 0x455118e7  // usublb z7.h, z7.b, z17.b\n"
    "ldr x22, [x15, #0xb0]\n"
    ".inst 0x449c427e  // smlalb z30.s, p4/M, z19.h, z28.h\n"
    ".inst 0x45511b18  // usublb z24.h, z24.b, z17.b\n"
    ".inst 0x449c4670  // smlalt z16.s, p4/M, z19.h, z28.h\n"
    "ld1b { z28.h }, p3/Z, [x21, x8]\n"
    ".inst 0x44864365  // smlalb z5.s, p4/M, z27.h, z6.h\n"
    ".inst 0x4486476b  // smlalt z11.s, p4/M, z27.h, z6.h\n"
    "ld1b { z27.h }, p3/Z, [x20, x8]\n"
    ".inst 0x45511ad6  // usublb z22.h, z22.b, z17.b\n"
    ".inst 0x44864004  // smlalb z4.s, p4/M, z0.h, z6.h\n"
    ".inst 0x448242ff  // smlalb z31.s, p4/M, z23.h, z2.h\n"
    "ldr x21, [x15, #0xb8]\n"
    "ldr x20, [x15, #0xc0]\n"
    ".inst 0x44864408  // smlalt z8.s, p4/M, z0.h, z6.h\n"
    "ld1b { z0.h }, p3/Z, [x24, x8]\n"
    ".inst 0x448246ea  // smlalt z10.s, p4/M, z23.h, z2.h\n"
    ".inst 0x45511b9c  // usublb z28.h, z28.b, z17.b\n"
    ".inst 0x4486403e  // smlalb z30.s, p4/M, z1.h, z6.h\n"
    ".inst 0x45511b7b  // usublb z27.h, z27.b, z17.b\n"
    "ld1b { z23.h }, p3/Z, [x23, x8]\n"
    ".inst 0x44864430  // smlalt z16.s, p4/M, z1.h, z6.h\n"
    ".inst 0x448d4265  // smlalb z5.s, p4/M, z19.h, z13.h\n"
    ".inst 0x448d466b  // smlalt z11.s, p4/M, z19.h, z13.h\n"
    "ld1b { z6.h }, p3/Z, [x22, x8]\n"
    "ld1b { z1.h }, p3/Z, [x21, x8]\n"
    ".inst 0x449440e4  // smlalb z4.s, p4/M, z7.h, z20.h\n"
    ".inst 0x448d431f  // smlalb z31.s, p4/M, z24.h, z13.h\n"
    ".inst 0x45511800  // usublb z0.h, z0.b, z17.b\n"
    "ld1w { z19.s }, p2/Z, [x13]\n"
    ".inst 0x449444e8  // smlalt z8.s, p4/M, z7.h, z20.h\n"
    ".inst 0x448d470a  // smlalt z10.s, p4/M, z24.h, z13.h\n"
    ".inst 0x45511af7  // usublb z23.h, z23.b, z17.b\n"
    "ld1w { z20.s }, p1/Z, [x13, #1, MUL VL]\n"
    ".inst 0x4482439e  // smlalb z30.s, p4/M, z28.h, z2.h\n"
    ".inst 0x455118c6  // usublb z6.h, z6.b, z17.b\n"
    ".inst 0x44824790  // smlalt z16.s, p4/M, z28.h, z2.h\n"
    "ld1b { z13.h }, p3/Z, [x20, x8]\n"
    ".inst 0x448242c5  // smlalb z5.s, p4/M, z22.h, z2.h\n"
    ".inst 0x448246cb  // smlalt z11.s, p4/M, z22.h, z2.h\n"
    ".inst 0x45511821  // usublb z1.h, z1.b, z17.b\n"
    "inch x8\n"
    ".inst 0x449a4364  // smlalb z4.s, p4/M, z27.h, z26.h\n"
    ".inst 0x4492401f  // smlalb z31.s, p4/M, z0.h, z18.h\n"
    "uzp1 z28.s, z19.s, z20.s\n"
    "inch x16\n"
    ".inst 0x449a4768  // smlalt z8.s, p4/M, z27.h, z26.h\n"
    ".inst 0x4492440a  // smlalt z10.s, p4/M, z0.h, z18.h\n"
    "uzp2 z20.s, z19.s, z20.s\n"
    "ld1w { z27.s }, p2/Z, [x12]\n"
    ".inst 0x449242de  // smlalb z30.s, p4/M, z22.h, z18.h\n"
    ".inst 0x449246d0  // smlalt z16.s, p4/M, z22.h, z18.h\n"
    "ld1w { z19.s }, p1/Z, [x12, #1, MUL VL]\n"
    ".inst 0x455119ad  // usublb z13.h, z13.b, z17.b\n"
    ".inst 0x449a43a5  // smlalb z5.s, p4/M, z29.h, z26.h\n"
    ".inst 0x449a47ab  // smlalt z11.s, p4/M, z29.h, z26.h\n"
    "mov x21, x8\n"
    "whilelt p2.s, x8, x17\n"
    ".inst 0x449542e4  // smlalb z4.s, p4/M, z23.h, z21.h\n"
    ".inst 0x449540df  // smlalb z31.s, p4/M, z6.h, z21.h\n"
    "ldr x20, [%x[params], %[offsetof_Params_bias]]\n"
    "addvl x13, x13, #2\n"
    ".inst 0x449546e8  // smlalt z8.s, p4/M, z23.h, z21.h\n"
    ".inst 0x449544ca  // smlalt z10.s, p4/M, z6.h, z21.h\n"
    "uzp1 z23.s, z27.s, z19.s\n"
    "addvl x12, x12, #2\n"
    ".inst 0x4495407e  // smlalb z30.s, p4/M, z3.h, z21.h\n"
    ".inst 0x44954470  // smlalt z16.s, p4/M, z3.h, z21.h\n"
    "uzp2 z6.s, z27.s, z19.s\n"
    "incw x21\n"
    ".inst 0x449540e5  // smlalb z5.s, p4/M, z7.h, z21.h\n"
    ".inst 0x449544eb  // smlalt z11.s, p4/M, z7.h, z21.h\n"
    ".inst 0x44824004  // smlalb z4.s, p4/M, z0.h, z2.h\n"
    ".inst 0x449a403f  // smlalb z31.s, p4/M, z1.h, z26.h\n"
    ".inst 0x44824408  // smlalt z8.s, p4/M, z0.h, z2.h\n"
    ".inst 0x449a442a  // smlalt z10.s, p4/M, z1.h, z26.h\n"
    "whilelt p1.s, x21, x17\n"
    "whilelt p3.h, x8, x17\n"
    ".inst 0x448f431e  // smlalb z30.s, p4/M, z24.h, z15.h\n"
    ".inst 0x448f4710  // smlalt z16.s, p4/M, z24.h, z15.h\n"
    ".inst 0x04bc74a5  // sqrdmulh z5.s, z5.s, z28.s\n"
    ".inst 0x04b4756b  // sqrdmulh z11.s, z11.s, z20.s\n"
    ".inst 0x448f4024  // smlalb z4.s, p4/M, z1.h, z15.h\n"
    ".inst 0x448f41bf  // smlalb z31.s, p4/M, z13.h, z15.h\n"
    "and z24.d, z5.d, z23.d\n"
    ".inst 0x448f4428  // smlalt z8.s, p4/M, z1.h, z15.h\n"
    ".inst 0x448f45aa  // smlalt z10.s, p4/M, z13.h, z15.h\n"
    "and z19.d, z11.d, z6.d\n"
    ".inst 0x04bc77de  // sqrdmulh z30.s, z30.s, z28.s\n"
    ".inst 0x04b47610  // sqrdmulh z16.s, z16.s, z20.s\n"
    "asr z24.s, z24.s, #0x1f\n"
    ".inst 0x04bc7484  // sqrdmulh z4.s, z4.s, z28.s\n"
    ".inst 0x04bc77ff  // sqrdmulh z31.s, z31.s, z28.s\n"
    "asr z19.s, z19.s, #0x1f\n"
    "and z7.d, z30.d, z23.d\n"
    "sqadd z5.s, z5.s, z24.s\n"
    ".inst 0x04b47508  // sqrdmulh z8.s, z8.s, z20.s\n"
    "and z15.d, z4.d, z23.d\n"
    "and z24.d, z31.d, z23.d\n"
    ".inst 0x04b4754a  // sqrdmulh z10.s, z10.s, z20.s\n"
    "sqadd z11.s, z11.s, z19.s\n"
    "asr z7.s, z7.s, #0x1f\n"
    "and z18.d, z16.d, z6.d\n"
    ".inst 0x448292e5  // srshl z5.s, p4/M, z5.s, z23.s\n"
    "asr z15.s, z15.s, #0x1f\n"
    "and z13.d, z8.d, z6.d\n"
    "asr z24.s, z24.s, #0x1f\n"
    "and z3.d, z10.d, z6.d\n"
    ".inst 0x448290cb  // srshl z11.s, p4/M, z11.s, z6.s\n"
    "sqadd z30.s, z30.s, z7.s\n"
    "asr z18.s, z18.s, #0x1f\n"
    "sqadd z4.s, z4.s, z15.s\n"
    "asr z13.s, z13.s, #0x1f\n"
    "sqadd z31.s, z31.s, z24.s\n"
    "asr z3.s, z3.s, #0x1f\n"
    ".inst 0x448292fe  // srshl z30.s, p4/M, z30.s, z23.s\n"
    "sqadd z16.s, z16.s, z18.s\n"
    ".inst 0x453040a5  // sqxtnb z5.h, z5.s\n"
    ".inst 0x448292e4  // srshl z4.s, p4/M, z4.s, z23.s\n"
    "sqadd z8.s, z8.s, z13.s\n"
    ".inst 0x448292ff  // srshl z31.s, p4/M, z31.s, z23.s\n"
    "sqadd z10.s, z10.s, z3.s\n"
    ".inst 0x453043de  // sqxtnb z30.h, z30.s\n"
    ".inst 0x448290d0  // srshl z16.s, p4/M, z16.s, z6.s\n"
    ".inst 0x45304084  // sqxtnb z4.h, z4.s\n"
    ".inst 0x45304565  // sqxtnt z5.h, z11.s\n"
    ".inst 0x448290c8  // srshl z8.s, p4/M, z8.s, z6.s\n"
    ".inst 0x448290ca  // srshl z10.s, p4/M, z10.s, z6.s\n"
    ".inst 0x453043ff  // sqxtnb z31.h, z31.s\n"
    ".inst 0x4530461e  // sqxtnt z30.h, z16.s\n"
    ".inst 0x45304504  // sqxtnt z4.h, z8.s\n"
    ".inst 0x4530455f  // sqxtnt z31.h, z10.s\n"
    "sqadd z5.h, z5.h, z25.h\n"
    "sqadd z30.h, z30.h, z25.h\n"
    "sqadd z4.h, z4.h, z25.h\n"
    "sqadd z31.h, z31.h, z25.h\n"
    "smax z5.h, p4/M, z5.h, z14.h\n"
    "smax z30.h, p4/M, z30.h, z14.h\n"
    "smax z4.h, p4/M, z4.h, z14.h\n"
    "smax z31.h, p4/M, z31.h, z14.h\n"
    "smin z5.h, p4/M, z5.h, z9.h\n"
    "smin z30.h, p4/M, z30.h, z9.h\n"
    "smin z4.h, p4/M, z4.h, z9.h\n"
    "smin z31.h, p4/M, z31.h, z9.h\n"
    "st1b { z5.h }, p0, [x11, x14]\n"
    "st1b { z30.h }, p0, [x10, x14]\n"
    "st1b { z4.h }, p0, [x9, x14]\n"
    "st1b { z31.h }, p0, [x28, x14]\n"
    "inch x14\n"
    "ld1sb { z28.h }, p4/Z, [x16]\n"
    "ld1sb { z20.h }, p4/Z, [x16, #1, MUL VL]\n"
    "ld1sb { z13.h }, p4/Z, [x16, #2, MUL VL]\n"
    "ld1sb { z18.h }, p4/Z, [x16, #3, MUL VL]\n"
    "ld1sb { z6.h }, p4/Z, [x16, #4, MUL VL]\n"
    "ld1sb { z2.h }, p4/Z, [x16, #5, MUL VL]\n"
    "ld1sb { z26.h }, p4/Z, [x16, #6, MUL VL]\n"
    "ld1sb { z21.h }, p4/Z, [x16, #7, MUL VL]\n"
    "inch x16, ALL, MUL #8\n"
    ".inst 0x454c139c  // ssublb z28.h, z28.b, z12.b\n"
    "ld1w { z10.s }, p2/Z, [x20]\n"
    "ld1w { z1.s }, p1/Z, [x20, #1, MUL VL]\n"
    "addvl x20, x20, #2\n"
    ".inst 0x454c1294  // ssublb z20.h, z20.b, z12.b\n"
    ".inst 0x454c11ad  // ssublb z13.h, z13.b, z12.b\n"
    ".inst 0x454c1252  // ssublb z18.h, z18.b, z12.b\n"
    "ld1sb { z15.h }, p4/Z, [x16]\n"
    "ldp x27, x26, [x15, #0x0]\n"
    ".inst 0x454c10c6  // ssublb z6.h, z6.b, z12.b\n"
    ".inst 0x454c1042  // ssublb z2.h, z2.b, z12.b\n"
    "uzp1 z5.s, z10.s, z1.s\n"
    "uzp2 z11.s, z10.s, z1.s\n"
    "str x20, [%x[params], %[offsetof_Params_bias]]\n"
    "ldp x25, x24, [x15, #0x10]\n"
    ".inst 0x454c135a  // ssublb z26.h, z26.b, z12.b\n"
    ".inst 0x454c12b5  // ssublb z21.h, z21.b, z12.b\n"
    ".inst 0x454c11ef  // ssublb z15.h, z15.b, z12.b\n"
    "ldp x23, x22, [x15, #0x20]\n"
    "mov z30.d, z5.d\n"
    "mov z16.d, z11.d\n"
    "mov z4.d, z5.d\n"
    "mov z8.d, z11.d\n"
    "mov z31.d, z5.d\n"
    "ldp x21, x20, [x15, #0x30]\n"
    "mov z10.d, z11.d\n"
    "ld1b { z3.h }, p3/Z, [x27, x8]\n"
    "ld1b { z29.h }, p3/Z, [x26, x8]\n"
    "ld1b { z23.h }, p3/Z, [x25, x8]\n"
    "ld1b { z0.h }, p3/Z, [x24, x8]\n"
    "ld1b { z24.h }, p3/Z, [x23, x8]\n"
    "ld1b { z22.h }, p3/Z, [x22, x8]\n"
    "ld1b { z27.h }, p3/Z, [x21, x8]\n"
    "ld1b { z19.h }, p3/Z, [x20, x8]\n"
    ".inst 0x45511863  // usublb z3.h, z3.b, z17.b\n"
    ".inst 0x45511bbd  // usublb z29.h, z29.b, z17.b\n"
    ".inst 0x45511af7  // usublb z23.h, z23.b, z17.b\n"
    ".inst 0x45511800  // usublb z0.h, z0.b, z17.b\n"
    ".inst 0x45511b18  // usublb z24.h, z24.b, z17.b\n"
    ".inst 0x45511ad6  // usublb z22.h, z22.b, z17.b\n"
    ".inst 0x45511b7b  // usublb z27.h, z27.b, z17.b\n"
    ".inst 0x45511a73  // usublb z19.h, z19.b, z17.b\n"
    "b.any 1b\n"
    :
    : [offsetof_Params_bias] "I" (offsetof(Params, bias)), [offsetof_Params_inptrs] "I" (offsetof(Params, inptrs)), [offsetof_Params_n_channels] "I" (offsetof(Params, n_channels)), [offsetof_Params_outptrs] "I" (offsetof(Params, outptrs)), [offsetof_Params_requant] "I" (offsetof(Params, requant)), [offsetof_Params_requant_muls] "I" (offsetof(Params, requant_muls)), [offsetof_Params_requant_shifts] "I" (offsetof(Params, requant_shifts)), [offsetof_Params_weights] "I" (offsetof(Params, weights)), [offsetof_Requantize32_a_offset] "I" (offsetof(arm_gemm::Requantize32, a_offset)), [offsetof_Requantize32_b_offset] "I" (offsetof(arm_gemm::Requantize32, b_offset)), [offsetof_Requantize32_c_offset] "I" (offsetof(arm_gemm::Requantize32, c_offset)), [offsetof_Requantize32_maxval] "I" (offsetof(arm_gemm::Requantize32, maxval)), [offsetof_Requantize32_minval] "I" (offsetof(arm_gemm::Requantize32, minval)), [params] "r" (&params)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace depthwise
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
