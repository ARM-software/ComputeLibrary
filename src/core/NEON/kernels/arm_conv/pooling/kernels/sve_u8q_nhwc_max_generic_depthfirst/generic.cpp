/*
 * Copyright (c) 2021-2024 Arm Limited.
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

#include "pooling.hpp"
#include <cstdint>
#include <cstddef>

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace pooling {


void sve_u8q_nhwc_max_generic_depthfirst_impl(
  const uint64_t,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const uint8_t *const *const inptrs,
  uint8_t *outptr,
  const Requantize32 &qp
)
{
  __asm__ __volatile__(
    "mov x9, #0x0\n"
    "cntb x28\n"
    "cntb x27, ALL, MUL #2\n"
    "cntb x26, ALL, MUL #3\n"
    "ptrue p4.b\n"
    "whilelt p3.b, x9, %x[n_channels]\n"
    "whilelt p2.b, x28, %x[n_channels]\n"
    "whilelt p1.b, x27, %x[n_channels]\n"
    "whilelt p0.b, x26, %x[n_channels]\n"
    "b.none 7f\n"
    "1:"  // 4-vectors of channels
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "mov z6.b, #0x0\n"
    "mov z5.b, #0x0\n"
    "mov x24, %x[inptrs]\n"
    "mov z4.b, #0x0\n"
    "mov z3.b, #0x0\n"
    "cbz x25, 4f\n"
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "subs x25, x25, #0x1\n"
    "add x24, x24, #0x20\n"
    "ld1b { z2.b }, p3/Z, [x23, x9]\n"
    "ld1b { z1.b }, p3/Z, [x22, x9]\n"
    "ld1b { z23.b }, p3/Z, [x21, x9]\n"
    "ld1b { z0.b }, p3/Z, [x20, x9]\n"
    "ld1b { z31.b }, p2/Z, [x23, x28]\n"
    "ld1b { z30.b }, p2/Z, [x22, x28]\n"
    "ld1b { z22.b }, p2/Z, [x21, x28]\n"
    "ld1b { z29.b }, p2/Z, [x20, x28]\n"
    "ld1b { z28.b }, p1/Z, [x23, x27]\n"
    "ld1b { z27.b }, p1/Z, [x22, x27]\n"
    "ld1b { z21.b }, p1/Z, [x21, x27]\n"
    "ld1b { z26.b }, p1/Z, [x20, x27]\n"
    "ld1b { z16.b }, p0/Z, [x23, x26]\n"
    "ld1b { z25.b }, p0/Z, [x22, x26]\n"
    "ld1b { z20.b }, p0/Z, [x21, x26]\n"
    "ld1b { z24.b }, p0/Z, [x20, x26]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 4 inputs loop
    "movprfx z19, z2\n umax z19.b, p4/M, z19.b, z1.b\n"
    "umax z23.b, p4/M, z23.b, z0.b\n"
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "movprfx z18, z31\n umax z18.b, p4/M, z18.b, z30.b\n"
    "umax z22.b, p4/M, z22.b, z29.b\n"
    "movprfx z17, z28\n umax z17.b, p4/M, z17.b, z27.b\n"
    "umax z21.b, p4/M, z21.b, z26.b\n"
    "umax z16.b, p4/M, z16.b, z25.b\n"
    "umax z20.b, p4/M, z20.b, z24.b\n"
    "ld1b { z2.b }, p3/Z, [x23, x9]\n"
    "ld1b { z1.b }, p3/Z, [x22, x9]\n"
    "umax z19.b, p4/M, z19.b, z23.b\n"
    "umax z18.b, p4/M, z18.b, z22.b\n"
    "ld1b { z23.b }, p3/Z, [x21, x9]\n"
    "ld1b { z0.b }, p3/Z, [x20, x9]\n"
    "umax z17.b, p4/M, z17.b, z21.b\n"
    "subs x25, x25, #0x1\n"
    "ld1b { z31.b }, p2/Z, [x23, x28]\n"
    "ld1b { z30.b }, p2/Z, [x22, x28]\n"
    "umax z16.b, p4/M, z16.b, z20.b\n"
    "add x24, x24, #0x20\n"
    "ld1b { z22.b }, p2/Z, [x21, x28]\n"
    "ld1b { z29.b }, p2/Z, [x20, x28]\n"
    "umax z6.b, p4/M, z6.b, z19.b\n"
    "umax z5.b, p4/M, z5.b, z18.b\n"
    "ld1b { z28.b }, p1/Z, [x23, x27]\n"
    "ld1b { z27.b }, p1/Z, [x22, x27]\n"
    "umax z4.b, p4/M, z4.b, z17.b\n"
    "ld1b { z21.b }, p1/Z, [x21, x27]\n"
    "ld1b { z26.b }, p1/Z, [x20, x27]\n"
    "umax z3.b, p4/M, z3.b, z16.b\n"
    "ld1b { z16.b }, p0/Z, [x23, x26]\n"
    "ld1b { z25.b }, p0/Z, [x22, x26]\n"
    "ld1b { z20.b }, p0/Z, [x21, x26]\n"
    "ld1b { z24.b }, p0/Z, [x20, x26]\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "movprfx z19, z2\n umax z19.b, p4/M, z19.b, z1.b\n"
    "umax z23.b, p4/M, z23.b, z0.b\n"
    "movprfx z18, z31\n umax z18.b, p4/M, z18.b, z30.b\n"
    "umax z22.b, p4/M, z22.b, z29.b\n"
    "movprfx z17, z28\n umax z17.b, p4/M, z17.b, z27.b\n"
    "umax z21.b, p4/M, z21.b, z26.b\n"
    "umax z16.b, p4/M, z16.b, z25.b\n"
    "umax z20.b, p4/M, z20.b, z24.b\n"
    "umax z19.b, p4/M, z19.b, z23.b\n"
    "umax z18.b, p4/M, z18.b, z22.b\n"
    "umax z17.b, p4/M, z17.b, z21.b\n"
    "umax z16.b, p4/M, z16.b, z20.b\n"
    "umax z6.b, p4/M, z6.b, z19.b\n"
    "umax z5.b, p4/M, z5.b, z18.b\n"
    "umax z4.b, p4/M, z4.b, z17.b\n"
    "umax z3.b, p4/M, z3.b, z16.b\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x20, [x24], #0x8\n"
    "subs x21, x21, #0x1\n"
    "ld1b { z19.b }, p3/Z, [x20, x9]\n"
    "ld1b { z18.b }, p2/Z, [x20, x28]\n"
    "ld1b { z17.b }, p1/Z, [x20, x27]\n"
    "ld1b { z16.b }, p0/Z, [x20, x26]\n"
    "umax z6.b, p4/M, z6.b, z19.b\n"
    "umax z5.b, p4/M, z5.b, z18.b\n"
    "umax z4.b, p4/M, z4.b, z17.b\n"
    "umax z3.b, p4/M, z3.b, z16.b\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "add x21, %x[quant_params], %[offsetof_qp_input_offset]\n"
    ".inst 0x4508a8d3  // ushllb z19.h, z6.b, #0x0\n"
    ".inst 0x4508acd1  // ushllt z17.h, z6.b, #0x0\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_left_shift]\n"
    "ld1rw { z6.s }, p4/Z, [x21]\n"
    ".inst 0x4508a8b2  // ushllb z18.h, z5.b, #0x0\n"
    ".inst 0x4508acb0  // ushllt z16.h, z5.b, #0x0\n"
    "ld1rw { z5.s }, p4/Z, [x20]\n"
    ".inst 0x4508a894  // ushllb z20.h, z4.b, #0x0\n"
    ".inst 0x4508ac98  // ushllt z24.h, z4.b, #0x0\n"
    "add x21, %x[quant_params], %[offsetof_qp_per_layer_mul]\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_right_shift]\n"
    ".inst 0x4508a877  // ushllb z23.h, z3.b, #0x0\n"
    ".inst 0x4508ac76  // ushllt z22.h, z3.b, #0x0\n"
    "ld1rw { z4.s }, p4/Z, [x21]\n"
    "ld1rw { z3.s }, p4/Z, [x20]\n"
    "neg z6.s, p4/M, z6.s\n"
    "add x20, %x[quant_params], %[offsetof_qp_output_offset]\n"
    "mov z2.s, #0x0\n"
    "mov z1.s, #0xff\n"
    "ld1rw { z0.s }, p4/Z, [x20]\n"
    ".inst 0x459340df  // saddwb z31.s, z6.s, z19.h\n"
    ".inst 0x459344d3  // saddwt z19.s, z6.s, z19.h\n"
    ".inst 0x459140de  // saddwb z30.s, z6.s, z17.h\n"
    ".inst 0x459144d1  // saddwt z17.s, z6.s, z17.h\n"
    ".inst 0x459240dd  // saddwb z29.s, z6.s, z18.h\n"
    ".inst 0x459244d2  // saddwt z18.s, z6.s, z18.h\n"
    ".inst 0x459040dc  // saddwb z28.s, z6.s, z16.h\n"
    ".inst 0x459044d0  // saddwt z16.s, z6.s, z16.h\n"
    ".inst 0x448290bf  // srshl z31.s, p4/M, z31.s, z5.s\n"
    ".inst 0x448290b3  // srshl z19.s, p4/M, z19.s, z5.s\n"
    ".inst 0x459440d5  // saddwb z21.s, z6.s, z20.h\n"
    ".inst 0x459444d4  // saddwt z20.s, z6.s, z20.h\n"
    ".inst 0x448290be  // srshl z30.s, p4/M, z30.s, z5.s\n"
    ".inst 0x448290b1  // srshl z17.s, p4/M, z17.s, z5.s\n"
    ".inst 0x459840db  // saddwb z27.s, z6.s, z24.h\n"
    ".inst 0x459844da  // saddwt z26.s, z6.s, z24.h\n"
    ".inst 0x448290bd  // srshl z29.s, p4/M, z29.s, z5.s\n"
    ".inst 0x448290b2  // srshl z18.s, p4/M, z18.s, z5.s\n"
    ".inst 0x459740d9  // saddwb z25.s, z6.s, z23.h\n"
    ".inst 0x459744d8  // saddwt z24.s, z6.s, z23.h\n"
    ".inst 0x448290bc  // srshl z28.s, p4/M, z28.s, z5.s\n"
    ".inst 0x448290b0  // srshl z16.s, p4/M, z16.s, z5.s\n"
    ".inst 0x459640d7  // saddwb z23.s, z6.s, z22.h\n"
    ".inst 0x459644d6  // saddwt z22.s, z6.s, z22.h\n"
    ".inst 0x448290b5  // srshl z21.s, p4/M, z21.s, z5.s\n"
    ".inst 0x448290b4  // srshl z20.s, p4/M, z20.s, z5.s\n"
    ".inst 0x448290bb  // srshl z27.s, p4/M, z27.s, z5.s\n"
    ".inst 0x448290ba  // srshl z26.s, p4/M, z26.s, z5.s\n"
    ".inst 0x04a477ff  // sqrdmulh z31.s, z31.s, z4.s\n"
    ".inst 0x04a47673  // sqrdmulh z19.s, z19.s, z4.s\n"
    ".inst 0x448290b9  // srshl z25.s, p4/M, z25.s, z5.s\n"
    ".inst 0x448290b8  // srshl z24.s, p4/M, z24.s, z5.s\n"
    ".inst 0x04a477de  // sqrdmulh z30.s, z30.s, z4.s\n"
    ".inst 0x04a47631  // sqrdmulh z17.s, z17.s, z4.s\n"
    ".inst 0x448290b7  // srshl z23.s, p4/M, z23.s, z5.s\n"
    ".inst 0x448290b6  // srshl z22.s, p4/M, z22.s, z5.s\n"
    ".inst 0x04a477bd  // sqrdmulh z29.s, z29.s, z4.s\n"
    ".inst 0x04a47652  // sqrdmulh z18.s, z18.s, z4.s\n"
    ".inst 0x04a4779c  // sqrdmulh z28.s, z28.s, z4.s\n"
    ".inst 0x04a47610  // sqrdmulh z16.s, z16.s, z4.s\n"
    ".inst 0x4482907f  // srshl z31.s, p4/M, z31.s, z3.s\n"
    ".inst 0x44829073  // srshl z19.s, p4/M, z19.s, z3.s\n"
    ".inst 0x04a476b5  // sqrdmulh z21.s, z21.s, z4.s\n"
    ".inst 0x04a47694  // sqrdmulh z20.s, z20.s, z4.s\n"
    ".inst 0x4482907e  // srshl z30.s, p4/M, z30.s, z3.s\n"
    ".inst 0x44829071  // srshl z17.s, p4/M, z17.s, z3.s\n"
    ".inst 0x04a4777b  // sqrdmulh z27.s, z27.s, z4.s\n"
    ".inst 0x04a4775a  // sqrdmulh z26.s, z26.s, z4.s\n"
    ".inst 0x4482907d  // srshl z29.s, p4/M, z29.s, z3.s\n"
    ".inst 0x44829072  // srshl z18.s, p4/M, z18.s, z3.s\n"
    ".inst 0x04a47739  // sqrdmulh z25.s, z25.s, z4.s\n"
    ".inst 0x04a47718  // sqrdmulh z24.s, z24.s, z4.s\n"
    ".inst 0x4482907c  // srshl z28.s, p4/M, z28.s, z3.s\n"
    ".inst 0x44829070  // srshl z16.s, p4/M, z16.s, z3.s\n"
    ".inst 0x04a476f7  // sqrdmulh z23.s, z23.s, z4.s\n"
    ".inst 0x04a476d6  // sqrdmulh z22.s, z22.s, z4.s\n"
    ".inst 0x44829075  // srshl z21.s, p4/M, z21.s, z3.s\n"
    ".inst 0x44829074  // srshl z20.s, p4/M, z20.s, z3.s\n"
    ".inst 0x4482907b  // srshl z27.s, p4/M, z27.s, z3.s\n"
    ".inst 0x4482907a  // srshl z26.s, p4/M, z26.s, z3.s\n"
    "add z31.s, z31.s, z0.s\n"
    "add z19.s, z19.s, z0.s\n"
    ".inst 0x44829079  // srshl z25.s, p4/M, z25.s, z3.s\n"
    ".inst 0x44829078  // srshl z24.s, p4/M, z24.s, z3.s\n"
    "add z30.s, z30.s, z0.s\n"
    "add z17.s, z17.s, z0.s\n"
    ".inst 0x44829077  // srshl z23.s, p4/M, z23.s, z3.s\n"
    ".inst 0x44829076  // srshl z22.s, p4/M, z22.s, z3.s\n"
    "add z29.s, z29.s, z0.s\n"
    "add z18.s, z18.s, z0.s\n"
    "add z28.s, z28.s, z0.s\n"
    "add z16.s, z16.s, z0.s\n"
    "smax z31.s, p4/M, z31.s, z2.s\n"
    "smax z19.s, p4/M, z19.s, z2.s\n"
    "add z21.s, z21.s, z0.s\n"
    "add z20.s, z20.s, z0.s\n"
    "smax z30.s, p4/M, z30.s, z2.s\n"
    "smax z17.s, p4/M, z17.s, z2.s\n"
    "add z27.s, z27.s, z0.s\n"
    "add z26.s, z26.s, z0.s\n"
    "smax z29.s, p4/M, z29.s, z2.s\n"
    "smax z18.s, p4/M, z18.s, z2.s\n"
    "add z25.s, z25.s, z0.s\n"
    "add z24.s, z24.s, z0.s\n"
    "smax z28.s, p4/M, z28.s, z2.s\n"
    "smax z16.s, p4/M, z16.s, z2.s\n"
    "add z23.s, z23.s, z0.s\n"
    "add z22.s, z22.s, z0.s\n"
    "smax z21.s, p4/M, z21.s, z2.s\n"
    "smax z20.s, p4/M, z20.s, z2.s\n"
    "smax z27.s, p4/M, z27.s, z2.s\n"
    "smax z26.s, p4/M, z26.s, z2.s\n"
    "smax z25.s, p4/M, z25.s, z2.s\n"
    "smax z24.s, p4/M, z24.s, z2.s\n"
    "smax z23.s, p4/M, z23.s, z2.s\n"
    "smax z22.s, p4/M, z22.s, z2.s\n"
    "smin z31.s, p4/M, z31.s, z1.s\n"
    "smin z19.s, p4/M, z19.s, z1.s\n"
    "smin z30.s, p4/M, z30.s, z1.s\n"
    "smin z17.s, p4/M, z17.s, z1.s\n"
    "smin z29.s, p4/M, z29.s, z1.s\n"
    "smin z18.s, p4/M, z18.s, z1.s\n"
    "smin z28.s, p4/M, z28.s, z1.s\n"
    "smin z16.s, p4/M, z16.s, z1.s\n"
    "trn1 z19.h, z31.h, z19.h\n"
    "smin z21.s, p4/M, z21.s, z1.s\n"
    "smin z20.s, p4/M, z20.s, z1.s\n"
    "trn1 z17.h, z30.h, z17.h\n"
    "smin z27.s, p4/M, z27.s, z1.s\n"
    "smin z26.s, p4/M, z26.s, z1.s\n"
    "trn1 z18.h, z29.h, z18.h\n"
    "smin z25.s, p4/M, z25.s, z1.s\n"
    "smin z24.s, p4/M, z24.s, z1.s\n"
    "trn1 z16.h, z28.h, z16.h\n"
    "smin z23.s, p4/M, z23.s, z1.s\n"
    "smin z22.s, p4/M, z22.s, z1.s\n"
    "trn1 z21.h, z21.h, z20.h\n"
    "trn1 z20.b, z19.b, z17.b\n"
    "trn1 z17.h, z27.h, z26.h\n"
    "trn1 z19.h, z25.h, z24.h\n"
    "trn1 z18.b, z18.b, z16.b\n"
    "trn1 z16.h, z23.h, z22.h\n"
    "st1b { z20.b }, p3, [%x[outptr], x9]\n"
    "incb x9, ALL, MUL #4\n"
    "trn1 z17.b, z21.b, z17.b\n"
    "trn1 z16.b, z19.b, z16.b\n"
    "st1b { z18.b }, p2, [%x[outptr], x28]\n"
    "incb x28, ALL, MUL #4\n"
    "st1b { z17.b }, p1, [%x[outptr], x27]\n"
    "incb x27, ALL, MUL #4\n"
    "st1b { z16.b }, p0, [%x[outptr], x26]\n"
    "incb x26, ALL, MUL #4\n"
    "whilelt p0.b, x26, %x[n_channels]\n"
    "b.any 1b\n"
    "7:"  // Single vector of channels
    "whilelt p3.b, x9, %x[n_channels]\n"
    "b.none 14f\n"
    "8:"  // Single vector of channels: Loop
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "mov z6.b, #0x0\n"
    "mov x24, %x[inptrs]\n"
    "cbz x25, 11f\n"
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "subs x25, x25, #0x1\n"
    "add x24, x24, #0x20\n"
    "ld1b { z2.b }, p3/Z, [x23, x9]\n"
    "ld1b { z1.b }, p3/Z, [x22, x9]\n"
    "ld1b { z23.b }, p3/Z, [x21, x9]\n"
    "ld1b { z0.b }, p3/Z, [x20, x9]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 4 inputs loop
    "movprfx z16, z2\n umax z16.b, p4/M, z16.b, z1.b\n"
    "movprfx z17, z23\n umax z17.b, p4/M, z17.b, z0.b\n"
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "subs x25, x25, #0x1\n"
    "add x24, x24, #0x20\n"
    "umax z16.b, p4/M, z16.b, z17.b\n"
    "ld1b { z2.b }, p3/Z, [x23, x9]\n"
    "ld1b { z1.b }, p3/Z, [x22, x9]\n"
    "ld1b { z23.b }, p3/Z, [x21, x9]\n"
    "ld1b { z0.b }, p3/Z, [x20, x9]\n"
    "umax z6.b, p4/M, z6.b, z16.b\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "movprfx z16, z2\n umax z16.b, p4/M, z16.b, z1.b\n"
    "movprfx z17, z23\n umax z17.b, p4/M, z17.b, z0.b\n"
    "umax z16.b, p4/M, z16.b, z17.b\n"
    "umax z6.b, p4/M, z6.b, z16.b\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x20, [x24], #0x8\n"
    "subs x21, x21, #0x1\n"
    "ld1b { z16.b }, p3/Z, [x20, x9]\n"
    "umax z6.b, p4/M, z6.b, z16.b\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "add x21, %x[quant_params], %[offsetof_qp_input_offset]\n"
    ".inst 0x4508a8d1  // ushllb z17.h, z6.b, #0x0\n"
    ".inst 0x4508acda  // ushllt z26.h, z6.b, #0x0\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_left_shift]\n"
    "ld1rw { z16.s }, p4/Z, [x21]\n"
    "ld1rw { z25.s }, p4/Z, [x20]\n"
    "add x21, %x[quant_params], %[offsetof_qp_per_layer_mul]\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_right_shift]\n"
    "ld1rw { z24.s }, p4/Z, [x21]\n"
    "ld1rw { z23.s }, p4/Z, [x20]\n"
    "add x20, %x[quant_params], %[offsetof_qp_output_offset]\n"
    "mov z22.s, #0x0\n"
    "ld1rw { z21.s }, p4/Z, [x20]\n"
    "mov z20.s, #0xff\n"
    "neg z16.s, p4/M, z16.s\n"
    ".inst 0x45914213  // saddwb z19.s, z16.s, z17.h\n"
    ".inst 0x45914611  // saddwt z17.s, z16.s, z17.h\n"
    ".inst 0x459a4212  // saddwb z18.s, z16.s, z26.h\n"
    ".inst 0x459a4610  // saddwt z16.s, z16.s, z26.h\n"
    ".inst 0x44829333  // srshl z19.s, p4/M, z19.s, z25.s\n"
    ".inst 0x44829331  // srshl z17.s, p4/M, z17.s, z25.s\n"
    ".inst 0x44829332  // srshl z18.s, p4/M, z18.s, z25.s\n"
    ".inst 0x44829330  // srshl z16.s, p4/M, z16.s, z25.s\n"
    ".inst 0x04b87673  // sqrdmulh z19.s, z19.s, z24.s\n"
    ".inst 0x04b87631  // sqrdmulh z17.s, z17.s, z24.s\n"
    ".inst 0x04b87652  // sqrdmulh z18.s, z18.s, z24.s\n"
    ".inst 0x04b87610  // sqrdmulh z16.s, z16.s, z24.s\n"
    ".inst 0x448292f3  // srshl z19.s, p4/M, z19.s, z23.s\n"
    ".inst 0x448292f1  // srshl z17.s, p4/M, z17.s, z23.s\n"
    ".inst 0x448292f2  // srshl z18.s, p4/M, z18.s, z23.s\n"
    ".inst 0x448292f0  // srshl z16.s, p4/M, z16.s, z23.s\n"
    "add z19.s, z19.s, z21.s\n"
    "add z17.s, z17.s, z21.s\n"
    "add z18.s, z18.s, z21.s\n"
    "add z16.s, z16.s, z21.s\n"
    "smax z19.s, p4/M, z19.s, z22.s\n"
    "smax z17.s, p4/M, z17.s, z22.s\n"
    "smax z18.s, p4/M, z18.s, z22.s\n"
    "smax z16.s, p4/M, z16.s, z22.s\n"
    "smin z19.s, p4/M, z19.s, z20.s\n"
    "smin z17.s, p4/M, z17.s, z20.s\n"
    "smin z18.s, p4/M, z18.s, z20.s\n"
    "smin z16.s, p4/M, z16.s, z20.s\n"
    "trn1 z17.h, z19.h, z17.h\n"
    "trn1 z16.h, z18.h, z16.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "st1b { z16.b }, p3, [%x[outptr], x9]\n"
    "incb x9\n"
    "whilelt p3.b, x9, %x[n_channels]\n"
    "b.any 8b\n"
    "14:"  // End
    :
    : [inptrs] "r" (inptrs), [n_channels] "r" (n_channels), [n_valid_cells] "r" (n_valid_cells), [offsetof_qp_input_offset] "I" (offsetof(Requantize32, input_offset)), [offsetof_qp_output_offset] "I" (offsetof(Requantize32, output_offset)), [offsetof_qp_per_layer_left_shift] "I" (offsetof(Requantize32, per_layer_left_shift)), [offsetof_qp_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_qp_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [outptr] "r" (outptr), [quant_params] "r" (&qp)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
