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


void sve_s8q_nhwc_max_generic_depthfirst_impl(
  const uint64_t,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const int8_t *const *const inptrs,
  int8_t *outptr,
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
    "mov z6.b, #0x80\n"
    "mov z3.b, #0x80\n"
    "mov x24, %x[inptrs]\n"
    "mov z5.b, #0x80\n"
    "mov z4.b, #0x80\n"
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
    "movprfx z19, z2\n smax z19.b, p4/M, z19.b, z1.b\n"
    "smax z23.b, p4/M, z23.b, z0.b\n"
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "movprfx z18, z31\n smax z18.b, p4/M, z18.b, z30.b\n"
    "smax z22.b, p4/M, z22.b, z29.b\n"
    "movprfx z17, z28\n smax z17.b, p4/M, z17.b, z27.b\n"
    "smax z21.b, p4/M, z21.b, z26.b\n"
    "smax z16.b, p4/M, z16.b, z25.b\n"
    "smax z20.b, p4/M, z20.b, z24.b\n"
    "ld1b { z2.b }, p3/Z, [x23, x9]\n"
    "ld1b { z1.b }, p3/Z, [x22, x9]\n"
    "smax z19.b, p4/M, z19.b, z23.b\n"
    "smax z18.b, p4/M, z18.b, z22.b\n"
    "ld1b { z23.b }, p3/Z, [x21, x9]\n"
    "ld1b { z0.b }, p3/Z, [x20, x9]\n"
    "smax z17.b, p4/M, z17.b, z21.b\n"
    "subs x25, x25, #0x1\n"
    "ld1b { z31.b }, p2/Z, [x23, x28]\n"
    "ld1b { z30.b }, p2/Z, [x22, x28]\n"
    "smax z16.b, p4/M, z16.b, z20.b\n"
    "add x24, x24, #0x20\n"
    "ld1b { z22.b }, p2/Z, [x21, x28]\n"
    "ld1b { z29.b }, p2/Z, [x20, x28]\n"
    "smax z6.b, p4/M, z6.b, z19.b\n"
    "smax z3.b, p4/M, z3.b, z18.b\n"
    "ld1b { z28.b }, p1/Z, [x23, x27]\n"
    "ld1b { z27.b }, p1/Z, [x22, x27]\n"
    "smax z5.b, p4/M, z5.b, z17.b\n"
    "ld1b { z21.b }, p1/Z, [x21, x27]\n"
    "ld1b { z26.b }, p1/Z, [x20, x27]\n"
    "smax z4.b, p4/M, z4.b, z16.b\n"
    "ld1b { z16.b }, p0/Z, [x23, x26]\n"
    "ld1b { z25.b }, p0/Z, [x22, x26]\n"
    "ld1b { z20.b }, p0/Z, [x21, x26]\n"
    "ld1b { z24.b }, p0/Z, [x20, x26]\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "movprfx z19, z2\n smax z19.b, p4/M, z19.b, z1.b\n"
    "smax z23.b, p4/M, z23.b, z0.b\n"
    "movprfx z18, z31\n smax z18.b, p4/M, z18.b, z30.b\n"
    "smax z22.b, p4/M, z22.b, z29.b\n"
    "movprfx z17, z28\n smax z17.b, p4/M, z17.b, z27.b\n"
    "smax z21.b, p4/M, z21.b, z26.b\n"
    "smax z16.b, p4/M, z16.b, z25.b\n"
    "smax z20.b, p4/M, z20.b, z24.b\n"
    "smax z19.b, p4/M, z19.b, z23.b\n"
    "smax z18.b, p4/M, z18.b, z22.b\n"
    "smax z17.b, p4/M, z17.b, z21.b\n"
    "smax z16.b, p4/M, z16.b, z20.b\n"
    "smax z6.b, p4/M, z6.b, z19.b\n"
    "smax z3.b, p4/M, z3.b, z18.b\n"
    "smax z5.b, p4/M, z5.b, z17.b\n"
    "smax z4.b, p4/M, z4.b, z16.b\n"
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
    "smax z6.b, p4/M, z6.b, z19.b\n"
    "smax z3.b, p4/M, z3.b, z18.b\n"
    "smax z5.b, p4/M, z5.b, z17.b\n"
    "smax z4.b, p4/M, z4.b, z16.b\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    ".inst 0x4508a0d3  // sshllb z19.h, z6.b, #0x0\n"
    ".inst 0x4508a4d1  // sshllt z17.h, z6.b, #0x0\n"
    "add x21, %x[quant_params], %[offsetof_qp_per_layer_left_shift]\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_mul]\n"
    ".inst 0x4508a072  // sshllb z18.h, z3.b, #0x0\n"
    ".inst 0x4508a478  // sshllt z24.h, z3.b, #0x0\n"
    "ld1rw { z3.s }, p4/Z, [x21]\n"
    "ld1rw { z2.s }, p4/Z, [x20]\n"
    ".inst 0x4508a0b5  // sshllb z21.h, z5.b, #0x0\n"
    ".inst 0x4508a4b7  // sshllt z23.h, z5.b, #0x0\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_right_shift]\n"
    ".inst 0x4508a096  // sshllb z22.h, z4.b, #0x0\n"
    ".inst 0x4508a494  // sshllt z20.h, z4.b, #0x0\n"
    "ld1rw { z16.s }, p4/Z, [x20]\n"
    ".inst 0x4510a261  // sshllb z1.s, z19.h, #0x0\n"
    ".inst 0x4510a673  // sshllt z19.s, z19.h, #0x0\n"
    ".inst 0x4510a220  // sshllb z0.s, z17.h, #0x0\n"
    ".inst 0x4510a631  // sshllt z17.s, z17.h, #0x0\n"
    ".inst 0x4510a25f  // sshllb z31.s, z18.h, #0x0\n"
    ".inst 0x4510a652  // sshllt z18.s, z18.h, #0x0\n"
    ".inst 0x4510a31e  // sshllb z30.s, z24.h, #0x0\n"
    ".inst 0x4510a71d  // sshllt z29.s, z24.h, #0x0\n"
    ".inst 0x44829061  // srshl z1.s, p4/M, z1.s, z3.s\n"
    ".inst 0x44829073  // srshl z19.s, p4/M, z19.s, z3.s\n"
    ".inst 0x4510a2bc  // sshllb z28.s, z21.h, #0x0\n"
    ".inst 0x4510a6b5  // sshllt z21.s, z21.h, #0x0\n"
    ".inst 0x44829060  // srshl z0.s, p4/M, z0.s, z3.s\n"
    ".inst 0x44829071  // srshl z17.s, p4/M, z17.s, z3.s\n"
    ".inst 0x4510a2fb  // sshllb z27.s, z23.h, #0x0\n"
    ".inst 0x4510a6fa  // sshllt z26.s, z23.h, #0x0\n"
    ".inst 0x4482907f  // srshl z31.s, p4/M, z31.s, z3.s\n"
    ".inst 0x44829072  // srshl z18.s, p4/M, z18.s, z3.s\n"
    ".inst 0x4510a2d9  // sshllb z25.s, z22.h, #0x0\n"
    ".inst 0x4510a6d8  // sshllt z24.s, z22.h, #0x0\n"
    ".inst 0x4482907e  // srshl z30.s, p4/M, z30.s, z3.s\n"
    ".inst 0x4482907d  // srshl z29.s, p4/M, z29.s, z3.s\n"
    ".inst 0x4510a297  // sshllb z23.s, z20.h, #0x0\n"
    ".inst 0x4510a696  // sshllt z22.s, z20.h, #0x0\n"
    ".inst 0x4482907c  // srshl z28.s, p4/M, z28.s, z3.s\n"
    ".inst 0x44829075  // srshl z21.s, p4/M, z21.s, z3.s\n"
    ".inst 0x4482907b  // srshl z27.s, p4/M, z27.s, z3.s\n"
    ".inst 0x4482907a  // srshl z26.s, p4/M, z26.s, z3.s\n"
    ".inst 0x04a27421  // sqrdmulh z1.s, z1.s, z2.s\n"
    ".inst 0x04a27673  // sqrdmulh z19.s, z19.s, z2.s\n"
    ".inst 0x44829079  // srshl z25.s, p4/M, z25.s, z3.s\n"
    ".inst 0x44829078  // srshl z24.s, p4/M, z24.s, z3.s\n"
    ".inst 0x04a27400  // sqrdmulh z0.s, z0.s, z2.s\n"
    ".inst 0x04a27631  // sqrdmulh z17.s, z17.s, z2.s\n"
    ".inst 0x44829077  // srshl z23.s, p4/M, z23.s, z3.s\n"
    ".inst 0x44829076  // srshl z22.s, p4/M, z22.s, z3.s\n"
    ".inst 0x04a277ff  // sqrdmulh z31.s, z31.s, z2.s\n"
    ".inst 0x04a27652  // sqrdmulh z18.s, z18.s, z2.s\n"
    ".inst 0x04a277de  // sqrdmulh z30.s, z30.s, z2.s\n"
    ".inst 0x04a277bd  // sqrdmulh z29.s, z29.s, z2.s\n"
    ".inst 0x44829201  // srshl z1.s, p4/M, z1.s, z16.s\n"
    ".inst 0x44829213  // srshl z19.s, p4/M, z19.s, z16.s\n"
    ".inst 0x04a2779c  // sqrdmulh z28.s, z28.s, z2.s\n"
    ".inst 0x04a276b5  // sqrdmulh z21.s, z21.s, z2.s\n"
    ".inst 0x44829200  // srshl z0.s, p4/M, z0.s, z16.s\n"
    ".inst 0x44829211  // srshl z17.s, p4/M, z17.s, z16.s\n"
    ".inst 0x04a2777b  // sqrdmulh z27.s, z27.s, z2.s\n"
    ".inst 0x04a2775a  // sqrdmulh z26.s, z26.s, z2.s\n"
    ".inst 0x4482921f  // srshl z31.s, p4/M, z31.s, z16.s\n"
    ".inst 0x44829212  // srshl z18.s, p4/M, z18.s, z16.s\n"
    ".inst 0x04a27739  // sqrdmulh z25.s, z25.s, z2.s\n"
    ".inst 0x04a27718  // sqrdmulh z24.s, z24.s, z2.s\n"
    ".inst 0x4482921e  // srshl z30.s, p4/M, z30.s, z16.s\n"
    ".inst 0x4482921d  // srshl z29.s, p4/M, z29.s, z16.s\n"
    ".inst 0x04a276f7  // sqrdmulh z23.s, z23.s, z2.s\n"
    ".inst 0x04a276d6  // sqrdmulh z22.s, z22.s, z2.s\n"
    ".inst 0x4482921c  // srshl z28.s, p4/M, z28.s, z16.s\n"
    ".inst 0x44829215  // srshl z21.s, p4/M, z21.s, z16.s\n"
    "mov z20.s, #0x7f\n"
    ".inst 0x4482921b  // srshl z27.s, p4/M, z27.s, z16.s\n"
    ".inst 0x4482921a  // srshl z26.s, p4/M, z26.s, z16.s\n"
    ".inst 0x44829219  // srshl z25.s, p4/M, z25.s, z16.s\n"
    ".inst 0x44829218  // srshl z24.s, p4/M, z24.s, z16.s\n"
    ".inst 0x44829217  // srshl z23.s, p4/M, z23.s, z16.s\n"
    ".inst 0x44829216  // srshl z22.s, p4/M, z22.s, z16.s\n"
    "not z16.s, p4/M, z20.s\n"
    "smax z1.s, p4/M, z1.s, z16.s\n"
    "smax z19.s, p4/M, z19.s, z16.s\n"
    "smax z0.s, p4/M, z0.s, z16.s\n"
    "smax z17.s, p4/M, z17.s, z16.s\n"
    "smax z31.s, p4/M, z31.s, z16.s\n"
    "smax z18.s, p4/M, z18.s, z16.s\n"
    "smax z30.s, p4/M, z30.s, z16.s\n"
    "smax z29.s, p4/M, z29.s, z16.s\n"
    "smax z28.s, p4/M, z28.s, z16.s\n"
    "smax z21.s, p4/M, z21.s, z16.s\n"
    "smax z27.s, p4/M, z27.s, z16.s\n"
    "smax z26.s, p4/M, z26.s, z16.s\n"
    "smax z25.s, p4/M, z25.s, z16.s\n"
    "smax z24.s, p4/M, z24.s, z16.s\n"
    "smax z23.s, p4/M, z23.s, z16.s\n"
    "smax z22.s, p4/M, z22.s, z16.s\n"
    "smin z1.s, p4/M, z1.s, z20.s\n"
    "smin z19.s, p4/M, z19.s, z20.s\n"
    "smin z0.s, p4/M, z0.s, z20.s\n"
    "smin z17.s, p4/M, z17.s, z20.s\n"
    "smin z31.s, p4/M, z31.s, z20.s\n"
    "smin z18.s, p4/M, z18.s, z20.s\n"
    "smin z30.s, p4/M, z30.s, z20.s\n"
    "smin z29.s, p4/M, z29.s, z20.s\n"
    "smin z28.s, p4/M, z28.s, z20.s\n"
    "trn1 z19.h, z1.h, z19.h\n"
    "smin z21.s, p4/M, z21.s, z20.s\n"
    "smin z27.s, p4/M, z27.s, z20.s\n"
    "trn1 z17.h, z0.h, z17.h\n"
    "smin z26.s, p4/M, z26.s, z20.s\n"
    "smin z25.s, p4/M, z25.s, z20.s\n"
    "trn1 z18.h, z31.h, z18.h\n"
    "smin z24.s, p4/M, z24.s, z20.s\n"
    "smin z23.s, p4/M, z23.s, z20.s\n"
    "trn1 z16.h, z30.h, z29.h\n"
    "smin z22.s, p4/M, z22.s, z20.s\n"
    "trn1 z21.h, z28.h, z21.h\n"
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
    "mov z6.b, #0x80\n"
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
    "movprfx z16, z2\n smax z16.b, p4/M, z16.b, z1.b\n"
    "movprfx z17, z23\n smax z17.b, p4/M, z17.b, z0.b\n"
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "subs x25, x25, #0x1\n"
    "add x24, x24, #0x20\n"
    "smax z16.b, p4/M, z16.b, z17.b\n"
    "ld1b { z2.b }, p3/Z, [x23, x9]\n"
    "ld1b { z1.b }, p3/Z, [x22, x9]\n"
    "ld1b { z23.b }, p3/Z, [x21, x9]\n"
    "ld1b { z0.b }, p3/Z, [x20, x9]\n"
    "smax z6.b, p4/M, z6.b, z16.b\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "movprfx z16, z2\n smax z16.b, p4/M, z16.b, z1.b\n"
    "movprfx z17, z23\n smax z17.b, p4/M, z17.b, z0.b\n"
    "smax z16.b, p4/M, z16.b, z17.b\n"
    "smax z6.b, p4/M, z6.b, z16.b\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x20, [x24], #0x8\n"
    "subs x21, x21, #0x1\n"
    "ld1b { z16.b }, p3/Z, [x20, x9]\n"
    "smax z6.b, p4/M, z6.b, z16.b\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    ".inst 0x4508a0d1  // sshllb z17.h, z6.b, #0x0\n"
    ".inst 0x4508a4d0  // sshllt z16.h, z6.b, #0x0\n"
    "add x21, %x[quant_params], %[offsetof_qp_per_layer_left_shift]\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_mul]\n"
    "ld1rw { z24.s }, p4/Z, [x21]\n"
    "ld1rw { z23.s }, p4/Z, [x20]\n"
    "add x20, %x[quant_params], %[offsetof_qp_per_layer_right_shift]\n"
    "mov z22.s, #0x7f\n"
    "ld1rw { z21.s }, p4/Z, [x20]\n"
    ".inst 0x4510a234  // sshllb z20.s, z17.h, #0x0\n"
    ".inst 0x4510a631  // sshllt z17.s, z17.h, #0x0\n"
    ".inst 0x4510a213  // sshllb z19.s, z16.h, #0x0\n"
    ".inst 0x4510a612  // sshllt z18.s, z16.h, #0x0\n"
    "not z16.s, p4/M, z22.s\n"
    ".inst 0x44829314  // srshl z20.s, p4/M, z20.s, z24.s\n"
    ".inst 0x44829311  // srshl z17.s, p4/M, z17.s, z24.s\n"
    ".inst 0x44829313  // srshl z19.s, p4/M, z19.s, z24.s\n"
    ".inst 0x44829312  // srshl z18.s, p4/M, z18.s, z24.s\n"
    ".inst 0x04b77694  // sqrdmulh z20.s, z20.s, z23.s\n"
    ".inst 0x04b77631  // sqrdmulh z17.s, z17.s, z23.s\n"
    ".inst 0x04b77673  // sqrdmulh z19.s, z19.s, z23.s\n"
    ".inst 0x04b77652  // sqrdmulh z18.s, z18.s, z23.s\n"
    ".inst 0x448292b4  // srshl z20.s, p4/M, z20.s, z21.s\n"
    ".inst 0x448292b1  // srshl z17.s, p4/M, z17.s, z21.s\n"
    ".inst 0x448292b3  // srshl z19.s, p4/M, z19.s, z21.s\n"
    ".inst 0x448292b2  // srshl z18.s, p4/M, z18.s, z21.s\n"
    "smax z20.s, p4/M, z20.s, z16.s\n"
    "smax z17.s, p4/M, z17.s, z16.s\n"
    "smax z19.s, p4/M, z19.s, z16.s\n"
    "smax z18.s, p4/M, z18.s, z16.s\n"
    "smin z20.s, p4/M, z20.s, z22.s\n"
    "smin z17.s, p4/M, z17.s, z22.s\n"
    "smin z19.s, p4/M, z19.s, z22.s\n"
    "smin z18.s, p4/M, z18.s, z22.s\n"
    "trn1 z17.h, z20.h, z17.h\n"
    "trn1 z16.h, z19.h, z18.h\n"
    "trn1 z16.b, z17.b, z16.b\n"
    "st1b { z16.b }, p3, [%x[outptr], x9]\n"
    "incb x9\n"
    "whilelt p3.b, x9, %x[n_channels]\n"
    "b.any 8b\n"
    "14:"  // End
    :
    : [inptrs] "r" (inptrs), [n_channels] "r" (n_channels), [n_valid_cells] "r" (n_valid_cells), [offsetof_qp_per_layer_left_shift] "I" (offsetof(Requantize32, per_layer_left_shift)), [offsetof_qp_per_layer_mul] "I" (offsetof(Requantize32, per_layer_mul)), [offsetof_qp_per_layer_right_shift] "I" (offsetof(Requantize32, per_layer_right_shift)), [outptr] "r" (outptr), [quant_params] "r" (&qp)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
