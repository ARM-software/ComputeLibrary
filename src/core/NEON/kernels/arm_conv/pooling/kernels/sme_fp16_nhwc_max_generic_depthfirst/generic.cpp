/*
 * Copyright (c) 2022 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SME)

#include <cstdint>

namespace arm_conv {
namespace pooling {


void sme_fp16_nhwc_max_generic_depthfirst_impl(
  const uint64_t,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const __fp16 *const *const inptrs,
  __fp16 *outptr
)
{
  __asm__ __volatile__(
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x28, #0x0\n"
    "cnth x27\n"
    "cnth x26, ALL, MUL #2\n"
    "cnth x25, ALL, MUL #3\n"
    "whilelt p4.h, x28, %x[n_channels]\n"
    "whilelt p3.h, x27, %x[n_channels]\n"
    "whilelt p2.h, x26, %x[n_channels]\n"
    "whilelt p1.h, x25, %x[n_channels]\n"
    "ptrue p0.b\n"
    "b.none 7f\n"
    "1:"  // 4-vectors of channels
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "mov z4.h, #0xfc00\n"
    "mov z3.h, #0xfc00\n"
    "mov x19, %x[inptrs]\n"
    "mov z2.h, #0xfc00\n"
    "mov z1.h, #0xfc00\n"
    "cbz x24, 4f\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "subs x24, x24, #0x1\n"
    "ld1h { z0.h }, p4/Z, [x23, x28, LSL #1]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "ld1h { z31.h }, p4/Z, [x22, x28, LSL #1]\n"
    "ld1h { z23.h }, p4/Z, [x21, x28, LSL #1]\n"
    "ld1h { z30.h }, p4/Z, [x20, x28, LSL #1]\n"
    "ld1h { z18.h }, p3/Z, [x23, x27, LSL #1]\n"
    "ld1h { z29.h }, p3/Z, [x22, x27, LSL #1]\n"
    "ld1h { z22.h }, p3/Z, [x21, x27, LSL #1]\n"
    "ld1h { z28.h }, p3/Z, [x20, x27, LSL #1]\n"
    "ld1h { z17.h }, p2/Z, [x23, x26, LSL #1]\n"
    "ld1h { z27.h }, p2/Z, [x22, x26, LSL #1]\n"
    "ld1h { z21.h }, p2/Z, [x21, x26, LSL #1]\n"
    "ld1h { z26.h }, p2/Z, [x20, x26, LSL #1]\n"
    "ld1h { z16.h }, p1/Z, [x23, x25, LSL #1]\n"
    "ld1h { z25.h }, p1/Z, [x22, x25, LSL #1]\n"
    "ld1h { z20.h }, p1/Z, [x21, x25, LSL #1]\n"
    "ld1h { z24.h }, p1/Z, [x20, x25, LSL #1]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 4 inputs loop
    "movprfx z19, z0\n fmax z19.h, p0/M, z19.h, z31.h\n"
    "fmax z23.h, p0/M, z23.h, z30.h\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "subs x24, x24, #0x1\n"
    "fmax z18.h, p0/M, z18.h, z29.h\n"
    "fmax z22.h, p0/M, z22.h, z28.h\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "fmax z17.h, p0/M, z17.h, z27.h\n"
    "fmax z21.h, p0/M, z21.h, z26.h\n"
    "ld1h { z0.h }, p4/Z, [x23, x28, LSL #1]\n"
    "fmax z16.h, p0/M, z16.h, z25.h\n"
    "fmax z20.h, p0/M, z20.h, z24.h\n"
    "ld1h { z31.h }, p4/Z, [x22, x28, LSL #1]\n"
    "fmax z19.h, p0/M, z19.h, z23.h\n"
    "fmax z18.h, p0/M, z18.h, z22.h\n"
    "ld1h { z23.h }, p4/Z, [x21, x28, LSL #1]\n"
    "fmax z17.h, p0/M, z17.h, z21.h\n"
    "fmax z16.h, p0/M, z16.h, z20.h\n"
    "ld1h { z30.h }, p4/Z, [x20, x28, LSL #1]\n"
    "fmax z4.h, p0/M, z4.h, z19.h\n"
    "fmax z3.h, p0/M, z3.h, z18.h\n"
    "ld1h { z18.h }, p3/Z, [x23, x27, LSL #1]\n"
    "fmax z2.h, p0/M, z2.h, z17.h\n"
    "fmax z1.h, p0/M, z1.h, z16.h\n"
    "ld1h { z29.h }, p3/Z, [x22, x27, LSL #1]\n"
    "ld1h { z22.h }, p3/Z, [x21, x27, LSL #1]\n"
    "ld1h { z28.h }, p3/Z, [x20, x27, LSL #1]\n"
    "ld1h { z17.h }, p2/Z, [x23, x26, LSL #1]\n"
    "ld1h { z27.h }, p2/Z, [x22, x26, LSL #1]\n"
    "ld1h { z21.h }, p2/Z, [x21, x26, LSL #1]\n"
    "ld1h { z26.h }, p2/Z, [x20, x26, LSL #1]\n"
    "ld1h { z16.h }, p1/Z, [x23, x25, LSL #1]\n"
    "ld1h { z25.h }, p1/Z, [x22, x25, LSL #1]\n"
    "ld1h { z20.h }, p1/Z, [x21, x25, LSL #1]\n"
    "ld1h { z24.h }, p1/Z, [x20, x25, LSL #1]\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "movprfx z19, z0\n fmax z19.h, p0/M, z19.h, z31.h\n"
    "fmax z23.h, p0/M, z23.h, z30.h\n"
    "fmax z18.h, p0/M, z18.h, z29.h\n"
    "fmax z22.h, p0/M, z22.h, z28.h\n"
    "fmax z17.h, p0/M, z17.h, z27.h\n"
    "fmax z21.h, p0/M, z21.h, z26.h\n"
    "fmax z16.h, p0/M, z16.h, z25.h\n"
    "fmax z20.h, p0/M, z20.h, z24.h\n"
    "fmax z19.h, p0/M, z19.h, z23.h\n"
    "fmax z18.h, p0/M, z18.h, z22.h\n"
    "fmax z17.h, p0/M, z17.h, z21.h\n"
    "fmax z16.h, p0/M, z16.h, z20.h\n"
    "fmax z4.h, p0/M, z4.h, z19.h\n"
    "fmax z3.h, p0/M, z3.h, z18.h\n"
    "fmax z2.h, p0/M, z2.h, z17.h\n"
    "fmax z1.h, p0/M, z1.h, z16.h\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x23, [x19], #0x8\n"
    "ld1h { z0.h }, p4/Z, [x23, x28, LSL #1]\n"
    "subs x20, x20, #0x1\n"
    "fmax z4.h, p0/M, z4.h, z0.h\n"
    "ld1h { z18.h }, p3/Z, [x23, x27, LSL #1]\n"
    "fmax z3.h, p0/M, z3.h, z18.h\n"
    "ld1h { z17.h }, p2/Z, [x23, x26, LSL #1]\n"
    "fmax z2.h, p0/M, z2.h, z17.h\n"
    "ld1h { z16.h }, p1/Z, [x23, x25, LSL #1]\n"
    "fmax z1.h, p0/M, z1.h, z16.h\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "st1h { z4.h }, p4, [%x[outptr], x28, LSL #1]\n"
    "inch x28, ALL, MUL #4\n"
    "st1h { z3.h }, p3, [%x[outptr], x27, LSL #1]\n"
    "inch x27, ALL, MUL #4\n"
    "st1h { z2.h }, p2, [%x[outptr], x26, LSL #1]\n"
    "inch x26, ALL, MUL #4\n"
    "st1h { z1.h }, p1, [%x[outptr], x25, LSL #1]\n"
    "inch x25, ALL, MUL #4\n"
    "whilelt p1.h, x25, %x[n_channels]\n"
    "b.any 1b\n"
    "7:"  // Single vector of channels
    "whilelt p4.h, x28, %x[n_channels]\n"
    "b.none 14f\n"
    "8:"  // Single vector of channels: Loop
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "mov z4.h, #0xfc00\n"
    "mov x19, %x[inptrs]\n"
    "cbz x24, 11f\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "subs x24, x24, #0x1\n"
    "ld1h { z0.h }, p4/Z, [x23, x28, LSL #1]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "ld1h { z31.h }, p4/Z, [x22, x28, LSL #1]\n"
    "ld1h { z23.h }, p4/Z, [x21, x28, LSL #1]\n"
    "ld1h { z30.h }, p4/Z, [x20, x28, LSL #1]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 4 inputs loop
    "movprfx z19, z0\n fmax z19.h, p0/M, z19.h, z31.h\n"
    "fmax z23.h, p0/M, z23.h, z30.h\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "subs x24, x24, #0x1\n"
    "fmax z19.h, p0/M, z19.h, z23.h\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "fmax z4.h, p0/M, z4.h, z19.h\n"
    "add x19, x19, #0x20\n"
    "ld1h { z0.h }, p4/Z, [x23, x28, LSL #1]\n"
    "ld1h { z31.h }, p4/Z, [x22, x28, LSL #1]\n"
    "ld1h { z23.h }, p4/Z, [x21, x28, LSL #1]\n"
    "ld1h { z30.h }, p4/Z, [x20, x28, LSL #1]\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "movprfx z19, z0\n fmax z19.h, p0/M, z19.h, z31.h\n"
    "fmax z23.h, p0/M, z23.h, z30.h\n"
    "fmax z19.h, p0/M, z19.h, z23.h\n"
    "fmax z4.h, p0/M, z4.h, z19.h\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x23, [x19], #0x8\n"
    "ld1h { z0.h }, p4/Z, [x23, x28, LSL #1]\n"
    "subs x20, x20, #0x1\n"
    "fmax z4.h, p0/M, z4.h, z0.h\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "st1h { z4.h }, p4, [%x[outptr], x28, LSL #1]\n"
    "inch x28\n"
    "whilelt p4.h, x28, %x[n_channels]\n"
    "b.any 8b\n"
    "14:"  // End
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [inptrs] "r" (inptrs), [n_channels] "r" (n_channels), [n_valid_cells] "r" (n_valid_cells), [outptr] "r" (outptr)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SME)
