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

#include <cstdint>

#if defined(__ARM_FEATURE_SVE)

namespace arm_conv {
namespace pooling {


void sve_fp32_nhwc_max_generic_depthfirst_impl(
  const uint64_t,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const float *const *const inptrs,
  float *outptr
)
{
  __asm__ __volatile__(
    "ptrue p4.b\n"
    "mov x28, #0x0\n"
    "cntw x27\n"
    "cntw x26, ALL, MUL #2\n"
    "cntw x25, ALL, MUL #3\n"
    "whilelt p3.s, x28, %x[n_channels]\n"
    "whilelt p2.s, x27, %x[n_channels]\n"
    "whilelt p1.s, x26, %x[n_channels]\n"
    "whilelt p0.s, x25, %x[n_channels]\n"
    "b.none 7f\n"
    "1:"  // 4-vectors of channels
    "mov z7.s, #0xff800000\n"
    "mov x19, %x[inptrs]\n"
    "mov z6.s, #0xff800000\n"
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "mov z5.s, #0xff800000\n"
    "mov z4.s, #0xff800000\n"
    "cbz x24, 4f\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "subs x24, x24, #0x1\n"
    "ld1w { z3.s }, p3/Z, [x23, x28, LSL #2]\n"
    "ld1w { z2.s }, p3/Z, [x22, x28, LSL #2]\n"
    "ld1w { z1.s }, p3/Z, [x21, x28, LSL #2]\n"
    "ld1w { z0.s }, p3/Z, [x20, x28, LSL #2]\n"
    "ld1w { z31.s }, p2/Z, [x23, x27, LSL #2]\n"
    "ld1w { z30.s }, p2/Z, [x22, x27, LSL #2]\n"
    "ld1w { z22.s }, p2/Z, [x21, x27, LSL #2]\n"
    "ld1w { z29.s }, p2/Z, [x20, x27, LSL #2]\n"
    "ld1w { z28.s }, p1/Z, [x23, x26, LSL #2]\n"
    "ld1w { z27.s }, p1/Z, [x22, x26, LSL #2]\n"
    "ld1w { z21.s }, p1/Z, [x21, x26, LSL #2]\n"
    "ld1w { z26.s }, p1/Z, [x20, x26, LSL #2]\n"
    "ld1w { z16.s }, p0/Z, [x23, x25, LSL #2]\n"
    "ld1w { z25.s }, p0/Z, [x22, x25, LSL #2]\n"
    "ld1w { z20.s }, p0/Z, [x21, x25, LSL #2]\n"
    "ld1w { z24.s }, p0/Z, [x20, x25, LSL #2]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 4 inputs loop
    "movprfx z19, z3\n fmax z19.s, p4/M, z19.s, z2.s\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "subs x24, x24, #0x1\n"
    "movprfx z23, z1\n fmax z23.s, p4/M, z23.s, z0.s\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "movprfx z18, z31\n fmax z18.s, p4/M, z18.s, z30.s\n"
    "ld1w { z3.s }, p3/Z, [x23, x28, LSL #2]\n"
    "fmax z22.s, p4/M, z22.s, z29.s\n"
    "movprfx z17, z28\n fmax z17.s, p4/M, z17.s, z27.s\n"
    "ld1w { z2.s }, p3/Z, [x22, x28, LSL #2]\n"
    "fmax z21.s, p4/M, z21.s, z26.s\n"
    "ld1w { z1.s }, p3/Z, [x21, x28, LSL #2]\n"
    "fmax z16.s, p4/M, z16.s, z25.s\n"
    "ld1w { z0.s }, p3/Z, [x20, x28, LSL #2]\n"
    "fmax z20.s, p4/M, z20.s, z24.s\n"
    "ld1w { z31.s }, p2/Z, [x23, x27, LSL #2]\n"
    "fmax z19.s, p4/M, z19.s, z23.s\n"
    "ld1w { z30.s }, p2/Z, [x22, x27, LSL #2]\n"
    "fmax z18.s, p4/M, z18.s, z22.s\n"
    "ld1w { z22.s }, p2/Z, [x21, x27, LSL #2]\n"
    "fmax z17.s, p4/M, z17.s, z21.s\n"
    "ld1w { z29.s }, p2/Z, [x20, x27, LSL #2]\n"
    "fmax z16.s, p4/M, z16.s, z20.s\n"
    "ld1w { z28.s }, p1/Z, [x23, x26, LSL #2]\n"
    "fmax z7.s, p4/M, z7.s, z19.s\n"
    "ld1w { z27.s }, p1/Z, [x22, x26, LSL #2]\n"
    "fmax z6.s, p4/M, z6.s, z18.s\n"
    "ld1w { z21.s }, p1/Z, [x21, x26, LSL #2]\n"
    "fmax z5.s, p4/M, z5.s, z17.s\n"
    "ld1w { z26.s }, p1/Z, [x20, x26, LSL #2]\n"
    "fmax z4.s, p4/M, z4.s, z16.s\n"
    "ld1w { z16.s }, p0/Z, [x23, x25, LSL #2]\n"
    "ld1w { z25.s }, p0/Z, [x22, x25, LSL #2]\n"
    "ld1w { z20.s }, p0/Z, [x21, x25, LSL #2]\n"
    "ld1w { z24.s }, p0/Z, [x20, x25, LSL #2]\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "movprfx z19, z3\n fmax z19.s, p4/M, z19.s, z2.s\n"
    "movprfx z23, z1\n fmax z23.s, p4/M, z23.s, z0.s\n"
    "movprfx z18, z31\n fmax z18.s, p4/M, z18.s, z30.s\n"
    "fmax z22.s, p4/M, z22.s, z29.s\n"
    "movprfx z17, z28\n fmax z17.s, p4/M, z17.s, z27.s\n"
    "fmax z21.s, p4/M, z21.s, z26.s\n"
    "fmax z16.s, p4/M, z16.s, z25.s\n"
    "fmax z20.s, p4/M, z20.s, z24.s\n"
    "fmax z19.s, p4/M, z19.s, z23.s\n"
    "fmax z18.s, p4/M, z18.s, z22.s\n"
    "fmax z17.s, p4/M, z17.s, z21.s\n"
    "fmax z16.s, p4/M, z16.s, z20.s\n"
    "fmax z7.s, p4/M, z7.s, z19.s\n"
    "fmax z6.s, p4/M, z6.s, z18.s\n"
    "fmax z5.s, p4/M, z5.s, z17.s\n"
    "fmax z4.s, p4/M, z4.s, z16.s\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x23, [x19], #0x8\n"
    "subs x20, x20, #0x1\n"
    "ld1w { z3.s }, p3/Z, [x23, x28, LSL #2]\n"
    "fmax z7.s, p4/M, z7.s, z3.s\n"
    "ld1w { z31.s }, p2/Z, [x23, x27, LSL #2]\n"
    "ld1w { z28.s }, p1/Z, [x23, x26, LSL #2]\n"
    "fmax z6.s, p4/M, z6.s, z31.s\n"
    "ld1w { z16.s }, p0/Z, [x23, x25, LSL #2]\n"
    "fmax z5.s, p4/M, z5.s, z28.s\n"
    "fmax z4.s, p4/M, z4.s, z16.s\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "st1w { z7.s }, p3, [%x[outptr], x28, LSL #2]\n"
    "incw x28, ALL, MUL #4\n"
    "st1w { z6.s }, p2, [%x[outptr], x27, LSL #2]\n"
    "incw x27, ALL, MUL #4\n"
    "st1w { z5.s }, p1, [%x[outptr], x26, LSL #2]\n"
    "incw x26, ALL, MUL #4\n"
    "st1w { z4.s }, p0, [%x[outptr], x25, LSL #2]\n"
    "incw x25, ALL, MUL #4\n"
    "whilelt p0.s, x25, %x[n_channels]\n"
    "b.any 1b\n"
    "7:"  // Single vector of channels
    "whilelt p3.s, x28, %x[n_channels]\n"
    "b.none 14f\n"
    "8:"  // Single vector of channels: Loop
    "mov z7.s, #0xff800000\n"
    "mov x19, %x[inptrs]\n"
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "cbz x24, 11f\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "subs x24, x24, #0x1\n"
    "ld1w { z3.s }, p3/Z, [x23, x28, LSL #2]\n"
    "ld1w { z2.s }, p3/Z, [x22, x28, LSL #2]\n"
    "ld1w { z1.s }, p3/Z, [x21, x28, LSL #2]\n"
    "ld1w { z0.s }, p3/Z, [x20, x28, LSL #2]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 4 inputs loop
    "movprfx z19, z3\n fmax z19.s, p4/M, z19.s, z2.s\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "subs x24, x24, #0x1\n"
    "movprfx z23, z1\n fmax z23.s, p4/M, z23.s, z0.s\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "fmax z19.s, p4/M, z19.s, z23.s\n"
    "ld1w { z3.s }, p3/Z, [x23, x28, LSL #2]\n"
    "ld1w { z2.s }, p3/Z, [x22, x28, LSL #2]\n"
    "fmax z7.s, p4/M, z7.s, z19.s\n"
    "ld1w { z1.s }, p3/Z, [x21, x28, LSL #2]\n"
    "ld1w { z0.s }, p3/Z, [x20, x28, LSL #2]\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "movprfx z19, z3\n fmax z19.s, p4/M, z19.s, z2.s\n"
    "movprfx z23, z1\n fmax z23.s, p4/M, z23.s, z0.s\n"
    "fmax z19.s, p4/M, z19.s, z23.s\n"
    "fmax z7.s, p4/M, z7.s, z19.s\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x23, [x19], #0x8\n"
    "subs x20, x20, #0x1\n"
    "ld1w { z3.s }, p3/Z, [x23, x28, LSL #2]\n"
    "fmax z7.s, p4/M, z7.s, z3.s\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "st1w { z7.s }, p3, [%x[outptr], x28, LSL #2]\n"
    "incw x28\n"
    "whilelt p3.s, x28, %x[n_channels]\n"
    "b.any 8b\n"
    "14:"  // End

    :
    : [inptrs] "r" (inptrs), [n_channels] "r" (n_channels), [n_valid_cells] "r" (n_valid_cells), [outptr] "r" (outptr)
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE)
