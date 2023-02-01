/*
 * Copyright (c) 2022-2023 Arm Limited.
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


#include <cstddef>
#include <cstdint>

#if defined(__ARM_FEATURE_SVE) && defined(ARM_COMPUTE_ENABLE_SME)

namespace arm_conv {
namespace pooling {

void sme_s8_nhwc_max_2x2_s1_output2x2_depthfirst_impl(
  const unsigned int n_channels,
  const int8_t *const *const inptrs,
  int8_t *const *const outptrs,
  const bool exclude_padding,
  const unsigned int pad_left,
  const unsigned int pad_top,
  const unsigned int pad_right,
  const unsigned int pad_bottom
)
{
  struct KernelArgs
  {
    const uint64_t n_channels;
    const int8_t *const *const inptrs;
    int8_t *const *const outptrs;
    KernelArgs(
      unsigned int channels,
      const int8_t *const *input_ptrs,
      int8_t *const * output_ptrs,
      bool, unsigned int, unsigned int, unsigned int, unsigned int
    ) : n_channels(channels),
        inptrs(input_ptrs),
        outptrs(output_ptrs)
    {
    }
  };

  const KernelArgs args(n_channels, inptrs, outptrs, exclude_padding,
                        pad_left, pad_top, pad_right, pad_bottom);

  __asm__ __volatile__(
    "ldr x21, [%x[args], %[offsetof_outptrs]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x15, #0x0\n"
    "ptrue p2.b\n"
    "ldr x20, [%x[args], %[offsetof_inptrs]]\n"
    "mov x14, #0x0\n"
    "ldr x13, [%x[args], %[offsetof_n_channels]]\n"
    "whilelt p1.b, x15, x13\n"
    "ldp x12, x11, [x21, #0x0]\n"
    "ldp x10, x9, [x21, #0x10]\n"
    "ldp x28, x27, [x20, #0x0]\n"
    "ld1b { z30.b }, p1/Z, [x27, x15]\n"
    "ldp x26, x25, [x20, #0x10]\n"
    "ld1b { z29.b }, p1/Z, [x25, x15]\n"
    "ldp x24, x23, [x20, #0x20]\n"
    "ld1b { z28.b }, p1/Z, [x24, x15]\n"
    "ldp x22, x21, [x20, #0x30]\n"
    "ld1b { z27.b }, p1/Z, [x21, x15]\n"
    "ldr x20, [x20, #0x40]\n"
    "ld1b { z26.b }, p1/Z, [x28, x15]\n"
    "ld1b { z25.b }, p1/Z, [x26, x15]\n"
    "ld1b { z24.b }, p1/Z, [x23, x15]\n"
    "ld1b { z23.b }, p1/Z, [x22, x15]\n"
    "ld1b { z19.b }, p1/Z, [x20, x15]\n"
    "incw x15\n"
    "whilelt p1.b, x15, x13\n"
    "b.none 2f\n"
    "1:"  // Vector: Loop
    "movprfx z22, z30\n smax z22.b, p2/M, z22.b, z28.b\n"
    "movprfx z21, z28\n smax z21.b, p2/M, z21.b, z27.b\n"
    "ld1b { z30.b }, p1/Z, [x27, x15]\n"
    "whilelt p0.b, x14, x13\n"
    "movprfx z20, z29\n smax z20.b, p2/M, z20.b, z26.b\n"
    "movprfx z18, z25\n smax z18.b, p2/M, z18.b, z24.b\n"
    "ld1b { z28.b }, p1/Z, [x24, x15]\n"
    "movprfx z17, z29\n smax z17.b, p2/M, z17.b, z23.b\n"
    "movprfx z16, z24\n smax z16.b, p2/M, z16.b, z19.b\n"
    "ld1b { z27.b }, p1/Z, [x21, x15]\n"
    "ld1b { z29.b }, p1/Z, [x25, x15]\n"
    "movprfx z19, z22\n smax z19.b, p2/M, z19.b, z20.b\n"
    "smax z18.b, p2/M, z18.b, z22.b\n"
    "ld1b { z26.b }, p1/Z, [x28, x15]\n"
    "smax z17.b, p2/M, z17.b, z21.b\n"
    "smax z16.b, p2/M, z16.b, z21.b\n"
    "ld1b { z25.b }, p1/Z, [x26, x15]\n"
    "st1b { z19.b }, p0, [x12, x14]\n"
    "ld1b { z24.b }, p1/Z, [x23, x15]\n"
    "st1b { z18.b }, p0, [x11, x14]\n"
    "ld1b { z23.b }, p1/Z, [x22, x15]\n"
    "st1b { z17.b }, p0, [x10, x14]\n"
    "ld1b { z19.b }, p1/Z, [x20, x15]\n"
    "incw x15\n"
    "whilelt p1.b, x15, x13\n"
    "st1b { z16.b }, p0, [x9, x14]\n"
    "incw x14\n"
    "b.any 1b\n"
    "2:"  // Vector: Tail
    "movprfx z22, z30\n smax z22.b, p2/M, z22.b, z28.b\n"
    "movprfx z21, z28\n smax z21.b, p2/M, z21.b, z27.b\n"
    "whilelt p0.b, x14, x13\n"
    "movprfx z20, z29\n smax z20.b, p2/M, z20.b, z26.b\n"
    "movprfx z18, z25\n smax z18.b, p2/M, z18.b, z24.b\n"
    "movprfx z17, z29\n smax z17.b, p2/M, z17.b, z23.b\n"
    "movprfx z16, z24\n smax z16.b, p2/M, z16.b, z19.b\n"
    "movprfx z19, z22\n smax z19.b, p2/M, z19.b, z20.b\n"
    "smax z18.b, p2/M, z18.b, z22.b\n"
    "st1b { z19.b }, p0, [x12, x14]\n"
    "smax z17.b, p2/M, z17.b, z21.b\n"
    "smax z16.b, p2/M, z16.b, z21.b\n"
    "st1b { z18.b }, p0, [x11, x14]\n"
    "st1b { z17.b }, p0, [x10, x14]\n"
    "st1b { z16.b }, p0, [x9, x14]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs))
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE) && defined(ARM_COMPUTE_ENABLE_SME)
