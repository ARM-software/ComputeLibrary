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


#include <cstddef>
#include <cstdint>

#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS) && defined(ARM_COMPUTE_ENABLE_SME)

namespace arm_conv {
namespace pooling {

void sme_fp16_nhwc_max_2x2_s1_output2x2_depthfirst_impl(
  const unsigned int n_channels,
  const __fp16 *const *const inptrs,
  __fp16 *const *const outptrs,
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
    const __fp16 *const *const inptrs;
    __fp16 *const *const outptrs;
    KernelArgs(
      unsigned int channels,
      const __fp16 *const *input_ptrs,
      __fp16 *const * output_ptrs,
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
    "ldr x20, [%x[args], %[offsetof_outptrs]]\n"
    ".inst 0xd503477f  // SMSTART ZA\n"
    "mov x14, #0x0\n"
    "ptrue p2.b\n"
    "ldr x19, [%x[args], %[offsetof_inptrs]]\n"
    "mov x13, #0x0\n"
    "ldr x12, [%x[args], %[offsetof_n_channels]]\n"
    "whilelt p1.h, x14, x12\n"
    "ldp x11, x10, [x20, #0x0]\n"
    "ldp x9, x28, [x20, #0x10]\n"
    "ldp x27, x26, [x19, #0x0]\n"
    "ld1h { z29.h }, p1/Z, [x26, x14, LSL #1]\n"
    "ldp x25, x24, [x19, #0x10]\n"
    "ld1h { z28.h }, p1/Z, [x24, x14, LSL #1]\n"
    "ldp x23, x22, [x19, #0x20]\n"
    "ld1h { z27.h }, p1/Z, [x23, x14, LSL #1]\n"
    "ldp x21, x20, [x19, #0x30]\n"
    "ld1h { z26.h }, p1/Z, [x20, x14, LSL #1]\n"
    "ldr x19, [x19, #0x40]\n"
    "ld1h { z20.h }, p1/Z, [x27, x14, LSL #1]\n"
    "ld1h { z25.h }, p1/Z, [x22, x14, LSL #1]\n"
    "ld1h { z24.h }, p1/Z, [x25, x14, LSL #1]\n"
    "ld1h { z23.h }, p1/Z, [x21, x14, LSL #1]\n"
    "ld1h { z19.h }, p1/Z, [x19, x14, LSL #1]\n"
    "incw x14\n"
    "whilelt p1.h, x14, x12\n"
    "b.none 2f\n"
    "1:"  // Vector: Loop
    "movprfx z22, z29\n fmax z22.h, p2/M, z22.h, z27.h\n"
    "movprfx z21, z27\n fmax z21.h, p2/M, z21.h, z26.h\n"
    "ld1h { z29.h }, p1/Z, [x26, x14, LSL #1]\n"
    "whilelt p0.h, x13, x12\n"
    "movprfx z18, z28\n fmax z18.h, p2/M, z18.h, z20.h\n"
    "movprfx z20, z25\n fmax z20.h, p2/M, z20.h, z24.h\n"
    "ld1h { z27.h }, p1/Z, [x23, x14, LSL #1]\n"
    "movprfx z17, z23\n fmax z17.h, p2/M, z17.h, z28.h\n"
    "movprfx z16, z25\n fmax z16.h, p2/M, z16.h, z19.h\n"
    "ld1h { z26.h }, p1/Z, [x20, x14, LSL #1]\n"
    "ld1h { z28.h }, p1/Z, [x24, x14, LSL #1]\n"
    "movprfx z19, z18\n fmax z19.h, p2/M, z19.h, z22.h\n"
    "movprfx z18, z22\n fmax z18.h, p2/M, z18.h, z20.h\n"
    "ld1h { z20.h }, p1/Z, [x27, x14, LSL #1]\n"
    "fmax z17.h, p2/M, z17.h, z21.h\n"
    "fmax z16.h, p2/M, z16.h, z21.h\n"
    "ld1h { z25.h }, p1/Z, [x22, x14, LSL #1]\n"
    "st1h { z19.h }, p0, [x11, x13, LSL #1]\n"
    "ld1h { z24.h }, p1/Z, [x25, x14, LSL #1]\n"
    "st1h { z18.h }, p0, [x10, x13, LSL #1]\n"
    "ld1h { z23.h }, p1/Z, [x21, x14, LSL #1]\n"
    "st1h { z17.h }, p0, [x9, x13, LSL #1]\n"
    "ld1h { z19.h }, p1/Z, [x19, x14, LSL #1]\n"
    "incw x14\n"
    "whilelt p1.h, x14, x12\n"
    "st1h { z16.h }, p0, [x28, x13, LSL #1]\n"
    "incw x13\n"
    "b.any 1b\n"
    "2:"  // Vector: Tail
    "movprfx z22, z29\n fmax z22.h, p2/M, z22.h, z27.h\n"
    "movprfx z21, z27\n fmax z21.h, p2/M, z21.h, z26.h\n"
    "whilelt p0.h, x13, x12\n"
    "movprfx z18, z28\n fmax z18.h, p2/M, z18.h, z20.h\n"
    "movprfx z20, z25\n fmax z20.h, p2/M, z20.h, z24.h\n"
    "movprfx z17, z23\n fmax z17.h, p2/M, z17.h, z28.h\n"
    "movprfx z16, z25\n fmax z16.h, p2/M, z16.h, z19.h\n"
    "movprfx z19, z18\n fmax z19.h, p2/M, z19.h, z22.h\n"
    "movprfx z18, z22\n fmax z18.h, p2/M, z18.h, z20.h\n"
    "st1h { z19.h }, p0, [x11, x13, LSL #1]\n"
    "fmax z17.h, p2/M, z17.h, z21.h\n"
    "fmax z16.h, p2/M, z16.h, z21.h\n"
    "st1h { z18.h }, p0, [x10, x13, LSL #1]\n"
    "st1h { z17.h }, p0, [x9, x13, LSL #1]\n"
    "st1h { z16.h }, p0, [x28, x13, LSL #1]\n"
    ".inst 0xd503467f  // SMSTOP\n"
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs))
    : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "x9", "x10", "x11", "x12", "x13", "x14", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__ARM_FEATURE_SVE) && defined(__ARM_FP16_ARGS) && defined(ARM_COMPUTE_ENABLE_SME)
