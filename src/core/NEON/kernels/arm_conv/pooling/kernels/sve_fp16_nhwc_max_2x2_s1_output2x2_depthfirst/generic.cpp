/*
 * Copyright (c) 2021-2022 Arm Limited.
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

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace pooling {

void sve_fp16_nhwc_max_2x2_s1_output2x2_depthfirst_impl(
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
    "ldr x14, [%x[args], %[offsetof_n_channels]]\n"
    "ldr x20, [%x[args], %[offsetof_outptrs]]\n"
    "mov x13, #0x0\n"
    "whilelt p2.h, x13, x14\n"
    "ldr x19, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x12, x11, [x20, #0x0]\n"
    "ptrue p1.b\n"
    "mov x10, #0x0\n"
    "ldp x9, x28, [x20, #0x10]\n"
    "ldp x27, x26, [x19, #0x0]\n"
    "ldp x25, x24, [x19, #0x10]\n"
    "ldp x23, x22, [x19, #0x20]\n"
    "ldp x21, x20, [x19, #0x30]\n"
    "ldr x19, [x19, #0x40]\n"
    "ld1h { z31.h }, p2/Z, [x26, x13, LSL #1]\n"
    "ld1h { z30.h }, p2/Z, [x23, x13, LSL #1]\n"
    "ld1h { z29.h }, p2/Z, [x20, x13, LSL #1]\n"
    "ld1h { z28.h }, p2/Z, [x24, x13, LSL #1]\n"
    "ld1h { z27.h }, p2/Z, [x27, x13, LSL #1]\n"
    "ld1h { z26.h }, p2/Z, [x22, x13, LSL #1]\n"
    "ld1h { z25.h }, p2/Z, [x25, x13, LSL #1]\n"
    "ld1h { z24.h }, p2/Z, [x21, x13, LSL #1]\n"
    "ld1h { z23.h }, p2/Z, [x19, x13, LSL #1]\n"
    "incw x13\n"
    "whilelt p2.h, x13, x14\n"
    "b.none 2f\n"
    "1:"  // Vector: Loop
    "movprfx z22, z31\n fmax z22.h, p1/M, z22.h, z30.h\n"
    "movprfx z21, z30\n fmax z21.h, p1/M, z21.h, z29.h\n"
    "ld1h { z31.h }, p2/Z, [x26, x13, LSL #1]\n"
    "ld1h { z30.h }, p2/Z, [x23, x13, LSL #1]\n"
    "movprfx z20, z28\n fmax z20.h, p1/M, z20.h, z27.h\n"
    "movprfx z17, z26\n fmax z17.h, p1/M, z17.h, z25.h\n"
    "ld1h { z29.h }, p2/Z, [x20, x13, LSL #1]\n"
    "ld1h { z27.h }, p2/Z, [x27, x13, LSL #1]\n"
    "movprfx z19, z24\n fmax z19.h, p1/M, z19.h, z28.h\n"
    "movprfx z18, z26\n fmax z18.h, p1/M, z18.h, z23.h\n"
    "ld1h { z28.h }, p2/Z, [x24, x13, LSL #1]\n"
    "ld1h { z26.h }, p2/Z, [x22, x13, LSL #1]\n"
    "ld1h { z25.h }, p2/Z, [x25, x13, LSL #1]\n"
    "ld1h { z24.h }, p2/Z, [x21, x13, LSL #1]\n"
    "whilelt p0.h, x10, x14\n"
    "movprfx z16, z22\n fmax z16.h, p1/M, z16.h, z20.h\n"
    "ld1h { z23.h }, p2/Z, [x19, x13, LSL #1]\n"
    "incw x13\n"
    "whilelt p2.h, x13, x14\n"
    "st1h { z16.h }, p0, [x12, x10, LSL #1]\n"
    "movprfx z16, z17\n fmax z16.h, p1/M, z16.h, z22.h\n"
    "movprfx z17, z21\n fmax z17.h, p1/M, z17.h, z19.h\n"
    "st1h { z16.h }, p0, [x11, x10, LSL #1]\n"
    "movprfx z16, z21\n fmax z16.h, p1/M, z16.h, z18.h\n"
    "st1h { z17.h }, p0, [x9, x10, LSL #1]\n"
    "st1h { z16.h }, p0, [x28, x10, LSL #1]\n"
    "incw x10\n"
    "b.any 1b\n"
    "2:"  // Vector: Tail
    "movprfx z22, z31\n fmax z22.h, p1/M, z22.h, z30.h\n"
    "movprfx z21, z30\n fmax z21.h, p1/M, z21.h, z29.h\n"
    "movprfx z20, z28\n fmax z20.h, p1/M, z20.h, z27.h\n"
    "movprfx z17, z26\n fmax z17.h, p1/M, z17.h, z25.h\n"
    "movprfx z19, z24\n fmax z19.h, p1/M, z19.h, z28.h\n"
    "movprfx z18, z26\n fmax z18.h, p1/M, z18.h, z23.h\n"
    "whilelt p0.h, x10, x14\n"
    "movprfx z16, z22\n fmax z16.h, p1/M, z16.h, z20.h\n"
    "st1h { z16.h }, p0, [x12, x10, LSL #1]\n"
    "movprfx z16, z17\n fmax z16.h, p1/M, z16.h, z22.h\n"
    "movprfx z17, z21\n fmax z17.h, p1/M, z17.h, z19.h\n"
    "st1h { z16.h }, p0, [x11, x10, LSL #1]\n"
    "movprfx z16, z21\n fmax z16.h, p1/M, z16.h, z18.h\n"
    "st1h { z17.h }, p0, [x9, x10, LSL #1]\n"
    "st1h { z16.h }, p0, [x28, x10, LSL #1]\n"
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs))
    : "cc", "memory", "p0", "p1", "p2", "x9", "x10", "x11", "x12", "x13", "x14", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)
