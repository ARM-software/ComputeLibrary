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


#include <cstddef>
#include <cstdint>

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace pooling {

void sve_s8_nhwc_max_2x2_s1_output2x2_depthfirst_impl(
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
    "ldr x15, [%x[args], %[offsetof_n_channels]]\n"
    "ldr x21, [%x[args], %[offsetof_outptrs]]\n"
    "mov x14, #0x0\n"
    "whilelt p2.b, x14, x15\n"
    "ldr x20, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x13, x12, [x21, #0x0]\n"
    "ptrue p1.b\n"
    "mov x11, #0x0\n"
    "ldp x10, x9, [x21, #0x10]\n"
    "ldp x28, x27, [x20, #0x0]\n"
    "ldp x26, x25, [x20, #0x10]\n"
    "ldp x24, x23, [x20, #0x20]\n"
    "ldp x22, x21, [x20, #0x30]\n"
    "ldr x20, [x20, #0x40]\n"
    "ld1b { z31.b }, p2/Z, [x27, x14]\n"
    "ld1b { z30.b }, p2/Z, [x24, x14]\n"
    "ld1b { z29.b }, p2/Z, [x21, x14]\n"
    "ld1b { z28.b }, p2/Z, [x25, x14]\n"
    "ld1b { z27.b }, p2/Z, [x28, x14]\n"
    "ld1b { z26.b }, p2/Z, [x26, x14]\n"
    "ld1b { z25.b }, p2/Z, [x23, x14]\n"
    "ld1b { z24.b }, p2/Z, [x22, x14]\n"
    "ld1b { z23.b }, p2/Z, [x20, x14]\n"
    "incw x14\n"
    "whilelt p2.b, x14, x15\n"
    "b.none 2f\n"
    "1:"  // Vector: Loop
    "movprfx z22, z31\n smax z22.b, p1/M, z22.b, z30.b\n"
    "movprfx z21, z30\n smax z21.b, p1/M, z21.b, z29.b\n"
    "ld1b { z31.b }, p2/Z, [x27, x14]\n"
    "ld1b { z30.b }, p2/Z, [x24, x14]\n"
    "movprfx z20, z28\n smax z20.b, p1/M, z20.b, z27.b\n"
    "movprfx z19, z26\n smax z19.b, p1/M, z19.b, z25.b\n"
    "ld1b { z29.b }, p2/Z, [x21, x14]\n"
    "ld1b { z27.b }, p2/Z, [x28, x14]\n"
    "movprfx z17, z28\n smax z17.b, p1/M, z17.b, z24.b\n"
    "movprfx z18, z25\n smax z18.b, p1/M, z18.b, z23.b\n"
    "ld1b { z28.b }, p2/Z, [x25, x14]\n"
    "ld1b { z26.b }, p2/Z, [x26, x14]\n"
    "ld1b { z25.b }, p2/Z, [x23, x14]\n"
    "ld1b { z24.b }, p2/Z, [x22, x14]\n"
    "whilelt p0.b, x11, x15\n"
    "movprfx z16, z22\n smax z16.b, p1/M, z16.b, z20.b\n"
    "ld1b { z23.b }, p2/Z, [x20, x14]\n"
    "incw x14\n"
    "whilelt p2.b, x14, x15\n"
    "st1b { z16.b }, p0, [x13, x11]\n"
    "movprfx z16, z19\n smax z16.b, p1/M, z16.b, z22.b\n"
    "smax z17.b, p1/M, z17.b, z21.b\n"
    "st1b { z16.b }, p0, [x12, x11]\n"
    "movprfx z16, z18\n smax z16.b, p1/M, z16.b, z21.b\n"
    "st1b { z17.b }, p0, [x10, x11]\n"
    "st1b { z16.b }, p0, [x9, x11]\n"
    "incw x11\n"
    "b.any 1b\n"
    "2:"  // Vector: Tail
    "movprfx z22, z31\n smax z22.b, p1/M, z22.b, z30.b\n"
    "movprfx z21, z30\n smax z21.b, p1/M, z21.b, z29.b\n"
    "movprfx z20, z28\n smax z20.b, p1/M, z20.b, z27.b\n"
    "movprfx z19, z26\n smax z19.b, p1/M, z19.b, z25.b\n"
    "movprfx z17, z28\n smax z17.b, p1/M, z17.b, z24.b\n"
    "movprfx z18, z25\n smax z18.b, p1/M, z18.b, z23.b\n"
    "whilelt p0.b, x11, x15\n"
    "movprfx z16, z22\n smax z16.b, p1/M, z16.b, z20.b\n"
    "st1b { z16.b }, p0, [x13, x11]\n"
    "movprfx z16, z19\n smax z16.b, p1/M, z16.b, z22.b\n"
    "smax z17.b, p1/M, z17.b, z21.b\n"
    "st1b { z16.b }, p0, [x12, x11]\n"
    "movprfx z16, z18\n smax z16.b, p1/M, z16.b, z21.b\n"
    "st1b { z17.b }, p0, [x10, x11]\n"
    "st1b { z16.b }, p0, [x9, x11]\n"
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs))
    : "cc", "memory", "p0", "p1", "p2", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
