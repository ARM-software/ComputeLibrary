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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(ARM_COMPUTE_ENABLE_SVE)

namespace arm_conv {
namespace pooling {

void sve_fp32_nhwc_avg_3x3_s1_output2x2_depthfirst_impl(
  const unsigned int n_channels,
  const float *const *const inptrs,
  float *const *const outptrs,
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
    const float *const *const inptrs;
    float *const *const outptrs;
    float rescale_vals[4];

    KernelArgs(
      unsigned int channels,
      const float *const *input_ptrs,
      float *const * output_ptrs,
      bool exclude_padding, unsigned int pad_left, unsigned int pad_top, unsigned int pad_right, unsigned int pad_bottom
    ) : n_channels(channels),
        inptrs(input_ptrs),
        outptrs(output_ptrs)
    {
      for (unsigned int i = 0; i < 2; i++)
      {
        const int start_i = 1*i - static_cast<int>(pad_top);
        const int end_i = std::min<int>(start_i + 3, 4 - pad_top - pad_bottom);
        const int valid_rows = end_i - std::max<int>(0, start_i);

        for (unsigned int j = 0; j < 2; j++)
        {
          const int start_j = 1*j - static_cast<int>(pad_left);
          const int end_j = std::min<int>(start_j + 3, 4 - pad_left - pad_right);
          const int valid_cols = end_j - std::max<int>(0, start_j);

          rescale_vals[i*2 + j] = static_cast<float>(1.0f / static_cast<float>(
            exclude_padding ? valid_rows * valid_cols : 9
          ));
        }
      }
    }
  };

  const KernelArgs args(n_channels, inptrs, outptrs, exclude_padding,
                        pad_left, pad_top, pad_right, pad_bottom);

  __asm__ __volatile__(
    "ldr x2, [%x[args], %[offsetof_n_channels]]\n"
    "ldr x21, [%x[args], %[offsetof_outptrs]]\n"
    "mov x3, #0x0\n"
    "mov x20, #0x4\n"
    "ldr x4, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x5, x6, [x21, #0x0]\n"
    "whilelt p0.s, XZR, x20\n"
    "whilelt p1.s, x3, x2\n"
    "ldp x7, x8, [x21, #0x10]\n"
    "ldp x17, x16, [x4, #0x0]\n"
    "add x15, %x[args], %[offsetof_rescale]\n"
    "mov x14, #0x0\n"
    "ldp x13, x12, [x4, #0x10]\n"
    "ldp x11, x10, [x4, #0x20]\n"
    "ldp x9, x28, [x4, #0x30]\n"
    "ldp x27, x26, [x4, #0x40]\n"
    "ldp x25, x24, [x4, #0x50]\n"
    "ldp x23, x22, [x4, #0x60]\n"
    "ldp x21, x20, [x4, #0x70]\n"
    "ld1w { z7.s }, p1/Z, [x10, x3, LSL #2]\n"
    "ld1w { z6.s }, p1/Z, [x9, x3, LSL #2]\n"
    "ld1w { z5.s }, p1/Z, [x26, x3, LSL #2]\n"
    "ld1w { z4.s }, p1/Z, [x25, x3, LSL #2]\n"
    "ld1w { z3.s }, p1/Z, [x16, x3, LSL #2]\n"
    "ld1w { z2.s }, p1/Z, [x13, x3, LSL #2]\n"
    "ld1w { z1.s }, p1/Z, [x11, x3, LSL #2]\n"
    "ld1w { z31.s }, p1/Z, [x27, x3, LSL #2]\n"
    "ld1w { z30.s }, p1/Z, [x28, x3, LSL #2]\n"
    "ld1w { z29.s }, p1/Z, [x24, x3, LSL #2]\n"
    "ld1w { z28.s }, p1/Z, [x22, x3, LSL #2]\n"
    "ld1w { z27.s }, p1/Z, [x21, x3, LSL #2]\n"
    "ld1w { z26.s }, p1/Z, [x17, x3, LSL #2]\n"
    "ld1w { z25.s }, p1/Z, [x12, x3, LSL #2]\n"
    "ld1w { z24.s }, p1/Z, [x23, x3, LSL #2]\n"
    "ld1w { z23.s }, p1/Z, [x20, x3, LSL #2]\n"
    "incw x3\n"
    "whilelt p1.s, x3, x2\n"
    "ld1rqw { z0.s }, p0/Z, [x15]\n"
    "b.none 2f\n"
    "1:"  // Vector: Loop
    "fadd z17.s, z7.s, z6.s\n"
    "fadd z16.s, z5.s, z4.s\n"
    "ld1w { z7.s }, p1/Z, [x10, x3, LSL #2]\n"
    "ld1w { z6.s }, p1/Z, [x9, x3, LSL #2]\n"
    "fadd z19.s, z17.s, z16.s\n"
    "fadd z18.s, z3.s, z2.s\n"
    "ld1w { z5.s }, p1/Z, [x26, x3, LSL #2]\n"
    "ld1w { z4.s }, p1/Z, [x25, x3, LSL #2]\n"
    "fadd z17.s, z1.s, z31.s\n"
    "fadd z22.s, z30.s, z29.s\n"
    "ld1w { z3.s }, p1/Z, [x16, x3, LSL #2]\n"
    "ld1w { z2.s }, p1/Z, [x13, x3, LSL #2]\n"
    "fadd z16.s, z28.s, z27.s\n"
    "fadd z21.s, z18.s, z19.s\n"
    "ld1w { z1.s }, p1/Z, [x11, x3, LSL #2]\n"
    "ld1w { z31.s }, p1/Z, [x27, x3, LSL #2]\n"
    "fadd z20.s, z16.s, z19.s\n"
    "fadd z19.s, z26.s, z17.s\n"
    "ld1w { z30.s }, p1/Z, [x28, x3, LSL #2]\n"
    "ld1w { z29.s }, p1/Z, [x24, x3, LSL #2]\n"
    "fadd z18.s, z25.s, z22.s\n"
    "fadd z17.s, z24.s, z17.s\n"
    "ld1w { z28.s }, p1/Z, [x22, x3, LSL #2]\n"
    "ld1w { z27.s }, p1/Z, [x21, x3, LSL #2]\n"
    "fadd z16.s, z23.s, z22.s\n"
    "ld1w { z26.s }, p1/Z, [x17, x3, LSL #2]\n"
    "ld1w { z25.s }, p1/Z, [x12, x3, LSL #2]\n"
    "fadd z19.s, z21.s, z19.s\n"
    "ld1w { z24.s }, p1/Z, [x23, x3, LSL #2]\n"
    "ld1w { z23.s }, p1/Z, [x20, x3, LSL #2]\n"
    "incw x3\n"
    "fadd z18.s, z21.s, z18.s\n"
    "fadd z17.s, z17.s, z20.s\n"
    "fadd z16.s, z16.s, z20.s\n"
    "whilelt p0.s, x14, x2\n"
    "whilelt p1.s, x3, x2\n"
    "fmul z19.s, z19.s, z0.s[0]\n"
    "fmul z18.s, z18.s, z0.s[1]\n"
    "st1w { z19.s }, p0, [x5, x14, LSL #2]\n"
    "fmul z17.s, z17.s, z0.s[2]\n"
    "fmul z16.s, z16.s, z0.s[3]\n"
    "st1w { z18.s }, p0, [x6, x14, LSL #2]\n"
    "st1w { z17.s }, p0, [x7, x14, LSL #2]\n"
    "st1w { z16.s }, p0, [x8, x14, LSL #2]\n"
    "incw x14\n"
    "b.any 1b\n"
    "2:"  // Vector: Tail
    "fadd z17.s, z7.s, z6.s\n"
    "fadd z16.s, z5.s, z4.s\n"
    "whilelt p0.s, x14, x2\n"
    "fadd z19.s, z17.s, z16.s\n"
    "fadd z18.s, z3.s, z2.s\n"
    "fadd z17.s, z1.s, z31.s\n"
    "fadd z22.s, z30.s, z29.s\n"
    "fadd z16.s, z28.s, z27.s\n"
    "fadd z21.s, z18.s, z19.s\n"
    "fadd z20.s, z16.s, z19.s\n"
    "fadd z19.s, z26.s, z17.s\n"
    "fadd z18.s, z25.s, z22.s\n"
    "fadd z17.s, z24.s, z17.s\n"
    "fadd z16.s, z23.s, z22.s\n"
    "fadd z19.s, z21.s, z19.s\n"
    "fmul z19.s, z19.s, z0.s[0]\n"
    "st1w { z19.s }, p0, [x5, x14, LSL #2]\n"
    "fadd z18.s, z21.s, z18.s\n"
    "fadd z17.s, z17.s, z20.s\n"
    "fmul z18.s, z18.s, z0.s[1]\n"
    "fmul z17.s, z17.s, z0.s[2]\n"
    "fadd z16.s, z16.s, z20.s\n"
    "fmul z16.s, z16.s, z0.s[3]\n"
    "st1w { z18.s }, p0, [x6, x14, LSL #2]\n"
    "st1w { z17.s }, p0, [x7, x14, LSL #2]\n"
    "st1w { z16.s }, p0, [x8, x14, LSL #2]\n"
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs)), [offsetof_rescale] "I" (offsetof(KernelArgs, rescale_vals))
    : "cc", "memory", "p0", "p1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE)
