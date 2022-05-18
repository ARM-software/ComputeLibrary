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

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)

namespace arm_conv {
namespace pooling {

void sve_fp16_nhwc_avg_3x3_s1_output2x2_depthfirst_impl(
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
    __fp16 rescale_vals[4];

    KernelArgs(
      unsigned int channels,
      const __fp16 *const *input_ptrs,
      __fp16 *const * output_ptrs,
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

          rescale_vals[i*2 + j] = static_cast<__fp16>(1.0f / static_cast<float>(
            exclude_padding ? valid_rows * valid_cols : 9
          ));
        }
      }
    }
  };

  const KernelArgs args(n_channels, inptrs, outptrs, exclude_padding,
                        pad_left, pad_top, pad_right, pad_bottom);

  __asm__ __volatile__(
    "ldr x3, [%x[args], %[offsetof_n_channels]]\n"
    "ldr x20, [%x[args], %[offsetof_outptrs]]\n"
    "mov x4, #0x0\n"
    "mov x19, #0x4\n"
    "ldr x5, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x6, x7, [x20, #0x0]\n"
    "whilelt p0.h, XZR, x19\n"
    "whilelt p1.h, x4, x3\n"
    "ldp x8, x17, [x20, #0x10]\n"
    "ldp x16, x15, [x5, #0x0]\n"
    "add x14, %x[args], %[offsetof_rescale]\n"
    "mov x13, #0x0\n"
    "ldp x12, x11, [x5, #0x10]\n"
    "ldp x10, x9, [x5, #0x20]\n"
    "ldp x28, x27, [x5, #0x30]\n"
    "ldp x26, x25, [x5, #0x40]\n"
    "ldp x24, x23, [x5, #0x50]\n"
    "ldp x22, x21, [x5, #0x60]\n"
    "ldp x20, x19, [x5, #0x70]\n"
    "ld1h { z7.h }, p1/Z, [x9, x4, LSL #1]\n"
    "ld1h { z6.h }, p1/Z, [x28, x4, LSL #1]\n"
    "ld1h { z5.h }, p1/Z, [x25, x4, LSL #1]\n"
    "ld1h { z4.h }, p1/Z, [x24, x4, LSL #1]\n"
    "ld1h { z3.h }, p1/Z, [x15, x4, LSL #1]\n"
    "ld1h { z2.h }, p1/Z, [x12, x4, LSL #1]\n"
    "ld1h { z1.h }, p1/Z, [x10, x4, LSL #1]\n"
    "ld1h { z31.h }, p1/Z, [x26, x4, LSL #1]\n"
    "ld1h { z30.h }, p1/Z, [x27, x4, LSL #1]\n"
    "ld1h { z29.h }, p1/Z, [x23, x4, LSL #1]\n"
    "ld1h { z28.h }, p1/Z, [x21, x4, LSL #1]\n"
    "ld1h { z27.h }, p1/Z, [x20, x4, LSL #1]\n"
    "ld1h { z26.h }, p1/Z, [x16, x4, LSL #1]\n"
    "ld1h { z25.h }, p1/Z, [x11, x4, LSL #1]\n"
    "ld1h { z24.h }, p1/Z, [x22, x4, LSL #1]\n"
    "ld1h { z23.h }, p1/Z, [x19, x4, LSL #1]\n"
    "incw x4\n"
    "whilelt p1.h, x4, x3\n"
    "ld1rqh { z0.h }, p0/Z, [x14]\n"
    "b.none 2f\n"
    "1:"  // Vector: Loop
    "fadd z17.h, z7.h, z6.h\n"
    "fadd z16.h, z5.h, z4.h\n"
    "ld1h { z7.h }, p1/Z, [x9, x4, LSL #1]\n"
    "ld1h { z6.h }, p1/Z, [x28, x4, LSL #1]\n"
    "fadd z19.h, z17.h, z16.h\n"
    "fadd z18.h, z3.h, z2.h\n"
    "ld1h { z5.h }, p1/Z, [x25, x4, LSL #1]\n"
    "ld1h { z4.h }, p1/Z, [x24, x4, LSL #1]\n"
    "fadd z17.h, z1.h, z31.h\n"
    "fadd z22.h, z30.h, z29.h\n"
    "ld1h { z3.h }, p1/Z, [x15, x4, LSL #1]\n"
    "ld1h { z2.h }, p1/Z, [x12, x4, LSL #1]\n"
    "fadd z16.h, z28.h, z27.h\n"
    "fadd z21.h, z18.h, z19.h\n"
    "ld1h { z1.h }, p1/Z, [x10, x4, LSL #1]\n"
    "ld1h { z31.h }, p1/Z, [x26, x4, LSL #1]\n"
    "fadd z20.h, z16.h, z19.h\n"
    "fadd z19.h, z26.h, z17.h\n"
    "ld1h { z30.h }, p1/Z, [x27, x4, LSL #1]\n"
    "ld1h { z29.h }, p1/Z, [x23, x4, LSL #1]\n"
    "fadd z18.h, z25.h, z22.h\n"
    "fadd z17.h, z24.h, z17.h\n"
    "ld1h { z28.h }, p1/Z, [x21, x4, LSL #1]\n"
    "ld1h { z27.h }, p1/Z, [x20, x4, LSL #1]\n"
    "fadd z16.h, z23.h, z22.h\n"
    "ld1h { z26.h }, p1/Z, [x16, x4, LSL #1]\n"
    "ld1h { z25.h }, p1/Z, [x11, x4, LSL #1]\n"
    "fadd z19.h, z19.h, z21.h\n"
    "ld1h { z24.h }, p1/Z, [x22, x4, LSL #1]\n"
    "ld1h { z23.h }, p1/Z, [x19, x4, LSL #1]\n"
    "incw x4\n"
    "fadd z18.h, z18.h, z21.h\n"
    "fadd z17.h, z17.h, z20.h\n"
    "fadd z16.h, z16.h, z20.h\n"
    "whilelt p0.h, x13, x3\n"
    "whilelt p1.h, x4, x3\n"
    "fmul z19.h, z19.h, z0.h[0]\n"
    "fmul z18.h, z18.h, z0.h[1]\n"
    "st1h { z19.h }, p0, [x6, x13, LSL #1]\n"
    "fmul z17.h, z17.h, z0.h[2]\n"
    "fmul z16.h, z16.h, z0.h[3]\n"
    "st1h { z18.h }, p0, [x7, x13, LSL #1]\n"
    "st1h { z17.h }, p0, [x8, x13, LSL #1]\n"
    "st1h { z16.h }, p0, [x17, x13, LSL #1]\n"
    "incw x13\n"
    "b.any 1b\n"
    "2:"  // Vector: Tail
    "fadd z17.h, z7.h, z6.h\n"
    "fadd z16.h, z5.h, z4.h\n"
    "whilelt p0.h, x13, x3\n"
    "fadd z19.h, z17.h, z16.h\n"
    "fadd z18.h, z3.h, z2.h\n"
    "fadd z17.h, z1.h, z31.h\n"
    "fadd z22.h, z30.h, z29.h\n"
    "fadd z16.h, z28.h, z27.h\n"
    "fadd z21.h, z18.h, z19.h\n"
    "fadd z20.h, z16.h, z19.h\n"
    "fadd z19.h, z26.h, z17.h\n"
    "fadd z18.h, z25.h, z22.h\n"
    "fadd z17.h, z24.h, z17.h\n"
    "fadd z16.h, z23.h, z22.h\n"
    "fadd z19.h, z19.h, z21.h\n"
    "fmul z19.h, z19.h, z0.h[0]\n"
    "st1h { z19.h }, p0, [x6, x13, LSL #1]\n"
    "fadd z18.h, z18.h, z21.h\n"
    "fadd z17.h, z17.h, z20.h\n"
    "fmul z18.h, z18.h, z0.h[1]\n"
    "fmul z17.h, z17.h, z0.h[2]\n"
    "fadd z16.h, z16.h, z20.h\n"
    "fmul z16.h, z16.h, z0.h[3]\n"
    "st1h { z18.h }, p0, [x7, x13, LSL #1]\n"
    "st1h { z17.h }, p0, [x8, x13, LSL #1]\n"
    "st1h { z16.h }, p0, [x17, x13, LSL #1]\n"
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs)), [offsetof_rescale] "I" (offsetof(KernelArgs, rescale_vals))
    : "cc", "memory", "p0", "p1", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(ARM_COMPUTE_ENABLE_SVE) && defined(__ARM_FP16_ARGS)
