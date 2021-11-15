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

#if defined(__aarch64__)
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace arm_conv {
namespace pooling {

void a64_fp32_nhwc_avg_3x3_s1_output2x2_depthfirst_impl(
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
    "ldr x4, [%x[args], %[offsetof_n_channels]]\n"
    "mov x5, #0x0\n"
    "ldr x20, [%x[args], %[offsetof_outptrs]]\n"
    "mov x6, #0x0\n"
    "ldr x19, [%x[args], %[offsetof_inptrs]]\n"
    "cmp x4, #0x4\n"
    "ldp x7, x8, [x20, #0x0]\n"
    "ldp x17, x16, [x20, #0x10]\n"
    "ldp x15, x14, [x19, #0x0]\n"
    "ldp x13, x12, [x19, #0x10]\n"
    "ldp x11, x10, [x19, #0x20]\n"
    "ldp x9, x28, [x19, #0x30]\n"
    "ldp x27, x26, [x19, #0x40]\n"
    "ldp x25, x24, [x19, #0x50]\n"
    "ldp x23, x22, [x19, #0x60]\n"
    "ldp x21, x20, [x19, #0x70]\n"
    "ldr q8, [%x[args], %[offsetof_rescale]]\n"
    "blt 3f\n"
    "ldr q7, [x10, x5]\n"
    "lsr x19, x4, #0x2\n"
    "ldr q6, [x9, x5]\n"
    "sub x4, x4, x19, LSL #2\n"
    "ldr q5, [x26, x5]\n"
    "subs x19, x19, #0x1\n"
    "ldr q4, [x25, x5]\n"
    "ldr q3, [x14, x5]\n"
    "ldr q2, [x13, x5]\n"
    "ldr q1, [x11, x5]\n"
    "ldr q0, [x27, x5]\n"
    "ldr q31, [x28, x5]\n"
    "ldr q30, [x24, x5]\n"
    "ldr q29, [x22, x5]\n"
    "ldr q28, [x21, x5]\n"
    "ldr q27, [x15, x5]\n"
    "ldr q26, [x12, x5]\n"
    "ldr q25, [x23, x5]\n"
    "ldr q24, [x20, x5]\n"
    "add x5, x5, #0x10\n"
    "beq 2f\n"
    "1:"  // Vector: Loop
    "fadd v17.4s, v7.4s, v6.4s\n"
    "ldr q7, [x10, x5]\n"
    "subs x19, x19, #0x1\n"
    "fadd v16.4s, v5.4s, v4.4s\n"
    "ldr q6, [x9, x5]\n"
    "fadd v18.4s, v3.4s, v2.4s\n"
    "ldr q5, [x26, x5]\n"
    "fadd v23.4s, v1.4s, v0.4s\n"
    "ldr q4, [x25, x5]\n"
    "fadd v22.4s, v31.4s, v30.4s\n"
    "ldr q3, [x14, x5]\n"
    "fadd v17.4s, v17.4s, v16.4s\n"
    "ldr q2, [x13, x5]\n"
    "fadd v16.4s, v29.4s, v28.4s\n"
    "ldr q1, [x11, x5]\n"
    "fadd v19.4s, v27.4s, v23.4s\n"
    "ldr q0, [x27, x5]\n"
    "fadd v21.4s, v18.4s, v17.4s\n"
    "ldr q31, [x28, x5]\n"
    "fadd v20.4s, v16.4s, v17.4s\n"
    "ldr q30, [x24, x5]\n"
    "fadd v18.4s, v26.4s, v22.4s\n"
    "ldr q29, [x22, x5]\n"
    "fadd v17.4s, v25.4s, v23.4s\n"
    "ldr q28, [x21, x5]\n"
    "fadd v16.4s, v24.4s, v22.4s\n"
    "ldr q27, [x15, x5]\n"
    "fadd v19.4s, v21.4s, v19.4s\n"
    "ldr q26, [x12, x5]\n"
    "fadd v18.4s, v21.4s, v18.4s\n"
    "ldr q25, [x23, x5]\n"
    "fadd v17.4s, v17.4s, v20.4s\n"
    "ldr q24, [x20, x5]\n"
    "add x5, x5, #0x10\n"
    "fadd v16.4s, v20.4s, v16.4s\n"
    "fmul v19.4s, v19.4s, v8.s[0]\n"
    "str q19, [x7, x6]\n"
    "fmul v18.4s, v18.4s, v8.s[1]\n"
    "fmul v17.4s, v17.4s, v8.s[2]\n"
    "str q18, [x8, x6]\n"
    "fmul v16.4s, v16.4s, v8.s[3]\n"
    "str q17, [x17, x6]\n"
    "str q16, [x16, x6]\n"
    "add x6, x6, #0x10\n"
    "bgt 1b\n"
    "2:"  // Vector: Tail
    "fadd v17.4s, v7.4s, v6.4s\n"
    "fadd v16.4s, v5.4s, v4.4s\n"
    "fadd v18.4s, v3.4s, v2.4s\n"
    "fadd v23.4s, v1.4s, v0.4s\n"
    "fadd v17.4s, v17.4s, v16.4s\n"
    "fadd v22.4s, v31.4s, v30.4s\n"
    "fadd v16.4s, v29.4s, v28.4s\n"
    "fadd v21.4s, v18.4s, v17.4s\n"
    "fadd v19.4s, v27.4s, v23.4s\n"
    "fadd v20.4s, v16.4s, v17.4s\n"
    "fadd v18.4s, v26.4s, v22.4s\n"
    "fadd v17.4s, v25.4s, v23.4s\n"
    "fadd v16.4s, v24.4s, v22.4s\n"
    "fadd v19.4s, v21.4s, v19.4s\n"
    "fadd v18.4s, v21.4s, v18.4s\n"
    "fadd v17.4s, v17.4s, v20.4s\n"
    "fadd v16.4s, v20.4s, v16.4s\n"
    "fmul v19.4s, v19.4s, v8.s[0]\n"
    "str q19, [x7, x6]\n"
    "fmul v18.4s, v18.4s, v8.s[1]\n"
    "fmul v17.4s, v17.4s, v8.s[2]\n"
    "str q18, [x8, x6]\n"
    "fmul v16.4s, v16.4s, v8.s[3]\n"
    "str q17, [x17, x6]\n"
    "str q16, [x16, x6]\n"
    "add x6, x6, #0x10\n"
    "cbz x4, 4f\n"
    "3:"  // Oddments
    "ldr s7, [x10, x5]\n"
    "subs x4, x4, #0x1\n"
    "ldr s6, [x9, x5]\n"
    "fadd v17.4s, v7.4s, v6.4s\n"
    "ldr s5, [x26, x5]\n"
    "ldr s4, [x25, x5]\n"
    "fadd v16.4s, v5.4s, v4.4s\n"
    "ldr s3, [x14, x5]\n"
    "ldr s2, [x13, x5]\n"
    "fadd v17.4s, v17.4s, v16.4s\n"
    "ldr s1, [x11, x5]\n"
    "ldr s0, [x27, x5]\n"
    "fadd v18.4s, v3.4s, v2.4s\n"
    "ldr s31, [x28, x5]\n"
    "fadd v23.4s, v1.4s, v0.4s\n"
    "ldr s30, [x24, x5]\n"
    "fadd v21.4s, v18.4s, v17.4s\n"
    "ldr s29, [x22, x5]\n"
    "ldr s28, [x21, x5]\n"
    "fadd v22.4s, v31.4s, v30.4s\n"
    "ldr s27, [x15, x5]\n"
    "ldr s26, [x12, x5]\n"
    "fadd v16.4s, v29.4s, v28.4s\n"
    "ldr s25, [x23, x5]\n"
    "fadd v20.4s, v16.4s, v17.4s\n"
    "ldr s24, [x20, x5]\n"
    "add x5, x5, #0x4\n"
    "fadd v19.4s, v27.4s, v23.4s\n"
    "fadd v18.4s, v26.4s, v22.4s\n"
    "fadd v17.4s, v25.4s, v23.4s\n"
    "fadd v16.4s, v24.4s, v22.4s\n"
    "fadd v19.4s, v21.4s, v19.4s\n"
    "fadd v18.4s, v21.4s, v18.4s\n"
    "fadd v17.4s, v17.4s, v20.4s\n"
    "fadd v16.4s, v20.4s, v16.4s\n"
    "fmul v19.4s, v19.4s, v8.s[0]\n"
    "str s19, [x7, x6]\n"
    "fmul v18.4s, v18.4s, v8.s[1]\n"
    "fmul v17.4s, v17.4s, v8.s[2]\n"
    "str s18, [x8, x6]\n"
    "fmul v16.4s, v16.4s, v8.s[3]\n"
    "str s17, [x17, x6]\n"
    "str s16, [x16, x6]\n"
    "add x6, x6, #0x4\n"
    "bgt 3b\n"
    "4:"  // End

    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs)), [offsetof_rescale] "I" (offsetof(KernelArgs, rescale_vals))
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv
#endif // defined(__aarch64__)
