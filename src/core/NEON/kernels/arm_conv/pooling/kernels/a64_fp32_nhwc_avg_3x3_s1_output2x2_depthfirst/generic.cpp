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

#if defined(__aarch64__)

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
    "ldr q7, [%x[args], %[offsetof_rescale]]\n"
    "ldr x3, [%x[args], %[offsetof_n_channels]]\n"
    "cmp x3, #0x4\n"
    "mov x4, #0x0\n"
    "ldr x21, [%x[args], %[offsetof_outptrs]]\n"
    "ldr x20, [%x[args], %[offsetof_inptrs]]\n"
    "mov x5, #0x0\n"
    "ldp x6, x7, [x21, #0x0]\n"
    "ldp x8, x17, [x21, #0x10]\n"
    "ldp x16, x15, [x20, #0x0]\n"
    "ldp x14, x13, [x20, #0x10]\n"
    "ldp x12, x11, [x20, #0x20]\n"
    "ldp x10, x9, [x20, #0x30]\n"
    "ldp x28, x27, [x20, #0x40]\n"
    "ldp x26, x25, [x20, #0x50]\n"
    "ldp x24, x23, [x20, #0x60]\n"
    "ldp x22, x21, [x20, #0x70]\n"
    "blt 3f\n"
    "ldr q6, [x11, x4]\n"
    "ldr q5, [x10, x4]\n"
    "lsr x20, x3, #0x2\n"
    "sub x3, x3, x20, LSL #2\n"
    "ldr q4, [x27, x4]\n"
    "ldr q3, [x26, x4]\n"
    "subs x20, x20, #0x1\n"
    "ldr q2, [x15, x4]\n"
    "ldr q1, [x14, x4]\n"
    "ldr q0, [x12, x4]\n"
    "ldr q31, [x28, x4]\n"
    "ldr q30, [x9, x4]\n"
    "ldr q29, [x25, x4]\n"
    "ldr q28, [x23, x4]\n"
    "ldr q27, [x22, x4]\n"
    "ldr q26, [x16, x4]\n"
    "ldr q25, [x13, x4]\n"
    "ldr q24, [x24, x4]\n"
    "ldr q23, [x21, x4]\n"
    "add x4, x4, #0x10\n"
    "beq 2f\n"
    "1:"  // Vector: Loop
    "fadd v17.4s, v6.4s, v5.4s\n"
    "ldr q6, [x11, x4]\n"
    "ldr q5, [x10, x4]\n"
    "fadd v16.4s, v4.4s, v3.4s\n"
    "ldr q4, [x27, x4]\n"
    "ldr q3, [x26, x4]\n"
    "fadd v19.4s, v17.4s, v16.4s\n"
    "fadd v18.4s, v2.4s, v1.4s\n"
    "ldr q2, [x15, x4]\n"
    "ldr q1, [x14, x4]\n"
    "fadd v17.4s, v0.4s, v31.4s\n"
    "fadd v22.4s, v30.4s, v29.4s\n"
    "ldr q0, [x12, x4]\n"
    "ldr q31, [x28, x4]\n"
    "fadd v16.4s, v28.4s, v27.4s\n"
    "fadd v21.4s, v18.4s, v19.4s\n"
    "ldr q30, [x9, x4]\n"
    "ldr q29, [x25, x4]\n"
    "fadd v20.4s, v16.4s, v19.4s\n"
    "fadd v19.4s, v26.4s, v17.4s\n"
    "ldr q28, [x23, x4]\n"
    "ldr q27, [x22, x4]\n"
    "fadd v18.4s, v25.4s, v22.4s\n"
    "fadd v17.4s, v24.4s, v17.4s\n"
    "ldr q26, [x16, x4]\n"
    "ldr q25, [x13, x4]\n"
    "fadd v16.4s, v23.4s, v22.4s\n"
    "fadd v19.4s, v21.4s, v19.4s\n"
    "ldr q24, [x24, x4]\n"
    "ldr q23, [x21, x4]\n"
    "fadd v18.4s, v21.4s, v18.4s\n"
    "fadd v17.4s, v17.4s, v20.4s\n"
    "fadd v16.4s, v16.4s, v20.4s\n"
    "subs x20, x20, #0x1\n"
    "fmul v19.4s, v19.4s, v7.s[0]\n"
    "add x4, x4, #0x10\n"
    "fmul v18.4s, v18.4s, v7.s[1]\n"
    "fmul v17.4s, v17.4s, v7.s[2]\n"
    "str q19, [x6, x5]\n"
    "fmul v16.4s, v16.4s, v7.s[3]\n"
    "str q18, [x7, x5]\n"
    "str q17, [x8, x5]\n"
    "str q16, [x17, x5]\n"
    "add x5, x5, #0x10\n"
    "bgt 1b\n"
    "2:"  // Vector: Tail
    "fadd v17.4s, v6.4s, v5.4s\n"
    "fadd v16.4s, v4.4s, v3.4s\n"
    "fadd v19.4s, v17.4s, v16.4s\n"
    "fadd v18.4s, v2.4s, v1.4s\n"
    "fadd v17.4s, v0.4s, v31.4s\n"
    "fadd v22.4s, v30.4s, v29.4s\n"
    "fadd v16.4s, v28.4s, v27.4s\n"
    "fadd v21.4s, v18.4s, v19.4s\n"
    "fadd v20.4s, v16.4s, v19.4s\n"
    "fadd v19.4s, v26.4s, v17.4s\n"
    "fadd v18.4s, v25.4s, v22.4s\n"
    "fadd v17.4s, v24.4s, v17.4s\n"
    "fadd v16.4s, v23.4s, v22.4s\n"
    "fadd v19.4s, v21.4s, v19.4s\n"
    "fadd v18.4s, v21.4s, v18.4s\n"
    "fadd v17.4s, v17.4s, v20.4s\n"
    "fadd v16.4s, v16.4s, v20.4s\n"
    "fmul v19.4s, v19.4s, v7.s[0]\n"
    "str q19, [x6, x5]\n"
    "fmul v18.4s, v18.4s, v7.s[1]\n"
    "fmul v17.4s, v17.4s, v7.s[2]\n"
    "str q18, [x7, x5]\n"
    "fmul v16.4s, v16.4s, v7.s[3]\n"
    "str q17, [x8, x5]\n"
    "str q16, [x17, x5]\n"
    "add x5, x5, #0x10\n"
    "cbz x3, 4f\n"
    "3:"  // Oddments
    "ldr s17, [x11, x4]\n"
    "ldr s16, [x10, x4]\n"
    "fadd v18.4s, v17.4s, v16.4s\n"
    "subs x3, x3, #0x1\n"
    "ldr s17, [x27, x4]\n"
    "ldr s16, [x26, x4]\n"
    "fadd v16.4s, v17.4s, v16.4s\n"
    "fadd v18.4s, v18.4s, v16.4s\n"
    "ldr s17, [x15, x4]\n"
    "ldr s16, [x14, x4]\n"
    "fadd v16.4s, v17.4s, v16.4s\n"
    "fadd v23.4s, v16.4s, v18.4s\n"
    "ldr s17, [x12, x4]\n"
    "ldr s16, [x28, x4]\n"
    "fadd v22.4s, v17.4s, v16.4s\n"
    "ldr s17, [x9, x4]\n"
    "ldr s16, [x25, x4]\n"
    "fadd v21.4s, v17.4s, v16.4s\n"
    "ldr s17, [x23, x4]\n"
    "ldr s16, [x22, x4]\n"
    "fadd v16.4s, v17.4s, v16.4s\n"
    "fadd v20.4s, v16.4s, v18.4s\n"
    "ldr s17, [x16, x4]\n"
    "ldr s16, [x13, x4]\n"
    "fadd v19.4s, v17.4s, v22.4s\n"
    "fadd v18.4s, v16.4s, v21.4s\n"
    "ldr s17, [x24, x4]\n"
    "ldr s16, [x21, x4]\n"
    "fadd v17.4s, v17.4s, v22.4s\n"
    "fadd v16.4s, v16.4s, v21.4s\n"
    "fadd v19.4s, v23.4s, v19.4s\n"
    "fadd v18.4s, v23.4s, v18.4s\n"
    "add x4, x4, #0x4\n"
    "fadd v17.4s, v17.4s, v20.4s\n"
    "fadd v16.4s, v16.4s, v20.4s\n"
    "fmul v19.4s, v19.4s, v7.s[0]\n"
    "fmul v18.4s, v18.4s, v7.s[1]\n"
    "str s19, [x6, x5]\n"
    "fmul v17.4s, v17.4s, v7.s[2]\n"
    "fmul v16.4s, v16.4s, v7.s[3]\n"
    "str s18, [x7, x5]\n"
    "str s17, [x8, x5]\n"
    "str s16, [x17, x5]\n"
    "add x5, x5, #0x4\n"
    "bgt 3b\n"
    "4:"  // End
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs)), [offsetof_rescale] "I" (offsetof(KernelArgs, rescale_vals))
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__aarch64__)
