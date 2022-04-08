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

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace pooling {

void a64_fp16_nhwc_avg_3x3_s1_output2x2_depthfirst_impl(
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
    "ldr x4, [%x[args], %[offsetof_n_channels]]\n"
    "ldr x20, [%x[args], %[offsetof_outptrs]]\n"
    "cmp x4, #0x8\n"
    "mov x5, #0x0\n"
    "ldr x19, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x6, x7, [x20, #0x0]\n"
    "mov x8, #0x0\n"
    "ldp x17, x16, [x20, #0x10]\n"
    "ldp x15, x14, [x19, #0x0]\n"
    "ldp x13, x12, [x19, #0x10]\n"
    "ldp x11, x10, [x19, #0x20]\n"
    "ldp x9, x28, [x19, #0x30]\n"
    "ldp x27, x26, [x19, #0x40]\n"
    "ldp x25, x24, [x19, #0x50]\n"
    "ldp x23, x22, [x19, #0x60]\n"
    "ldp x21, x20, [x19, #0x70]\n"
    "ldr d7, [%x[args], %[offsetof_rescale]]\n"
    "blt 3f\n"
    "lsr x19, x4, #0x3\n"
    "sub x4, x4, x19, LSL #3\n"
    "ldr q6, [x10, x5]\n"
    "ldr q5, [x9, x5]\n"
    "subs x19, x19, #0x1\n"
    "ldr q4, [x26, x5]\n"
    "ldr q3, [x25, x5]\n"
    "ldr q2, [x14, x5]\n"
    "ldr q1, [x13, x5]\n"
    "ldr q0, [x11, x5]\n"
    "ldr q31, [x27, x5]\n"
    "ldr q30, [x28, x5]\n"
    "ldr q29, [x24, x5]\n"
    "ldr q28, [x22, x5]\n"
    "ldr q27, [x21, x5]\n"
    "ldr q26, [x15, x5]\n"
    "ldr q25, [x12, x5]\n"
    "ldr q24, [x23, x5]\n"
    "ldr q23, [x20, x5]\n"
    "add x5, x5, #0x10\n"
    "beq 2f\n"
    "1:"  // Vector: Loop
    "fadd v17.8h, v6.8h, v5.8h\n"
    "fadd v16.8h, v4.8h, v3.8h\n"
    "subs x19, x19, #0x1\n"
    "ldr q6, [x10, x5]\n"
    "fadd v19.8h, v17.8h, v16.8h\n"
    "fadd v18.8h, v2.8h, v1.8h\n"
    "ldr q5, [x9, x5]\n"
    "ldr q4, [x26, x5]\n"
    "fadd v17.8h, v0.8h, v31.8h\n"
    "fadd v22.8h, v30.8h, v29.8h\n"
    "ldr q3, [x25, x5]\n"
    "ldr q2, [x14, x5]\n"
    "fadd v16.8h, v28.8h, v27.8h\n"
    "fadd v21.8h, v18.8h, v19.8h\n"
    "ldr q1, [x13, x5]\n"
    "ldr q0, [x11, x5]\n"
    "fadd v20.8h, v16.8h, v19.8h\n"
    "fadd v19.8h, v26.8h, v17.8h\n"
    "ldr q31, [x27, x5]\n"
    "ldr q30, [x28, x5]\n"
    "fadd v18.8h, v25.8h, v22.8h\n"
    "fadd v17.8h, v24.8h, v17.8h\n"
    "ldr q29, [x24, x5]\n"
    "ldr q28, [x22, x5]\n"
    "fadd v16.8h, v23.8h, v22.8h\n"
    "fadd v19.8h, v19.8h, v21.8h\n"
    "ldr q27, [x21, x5]\n"
    "ldr q26, [x15, x5]\n"
    "fadd v18.8h, v18.8h, v21.8h\n"
    "fadd v17.8h, v17.8h, v20.8h\n"
    "ldr q25, [x12, x5]\n"
    "ldr q24, [x23, x5]\n"
    "fadd v16.8h, v16.8h, v20.8h\n"
    "fmul v19.8h, v19.8h, v7.h[0]\n"
    "ldr q23, [x20, x5]\n"
    "add x5, x5, #0x10\n"
    "fmul v18.8h, v18.8h, v7.h[1]\n"
    "fmul v17.8h, v17.8h, v7.h[2]\n"
    "str q19, [x6, x8]\n"
    "fmul v16.8h, v16.8h, v7.h[3]\n"
    "str q18, [x7, x8]\n"
    "str q17, [x17, x8]\n"
    "str q16, [x16, x8]\n"
    "add x8, x8, #0x10\n"
    "bgt 1b\n"
    "2:"  // Vector: Tail
    "fadd v17.8h, v6.8h, v5.8h\n"
    "fadd v16.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v17.8h, v16.8h\n"
    "fadd v18.8h, v2.8h, v1.8h\n"
    "fadd v17.8h, v0.8h, v31.8h\n"
    "fadd v22.8h, v30.8h, v29.8h\n"
    "fadd v16.8h, v28.8h, v27.8h\n"
    "fadd v21.8h, v18.8h, v19.8h\n"
    "fadd v20.8h, v16.8h, v19.8h\n"
    "fadd v19.8h, v26.8h, v17.8h\n"
    "fadd v18.8h, v25.8h, v22.8h\n"
    "fadd v17.8h, v24.8h, v17.8h\n"
    "fadd v16.8h, v23.8h, v22.8h\n"
    "fadd v19.8h, v19.8h, v21.8h\n"
    "fadd v18.8h, v18.8h, v21.8h\n"
    "fadd v17.8h, v17.8h, v20.8h\n"
    "fadd v16.8h, v16.8h, v20.8h\n"
    "fmul v19.8h, v19.8h, v7.h[0]\n"
    "str q19, [x6, x8]\n"
    "fmul v18.8h, v18.8h, v7.h[1]\n"
    "fmul v17.8h, v17.8h, v7.h[2]\n"
    "str q18, [x7, x8]\n"
    "fmul v16.8h, v16.8h, v7.h[3]\n"
    "str q17, [x17, x8]\n"
    "str q16, [x16, x8]\n"
    "add x8, x8, #0x10\n"
    "cbz x4, 4f\n"
    "3:"  // Oddments
    "ldr h6, [x10, x5]\n"
    "ldr h5, [x9, x5]\n"
    "fadd v17.8h, v6.8h, v5.8h\n"
    "subs x4, x4, #0x1\n"
    "ldr h4, [x26, x5]\n"
    "ldr h3, [x25, x5]\n"
    "fadd v16.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v17.8h, v16.8h\n"
    "ldr h2, [x14, x5]\n"
    "ldr h1, [x13, x5]\n"
    "fadd v18.8h, v2.8h, v1.8h\n"
    "fadd v21.8h, v18.8h, v19.8h\n"
    "ldr h0, [x11, x5]\n"
    "ldr h31, [x27, x5]\n"
    "fadd v17.8h, v0.8h, v31.8h\n"
    "ldr h30, [x28, x5]\n"
    "ldr h29, [x24, x5]\n"
    "fadd v22.8h, v30.8h, v29.8h\n"
    "ldr h28, [x22, x5]\n"
    "ldr h27, [x21, x5]\n"
    "fadd v16.8h, v28.8h, v27.8h\n"
    "fadd v20.8h, v16.8h, v19.8h\n"
    "ldr h26, [x15, x5]\n"
    "ldr h25, [x12, x5]\n"
    "fadd v19.8h, v26.8h, v17.8h\n"
    "fadd v18.8h, v25.8h, v22.8h\n"
    "ldr h24, [x23, x5]\n"
    "ldr h23, [x20, x5]\n"
    "fadd v17.8h, v24.8h, v17.8h\n"
    "fadd v16.8h, v23.8h, v22.8h\n"
    "fadd v19.8h, v19.8h, v21.8h\n"
    "fadd v18.8h, v18.8h, v21.8h\n"
    "add x5, x5, #0x2\n"
    "fadd v17.8h, v17.8h, v20.8h\n"
    "fadd v16.8h, v16.8h, v20.8h\n"
    "fmul v19.8h, v19.8h, v7.h[0]\n"
    "fmul v18.8h, v18.8h, v7.h[1]\n"
    "str h19, [x6, x8]\n"
    "fmul v17.8h, v17.8h, v7.h[2]\n"
    "fmul v16.8h, v16.8h, v7.h[3]\n"
    "str h18, [x7, x8]\n"
    "str h17, [x17, x8]\n"
    "str h16, [x16, x8]\n"
    "add x8, x8, #0x2\n"
    "bgt 3b\n"
    "4:"  // End
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs)), [offsetof_rescale] "I" (offsetof(KernelArgs, rescale_vals))
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
