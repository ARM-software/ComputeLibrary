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

#if defined(__aarch64__)

namespace arm_conv {
namespace pooling {

void a64_fp32_nhwc_max_2x2_s1_output2x2_depthfirst_impl(
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
    KernelArgs(
      unsigned int channels,
      const float *const *input_ptrs,
      float *const * output_ptrs,
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
    "ldr x16, [%x[args], %[offsetof_n_channels]]\n"
    "ldr x21, [%x[args], %[offsetof_outptrs]]\n"
    "cmp x16, #0x4\n"
    "mov x15, #0x0\n"
    "ldr x20, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x14, x13, [x21, #0x0]\n"
    "mov x12, #0x0\n"
    "ldp x11, x10, [x21, #0x10]\n"
    "ldp x9, x28, [x20, #0x0]\n"
    "ldp x27, x26, [x20, #0x10]\n"
    "ldp x25, x24, [x20, #0x20]\n"
    "ldp x23, x22, [x20, #0x30]\n"
    "ldr x21, [x20, #0x40]\n"
    "blt 3f\n"
    "ldr q30, [x28, x15]\n"
    "ldr q29, [x25, x15]\n"
    "lsr x20, x16, #0x2\n"
    "sub x16, x16, x20, LSL #2\n"
    "ldr q28, [x22, x15]\n"
    "ldr q27, [x26, x15]\n"
    "subs x20, x20, #0x1\n"
    "ldr q26, [x9, x15]\n"
    "ldr q25, [x27, x15]\n"
    "ldr q24, [x24, x15]\n"
    "ldr q23, [x23, x15]\n"
    "ldr q22, [x21, x15]\n"
    "add x15, x15, #0x10\n"
    "beq 2f\n"
    "1:"  // Vector: Loop
    "fmax v21.4s, v30.4s, v29.4s\n"
    "ldr q30, [x28, x15]\n"
    "fmax v20.4s, v29.4s, v28.4s\n"
    "ldr q29, [x25, x15]\n"
    "ldr q28, [x22, x15]\n"
    "fmax v19.4s, v27.4s, v26.4s\n"
    "ldr q26, [x9, x15]\n"
    "fmax v18.4s, v25.4s, v24.4s\n"
    "ldr q25, [x27, x15]\n"
    "fmax v17.4s, v27.4s, v23.4s\n"
    "ldr q27, [x26, x15]\n"
    "fmax v16.4s, v24.4s, v22.4s\n"
    "ldr q24, [x24, x15]\n"
    "ldr q23, [x23, x15]\n"
    "subs x20, x20, #0x1\n"
    "fmax v19.4s, v21.4s, v19.4s\n"
    "ldr q22, [x21, x15]\n"
    "fmax v18.4s, v18.4s, v21.4s\n"
    "fmax v17.4s, v17.4s, v20.4s\n"
    "add x15, x15, #0x10\n"
    "fmax v16.4s, v20.4s, v16.4s\n"
    "str q19, [x14, x12]\n"
    "str q18, [x13, x12]\n"
    "str q17, [x11, x12]\n"
    "str q16, [x10, x12]\n"
    "add x12, x12, #0x10\n"
    "bgt 1b\n"
    "2:"  // Vector: Tail
    "fmax v21.4s, v30.4s, v29.4s\n"
    "fmax v20.4s, v29.4s, v28.4s\n"
    "fmax v16.4s, v27.4s, v26.4s\n"
    "fmax v18.4s, v25.4s, v24.4s\n"
    "fmax v17.4s, v27.4s, v23.4s\n"
    "fmax v19.4s, v24.4s, v22.4s\n"
    "fmax v16.4s, v21.4s, v16.4s\n"
    "fmax v18.4s, v18.4s, v21.4s\n"
    "str q16, [x14, x12]\n"
    "fmax v17.4s, v17.4s, v20.4s\n"
    "fmax v16.4s, v20.4s, v19.4s\n"
    "str q18, [x13, x12]\n"
    "str q17, [x11, x12]\n"
    "str q16, [x10, x12]\n"
    "add x12, x12, #0x10\n"
    "cbz x16, 4f\n"
    "3:"  // Oddments
    "ldr s16, [x28, x15]\n"
    "ldr s17, [x25, x15]\n"
    "fmax v23.4s, v16.4s, v17.4s\n"
    "subs x16, x16, #0x1\n"
    "ldr s16, [x22, x15]\n"
    "ldr s22, [x26, x15]\n"
    "fmax v21.4s, v17.4s, v16.4s\n"
    "ldr s16, [x9, x15]\n"
    "ldr s17, [x27, x15]\n"
    "fmax v16.4s, v22.4s, v16.4s\n"
    "fmax v20.4s, v23.4s, v16.4s\n"
    "ldr s19, [x24, x15]\n"
    "ldr s16, [x23, x15]\n"
    "fmax v18.4s, v17.4s, v19.4s\n"
    "fmax v17.4s, v22.4s, v16.4s\n"
    "ldr s16, [x21, x15]\n"
    "fmax v16.4s, v19.4s, v16.4s\n"
    "add x15, x15, #0x4\n"
    "fmax v18.4s, v18.4s, v23.4s\n"
    "fmax v17.4s, v17.4s, v21.4s\n"
    "fmax v16.4s, v21.4s, v16.4s\n"
    "str s20, [x14, x12]\n"
    "str s18, [x13, x12]\n"
    "str s17, [x11, x12]\n"
    "str s16, [x10, x12]\n"
    "add x12, x12, #0x4\n"
    "bgt 3b\n"
    "4:"  // End
    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs))
    : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__aarch64__)
