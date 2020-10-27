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


#include <cstddef>
#include <cstdint>

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
    "ldr x15, [%x[args], %[offsetof_n_channels]]\n"
    "mov x14, #0x0\n"
    "ldr x19, [%x[args], %[offsetof_outptrs]]\n"
    "mov x13, #0x0\n"
    "ldp x12, x11, [x19, #0x0]\n"
    "cmp x15, #0x4\n"
    "ldp x10, x9, [x19, #0x10]\n"
    "ldr x19, [%x[args], %[offsetof_inptrs]]\n"
    "ldp x28, x27, [x19, #0x0]\n"
    "ldp x26, x25, [x19, #0x10]\n"
    "ldp x24, x23, [x19, #0x20]\n"
    "ldp x22, x21, [x19, #0x30]\n"
    "ldr x20, [x19, #0x40]\n"
    "blt 3f\n"
    "lsr x19, x15, #0x2\n"
    "sub x15, x15, x19, LSL #2\n"
    "ldr q30, [x27, x14]\n"
    "ldr q29, [x24, x14]\n"
    "ldr q28, [x21, x14]\n"
    "ldr q27, [x25, x14]\n"
    "ldr q26, [x28, x14]\n"
    "ldr q25, [x23, x14]\n"
    "ldr q24, [x26, x14]\n"
    "ldr q23, [x22, x14]\n"
    "ldr q22, [x20, x14]\n"
    "add x14, x14, #0x10\n"
    "subs x19, x19, #0x1\n"
    "beq 2f\n"
    "1:"  // Vector: Loop
    "fmax v21.4s, v30.4s, v29.4s\n"
    "ldr q30, [x27, x14]\n"
    "fmax v20.4s, v29.4s, v28.4s\n"
    "ldr q29, [x24, x14]\n"
    "fmax v19.4s, v27.4s, v26.4s\n"
    "ldr q28, [x21, x14]\n"
    "fmax v18.4s, v25.4s, v24.4s\n"
    "ldr q26, [x28, x14]\n"
    "fmax v17.4s, v23.4s, v27.4s\n"
    "ldr q27, [x25, x14]\n"
    "fmax v16.4s, v25.4s, v22.4s\n"
    "ldr q25, [x23, x14]\n"
    "fmax v19.4s, v21.4s, v19.4s\n"
    "ldr q24, [x26, x14]\n"
    "fmax v18.4s, v21.4s, v18.4s\n"
    "ldr q23, [x22, x14]\n"
    "fmax v17.4s, v20.4s, v17.4s\n"
    "ldr q22, [x20, x14]\n"
    "fmax v16.4s, v20.4s, v16.4s\n"
    "add x14, x14, #0x10\n"
    "str q19, [x12, x13]\n"
    "str q18, [x11, x13]\n"
    "str q17, [x10, x13]\n"
    "str q16, [x9, x13]\n"
    "add x13, x13, #0x10\n"
    "subs x19, x19, #0x1\n"
    "bgt 1b\n"
    "2:"  // Vector: Tail
    "fmax v21.4s, v30.4s, v29.4s\n"
    "fmax v20.4s, v29.4s, v28.4s\n"
    "fmax v19.4s, v27.4s, v26.4s\n"
    "fmax v18.4s, v25.4s, v24.4s\n"
    "fmax v17.4s, v23.4s, v27.4s\n"
    "fmax v16.4s, v25.4s, v22.4s\n"
    "fmax v19.4s, v21.4s, v19.4s\n"
    "str q19, [x12, x13]\n"
    "fmax v18.4s, v21.4s, v18.4s\n"
    "fmax v17.4s, v20.4s, v17.4s\n"
    "str q18, [x11, x13]\n"
    "fmax v16.4s, v20.4s, v16.4s\n"
    "str q17, [x10, x13]\n"
    "str q16, [x9, x13]\n"
    "add x13, x13, #0x10\n"
    "cbz x15, 4f\n"
    "3:"  // Oddments
    "ldr s30, [x27, x14]\n"
    "ldr s29, [x24, x14]\n"
    "fmax v21.4s, v30.4s, v29.4s\n"
    "ldr s28, [x21, x14]\n"
    "ldr s27, [x25, x14]\n"
    "fmax v20.4s, v29.4s, v28.4s\n"
    "ldr s26, [x28, x14]\n"
    "ldr s25, [x23, x14]\n"
    "fmax v19.4s, v27.4s, v26.4s\n"
    "ldr s24, [x26, x14]\n"
    "ldr s23, [x22, x14]\n"
    "fmax v19.4s, v21.4s, v19.4s\n"
    "ldr s22, [x20, x14]\n"
    "add x14, x14, #0x4\n"
    "fmax v18.4s, v25.4s, v24.4s\n"
    "subs x15, x15, #0x1\n"
    "fmax v17.4s, v23.4s, v27.4s\n"
    "str s19, [x12, x13]\n"
    "fmax v16.4s, v25.4s, v22.4s\n"
    "fmax v18.4s, v21.4s, v18.4s\n"
    "str s18, [x11, x13]\n"
    "fmax v17.4s, v20.4s, v17.4s\n"
    "fmax v16.4s, v20.4s, v16.4s\n"
    "str s17, [x10, x13]\n"
    "str s16, [x9, x13]\n"
    "add x13, x13, #0x4\n"
    "bgt 3b\n"
    "4:"  // End

    :
    : [args] "r" (&args), [offsetof_inptrs] "I" (offsetof(KernelArgs, inptrs)), [offsetof_n_channels] "I" (offsetof(KernelArgs, n_channels)), [offsetof_outptrs] "I" (offsetof(KernelArgs, outptrs))
    : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv
