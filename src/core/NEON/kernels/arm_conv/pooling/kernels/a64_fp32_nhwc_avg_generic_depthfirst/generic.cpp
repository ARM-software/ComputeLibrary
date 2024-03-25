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

#include <cstdint>
#include <cstddef>

#if defined(__aarch64__)

namespace arm_conv {
namespace pooling {


void a64_fp32_nhwc_avg_generic_depthfirst_impl(
  const uint64_t window_cells,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const float *const *const inptrs,
  float *outptr
)
{
  const auto rescale_value = static_cast<float>(1.0f / static_cast<float>(window_cells));

  __asm__ __volatile__(
    "ld1r { v9.4s }, [%x[rescale_ptr]]\n"
    "cmp %x[n_channels], #0x10\n"
    "mov x27, #0x0\n"
    "mov x26, #0x10\n"  // cntb _, ALL, #1
    "mov x24, #0x20\n"  // cntb _, ALL, #2
    "mov x23, #0x30\n"  // cntb _, ALL, #3
    "blt 7f\n"
    "1:"  // 4-vectors of channels
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "movi v8.16b, #0x0\n"
    "movi v7.16b, #0x0\n"
    "mov x22, %x[inptrs]\n"
    "movi v6.16b, #0x0\n"
    "movi v5.16b, #0x0\n"
    "cbz x25, 4f\n"
    "ldp x21, x20, [x22, #0x0]\n"
    "ldr q4, [x21, x27]\n"
    "subs x25, x25, #0x1\n"
    "ldr q3, [x20, x27]\n"
    "ldr q2, [x21, x26]\n"
    "ldr q1, [x20, x26]\n"
    "ldr q0, [x21, x24]\n"
    "ldr q31, [x20, x24]\n"
    "ldr q30, [x21, x23]\n"
    "ldr q29, [x20, x23]\n"
    "ldp x21, x20, [x22, #0x10]\n"
    "add x22, x22, #0x20\n"
    "ldr q28, [x21, x27]\n"
    "ldr q22, [x20, x27]\n"
    "ldr q27, [x21, x26]\n"
    "ldr q21, [x20, x26]\n"
    "ldr q26, [x21, x24]\n"
    "ldr q20, [x20, x24]\n"
    "ldr q25, [x21, x23]\n"
    "ldr q24, [x20, x23]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 4 inputs loop
    "fadd v23.4s, v4.4s, v3.4s\n"
    "fadd v19.4s, v28.4s, v22.4s\n"
    "ldp x21, x20, [x22, #0x0]\n"
    "ldr q4, [x21, x27]\n"
    "ldr q3, [x20, x27]\n"
    "fadd v22.4s, v2.4s, v1.4s\n"
    "ldr q2, [x21, x26]\n"
    "fadd v18.4s, v27.4s, v21.4s\n"
    "ldr q1, [x20, x26]\n"
    "fadd v21.4s, v0.4s, v31.4s\n"
    "ldr q0, [x21, x24]\n"
    "fadd v17.4s, v26.4s, v20.4s\n"
    "ldr q31, [x20, x24]\n"
    "fadd v20.4s, v30.4s, v29.4s\n"
    "ldr q30, [x21, x23]\n"
    "fadd v16.4s, v25.4s, v24.4s\n"
    "ldr q29, [x20, x23]\n"
    "fadd v19.4s, v23.4s, v19.4s\n"
    "fadd v18.4s, v22.4s, v18.4s\n"
    "ldp x21, x20, [x22, #0x10]\n"
    "ldr q28, [x21, x27]\n"
    "ldr q22, [x20, x27]\n"
    "fadd v17.4s, v21.4s, v17.4s\n"
    "fadd v16.4s, v20.4s, v16.4s\n"
    "ldr q27, [x21, x26]\n"
    "ldr q21, [x20, x26]\n"
    "subs x25, x25, #0x1\n"
    "fadd v8.4s, v8.4s, v19.4s\n"
    "ldr q26, [x21, x24]\n"
    "ldr q20, [x20, x24]\n"
    "fadd v7.4s, v7.4s, v18.4s\n"
    "fadd v6.4s, v6.4s, v17.4s\n"
    "ldr q25, [x21, x23]\n"
    "ldr q24, [x20, x23]\n"
    "fadd v5.4s, v5.4s, v16.4s\n"
    "add x22, x22, #0x20\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "fadd v23.4s, v4.4s, v3.4s\n"
    "fadd v19.4s, v28.4s, v22.4s\n"
    "fadd v22.4s, v2.4s, v1.4s\n"
    "fadd v18.4s, v27.4s, v21.4s\n"
    "fadd v21.4s, v0.4s, v31.4s\n"
    "fadd v17.4s, v26.4s, v20.4s\n"
    "fadd v20.4s, v30.4s, v29.4s\n"
    "fadd v16.4s, v25.4s, v24.4s\n"
    "fadd v19.4s, v23.4s, v19.4s\n"
    "fadd v18.4s, v22.4s, v18.4s\n"
    "fadd v17.4s, v21.4s, v17.4s\n"
    "fadd v16.4s, v20.4s, v16.4s\n"
    "fadd v8.4s, v8.4s, v19.4s\n"
    "fadd v7.4s, v7.4s, v18.4s\n"
    "fadd v6.4s, v6.4s, v17.4s\n"
    "fadd v5.4s, v5.4s, v16.4s\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x20, [x22], #0x8\n"
    "ldr q16, [x20, x27]\n"
    "subs x21, x21, #0x1\n"
    "fadd v8.4s, v8.4s, v16.4s\n"
    "ldr q17, [x20, x26]\n"
    "ldr q16, [x20, x24]\n"
    "fadd v7.4s, v7.4s, v17.4s\n"
    "fadd v6.4s, v6.4s, v16.4s\n"
    "ldr q16, [x20, x23]\n"
    "fadd v5.4s, v5.4s, v16.4s\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x10\n"
    "cmp %x[n_channels], #0x10\n"
    "fmul v8.4s, v8.4s, v9.4s\n"
    "fmul v7.4s, v7.4s, v9.4s\n"
    "fmul v6.4s, v6.4s, v9.4s\n"
    "fmul v5.4s, v5.4s, v9.4s\n"
    "str q8, [%x[outptr], x27]\n"
    "add x27, x27, #0x40\n"
    "str q7, [%x[outptr], x26]\n"
    "add x26, x26, #0x40\n"
    "str q6, [%x[outptr], x24]\n"
    "add x24, x24, #0x40\n"
    "str q5, [%x[outptr], x23]\n"
    "add x23, x23, #0x40\n"
    "bge 1b\n"
    "cbz %x[n_channels], 25f\n"
    "7:"  // Single vector of channels
    "cmp %x[n_channels], #0x4\n"
    "blt 14f\n"
    "8:"  // Single vector of channels: Loop
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "movi v8.16b, #0x0\n"
    "mov x22, %x[inptrs]\n"
    "cbz x25, 11f\n"
    "ldp x21, x20, [x22, #0x0]\n"
    "ldr q4, [x21, x27]\n"
    "subs x25, x25, #0x1\n"
    "ldr q3, [x20, x27]\n"
    "ldp x21, x20, [x22, #0x10]\n"
    "add x22, x22, #0x20\n"
    "ldr q28, [x21, x27]\n"
    "ldr q22, [x20, x27]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 4 inputs loop
    "fadd v17.4s, v4.4s, v3.4s\n"
    "fadd v16.4s, v28.4s, v22.4s\n"
    "ldp x21, x20, [x22, #0x0]\n"
    "ldr q4, [x21, x27]\n"
    "ldr q3, [x20, x27]\n"
    "fadd v16.4s, v17.4s, v16.4s\n"
    "ldp x21, x20, [x22, #0x10]\n"
    "subs x25, x25, #0x1\n"
    "ldr q28, [x21, x27]\n"
    "ldr q22, [x20, x27]\n"
    "fadd v8.4s, v8.4s, v16.4s\n"
    "add x22, x22, #0x20\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "fadd v17.4s, v4.4s, v3.4s\n"
    "fadd v16.4s, v28.4s, v22.4s\n"
    "fadd v16.4s, v17.4s, v16.4s\n"
    "fadd v8.4s, v8.4s, v16.4s\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x20, [x22], #0x8\n"
    "ldr q16, [x20, x27]\n"
    "subs x21, x21, #0x1\n"
    "fadd v8.4s, v8.4s, v16.4s\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x4\n"
    "cmp %x[n_channels], #0x4\n"
    "fmul v8.4s, v8.4s, v9.4s\n"
    "str q8, [%x[outptr], x27]\n"
    "add x27, x27, #0x10\n"
    "bge 8b\n"
    "cbz %x[n_channels], 25f\n"
    "14:"  // Oddments
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "add %x[outptr], %x[outptr], x27\n"
    "movi v8.16b, #0x0\n"
    "mov x24, %x[inptrs]\n"
    "cbz x25, 18f\n"
    "15:"  // Oddments: 4 inputs loop
    "ldp x23, x22, [x24, #0x0]\n"
    "ldp x21, x20, [x24, #0x10]\n"
    "add x24, x24, #0x20\n"
    "add x23, x23, x27\n"
    "add x22, x22, x27\n"
    "add x21, x21, x27\n"
    "movi v4.16b, #0x0\n"
    "movi v3.16b, #0x0\n"
    "add x20, x20, x27\n"
    "movi v28.16b, #0x0\n"
    "movi v22.16b, #0x0\n"
    "tbz %x[n_channels], #1, 16f\n"
    "ldr d4, [x23], #0x8\n"
    "ldr d3, [x22], #0x8\n"
    "ldr d28, [x21], #0x8\n"
    "ldr d22, [x20], #0x8\n"
    "tbz %x[n_channels], #0, 17f\n"
    "ld1 { v4.s }[2], [x23], #0x4\n"
    "ld1 { v3.s }[2], [x22], #0x4\n"
    "ld1 { v28.s }[2], [x21], #0x4\n"
    "ld1 { v22.s }[2], [x20], #0x4\n"
    "b 17f\n"
    "16:"  // Oddments: 4 inputs loop: Load: Bit 1: Unset
    "tbz %x[n_channels], #0, 17f\n"
    "ldr s4, [x23], #0x4\n"
    "ldr s3, [x22], #0x4\n"
    "ldr s28, [x21], #0x4\n"
    "ldr s22, [x20], #0x4\n"
    "17:"  // Oddments: 4 inputs loop: Load: Bit 1: End
    "fadd v17.4s, v4.4s, v3.4s\n"
    "fadd v16.4s, v28.4s, v22.4s\n"
    "subs x25, x25, #0x1\n"
    "fadd v16.4s, v17.4s, v16.4s\n"
    "fadd v8.4s, v8.4s, v16.4s\n"
    "bgt 15b\n"
    "18:"  // Oddments: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 22f\n"
    "19:"  // Oddments: Single input loop
    "ldr x23, [x24], #0x8\n"
    "add x23, x23, x27\n"
    "movi v4.16b, #0x0\n"
    "tbz %x[n_channels], #1, 20f\n"
    "ldr d4, [x23], #0x8\n"
    "tbz %x[n_channels], #0, 21f\n"
    "ld1 { v4.s }[2], [x23], #0x4\n"
    "b 21f\n"
    "20:"  // Oddments: Single input loop: Load: Bit 1: Unset
    "tbz %x[n_channels], #0, 21f\n"
    "ldr s4, [x23], #0x4\n"
    "21:"  // Oddments: Single input loop: Load: Bit 1: End
    "subs x21, x21, #0x1\n"
    "fadd v8.4s, v8.4s, v4.4s\n"
    "bgt 19b\n"
    "22:"  // Oddments: Single input loop: End
    "fmul v8.4s, v8.4s, v9.4s\n"
    "tbz %x[n_channels], #1, 23f\n"
    "st1 { v8.d }[0], [%x[outptr]], #0x8\n"
    "tbz %x[n_channels], #0, 24f\n"
    "st1 { v8.s }[2], [%x[outptr]], #0x4\n"
    "b 24f\n"
    "23:"  // Oddments: Store: Bit 1: Unset
    "tbz %x[n_channels], #0, 24f\n"
    "st1 { v8.s }[0], [%x[outptr]], #0x4\n"
    "24:"  // Oddments: Store: Bit 1: End
    "25:"  // End
    : [n_channels] "+&r" (n_channels), [outptr] "+&r" (outptr)
    : [inptrs] "r" (inptrs), [n_valid_cells] "r" (n_valid_cells), [rescale_ptr] "r" (&rescale_value)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__aarch64__)
