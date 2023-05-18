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

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace pooling {


void a64_fp16_nhwc_avg_generic_depthfirst_impl(
  const uint64_t window_cells,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const __fp16 *const *const inptrs,
  __fp16 *outptr
)
{
  const auto rescale_value = static_cast<__fp16>(1.0f / static_cast<float>(window_cells));

  __asm__ __volatile__(
    "ld1r { v9.8h }, [%x[rescale_ptr]]\n"
    "cmp %x[n_channels], #0x20\n"
    "mov x9, #0x0\n"
    "mov x28, #0x10\n"  // cntb _, ALL, #1
    "mov x27, #0x20\n"  // cntb _, ALL, #2
    "mov x26, #0x30\n"  // cntb _, ALL, #3
    "blt 7f\n"
    "1:"  // 4-vectors of channels
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "movi v8.16b, #0x0\n"
    "movi v7.16b, #0x0\n"
    "mov x20, %x[inptrs]\n"
    "movi v6.16b, #0x0\n"
    "movi v5.16b, #0x0\n"
    "cbz x25, 4f\n"
    "ldp x24, x23, [x20, #0x0]\n"
    "ldr q4, [x24, x9]\n"
    "subs x25, x25, #0x1\n"
    "ldr q3, [x23, x9]\n"
    "ldr q2, [x24, x28]\n"
    "ldr q1, [x23, x28]\n"
    "ldr q0, [x24, x27]\n"
    "ldr q31, [x23, x27]\n"
    "ldr q30, [x24, x26]\n"
    "ldr q29, [x23, x26]\n"
    "ldp x22, x21, [x20, #0x10]\n"
    "add x20, x20, #0x20\n"
    "ldr q28, [x22, x9]\n"
    "ldr q22, [x21, x9]\n"
    "ldr q27, [x22, x28]\n"
    "ldr q21, [x21, x28]\n"
    "ldr q26, [x22, x27]\n"
    "ldr q20, [x21, x27]\n"
    "ldr q25, [x22, x26]\n"
    "ldr q24, [x21, x26]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 4 inputs loop
    "fadd v23.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v28.8h, v22.8h\n"
    "ldp x24, x23, [x20, #0x0]\n"
    "ldr q4, [x24, x9]\n"
    "ldr q3, [x23, x9]\n"
    "fadd v22.8h, v2.8h, v1.8h\n"
    "ldr q2, [x24, x28]\n"
    "fadd v18.8h, v27.8h, v21.8h\n"
    "ldr q1, [x23, x28]\n"
    "fadd v21.8h, v0.8h, v31.8h\n"
    "ldr q0, [x24, x27]\n"
    "fadd v17.8h, v26.8h, v20.8h\n"
    "ldr q31, [x23, x27]\n"
    "fadd v20.8h, v30.8h, v29.8h\n"
    "ldr q30, [x24, x26]\n"
    "fadd v16.8h, v25.8h, v24.8h\n"
    "ldr q29, [x23, x26]\n"
    "fadd v19.8h, v23.8h, v19.8h\n"
    "fadd v18.8h, v22.8h, v18.8h\n"
    "ldp x22, x21, [x20, #0x10]\n"
    "ldr q28, [x22, x9]\n"
    "ldr q22, [x21, x9]\n"
    "fadd v17.8h, v21.8h, v17.8h\n"
    "fadd v16.8h, v20.8h, v16.8h\n"
    "ldr q27, [x22, x28]\n"
    "ldr q21, [x21, x28]\n"
    "subs x25, x25, #0x1\n"
    "fadd v8.8h, v8.8h, v19.8h\n"
    "ldr q26, [x22, x27]\n"
    "ldr q20, [x21, x27]\n"
    "fadd v7.8h, v7.8h, v18.8h\n"
    "fadd v6.8h, v6.8h, v17.8h\n"
    "ldr q25, [x22, x26]\n"
    "ldr q24, [x21, x26]\n"
    "fadd v5.8h, v5.8h, v16.8h\n"
    "add x20, x20, #0x20\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "fadd v23.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v28.8h, v22.8h\n"
    "fadd v22.8h, v2.8h, v1.8h\n"
    "fadd v18.8h, v27.8h, v21.8h\n"
    "fadd v21.8h, v0.8h, v31.8h\n"
    "fadd v17.8h, v26.8h, v20.8h\n"
    "fadd v20.8h, v30.8h, v29.8h\n"
    "fadd v16.8h, v25.8h, v24.8h\n"
    "fadd v19.8h, v23.8h, v19.8h\n"
    "fadd v18.8h, v22.8h, v18.8h\n"
    "fadd v17.8h, v21.8h, v17.8h\n"
    "fadd v16.8h, v20.8h, v16.8h\n"
    "fadd v8.8h, v8.8h, v19.8h\n"
    "fadd v7.8h, v7.8h, v18.8h\n"
    "fadd v6.8h, v6.8h, v17.8h\n"
    "fadd v5.8h, v5.8h, v16.8h\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x24, [x20], #0x8\n"
    "ldr q4, [x24, x9]\n"
    "subs x21, x21, #0x1\n"
    "fadd v8.8h, v8.8h, v4.8h\n"
    "ldr q2, [x24, x28]\n"
    "ldr q0, [x24, x27]\n"
    "fadd v7.8h, v7.8h, v2.8h\n"
    "fadd v6.8h, v6.8h, v0.8h\n"
    "ldr q30, [x24, x26]\n"
    "fadd v5.8h, v5.8h, v30.8h\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x20\n"
    "cmp %x[n_channels], #0x20\n"
    "fmul v8.8h, v8.8h, v9.8h\n"
    "fmul v7.8h, v7.8h, v9.8h\n"
    "fmul v6.8h, v6.8h, v9.8h\n"
    "fmul v5.8h, v5.8h, v9.8h\n"
    "str q8, [%x[outptr], x9]\n"
    "add x9, x9, #0x40\n"
    "str q7, [%x[outptr], x28]\n"
    "add x28, x28, #0x40\n"
    "str q6, [%x[outptr], x27]\n"
    "add x27, x27, #0x40\n"
    "str q5, [%x[outptr], x26]\n"
    "add x26, x26, #0x40\n"
    "bge 1b\n"
    "cbz %x[n_channels], 31f\n"
    "7:"  // Single vector of channels
    "cmp %x[n_channels], #0x8\n"
    "blt 14f\n"
    "8:"  // Single vector of channels: Loop
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "movi v8.16b, #0x0\n"
    "mov x20, %x[inptrs]\n"
    "cbz x25, 11f\n"
    "ldp x24, x23, [x20, #0x0]\n"
    "ldr q4, [x24, x9]\n"
    "subs x25, x25, #0x1\n"
    "ldr q3, [x23, x9]\n"
    "ldp x22, x21, [x20, #0x10]\n"
    "add x20, x20, #0x20\n"
    "ldr q28, [x22, x9]\n"
    "ldr q22, [x21, x9]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 4 inputs loop
    "fadd v23.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v28.8h, v22.8h\n"
    "ldp x24, x23, [x20, #0x0]\n"
    "ldr q4, [x24, x9]\n"
    "ldr q3, [x23, x9]\n"
    "fadd v19.8h, v23.8h, v19.8h\n"
    "ldp x22, x21, [x20, #0x10]\n"
    "subs x25, x25, #0x1\n"
    "ldr q28, [x22, x9]\n"
    "ldr q22, [x21, x9]\n"
    "fadd v8.8h, v8.8h, v19.8h\n"
    "add x20, x20, #0x20\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "fadd v23.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v28.8h, v22.8h\n"
    "fadd v19.8h, v23.8h, v19.8h\n"
    "fadd v8.8h, v8.8h, v19.8h\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x24, [x20], #0x8\n"
    "ldr q4, [x24, x9]\n"
    "subs x21, x21, #0x1\n"
    "fadd v8.8h, v8.8h, v4.8h\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x8\n"
    "cmp %x[n_channels], #0x8\n"
    "fmul v8.8h, v8.8h, v9.8h\n"
    "str q8, [%x[outptr], x9]\n"
    "add x9, x9, #0x10\n"
    "bge 8b\n"
    "cbz %x[n_channels], 31f\n"
    "14:"  // Oddments
    "lsr x25, %x[n_valid_cells], #0x2\n"
    "add %x[outptr], %x[outptr], x9\n"
    "movi v8.16b, #0x0\n"
    "mov x20, %x[inptrs]\n"
    "cbz x25, 20f\n"
    "15:"  // Oddments: 4 inputs loop
    "ldp x24, x23, [x20, #0x0]\n"
    "ldp x22, x21, [x20, #0x10]\n"
    "add x20, x20, #0x20\n"
    "add x24, x24, x9\n"
    "add x23, x23, x9\n"
    "add x22, x22, x9\n"
    "movi v4.16b, #0x0\n"
    "movi v3.16b, #0x0\n"
    "add x21, x21, x9\n"
    "movi v28.16b, #0x0\n"
    "movi v22.16b, #0x0\n"
    "tbz %x[n_channels], #2, 17f\n"
    "ldr d4, [x24], #0x8\n"
    "ldr d3, [x23], #0x8\n"
    "ldr d28, [x22], #0x8\n"
    "ldr d22, [x21], #0x8\n"
    "tbz %x[n_channels], #1, 16f\n"
    "ld1 { v4.s }[2], [x24], #0x4\n"
    "ld1 { v3.s }[2], [x23], #0x4\n"
    "ld1 { v28.s }[2], [x22], #0x4\n"
    "ld1 { v22.s }[2], [x21], #0x4\n"
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v4.h }[6], [x24], #0x2\n"
    "ld1 { v3.h }[6], [x23], #0x2\n"
    "ld1 { v28.h }[6], [x22], #0x2\n"
    "ld1 { v22.h }[6], [x21], #0x2\n"
    "b 19f\n"
    "16:"  // Oddments: 4 inputs loop: Load: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v4.h }[4], [x24], #0x2\n"
    "ld1 { v3.h }[4], [x23], #0x2\n"
    "ld1 { v28.h }[4], [x22], #0x2\n"
    "ld1 { v22.h }[4], [x21], #0x2\n"
    "b 19f\n"
    "17:"  // Oddments: 4 inputs loop: Load: Bit 2: Unset
    "tbz %x[n_channels], #1, 18f\n"
    "ldr s4, [x24], #0x4\n"
    "ldr s3, [x23], #0x4\n"
    "ldr s28, [x22], #0x4\n"
    "ldr s22, [x21], #0x4\n"
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v4.h }[2], [x24], #0x2\n"
    "ld1 { v3.h }[2], [x23], #0x2\n"
    "ld1 { v28.h }[2], [x22], #0x2\n"
    "ld1 { v22.h }[2], [x21], #0x2\n"
    "b 19f\n"
    "18:"  // Oddments: 4 inputs loop: Load: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 19f\n"
    "ldr h4, [x24], #0x2\n"
    "ldr h3, [x23], #0x2\n"
    "ldr h28, [x22], #0x2\n"
    "ldr h22, [x21], #0x2\n"
    "19:"  // Oddments: 4 inputs loop: Load: Bit 2: End
    "fadd v23.8h, v4.8h, v3.8h\n"
    "fadd v19.8h, v28.8h, v22.8h\n"
    "subs x25, x25, #0x1\n"
    "fadd v19.8h, v23.8h, v19.8h\n"
    "fadd v8.8h, v8.8h, v19.8h\n"
    "bgt 15b\n"
    "20:"  // Oddments: After loop
    "ands x21, %x[n_valid_cells], #0x3\n"
    "beq 26f\n"
    "21:"  // Oddments: Single input loop
    "ldr x24, [x20], #0x8\n"
    "add x24, x24, x9\n"
    "movi v4.16b, #0x0\n"
    "tbz %x[n_channels], #2, 23f\n"
    "ldr d4, [x24], #0x8\n"
    "tbz %x[n_channels], #1, 22f\n"
    "ld1 { v4.s }[2], [x24], #0x4\n"
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v4.h }[6], [x24], #0x2\n"
    "b 25f\n"
    "22:"  // Oddments: Single input loop: Load: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v4.h }[4], [x24], #0x2\n"
    "b 25f\n"
    "23:"  // Oddments: Single input loop: Load: Bit 2: Unset
    "tbz %x[n_channels], #1, 24f\n"
    "ldr s4, [x24], #0x4\n"
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v4.h }[2], [x24], #0x2\n"
    "b 25f\n"
    "24:"  // Oddments: Single input loop: Load: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 25f\n"
    "ldr h4, [x24], #0x2\n"
    "25:"  // Oddments: Single input loop: Load: Bit 2: End
    "subs x21, x21, #0x1\n"
    "fadd v8.8h, v8.8h, v4.8h\n"
    "bgt 21b\n"
    "26:"  // Oddments: Single input loop: End
    "fmul v8.8h, v8.8h, v9.8h\n"
    "tbz %x[n_channels], #2, 28f\n"
    "st1 { v8.d }[0], [%x[outptr]], #0x8\n"
    "tbz %x[n_channels], #1, 27f\n"
    "st1 { v8.s }[2], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v8.h }[6], [%x[outptr]], #0x2\n"
    "b 30f\n"
    "27:"  // Oddments: Store: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v8.h }[4], [%x[outptr]], #0x2\n"
    "b 30f\n"
    "28:"  // Oddments: Store: Bit 2: Unset
    "tbz %x[n_channels], #1, 29f\n"
    "st1 { v8.s }[0], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v8.h }[2], [%x[outptr]], #0x2\n"
    "b 30f\n"
    "29:"  // Oddments: Store: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v8.h }[0], [%x[outptr]], #0x2\n"
    "30:"  // Oddments: Store: Bit 2: End
    "31:"  // End
    : [n_channels] "+&r" (n_channels), [outptr] "+&r" (outptr)
    : [inptrs] "r" (inptrs), [n_valid_cells] "r" (n_valid_cells), [rescale_ptr] "r" (&rescale_value)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
