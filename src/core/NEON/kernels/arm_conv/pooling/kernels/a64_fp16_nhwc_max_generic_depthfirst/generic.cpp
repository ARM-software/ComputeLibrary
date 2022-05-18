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

#include <cstdint>
#include <cstddef>

#if defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

namespace arm_conv {
namespace pooling {


void a64_fp16_nhwc_max_generic_depthfirst_impl(
  const uint64_t,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const __fp16 *const *const inptrs,
  __fp16 *outptr
)
{
  __asm__ __volatile__(
    "cmp %x[n_channels], #0x20\n"
    "mov x28, #0x0\n"
    "mov x27, #0x10\n"  // cntb _, ALL, #1
    "mov x26, #0x20\n"  // cntb _, ALL, #2
    "mov x25, #0x30\n"  // cntb _, ALL, #3
    "blt 7f\n"
    "1:"  // 4-vectors of channels
    "mov w19, #0xfc00\n"
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "dup v6.8h, w19\n"
    "dup v5.8h, w19\n"
    "dup v4.8h, w19\n"
    "dup v3.8h, w19\n"
    "mov x19, %x[inptrs]\n"
    "cbz x24, 4f\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "subs x24, x24, #0x1\n"
    "add x19, x19, #0x20\n"
    "ldr q2, [x23, x28]\n"
    "ldr q1, [x22, x28]\n"
    "ldr q0, [x21, x28]\n"
    "ldr q31, [x20, x28]\n"
    "ldr q30, [x23, x27]\n"
    "ldr q22, [x22, x27]\n"
    "ldr q29, [x21, x27]\n"
    "ldr q28, [x20, x27]\n"
    "ldr q27, [x23, x26]\n"
    "ldr q21, [x22, x26]\n"
    "ldr q26, [x21, x26]\n"
    "ldr q17, [x20, x26]\n"
    "ldr q25, [x23, x25]\n"
    "ldr q20, [x22, x25]\n"
    "ldr q24, [x21, x25]\n"
    "ldr q16, [x20, x25]\n"
    "beq 3f\n"
    "2:"  // 4-vectors of channels: 4 inputs loop
    "fmax v23.8h, v2.8h, v1.8h\n"
    "fmax v19.8h, v0.8h, v31.8h\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "fmax v22.8h, v30.8h, v22.8h\n"
    "fmax v18.8h, v29.8h, v28.8h\n"
    "subs x24, x24, #0x1\n"
    "add x19, x19, #0x20\n"
    "fmax v21.8h, v27.8h, v21.8h\n"
    "fmax v17.8h, v26.8h, v17.8h\n"
    "ldr q2, [x23, x28]\n"
    "ldr q1, [x22, x28]\n"
    "fmax v20.8h, v25.8h, v20.8h\n"
    "fmax v16.8h, v24.8h, v16.8h\n"
    "ldr q0, [x21, x28]\n"
    "ldr q31, [x20, x28]\n"
    "fmax v19.8h, v23.8h, v19.8h\n"
    "fmax v18.8h, v22.8h, v18.8h\n"
    "ldr q30, [x23, x27]\n"
    "ldr q22, [x22, x27]\n"
    "fmax v17.8h, v21.8h, v17.8h\n"
    "fmax v16.8h, v20.8h, v16.8h\n"
    "ldr q29, [x21, x27]\n"
    "ldr q28, [x20, x27]\n"
    "fmax v6.8h, v6.8h, v19.8h\n"
    "fmax v5.8h, v5.8h, v18.8h\n"
    "ldr q27, [x23, x26]\n"
    "ldr q21, [x22, x26]\n"
    "fmax v4.8h, v4.8h, v17.8h\n"
    "fmax v3.8h, v3.8h, v16.8h\n"
    "ldr q26, [x21, x26]\n"
    "ldr q17, [x20, x26]\n"
    "ldr q25, [x23, x25]\n"
    "ldr q20, [x22, x25]\n"
    "ldr q24, [x21, x25]\n"
    "ldr q16, [x20, x25]\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "fmax v23.8h, v2.8h, v1.8h\n"
    "fmax v19.8h, v0.8h, v31.8h\n"
    "fmax v22.8h, v30.8h, v22.8h\n"
    "fmax v18.8h, v29.8h, v28.8h\n"
    "fmax v21.8h, v27.8h, v21.8h\n"
    "fmax v17.8h, v26.8h, v17.8h\n"
    "fmax v20.8h, v25.8h, v20.8h\n"
    "fmax v16.8h, v24.8h, v16.8h\n"
    "fmax v19.8h, v23.8h, v19.8h\n"
    "fmax v18.8h, v22.8h, v18.8h\n"
    "fmax v17.8h, v21.8h, v17.8h\n"
    "fmax v16.8h, v20.8h, v16.8h\n"
    "fmax v6.8h, v6.8h, v19.8h\n"
    "fmax v5.8h, v5.8h, v18.8h\n"
    "fmax v4.8h, v4.8h, v17.8h\n"
    "fmax v3.8h, v3.8h, v16.8h\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x23, [x19], #0x8\n"
    "ldr q2, [x23, x28]\n"
    "subs x20, x20, #0x1\n"
    "fmax v6.8h, v6.8h, v2.8h\n"
    "ldr q30, [x23, x27]\n"
    "ldr q27, [x23, x26]\n"
    "fmax v5.8h, v5.8h, v30.8h\n"
    "fmax v4.8h, v4.8h, v27.8h\n"
    "ldr q25, [x23, x25]\n"
    "fmax v3.8h, v3.8h, v25.8h\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x20\n"
    "cmp %x[n_channels], #0x20\n"
    "str q6, [%x[outptr], x28]\n"
    "str q5, [%x[outptr], x27]\n"
    "add x28, x28, #0x40\n"
    "add x27, x27, #0x40\n"
    "str q4, [%x[outptr], x26]\n"
    "add x26, x26, #0x40\n"
    "str q3, [%x[outptr], x25]\n"
    "add x25, x25, #0x40\n"
    "bge 1b\n"
    "cbz %x[n_channels], 31f\n"
    "7:"  // Single vector of channels
    "cmp %x[n_channels], #0x8\n"
    "blt 14f\n"
    "8:"  // Single vector of channels: Loop
    "mov w19, #0xfc00\n"
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "dup v6.8h, w19\n"
    "mov x19, %x[inptrs]\n"
    "cbz x24, 11f\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "subs x24, x24, #0x1\n"
    "add x19, x19, #0x20\n"
    "ldr q2, [x23, x28]\n"
    "ldr q1, [x22, x28]\n"
    "ldr q0, [x21, x28]\n"
    "ldr q31, [x20, x28]\n"
    "beq 10f\n"
    "9:"  // Single vector of channels: Loop: 4 inputs loop
    "fmax v23.8h, v2.8h, v1.8h\n"
    "fmax v19.8h, v0.8h, v31.8h\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "fmax v19.8h, v23.8h, v19.8h\n"
    "subs x24, x24, #0x1\n"
    "fmax v6.8h, v6.8h, v19.8h\n"
    "add x19, x19, #0x20\n"
    "ldr q2, [x23, x28]\n"
    "ldr q1, [x22, x28]\n"
    "ldr q0, [x21, x28]\n"
    "ldr q31, [x20, x28]\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "fmax v23.8h, v2.8h, v1.8h\n"
    "fmax v19.8h, v0.8h, v31.8h\n"
    "fmax v19.8h, v23.8h, v19.8h\n"
    "fmax v6.8h, v6.8h, v19.8h\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x23, [x19], #0x8\n"
    "ldr q2, [x23, x28]\n"
    "subs x20, x20, #0x1\n"
    "fmax v6.8h, v6.8h, v2.8h\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x8\n"
    "cmp %x[n_channels], #0x8\n"
    "str q6, [%x[outptr], x28]\n"
    "add x28, x28, #0x10\n"
    "bge 8b\n"
    "cbz %x[n_channels], 31f\n"
    "14:"  // Oddments
    "mov w19, #0xfc00\n"
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "dup v6.8h, w19\n"
    "add %x[outptr], %x[outptr], x28\n"
    "mov x19, %x[inptrs]\n"
    "cbz x24, 20f\n"
    "15:"  // Oddments: 4 inputs loop
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "add x19, x19, #0x20\n"
    "add x23, x23, x28\n"
    "add x22, x22, x28\n"
    "add x21, x21, x28\n"
    "movi v2.16b, #0x0\n"
    "movi v1.16b, #0x0\n"
    "add x20, x20, x28\n"
    "movi v0.16b, #0x0\n"
    "movi v31.16b, #0x0\n"
    "tbz %x[n_channels], #2, 17f\n"
    "ldr d2, [x23], #0x8\n"
    "ldr d1, [x22], #0x8\n"
    "ldr d0, [x21], #0x8\n"
    "ldr d31, [x20], #0x8\n"
    "tbz %x[n_channels], #1, 16f\n"
    "ld1 { v2.s }[2], [x23], #0x4\n"
    "ld1 { v1.s }[2], [x22], #0x4\n"
    "ld1 { v0.s }[2], [x21], #0x4\n"
    "ld1 { v31.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v2.h }[6], [x23], #0x2\n"
    "ld1 { v1.h }[6], [x22], #0x2\n"
    "ld1 { v0.h }[6], [x21], #0x2\n"
    "ld1 { v31.h }[6], [x20], #0x2\n"
    "b 19f\n"
    "16:"  // Oddments: 4 inputs loop: Load: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v2.h }[4], [x23], #0x2\n"
    "ld1 { v1.h }[4], [x22], #0x2\n"
    "ld1 { v0.h }[4], [x21], #0x2\n"
    "ld1 { v31.h }[4], [x20], #0x2\n"
    "b 19f\n"
    "17:"  // Oddments: 4 inputs loop: Load: Bit 2: Unset
    "tbz %x[n_channels], #1, 18f\n"
    "ldr s2, [x23], #0x4\n"
    "ldr s1, [x22], #0x4\n"
    "ldr s0, [x21], #0x4\n"
    "ldr s31, [x20], #0x4\n"
    "tbz %x[n_channels], #0, 19f\n"
    "ld1 { v2.h }[2], [x23], #0x2\n"
    "ld1 { v1.h }[2], [x22], #0x2\n"
    "ld1 { v0.h }[2], [x21], #0x2\n"
    "ld1 { v31.h }[2], [x20], #0x2\n"
    "b 19f\n"
    "18:"  // Oddments: 4 inputs loop: Load: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 19f\n"
    "ldr h2, [x23], #0x2\n"
    "ldr h1, [x22], #0x2\n"
    "ldr h0, [x21], #0x2\n"
    "ldr h31, [x20], #0x2\n"
    "19:"  // Oddments: 4 inputs loop: Load: Bit 2: End
    "fmax v23.8h, v2.8h, v1.8h\n"
    "fmax v19.8h, v0.8h, v31.8h\n"
    "subs x24, x24, #0x1\n"
    "fmax v19.8h, v23.8h, v19.8h\n"
    "fmax v6.8h, v6.8h, v19.8h\n"
    "bgt 15b\n"
    "20:"  // Oddments: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 26f\n"
    "21:"  // Oddments: Single input loop
    "ldr x23, [x19], #0x8\n"
    "add x23, x23, x28\n"
    "movi v2.16b, #0x0\n"
    "tbz %x[n_channels], #2, 23f\n"
    "ldr d2, [x23], #0x8\n"
    "tbz %x[n_channels], #1, 22f\n"
    "ld1 { v2.s }[2], [x23], #0x4\n"
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v2.h }[6], [x23], #0x2\n"
    "b 25f\n"
    "22:"  // Oddments: Single input loop: Load: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v2.h }[4], [x23], #0x2\n"
    "b 25f\n"
    "23:"  // Oddments: Single input loop: Load: Bit 2: Unset
    "tbz %x[n_channels], #1, 24f\n"
    "ldr s2, [x23], #0x4\n"
    "tbz %x[n_channels], #0, 25f\n"
    "ld1 { v2.h }[2], [x23], #0x2\n"
    "b 25f\n"
    "24:"  // Oddments: Single input loop: Load: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 25f\n"
    "ldr h2, [x23], #0x2\n"
    "25:"  // Oddments: Single input loop: Load: Bit 2: End
    "subs x20, x20, #0x1\n"
    "fmax v6.8h, v6.8h, v2.8h\n"
    "bgt 21b\n"
    "26:"  // Oddments: Single input loop: End
    "tbz %x[n_channels], #2, 28f\n"
    "st1 { v6.d }[0], [%x[outptr]], #0x8\n"
    "tbz %x[n_channels], #1, 27f\n"
    "st1 { v6.s }[2], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v6.h }[6], [%x[outptr]], #0x2\n"
    "b 30f\n"
    "27:"  // Oddments: Store: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v6.h }[4], [%x[outptr]], #0x2\n"
    "b 30f\n"
    "28:"  // Oddments: Store: Bit 2: Unset
    "tbz %x[n_channels], #1, 29f\n"
    "st1 { v6.s }[0], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v6.h }[2], [%x[outptr]], #0x2\n"
    "b 30f\n"
    "29:"  // Oddments: Store: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 30f\n"
    "st1 { v6.h }[0], [%x[outptr]], #0x2\n"
    "30:"  // Oddments: Store: Bit 2: End
    "31:"  // End
    : [n_channels] "+&r" (n_channels), [outptr] "+&r" (outptr)
    : [inptrs] "r" (inptrs), [n_valid_cells] "r" (n_valid_cells)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv

#endif  // defined(__aarch64__) && defined(__ARM_FP16_ARGS) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
