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

#if defined(__aarch64__)

#include <cstdint>
#include <cstddef>

namespace arm_conv {
namespace pooling {


void a64_s8_nhwc_max_generic_depthfirst_impl(
  const uint64_t,
  const uint64_t n_valid_cells,
  uint64_t n_channels,
  const int8_t *const *const inptrs,
  int8_t *outptr
)
{
  __asm__ __volatile__(
    "cmp %x[n_channels], #0x40\n"
    "mov x28, #0x0\n"
    "mov x27, #0x10\n"  // cntb _, ALL, #1
    "mov x26, #0x20\n"  // cntb _, ALL, #2
    "mov x25, #0x30\n"  // cntb _, ALL, #3
    "blt 7f\n"
    "1:"  // 4-vectors of channels
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "movi v6.16b, #0x80\n"
    "movi v5.16b, #0x80\n"
    "mov x19, %x[inptrs]\n"
    "movi v4.16b, #0x80\n"
    "movi v3.16b, #0x80\n"
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
    "smax v23.16b, v2.16b, v1.16b\n"
    "smax v19.16b, v0.16b, v31.16b\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "smax v22.16b, v30.16b, v22.16b\n"
    "smax v18.16b, v29.16b, v28.16b\n"
    "subs x24, x24, #0x1\n"
    "add x19, x19, #0x20\n"
    "smax v21.16b, v27.16b, v21.16b\n"
    "smax v17.16b, v26.16b, v17.16b\n"
    "ldr q2, [x23, x28]\n"
    "ldr q1, [x22, x28]\n"
    "smax v20.16b, v25.16b, v20.16b\n"
    "smax v16.16b, v24.16b, v16.16b\n"
    "ldr q0, [x21, x28]\n"
    "ldr q31, [x20, x28]\n"
    "smax v19.16b, v23.16b, v19.16b\n"
    "smax v18.16b, v22.16b, v18.16b\n"
    "ldr q30, [x23, x27]\n"
    "ldr q22, [x22, x27]\n"
    "smax v17.16b, v21.16b, v17.16b\n"
    "smax v16.16b, v20.16b, v16.16b\n"
    "ldr q29, [x21, x27]\n"
    "ldr q28, [x20, x27]\n"
    "smax v6.16b, v6.16b, v19.16b\n"
    "smax v5.16b, v5.16b, v18.16b\n"
    "ldr q27, [x23, x26]\n"
    "ldr q21, [x22, x26]\n"
    "smax v4.16b, v4.16b, v17.16b\n"
    "smax v3.16b, v3.16b, v16.16b\n"
    "ldr q26, [x21, x26]\n"
    "ldr q17, [x20, x26]\n"
    "ldr q25, [x23, x25]\n"
    "ldr q20, [x22, x25]\n"
    "ldr q24, [x21, x25]\n"
    "ldr q16, [x20, x25]\n"
    "bgt 2b\n"
    "3:"  // 4-vectors of channels: 4 inputs tail
    "smax v23.16b, v2.16b, v1.16b\n"
    "smax v19.16b, v0.16b, v31.16b\n"
    "smax v22.16b, v30.16b, v22.16b\n"
    "smax v18.16b, v29.16b, v28.16b\n"
    "smax v21.16b, v27.16b, v21.16b\n"
    "smax v17.16b, v26.16b, v17.16b\n"
    "smax v20.16b, v25.16b, v20.16b\n"
    "smax v16.16b, v24.16b, v16.16b\n"
    "smax v19.16b, v23.16b, v19.16b\n"
    "smax v18.16b, v22.16b, v18.16b\n"
    "smax v17.16b, v21.16b, v17.16b\n"
    "smax v16.16b, v20.16b, v16.16b\n"
    "smax v6.16b, v6.16b, v19.16b\n"
    "smax v5.16b, v5.16b, v18.16b\n"
    "smax v4.16b, v4.16b, v17.16b\n"
    "smax v3.16b, v3.16b, v16.16b\n"
    "4:"  // 4-vectors of channels: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 6f\n"
    "5:"  // 4-vectors of channels: Single input loop
    "ldr x23, [x19], #0x8\n"
    "ldr q2, [x23, x28]\n"
    "subs x20, x20, #0x1\n"
    "smax v6.16b, v6.16b, v2.16b\n"
    "ldr q30, [x23, x27]\n"
    "ldr q27, [x23, x26]\n"
    "smax v5.16b, v5.16b, v30.16b\n"
    "smax v4.16b, v4.16b, v27.16b\n"
    "ldr q25, [x23, x25]\n"
    "smax v3.16b, v3.16b, v25.16b\n"
    "bgt 5b\n"
    "6:"  // 4-vectors of channels: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x40\n"
    "cmp %x[n_channels], #0x40\n"
    "str q6, [%x[outptr], x28]\n"
    "str q5, [%x[outptr], x27]\n"
    "add x28, x28, #0x40\n"
    "add x27, x27, #0x40\n"
    "str q4, [%x[outptr], x26]\n"
    "add x26, x26, #0x40\n"
    "str q3, [%x[outptr], x25]\n"
    "add x25, x25, #0x40\n"
    "bge 1b\n"
    "cbz %x[n_channels], 43f\n"
    "7:"  // Single vector of channels
    "cmp %x[n_channels], #0x10\n"
    "blt 14f\n"
    "8:"  // Single vector of channels: Loop
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "movi v6.16b, #0x80\n"
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
    "smax v23.16b, v2.16b, v1.16b\n"
    "smax v19.16b, v0.16b, v31.16b\n"
    "ldp x23, x22, [x19, #0x0]\n"
    "ldp x21, x20, [x19, #0x10]\n"
    "smax v19.16b, v23.16b, v19.16b\n"
    "subs x24, x24, #0x1\n"
    "smax v6.16b, v6.16b, v19.16b\n"
    "add x19, x19, #0x20\n"
    "ldr q2, [x23, x28]\n"
    "ldr q1, [x22, x28]\n"
    "ldr q0, [x21, x28]\n"
    "ldr q31, [x20, x28]\n"
    "bgt 9b\n"
    "10:"  // Single vector of channels: Loop: 4 inputs tail
    "smax v23.16b, v2.16b, v1.16b\n"
    "smax v19.16b, v0.16b, v31.16b\n"
    "smax v19.16b, v23.16b, v19.16b\n"
    "smax v6.16b, v6.16b, v19.16b\n"
    "11:"  // Single vector of channels: Loop: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 13f\n"
    "12:"  // Single vector of channels: Loop: Single input loop
    "ldr x23, [x19], #0x8\n"
    "ldr q2, [x23, x28]\n"
    "subs x20, x20, #0x1\n"
    "smax v6.16b, v6.16b, v2.16b\n"
    "bgt 12b\n"
    "13:"  // Single vector of channels: Loop: Single input loop: End
    "sub %x[n_channels], %x[n_channels], #0x10\n"
    "cmp %x[n_channels], #0x10\n"
    "str q6, [%x[outptr], x28]\n"
    "add x28, x28, #0x10\n"
    "bge 8b\n"
    "cbz %x[n_channels], 43f\n"
    "14:"  // Oddments
    "lsr x24, %x[n_valid_cells], #0x2\n"
    "add %x[outptr], %x[outptr], x28\n"
    "movi v6.16b, #0x80\n"
    "mov x19, %x[inptrs]\n"
    "cbz x24, 24f\n"
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
    "tbz %x[n_channels], #3, 19f\n"
    "ldr d2, [x23], #0x8\n"
    "ldr d1, [x22], #0x8\n"
    "ldr d0, [x21], #0x8\n"
    "ldr d31, [x20], #0x8\n"
    "tbz %x[n_channels], #2, 17f\n"
    "ld1 { v2.s }[2], [x23], #0x4\n"
    "ld1 { v1.s }[2], [x22], #0x4\n"
    "ld1 { v0.s }[2], [x21], #0x4\n"
    "ld1 { v31.s }[2], [x20], #0x4\n"
    "tbz %x[n_channels], #1, 16f\n"
    "ld1 { v2.h }[6], [x23], #0x2\n"
    "ld1 { v1.h }[6], [x22], #0x2\n"
    "ld1 { v0.h }[6], [x21], #0x2\n"
    "ld1 { v31.h }[6], [x20], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[14], [x23], #0x1\n"
    "ld1 { v1.b }[14], [x22], #0x1\n"
    "ld1 { v0.b }[14], [x21], #0x1\n"
    "ld1 { v31.b }[14], [x20], #0x1\n"
    "b 23f\n"
    "16:"  // Oddments: 4 inputs loop: Load: Bit 3: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[12], [x23], #0x1\n"
    "ld1 { v1.b }[12], [x22], #0x1\n"
    "ld1 { v0.b }[12], [x21], #0x1\n"
    "ld1 { v31.b }[12], [x20], #0x1\n"
    "b 23f\n"
    "17:"  // Oddments: 4 inputs loop: Load: Bit 3: Bit 2: Unset
    "tbz %x[n_channels], #1, 18f\n"
    "ld1 { v2.h }[4], [x23], #0x2\n"
    "ld1 { v1.h }[4], [x22], #0x2\n"
    "ld1 { v0.h }[4], [x21], #0x2\n"
    "ld1 { v31.h }[4], [x20], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[10], [x23], #0x1\n"
    "ld1 { v1.b }[10], [x22], #0x1\n"
    "ld1 { v0.b }[10], [x21], #0x1\n"
    "ld1 { v31.b }[10], [x20], #0x1\n"
    "b 23f\n"
    "18:"  // Oddments: 4 inputs loop: Load: Bit 3: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[8], [x23], #0x1\n"
    "ld1 { v1.b }[8], [x22], #0x1\n"
    "ld1 { v0.b }[8], [x21], #0x1\n"
    "ld1 { v31.b }[8], [x20], #0x1\n"
    "b 23f\n"
    "19:"  // Oddments: 4 inputs loop: Load: Bit 3: Unset
    "tbz %x[n_channels], #2, 21f\n"
    "ldr s2, [x23], #0x4\n"
    "ldr s1, [x22], #0x4\n"
    "ldr s0, [x21], #0x4\n"
    "ldr s31, [x20], #0x4\n"
    "tbz %x[n_channels], #1, 20f\n"
    "ld1 { v2.h }[2], [x23], #0x2\n"
    "ld1 { v1.h }[2], [x22], #0x2\n"
    "ld1 { v0.h }[2], [x21], #0x2\n"
    "ld1 { v31.h }[2], [x20], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[6], [x23], #0x1\n"
    "ld1 { v1.b }[6], [x22], #0x1\n"
    "ld1 { v0.b }[6], [x21], #0x1\n"
    "ld1 { v31.b }[6], [x20], #0x1\n"
    "b 23f\n"
    "20:"  // Oddments: 4 inputs loop: Load: Bit 3: Unset: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[4], [x23], #0x1\n"
    "ld1 { v1.b }[4], [x22], #0x1\n"
    "ld1 { v0.b }[4], [x21], #0x1\n"
    "ld1 { v31.b }[4], [x20], #0x1\n"
    "b 23f\n"
    "21:"  // Oddments: 4 inputs loop: Load: Bit 3: Unset: Bit 2: Unset
    "tbz %x[n_channels], #1, 22f\n"
    "ldr h2, [x23], #0x2\n"
    "ldr h1, [x22], #0x2\n"
    "ldr h0, [x21], #0x2\n"
    "ldr h31, [x20], #0x2\n"
    "tbz %x[n_channels], #0, 23f\n"
    "ld1 { v2.b }[2], [x23], #0x1\n"
    "ld1 { v1.b }[2], [x22], #0x1\n"
    "ld1 { v0.b }[2], [x21], #0x1\n"
    "ld1 { v31.b }[2], [x20], #0x1\n"
    "b 23f\n"
    "22:"  // Oddments: 4 inputs loop: Load: Bit 3: Unset: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 23f\n"
    "ldr b2, [x23], #0x1\n"
    "ldr b1, [x22], #0x1\n"
    "ldr b0, [x21], #0x1\n"
    "ldr b31, [x20], #0x1\n"
    "23:"  // Oddments: 4 inputs loop: Load: Bit 3: End
    "smax v23.16b, v2.16b, v1.16b\n"
    "smax v19.16b, v0.16b, v31.16b\n"
    "subs x24, x24, #0x1\n"
    "smax v19.16b, v23.16b, v19.16b\n"
    "smax v6.16b, v6.16b, v19.16b\n"
    "bgt 15b\n"
    "24:"  // Oddments: After loop
    "ands x20, %x[n_valid_cells], #0x3\n"
    "beq 34f\n"
    "25:"  // Oddments: Single input loop
    "ldr x23, [x19], #0x8\n"
    "add x23, x23, x28\n"
    "movi v2.16b, #0x0\n"
    "tbz %x[n_channels], #3, 29f\n"
    "ldr d2, [x23], #0x8\n"
    "tbz %x[n_channels], #2, 27f\n"
    "ld1 { v2.s }[2], [x23], #0x4\n"
    "tbz %x[n_channels], #1, 26f\n"
    "ld1 { v2.h }[6], [x23], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[14], [x23], #0x1\n"
    "b 33f\n"
    "26:"  // Oddments: Single input loop: Load: Bit 3: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[12], [x23], #0x1\n"
    "b 33f\n"
    "27:"  // Oddments: Single input loop: Load: Bit 3: Bit 2: Unset
    "tbz %x[n_channels], #1, 28f\n"
    "ld1 { v2.h }[4], [x23], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[10], [x23], #0x1\n"
    "b 33f\n"
    "28:"  // Oddments: Single input loop: Load: Bit 3: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[8], [x23], #0x1\n"
    "b 33f\n"
    "29:"  // Oddments: Single input loop: Load: Bit 3: Unset
    "tbz %x[n_channels], #2, 31f\n"
    "ldr s2, [x23], #0x4\n"
    "tbz %x[n_channels], #1, 30f\n"
    "ld1 { v2.h }[2], [x23], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[6], [x23], #0x1\n"
    "b 33f\n"
    "30:"  // Oddments: Single input loop: Load: Bit 3: Unset: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[4], [x23], #0x1\n"
    "b 33f\n"
    "31:"  // Oddments: Single input loop: Load: Bit 3: Unset: Bit 2: Unset
    "tbz %x[n_channels], #1, 32f\n"
    "ldr h2, [x23], #0x2\n"
    "tbz %x[n_channels], #0, 33f\n"
    "ld1 { v2.b }[2], [x23], #0x1\n"
    "b 33f\n"
    "32:"  // Oddments: Single input loop: Load: Bit 3: Unset: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 33f\n"
    "ldr b2, [x23], #0x1\n"
    "33:"  // Oddments: Single input loop: Load: Bit 3: End
    "subs x20, x20, #0x1\n"
    "smax v6.16b, v6.16b, v2.16b\n"
    "bgt 25b\n"
    "34:"  // Oddments: Single input loop: End
    "tbz %x[n_channels], #3, 38f\n"
    "st1 { v6.d }[0], [%x[outptr]], #0x8\n"
    "tbz %x[n_channels], #2, 36f\n"
    "st1 { v6.s }[2], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #1, 35f\n"
    "st1 { v6.h }[6], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[14], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "35:"  // Oddments: Store: Bit 3: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[12], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "36:"  // Oddments: Store: Bit 3: Bit 2: Unset
    "tbz %x[n_channels], #1, 37f\n"
    "st1 { v6.h }[4], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[10], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "37:"  // Oddments: Store: Bit 3: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[8], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "38:"  // Oddments: Store: Bit 3: Unset
    "tbz %x[n_channels], #2, 40f\n"
    "st1 { v6.s }[0], [%x[outptr]], #0x4\n"
    "tbz %x[n_channels], #1, 39f\n"
    "st1 { v6.h }[2], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[6], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "39:"  // Oddments: Store: Bit 3: Unset: Bit 2: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[4], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "40:"  // Oddments: Store: Bit 3: Unset: Bit 2: Unset
    "tbz %x[n_channels], #1, 41f\n"
    "st1 { v6.h }[0], [%x[outptr]], #0x2\n"
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[2], [%x[outptr]], #0x1\n"
    "b 42f\n"
    "41:"  // Oddments: Store: Bit 3: Unset: Bit 2: Unset: Bit 1: Unset
    "tbz %x[n_channels], #0, 42f\n"
    "st1 { v6.b }[0], [%x[outptr]], #0x1\n"
    "42:"  // Oddments: Store: Bit 3: End
    "43:"  // End
    : [n_channels] "+&r" (n_channels), [outptr] "+&r" (outptr)
    : [inptrs] "r" (inptrs), [n_valid_cells] "r" (n_valid_cells)
    : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
  );
}

}  // namespace pooling
}  // namespace arm_conv
#endif  // defined(__aarch64__)
