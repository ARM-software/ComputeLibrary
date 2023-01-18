/*
 * Copyright (c) 2019-2021 Arm Limited.
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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifdef __aarch64__

template<>
void interleave_block<4, 16, VLType::None, true>(
  uint8_t * &out_ptr, const uint8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool first
)
{
  __asm__ __volatile__(
      "movi v28.8h, #0x0\n"
      "ldr x23, [%x[in], #0x0]\n"
      "mov x22, #0x0\n"
      "movi v27.8h, #0x0\n"
      "ldr x21, [%x[in], #0x8]\n"
      "cmp %x[height], #0x4\n"
      "movi v26.8h, #0x0\n"
      "ldr x20, [%x[in], #0x10]\n"
      "add x23, x23, %x[row_offset]\n"
      "movi v25.8h, #0x0\n"
      "ldr x19, [%x[in], #0x18]\n"
      "movi v24.4s, #0x0\n"
      "add x21, x21, %x[row_offset]\n"
      "movi v23.4s, #0x0\n"
      "add x20, x20, %x[row_offset]\n"
      "movi v22.4s, #0x0\n"
      "add x19, x19, %x[row_offset]\n"
      "movi v21.4s, #0x0\n"
      "beq 1f\n"
      "mov x19, x23\n"
      "cmp %x[height], #0x2\n"
      "csel x21, x21, x23, GE\n"
      "csel x20, x20, x23, GT\n"
      "1:"  // no_pointer_adj
      "movi v20.4s, #0x0\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x21, #0x0]\n"
      "prfm pldl1keep, [x20, #0x0]\n"
      "prfm pldl1keep, [x19, #0x0]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "prfm pldl1keep, [x20, #0x40]\n"
      "prfm pldl1keep, [x19, #0x40]\n"
      "cbnz %w[first], 2f\n"
      "sub %x[out_ptr], %x[out_ptr], #0x10\n"
      "ld1 { v20.4s }, [%x[out_ptr]]\n"
      "2:"  // first_pass
      "cmp %x[width], #0x10\n"
      "blt 5f\n"
      "3:"  // Main loop head
      "cmp x22, #0x7e\n"
      "ble 4f\n"
      "uadalp v24.4s, v28.8h\n"
      "movi v28.8h, #0x0\n"
      "uadalp v23.4s, v27.8h\n"
      "movi v27.8h, #0x0\n"
      "uadalp v22.4s, v26.8h\n"
      "movi v26.8h, #0x0\n"
      "uadalp v21.4s, v25.8h\n"
      "movi v25.8h, #0x0\n"
      "mov x22, #0x0\n"
      "4:"  // no_accumulate_16
      "ldr q19, [x23], #0x10\n"
      "add x22, x22, #0x1\n"
      "ldr q18, [x21], #0x10\n"
      "subs %x[width], %x[width], #0x10\n"
      "ldr q17, [x20], #0x10\n"
      "cmp %x[width], #0x10\n"
      "ldr q16, [x19], #0x10\n"
      "uadalp v28.8h, v19.16b\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      "uadalp v27.8h, v18.16b\n"
      "prfm pldl1keep, [x20, #0x70]\n"
      "uadalp v26.8h, v17.16b\n"
      "prfm pldl1keep, [x19, #0x70]\n"
      "uadalp v25.8h, v16.16b\n"
      "str q19, [%x[out_ptr], #0x0]\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "bge 3b\n"
      "5:"  // Main loop skip
      "cbz %x[width], 14f\n"
      "tbz %x[width], #3, 9f\n"
      "ldr d19, [x23], #0x8\n"
      "ldr d18, [x21], #0x8\n"
      "ldr d17, [x20], #0x8\n"
      "ldr d16, [x19], #0x8\n"
      "tbz %x[width], #2, 7f\n"
      "ld1 { v19.s }[2], [x23], #0x4\n"
      "ld1 { v18.s }[2], [x21], #0x4\n"
      "ld1 { v17.s }[2], [x20], #0x4\n"
      "ld1 { v16.s }[2], [x19], #0x4\n"
      "tbz %x[width], #1, 6f\n"
      "ld1 { v19.h }[6], [x23], #0x2\n"
      "ld1 { v18.h }[6], [x21], #0x2\n"
      "ld1 { v17.h }[6], [x20], #0x2\n"
      "ld1 { v16.h }[6], [x19], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[14], [x23]\n"
      "ld1 { v18.b }[14], [x21]\n"
      "ld1 { v17.b }[14], [x20]\n"
      "ld1 { v16.b }[14], [x19]\n"
      "b 13f\n"
      "6:"  // odd_loads_1_12
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[12], [x23]\n"
      "ld1 { v18.b }[12], [x21]\n"
      "ld1 { v17.b }[12], [x20]\n"
      "ld1 { v16.b }[12], [x19]\n"
      "b 13f\n"
      "7:"  // odd_loads_2_8
      "tbz %x[width], #1, 8f\n"
      "ld1 { v19.h }[4], [x23], #0x2\n"
      "ld1 { v18.h }[4], [x21], #0x2\n"
      "ld1 { v17.h }[4], [x20], #0x2\n"
      "ld1 { v16.h }[4], [x19], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[10], [x23]\n"
      "ld1 { v18.b }[10], [x21]\n"
      "ld1 { v17.b }[10], [x20]\n"
      "ld1 { v16.b }[10], [x19]\n"
      "b 13f\n"
      "8:"  // odd_loads_1_8
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[8], [x23]\n"
      "ld1 { v18.b }[8], [x21]\n"
      "ld1 { v17.b }[8], [x20]\n"
      "ld1 { v16.b }[8], [x19]\n"
      "b 13f\n"
      "9:"  // odd_loads_4_0
      "tbz %x[width], #2, 11f\n"
      "ldr s19, [x23], #0x4\n"
      "ldr s18, [x21], #0x4\n"
      "ldr s17, [x20], #0x4\n"
      "ldr s16, [x19], #0x4\n"
      "tbz %x[width], #1, 10f\n"
      "ld1 { v19.h }[2], [x23], #0x2\n"
      "ld1 { v18.h }[2], [x21], #0x2\n"
      "ld1 { v17.h }[2], [x20], #0x2\n"
      "ld1 { v16.h }[2], [x19], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[6], [x23]\n"
      "ld1 { v18.b }[6], [x21]\n"
      "ld1 { v17.b }[6], [x20]\n"
      "ld1 { v16.b }[6], [x19]\n"
      "b 13f\n"
      "10:"  // odd_loads_1_4
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[4], [x23]\n"
      "ld1 { v18.b }[4], [x21]\n"
      "ld1 { v17.b }[4], [x20]\n"
      "ld1 { v16.b }[4], [x19]\n"
      "b 13f\n"
      "11:"  // odd_loads_2_0
      "tbz %x[width], #1, 12f\n"
      "ldr h19, [x23], #0x2\n"
      "ldr h18, [x21], #0x2\n"
      "ldr h17, [x20], #0x2\n"
      "ldr h16, [x19], #0x2\n"
      "tbz %x[width], #0, 13f\n"
      "ld1 { v19.b }[2], [x23]\n"
      "ld1 { v18.b }[2], [x21]\n"
      "ld1 { v17.b }[2], [x20]\n"
      "ld1 { v16.b }[2], [x19]\n"
      "b 13f\n"
      "12:"  // odd_loads_1_0
      "ldr b19, [x23, #0x0]\n"
      "ldr b18, [x21, #0x0]\n"
      "ldr b17, [x20, #0x0]\n"
      "ldr b16, [x19, #0x0]\n"
      "13:"  // Odd load end
      "str q19, [%x[out_ptr], #0x0]\n"
      "uadalp v28.8h, v19.16b\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "uadalp v27.8h, v18.16b\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "uadalp v26.8h, v17.16b\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "uadalp v25.8h, v16.16b\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "14:"  // Odds skip
      "uadalp v24.4s, v28.8h\n"
      "uadalp v23.4s, v27.8h\n"
      "addp v24.4s, v24.4s, v23.4s\n"
      "uadalp v22.4s, v26.8h\n"
      "uadalp v21.4s, v25.8h\n"
      "addp v23.4s, v22.4s, v21.4s\n"
      "addp v24.4s, v24.4s, v23.4s\n"
      "add v24.4s, v24.4s, v20.4s\n"
      "str q24, [%x[out_ptr], #0x0]\n"
      "add %x[out_ptr], %x[out_ptr], #0x10\n"
      : [out_ptr] "+&r" (out_ptr), [width] "+&r" (width)
      : [first] "r" (first), [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "x19", "x20", "x21", "x22", "x23"
    );
}


#endif // __aarch64__
