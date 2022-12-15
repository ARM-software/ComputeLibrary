/*
 * Copyright (c) 2019-2021, 2023 Arm Limited.
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

#ifdef __aarch64__

template<>
void interleave_block<4, 16, VLType::None, false>(
  int8_t * &out_ptr, const int8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool
)
{
  __asm__ __volatile__(
      "ldr x23, [%x[in], #0x0]\n"
      "ldr x22, [%x[in], #0x8]\n"
      "cmp %x[height], #0x4\n"
      "add x23, x23, %x[row_offset]\n"
      "ldr x21, [%x[in], #0x10]\n"
      "ldr x20, [%x[in], #0x18]\n"
      "add x22, x22, %x[row_offset]\n"
      "add x21, x21, %x[row_offset]\n"
      "add x20, x20, %x[row_offset]\n"
      "beq 1f\n"
      "cmp %x[height], #0x2\n"
      "mov x20, x23\n"
      "csel x22, x22, x23, GE\n"
      "csel x21, x21, x23, GT\n"
      "1:"  // no_pointer_adj
      "cmp %x[width], #0x10\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x22, #0x0]\n"
      "prfm pldl1keep, [x21, #0x0]\n"
      "prfm pldl1keep, [x20, #0x0]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "prfm pldl1keep, [x20, #0x40]\n"
      "blt 3f\n"
      "2:"  // Main loop head
      "ldr q19, [x23], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "subs %x[width], %x[width], #0x10\n"
      "cmp %x[width], #0x10\n"
      "ldr q17, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "str q19, [%x[out_ptr], #0x0]\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      "prfm pldl1keep, [x22, #0x70]\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      "prfm pldl1keep, [x20, #0x70]\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "bge 2b\n"
      "3:"  // Main loop skip
      "cbz %x[width], 12f\n"
      "tbz %x[width], #3, 7f\n"
      "ldr d19, [x23], #0x8\n"
      "ldr d18, [x22], #0x8\n"
      "ldr d17, [x21], #0x8\n"
      "ldr d16, [x20], #0x8\n"
      "tbz %x[width], #2, 5f\n"
      "ld1 { v19.s }[2], [x23], #0x4\n"
      "ld1 { v18.s }[2], [x22], #0x4\n"
      "ld1 { v17.s }[2], [x21], #0x4\n"
      "ld1 { v16.s }[2], [x20], #0x4\n"
      "tbz %x[width], #1, 4f\n"
      "ld1 { v19.h }[6], [x23], #0x2\n"
      "ld1 { v18.h }[6], [x22], #0x2\n"
      "ld1 { v17.h }[6], [x21], #0x2\n"
      "ld1 { v16.h }[6], [x20], #0x2\n"
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[14], [x23]\n"
      "ld1 { v18.b }[14], [x22]\n"
      "ld1 { v17.b }[14], [x21]\n"
      "ld1 { v16.b }[14], [x20]\n"
      "b 11f\n"
      "4:"  // odd_loads_1_12
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[12], [x23]\n"
      "ld1 { v18.b }[12], [x22]\n"
      "ld1 { v17.b }[12], [x21]\n"
      "ld1 { v16.b }[12], [x20]\n"
      "b 11f\n"
      "5:"  // odd_loads_2_8
      "tbz %x[width], #1, 6f\n"
      "ld1 { v19.h }[4], [x23], #0x2\n"
      "ld1 { v18.h }[4], [x22], #0x2\n"
      "ld1 { v17.h }[4], [x21], #0x2\n"
      "ld1 { v16.h }[4], [x20], #0x2\n"
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[10], [x23]\n"
      "ld1 { v18.b }[10], [x22]\n"
      "ld1 { v17.b }[10], [x21]\n"
      "ld1 { v16.b }[10], [x20]\n"
      "b 11f\n"
      "6:"  // odd_loads_1_8
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[8], [x23]\n"
      "ld1 { v18.b }[8], [x22]\n"
      "ld1 { v17.b }[8], [x21]\n"
      "ld1 { v16.b }[8], [x20]\n"
      "b 11f\n"
      "7:"  // odd_loads_4_0
      "tbz %x[width], #2, 9f\n"
      "ldr s19, [x23], #0x4\n"
      "ldr s18, [x22], #0x4\n"
      "ldr s17, [x21], #0x4\n"
      "ldr s16, [x20], #0x4\n"
      "tbz %x[width], #1, 8f\n"
      "ld1 { v19.h }[2], [x23], #0x2\n"
      "ld1 { v18.h }[2], [x22], #0x2\n"
      "ld1 { v17.h }[2], [x21], #0x2\n"
      "ld1 { v16.h }[2], [x20], #0x2\n"
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[6], [x23]\n"
      "ld1 { v18.b }[6], [x22]\n"
      "ld1 { v17.b }[6], [x21]\n"
      "ld1 { v16.b }[6], [x20]\n"
      "b 11f\n"
      "8:"  // odd_loads_1_4
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[4], [x23]\n"
      "ld1 { v18.b }[4], [x22]\n"
      "ld1 { v17.b }[4], [x21]\n"
      "ld1 { v16.b }[4], [x20]\n"
      "b 11f\n"
      "9:"  // odd_loads_2_0
      "tbz %x[width], #1, 10f\n"
      "ldr h19, [x23], #0x2\n"
      "ldr h18, [x22], #0x2\n"
      "ldr h17, [x21], #0x2\n"
      "ldr h16, [x20], #0x2\n"
      "tbz %x[width], #0, 11f\n"
      "ld1 { v19.b }[2], [x23]\n"
      "ld1 { v18.b }[2], [x22]\n"
      "ld1 { v17.b }[2], [x21]\n"
      "ld1 { v16.b }[2], [x20]\n"
      "b 11f\n"
      "10:"  // odd_loads_1_0
      "ldr b19, [x23, #0x0]\n"
      "ldr b18, [x22, #0x0]\n"
      "ldr b17, [x21, #0x0]\n"
      "ldr b16, [x20, #0x0]\n"
      "11:"  // Odd load end
      "str q19, [%x[out_ptr], #0x0]\n"
      "str q18, [%x[out_ptr], #0x10]\n"
      "str q17, [%x[out_ptr], #0x20]\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "12:"  // Odds skip

      : [out_ptr] "+&r" (out_ptr), [width] "+&r" (width)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v16", "v17", "v18", "v19", "x20", "x21", "x22", "x23"
    );
}

template<>
void interleave_block<4, 16, VLType::None, false>(
  uint8_t * &out_ptr, const uint8_t * const * in, size_t width, size_t height,
  size_t row_offset, bool
)
{
  int8_t * &out_cast = reinterpret_cast<int8_t * &>(out_ptr);
  const int8_t * const * in_cast = reinterpret_cast<const int8_t * const *>(in);

  interleave_block<4, 16, VLType::None, false>(out_cast, in_cast, width, height, row_offset, false);
}


#endif // __aarch64__
