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
void interleave_block<8, 2, VLType::None, false>(
  float * &out_ptr, const float * const * in, size_t width, size_t height,
  size_t row_offset, bool
)
{
  __asm__ __volatile__(
      "ldr x27, [%x[in], #0x0]\n"
      "cmp %x[height], #0x8\n"
      "ldr x26, [%x[in], #0x8]\n"
      "add x27, x27, %x[row_offset], LSL #2\n"
      "ldr x25, [%x[in], #0x10]\n"
      "ldr x24, [%x[in], #0x18]\n"
      "add x26, x26, %x[row_offset], LSL #2\n"
      "ldr x23, [%x[in], #0x20]\n"
      "add x25, x25, %x[row_offset], LSL #2\n"
      "ldr x22, [%x[in], #0x28]\n"
      "ldr x21, [%x[in], #0x30]\n"
      "add x24, x24, %x[row_offset], LSL #2\n"
      "ldr x20, [%x[in], #0x38]\n"
      "add x23, x23, %x[row_offset], LSL #2\n"
      "add x22, x22, %x[row_offset], LSL #2\n"
      "add x21, x21, %x[row_offset], LSL #2\n"
      "add x20, x20, %x[row_offset], LSL #2\n"
      "beq 1f\n"
      "mov x20, x27\n"
      "cmp %x[height], #0x2\n"
      "csel x26, x26, x27, GE\n"
      "csel x25, x25, x27, GT\n"
      "cmp %x[height], #0x4\n"
      "csel x24, x24, x27, GE\n"
      "csel x23, x23, x27, GT\n"
      "cmp %x[height], #0x6\n"
      "csel x22, x22, x27, GE\n"
      "csel x21, x21, x27, GT\n"
      "1:"  // no_pointer_adj
      "prfm pldl1keep, [x27, #0x0]\n"
      "cmp %x[width], #0x4\n"
      "prfm pldl1keep, [x26, #0x0]\n"
      "prfm pldl1keep, [x25, #0x0]\n"
      "prfm pldl1keep, [x24, #0x0]\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x22, #0x0]\n"
      "prfm pldl1keep, [x21, #0x0]\n"
      "prfm pldl1keep, [x20, #0x0]\n"
      "prfm pldl1keep, [x27, #0x40]\n"
      "prfm pldl1keep, [x26, #0x40]\n"
      "prfm pldl1keep, [x25, #0x40]\n"
      "prfm pldl1keep, [x24, #0x40]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "prfm pldl1keep, [x20, #0x40]\n"
      "blt 3f\n"
      "2:"  // Main loop head
      "ldr q27, [x27], #0x10\n"
      "subs %x[width], %x[width], #0x4\n"
      "ldr q24, [x26], #0x10\n"
      "zip1 v26.2d, v27.2d, v24.2d\n"
      "ldr q25, [x25], #0x10\n"
      "cmp %x[width], #0x4\n"
      "zip2 v24.2d, v27.2d, v24.2d\n"
      "ldr q21, [x24], #0x10\n"
      "ldr q23, [x23], #0x10\n"
      "zip1 v22.2d, v25.2d, v21.2d\n"
      "ldr q18, [x22], #0x10\n"
      "zip2 v21.2d, v25.2d, v21.2d\n"
      "ldr q20, [x21], #0x10\n"
      "ldr q16, [x20], #0x10\n"
      "zip1 v19.2d, v23.2d, v18.2d\n"
      "prfm pldl1keep, [x27, #0x70]\n"
      "zip2 v18.2d, v23.2d, v18.2d\n"
      "prfm pldl1keep, [x26, #0x70]\n"
      "zip1 v17.2d, v20.2d, v16.2d\n"
      "prfm pldl1keep, [x25, #0x70]\n"
      "zip2 v16.2d, v20.2d, v16.2d\n"
      "prfm pldl1keep, [x24, #0x70]\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      "prfm pldl1keep, [x22, #0x70]\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      "prfm pldl1keep, [x20, #0x70]\n"
      "str q26, [%x[out_ptr], #0x0]\n"
      "str q22, [%x[out_ptr], #0x10]\n"
      "str q19, [%x[out_ptr], #0x20]\n"
      "str q17, [%x[out_ptr], #0x30]\n"
      "str q24, [%x[out_ptr], #0x40]\n"
      "str q21, [%x[out_ptr], #0x50]\n"
      "str q18, [%x[out_ptr], #0x60]\n"
      "str q16, [%x[out_ptr], #0x70]\n"
      "add %x[out_ptr], %x[out_ptr], #0x80\n"
      "bge 2b\n"
      "3:"  // Main loop skip
      "cbz %x[width], 6f\n"
      "tbz %x[width], #1, 4f\n"
      "ldr d27, [x27], #0x8\n"
      "ldr d24, [x26], #0x8\n"
      "mov x19, #0x1\n"
      "ldr d25, [x25], #0x8\n"
      "ldr d21, [x24], #0x8\n"
      "ldr d23, [x23], #0x8\n"
      "ldr d18, [x22], #0x8\n"
      "ldr d20, [x21], #0x8\n"
      "ldr d16, [x20], #0x8\n"
      "tbz %x[width], #0, 5f\n"
      "ld1 { v27.s }[2], [x27]\n"
      "mov x19, #0x2\n"
      "ld1 { v24.s }[2], [x26]\n"
      "ld1 { v25.s }[2], [x25]\n"
      "ld1 { v21.s }[2], [x24]\n"
      "ld1 { v23.s }[2], [x23]\n"
      "ld1 { v18.s }[2], [x22]\n"
      "ld1 { v20.s }[2], [x21]\n"
      "ld1 { v16.s }[2], [x20]\n"
      "b 5f\n"
      "4:"  // odd_loads_1_0
      "ldr s27, [x27, #0x0]\n"
      "mov x19, #0x1\n"
      "ldr s24, [x26, #0x0]\n"
      "ldr s25, [x25, #0x0]\n"
      "ldr s21, [x24, #0x0]\n"
      "ldr s23, [x23, #0x0]\n"
      "ldr s18, [x22, #0x0]\n"
      "ldr s20, [x21, #0x0]\n"
      "ldr s16, [x20, #0x0]\n"
      "5:"  // Odd load end
      "zip1 v26.2d, v27.2d, v24.2d\n"
      "str q26, [%x[out_ptr], #0x0]\n"
      "zip1 v22.2d, v25.2d, v21.2d\n"
      "subs x19, x19, #0x1\n"
      "zip1 v19.2d, v23.2d, v18.2d\n"
      "str q22, [%x[out_ptr], #0x10]\n"
      "zip1 v17.2d, v20.2d, v16.2d\n"
      "str q19, [%x[out_ptr], #0x20]\n"
      "str q17, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "beq 6f\n"
      "zip2 v24.2d, v27.2d, v24.2d\n"
      "str q24, [%x[out_ptr], #0x0]\n"
      "zip2 v21.2d, v25.2d, v21.2d\n"
      "zip2 v18.2d, v23.2d, v18.2d\n"
      "str q21, [%x[out_ptr], #0x10]\n"
      "zip2 v16.2d, v20.2d, v16.2d\n"
      "str q18, [%x[out_ptr], #0x20]\n"
      "str q16, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "6:"  // Odds skip

      : [out_ptr] "+&r" (out_ptr), [width] "+&r" (width)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27"
    );
}


#endif // __aarch64__
