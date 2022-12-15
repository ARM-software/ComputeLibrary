/*
 * Copyright (c) 2021, 2023 Arm Limited.
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
void interleave_block<8, 4, VLType::None, false>(
  bfloat16 * &out_ptr, const float * const * in, size_t width, size_t height,
  size_t row_offset, bool
)
{
  __asm__ __volatile__(
      "ldr x28, [%x[in], #0x0]\n"
      "ldr x27, [%x[in], #0x8]\n"
      "cmp %x[height], #0x8\n"
      "add x28, x28, %x[row_offset], LSL #2\n"
      "ldr x26, [%x[in], #0x10]\n"
      "ldr x25, [%x[in], #0x18]\n"
      "add x27, x27, %x[row_offset], LSL #2\n"
      "add x26, x26, %x[row_offset], LSL #2\n"
      "ldr x24, [%x[in], #0x20]\n"
      "ldr x23, [%x[in], #0x28]\n"
      "add x25, x25, %x[row_offset], LSL #2\n"
      "add x24, x24, %x[row_offset], LSL #2\n"
      "ldr x22, [%x[in], #0x30]\n"
      "ldr x21, [%x[in], #0x38]\n"
      "add x23, x23, %x[row_offset], LSL #2\n"
      "add x22, x22, %x[row_offset], LSL #2\n"
      "add x21, x21, %x[row_offset], LSL #2\n"
      "beq 1f\n"
      "cmp %x[height], #0x2\n"
      "csel x27, x27, x28, GE\n"
      "csel x26, x26, x28, GT\n"
      "cmp %x[height], #0x4\n"
      "csel x25, x25, x28, GE\n"
      "csel x24, x24, x28, GT\n"
      "cmp %x[height], #0x6\n"
      "mov x21, x28\n"
      "csel x23, x23, x28, GE\n"
      "csel x22, x22, x28, GT\n"
      "1:"  // no_pointer_adj
      "cmp %x[width], #0x4\n"
      "prfm pldl1keep, [x28, #0x0]\n"
      "prfm pldl1keep, [x27, #0x0]\n"
      "prfm pldl1keep, [x26, #0x0]\n"
      "prfm pldl1keep, [x25, #0x0]\n"
      "prfm pldl1keep, [x24, #0x0]\n"
      "prfm pldl1keep, [x23, #0x0]\n"
      "prfm pldl1keep, [x22, #0x0]\n"
      "prfm pldl1keep, [x21, #0x0]\n"
      "prfm pldl1keep, [x28, #0x40]\n"
      "prfm pldl1keep, [x27, #0x40]\n"
      "prfm pldl1keep, [x26, #0x40]\n"
      "prfm pldl1keep, [x25, #0x40]\n"
      "prfm pldl1keep, [x24, #0x40]\n"
      "prfm pldl1keep, [x23, #0x40]\n"
      "prfm pldl1keep, [x22, #0x40]\n"
      "prfm pldl1keep, [x21, #0x40]\n"
      "blt 3f\n"
      "2:"  // Main loop head
      "ldr q23, [x28], #0x10\n"
      "ldr q22, [x26], #0x10\n"
      ".inst 0x0ea16af7  // bfcvtn v23.4h, v23.4s\n"
      ".inst 0x0ea16ad6  // bfcvtn v22.4h, v22.4s\n"
      "ldr q21, [x24], #0x10\n"
      "ldr q20, [x22], #0x10\n"
      ".inst 0x0ea16ab5  // bfcvtn v21.4h, v21.4s\n"
      ".inst 0x0ea16a94  // bfcvtn v20.4h, v20.4s\n"
      "ldr q19, [x27], #0x10\n"
      "ldr q18, [x25], #0x10\n"
      "subs %x[width], %x[width], #0x4\n"
      "cmp %x[width], #0x4\n"
      "ldr q17, [x23], #0x10\n"
      "ldr q16, [x21], #0x10\n"
      ".inst 0x4ea16a77  // bfcvtn2 v23.8h, v19.4s\n"
      ".inst 0x4ea16a56  // bfcvtn2 v22.8h, v18.4s\n"
      "prfm pldl1keep, [x28, #0x70]\n"
      "prfm pldl1keep, [x27, #0x70]\n"
      ".inst 0x4ea16a35  // bfcvtn2 v21.8h, v17.4s\n"
      ".inst 0x4ea16a14  // bfcvtn2 v20.8h, v16.4s\n"
      "prfm pldl1keep, [x26, #0x70]\n"
      "prfm pldl1keep, [x25, #0x70]\n"
      "str q23, [%x[out_ptr], #0x0]\n"
      "prfm pldl1keep, [x24, #0x70]\n"
      "prfm pldl1keep, [x23, #0x70]\n"
      "str q22, [%x[out_ptr], #0x10]\n"
      "prfm pldl1keep, [x22, #0x70]\n"
      "prfm pldl1keep, [x21, #0x70]\n"
      "str q21, [%x[out_ptr], #0x20]\n"
      "str q20, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "bge 2b\n"
      "3:"  // Main loop skip
      "cbz %x[width], 6f\n"
      "tbz %x[width], #1, 4f\n"
      "ldr d23, [x28], #0x8\n"
      "ldr d19, [x27], #0x8\n"
      "mov x20, #0x1\n"
      "ldr d22, [x26], #0x8\n"
      "ldr d18, [x25], #0x8\n"
      "ldr d21, [x24], #0x8\n"
      "ldr d17, [x23], #0x8\n"
      "ldr d20, [x22], #0x8\n"
      "ldr d16, [x21], #0x8\n"
      "tbz %x[width], #0, 5f\n"
      "ld1 { v23.s }[2], [x28]\n"
      "ld1 { v19.s }[2], [x27]\n"
      "ld1 { v22.s }[2], [x26]\n"
      "ld1 { v18.s }[2], [x25]\n"
      "ld1 { v21.s }[2], [x24]\n"
      "ld1 { v17.s }[2], [x23]\n"
      "ld1 { v20.s }[2], [x22]\n"
      "ld1 { v16.s }[2], [x21]\n"
      "b 5f\n"
      "4:"  // odd_loads_1_0
      "ldr s23, [x28, #0x0]\n"
      "ldr s19, [x27, #0x0]\n"
      "mov x20, #0x1\n"
      "ldr s22, [x26, #0x0]\n"
      "ldr s18, [x25, #0x0]\n"
      "ldr s21, [x24, #0x0]\n"
      "ldr s17, [x23, #0x0]\n"
      "ldr s20, [x22, #0x0]\n"
      "ldr s16, [x21, #0x0]\n"
      "5:"  // Odd load end
      ".inst 0x0ea16af7  // bfcvtn v23.4h, v23.4s\n"
      ".inst 0x0ea16ad6  // bfcvtn v22.4h, v22.4s\n"
      ".inst 0x0ea16ab5  // bfcvtn v21.4h, v21.4s\n"
      ".inst 0x0ea16a94  // bfcvtn v20.4h, v20.4s\n"
      ".inst 0x4ea16a77  // bfcvtn2 v23.8h, v19.4s\n"
      ".inst 0x4ea16a56  // bfcvtn2 v22.8h, v18.4s\n"
      "str q23, [%x[out_ptr], #0x0]\n"
      ".inst 0x4ea16a35  // bfcvtn2 v21.8h, v17.4s\n"
      ".inst 0x4ea16a14  // bfcvtn2 v20.8h, v16.4s\n"
      "str q22, [%x[out_ptr], #0x10]\n"
      "str q21, [%x[out_ptr], #0x20]\n"
      "str q20, [%x[out_ptr], #0x30]\n"
      "add %x[out_ptr], %x[out_ptr], #0x40\n"
      "6:"  // Odds skip

      : [out_ptr] "+&r" (out_ptr), [width] "+&r" (width)
      : [height] "r" (height), [in] "r" (in), [row_offset] "r" (row_offset)
      : "cc", "memory", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}


#endif // __aarch64__
