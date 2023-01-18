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

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_4_1x4(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 4) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 4 * roundup<size_t>(height, 4) * sizeof(uint8_t);

    __asm__ __volatile__(
      "cmp %x[height], #0x10\n"
      "blt 8f\n"
      "1:"  // Main row loop: Head
      "mov x16, %x[in]\n"
      "mov x15, %x[out]\n"
      "add x14, x16, %x[in_stride]\n"
      "add x13, x14, %x[in_stride]\n"
      "add x12, x13, %x[in_stride]\n"
      "add x11, x12, %x[in_stride]\n"
      "add x10, x11, %x[in_stride]\n"
      "add x9, x10, %x[in_stride]\n"
      "add x28, x9, %x[in_stride]\n"
      "add x27, x28, %x[in_stride]\n"
      "add x26, x27, %x[in_stride]\n"
      "add x25, x26, %x[in_stride]\n"
      "add x24, x25, %x[in_stride]\n"
      "add x23, x24, %x[in_stride]\n"
      "add x22, x23, %x[in_stride]\n"
      "add x21, x22, %x[in_stride]\n"
      "add x20, x21, %x[in_stride]\n"
      "add %x[in], x20, %x[in_stride]\n"
      "sub %x[height], %x[height], #0x10\n"
      "mov x19, %x[width]\n"
      "cmp x19, #0x10\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ldr q20, [x16], #0x10\n"
      "sub x19, x19, #0x10\n"
      "ldr q19, [x14], #0x10\n"
      "cmp x19, #0x10\n"
      "ldr q16, [x13], #0x10\n"
      "zip1 v18.16b, v20.16b, v16.16b\n"
      "ldr q17, [x12], #0x10\n"
      "zip2 v5.16b, v20.16b, v16.16b\n"
      "ldr q4, [x11], #0x10\n"
      "ldr q3, [x10], #0x10\n"
      "zip1 v16.16b, v19.16b, v17.16b\n"
      "ldr q2, [x9], #0x10\n"
      "zip2 v1.16b, v19.16b, v17.16b\n"
      "ldr q0, [x28], #0x10\n"
      "zip1 v22.16b, v18.16b, v16.16b\n"
      "ldr q31, [x27], #0x10\n"
      "zip2 v21.16b, v18.16b, v16.16b\n"
      "ldr q30, [x26], #0x10\n"
      "zip1 v29.16b, v5.16b, v1.16b\n"
      "ldr q28, [x25], #0x10\n"
      "zip1 v17.16b, v4.16b, v2.16b\n"
      "ldr q27, [x24], #0x10\n"
      "zip1 v16.16b, v3.16b, v0.16b\n"
      "ldr q26, [x23], #0x10\n"
      "zip1 v19.16b, v17.16b, v16.16b\n"
      "ldr q25, [x22], #0x10\n"
      "zip2 v20.16b, v17.16b, v16.16b\n"
      "ldr q24, [x21], #0x10\n"
      "zip1 v18.16b, v31.16b, v28.16b\n"
      "ldr q23, [x20], #0x10\n"
      "zip1 v17.16b, v30.16b, v27.16b\n"
      "str q22, [x15, #0x0]\n"
      "zip1 v16.16b, v18.16b, v17.16b\n"
      "str q19, [x15, #0x10]\n"
      "zip2 v19.16b, v18.16b, v17.16b\n"
      "str q16, [x15, #0x20]\n"
      "zip1 v18.16b, v26.16b, v24.16b\n"
      "zip1 v17.16b, v25.16b, v23.16b\n"
      "zip1 v16.16b, v18.16b, v17.16b\n"
      "str q16, [x15, #0x30]\n"
      "add x15, x15, %x[out_stride]\n"
      "zip2 v17.16b, v18.16b, v17.16b\n"
      "str q21, [x15, #0x0]\n"
      "zip2 v22.16b, v4.16b, v2.16b\n"
      "str q20, [x15, #0x10]\n"
      "zip2 v21.16b, v3.16b, v0.16b\n"
      "str q19, [x15, #0x20]\n"
      "zip1 v16.16b, v22.16b, v21.16b\n"
      "str q17, [x15, #0x30]\n"
      "add x15, x15, %x[out_stride]\n"
      "zip2 v20.16b, v31.16b, v28.16b\n"
      "str q29, [x15, #0x0]\n"
      "zip2 v17.16b, v30.16b, v27.16b\n"
      "str q16, [x15, #0x10]\n"
      "zip1 v16.16b, v20.16b, v17.16b\n"
      "str q16, [x15, #0x20]\n"
      "zip2 v19.16b, v26.16b, v24.16b\n"
      "zip2 v18.16b, v25.16b, v23.16b\n"
      "zip1 v16.16b, v19.16b, v18.16b\n"
      "str q16, [x15, #0x30]\n"
      "add x15, x15, %x[out_stride]\n"
      "zip2 v16.16b, v5.16b, v1.16b\n"
      "str q16, [x15, #0x0]\n"
      "zip2 v16.16b, v22.16b, v21.16b\n"
      "zip2 v17.16b, v20.16b, v17.16b\n"
      "str q16, [x15, #0x10]\n"
      "zip2 v16.16b, v19.16b, v18.16b\n"
      "str q17, [x15, #0x20]\n"
      "str q16, [x15, #0x30]\n"
      "add x15, x15, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cmp x19, #0x4\n"
      "blt 5f\n"
      "4:"  // Main row loop: Column loop
      "ldr s18, [x16], #0x4\n"
      "sub x19, x19, #0x4\n"
      "ldr s17, [x14], #0x4\n"
      "cmp x19, #0x4\n"
      "ldr s16, [x13], #0x4\n"
      "zip1 v19.16b, v18.16b, v16.16b\n"
      "ldr s16, [x12], #0x4\n"
      "ldr s18, [x11], #0x4\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "ldr s20, [x10], #0x4\n"
      "ldr s17, [x9], #0x4\n"
      "zip1 v23.16b, v19.16b, v16.16b\n"
      "ldr s16, [x28], #0x4\n"
      "zip1 v19.16b, v18.16b, v17.16b\n"
      "ldr s18, [x27], #0x4\n"
      "ldr s22, [x26], #0x4\n"
      "zip1 v16.16b, v20.16b, v16.16b\n"
      "ldr s17, [x25], #0x4\n"
      "zip1 v21.16b, v19.16b, v16.16b\n"
      "ldr s16, [x24], #0x4\n"
      "zip1 v18.16b, v18.16b, v17.16b\n"
      "ldr s20, [x23], #0x4\n"
      "ldr s19, [x22], #0x4\n"
      "zip1 v16.16b, v22.16b, v16.16b\n"
      "ldr s17, [x21], #0x4\n"
      "zip1 v18.16b, v18.16b, v16.16b\n"
      "ldr s16, [x20], #0x4\n"
      "zip1 v17.16b, v20.16b, v17.16b\n"
      "str q23, [x15, #0x0]\n"
      "str q21, [x15, #0x10]\n"
      "zip1 v16.16b, v19.16b, v16.16b\n"
      "str q18, [x15, #0x20]\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "str q16, [x15, #0x30]\n"
      "add x15, x15, %x[out_stride]\n"
      "bge 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x19, #0x1\n"
      "blt 7f\n"
      "6:"  // Main row loop: width 1 loop: loop
      "ldr b18, [x16], #0x1\n"
      "sub x19, x19, #0x1\n"
      "ldr b17, [x14], #0x1\n"
      "cmp x19, #0x1\n"
      "ldr b16, [x13], #0x1\n"
      "zip1 v19.16b, v18.16b, v16.16b\n"
      "ldr b16, [x12], #0x1\n"
      "ldr b18, [x11], #0x1\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "ldr b20, [x10], #0x1\n"
      "ldr b17, [x9], #0x1\n"
      "zip1 v23.16b, v19.16b, v16.16b\n"
      "ldr b16, [x28], #0x1\n"
      "zip1 v19.16b, v18.16b, v17.16b\n"
      "ldr b18, [x27], #0x1\n"
      "ldr b22, [x26], #0x1\n"
      "zip1 v16.16b, v20.16b, v16.16b\n"
      "ldr b17, [x25], #0x1\n"
      "zip1 v21.16b, v19.16b, v16.16b\n"
      "ldr b16, [x24], #0x1\n"
      "zip1 v18.16b, v18.16b, v17.16b\n"
      "ldr b20, [x23], #0x1\n"
      "ldr b19, [x22], #0x1\n"
      "zip1 v16.16b, v22.16b, v16.16b\n"
      "ldr b17, [x21], #0x1\n"
      "zip1 v18.16b, v18.16b, v16.16b\n"
      "ldr b16, [x20], #0x1\n"
      "zip1 v17.16b, v20.16b, v17.16b\n"
      "str s23, [x15, #0x0]\n"
      "str s21, [x15, #0x10]\n"
      "zip1 v16.16b, v19.16b, v16.16b\n"
      "str s18, [x15, #0x20]\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "str s16, [x15, #0x30]\n"
      "add x15, x15, #0x4\n"
      "bge 6b\n"
      "7:"  // Main row loop: width 1 loop: skip
      "add %x[out], %x[out], #0x40\n"
      "cmp %x[height], #0x10\n"
      "bge 1b\n"
      "cbz %x[height], 16f\n"
      "8:"  // Main loop skip

      "9:"  // Tail row loop: Head
      "mov x16, %x[in]\n"
      "mov x15, %x[out]\n"
      "add x14, x16, %x[in_stride]\n"
      "add x13, x14, %x[in_stride]\n"
      "add x12, x13, %x[in_stride]\n"
      "add %x[in], x12, %x[in_stride]\n"
      "cmp %x[height], #0x3\n"
      "csel x12, x12, %x[pad_row], GT\n"
      "csel x13, x13, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x14, x14, %x[pad_row], GT\n"
      "sub %x[height], %x[height], #0x4\n"
      "mov x19, %x[width]\n"
      "cmp x19, #0x10\n"
      "blt 11f\n"
      "10:"  // Tail row loop: Unroll column loop
      "ldr q19, [x16], #0x10\n"
      "sub x19, x19, #0x10\n"
      "ldr q18, [x14], #0x10\n"
      "cmp x19, #0x10\n"
      "ldr q17, [x13], #0x10\n"
      "zip1 v20.16b, v19.16b, v17.16b\n"
      "ldr q16, [x12], #0x10\n"
      "zip2 v19.16b, v19.16b, v17.16b\n"
      "zip1 v17.16b, v18.16b, v16.16b\n"
      "zip2 v18.16b, v18.16b, v16.16b\n"
      "zip1 v16.16b, v20.16b, v17.16b\n"
      "str q16, [x15, #0x0]\n"
      "add x15, x15, %x[out_stride]\n"
      "zip2 v16.16b, v20.16b, v17.16b\n"
      "str q16, [x15, #0x0]\n"
      "zip1 v17.16b, v19.16b, v18.16b\n"
      "add x15, x15, %x[out_stride]\n"
      "zip2 v16.16b, v19.16b, v18.16b\n"
      "str q17, [x15, #0x0]\n"
      "add x15, x15, %x[out_stride]\n"
      "str q16, [x15, #0x0]\n"
      "add x15, x15, %x[out_stride]\n"
      "bge 10b\n"
      "11:"  // Tail row loop: Unroll column loop skip
      "cmp x19, #0x4\n"
      "blt 13f\n"
      "12:"  // Tail row loop: Column loop
      "ldr s17, [x16], #0x4\n"
      "sub x19, x19, #0x4\n"
      "ldr s18, [x14], #0x4\n"
      "cmp x19, #0x4\n"
      "ldr s16, [x13], #0x4\n"
      "zip1 v17.16b, v17.16b, v16.16b\n"
      "ldr s16, [x12], #0x4\n"
      "zip1 v16.16b, v18.16b, v16.16b\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "str q16, [x15, #0x0]\n"
      "add x15, x15, %x[out_stride]\n"
      "bge 12b\n"
      "13:"  // Tail row loop: Column loop skip
      "cmp x19, #0x1\n"
      "blt 15f\n"
      "14:"  // Tail row loop: width 1 loop: loop
      "ldr b17, [x16], #0x1\n"
      "sub x19, x19, #0x1\n"
      "ldr b18, [x14], #0x1\n"
      "cmp x19, #0x1\n"
      "ldr b16, [x13], #0x1\n"
      "zip1 v17.16b, v17.16b, v16.16b\n"
      "ldr b16, [x12], #0x1\n"
      "zip1 v16.16b, v18.16b, v16.16b\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "str s16, [x15, #0x0]\n"
      "add x15, x15, #0x4\n"
      "bge 14b\n"
      "15:"  // Tail row loop: width 1 loop: skip
      "add %x[out], %x[out], #0x10\n"
      "cmp %x[height], #0x1\n"
      "bge 9b\n"
      "16:"  // Done

      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x19", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

} // anonymous namespace

template<>
void Transform<4, 4, true, VLType::None>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_4_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<4, 4, true, VLType::None>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_4_1x4(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif
