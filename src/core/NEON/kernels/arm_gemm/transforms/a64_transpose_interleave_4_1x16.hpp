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

#pragma once

#ifdef __aarch64__

namespace {

void a64_transpose_interleave_4_1x16(uint8_t *out, const uint8_t *in, size_t width, size_t in_stride, size_t height)
{
    uint8_t *pad_row = reinterpret_cast<uint8_t *>(alloca(width * sizeof(uint8_t)));

    if (height % 16) {
        memset(pad_row, 0, width * sizeof(uint8_t));
    }

    size_t out_stride = 4 * roundup<size_t>(height, 16) * sizeof(uint8_t);

    __asm__ __volatile__(

      "1:"  // Main row loop: Head
      "mov x17, %x[in]\n"
      "add x16, x17, %x[in_stride]\n"
      "add x15, x16, %x[in_stride]\n"
      "add x14, x15, %x[in_stride]\n"
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
      "cmp %x[height], #0xf\n"
      "add %x[in], x22, %x[in_stride]\n"
      "csel x22, x22, %x[pad_row], GT\n"
      "csel x23, x23, %x[pad_row], GE\n"
      "cmp %x[height], #0xd\n"
      "csel x24, x24, %x[pad_row], GT\n"
      "csel x25, x25, %x[pad_row], GE\n"
      "cmp %x[height], #0xb\n"
      "csel x26, x26, %x[pad_row], GT\n"
      "csel x27, x27, %x[pad_row], GE\n"
      "cmp %x[height], #0x9\n"
      "csel x28, x28, %x[pad_row], GT\n"
      "csel x9, x9, %x[pad_row], GE\n"
      "cmp %x[height], #0x7\n"
      "csel x10, x10, %x[pad_row], GT\n"
      "csel x11, x11, %x[pad_row], GE\n"
      "cmp %x[height], #0x5\n"
      "mov x21, %x[width]\n"
      "csel x12, x12, %x[pad_row], GT\n"
      "csel x13, x13, %x[pad_row], GE\n"
      "cmp %x[height], #0x3\n"
      "csel x14, x14, %x[pad_row], GT\n"
      "csel x15, x15, %x[pad_row], GE\n"
      "cmp %x[height], #0x1\n"
      "csel x16, x16, %x[pad_row], GT\n"
      "cmp x21, #0x10\n"
      "mov x20, %x[out]\n"
      "sub %x[height], %x[height], #0x10\n"
      "blt 3f\n"
      "2:"  // Main row loop: Unroll column loop
      "ldr q3, [x17], #0x10\n"
      "ldr q9, [x16], #0x10\n"
      "sub x21, x21, #0x10\n"
      "cmp x21, #0x10\n"
      "ldr q2, [x15], #0x10\n"
      "ldr q8, [x14], #0x10\n"
      "ldr q0, [x13], #0x10\n"
      "ldr q31, [x12], #0x10\n"
      "ldr q30, [x11], #0x10\n"
      "ldr q7, [x10], #0x10\n"
      "ldr q29, [x9], #0x10\n"
      "ldr q28, [x28], #0x10\n"
      "zip1 v27.16b, v3.16b, v29.16b\n"
      "zip1 v6.16b, v9.16b, v28.16b\n"
      "ldr q25, [x27], #0x10\n"
      "ldr q24, [x26], #0x10\n"
      "zip1 v26.16b, v2.16b, v25.16b\n"
      "zip1 v1.16b, v8.16b, v24.16b\n"
      "ldr q23, [x25], #0x10\n"
      "ldr q22, [x24], #0x10\n"
      "zip1 v21.16b, v0.16b, v23.16b\n"
      "zip1 v20.16b, v31.16b, v22.16b\n"
      "ldr q19, [x23], #0x10\n"
      "ldr q18, [x22], #0x10\n"
      "zip1 v17.16b, v30.16b, v19.16b\n"
      "zip1 v16.16b, v7.16b, v18.16b\n"
      "zip2 v5.16b, v3.16b, v29.16b\n"
      "zip2 v0.16b, v0.16b, v23.16b\n"
      "zip2 v4.16b, v2.16b, v25.16b\n"
      "zip2 v3.16b, v30.16b, v19.16b\n"
      "zip2 v2.16b, v9.16b, v28.16b\n"
      "zip2 v31.16b, v31.16b, v22.16b\n"
      "zip2 v30.16b, v8.16b, v24.16b\n"
      "zip2 v29.16b, v7.16b, v18.16b\n"
      "zip1 v25.16b, v27.16b, v21.16b\n"
      "zip1 v24.16b, v26.16b, v17.16b\n"
      "zip1 v23.16b, v6.16b, v20.16b\n"
      "zip1 v22.16b, v1.16b, v16.16b\n"
      "zip2 v28.16b, v27.16b, v21.16b\n"
      "zip2 v27.16b, v26.16b, v17.16b\n"
      "zip2 v26.16b, v6.16b, v20.16b\n"
      "zip2 v21.16b, v1.16b, v16.16b\n"
      "zip1 v1.16b, v5.16b, v0.16b\n"
      "zip1 v20.16b, v4.16b, v3.16b\n"
      "zip1 v19.16b, v2.16b, v31.16b\n"
      "zip1 v16.16b, v30.16b, v29.16b\n"
      "zip1 v18.16b, v25.16b, v24.16b\n"
      "zip1 v17.16b, v23.16b, v22.16b\n"
      "zip2 v25.16b, v25.16b, v24.16b\n"
      "zip2 v24.16b, v23.16b, v22.16b\n"
      "zip2 v0.16b, v5.16b, v0.16b\n"
      "zip2 v23.16b, v4.16b, v3.16b\n"
      "zip2 v31.16b, v2.16b, v31.16b\n"
      "zip2 v22.16b, v30.16b, v29.16b\n"
      "zip1 v30.16b, v28.16b, v27.16b\n"
      "zip1 v29.16b, v26.16b, v21.16b\n"
      "zip2 v28.16b, v28.16b, v27.16b\n"
      "zip2 v27.16b, v26.16b, v21.16b\n"
      "zip1 v26.16b, v1.16b, v20.16b\n"
      "zip1 v21.16b, v19.16b, v16.16b\n"
      "zip2 v20.16b, v1.16b, v20.16b\n"
      "zip2 v19.16b, v19.16b, v16.16b\n"
      "zip1 v16.16b, v18.16b, v17.16b\n"
      "zip2 v18.16b, v18.16b, v17.16b\n"
      "str q16, [x20, #0x0]\n"
      "zip1 v17.16b, v25.16b, v24.16b\n"
      "zip2 v16.16b, v25.16b, v24.16b\n"
      "str q18, [x20, #0x10]\n"
      "str q17, [x20, #0x20]\n"
      "zip1 v25.16b, v0.16b, v23.16b\n"
      "zip1 v24.16b, v31.16b, v22.16b\n"
      "str q16, [x20, #0x30]\n"
      "add x20, x20, %x[out_stride]\n"
      "zip2 v23.16b, v0.16b, v23.16b\n"
      "zip2 v22.16b, v31.16b, v22.16b\n"
      "zip1 v16.16b, v30.16b, v29.16b\n"
      "zip2 v17.16b, v30.16b, v29.16b\n"
      "str q16, [x20, #0x0]\n"
      "zip1 v16.16b, v28.16b, v27.16b\n"
      "zip2 v18.16b, v28.16b, v27.16b\n"
      "str q17, [x20, #0x10]\n"
      "str q16, [x20, #0x20]\n"
      "zip1 v17.16b, v26.16b, v21.16b\n"
      "zip2 v16.16b, v26.16b, v21.16b\n"
      "str q18, [x20, #0x30]\n"
      "add x20, x20, %x[out_stride]\n"
      "zip1 v21.16b, v20.16b, v19.16b\n"
      "zip2 v20.16b, v20.16b, v19.16b\n"
      "str q17, [x20, #0x0]\n"
      "zip1 v19.16b, v25.16b, v24.16b\n"
      "zip2 v18.16b, v25.16b, v24.16b\n"
      "str q16, [x20, #0x10]\n"
      "zip1 v17.16b, v23.16b, v22.16b\n"
      "zip2 v16.16b, v23.16b, v22.16b\n"
      "str q21, [x20, #0x20]\n"
      "str q20, [x20, #0x30]\n"
      "add x20, x20, %x[out_stride]\n"
      "str q19, [x20, #0x0]\n"
      "str q18, [x20, #0x10]\n"
      "str q17, [x20, #0x20]\n"
      "str q16, [x20, #0x30]\n"
      "add x20, x20, %x[out_stride]\n"
      "bge 2b\n"
      "3:"  // Main row loop: Unroll column loop skip
      "cmp x21, #0x4\n"
      "blt 5f\n"
      "4:"  // Main row loop: Column loop
      "ldr s21, [x17], #0x4\n"
      "ldr s23, [x16], #0x4\n"
      "sub x21, x21, #0x4\n"
      "cmp x21, #0x4\n"
      "ldr s20, [x15], #0x4\n"
      "ldr s22, [x14], #0x4\n"
      "ldr s19, [x13], #0x4\n"
      "ldr s18, [x12], #0x4\n"
      "ldr s25, [x11], #0x4\n"
      "ldr s24, [x10], #0x4\n"
      "ldr s17, [x9], #0x4\n"
      "ldr s16, [x28], #0x4\n"
      "zip1 v21.16b, v21.16b, v17.16b\n"
      "zip1 v23.16b, v23.16b, v16.16b\n"
      "ldr s17, [x27], #0x4\n"
      "ldr s16, [x26], #0x4\n"
      "zip1 v20.16b, v20.16b, v17.16b\n"
      "zip1 v22.16b, v22.16b, v16.16b\n"
      "ldr s17, [x25], #0x4\n"
      "ldr s16, [x24], #0x4\n"
      "zip1 v19.16b, v19.16b, v17.16b\n"
      "zip1 v18.16b, v18.16b, v16.16b\n"
      "ldr s17, [x23], #0x4\n"
      "ldr s16, [x22], #0x4\n"
      "zip1 v17.16b, v25.16b, v17.16b\n"
      "zip1 v16.16b, v24.16b, v16.16b\n"
      "zip1 v21.16b, v21.16b, v19.16b\n"
      "zip1 v20.16b, v20.16b, v17.16b\n"
      "zip1 v19.16b, v23.16b, v18.16b\n"
      "zip1 v16.16b, v22.16b, v16.16b\n"
      "zip1 v18.16b, v21.16b, v20.16b\n"
      "zip1 v17.16b, v19.16b, v16.16b\n"
      "zip2 v20.16b, v21.16b, v20.16b\n"
      "zip2 v19.16b, v19.16b, v16.16b\n"
      "zip1 v16.16b, v18.16b, v17.16b\n"
      "zip2 v18.16b, v18.16b, v17.16b\n"
      "str q16, [x20, #0x0]\n"
      "zip1 v17.16b, v20.16b, v19.16b\n"
      "zip2 v16.16b, v20.16b, v19.16b\n"
      "str q18, [x20, #0x10]\n"
      "str q17, [x20, #0x20]\n"
      "str q16, [x20, #0x30]\n"
      "add x20, x20, %x[out_stride]\n"
      "bge 4b\n"
      "5:"  // Main row loop: Column loop skip
      "cmp x21, #0x1\n"
      "blt 7f\n"
      "6:"  // Main row loop: width 1 loop: loop
      "ldr b23, [x17], #0x1\n"
      "ldr b22, [x16], #0x1\n"
      "sub x21, x21, #0x1\n"
      "cmp x21, #0x1\n"
      "ldr b21, [x15], #0x1\n"
      "ldr b20, [x14], #0x1\n"
      "ldr b19, [x13], #0x1\n"
      "ldr b18, [x12], #0x1\n"
      "ldr b25, [x11], #0x1\n"
      "ldr b24, [x10], #0x1\n"
      "ldr b17, [x9], #0x1\n"
      "ldr b16, [x28], #0x1\n"
      "zip1 v23.16b, v23.16b, v17.16b\n"
      "zip1 v22.16b, v22.16b, v16.16b\n"
      "ldr b17, [x27], #0x1\n"
      "ldr b16, [x26], #0x1\n"
      "zip1 v21.16b, v21.16b, v17.16b\n"
      "zip1 v20.16b, v20.16b, v16.16b\n"
      "ldr b17, [x25], #0x1\n"
      "ldr b16, [x24], #0x1\n"
      "zip1 v19.16b, v19.16b, v17.16b\n"
      "zip1 v18.16b, v18.16b, v16.16b\n"
      "ldr b17, [x23], #0x1\n"
      "ldr b16, [x22], #0x1\n"
      "zip1 v17.16b, v25.16b, v17.16b\n"
      "zip1 v16.16b, v24.16b, v16.16b\n"
      "zip1 v19.16b, v23.16b, v19.16b\n"
      "zip1 v17.16b, v21.16b, v17.16b\n"
      "zip1 v18.16b, v22.16b, v18.16b\n"
      "zip1 v16.16b, v20.16b, v16.16b\n"
      "zip1 v17.16b, v19.16b, v17.16b\n"
      "zip1 v16.16b, v18.16b, v16.16b\n"
      "zip1 v16.16b, v17.16b, v16.16b\n"
      "str q16, [x20, #0x0]\n"
      "add x20, x20, #0x10\n"
      "bge 6b\n"
      "7:"  // Main row loop: width 1 loop: skip
      "cmp %x[height], #0x1\n"
      "add %x[out], %x[out], #0x40\n"
      "bge 1b\n"
      : [height] "+&r" (height), [in] "+&r" (in), [out] "+&r" (out)
      : [in_stride] "r" (in_stride), [out_stride] "r" (out_stride), [pad_row] "r" (pad_row), [width] "r" (width)
      : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
    );
}

} // anonymous namespace

template<>
void Transform<4, 16, true, VLType::None>(
    uint8_t *out, const uint8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_4_1x16(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(uint8_t) / 1,
        stride * sizeof(uint8_t),
        (kmax-k0)
    );
}

template<>
void Transform<4, 16, true, VLType::None>(
    int8_t *out, const int8_t *in, int stride, int x0, int xmax, int k0, int kmax)
{
    a64_transpose_interleave_4_1x16(
        reinterpret_cast<uint8_t *>(out),
        reinterpret_cast<const uint8_t *>(in + k0 * stride + x0),
        (xmax-x0) * sizeof(int8_t) / 1,
        stride * sizeof(int8_t),
        (kmax-k0)
    );
}

#endif
