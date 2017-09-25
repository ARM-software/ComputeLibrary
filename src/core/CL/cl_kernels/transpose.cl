/*
 * Copyright (c) 2017 ARM Limited.
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
#include "helpers.h"

#define SWAP_ROW(u0, l0)     \
    ({                       \
        tmp_swap = u0;       \
        u0       = l0;       \
        l0       = tmp_swap; \
    })

#define SWAP_4x4(u0, u1, u2, u3, l0, l1, l2, l3) \
    ({                                           \
        VEC_DATA_TYPE(DATA_TYPE, 4)              \
        tmp_swap;                                \
        SWAP_ROW(u0, l0);                        \
        SWAP_ROW(u1, l1);                        \
        SWAP_ROW(u2, l2);                        \
        SWAP_ROW(u3, l3);                        \
    })

#define SWAP_8x8(u0, u1, u2, u3, u4, u5, u6, u7, l0, l1, l2, l3, l4, l5, l6, l7) \
    ({                                                                           \
        VEC_DATA_TYPE(DATA_TYPE, 8)                                              \
        tmp_swap;                                                                \
        SWAP_ROW(u0, l0);                                                        \
        SWAP_ROW(u1, l1);                                                        \
        SWAP_ROW(u2, l2);                                                        \
        SWAP_ROW(u3, l3);                                                        \
        SWAP_ROW(u4, l4);                                                        \
        SWAP_ROW(u5, l5);                                                        \
        SWAP_ROW(u6, l6);                                                        \
        SWAP_ROW(u7, l7);                                                        \
    })

#define TRANSPOSE_4x4(u0, u1, u2, u3) \
    ({                                \
        VEC_DATA_TYPE(DATA_TYPE, 4)   \
        tmp;                          \
        tmp.s012 = u0.s123;           \
        u0.s1    = u1.s0;             \
        u0.s2    = u2.s0;             \
        u0.s3    = u3.s0;             \
        u1.s0    = tmp.s0;            \
        u2.s0    = tmp.s1;            \
        u3.s0    = tmp.s2;            \
        \
        tmp.s01 = u1.s23;             \
        u1.s2   = u2.s1;              \
        u1.s3   = u3.s1;              \
        u2.s1   = tmp.s0;             \
        u3.s1   = tmp.s1;             \
        \
        tmp.s0 = u2.s3;               \
        u2.s3  = u3.s2;               \
        u3.s2  = tmp.s0;              \
    })

#define TRANSPOSE_8x8(u0, u1, u2, u3, u4, u5, u6, u7)                                             \
    ({                                                                                            \
        TRANSPOSE_4x4(u0.s0123, u1.s0123, u2.s0123, u3.s0123);                                    \
        TRANSPOSE_4x4(u0.s4567, u1.s4567, u2.s4567, u3.s4567);                                    \
        TRANSPOSE_4x4(u4.s0123, u5.s0123, u6.s0123, u7.s0123);                                    \
        TRANSPOSE_4x4(u4.s4567, u5.s4567, u6.s4567, u7.s4567);                                    \
        SWAP_4x4(u0.s4567, u1.s4567, u2.s4567, u3.s4567, u4.s0123, u5.s0123, u6.s0123, u7.s0123); \
    })

#define TRANSPOSE_16x16(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15)                                                \
    ({                                                                                                                                       \
        TRANSPOSE_8x8(u0.s01234567, u1.s01234567, u2.s01234567, u3.s01234567, u4.s01234567, u5.s01234567, u6.s01234567, u7.s01234567);       \
        TRANSPOSE_8x8(u0.s89ABCDEF, u1.s89ABCDEF, u2.s89ABCDEF, u3.s89ABCDEF, u4.s89ABCDEF, u5.s89ABCDEF, u6.s89ABCDEF, u7.s89ABCDEF);       \
        TRANSPOSE_8x8(u8.s01234567, u9.s01234567, u10.s01234567, u11.s01234567, u12.s01234567, u13.s01234567, u14.s01234567, u15.s01234567); \
        TRANSPOSE_8x8(u8.s89ABCDEF, u9.s89ABCDEF, u10.s89ABCDEF, u11.s89ABCDEF, u12.s89ABCDEF, u13.s89ABCDEF, u14.s89ABCDEF, u15.s89ABCDEF); \
        SWAP_8x8(u0.s89ABCDEF, u1.s89ABCDEF, u2.s89ABCDEF, u3.s89ABCDEF, u4.s89ABCDEF, u5.s89ABCDEF, u6.s89ABCDEF, u7.s89ABCDEF,             \
                 u8.s01234567, u9.s01234567, u10.s01234567, u11.s01234567, u12.s01234567, u13.s01234567, u14.s01234567, u15.s01234567);      \
    })

#ifndef DATA_TYPE_IN_BYTES
#error DATA_TYPE_IN_BYTES not set for the transpose OpenCL kernel
#endif /* not DATA_TYPE_IN_BYTES */

#undef VLOAD
#undef VSTORE

#if DATA_TYPE_IN_BYTES == 4
#define DATA_TYPE uint
#define TRANSPOSE() TRANSPOSE_4x4(u0, u1, u2, u3)
#define VLOAD(x, y) vload4(x, y)
#define VSTORE(x, y, z) vstore4(x, y, z)
#define BLOCK_SIZE 4
#elif DATA_TYPE_IN_BYTES == 2
#define DATA_TYPE ushort
#define TRANSPOSE() TRANSPOSE_8x8(u0, u1, u2, u3, u4, u5, u6, u7)
#define VLOAD(x, y) vload8(x, y)
#define VSTORE(x, y, z) vstore8(x, y, z)
#define BLOCK_SIZE 8
#elif DATA_TYPE_IN_BYTES == 1
#define DATA_TYPE uchar
#define TRANSPOSE() TRANSPOSE_16x16(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15)
#define VLOAD(x, y) vload16(x, y)
#define VSTORE(x, y, z) vstore16(x, y, z)
#define BLOCK_SIZE 16
#else /* switch DATA_TYPE_IN_BYTES */
#error DATA_TYPE_IN_BYTES not supported for transpose
#endif /* switch DATA_TYPE_IN_BYTES */

/** This OpenCL kernel computes the matrix transposition of input matrix
 *
 * @attention The number of bytes of the data type need to be passed at compile time using -DDATA_TYPE_IN_BYTES. DATA_TYPE_IN_BYTES can be:
 *  -# -DDATA_TYPE_IN_BYTES=1 for transposing U8 or S8 matrices
 *  -# -DDATA_TYPE_IN_BYTES=2 for transposing U16, S16 or FP16 matrices
 *  -# -DDATA_TYPE_IN_BYTES=4 for transposing U32, S32 or FP32 matrices
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U8/S8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void transpose(IMAGE_DECLARATION(src),
                        IMAGE_DECLARATION(dst))
{
    uint x = get_global_id(0) * BLOCK_SIZE;
    uint y = get_global_id(1) * BLOCK_SIZE;

    // Compute source address
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    // Load the NxN block at (x, y)
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u0 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 0)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u1 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 1)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u2 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 2)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u3 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 3)));
#if BLOCK_SIZE > 4
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u4 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 4)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u5 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 5)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u6 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 6)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u7 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 7)));
#if BLOCK_SIZE == 16
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u8 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 8)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u9 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 9)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u10 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 10)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u11 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 11)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u12 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 12)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u13 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 13)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u14 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 14)));
    VEC_DATA_TYPE(DATA_TYPE, BLOCK_SIZE)
    u15 = VLOAD(0, (__global DATA_TYPE *)(offset(&src, 0, 15)));
#endif /* BLOCK_SIZE == 16 */
#endif /* BLOCK_SIZE > 4 */

    // Transpose the block
    TRANSPOSE();

    // Store the block at (y, x)
    uint dst_offset_in_bytes = y * DATA_TYPE_IN_BYTES + x * dst_stride_y + dst_offset_first_element_in_bytes;
    VSTORE(u0, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 0 * dst_stride_y));
    VSTORE(u1, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 1 * dst_stride_y));
    VSTORE(u2, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 2 * dst_stride_y));
    VSTORE(u3, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 3 * dst_stride_y));
#if BLOCK_SIZE > 4
    VSTORE(u4, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 4 * dst_stride_y));
    VSTORE(u5, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 5 * dst_stride_y));
    VSTORE(u6, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 6 * dst_stride_y));
    VSTORE(u7, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 7 * dst_stride_y));
#if BLOCK_SIZE == 16
    VSTORE(u8, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 8 * dst_stride_y));
    VSTORE(u9, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 9 * dst_stride_y));
    VSTORE(u10, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 10 * dst_stride_y));
    VSTORE(u11, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 11 * dst_stride_y));
    VSTORE(u12, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 12 * dst_stride_y));
    VSTORE(u13, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 13 * dst_stride_y));
    VSTORE(u14, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 14 * dst_stride_y));
    VSTORE(u15, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_in_bytes + 15 * dst_stride_y));
#endif /* BLOCK_SIZE == 16 */
#endif /* BLOCK_SIZE > 4 */
}
