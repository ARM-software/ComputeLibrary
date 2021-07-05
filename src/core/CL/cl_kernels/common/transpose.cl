/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#define PARTIAL_STORE_M0 VEC_SIZE_LEFTOVER_X
#define PARTIAL_STORE_N0 VEC_SIZE_LEFTOVER_Y

#include "helpers.h"
#include "repeat.h"

#if defined(DATA_TYPE_IN_BYTES) && defined(VEC_SIZE_X) && defined(VEC_SIZE_LEFTOVER_X) && defined(VEC_SIZE_Y) && defined(VEC_SIZE_LEFTOVER_Y)

#if VEC_SIZE_X == 1
#if VEC_SIZE_Y == 1
#define TRANSPOSED_U(val) \
    {                     \
        u0                \
    }
#elif VEC_SIZE_Y == 2
#define TRANSPOSED_U(val) \
    {                     \
        u0, u1            \
    }
#elif VEC_SIZE_Y == 3
#define TRANSPOSED_U(val) \
    {                     \
        u0, u1, u2        \
    }
#elif VEC_SIZE_Y == 4
#define TRANSPOSED_U(val) \
    {                     \
        u0, u1, u2, u3    \
    }
#elif VEC_SIZE_Y == 8
#define TRANSPOSED_U(val)              \
    {                                  \
        u0, u1, u2, u3, u4, u5, u6, u7 \
    }
#elif VEC_SIZE_Y == 16
#define TRANSPOSED_U(val)                        \
    {                                            \
        u0, u1, u2, u3, u4, u5, u6, u7,          \
        u8, u9, u10, u11, u12, u13, u14, u15 \
    }
#endif /* switch VEC_SIZE_Y */
#else  // VEC_SIZE_X == 1
#if VEC_SIZE_Y == 1
#define TRANSPOSED_U(val) \
    {                     \
        u0.val            \
    }
#elif VEC_SIZE_Y == 2
#define TRANSPOSED_U(val) \
    {                     \
        u0.val, u1.val    \
    }
#elif VEC_SIZE_Y == 3
#define TRANSPOSED_U(val)      \
    {                          \
        u0.val, u1.val, u2.val \
    }
#elif VEC_SIZE_Y == 4
#define TRANSPOSED_U(val)              \
    {                                  \
        u0.val, u1.val, u2.val, u3.val \
    }
#elif VEC_SIZE_Y == 8
#define TRANSPOSED_U(val)                                              \
    {                                                                  \
        u0.val, u1.val, u2.val, u3.val, u4.val, u5.val, u6.val, u7.val \
    }
#elif VEC_SIZE_Y == 16
#define TRANSPOSED_U(val)                                                        \
    {                                                                            \
        u0.val, u1.val, u2.val, u3.val, u4.val, u5.val, u6.val, u7.val,          \
        u8.val, u9.val, u10.val, u11.val, u12.val, u13.val, u14.val, u15.val \
    }
#endif /* switch VEC_SIZE_Y */
#endif // VEC_SIZE_X == 1

#if DATA_TYPE_IN_BYTES == 4
#define DATA_TYPE uint
#elif DATA_TYPE_IN_BYTES == 2
#define DATA_TYPE ushort
#elif DATA_TYPE_IN_BYTES == 1
#define DATA_TYPE uchar
#else /* switch DATA_TYPE_IN_BYTES */
#error DATA_TYPE_IN_BYTES not supported for transpose
#endif /* switch DATA_TYPE_IN_BYTES */

/** This OpenCL kernel computes the matrix transposition of input matrix
 *
 * @note The number of bytes of the data type need to be passed at compile time using -DDATA_TYPE_IN_BYTES. DATA_TYPE_IN_BYTES can be:
 *  -# -DDATA_TYPE_IN_BYTES=1 for transposing U8 or S8 matrices
 *  -# -DDATA_TYPE_IN_BYTES=2 for transposing U16, S16 or FP16 matrices
 *  -# -DDATA_TYPE_IN_BYTES=4 for transposing U32, S32 or FP32 matrices
 *  -# -DVEC_SIZE_X is the number of elements processed in X dimension
 *  -# -DVEC_SIZE_LEFTOVER_X is the leftover size in the X dimension; x_dimension % VEC_SIZE_X
 *  -# -DVEC_SIZE_Y is the number of elements processed in Y dimension
 *  -# -DVEC_SIZE_LEFTOVER_Y is the leftover size in the Y dimension; y_dimension % VEC_SIZE_Y
 *
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: All
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
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE_X - (VEC_SIZE_X - VEC_SIZE_LEFTOVER_X) % VEC_SIZE_X), 0);
    uint y_offs = max((int)(get_global_id(1) * VEC_SIZE_Y - (VEC_SIZE_Y - VEC_SIZE_LEFTOVER_Y) % VEC_SIZE_Y), 0);

    // Compute addresses
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x_offs * DATA_TYPE_IN_BYTES + y_offs * src_stride_y;
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + y_offs * DATA_TYPE_IN_BYTES + x_offs * dst_stride_y;

    // Load the NxM block at (x, y)
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u0 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)src_addr);
#if VEC_SIZE_Y > 1
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u1 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + src_stride_y));
#endif /* VEC_SIZE_Y > 1 */
#if VEC_SIZE_Y > 2
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u2 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));
#endif /* VEC_SIZE_Y > 2 */
#if VEC_SIZE_Y > 3
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u3 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));
#endif /* VEC_SIZE_Y > 3 */
#if VEC_SIZE_Y > 4
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u4 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u5 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 5 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u6 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 6 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u7 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 7 * src_stride_y));
#endif /* VEC_SIZE_Y > 4 */
#if VEC_SIZE_Y > 8
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u8 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 8 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u9 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 9 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u10 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 10 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u11 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 11 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u12 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 12 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u13 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 13 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u14 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 14 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    u15 = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)(src_addr + 15 * src_stride_y));
#endif /* VEC_SIZE_Y > 8 */

    //Create transposed vectors
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t0 = TRANSPOSED_U(s0);
#if VEC_SIZE_X > 1
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t1 = TRANSPOSED_U(s1);
#endif /* VEC_SIZE_X > 1 */
#if VEC_SIZE_X > 2
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t2 = TRANSPOSED_U(s2);
#endif /* VEC_SIZE_X > 2 */
#if VEC_SIZE_X > 3
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t3 = TRANSPOSED_U(s3);
#endif /* VEC_SIZE_X > 3 */
#if VEC_SIZE_X > 4
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t4 = TRANSPOSED_U(s4);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t5 = TRANSPOSED_U(s5);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t6 = TRANSPOSED_U(s6);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t7 = TRANSPOSED_U(s7);
#endif /* VEC_SIZE_X > 4 */
#if VEC_SIZE_X > 8
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t8 = TRANSPOSED_U(s8);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    t9 = TRANSPOSED_U(s9);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    tA = TRANSPOSED_U(sA);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    tB = TRANSPOSED_U(sB);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    tC = TRANSPOSED_U(sC);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    tD = TRANSPOSED_U(sD);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    tE = TRANSPOSED_U(sE);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_Y)
    tF = TRANSPOSED_U(sF);
#endif /* VEC_SIZE_X > 8 */

    // Store the block at (y, x)
    REPEAT_VAR_INIT_TO_CONST(VEC_SIZE_X, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;
    STORE_BLOCK_BOUNDARY_AWARE(VEC_SIZE_X, VEC_SIZE_Y, DATA_TYPE, t, (__global uchar *)dst_addr, dst_stride_y, zout, VEC_SIZE_LEFTOVER_X, VEC_SIZE_LEFTOVER_Y, VEC_SIZE_LEFTOVER_X != 0
                               && get_global_id(0) == 0,
                               VEC_SIZE_LEFTOVER_Y != 0 && get_global_id(1) == 0);
}

#endif // defined(DATA_TYPE_IN_BYTES) && defined(VEC_SIZE_X) && defined(VEC_SIZE_LEFTOVER_X) && defined(VEC_SIZE_Y) && defined(VEC_SIZE_LEFTOVER_Y)