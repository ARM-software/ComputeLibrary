/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#include "helpers_asymm.h"

#if defined(COLS_B) && defined(MULT_INTERLEAVE4X4_HEIGHT) && defined(TRANSPOSE1XW_WIDTH_STEP)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel before running the matrix multiplication
 *
 * @note The number of matrix B columns needs to be passed at compile time using -DCOLS_B: e.g. -DCOLS_B=1024
 * @note The transposition width step (mult_transpose1xW_width * 4) must be passed at compile time using -DTRANSPOSE1XW_WIDTH_STEP (i.e. -DTRANSPOSE1XW_WIDTH_STEP=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data type: QASYMM8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data type: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemmlowp_mm_interleaved_transposed_midgard(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
                                                         IMAGE_DECLARATION(dst))
{
    int x = get_global_id(0) / TRANSPOSE1XW_WIDTH_STEP;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % TRANSPOSE1XW_WIDTH_STEP) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    __global uchar *src_addr_a = (__global uchar *)(src0_ptr + y * src0_stride_y + src0_offset_first_element_in_bytes);
    __global uchar *src_addr_b = (__global uchar *)(src1_ptr + x * src1_stride_y + src1_offset_first_element_in_bytes);

    // Compute end row address for matrix B
    __global uchar *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    int4 c00 = 0;
    int4 c10 = 0;
    int4 c20 = 0;
    int4 c30 = 0;

    for(; src_addr_b <= (src_end_addr_b - (int)(8 * TRANSPOSE1XW_WIDTH_STEP)); src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 8 * TRANSPOSE1XW_WIDTH_STEP)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        int4 a0 = convert_int4(vload4(0, src_addr_a));
        int4 b0 = convert_int4(vload4(0, src_addr_b));

        c00 += (int4)a0.s0 * b0;
        c10 += (int4)a0.s1 * b0;
        c20 += (int4)a0.s2 * b0;
        c30 += (int4)a0.s3 * b0;

        a0 = convert_int4(vload4(0, src_addr_a + 4 * MULT_INTERLEAVE4X4_HEIGHT));
        b0 = convert_int4(vload4(0, src_addr_b + 4 * TRANSPOSE1XW_WIDTH_STEP));

        c00 += (int4)a0.s0 * b0;
        c10 += (int4)a0.s1 * b0;
        c20 += (int4)a0.s2 * b0;
        c30 += (int4)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += (4 * MULT_INTERLEAVE4X4_HEIGHT), src_addr_b += (4 * TRANSPOSE1XW_WIDTH_STEP))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        int4 a0 = convert_int4(vload4(0, src_addr_a));
        int4 b0 = convert_int4(vload4(0, src_addr_b));

        c00 += (int4)a0.s0 * b0;
        c10 += (int4)a0.s1 * b0;
        c20 += (int4)a0.s2 * b0;
        c30 += (int4)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Store 4x4 block
    vstore4(c00, 0, (__global int *)(offset(&dst, 0, 0)));
    vstore4(c10, 0, (__global int *)(offset(&dst, 0, 1)));
    vstore4(c20, 0, (__global int *)(offset(&dst, 0, 2)));
    vstore4(c30, 0, (__global int *)(offset(&dst, 0, 3)));
}

/** This OpenCL kernel is optimized for Bifrost and computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel before running the matrix multiplication
 *
 * @attention The number of matrix B columns needs to be passed at compile time using -DCOLS_B
 * @note The transposition width step (mult_transpose1xW_width * 4) must be passed at compile time using -DTRANSPOSE1XW_WIDTH_STEP (i.e. -DTRANSPOSE1XW_WIDTH_STEP=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data type: QASYMM8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data type: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemmlowp_mm_interleaved_transposed_bifrost(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
                                                         IMAGE_DECLARATION(dst))
{
    int x = get_global_id(0) / TRANSPOSE1XW_WIDTH_STEP;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % TRANSPOSE1XW_WIDTH_STEP) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    __global uchar *src_addr_a = (__global uchar *)(src0_ptr + y * src0_stride_y + src0_offset_first_element_in_bytes);
    __global uchar *src_addr_b = (__global uchar *)(src1_ptr + x * src1_stride_y + src1_offset_first_element_in_bytes);

    // Compute end row address for matrix B
    __global uchar *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    uint c00 = 0;
    uint c01 = 0;
    uint c02 = 0;
    uint c03 = 0;
    uint c10 = 0;
    uint c11 = 0;
    uint c12 = 0;
    uint c13 = 0;
    uint c20 = 0;
    uint c21 = 0;
    uint c22 = 0;
    uint c23 = 0;
    uint c30 = 0;
    uint c31 = 0;
    uint c32 = 0;
    uint c33 = 0;

#if MULT_INTERLEAVE4X4_HEIGHT == 1
    for(; src_addr_b <= (src_end_addr_b - (int)(32 * TRANSPOSE1XW_WIDTH_STEP)); src_addr_a += (32 * MULT_INTERLEAVE4X4_HEIGHT), src_addr_b += (32 * TRANSPOSE1XW_WIDTH_STEP))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        uchar16 a0 = vload16(0, src_addr_a);
        uchar4  b0 = vload4(0, src_addr_b);

        c00 += (ushort)a0.s0 * b0.s0;
        c01 += (ushort)a0.s0 * b0.s1;
        c02 += (ushort)a0.s0 * b0.s2;
        c03 += (ushort)a0.s0 * b0.s3;

        c10 += (ushort)a0.s1 * b0.s0;
        c11 += (ushort)a0.s1 * b0.s1;
        c12 += (ushort)a0.s1 * b0.s2;
        c13 += (ushort)a0.s1 * b0.s3;

        c20 += (ushort)a0.s2 * b0.s0;
        c21 += (ushort)a0.s2 * b0.s1;
        c22 += (ushort)a0.s2 * b0.s2;
        c23 += (ushort)a0.s2 * b0.s3;

        c30 += (ushort)a0.s3 * b0.s0;
        c31 += (ushort)a0.s3 * b0.s1;
        c32 += (ushort)a0.s3 * b0.s2;
        c33 += (ushort)a0.s3 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 4 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s4 * b0.s0;
        c01 += (ushort)a0.s4 * b0.s1;
        c02 += (ushort)a0.s4 * b0.s2;
        c03 += (ushort)a0.s4 * b0.s3;

        c10 += (ushort)a0.s5 * b0.s0;
        c11 += (ushort)a0.s5 * b0.s1;
        c12 += (ushort)a0.s5 * b0.s2;
        c13 += (ushort)a0.s5 * b0.s3;

        c20 += (ushort)a0.s6 * b0.s0;
        c21 += (ushort)a0.s6 * b0.s1;
        c22 += (ushort)a0.s6 * b0.s2;
        c23 += (ushort)a0.s6 * b0.s3;

        c30 += (ushort)a0.s7 * b0.s0;
        c31 += (ushort)a0.s7 * b0.s1;
        c32 += (ushort)a0.s7 * b0.s2;
        c33 += (ushort)a0.s7 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 8 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s8 * b0.s0;
        c01 += (ushort)a0.s8 * b0.s1;
        c02 += (ushort)a0.s8 * b0.s2;
        c03 += (ushort)a0.s8 * b0.s3;

        c10 += (ushort)a0.s9 * b0.s0;
        c11 += (ushort)a0.s9 * b0.s1;
        c12 += (ushort)a0.s9 * b0.s2;
        c13 += (ushort)a0.s9 * b0.s3;

        c20 += (ushort)a0.sA * b0.s0;
        c21 += (ushort)a0.sA * b0.s1;
        c22 += (ushort)a0.sA * b0.s2;
        c23 += (ushort)a0.sA * b0.s3;

        c30 += (ushort)a0.sB * b0.s0;
        c31 += (ushort)a0.sB * b0.s1;
        c32 += (ushort)a0.sB * b0.s2;
        c33 += (ushort)a0.sB * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 12 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.sC * b0.s0;
        c01 += (ushort)a0.sC * b0.s1;
        c02 += (ushort)a0.sC * b0.s2;
        c03 += (ushort)a0.sC * b0.s3;

        c10 += (ushort)a0.sD * b0.s0;
        c11 += (ushort)a0.sD * b0.s1;
        c12 += (ushort)a0.sD * b0.s2;
        c13 += (ushort)a0.sD * b0.s3;

        c20 += (ushort)a0.sE * b0.s0;
        c21 += (ushort)a0.sE * b0.s1;
        c22 += (ushort)a0.sE * b0.s2;
        c23 += (ushort)a0.sE * b0.s3;

        c30 += (ushort)a0.sF * b0.s0;
        c31 += (ushort)a0.sF * b0.s1;
        c32 += (ushort)a0.sF * b0.s2;
        c33 += (ushort)a0.sF * b0.s3;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload16(0, src_addr_a + 16);
        b0 = vload4(0, src_addr_b + 16 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s0 * b0.s0;
        c01 += (ushort)a0.s0 * b0.s1;
        c02 += (ushort)a0.s0 * b0.s2;
        c03 += (ushort)a0.s0 * b0.s3;

        c10 += (ushort)a0.s1 * b0.s0;
        c11 += (ushort)a0.s1 * b0.s1;
        c12 += (ushort)a0.s1 * b0.s2;
        c13 += (ushort)a0.s1 * b0.s3;

        c20 += (ushort)a0.s2 * b0.s0;
        c21 += (ushort)a0.s2 * b0.s1;
        c22 += (ushort)a0.s2 * b0.s2;
        c23 += (ushort)a0.s2 * b0.s3;

        c30 += (ushort)a0.s3 * b0.s0;
        c31 += (ushort)a0.s3 * b0.s1;
        c32 += (ushort)a0.s3 * b0.s2;
        c33 += (ushort)a0.s3 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 20 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s4 * b0.s0;
        c01 += (ushort)a0.s4 * b0.s1;
        c02 += (ushort)a0.s4 * b0.s2;
        c03 += (ushort)a0.s4 * b0.s3;

        c10 += (ushort)a0.s5 * b0.s0;
        c11 += (ushort)a0.s5 * b0.s1;
        c12 += (ushort)a0.s5 * b0.s2;
        c13 += (ushort)a0.s5 * b0.s3;

        c20 += (ushort)a0.s6 * b0.s0;
        c21 += (ushort)a0.s6 * b0.s1;
        c22 += (ushort)a0.s6 * b0.s2;
        c23 += (ushort)a0.s6 * b0.s3;

        c30 += (ushort)a0.s7 * b0.s0;
        c31 += (ushort)a0.s7 * b0.s1;
        c32 += (ushort)a0.s7 * b0.s2;
        c33 += (ushort)a0.s7 * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 24 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.s8 * b0.s0;
        c01 += (ushort)a0.s8 * b0.s1;
        c02 += (ushort)a0.s8 * b0.s2;
        c03 += (ushort)a0.s8 * b0.s3;

        c10 += (ushort)a0.s9 * b0.s0;
        c11 += (ushort)a0.s9 * b0.s1;
        c12 += (ushort)a0.s9 * b0.s2;
        c13 += (ushort)a0.s9 * b0.s3;

        c20 += (ushort)a0.sA * b0.s0;
        c21 += (ushort)a0.sA * b0.s1;
        c22 += (ushort)a0.sA * b0.s2;
        c23 += (ushort)a0.sA * b0.s3;

        c30 += (ushort)a0.sB * b0.s0;
        c31 += (ushort)a0.sB * b0.s1;
        c32 += (ushort)a0.sB * b0.s2;
        c33 += (ushort)a0.sB * b0.s3;

        // Load values from matrix B (transposed)
        b0 = vload4(0, src_addr_b + 28 * TRANSPOSE1XW_WIDTH_STEP);

        c00 += (ushort)a0.sC * b0.s0;
        c01 += (ushort)a0.sC * b0.s1;
        c02 += (ushort)a0.sC * b0.s2;
        c03 += (ushort)a0.sC * b0.s3;

        c10 += (ushort)a0.sD * b0.s0;
        c11 += (ushort)a0.sD * b0.s1;
        c12 += (ushort)a0.sD * b0.s2;
        c13 += (ushort)a0.sD * b0.s3;

        c20 += (ushort)a0.sE * b0.s0;
        c21 += (ushort)a0.sE * b0.s1;
        c22 += (ushort)a0.sE * b0.s2;
        c23 += (ushort)a0.sE * b0.s3;

        c30 += (ushort)a0.sF * b0.s0;
        c31 += (ushort)a0.sF * b0.s1;
        c32 += (ushort)a0.sF * b0.s2;
        c33 += (ushort)a0.sF * b0.s3;
    }
#endif // MULT_INTERLEAVE4X4_HEIGHT == 1

    for(; src_addr_b < src_end_addr_b; src_addr_a += (4 * MULT_INTERLEAVE4X4_HEIGHT), src_addr_b += (4 * TRANSPOSE1XW_WIDTH_STEP))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        uchar4 a0 = vload4(0, src_addr_a);
        uchar4 b0 = vload4(0, src_addr_b);

        c00 += (ushort)a0.s0 * b0.s0;
        c01 += (ushort)a0.s0 * b0.s1;
        c02 += (ushort)a0.s0 * b0.s2;
        c03 += (ushort)a0.s0 * b0.s3;

        c10 += (ushort)a0.s1 * b0.s0;
        c11 += (ushort)a0.s1 * b0.s1;
        c12 += (ushort)a0.s1 * b0.s2;
        c13 += (ushort)a0.s1 * b0.s3;

        c20 += (ushort)a0.s2 * b0.s0;
        c21 += (ushort)a0.s2 * b0.s1;
        c22 += (ushort)a0.s2 * b0.s2;
        c23 += (ushort)a0.s2 * b0.s3;

        c30 += (ushort)a0.s3 * b0.s0;
        c31 += (ushort)a0.s3 * b0.s1;
        c32 += (ushort)a0.s3 * b0.s2;
        c33 += (ushort)a0.s3 * b0.s3;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Store 4x4 block
    vstore4((int4)(c00, c01, c02, c03), 0, (__global int *)(offset(&dst, 0, 0)));
    vstore4((int4)(c10, c11, c12, c13), 0, (__global int *)(offset(&dst, 0, 1)));
    vstore4((int4)(c20, c21, c22, c23), 0, (__global int *)(offset(&dst, 0, 2)));
    vstore4((int4)(c30, c31, c32, c33), 0, (__global int *)(offset(&dst, 0, 3)));
}
#endif // defined(COLS_B) && defined(MULT_INTERLEAVE4X4_HEIGHT) && defined(TRANSPOSE1XW_WIDTH_STEP)

#if defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) && defined(COLS_A)
#define VECTOR_UCHAR VEC_DATA_TYPE(uchar, NUM_ELEMS_PROCESSED_PER_THREAD_X)
#define VECTOR_UINT VEC_DATA_TYPE(uint, NUM_ELEMS_PROCESSED_PER_THREAD_X)
#define VECTOR_INT VEC_DATA_TYPE(int, NUM_ELEMS_PROCESSED_PER_THREAD_X)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data type: QASYMM8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data type: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemmlowp_mm_midgard(IMAGE_DECLARATION(src0),
                                  IMAGE_DECLARATION(src1),
                                  IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx;

    int end_row_vec_a = src_addr.s0 + COLS_A;

    VECTOR_UINT acc0 = 0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VECTOR_UINT acc1 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VECTOR_UINT acc2 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VECTOR_UINT acc3 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    VECTOR_UINT acc4 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

    for(; src_addr.s0 <= (end_row_vec_a - 2); src_addr += (int2)(2, 2 * src1_stride_y))
    {
        // Load values from matrix A
        uchar2 a0 = vload2(0, src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar2 a1 = vload2(0, src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar2 a2 = vload2(0, src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar2 a3 = vload2(0, src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar2 a4 = vload2(0, src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        VECTOR_UCHAR b0 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src1_ptr + src_addr.s1);
        VECTOR_UCHAR b1 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src1_ptr + src_addr.s1 + src1_stride_y);

        // Accumulate
        acc0 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a0.s0;
        acc0 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a0.s1;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a1.s0;
        acc1 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a1.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a2.s0;
        acc2 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a2.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a3.s0;
        acc3 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a3.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        acc4 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a4.s0;
        acc4 += CONVERT(b1, VECTOR_UINT) * (VECTOR_UINT)a4.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(1, src1_stride_y))
    {
        // Load values from matrix A
        uchar a0 = *(src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar a1 = *(src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar a2 = *(src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar a3 = *(src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar a4 = *(src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        VECTOR_UCHAR b0 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src1_ptr + src_addr.s1);

        // Accumulate
        acc0 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a2;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        acc4 += CONVERT(b0, VECTOR_UINT) * (VECTOR_UINT)a4;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Store the result
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc0, VECTOR_INT), 0, (__global int *)(offset(&dst, 0, 0)));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc1, VECTOR_INT), 0, (__global int *)(offset(&dst, 0, 1)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc2, VECTOR_INT), 0, (__global int *)(offset(&dst, 0, 2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc3, VECTOR_INT), 0, (__global int *)(offset(&dst, 0, 3)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (CONVERT(acc4, VECTOR_INT), 0, (__global int *)(offset(&dst, 0, 4)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
}

/** OpenCL kernel optimized for Bifrost architectures that computes the matrix multiplication between matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data type: QASYMM8
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data type: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: S32
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemmlowp_mm_bifrost(IMAGE_DECLARATION(src0),
                                  IMAGE_DECLARATION(src1),
                                  IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx;

    int end_row_vec_a = src_addr.s0 + COLS_A;

    uint acc00 = 0;
    uint acc01 = 0;
    uint acc02 = 0;
    uint acc03 = 0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    uint acc10 = 0;
    uint acc11 = 0;
    uint acc12 = 0;
    uint acc13 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    uint acc20 = 0;
    uint acc21 = 0;
    uint acc22 = 0;
    uint acc23 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    uint acc30 = 0;
    uint acc31 = 0;
    uint acc32 = 0;
    uint acc33 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    uint acc40 = 0;
    uint acc41 = 0;
    uint acc42 = 0;
    uint acc43 = 0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4

    for(; src_addr.s0 <= (end_row_vec_a - 4); src_addr += (int2)(4, 4 * src1_stride_y))
    {
        // Load values from matrix A
        uchar4 a0 = vload4(0, src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar4 a1 = vload4(0, src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar4 a2 = vload4(0, src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar4 a3 = vload4(0, src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar4 a4 = vload4(0, src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        uchar4 b0 = vload4(0, src1_ptr + src_addr.s1 + 0 * src1_stride_y);
        uchar4 b1 = vload4(0, src1_ptr + src_addr.s1 + 1 * src1_stride_y);
        uchar4 b2 = vload4(0, src1_ptr + src_addr.s1 + 2 * src1_stride_y);
        uchar4 b3 = vload4(0, src1_ptr + src_addr.s1 + 3 * src1_stride_y);

        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a0.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a0.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a0.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a0.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a0.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a0.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a0.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a0.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a0.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a0.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a0.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a0.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a0.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a0.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a0.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a0.s3;

            acc00 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc01 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc02 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc03 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a1.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a1.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a1.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a1.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a1.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a1.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a1.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a1.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a1.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a1.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a1.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a1.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a1.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a1.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a1.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a1.s3;

            acc10 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc11 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc12 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc13 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a2.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a2.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a2.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a2.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a2.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a2.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a2.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a2.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a2.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a2.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a2.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a2.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a2.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a2.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a2.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a2.s3;

            acc20 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc21 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc22 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc23 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a3.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a3.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a3.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a3.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a3.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a3.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a3.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a3.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a3.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a3.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a3.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a3.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a3.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a3.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a3.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a3.s3;

            acc30 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc31 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc32 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc33 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a4.s0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a4.s0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a4.s0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a4.s0;

            ushort tmp4 = (ushort)b1.s0 * (ushort)a4.s1;
            ushort tmp5 = (ushort)b1.s1 * (ushort)a4.s1;
            ushort tmp6 = (ushort)b1.s2 * (ushort)a4.s1;
            ushort tmp7 = (ushort)b1.s3 * (ushort)a4.s1;

            ushort tmp8 = (ushort)b2.s0 * (ushort)a4.s2;
            ushort tmp9 = (ushort)b2.s1 * (ushort)a4.s2;
            ushort tmpA = (ushort)b2.s2 * (ushort)a4.s2;
            ushort tmpB = (ushort)b2.s3 * (ushort)a4.s2;

            ushort tmpC = (ushort)b3.s0 * (ushort)a4.s3;
            ushort tmpD = (ushort)b3.s1 * (ushort)a4.s3;
            ushort tmpE = (ushort)b3.s2 * (ushort)a4.s3;
            ushort tmpF = (ushort)b3.s3 * (ushort)a4.s3;

            acc40 += ((uint)tmp0 + (uint)tmp4 + (uint)tmp8 + (uint)tmpC);
            acc41 += ((uint)tmp1 + (uint)tmp5 + (uint)tmp9 + (uint)tmpD);
            acc42 += ((uint)tmp2 + (uint)tmp6 + (uint)tmpA + (uint)tmpE);
            acc43 += ((uint)tmp3 + (uint)tmp7 + (uint)tmpB + (uint)tmpF);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(1, src1_stride_y))
    {
        // Load values from matrix A
        uchar a0 = *(src0_ptr + src_addr.s0 + 0 * src0_stride_y);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        uchar a1 = *(src0_ptr + src_addr.s0 + 1 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        uchar a2 = *(src0_ptr + src_addr.s0 + 2 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        uchar a3 = *(src0_ptr + src_addr.s0 + 3 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        uchar a4 = *(src0_ptr + src_addr.s0 + 4 * src0_stride_y);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        // Load values from matrix B
        uchar4 b0 = vload4(0, src1_ptr + src_addr.s1);

        // Accumulate
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a0;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a0;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a0;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a0;

            acc00 += ((uint)tmp0);
            acc01 += ((uint)tmp1);
            acc02 += ((uint)tmp2);
            acc03 += ((uint)tmp3);
        }
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a1;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a1;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a1;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a1;

            acc10 += ((uint)tmp0);
            acc11 += ((uint)tmp1);
            acc12 += ((uint)tmp2);
            acc13 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a2;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a2;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a2;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a2;

            acc20 += ((uint)tmp0);
            acc21 += ((uint)tmp1);
            acc22 += ((uint)tmp2);
            acc23 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a3;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a3;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a3;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a3;

            acc30 += ((uint)tmp0);
            acc31 += ((uint)tmp1);
            acc32 += ((uint)tmp2);
            acc33 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
        {
            // Accumulate
            ushort tmp0 = (ushort)b0.s0 * (ushort)a4;
            ushort tmp1 = (ushort)b0.s1 * (ushort)a4;
            ushort tmp2 = (ushort)b0.s2 * (ushort)a4;
            ushort tmp3 = (ushort)b0.s3 * (ushort)a4;

            acc40 += ((uint)tmp0);
            acc41 += ((uint)tmp1);
            acc42 += ((uint)tmp2);
            acc43 += ((uint)tmp3);
        }
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Store the result
    vstore4((int4)(acc00, acc01, acc02, acc03), 0, (__global int *)(offset(&dst, 0, 0)));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((int4)(acc10, acc11, acc12, acc13), 0, (__global int *)(offset(&dst, 0, 1)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((int4)(acc20, acc21, acc22, acc23), 0, (__global int *)(offset(&dst, 0, 2)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((int4)(acc30, acc31, acc32, acc33), 0, (__global int *)(offset(&dst, 0, 3)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
    vstore4((int4)(acc40, acc41, acc42, acc43), 0, (__global int *)(offset(&dst, 0, 4)));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 4
}
#endif // defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_Y) && defined(COLS_A)

#if defined(COLS_A)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each row of Matrix A.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix A columns needs to be passed at compile time using -DCOLS_A
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_matrix_a_reduction(TENSOR3D_DECLARATION(src),
                                          IMAGE_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Image    dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uint4 sum_row_u32 = (uint4)0;
    uint  sum_row     = 0;

    __global const uchar *matrix_a = (__global const uchar *)(src.ptr + get_global_id(0) * src_stride_y + get_global_id(1) * src_stride_z);

    int i = 0;

    // This for loop performs 16 accumulations
    for(; i <= ((int)COLS_A - 16); i += 16)
    {
        const uchar16 a0_u8 = vload16(0, matrix_a + i);

        sum_row_u32 += convert_uint4(a0_u8.s0123) + convert_uint4(a0_u8.s4567) + convert_uint4(a0_u8.s89AB) + convert_uint4(a0_u8.sCDEF);
    }

    // This for loop performs the leftover accumulations
    for(; i < COLS_A; ++i)
    {
        sum_row += matrix_a[i];
    }

    sum_row += sum_row_u32.s0 + sum_row_u32.s1 + sum_row_u32.s2 + sum_row_u32.s3;

    *((__global int *)dst.ptr) = (int)sum_row;
}
#endif // defined(COLS_A)

#if defined(COLS_B) && defined(ROWS_B)
/** OpenCL kernel used to compute the row-vectors of sums of all the entries in each column of Matrix B.
 *
 * @note This stage is needed to handle the offset of matrix product
 *       https://github.com/google/gemmlowp/blob/master/doc/low-precision.md
 *
 * @attention The number of matrix B columns and rows needs to be passed at compile time using -DCOLS_B and -DROWS_B
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data type: QASYMM8
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor Supported data type: S32
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_matrix_b_reduction(TENSOR3D_DECLARATION(src),
                                          IMAGE_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Image    dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uint16 sum_col_u32 = (uint16)0;

    __global const uchar *matrix_b = (__global const uchar *)(src.ptr + get_global_id(1) * src_stride_z);

    int i = 0;
    // This for loop performs 4 accumulations
    for(; i <= ((int)ROWS_B - 4); i += 4)
    {
        const uchar16 b0_u8 = vload16(0, matrix_b + 0 * src_stride_y);
        const uchar16 b1_u8 = vload16(0, matrix_b + 1 * src_stride_y);
        const uchar16 b2_u8 = vload16(0, matrix_b + 2 * src_stride_y);
        const uchar16 b3_u8 = vload16(0, matrix_b + 3 * src_stride_y);

        sum_col_u32 += convert_uint16(b0_u8) + convert_uint16(b1_u8) + convert_uint16(b2_u8) + convert_uint16(b3_u8);

        matrix_b += 4 * src_stride_y;
    }

    // This for loop perfoms the leftover accumulations
    for(; i < (int)ROWS_B; ++i)
    {
        const uchar16 b0_u8 = vload16(0, matrix_b);

        sum_col_u32 += convert_uint16(b0_u8);

        matrix_b += src_stride_y;
    }

    vstore16(convert_int16(sum_col_u32), 0, (__global int *)dst.ptr);
}
#endif // defined(COLS_B) && defined(ROWS_B)

#if defined(K_OFFSET)
/* OpenCL kernel used to add the offset contribution after @ref CLGEMMLowpMatrixMultiplyKernel. The computation is performed in-place
 *
 * This kernel takes a final int32 accumulator value (the output of @CLGEMMLowpMatrixMultiplyKernel),
 * and adds to it the offset contribution of matrix A and matrix B in-place.
 *
 * @attention The k_offset = a_offset * b_offset * k (where k is the number of matrix A columns) needs to be passed at compile time using -DK_OFFSET (i.e. -DK_OFFSET=1200)
 * @note In case the offset contribution due to a_offset is required, a_offset needs to be passed at compile time using -DA_OFFSET (i.e. -DA_OFFSET=1)
 * @note In case the offset contribution due to b_offset is required, b_offset needs to be passed at compile time using -DB_OFFSET (i.e. -DB_OFFSET=6)
 * @note In case sum_col has batches, -DSUM_COL_HAS_BATCHES must be passed at compile time. Usually if gemmlowp is used to accelerate convolution layer, sum_col will not have batches
 *
 * The final result is:
 *
 * mm_result[i][k] = mm_result[i][k] +
 *                   (sum_col[k] * A_OFFSET) +
 *                   (sum_row[i] * B_OFFSET) +
 *                   (K_OFFSET)
 *
 * @param[in] mm_result_ptr                                Pointer to the source tensor. Supported data type: S32
 * @param[in] mm_result_stride_x                           Stride of the source tensor in X dimension (in bytes)
 * @param[in] mm_result_step_x                             mm_result_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] mm_result_stride_y                           Stride of the source tensor in Y dimension (in bytes)
 * @param[in] mm_result_step_y                             mm_result_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] mm_result_stride_z                           Stride of the source tensor in Z dimension (in bytes)
 * @param[in] mm_result_step_z                             mm_result_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] mm_result_offset_first_element_in_bytes      The offset of the first element in the source tensor
 * @param[in] sum_col_result_ptr                           Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in] sum_col_result_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in] sum_col_result_step_x                        sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] sum_col_result_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] sum_col_result_step_y                        sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] sum_col_result_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] sum_row_result_ptr                           Pointer to the source tensor. Supported data type: same as @p mm_result_ptr
 * @param[in] sum_row_result_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in] sum_row_result_step_x                        sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] sum_row_result_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] sum_row_result_step_y                        sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] sum_row_result_offset_first_element_in_bytes The offset of the first element in the source tensor
 */
__kernel void gemmlowp_offset_contribution(TENSOR3D_DECLARATION(mm_result)
#if defined(A_OFFSET)
                                           ,
                                           IMAGE_DECLARATION(sum_col)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
                                           ,
                                           IMAGE_DECLARATION(sum_row)
#endif // defined(B_OFFSET)
                                          )
{
    Tensor3D mm_result = CONVERT_TO_TENSOR3D_STRUCT(mm_result);

    int4 a_offset_s32 = (int4)0;
    int4 b_offset_s32 = (int4)0;

#if defined(A_OFFSET)
    Image sum_col = CONVERT_TO_IMAGE_STRUCT(sum_col);

    // Compute the offset contribution due to A_OFFSET
#if defined(SUM_COL_HAS_BATCHES)
    a_offset_s32 = vload4(0, (__global int *)(sum_col.ptr + get_global_id(2) * sum_col_stride_y));
#else  // defined(MATRIX_B_HAS_BATCHES)
    a_offset_s32 = vload4(0, (__global int *)(sum_col.ptr));
#endif // defined(MATRIX_B_HAS_BATCHES)

    a_offset_s32 *= (int4)A_OFFSET;
#endif // defined(A_OFFSET)

#if defined(B_OFFSET)
    Image sum_row = CONVERT_TO_IMAGE_STRUCT(sum_row);

    // Compute the offset contribution due to B_OFFSET
    b_offset_s32 = (int4) * (((__global int *)(sum_row.ptr + get_global_id(2) * sum_row_stride_y)) + get_global_id(1));
    b_offset_s32 *= (int4)B_OFFSET;
#endif // defined(B_OFFSET)

    const int4 offset_term_s32 = (int4)K_OFFSET + a_offset_s32 + b_offset_s32;

    int4 in_s32 = vload4(0, (__global int *)mm_result.ptr);

    // Add the offset terms to GEMM's result
    in_s32 += offset_term_s32;

    // Store the result with the offset contribution
    vstore4(in_s32, 0, (__global int *)mm_result.ptr);
}
#endif // defined(K_OFFSET)

#if defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Add offset terms to final result
 *  -# Multiply each entry of result by result_mult_int
 *  -# Add bias to final result (if -DADD_BIAS is passed at compile time)
 *  -# Shift the int32 accumulator by result_shift
 *  -# Clamp the value between the specified min and max bounds (if -DMIN_BOUND and/or -DMAX_BOUND are passed at compile time)
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data type: S32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in]  biases_ptr                           Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                      Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                        biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes The offset of the first element in the biases tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_output_stage_quantize_down(TENSOR3D_DECLARATION(src),
#if defined(ADD_BIAS)
                                                  VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                  TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
#if defined(ADD_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);
#endif // defined(ADD_BIAS)

    int16 input_values = vload16(0, (__global int *)src.ptr);

    // Add the offset terms to GEMM's result
    input_values += (int16)RESULT_OFFSET;

#if defined(ADD_BIAS)
    // Add bias
    const int16 biases_values = vload16(0, (__global int *)biases.ptr);
    input_values += (int16)biases_values;
#endif // defined(ADD_BIAS)

    // Multiply by result_mult_int and shift
    input_values *= RESULT_MULT_INT;

    input_values >>= RESULT_SHIFT;

    uchar16 res = convert_uchar16_sat(input_values);

#if defined(MIN_BOUND)
    res = max(res, (uchar16)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar16)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore16(res, 0, dst.ptr);
}
#endif // defined(RESULT_OFFSET) && defined(RESULT_MULT_INT) && defined(RESULT_SHIFT)

#if defined(RESULT_OFFSET_AFTER_SHIFT) && defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)
/** This OpenCL kernel is used to quantize down the int32 accumulator values of GEMMLowp to QASYMM8
 *
 * This kernel takes a final int32 accumulator value (the output of @ref CLGEMMLowpMatrixMultiplyKernel), and processes it to obtain the final QASYMM8 value.
 * The following computations will be performed by the kernel:
 *
 *  -# Compute fixed point multiplication between each entry of input by result_fixedpoint_multiplier
 *  -# Add bias to final result if bias tensor is not a nullptr
 *  -# Round to nearest division by a power-of-two using result_shift
 *  -# Add offset to each result
 *  -# Clamp the value between the specified min and max bounds
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to QASYMM8.
 *
 * @attention The offset, scalar scale factor and number of bits to shift right of output tensor must be passed at compile time using -DRESULT_OFFSET, -RESULT_MULT_INT and -DRESULT_SHIFT
 *
 * @note In case the addition of int32 biases is required, -DADD_BIAS should be passed at compile time
 * @note In case the clamping of the result is required, the min and max bounds can be passed at compile time using -DMIN_BOUND and -DMAX_BOUND.
 *       These values can be used to implement "rectified linear unit" activation functions
 *
 * @param[in]  src_ptr                              Pointer to the source tensor. Supported data type: S32
 * @param[in]  src_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[in]  biases_ptr                           Pointer to the biases tensor. Supported data type: same as @p src_ptr
 * @param[in]  biases_stride_x                      Stride of the biases tensor in X dimension (in bytes)
 * @param[in]  biases_step_x                        biases_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  biases_offset_first_element_in_bytes The offset of the first element in the biases tensor
 * @param[out] dst_ptr                              Pointer to the destination tensor Supported data type: QASYMM8
 * @param[in]  dst_stride_x                         Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                           dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                         Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                           dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                           src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes    The offset of the first element in the destination tensor
 */
__kernel void gemmlowp_output_stage_quantize_down_fixedpoint(TENSOR3D_DECLARATION(src),
#if defined(ADD_BIAS)
                                                             VECTOR_DECLARATION(biases),
#endif // defined(ADD_BIAS)
                                                             TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
#if defined(ADD_BIAS)
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);
#endif // defined(ADD_BIAS)

    int16 input_values = vload16(0, (__global int *)src.ptr);

#if defined(ADD_BIAS)
    // Add bias
    const int16 biases_values = vload16(0, (__global int *)biases.ptr);
    input_values += (int16)biases_values;
#endif // defined(ADD_BIAS)

    // Multiply by result_mult_int and shift
    input_values = ASYMM_MULT_BY_QUANT_MULTIPLIER_LESS_THAN_ONE(input_values, RESULT_FIXEDPOINT_MULTIPLIER, RESULT_SHIFT, 16);

    // Add the offset terms to GEMM's result
    input_values += (int16)RESULT_OFFSET_AFTER_SHIFT;

    uchar16 res = convert_uchar16_sat(input_values);

#if defined(MIN_BOUND)
    res = max(res, (uchar16)MIN_BOUND);
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    res = min(res, (uchar16)MAX_BOUND);
#endif // defined(MAX_BOUND)

    // Store the result
    vstore16(res, 0, dst.ptr);
}
#endif // defined(RESULT_OFFSET_AFTER_SHIFT) && defined(RESULT_FIXEDPOINT_MULTIPLIER) && defined(RESULT_SHIFT)
