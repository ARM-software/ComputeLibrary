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
#include "helpers_asymm.h"

#if defined(COLS_B)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_8bit and @ref gemm_transpose1x16 before running the matrix multiplication
 *
 * @attention The number of matrix B columns needs to be passed at compile time using -DCOLS_B
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
__kernel void gemmlowp_mm_interleaved_transposed(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
                                                 IMAGE_DECLARATION(dst))
{
    // src_addr.s0 = address of matrix A
    // src_addr.s1 = address of matrix B
    // Compute address for matrix A and B
    int2 src_addr = (int2)(get_global_id(1), get_global_id(0)) * (int2)((src0_stride_y),
                                                                        (src1_stride_y));

    // Add offset_first_element_in_bytes
    src_addr = src_addr + ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Compute end row address for matrix B
    int end_row_mtx_b = src_addr.s1 + COLS_B;

    // Reset accumulators
    int16 c00 = 0;
    int16 c10 = 0;
    int16 c20 = 0;
    int16 c30 = 0;

    for(; src_addr.s1 <= (end_row_mtx_b - 32); src_addr += (int2)(8, 32))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        int8 a0  = convert_int8(vload8(0, ((__global uchar *)src0_ptr) + src_addr.s0));
        int16 b0 = convert_int16(vload16(0, ((__global uchar *)src1_ptr) + src_addr.s1));

        c00 += (int16)a0.s0 * b0;
        c10 += (int16)a0.s1 * b0;
        c20 += (int16)a0.s2 * b0;
        c30 += (int16)a0.s3 * b0;

        int16 b1 = convert_int16(vload16(0, ((__global uchar *)src1_ptr) + src_addr.s1 + 16));

        c00 += (int16)a0.s4 * b1;
        c10 += (int16)a0.s5 * b1;
        c20 += (int16)a0.s6 * b1;
        c30 += (int16)a0.s7 * b1;
    }

    for(; src_addr.s1 < end_row_mtx_b; src_addr += (int2)(4, 16))
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        int4 a0  = convert_int4(vload4(0, ((__global uchar *)src0_ptr) + src_addr.s0));
        int16 b0 = convert_int16(vload16(0, ((__global uchar *)src1_ptr) + src_addr.s1));

        c00 += (int16)a0.s0 * b0;
        c10 += (int16)a0.s1 * b0;
        c20 += (int16)a0.s2 * b0;
        c30 += (int16)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Store 4x16 block
    vstore16(c00, 0, (__global int *)(offset(&dst, 0, 0)));
    vstore16(c10, 0, (__global int *)(offset(&dst, 0, 1)));
    vstore16(c20, 0, (__global int *)(offset(&dst, 0, 2)));
    vstore16(c30, 0, (__global int *)(offset(&dst, 0, 3)));
}
#endif // defined(COLS_B)

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
__kernel void gemmlowp_mm(IMAGE_DECLARATION(src0),
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

    int16 a_offset_s32 = (int16)0;
    int16 b_offset_s32 = (int16)0;

#if defined(A_OFFSET)
    Image sum_col = CONVERT_TO_IMAGE_STRUCT(sum_col);

    // Compute the offset contribution due to A_OFFSET
#if defined(SUM_COL_HAS_BATCHES)
    a_offset_s32 = vload16(0, (__global int *)(sum_col.ptr + get_global_id(2) * sum_col_stride_y));
#else  // defined(MATRIX_B_HAS_BATCHES)
    a_offset_s32 = vload16(0, (__global int *)(sum_col.ptr));
#endif // defined(MATRIX_B_HAS_BATCHES)

    a_offset_s32 *= (int16)A_OFFSET;
#endif // defined(A_OFFSET)

#if defined(B_OFFSET)
    Image sum_row = CONVERT_TO_IMAGE_STRUCT(sum_row);

    // Compute the offset contribution due to B_OFFSET
    b_offset_s32 = (int16) * (((__global int *)(sum_row.ptr + get_global_id(2) * sum_row_stride_y)) + get_global_id(1));
    b_offset_s32 *= (int16)B_OFFSET;
#endif // defined(B_OFFSET)

    const int16 offset_term_s32 = (int16)K_OFFSET + a_offset_s32 + b_offset_s32;

    int16 in_s32 = vload16(0, (__global int *)mm_result.ptr);

    // Add the offset terms to GEMM's result
    in_s32 += offset_term_s32;

    // Store the result with the offset contribution
    vstore16(in_s32, 0, (__global int *)mm_result.ptr);
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
