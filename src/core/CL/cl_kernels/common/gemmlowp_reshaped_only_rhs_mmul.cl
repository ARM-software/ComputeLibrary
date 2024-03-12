/*
 * Copyright (c) 2022 Arm Limited.
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
#include "activation_float_helpers.h"
#include "helpers.h"
#include "tile_helpers.h"
#if defined(GEMMLOWP_MM_RESHAPED_ONLY_RHS_MMUL)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices using the MMUL extension:
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is transposed
 *
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=1, -DK0=1).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=1)
 * @note The number of output columns processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_N0 (e.g., -DMMUL_N0=4)
 * @note The number of output rows processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_M0 (e.g., -DMMUL_M0=4)
 * @note The number of lhs columns (or rhs rows) processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_K0 (e.g., -DMMUL_K0=16)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 4
 *  - N0 = 1, 4, 8
 *  - K0 = 4
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 *
 * @param[in]  lhs_ptr                               Pointer to the LHS tensor. Supported data types: QASYMM8/QASYMM8_SIGNED
 * @param[in]  lhs_stride_y                          Stride of the LHS tensor in Y dimension (in bytes)
 * @param[in]  lhs_stride_z                          Stride of the LHS tensor in Z dimension (in bytes)
 * @param[in]  lhs_w                                 The size of the width dimension of the LHS tensor
 * @param[in]  lhs_h                                 The size of the height dimension of the LHS tensor
 * @param[in]  lhs_n                                 The size of the depth dimension of the LHS tensor
 * @param[in]  lhs_offset_first_element_in_bytes     The offset of the first element in the LHS tensor
 * @param[in]  rhs_ptr                               Pointer to the RHS reshaped tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                          Stride of the RHS tensor in Y dimension (in bytes)
 * @param[in]  rhs_stride_z                          Stride of the RHS tensor in Z dimension (in bytes)
 * @param[in]  rhs_w                                 The size of the width dimension of the RHS tensor
 * @param[in]  rhs_h                                 The size of the height dimension of the RHS tensor
 * @param[in]  rhs_n                                 The size of the depth dimension of the RHS tensor
 * @param[in]  rhs_offset_first_element_in_bytes     The offset of the first element in the RHS tensor
 * @param[in]  bia_ptr                               (Optional) Pointer to the bias tensor. Supported data type: S32
 * @param[in]  bia_stride_y                          (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bia_stride_z                          (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bia_w                                 (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bia_h                                 (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bia_n                                 (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bia_offset_first_element_in_bytes     (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                               Pointer to the destination tensor. Supported data type: same as @p lhs_ptr or S32
 * @param[in]  dst_stride_y                          Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                          Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                                 The size of the width dimension of the destination tensor
 * @param[in]  dst_h                                 The size of the height dimension of the destination tensor
 * @param[in]  dst_n                                 The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes     The offset of the first element in the destination tensor
 * @param[in]  M                                     Number of rows in LHS matrix not reshaped
 * @param[in]  N                                     Number of columns in RHS matrix not reshaped
 * @param[in]  K                                     Number of columns in LHS matrix and rows in RHS matrix not reshaped
 * @param[in]  sum_col_ptr                           (Optional) Pointer to the source tensor. Supported data type: S32
 * @param[in]  sum_col_stride_x                      (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_col_step_x                        (Optional) sum_col_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_col_stride_y                      (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_col_step_y                        (Optional) sum_col_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_col_offset_first_element_in_bytes (Optional) The offset of the first element in the source tensor
 * @param[in]  sum_row_ptr                           (Optional) Pointer to the source tensor. Supported data type: S32
 * @param[in]  sum_row_stride_x                      (Optional) Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_row_step_x                        (Optional) sum_row_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_row_stride_y                      (Optional) Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_row_step_y                        (Optional) sum_row_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_row_offset_first_element_in_bytes (Optional) The offset of the first element in the source tensor
 */
__kernel void gemmlowp_mm_reshaped_only_rhs_mmul(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#if defined(ADD_BIAS)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(ADD_BIAS)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K
#if defined(A_OFFSET)
    ,
    TENSOR3D_T(sum_col, BUFFER)
#endif // defined(A_OFFSET)
#if defined(B_OFFSET)
    ,
    TENSOR3D_T(sum_row, BUFFER)
#endif // defined(B_OFFSET)
)
{
#define MMUL_BLOCK_SIZE (MMUL_N0 * MMUL_M0)
#define VEC_SIZE 4 // For int8 types input to mmul instruction is a length 4 vector

    uint x0 = get_global_id(0);
    uint y0 = get_global_id(1);
    uint z  = get_global_id(2);

    // Get block ID and thread ID within the block
    uint block_id  = (x0 / MMUL_BLOCK_SIZE);
    uint thread_id = (x0 % MMUL_BLOCK_SIZE);

    // Coordinate within a block
    uint block_x = thread_id % MMUL_N0;
    uint block_y = (thread_id / MMUL_M0);

    // Starting destination coordinates
    uint dst_x = min(block_x * N0 + block_id * MMUL_N0 * N0, (uint)(N - 1));
    uint dst_y = min(block_y * M0 + y0 * M0 * MMUL_M0, (uint)(M - M0));

    uint lhs_x = VEC_SIZE * block_x;
    uint lhs_y = dst_y;

    uint rhs_x = VEC_SIZE * N0 * block_y;
    uint rhs_y = 4 * block_id + block_x;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(OUT_DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    TILE(ACC_DATA_TYPE, M0, N0, c);
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    for(int k = 0; k <= K - MMUL_K0; k += MMUL_K0)
    {
        TILE(DATA_TYPE, M0, VEC_SIZE, a);
        T_LOAD(DATA_TYPE, M0, VEC_SIZE, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);

        TILE(DATA_TYPE, N0, VEC_SIZE, b);
        T_LOAD(DATA_TYPE, N0, VEC_SIZE, BUFFER, rhs, 0, 0, 1, VEC_SIZE, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                VEC_TYPE vec_a = (VEC_TYPE)(a[m0].s[0], a[m0].s[1], a[m0].s[2], a[m0].s[3]);
                VEC_TYPE vec_b = (VEC_TYPE)(b[n0].s[0], b[n0].s[1], b[n0].s[2], b[n0].s[3]);
                c[m0].s[n0]    = arm_matrix_multiply(vec_a, vec_b, c[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_K0 * N0 * sizeof(DATA_TYPE);
    }

    if(block_x * N0 + block_id * MMUL_N0 * N0 >= N)
    {
        return;
    }

    if(block_y * M0 + y0 * M0 * MMUL_M0 >= M)
    {
        return;
    }

#if defined(FUSED_OUTPUT_STAGE_FIXED_POINT)

    TILE(int, M0, N0, offset_s32);
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        offset_s32[i].v = (VEC_DATA_TYPE(int, N0))K_OFFSET;
    })

#if defined(A_OFFSET)

    TILE(int, 1, N0, a_offset_s32);

    T_LOAD(int, 1, N0, BUFFER, sum_col, dst_x, z, 1, sum_col_stride_z, a_offset_s32);

    a_offset_s32[0].v *= A_OFFSET;

    T_ELTWISE_BROADCAST_ADD_X(int, M0, N0, offset_s32, a_offset_s32, offset_s32);
#endif // defined(A_OFFSET)

#if defined(B_OFFSET)

    TILE(int, M0, 1, b_offset_s32);

    T_LOAD(int, M0, 1, BUFFER, sum_row, dst_y, z * M, 1, 4, b_offset_s32);

    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        offset_s32[m0].v += b_offset_s32[m0].v *B_OFFSET;
    })

#endif // defined(B_OFFSET)

#if defined(ADD_BIAS)
#if defined(BROADCAST_BIAS)
    bia_offset_first_element_in_bytes += dst_x * sizeof(ACC_DATA_TYPE) + z * bia_stride_y;

    TILE(int, M0, N0, bias);

    T_LOAD(int, M0, N0, BUFFER, bia, dst_x, dst_y, 1, 1, bias);

    T_ADD(ACC_DATA_TYPE, M0, N0, offset_s32, bias, offset_s32);

#else // defined(BROADCAST_BIAS)
    bia_offset_first_element_in_bytes += dst_x * sizeof(ACC_DATA_TYPE);

    TILE(int, 1, N0, bias);

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        bias[0].v = VLOAD(N0)(0, (ACC_DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes));
    }
    else
    {
        VLOAD_PARTIAL(N0, N0_LEFTOVER)
        (bias[0].v, 0, (ACC_DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes));
    }

    T_ELTWISE_BROADCAST_ADD_X(int, M0, N0, offset_s32, bias, offset_s32);

#endif // defined(BROADCAST_BIAS)
#endif // defined(ADD_BIAS)

    T_ADD(ACC_DATA_TYPE, M0, N0, c, offset_s32, c);
    TILE(OUT_DATA_TYPE, M0, N0, c_lp);
    T_QUANTIZE8(ACC_DATA_TYPE, OUT_DATA_TYPE, PER_TENSOR, M0, N0, RESULT_OFFSET, RESULT_SHIFT, RESULT_MULTIPLIER, c, 0, 0, c_lp);

#if defined(MIN_BOUND)
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_lp[i].v = max(c_lp[i].v, (VEC_DATA_TYPE(OUT_DATA_TYPE, N0))MIN_BOUND);
    })
#endif // defined(MIN_BOUND)
#if defined(MAX_BOUND)
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_lp[i].v = min(c_lp[i].v, (VEC_DATA_TYPE(OUT_DATA_TYPE, N0))MAX_BOUND);
    })
#endif // defined(MAX_BOUND)

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, c, c);

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c_lp[m0].v, 0, (__global OUT_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE_PARTIAL(N0, N0_LEFTOVER)
                (c_lp[m0].v, 0, (__global OUT_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#else  // FUSED_OUTPUT_STAGE_FIXED_POINT
    // Store
    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global OUT_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE_PARTIAL(N0, N0_LEFTOVER)
                (c[m0].v, 0, (__global OUT_DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
#endif // FUSED_OUTPUT_STAGE_FIXED_POINT
}

#endif // defined(GEMMLOWP_MM_RESHAPED_ONLY_RHS_MMUL)
