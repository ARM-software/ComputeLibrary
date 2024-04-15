/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_MMUL)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices using the MMUL extension:
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is NOT transposed
 *
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of output columns processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_N0 (e.g., -DMMUL_N0=2)
 * @note The number of output rows processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_M0 (e.g., -DMMUL_M0=2)
 * @note The number of lhs columns (or rhs rows) processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_K0 (e.g., -DMMUL_K0=2)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 *
 * @param[in]  lhs_ptr                           Pointer to the LHS tensor. Supported data types: F16/F32
 * @param[in]  lhs_stride_y                      Stride of the LHS tensor in Y dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the LHS tensor in Z dimension (in bytes)
 * @param[in]  lhs_w                             The size of the width dimension of the LHS tensor
 * @param[in]  lhs_h                             The size of the height dimension of the LHS tensor
 * @param[in]  lhs_n                             The size of the depth dimension of the LHS tensor
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the LHS tensor
 * @param[in]  rhs_ptr                           Pointer to the RHS reshaped tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the RHS tensor in Y dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the RHS tensor in Z dimension (in bytes)
 * @param[in]  rhs_w                             The size of the width dimension of the RHS tensor
 * @param[in]  rhs_h                             The size of the height dimension of the RHS tensor
 * @param[in]  rhs_n                             The size of the depth dimension of the RHS tensor
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the RHS tensor
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  bia_stride_y                      (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bia_stride_z                      (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bia_w                             (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bia_h                             (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bia_n                             (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                             The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  M                                 Number of rows in LHS matrix not reshaped
 * @param[in]  N                                 Number of columns in RHS matrix not reshaped
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped
 */
__kernel void gemm_mm_reshaped_only_rhs_nt_mmul(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#if defined(BETA)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(BETA)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_N0 * MMUL_K0)

    uint x0 = get_global_id(0); // (N / N0) * MMUL_K0
    uint y0 = get_global_id(1); // (M / M0) / MMUL_M0
    uint z  = get_global_id(2); // Batch

    // Get block ID and thread ID within the block
    uint block_id  = (x0 / MMUL_BLOCK_SIZE);
    uint thread_id = (x0 % MMUL_BLOCK_SIZE);

    // Coordinate within a block
    uint block_x = thread_id % MMUL_N0;
    uint block_y = (thread_id / MMUL_M0);

    // Starting destination coordinates
    uint dst_x = min(block_x * N0 + block_id * MMUL_N0 * N0, (uint)(N - 1));
    uint dst_y = min(block_y * M0 + y0 * M0 * MMUL_M0, (uint)(M - M0));

    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results

    // Starting LHS coordinates
    uint lhs_x = block_x;
    uint lhs_y = dst_y;

    // Starting RHS coordinates
    uint rhs_x = block_y * N0 * MMUL_N0 + block_x * N0;
    uint rhs_y = block_id;

    // Compute LHS/RHS/DST matrix address
#ifdef REINTERPRET_INPUT_AS_3D
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + (lhs_y + z * M) * lhs_stride_y;
#else // REINTERPRET_INPUT_AS_3D
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
#endif // REINTERPRET_INPUT_AS_3D

#ifdef BATCHED_RHS
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
#else // BATCHED_RHS
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y;
#endif // BATCHED_RHS

#ifdef REINTERPRET_OUTPUT_AS_3D
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + (dst_y + z * M) * dst_stride_y;
#else // REINTERPRET_OUTPUT_AS_3D
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;
#endif // REINTERPRET_OUTPUT_AS_3D

    // Note: If RHS derives from the weights of convolution 2d layer, RHS will always be 2D and rhs_stride_z will always be equal to 0 for
    // not sliding the tensor

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

#if !defined(HALF_PRECISION)
#define c c_f32
#endif // !defined(HALF_PRECISION)

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k <= K - MMUL_K0; k += MMUL_K0)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, 0, 0, 1, 0, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[m0].s[0], b[0].s[n0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_K0 * MMUL_N0 * N0 * sizeof(DATA_TYPE);
    }

    if(block_x * N0 + block_id * MMUL_N0 * N0 >= N)
    {
        return;
    }

    if(block_y * M0 + y0 * M0 * MMUL_M0 >= M)
    {
        return;
    }

#if defined(HALF_PRECISION)
    TILE(DATA_TYPE, M0, N0, c);

    // Conversion required for the half precision
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            c[m0].s[n0] = c_f32[m0].s[n0];
        })
    })
#endif // defined(HALF_PRECISION)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, c, (DATA_TYPE)ALPHA, c);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    bia_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE);

    TILE(DATA_TYPE, 1, N0, bias0);

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        bias0[0].v = VLOAD(N0)(0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes));
    }
    else
    {
        VLOAD_PARTIAL(N0, N0_LEFTOVER)
        (bias0[0].v, 0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes));
    }

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, 1, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_X(V_ADD, DATA_TYPE, M0, N0, c, bias0, c);
#else // defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, M0, N0, bias0);

    bia_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * bia_stride_y + z * bia_stride_z;

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                bias0[m0].v = VLOAD(N0)(0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes + m0 * bia_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VLOAD_PARTIAL(N0, N0_LEFTOVER)
                (bias0[m0].v, 0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes + m0 * bia_stride_y));
            }
        })
    }

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias
    T_ADD(DATA_TYPE, M0, N0, c, bias0, c);
    // c = c + bias
#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, c, c);

    // Store
    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
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
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_MM_RESHAPED_ONLY_RHS_MMUL)

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_MMUL_TEXTURE)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices using the MMUL extension and the OpenCL image for RHS:
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is NOT transposed
 *
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of output columns processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_N0 (e.g., -DMMUL_N0=2)
 * @note The number of output rows processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_M0 (e.g., -DMMUL_M0=2)
 * @note The number of lhs columns (or rhs rows) processed by the the cooperative mmul extension must be passed at compile time using -DMMUL_K0 (e.g., -DMMUL_K0=2)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 *
 * @param[in]  lhs_ptr                           Pointer to the LHS tensor. Supported data types: F16/F32
 * @param[in]  lhs_stride_y                      Stride of the LHS tensor in Y dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the LHS tensor in Z dimension (in bytes)
 * @param[in]  lhs_w                             The size of the width dimension of the LHS tensor
 * @param[in]  lhs_h                             The size of the height dimension of the LHS tensor
 * @param[in]  lhs_n                             The size of the depth dimension of the LHS tensor
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the LHS tensor
 * @param[in]  rhs_ptr                           Pointer to the RHS reshaped tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the RHS tensor in Y dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the RHS tensor in Z dimension (in bytes)
 * @param[in]  rhs_w                             The size of the width dimension of the RHS tensor
 * @param[in]  rhs_h                             The size of the height dimension of the RHS tensor
 * @param[in]  rhs_n                             The size of the depth dimension of the RHS tensor
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the RHS tensor
 * @param[in]  bia_ptr                           (Optional) Pointer to the bias tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  bia_stride_y                      (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bia_stride_z                      (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bia_w                             (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bia_h                             (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bia_n                             (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bia_offset_first_element_in_bytes (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                             The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  M                                 Number of rows in LHS matrix not reshaped
 * @param[in]  N                                 Number of columns in RHS matrix not reshaped
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped
 */
__kernel void gemm_mm_reshaped_only_rhs_nt_mmul_texture(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, IMAGE),
#if defined(BETA)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(BETA)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_N0 * MMUL_K0)

    uint x0 = get_global_id(0); // (N / N0) * MMUL_K0
    uint y0 = get_global_id(1); // (M / M0) / MMUL_M0
    uint z  = get_global_id(2); // Batch

    // Get block ID and thread ID within the block
    uint block_id  = (x0 / MMUL_BLOCK_SIZE);
    uint thread_id = (x0 % MMUL_BLOCK_SIZE);

    // Coordinate within a block
    uint block_x = thread_id % MMUL_N0;
    uint block_y = (thread_id / MMUL_M0);

    // Starting destination coordinates
    uint dst_x = min(block_x * N0 + block_id * MMUL_N0 * N0, (uint)(N - 1));
    uint dst_y = min(block_y * M0 + y0 * M0 * MMUL_M0, (uint)(M - M0));

    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results

    // Starting LHS coordinates
    uint lhs_x = block_x;
    uint lhs_y = dst_y;

    // Starting RHS coordinates
    uint rhs_x = block_y * N0 * MMUL_N0 + block_x * N0;

#ifdef BATCHED_RHS
    uint rhs_y = block_id + z * rhs_h;
#else // BATCHED_RHS
    uint rhs_y = block_id;
#endif // BATCHED_RHS

    // Compute LHS/RHS/DST matrix address
#ifdef REINTERPRET_INPUT_AS_3D
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + (lhs_y + z * M) * lhs_stride_y;
#else // REINTERPRET_INPUT_AS_3D
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
#endif // REINTERPRET_INPUT_AS_3D

#ifdef REINTERPRET_OUTPUT_AS_3D
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + (dst_y + z * M) * dst_stride_y;
#else // REINTERPRET_OUTPUT_AS_3D
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;
#endif // REINTERPRET_OUTPUT_AS_3D

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

#if !defined(HALF_PRECISION)
#define c c_f32
#endif // !defined(HALF_PRECISION)

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k <= K - MMUL_K0; k += MMUL_K0)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, IMAGE, rhs, rhs_x, rhs_y, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[m0].s[0], b[0].s[n0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_x += MMUL_K0 * MMUL_N0 * N0;
    }

    if(block_x * N0 + block_id * MMUL_N0 * N0 >= N)
    {
        return;
    }

    if(block_y * M0 + y0 * M0 * MMUL_M0 >= M)
    {
        return;
    }

#if defined(HALF_PRECISION)
    TILE(DATA_TYPE, M0, N0, c);

    // Conversion required for the half precision
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            c[m0].s[n0] = c_f32[m0].s[n0];
        })
    })
#endif // defined(HALF_PRECISION)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, c, (DATA_TYPE)ALPHA, c);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    bia_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE);

    TILE(DATA_TYPE, 1, N0, bias0);

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        bias0[0].v = VLOAD(N0)(0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes));
    }
    else
    {
        VLOAD_PARTIAL(N0, N0_LEFTOVER)
        (bias0[0].v, 0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes));
    }

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, 1, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_X(V_ADD, DATA_TYPE, M0, N0, c, bias0, c);
#else // defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, M0, N0, bias0);

    bia_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * bia_stride_y + z * bia_stride_z;

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                bias0[m0].v = VLOAD(N0)(0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes + m0 * bia_stride_y));
            }
        })
    }
    else
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VLOAD_PARTIAL(N0, N0_LEFTOVER)
                (bias0[m0].v, 0, (DATA_TYPE *)(bia_ptr + bia_offset_first_element_in_bytes + m0 * bia_stride_y));
            }
        })
    }

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias
    T_ADD(DATA_TYPE, M0, N0, c, bias0, c);
    // c = c + bias
#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, c, c);

    // Store
    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
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
                (c[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_MM_RESHAPED_ONLY_RHS_MMUL_TEXTURE)
