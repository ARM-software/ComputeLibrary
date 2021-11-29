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

// *INDENT-OFF*
// clang-format off
#if defined(GEMM_MM_RESHAPED_ONLY_RHS_T)
//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops:
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is transposed
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - H0 >= 1
 *
 * @note In case of post ops, the following information must be passed at compile time:
 * @note -DPOST_OP1, -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 1
 * @note -DPOST_OP2: The arithmetic addition post op to perform at slot 2
 * @note -DPOST_OP3, -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
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
 * @param[in]  ex0_ptr                           (Optional) Pointer to the tensor added with POST_OP2. Supported data type: same as @p lhs_ptr
 * @param[in]  ex0_stride_y                      (Optional) Stride of the tensor added with POST_OP2 in Y dimension (in bytes)
 * @param[in]  ex0_stride_z                      (Optional) Stride of the tensor added with POST_OP2 in Z dimension (in bytes)
 * @param[in]  ex0_w                             (Optional) The size of the width dimension of the tensor added with POST_OP2
 * @param[in]  ex0_h                             (Optional) The size of the height dimension of the tensor added with POST_OP2
 * @param[in]  ex0_n                             (Optional) The size of the depth dimension of the tensor added with POST_OP2
 * @param[in]  ex0_offset_first_element_in_bytes (Optional) The offset of the first element in the tensor added with POST_OP2
 * @param[out] dst_ptr                           (Optional) Pointer to the destination tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      (Optional) Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                             (Optional) The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             (Optional) The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             (Optional) The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 * @param[in]  M                                 Number of rows in LHS matrix not reshaped
 * @param[in]  N                                 Number of columns in RHS matrix not reshaped
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped
 */
//! @endcond
__kernel void gemm_mm_reshaped_only_rhs_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#if defined(BETA)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(BETA)
#if defined(POST_OP2)
    TENSOR3D_T(ex0, BUFFER),
#endif // defined(POST_OP_ADD)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K
)
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (K0)
#define RHS_STEP_X ((K0) * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (K0)
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    const uint x = GET_SPATIAL_IDX(0, N0, 0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

#if defined(DUMMY_WORK_ITEMS)
    if((x >= N) || (y >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    bool x_cond = PARTIAL_STORE_N0 != 0 && ((x + N0) > N);
    bool y_cond = PARTIAL_STORE_M0 != 0 && y == 0;

    TILE(uint, M0, 1, dst_indirect_y);
    INITIALIZE_INDIRECT_Y(M0, PARTIAL_STORE_M0, y_cond, dst_indirect_y);

    const uint x_rhs = x / N0;

    lhs_offset_first_element_in_bytes += y * (uint)lhs_stride_y + z * (uint)lhs_stride_y * M;
    rhs_offset_first_element_in_bytes += (x_rhs % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x_rhs / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset_first_element_in_bytes += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset_first_element_in_bytes += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    int i = 0;
    for(; i <= (K - K0); i+=K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, N0, K0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, K0, BUFFER, rhs, 0, 0, 1, RHS_STEP_X * sizeof(DATA_TYPE), b);

        // Compute the matrix multiplication between the two tiles
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, T, a, b, c);

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += (N0 * RHS_STEP_X * RHS_STEP_LOOP) * sizeof(DATA_TYPE);
    }
#if defined(RUN_LEFTOVER_K0)
    for(; i < K; ++i)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, N0, 1, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, 1, BUFFER, rhs, 0, 0, 1, RHS_STEP_X * sizeof(DATA_TYPE), b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, T, a, b, c);

        lhs_offset_first_element_in_bytes += sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += sizeof(DATA_TYPE);
    }
#endif // defined(RUN_LEFTOVER_K0)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, c, (DATA_TYPE)ALPHA, c);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, 1, N0, bias0);

    T_LOAD_WIDTH_SELECT(DATA_TYPE, 1, N0, PARTIAL_STORE_N0, BUFFER, bia, x, 0, 0, x_cond, bias0);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, 1, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, M0, N0, c, bias0, c);
#else // defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, M0, N0, bias0);

    bia_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * bia_stride_y) + (z * bia_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, bia, 0, bia_stride_y, x_cond, bias0, dst_indirect_y);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias
    T_ADD(DATA_TYPE, M0, N0, c, bias0, c);
    // c = c + bias
#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(POST_OP1)
    T_ACTIVATION(DATA_TYPE, M0, N0, P1_ACTIVATION_TYPE, P1_ACTIVATION_A_VAL, P1_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP1)

#if defined(POST_OP2)
    TILE(DATA_TYPE, M0, N0, extra0);

    ex0_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * ex0_stride_y) + (z * ex0_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, ex0, 0, ex0_stride_y, x_cond, extra0, dst_indirect_y);

    T_ADD(DATA_TYPE, M0, N0, c, extra0, c);
#endif // defined(POST_OP2)

#if defined(POST_OP3)
    T_ACTIVATION(DATA_TYPE, M0, N0, P3_ACTIVATION_TYPE, P3_ACTIVATION_A_VAL, P3_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP3)

    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * dst_stride_y) + (z * dst_stride_y * M);

    // Store the tile in reverse order so that the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, c, dst_indirect_y);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_RESHAPED_RHS_ONLY_T)

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_T_TEXTURE)
//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops:
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is transposed
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - H0 >= 1
 *
 * @note In case of post ops, the following information must be passed at compile time:
 * @note -DPOST_OP1, -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 1
 * @note -DPOST_OP2: The arithmetic addition post op to perform at slot 2
 * @note -DPOST_OP3, -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
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
 * @param[in]  ex0_ptr                           (Optional) Pointer to the tensor added with POST_OP2. Supported data type: same as @p lhs_ptr
 * @param[in]  ex0_stride_y                      (Optional) Stride of the tensor added with POST_OP2 in Y dimension (in bytes)
 * @param[in]  ex0_stride_z                      (Optional) Stride of the tensor added with POST_OP2 in Z dimension (in bytes)
 * @param[in]  ex0_w                             (Optional) The size of the width dimension of the tensor added with POST_OP2
 * @param[in]  ex0_h                             (Optional) The size of the height dimension of the tensor added with POST_OP2
 * @param[in]  ex0_n                             (Optional) The size of the depth dimension of the tensor added with POST_OP2
 * @param[in]  ex0_offset_first_element_in_bytes (Optional) The offset of the first element in the tensor added with POST_OP2
 * @param[out] dst_ptr                           (Optional) Pointer to the destination tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      (Optional) Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                             (Optional) The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             (Optional) The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             (Optional) The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 * @param[in]  M                                 Number of rows in LHS matrix not reshaped
 * @param[in]  N                                 Number of columns in RHS matrix not reshaped
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped
 */
//! @endcond
__kernel void gemm_mm_reshaped_only_rhs_t_texture(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, IMAGE),
#if defined(BETA)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(BETA)
#if defined(POST_OP2)
    TENSOR3D_T(ex0, BUFFER),
#endif // defined(POST_OP_ADD)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K
)
{
    // Block size
#define RHS_BLOCK_SIZE (K0 * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (K0)
#define RHS_STEP_X (K0 * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X K0
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    const uint x = GET_SPATIAL_IDX(0, N0, 0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

#if defined(DUMMY_WORK_ITEMS)
    if((x >= N) || (y >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    bool x_cond = PARTIAL_STORE_N0 != 0 && ((x + N0) > N);
    bool y_cond = PARTIAL_STORE_M0 != 0 && y == 0;

    TILE(uint, M0, 1, dst_indirect_y);
    INITIALIZE_INDIRECT_Y(M0, PARTIAL_STORE_M0, y_cond, dst_indirect_y);

    lhs_offset_first_element_in_bytes += y * (uint)lhs_stride_y + z * lhs_stride_y * M;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = z % MATRIX_B_DEPTH;
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = z;
#endif // defined(MATRIX_B_DEPTH)

    uint x_rhs = ((x / N0) % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = ((x / N0) / H0) + z_rhs * rhs_h;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    TILE(DATA_TYPE, M0, K0, a);
    TILE(DATA_TYPE, N0, K0, b);

    int i = 0;
    for(; i <= (K - K0); i+=K0)
    {
        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD_DILATED(DATA_TYPE, N0, K0, IMAGE, rhs, x_rhs, y_rhs, RHS_STEP_X, 0, 1, b);

        // Compute the matrix multiplication between the two tiles
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, T, a, b, c);

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
        x_rhs += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }
#if defined(RUN_LEFTOVER_K0)
    T_LOAD_DILATED(DATA_TYPE, N0, K0, IMAGE, rhs, x_rhs, y_rhs, RHS_STEP_X, 0, 1, b);

        LOOP_UNROLLING(int, k0, 0, 1, PARTIAL_K,
        {
            LOOP_UNROLLING(int, m0, 0, 1, M0,
            {
                DATA_TYPE a0 = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset_first_element_in_bytes + m0 * lhs_stride_y);
                LOOP_UNROLLING(int, n0, 0, 1, N0,
                {
                    c[m0].s[n0] += a0 * b[n0].s[k0];
                })
            })
            lhs_offset_first_element_in_bytes += sizeof(DATA_TYPE);
        })
#endif // defined(RUN_LEFTOVER_K0)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, c, (DATA_TYPE)ALPHA, c);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, 1, N0, bias0);

    T_LOAD_WIDTH_SELECT(DATA_TYPE, 1, N0, PARTIAL_STORE_N0, BUFFER, bia, x, 0, 0, x_cond, bias0);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, 1, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, M0, N0, c, bias0, c);
#else // defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, M0, N0, bias0);

    bia_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * bia_stride_y) + (z * bia_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, bia, 0, bia_stride_y, x_cond, bias0, dst_indirect_y);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias
    T_ADD(DATA_TYPE, M0, N0, c, bias0, c);
    // c = c + bias
#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(POST_OP1)
    T_ACTIVATION(DATA_TYPE, M0, N0, P1_ACTIVATION_TYPE, P1_ACTIVATION_A_VAL, P1_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP1)

#if defined(POST_OP2)
    TILE(DATA_TYPE, M0, N0, extra0);

    ex0_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * ex0_stride_y) + (z * ex0_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, ex0, 0, ex0_stride_y, x_cond, extra0, dst_indirect_y);

    T_ADD(DATA_TYPE, M0, N0, c, extra0, c);
#endif // defined(POST_OP2)

#if defined(POST_OP3)
    T_ACTIVATION(DATA_TYPE, M0, N0, P3_ACTIVATION_TYPE, P3_ACTIVATION_A_VAL, P3_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP3)

    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_y * M;

    // Store the tile in reverse order so that the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, c, dst_indirect_y);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_RESHAPED_RHS_ONLY_T_TEXTURE)

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_NT)
//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops:
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is not transposed
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - H0 >= 1
 *
 * @note In case of post ops, the following information must be passed at compile time:
 * @note -DPOST_OP1, -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 1
 * @note -DPOST_OP2: The arithmetic addition post op to perform at slot 2
 * @note -DPOST_OP3, -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
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
 * @param[in]  ex0_ptr                           (Optional) Pointer to the tensor added with POST_OP2. Supported data type: same as @p lhs_ptr
 * @param[in]  ex0_stride_y                      (Optional) Stride of the tensor added with POST_OP2 in Y dimension (in bytes)
 * @param[in]  ex0_stride_z                      (Optional) Stride of the tensor added with POST_OP2 in Z dimension (in bytes)
 * @param[in]  ex0_w                             (Optional) The size of the width dimension of the tensor added with POST_OP2
 * @param[in]  ex0_h                             (Optional) The size of the height dimension of the tensor added with POST_OP2
 * @param[in]  ex0_n                             (Optional) The size of the depth dimension of the tensor added with POST_OP2
 * @param[in]  ex0_offset_first_element_in_bytes (Optional) The offset of the first element in the tensor added with POST_OP2
 * @param[out] dst_ptr                           (Optional) Pointer to the destination tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      (Optional) Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                             (Optional) The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             (Optional) The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             (Optional) The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 * @param[in]  M                                 Number of rows in LHS matrix not reshaped
 * @param[in]  N                                 Number of columns in RHS matrix not reshaped
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped
 */
//! @endcond
__kernel void gemm_mm_reshaped_only_rhs_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#if defined(BETA)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(BETA)
#if defined(POST_OP2)
    TENSOR3D_T(ex0, BUFFER),
#endif // defined(POST_OP_ADD)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K
)
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (N0)
#define RHS_STEP_X ((N0) * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (N0)
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    const uint x = GET_SPATIAL_IDX(0, N0, 0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

#if defined(DUMMY_WORK_ITEMS)
    if((x >= N) || (y >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    bool x_cond = PARTIAL_STORE_N0 != 0 && ((x + N0) > N);
    bool y_cond = PARTIAL_STORE_M0 != 0 && y == 0;

    TILE(uint, M0, 1, dst_indirect_y);
    INITIALIZE_INDIRECT_Y(M0, PARTIAL_STORE_M0, y_cond, dst_indirect_y);

    const uint x_rhs = x / N0;

    lhs_offset_first_element_in_bytes += y * (uint)lhs_stride_y + z * lhs_stride_y * M;
    rhs_offset_first_element_in_bytes += (x_rhs % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x_rhs / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset_first_element_in_bytes += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset_first_element_in_bytes += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    int i = 0;
    for(; i <= (K - K0); i+=K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, BUFFER, rhs, 0, 0, 1, RHS_STEP_X * sizeof(DATA_TYPE), b);

        // Compute the matrix multiplication between the two tiles
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, c);

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += K0 * RHS_STEP_X * RHS_STEP_LOOP * sizeof(DATA_TYPE);
    }
#if defined(RUN_LEFTOVER_K0)
    for(; i < K; ++i)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, 0, 0, 1, RHS_STEP_X * sizeof(DATA_TYPE), b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, a, b, c);

        lhs_offset_first_element_in_bytes += sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += RHS_STEP_X * sizeof(DATA_TYPE);
    }
#endif // defined(RUN_LEFTOVER_K0)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, c, (DATA_TYPE)ALPHA, c);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, 1, N0, bias0);

    T_LOAD_WIDTH_SELECT(DATA_TYPE, 1, N0, PARTIAL_STORE_N0, BUFFER, bia, x, 0, 0, x_cond, bias0);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, 1, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, M0, N0, c, bias0, c);
#else // defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, M0, N0, bias0);

    bia_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * bia_stride_y) + (z * bia_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, bia, 0, bia_stride_y, x_cond, bias0, dst_indirect_y);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias
    T_ADD(DATA_TYPE, M0, N0, c, bias0, c);
    // c = c + bias
#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(POST_OP1)
    T_ACTIVATION(DATA_TYPE, M0, N0, P1_ACTIVATION_TYPE, P1_ACTIVATION_A_VAL, P1_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP1)

#if defined(POST_OP2)
    TILE(DATA_TYPE, M0, N0, extra0);

    ex0_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * ex0_stride_y) + (z * ex0_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, ex0, 0, ex0_stride_y, x_cond, extra0, dst_indirect_y);

    T_ADD(DATA_TYPE, M0, N0, c, extra0, c);
#endif // defined(POST_OP2)

#if defined(POST_OP3)
    T_ACTIVATION(DATA_TYPE, M0, N0, P3_ACTIVATION_TYPE, P3_ACTIVATION_A_VAL, P3_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP3)

    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_y * M;

    // Store the tile in reverse order so that the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, c, dst_indirect_y);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_RESHAPED_RHS_ONLY_NT)

#if defined(GEMM_MM_RESHAPED_ONLY_RHS_NT_TEXTURE)
//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the matrix multiplication between 2 matrices plus 3 post ops:
 * Post op 1: activation (optional)
 * Post op 2: elementwise op
 * Post op 3: activation (optional)
 *
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref ClGemmMatrixMultiplyReshapedOnlyRhsKernel and the block K0xN0 is not transposed
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - H0 >= 1
 *
 * @note In case of post ops, the following information must be passed at compile time:
 * @note -DPOST_OP1, -DP1_ACTIVATION_TYPE, -DP1_ACTIVATION_A_VAL, -DP1_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 1
 * @note -DPOST_OP2: The arithmetic addition post op to perform at slot 2
 * @note -DPOST_OP3, -DP3_ACTIVATION_TYPE, -DP3_ACTIVATION_A_VAL, -DP3_ACTIVATION_B_VAL: The activation type, alpha and beta values of the activation post op at slot 3
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
 * @param[in]  ex0_ptr                           (Optional) Pointer to the tensor added with POST_OP2. Supported data type: same as @p lhs_ptr
 * @param[in]  ex0_stride_y                      (Optional) Stride of the tensor added with POST_OP2 in Y dimension (in bytes)
 * @param[in]  ex0_stride_z                      (Optional) Stride of the tensor added with POST_OP2 in Z dimension (in bytes)
 * @param[in]  ex0_w                             (Optional) The size of the width dimension of the tensor added with POST_OP2
 * @param[in]  ex0_h                             (Optional) The size of the height dimension of the tensor added with POST_OP2
 * @param[in]  ex0_n                             (Optional) The size of the depth dimension of the tensor added with POST_OP2
 * @param[in]  ex0_offset_first_element_in_bytes (Optional) The offset of the first element in the tensor added with POST_OP2
 * @param[out] dst_ptr                           (Optional) Pointer to the destination tensor. Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_stride_z                      (Optional) Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_w                             (Optional) The size of the width dimension of the destination tensor
 * @param[in]  dst_h                             (Optional) The size of the height dimension of the destination tensor
 * @param[in]  dst_n                             (Optional) The size of the depth dimension of the destination tensor
 * @param[in]  dst_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 * @param[in]  M                                 Number of rows in LHS matrix not reshaped
 * @param[in]  N                                 Number of columns in RHS matrix not reshaped
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped
 */
//! @endcond
__kernel void gemm_mm_reshaped_only_rhs_nt_texture(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, IMAGE),
#if defined(BETA)
    TENSOR3D_T(bia, BUFFER),
#endif // defined(BETA)
#if defined(POST_OP2)
    TENSOR3D_T(ex0, BUFFER),
#endif // defined(POST_OP_ADD)
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K
)
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (N0)
#define RHS_STEP_X ((N0) * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (N0)
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    const uint x = GET_SPATIAL_IDX(0, N0, 0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

#if defined(DUMMY_WORK_ITEMS)
    if((x >= N) || (y >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    bool x_cond = PARTIAL_STORE_N0 != 0 && ((x + N0) > N);
    bool y_cond = PARTIAL_STORE_M0 != 0 && y == 0;

    TILE(uint, M0, 1, dst_indirect_y);
    INITIALIZE_INDIRECT_Y(M0, PARTIAL_STORE_M0, y_cond, dst_indirect_y);

    lhs_offset_first_element_in_bytes += y * (uint)lhs_stride_y + z * lhs_stride_y * M;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = z % MATRIX_B_DEPTH;
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = z;
#endif // defined(MATRIX_B_DEPTH)

    uint x_rhs = ((x / N0) % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = ((x / N0) / H0) + z_rhs * rhs_h;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, c);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = 0;
    })

    int i = 0;
    for(; i <= (K - K0); i+=K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD_DILATED(DATA_TYPE, K0, N0, IMAGE, rhs, x_rhs, y_rhs, RHS_STEP_X, 0, 1, b);

        // Compute the matrix multiplication between the two tiles
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, c);

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
        x_rhs += K0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

#if defined(RUN_LEFTOVER_K0)
    for(; i < K; ++i)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD_DILATED(DATA_TYPE, 1, N0, IMAGE, rhs, x_rhs, y_rhs, RHS_STEP_X, 0, 1, b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, a, b, c);

        lhs_offset_first_element_in_bytes += sizeof(DATA_TYPE);
        x_rhs += RHS_STEP_X;
    }
#endif // defined(RUN_LEFTOVER_K0)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, c, (DATA_TYPE)ALPHA, c);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, 1, N0, bias0);

    T_LOAD_WIDTH_SELECT(DATA_TYPE, 1, N0, PARTIAL_STORE_N0, BUFFER, bia, x, 0, 0, x_cond, bias0);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, 1, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    T_ADD_BROADCAST_X(DATA_TYPE, M0, N0, c, bias0, c);
#else // defined(BROADCAST_BIAS)
    TILE(DATA_TYPE, M0, N0, bias0);

    bia_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * bia_stride_y) + (z * bia_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, bia, 0, bia_stride_y, x_cond, bias0, dst_indirect_y);

#ifndef UNIT_BETA
    T_SCALE_CONSTANT(DATA_TYPE, M0, N0, bias0, (DATA_TYPE)BETA, bias0);
#endif // UNIT_BIAS

    // c = c + bias
    T_ADD(DATA_TYPE, M0, N0, c, bias0, c);
    // c = c + bias
#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(POST_OP1)
    T_ACTIVATION(DATA_TYPE, M0, N0, P1_ACTIVATION_TYPE, P1_ACTIVATION_A_VAL, P1_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP1)

#if defined(POST_OP2)
    TILE(DATA_TYPE, M0, N0, extra0);

    ex0_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + (y * ex0_stride_y) + (z * ex0_stride_y * M);

    T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, ex0, 0, ex0_stride_y, x_cond, extra0, dst_indirect_y);

    T_ADD(DATA_TYPE, M0, N0, c, extra0, c);
#endif // defined(POST_OP2)

#if defined(POST_OP3)
    T_ACTIVATION(DATA_TYPE, M0, N0, P3_ACTIVATION_TYPE, P3_ACTIVATION_A_VAL, P3_ACTIVATION_B_VAL, c, c);
#endif // defined(POST_OP3)

    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_y * M;

    // Store the tile in reverse order so that the invalid values are overwritten with the valid ones
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, c, dst_indirect_y);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(GEMM_RESHAPED_RHS_ONLY_NT_TEXTURE)