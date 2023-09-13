/*
 * Copyright (c) 2023 Arm Limited.
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

#ifdef BIAS
// This function performs in-place bias addition for integer datatype when bias is enabled.
// Note The tile's dimensions used for the LHS and RHS matrices (M0, N0) must be passed at compile time using -DN0, -DM0 (e.g. -DN0=8, -DM0=4).
inline void perform_bias_addition(uchar *bias_ptr, uint bias_offset_first_element_in_bytes, TILE(int, M0, N0, acc), uint x)
{
    TILE(int, 1, N0, bias_tile);

    // below expands to use bias_ptr and bias_offset_first_element_in_bytes
    T_LOAD(int, 1, N0, BUFFER, bias, x, 0, 1, 0, bias_tile);

    // c = c + bias[broadcasted]
    T_ELTWISE_BROADCAST_ADD_X(int, M0, N0, acc, bias_tile, acc);
}
#endif // defined(BIAS)

#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0) // MMUL block size for the output matrix

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=uchar)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at
 *       compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER
 *       (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The dimensions M, N, K must be passed at compile time using -DK (e.g. -DM=5, -DN=8, -DK=6).
 *       K must be a multiple of 16.
 * @note MMUL block sizes must be passed at compile time using -DMMUL_K0, -DMMUL_M0, -DMMUL_N0
 *       (e.g. -DMMUL_K0=16, -DMMUL_M0=4, -DMMUL_N0=4)
 * @note If there is bias -DBIAS option must be passed at compile time
 * @note Quantization offsets of lhs, rhs and dst tensors must be passed at compile time using -DLHS_OFFSET,
 *       -DRHS_OFFSET, -DDST_OFFSET (e.g. -DLHS_OFFSET=10, -DRHS_OFFSET=0, -DDST_OFFSET=-6)
 * @note Effective quantization multiplier and shift for the destination tensor must be passed at compile time using
 *       -DDST_MULTIPLIER and -DDST_SHIFT (e.g. -DDST_MULTIPLIER=2091, -DST_SHIFT=8)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_QUANTIZED_MMUL_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 4
 * @note For a generic view on how the MMUL works, see mat_mul_mmul.cl
 *
 * @param[in]  lhs_ptr                            Pointer to the lhs matrix. Supported data types: QASYMM8_SIGNED/QASYMM8
 * @param[in]  lhs_stride_y                       Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                       Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                              The width of the lhs tensor
 * @param[in]  lhs_h                              The height of the lhs tensor
 * @param[in]  lhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                            Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                       Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                              The width of the rhs tensor
 * @param[in]  rhs_h                              The height of the rhs tensor
 * @param[in]  rhs_n                              Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the rhs matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias tensor. Supported data type: S32
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias tensor in Y dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias tensor in Z dimension (in bytes)
 * @param[in]  bias_w                             (Optional) The size of the width dimension of the bias tensor
 * @param[in]  bias_h                             (Optional) The size of the height dimension of the bias tensor
 * @param[in]  bias_n                             (Optional) The size of the depth dimension of the bias tensor
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias tensor
 * @param[out] dst_ptr                            Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                       Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                              The width of the dst tensor
 * @param[in]  dst_h                              The height of the dst tensor
 * @param[in]  dst_n                              Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the dst matrix
 */
__kernel void mat_mul_native_quantized_mmul_nt_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
    // The explanation of how this kernel works is very similar to the explanation given in
    // mat_mul_mmul.cl. The MMUL logic, and terminology is the same. The only difference is
    // in quantization multiplication, the MMUL block sizes are (4 x 16) for Lhs matrix and
    // (16 x 4) for Rhs matrix, resulting in (4 x 4) MMUL block size for the destination.
    //
    // Figures 1, 2 and 3 in the previous explanation works the same. Since the Lhs and Rhs
    // MMUL block sizes are different in quantized extension, the thread access pattern is
    // slightly different. We can redraw Figure 4 (Thread access pattern) as follows:
    //
    //                                 (Modified Figure 4 from mat_mul_mmul.cl)
    //                               Thread Access Layouts in LHS & RHS matrices
    //
    //                                    LHS matrix
    //                 4 times      4 times          4 times        4 times
    //           _______________________________________________________________
    //          |T0_|T0_|T0_|T0_|T1_|T1_|T1_|T1_|T2_|T2_|T2_|T2_|T3_|T3_|T3_|T3_|
    //          |T0_| ...                                                       |
    //    M0    |   .    .                                                      |
    //   Times  |   .       .                                                   |
    //          |   .           .                                               |
    //          |T0_|T0_|T0_|T0_|T1_|T1_|T1_|T1_|T2_|T2_|T2_|T2_|T3_|T3_|T3_|T3_|
    //          |T4_|T4_|T4_|T4_|T5_|T5_|T5_|T5_|T6_|T6_|T6_|T6_|T7_|T7_|T7_|T7_|
    //          |T4_|T4_|T4_|T4_|T5_|T5_|T5_|T5_|T6_|T6_|T6_|T6_|T7_|T7_|T7_|T7_|
    //    M0    |   .    .                                                      |
    //   Times  |   .       .                                                   |
    //          |   .           .                                               |
    //          |T4_|T4_|T4_|T4_|T5_|T5_|T5_|T5_|T6_|T6_|T6_|T6_|T7_|T7_|T7_|T7_|
    //          |T8_|T8_|T8_|T8_|T9_|T9_|T9_|T9_|T10|T10|T10|T10|T11|T11|T11|T11|
    //    M0    |   .                                                           |
    //   Times  |   .                                                           |
    //          |   .                                                           |
    //          |T8_|T8_|T8_|T8_|T9_|T9_|T9_|T9_|T10|T10|T10|T10|T11|T11|T11|T11|
    //    M0    |   .                                                           |
    //   Times  |   .                                                           |
    //          |   .                                                           |
    //          |T12|T12|T12|T12|T13|T13|T13|T13|T14|T14|T14|T14|T15|T15|T15|T15|
    //
    //
    //                                                  RHS Matrix
    //
    //                   __________N0 times______N0 times____________________N0 times_______
    //                  |__T0__| ... |__T0__|__T1__| ...  |__T1__| ... |__T3__| ... |__T3__|
    //       4 times    |__T0__| ... |__T0__|__T1__| ...  |__T1__| ... |__T3__| ... |__T3__|
    //                  |__T0__| ... |__T0__|__T1__| ...  |__T1__| ... |__T3__| ... |__T3__|
    //                  |__T0__| ... |__T0__|__T1__| ...  |__T1__| ... |__T3__| ... |__T3__|
    //                  |__T4__| ... |__T4__|__T5__| ...  |__T5__| ... |__T7__| ... |__T7__|
    //       4 times    |__T4__| ... |__T4__|__T5__| ...  |__T5__| ... |__T7__| ... |__T7__|
    //                  |__T4__| ... |__T4__|__T5__| ...  |__T5__| ... |__T7__| ... |__T7__|
    //           X      |__T4__| ... |__T4__|__T5__| ...  |__T5__| ... |__T7__| ... |__T7__|
    //                  |__T8__| ... |__T8__|__T9__| ...  |__T9__| ... |__T11_| ... |__T11_|
    //                  |__T8__| ... |__T8__|__T9__| ...  |__T9__| ... |__T11_| ... |__T11_|
    //       4 times    |__T8__| ... |__T8__|__T9__| ...  |__T9__| ... |__T11_| ... |__T11_|
    //                  |__T8__| ... |__T8__|__T9__| ...  |__T9__| ... |__T11_| ... |__T11_|
    //                  |__T12_| ... |__T12_|__T13_| ...  |__T13_| ... |__T15_| ... |__T15_|
    //       4 times    |__T12_| ... |__T12_|__T13_| ...  |__T13_| ... |__T15_| ... |__T15_|
    //                  |__T12_| ... |__T12_|__T13_| ...  |__T13_| ... |__T15_| ... |__T15_|
    //                  |__T12_|_____|__T12_|__T13_|______|__T13_|_____|__T15_|_____|__T15_|
    //
    //
    // The logic behind this thread access pattern is already descried in the explanation
    // in mat_mul_mmul.cl. The only change is threads accesses are extended to 4 elements
    // from 1, in rightward direction in Lhs, and in downward direction in Rhs, because they
    // are now operating on 4 char/uchar's (again 32-bit data), instead of one 32-bit floating point.
    //
    // The mathematical view of the matrix multiplication explained in Figure 5 also holds for this,
    // except the dimension 4 is 16 instead, but the vector notations do not change, i.e. it's as follows:
    //
    //   Settings:
    //          - a 8 x 16 LHS section
    //          - 16 x 8 RHS section
    //          - Each vector variable ai, bj represent a 16x1 vector
    //          - ^T (superscript T) denotes transpose
    //          - M0 = N0 = 2
    //          - MMUL_N0 = MMUL_M0 = 4, MMUL_K0 = 16
    //
    //
    //                                             (Modified Figure 5)
    //                              Mathematical view of the Matrix Multiplication
    //
    //      LHS                           RHS                                           DST
    //    [  a1^T  ]            [ b1 b2 b3 b4 b5 b6 b7 ]                [ a1^Tb1  a1^Tb2  a1^Tb3 ... a1^Tb7 ]
    //    [  a2^T  ]                                    16 x 8          [ a2^Tb1  a2^Tb2  a2^Tb3 ... a2^Tb7 ]
    //    [  a3^T  ]                                                    [                                   ]
    //    [  a4^T  ]                                                =   [   .       .                       ]
    //    [  a5^T  ]        X                                           [   .          .                    ]
    //    [  a6^T  ]                                                    [   .             .                 ]
    //    [  a7^T  ]                                                    [                                   ]
    //    [  a8^T  ]                                                    [ a7^Tb1  a7^Tb2  a7^Tb3 ... a7^Tb7 ]
    //              8 x 16                                                                                     8 x 8
    //
    //
    //  For the first iteration, i.e. (m0, n0) = (0, 0), the arm_matrix_multiply would multiply the following matrices:
    //
    //    [  a1^T  ]            [  b1 b3 b5 b7 ]                [ a1^Tb1  a1^Tb3  a1^Tb5  a1^Tb7 ]
    //    [  a3^T  ]        x                   4 x 4     =     [ a3^Tb1  a1^Tb3  a1^Tb5  a1^Tb7 ]
    //    [  a5^T  ]                                            [ a5^Tb1  a1^Tb3  a1^Tb5  a1^Tb7 ]
    //    [  a7^T  ]                                            [ a7^Tb1  a7^Tb3  a7^Tb5  a7^Tb7 ]
    //              4 x 4                                                                         4 x 4
    // The elements calculated in the 4x4 output block are the "interleaved" elements in the DST above.
    // When we follow for each combination of (m0, n0), every element of the DST matrix "section" is filled.
    //
    // Please refer to mat_mul_mmul.cl for more details.

    const uint x0 = get_global_id(0); // [0, (N / N0) * MMUL_M0)
    // The upper limit is a simplified version of (N / N0) / MMUL_N0) * MMUL_BLOCK_SIZE)
    const uint y0 = get_global_id(1); // [0, (M / M0) / MMUL_M0)
    const uint z  = get_global_id(2); // Batch

    // Get section coordinates
    const uint section_x = (x0 / MMUL_BLOCK_SIZE);
    const uint section_y = y0;

    // Get thread coordinates within an mmul block
    const uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    const uint thread_x  = thread_id % MMUL_N0;
    const uint thread_y  = (thread_id / MMUL_N0);

    // Calculate dst coordinates
    const uint dst_x_unclamped = thread_x * N0 + section_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + section_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    const uint lhs_x = K0 * thread_x;
    const uint lhs_y = dst_y;

    // Starting RHS coordinates
    const uint rhs_x = dst_x;
    const uint rhs_y = K0 * thread_y;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(int, M0, N0, c);
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c[i].v = K * ((int)LHS_OFFSET) * ((int)RHS_OFFSET);
    })

    // Calculate row and column sums
    TILE(int, 1, N0, b_sum);
    b_sum[0].v = 0;

    TILE(int, 1, M0, a_sum);
    a_sum[0].v = 0;

    VEC_DATA_TYPE(DATA_TYPE, K0)
    vec_1 = (VEC_DATA_TYPE(DATA_TYPE, K0))(1, 1, 1, 1);

    for(int k = 0; k < lhs_w; k += MMUL_K0)
    {
        // A tile of M0xK0 but K0 must be set to K0
        TILE(DATA_TYPE, M0, K0, a);
        // A tile of K0xN0 but K0 must be set to K0
        TILE(DATA_TYPE, K0, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                VEC_DATA_TYPE(DATA_TYPE, K0)
                vec_b       = (VEC_DATA_TYPE(DATA_TYPE, K0))(b[0].s[n0], b[1].s[n0], b[2].s[n0], b[3].s[n0]);
                c[m0].s[n0] = arm_matrix_multiply(a[m0].v, vec_b, c[m0].s[n0]);
            })
        })

#if RHS_OFFSET != 0
        // Row Sum of A: Calculate the sum of rows by multiplying A with
        // a matrix of 1's from Right
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            a_sum[0].s[m0] = arm_matrix_multiply(a[m0].v, vec_1, a_sum[0].s[m0]);
        })
#endif // RHS_OFFSET != 0

#if LHS_OFFSET != 0
        // Column Sum of B: Calculate the sum of columns by multiplying B
        // with a matrix of 1's from Left
        LOOP_UNROLLING(int, n0, 0, 1, N0,
        {
            VEC_DATA_TYPE(DATA_TYPE, K0)
            vec_b          = (VEC_DATA_TYPE(DATA_TYPE, K0))(b[0].s[n0], b[1].s[n0], b[2].s[n0], b[3].s[n0]);
            b_sum[0].s[n0] = arm_matrix_multiply(vec_1, vec_b, b_sum[0].s[n0]);
        })
#endif // LHS_OFFSET != 0

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_K0 * rhs_stride_y;
    }

    // Do not write if the coordinates are out of bound
    // But, read has to happen as arm_matrix_multiply() expects certain number of calls
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
    {
        return;
    }

#if RHS_OFFSET != 0 || LHS_OFFSET != 0
    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        const int A = ((int)RHS_OFFSET) * a_sum[0].s[i];
        LOOP_UNROLLING(int, j, 0, 1, N0,
        {
            c[i].s[j] -= A + ((int)(LHS_OFFSET)) * b_sum[0].s[j];
        })
    })
#endif // RHS_OFFSET != 0 || LHS_OFFSET != 0

#ifdef BIAS
    perform_bias_addition(bias_ptr, bias_offset_first_element_in_bytes, c, dst_x);
#endif // defined(BIAS)

    // Quantize the tile
    TILE(DATA_TYPE, M0, N0, cq);
    T_QUANTIZE8_ASYMMETRIC(int, DATA_TYPE, M0, N0, DST_OFFSET, DST_SHIFT, DST_MULTIPLIER, c, cq);

    if(dst_x + N0 <= N || N0_LEFTOVER == 0)
    {
        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            if(dst_y + m0 < M || M0_LEFTOVER == 0)
            {
                VSTORE(N0)
                (cq[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
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
                (cq[m0].v, 0, (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + m0 * dst_stride_y));
            }
        })
    }
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_NT)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS transposed - buffer only
 *
 * Supported block configurations:
 *     TODO: Report supported M0, N0, K0
 *
 * Similar to mat_mul_native_quantized_mmul_nt_nt()
 */
__kernel void mat_mul_native_quantized_mmul_nt_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_NT_T)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS transposed, RHS non-transposed
 *
 * Supported block configurations:
 *     TODO: Report supported M0, N0, K0
 *
 * Similar to mat_mul_native_quantized_mmul_nt_nt()
 */
__kernel void mat_mul_native_quantized_mmul_t_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_NT)

#if defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS transposed, RHS transposed
 *
 * Supported block configurations:
 *     TODO: Report supported M0, N0, K0
 *
 * Similar to mat_mul_native_quantized_mmul_nt_nt()
 */
__kernel void mat_mul_native_quantized_mmul_t_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
#ifdef BIAS
    TENSOR3D_T(bias, BUFFER),
#endif // defined(BIAS)
    TENSOR3D_T(dst, BUFFER))
{
}
#endif // defined(MAT_MUL_NATIVE_QUANTIZED_MMUL_T_T)
