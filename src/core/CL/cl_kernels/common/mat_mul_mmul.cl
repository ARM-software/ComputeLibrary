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
#include "helpers.h"
#include "tile_helpers.h"

#if defined(MAT_MUL_NATIVE_MMUL_NT_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                           Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                             The width of the rhs tensor
 * @param[in]  rhs_h                             The height of the rhs tensor
 * @param[in]  rhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the rhs matrix
 * @param[out] dst_ptr                           Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                             The width of the dst tensor
 * @param[in]  dst_h                             The height of the dst tensor
 * @param[in]  dst_n                             Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the dst matrix
 * @param[in]  M                                 Number of rows in LHS matrix
 * @param[in]  N                                 Number of columns in RHS matrix
 * @param[in]  K                                 Number of columns in LHS matrix and rows in RHS matrix, both not transposed.
 */
__kernel void mat_mul_native_mmul_nt_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0)

    const uint x0 = get_global_id(0); // (N / N0) * MMUL_M0
    const uint y0 = get_global_id(1); // (M / M0) / MMUL_M0
    const uint z  = get_global_id(2); // Batch

    // Get block coordinates
    const uint block_x = (x0 / MMUL_BLOCK_SIZE);
    const uint block_y = y0;

    // Get thread coordinates within a block
    const uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    const uint thread_x  = thread_id % MMUL_N0;
    const uint thread_y  = (thread_id / MMUL_N0);

    // Starting destination coordinates
    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results
    // Although we will never write out-of-bound, we still need this clamp to ensure that we do not read out-of-bound either.
    const uint dst_x_unclamped = thread_x * N0 + block_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + block_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    const uint lhs_x = thread_x;
    const uint lhs_y = dst_y;

    // Starting RHS coordinates
    const uint rhs_x = dst_x;
    const uint rhs_y = thread_y;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k < K; k += MMUL_K0)
    {
        // A tile of M0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, M0, 1, a);
        // A tile of K0xN0 but K0 must be set to 1
        TILE(DATA_TYPE, 1, N0, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[m0].s[0], b[0].s[n0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_K0 * rhs_stride_y;
    }

    // For threads "outside" of the dst bound, we do not write but we have to "read" (arm_matrix_multiply). That's why this needs to happen after arm_matrix_multiply
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
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
#else // defined(HALF_PRECISION)
#define c c_f32
#endif // defined(HALF_PRECISION)

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

#undef MMUL_BLOCK_SIZE
}
#endif // defined(MAT_MUL_NATIVE_MMUL_NT_NT)

#if defined(MAT_MUL_NATIVE_MMUL_NT_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul) using MMUL: LHS non-transposed, RHS transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The tile's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=1).
 * @note The number of leftover outputs rows/columns must be passed using -DN0_LEFTOVER and -DM0_LEFTOVER (e.g. -DN0_LEFTOVER=2, -DM0_LEFTOVER=3)
 * @note The MMUL block dimension (MMUL_M0, MMUL_N0, MMUL_K0) must be passed at compile time using -DMMUL_M0, -DMMUL_N0 and -DMMUL_K0 (e.g. -DMMUL_M0=4, -DMMUL_N0=4, -DMMUL_K0=4).
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_MMUL_NT_T)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_ptr                           Pointer to the rhs matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  rhs_stride_y                      Stride of the rhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the rhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  rhs_w                             The width of the rhs tensor
 * @param[in]  rhs_h                             The height of the rhs tensor
 * @param[in]  rhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the rhs matrix
 * @param[out] dst_ptr                           Pointer to the dst matrix. Supported data types: same as @p lhs_ptr
 * @param[in]  dst_stride_y                      Stride of the dst matrix in Y (2nd) dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the dst tensor in Z (3rd) dimension (in bytes)
 * @param[in]  dst_w                             The width of the dst tensor
 * @param[in]  dst_h                             The height of the dst tensor
 * @param[in]  dst_n                             Number of the matrices (buffers) in the batch
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the dst matrix
 * @param[in]  M                                 Number of rows in LHS matrix
 * @param[in]  N                                 Number of columns in RHS matrix
 * @param[in]  K                                 Number of columns in LHS matrix and columns in RHS-Transposed matrix, which is multiple of MMUL_K0.
 */
__kernel void mat_mul_native_mmul_nt_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, BUFFER),
    TENSOR3D_T(dst, BUFFER),
    const int M,
    const int N,
    const int K)
{
#define MMUL_BLOCK_SIZE (MMUL_M0 * MMUL_N0)

    const uint x0 = get_global_id(0); // (N / N0) * MMUL_M0
    const uint y0 = get_global_id(1); // (M / M0) / MMUL_M0
    const uint z  = get_global_id(2); // Batch

    // Get block coordinates
    const uint block_x = (x0 / MMUL_BLOCK_SIZE);
    const uint block_y = y0;

    // Get thread coordinates within a block
    const uint thread_id = (x0 % MMUL_BLOCK_SIZE);
    const uint thread_x  = thread_id % MMUL_N0;
    const uint thread_y  = (thread_id / MMUL_N0);

    // Starting destination coordinates
    // Note: We need to clamp dst_x and dst_y because we always need to execute a complete MMUL block! Only after the matrix multiplication
    // part can we exit the kernel if it is out-of-bound. Remember, we have a cooperative matrix multiplication. Therefore, we need a full block to get the correct results
    // Although we will never write out-of-bound, we still need this clamp to ensure that we do not read out-of-bound either.
    const uint dst_x_unclamped = thread_x * N0 + block_x * N0 * MMUL_N0;
    const uint dst_y_unclamped = thread_y * M0 + block_y * M0 * MMUL_M0;
    const uint dst_x           = min(dst_x_unclamped, (uint)(N - N0));
    const uint dst_y           = min(dst_y_unclamped, (uint)(M - M0));

    // Starting LHS coordinates
    const uint lhs_x = thread_x;
    const uint lhs_y = dst_y;

    // Starting RHS coordinates
    const uint rhs_x = thread_y;
    const uint rhs_y = dst_x;

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += lhs_x * sizeof(DATA_TYPE) + lhs_y * lhs_stride_y + z * lhs_stride_z;
    rhs_offset_first_element_in_bytes += rhs_x * sizeof(DATA_TYPE) + rhs_y * rhs_stride_y + z * rhs_stride_z;
    dst_offset_first_element_in_bytes += dst_x * sizeof(DATA_TYPE) + dst_y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    // MMUL extension accumulate the result in F32 for both F32 and F16
    TILE(float, M0, N0, c_f32);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        c_f32[i].v = 0;
    })

    for(int k = 0; k < K; k += MMUL_K0)
    {
        // A tile of M0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, M0, 1, a);
        // A tile of N0xK0 but K0 must be set to 1
        TILE(DATA_TYPE, N0, 1, b);

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, 1, BUFFER, rhs, 0, 0, 1, rhs_stride_y, b);

        LOOP_UNROLLING(int, m0, 0, 1, M0,
        {
            LOOP_UNROLLING(int, n0, 0, 1, N0,
            {
                c_f32[m0].s[n0] = arm_matrix_multiply(a[m0].s[0], b[n0].s[0], c_f32[m0].s[n0]);
            })
        })

        lhs_offset_first_element_in_bytes += MMUL_K0 * sizeof(DATA_TYPE);
        rhs_offset_first_element_in_bytes += MMUL_N0 * sizeof(DATA_TYPE);
    }

    // For threads "outside" of the dst bound, we do not write but we have to "read" (arm_matrix_multiply). That's why this needs to happen after arm_matrix_multiply
    if(dst_x_unclamped >= N || dst_y_unclamped >= M)
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
#else // defined(HALF_PRECISION)
#define c c_f32
#endif // defined(HALF_PRECISION)

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

#undef MMUL_BLOCK_SIZE
}
#endif // defined(MAT_MUL_NATIVE_MMUL_NT_T)
