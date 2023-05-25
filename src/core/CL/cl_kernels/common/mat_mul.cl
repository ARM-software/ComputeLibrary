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

#if defined(MAT_MUL_NATIVE_NT_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The fused activation function used should be passed with -DACTIVATION_TYPE, -DA_VAL and -DB_VAL are used for min and max output bounded activation functions.
 * @note The number of leftover outputs rows/columns must be passed using -DPARTIAL_STORE_N0 and -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_N0=2, -DPARTIAL_STORE_M0=3)
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=6)
 * @note The tensor type ("BUFFER" or "IMAGE") of the rhs tensor must be passed at compile time using -DRHS_TENSOR_TYPE (e.g. -DRHS_TENSOR_TYPE=BUFFER)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_NT_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16 (only 4, 8, 16 if RHS_TENSOR_TYPE=IMAGE)
 *  - K0 = 1, 2, 3, 4, 8, 16
 * @note Values > 8 for M0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_img                           (Optional) Read only cl_image object for the rhs tensor. Included when RHS_TENSOR_TYPE=IMAGE
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
 */
__kernel void mat_mul_native_nt_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
    TENSOR3D_T(dst, BUFFER))
{
    const uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc[i].v = 0.f;
    })

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, K0, N0, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, b, acc);

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, 1, N0, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, 1,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, a, b, acc);

        lhs_offset_first_element_in_bytes += 1 * sizeof(DATA_TYPE);
    }
#endif // K % K0 != 0

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, acc, acc);

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
}
#endif // defined(MAT_MUL_NATIVE_NT_NT)

#if defined(MAT_MUL_NATIVE_NT_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS non-transposed, RHS transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The fused activation function used should be passed with -DACTIVATION_TYPE, -DA_VAL and -DB_VAL are used for min and max output bounded activation functions.
 * @note The number of leftover outputs rows/columns must be passed using -DPARTIAL_STORE_N0 and -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_N0=2, -DPARTIAL_STORE_M0=3)
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=6)
 * @note The tensor type ("BUFFER" or "IMAGE") of the rhs tensor must be passed at compile time using -DRHS_TENSOR_TYPE (e.g. -DRHS_TENSOR_TYPE=BUFFER)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_NT_T)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 > 0
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1, 2, 3, 4, 8, 16 (only 4, 8, 16 if RHS_TENSOR_TYPE=IMAGE)
 * @note Values > 8 for M0, N0 and K0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_img                           (Optional) Read only cl_image object for the rhs tensor. Included when RHS_TENSOR_TYPE=IMAGE
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
 */
__kernel void mat_mul_native_nt_t(TENSOR3D_T(lhs, BUFFER),
                                  TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
                                  TENSOR3D_T(dst, BUFFER))

{
    const uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * lhs_stride_y + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc[i].v = 0.f;
    })

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        TILE(DATA_TYPE, M0, K0, a);
        TILE(DATA_TYPE, N0, K0, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, K0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, K0, RHS_TENSOR_TYPE, rhs, k, x + rhs_z, 1, rhs_stride_y, b);

#if GPU_ARCH == GPU_ARCH_MIDGARD
        // This part is written to decrease the number of loop unrollings caused
        // by T_MMUL. The NT/NT version is partly vectorized and uses less number
        // of loop unrollings, and code behaves as expected. Although this is not
        // a performant solution for the specified architecture, it is necessary
        // to overcome some limitations.
        TILE(DATA_TYPE, K0, N0, bt);
        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            LOOP_UNROLLING(int, j, 0, 1, K0,
            {
                bt[j].s[i] = b[i].s[j];
            })
        })
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, a, bt, acc);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, T, a, b, acc);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        lhs_offset_first_element_in_bytes += K0 * sizeof(DATA_TYPE);
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        TILE(DATA_TYPE, M0, 1, a);
        TILE(DATA_TYPE, N0, 1, b);

        LOOP_UNROLLING(int, i, 0, 1, M0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, M0, 1, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, 1, BUFFER, rhs, k, x + rhs_z, 1, rhs_stride_y, b);

#if GPU_ARCH == GPU_ARCH_MIDGARD
        // See the main loop for the explanation of this part
        TILE(DATA_TYPE, 1, N0, bt);
        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            bt[0].s[i] = b[i].s[0];
        })
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, a, bt, acc);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, T, a, b, acc);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        lhs_offset_first_element_in_bytes += 1 * sizeof(DATA_TYPE);
    }
#endif // K % K0 != 0

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, acc, acc);

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
}
#endif // defined(MAT_MUL_NATIVE_NT_T)

#if defined(MAT_MUL_NATIVE_T_NT)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS transposed, RHS non-transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The fused activation function used should be passed with -DACTIVATION_TYPE, -DA_VAL and -DB_VAL are used for min and max output bounded activation functions.
 * @note The number of leftover outputs rows/columns must be passed using -DPARTIAL_STORE_N0 and -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_N0=2, -DPARTIAL_STORE_M0=3)
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=6)
 * @note The tensor type ("BUFFER" or "IMAGE") of the rhs tensor must be passed at compile time using -DRHS_TENSOR_TYPE (e.g. -DRHS_TENSOR_TYPE=BUFFER)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_T_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 8, 16
 *  - N0 = 1, 2, 3, 4, 8, 16 (only 4, 8, 16 if RHS_TENSOR_TYPE=IMAGE)
 *  - K0 > 0
 * * @note Values > 8 for M0, and K0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_img                           (Optional) Read only cl_image object for the rhs tensor. Included when RHS_TENSOR_TYPE=IMAGE
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
 */
__kernel void mat_mul_native_t_nt(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
    TENSOR3D_T(dst, BUFFER))
{
    const uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * sizeof(DATA_TYPE) + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc[i].v = 0.f;
    })

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        TILE(DATA_TYPE, K0, M0, a);
        TILE(DATA_TYPE, K0, N0, b);

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, K0, M0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, K0, N0, RHS_TENSOR_TYPE, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

#if GPU_ARCH == GPU_ARCH_MIDGARD
        // For explanation, see mat_mul_native_nt_t
        TILE(DATA_TYPE, M0, K0, at);
        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            LOOP_UNROLLING(int, j, 0, 1, M0,
            {
                at[j].s[i] = a[i].s[j];
            })
        })
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, at, b, acc);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, T, NT, a, b, acc);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        lhs_offset_first_element_in_bytes += K0 * lhs_stride_y;
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        TILE(DATA_TYPE, 1, M0, a);
        TILE(DATA_TYPE, 1, N0, b);

        LOOP_UNROLLING(int, i, 0, 1, 1,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, 1,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, 1, M0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, 1, N0, BUFFER, rhs, x, k + rhs_z, 1, rhs_stride_y, b);

#if GPU_ARCH == GPU_ARCH_MIDGARD
        // For explanation, see mat_mul_native_nt_t
        TILE(DATA_TYPE, M0, 1, at);
        LOOP_UNROLLING(int, j, 0, 1, M0,
        {
            at[j].s[0] = a[0].s[j];
        })
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, at, b, acc);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, T, NT, a, b, acc);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        lhs_offset_first_element_in_bytes += 1 * lhs_stride_y;
    }
#endif // K % K0 != 0

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, acc, acc);

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
}
#endif // defined(MAT_MUL_NATIVE_T_NT)

#if defined(MAT_MUL_NATIVE_T_T)
/** This OpenCL kernel performs the batch matrix multiplication (BatchMatMul): LHS transposed, RHS transposed - buffer only
 *
 * @note the "batch" here expresses the number of matrix multiplications to run in parallel. However, it
 *       should NOT be confused with the batch size of the model. For NHWC the "batch" is the "H" dimension
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions used for the LHS and RHS matrices (M0, N0 and K0) must be passed at compile time using -DN0, -DM0 and -DK0 (e.g. -DN0=8, -DM0=4, -DK0=4).
 * @note The fused activation function used should be passed with -DACTIVATION_TYPE, -DA_VAL and -DB_VAL are used for min and max output bounded activation functions.
 * @note The number of leftover outputs rows/columns must be passed using -DPARTIAL_STORE_N0 and -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_N0=2, -DPARTIAL_STORE_M0=3)
 * @note The dimension K must be passed at compile time using -DK (e.g. -DK=6)
 * @note The tensor type ("BUFFER" or "IMAGE") of the rhs tensor must be passed at compile time using -DRHS_TENSOR_TYPE (e.g. -DRHS_TENSOR_TYPE=BUFFER)
 * @note The kernel name in uppercase must be passed at compile time (e.g. -DMAT_MUL_NATIVE_T_NT)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 8, 16
 *  - N0 = 1, 2, 3, 4, 8, 16
 *  - K0 = 1, 2, 3, 4, 8, 16 (only 4, 8, 16 if RHS_TENSOR_TYPE=IMAGE)
 * @note Values > 8 for M0, N0 and K0 are not expected to be efficient
 *
 * @param[in]  lhs_ptr                           Pointer to the lhs matrix. Supported data types: F32/F16
 * @param[in]  lhs_stride_y                      Stride of the lhs matrix in Y (2nd) dimension (in bytes)
 * @param[in]  lhs_stride_z                      Stride of the lhs tensor in Z (3rd) dimension (in bytes)
 * @param[in]  lhs_w                             The width of the lhs tensor
 * @param[in]  lhs_h                             The height of the lhs tensor
 * @param[in]  lhs_n                             Number of the matrices (buffers) in the batch
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the lhs matrix
 * @param[in]  rhs_img                           (Optional) Read only cl_image object for the rhs tensor. Included when RHS_TENSOR_TYPE=IMAGE
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
 */
__kernel void mat_mul_native_t_t(
    TENSOR3D_T(lhs, BUFFER),
    TENSOR3D_T(rhs, RHS_TENSOR_TYPE),
    TENSOR3D_T(dst, BUFFER))
{
    const uint x = GET_SPATIAL_IDX(0, N0, PARTIAL_STORE_N0);
    const uint y = GET_SPATIAL_IDX(1, M0, PARTIAL_STORE_M0);
    const uint z = GET_SPATIAL_IDX(2, 1, 0);

    // Compute LHS/RHS/DST matrix address
    lhs_offset_first_element_in_bytes += y * sizeof(DATA_TYPE) + z * lhs_stride_z;
    dst_offset_first_element_in_bytes += x * sizeof(DATA_TYPE) + y * dst_stride_y + z * dst_stride_z;

    // Initialize the accumulators
    TILE(DATA_TYPE, M0, N0, acc);

    LOOP_UNROLLING(int, i, 0, 1, M0,
    {
        acc[i].v = 0.f;
    })

    const int rhs_z = z * rhs_h;
    int       k;
    for(k = 0; k <= K - K0; k += K0)
    {
        TILE(DATA_TYPE, K0, M0, a);
        TILE(DATA_TYPE, N0, K0, b);

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, K0, M0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, K0, RHS_TENSOR_TYPE, rhs, k, x + rhs_z, 1, rhs_stride_y, b);
#if GPU_ARCH == GPU_ARCH_MIDGARD
        // For explanation, see mat_mul_native_nt_t
        TILE(DATA_TYPE, M0, K0, at);
        TILE(DATA_TYPE, K0, N0, bt);

        LOOP_UNROLLING(int, i, 0, 1, K0,
        {
            LOOP_UNROLLING(int, j, 0, 1, M0,
            {
                at[j].s[i] = a[i].s[j];
            })
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            LOOP_UNROLLING(int, j, 0, 1, K0,
            {
                bt[j].s[i] = b[i].s[j];
            })
        })

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, NT, NT, at, bt, acc);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, K0, T, T, a, b, acc);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        lhs_offset_first_element_in_bytes += K0 * lhs_stride_y;
    }

#if K % K0 != 0
    /* Leftover Loop */
    for(; k < K; ++k)
    {
        TILE(DATA_TYPE, 1, M0, a);
        TILE(DATA_TYPE, N0, 1, b);

        LOOP_UNROLLING(int, i, 0, 1, 1,
        {
            a[i].v = 0.f;
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            b[i].v = 0.f;
        })

        // Load tile from the lhs/rhs tensors
        T_LOAD(DATA_TYPE, 1, M0, BUFFER, lhs, 0, 0, 1, lhs_stride_y, a);
        T_LOAD(DATA_TYPE, N0, 1, BUFFER, rhs, k, x + rhs_z, 1, rhs_stride_y, b);

#if GPU_ARCH == GPU_ARCH_MIDGARD
        // For explanation, see mat_mul_native_nt_t
        TILE(DATA_TYPE, M0, 1, at);
        TILE(DATA_TYPE, 1, N0, bt);

        LOOP_UNROLLING(int, j, 0, 1, M0,
        {
            at[j].s[0] = a[0].s[j];
        })

        LOOP_UNROLLING(int, i, 0, 1, N0,
        {
            bt[0].s[i] = b[i].s[0];
        })

        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, NT, NT, at, bt, acc);
#else  // GPU_ARCH == GPU_ARCH_MIDGARD
        T_MMUL(DATA_TYPE, DATA_TYPE, DATA_TYPE, M0, N0, 1, T, T, a, b, acc);
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

        lhs_offset_first_element_in_bytes += 1 * lhs_stride_y;
    }
#endif // K % K0 != 0

    const bool x_cond = PARTIAL_STORE_N0 != 0 && get_global_id(0) == 0;
    const bool y_cond = PARTIAL_STORE_M0 != 0 && get_global_id(1) == 0;

    TILE(int, M0, 1, indirect_buffer);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        indirect_buffer[_i].v = min(_i, select(M0 - 1, PARTIAL_STORE_M0 - 1, y_cond));
    });

    T_ACTIVATION(DATA_TYPE, M0, N0, ACTIVATION_TYPE, A_VAL, B_VAL, acc, acc);

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, N0, PARTIAL_STORE_N0, BUFFER, dst, 0, dst_stride_y, x_cond, acc, indirect_buffer);
}
#endif // defined(MAT_MUL_NATIVE_T_T)
