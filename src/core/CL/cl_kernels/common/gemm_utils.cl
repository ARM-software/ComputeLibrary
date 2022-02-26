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
#include "gemm_helpers.h"
#include "helpers.h"
#include "repeat.h"
#include "tile_helpers.h"

#if defined(RESHAPE_LHS_NT)
/** This OpenCL kernel reshapes the lhs input matrix. The kernel splits the input matrix in blocks of size M0xK0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The width of the input tensor must be passed at compile time using -DSRC_WIDTH (e.g. -DSRC_WIDTH=16)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (e.g. -DSRC_HEIGHT=16)
 * @note The block's dimensions (M0 and K0) must be passed at compile time using -DM0 and -DK0 (e.g. -DM0=2, -DK0=2).
 * @note The size of the partial load block in y must be passed at compile time using -DPARTIAL_M0 (e.g. -DPARTIAL_M0=1)
 * @note The size of the partial load block in x must be passed at compile time using -DPARTIAL_K0 (e.g. -DPARTIAL_K0=1)
 * @note Only the following values for M0, K0 and V0 are supported:
 *                                      M0: 2,3,4,5,6,7,8
 *                                      K0: 2,3,4,8,16
 *                                      V0: greater than 0
 * @note If the M0xK0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: All
 * @param[in] src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_w                             The size of the width dimension of the source tensor
 * @param[in] src_h                             The size of the height dimension of the source tensor
 * @param[in] src_n                             The size of the depth dimension of the source tensor
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: All
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_w                             The size of the width dimension of the destination tensor
 * @param[in] dst_h                             The size of the height dimension of the destination tensor
 * @param[in] dst_n                             The size of the depth dimension of the destination tensor
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] M                                 The size of height dimension of the source tensor, affected by reinterpret_input_as_3d
 * @param[in] V0                                The number of blocks to place on the same row. It must be greater than 0.
 */
__kernel void gemm_reshape_lhs_matrix_nt(TENSOR3D_T(src, BUFFER),
                                         TENSOR3D_T(dst, BUFFER),
                                         const int M,
                                         const int V0)
{
    // Block size
#define BLOCK_SIZE ((M0) * (K0))

    // Output offset X
#if defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (K0)
#else // defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (BLOCK_SIZE)
#endif // defined(INTERLEAVE)

    // Output step X
#if defined(INTERLEAVE)
#define OUTPUT_STEP_X (K0) * (V0)
#else // Do not interleave
#define OUTPUT_STEP_X (K0)
#endif // defined(INTERLEAVE)

    const int x = GET_SPATIAL_IDX(0, 1, 0); // K
    const int y = GET_SPATIAL_IDX(1, 1, 0); // M
    const int z = GET_SPATIAL_IDX(2, 1, 0); // Batch size

    const int xi = x * K0;
    const int yi = y * M0;

    const int xo = x * BLOCK_SIZE * V0 + (y % V0) * OUTPUT_OFFSET_X;
    const int yo = (y / V0);

    // src_stride_z is expressed as M * src_stride_y, to handle case where reinterpret_input_as_3d=true
    src_offset_first_element_in_bytes += yi * src_stride_y + z * M * src_stride_y;
    dst_offset_first_element_in_bytes += yo * dst_stride_y + z * dst_stride_z;

    TILE(DATA_TYPE, M0, K0, in);

    // Initialize the input tile to zero
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        in[_i].v = 0;
    });

    bool x_cond = (xi + K0 >= src_w) && (PARTIAL_K0 != 0);
    bool y_cond = (yi + M0 >= M) && (PARTIAL_M0 != 0);
    // Load input tile
    TILE(uint, M0, 1, in_indirect_y);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        in_indirect_y[_i].v = _i;

    });
#if PARTIAL_M0 != 0
    if(y_cond)
    {
        T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, PARTIAL_M0, K0, PARTIAL_K0, BUFFER, src, xi, src_stride_y, x_cond, in, in_indirect_y);
    }
    else
#endif // PARTIAL_M0 != 0
    {
        T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, K0, PARTIAL_K0, BUFFER, src, xi, src_stride_y, x_cond, in, in_indirect_y);
    }

    // Store output tile
    TILE(uint, M0, 1, dst_indirect_y);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        dst_indirect_y[_i].v = _i;
    });

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, K0, 0, BUFFER, dst, xo, (OUTPUT_STEP_X * sizeof(DATA_TYPE)), false, in, dst_indirect_y);
#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}
#endif // defined(RESHAPE_LHS_NT)

#if defined(RESHAPE_LHS_T)
/** This OpenCL kernel reshapes the lhs input matrix. The kernel splits the input matrix in blocks of size M0xK0 and stores each one (transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The width of the input tensor must be passed at compile time using -DSRC_WIDTH (e.g. -DSRC_WIDTH=16)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (e.g. -DSRC_HEIGHT=16)
 * @note The block's dimensions (M0 and K0) must be passed at compile time using -DM0 and -DK0 (e.g. -DM0=2, -DK0=2).
 * @note The size of the partial load block in y must be passed at compile time using -DPARTIAL_M0 (e.g. -DPARTIAL_M0=1)
 * @note The size of the partial load block in x must be passed at compile time using -DPARTIAL_K0 (e.g. -DPARTIAL_K0=1)
 * @note Only the following values for M0, K0 and V0 are supported:
 *                                      M0: 2,3,4,8,16
 *                                      K0: 2,3,4,8,16
 *                                      V0: greater than 0
 * @note If the M0xK0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: All
 * @param[in] src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_w                             The size of the width dimension of the source tensor
 * @param[in] src_h                             The size of the height dimension of the source tensor
 * @param[in] src_n                             The size of the depth dimension of the source tensor
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: All
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_w                             The size of the width dimension of the destination tensor
 * @param[in] dst_h                             The size of the height dimension of the destination tensor
 * @param[in] dst_n                             The size of the depth dimension of the destination tensor
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] M                                 The size of height dimension of the source tensor, affected by reinterpret_input_as_3d
 * @param[in] V0                                The number of blocks to place on the same row. It must be greater than 0
 */
__kernel void gemm_reshape_lhs_matrix_t(TENSOR3D_T(src, BUFFER),
                                        TENSOR3D_T(dst, BUFFER),
                                        const int M,
                                        const int V0)
{
    // Block size
#define BLOCK_SIZE ((M0) * (K0))

    // Output offset X
#if defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (M0)
#else // defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (BLOCK_SIZE)
#endif // defined(INTERLEAVE)

    // Output step X
#if defined(INTERLEAVE)
#define OUTPUT_STEP_X (M0) * (V0)
#else // Do not interleave
#define OUTPUT_STEP_X (M0)
#endif // defined(INTERLEAVE)

    const int x = GET_SPATIAL_IDX(0, 1, 0); // K
    const int y = GET_SPATIAL_IDX(1, 1, 0); // M
    const int z = GET_SPATIAL_IDX(2, 1, 0); // Batch size

    const int xi = x * K0;
    const int yi = y * M0;

    const int xo = x * BLOCK_SIZE * V0 + ((y % V0) * OUTPUT_OFFSET_X);
    const int yo = (y / V0);

    // src_stride_z is expressed as M * src_stride_y, to handle case where reinterpret_input_as_3d=true
    src_offset_first_element_in_bytes += yi * src_stride_y + z * M * src_stride_y;
    dst_offset_first_element_in_bytes += yo * dst_stride_y + z * dst_stride_z;

    TILE(DATA_TYPE, M0, K0, in);
    TILE(DATA_TYPE, K0, M0, in_tr);

    // Initialize the tile to zero
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        in[_i].v = 0;
    });

    // Load input tile
    bool x_cond = (xi + K0 >= src_w) && (PARTIAL_K0 != 0);
    bool y_cond = (yi + M0 >= M) && (PARTIAL_M0 != 0);

    TILE(uint, M0, 1, in_indirect_y);
    LOOP_UNROLLING(int, _i, 0, 1, M0,
    {
        in_indirect_y[_i].v = _i;

    });
#if PARTIAL_M0 != 0
    if(y_cond)
    {
        T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, PARTIAL_M0, K0, PARTIAL_K0, BUFFER, src, xi, src_stride_y, x_cond, in, in_indirect_y);
    }
    else
#endif // PARTIAL_M0 != 0
    {
        T_LOAD_INDIRECT_WIDTH_SELECT(DATA_TYPE, M0, K0, PARTIAL_K0, BUFFER, src, xi, src_stride_y, x_cond, in, in_indirect_y);
    }
    // Transpose input tile
    LOOP_UNROLLING(int, m0, 0, 1, M0,
    {
        LOOP_UNROLLING(int, k0, 0, 1, K0,
        {
            in_tr[k0].s[m0] = in[m0].s[k0];
        })
    });

    TILE(uint, K0, 1, dst_indirect_y);
    LOOP_UNROLLING(int, _i, 0, 1, K0,
    {
        dst_indirect_y[_i].v = _i;
    });

    // Store output tile
    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, K0, M0, 0, BUFFER, dst, xo, (OUTPUT_STEP_X * sizeof(DATA_TYPE)), false, in_tr, dst_indirect_y);

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}
#endif // defined(RESHAPE_LHS_T)

#if defined(RESHAPE_RHS_NT)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (e.g. -DK0=2, -DN0=2).
 * @note If the K0xN0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 * @note Only the following values for K0, N0 and H0 are supported:
 *                                      N0: 2,3,4,8,16
 *                                      K0: 1,2,3,4,8,16
 *                                      H0: greater than 0
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: All
 * @param[in] src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_w                             The size of the width dimension of the source tensor
 * @param[in] src_h                             The size of the height dimension of the source tensor
 * @param[in] src_n                             The size of the depth dimension of the source tensor
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: All
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_w                             The size of the width dimension of the destination tensor
 * @param[in] dst_h                             The size of the height dimension of the destination tensor
 * @param[in] dst_n                             The size of the depth dimension of the destination tensor
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] H0                                The number of blocks to place on the same row. It must be greater than 0
 */
__kernel void gemm_reshape_rhs_matrix_nt(TENSOR3D_T(src, BUFFER),
                                         TENSOR3D_T(dst, BUFFER),
                                         const int H0)
{
    // Block size
#define BLOCK_SIZE ((K0) * (N0))

    // Output offset X
#if defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (N0)
#else // defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (BLOCK_SIZE)
#endif // defined(INTERLEAVE)

    // Output step X
#if defined(INTERLEAVE)
#define OUTPUT_STEP_X (N0) * (H0)
#else // Do not interleave
#define OUTPUT_STEP_X (N0)
#endif // defined(INTERLEAVE)

    const int x = GET_SPATIAL_IDX(0, 1, 0);
    const int y = GET_SPATIAL_IDX(1, 1, 0);
    const int z = GET_SPATIAL_IDX(2, 1, 0);

    const int xi = x * N0;
    const int yi = y * K0;

    const int xo = y * BLOCK_SIZE * H0 + (x % H0) * OUTPUT_OFFSET_X;
    const int yo = (x / H0);

    src_offset_first_element_in_bytes += yi * src_stride_y + z * src_stride_z;
    dst_offset_first_element_in_bytes += yo * dst_stride_y + z * dst_stride_z;

    TILE(DATA_TYPE, K0, N0, in);

    // Initialize the tile to zero
    for(int i = 0; i < K0; ++i)
    {
        in[i].v = 0;
    }

    // Load input tile
    for(int i = 0; i < K0; ++i)
    {
        if(yi + i < src_h)
        {
            in[i].v = V_LOAD(DATA_TYPE, N0, BUFFER, src, xi, i, src_stride_y);
        }
    }

    TILE(uint, K0, 1, dst_indirect_y);
    for(int i = 0; i < K0; ++i)
    {
        dst_indirect_y[i].v = i;
    }

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, K0, N0, 0, BUFFER, dst, xo, (OUTPUT_STEP_X * sizeof(DATA_TYPE)), false, in, dst_indirect_y);

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}
#endif // defined(RESHAPE_RHS_NT)

#if defined(RESHAPE_RHS_T)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (e.g. -DK0=2, -DN0=2).
 * @note If the K0xN0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 * @note The option -DTRANSPOSE must passed at compile time.
 * @note Only the following values for K0, N0 and H0 are supported:
 *                                      N0: 2,3,4,8,16
 *                                      K0: 2,3,4,8,16
 *                                      H0: greater than 0
 *
 * @param[in] src_ptr                           Pointer to the source tensor. Supported data types: All
 * @param[in] src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_w                             The size of the width dimension of the source tensor
 * @param[in] src_h                             The size of the height dimension of the source tensor
 * @param[in] src_n                             The size of the depth dimension of the source tensor
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: All
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_w                             The size of the width dimension of the destination tensor
 * @param[in] dst_h                             The size of the height dimension of the destination tensor
 * @param[in] dst_n                             The size of the depth dimension of the destination tensor
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] H0                                The number of blocks to place on the same row. It must be greater than 0.
 */
__kernel void gemm_reshape_rhs_matrix_t(TENSOR3D_T(src, BUFFER),
                                        TENSOR3D_T(dst, BUFFER),
                                        const int H0)
{
    // Block size
#define BLOCK_SIZE ((K0) * (N0))

    // Output offset X
#if defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (K0)
#else // defined(INTERLEAVE)
#define OUTPUT_OFFSET_X (BLOCK_SIZE)
#endif // defined(INTERLEAVE)

    // Output step X
#if defined(INTERLEAVE)
#define OUTPUT_STEP_X (K0) * (H0)
#else // Do not interleave
#define OUTPUT_STEP_X (K0)
#endif // defined(INTERLEAVE)

    const int x = GET_SPATIAL_IDX(0, 1, 0);
    const int y = GET_SPATIAL_IDX(1, 1, 0);
    const int z = GET_SPATIAL_IDX(2, 1, 0);

    const int xi = x * N0;
    const int yi = y * K0;

    const int xo = y * BLOCK_SIZE * H0 + (x % H0) * OUTPUT_OFFSET_X;
    const int yo = (x / H0);

    src_offset_first_element_in_bytes += yi * src_stride_y + z * src_stride_z;
    dst_offset_first_element_in_bytes += yo * dst_stride_y + z * dst_stride_z;

    TILE(DATA_TYPE, K0, N0, in);
    TILE(DATA_TYPE, N0, K0, in_tr);

    // Initialize the tile to zero
    for(int i = 0; i < K0; ++i)
    {
        in[i].v = 0;
    }

    // Load input tile
    for(int i = 0; i < K0; ++i)
    {
        if(yi + i < src_h)
        {
            in[i].v = V_LOAD(DATA_TYPE, N0, BUFFER, src, xi, i, src_stride_y);
        }
    }

    // Transpose input tile
    for(int k0 = 0; k0 < K0; ++k0)
    {
        for(int n0 = 0; n0 < N0; ++n0)
        {
            in_tr[n0].s[k0] = in[k0].s[n0];
        }
    }

    TILE(uint, N0, 1, dst_indirect_y);
    for(int i = 0; i < N0; ++i)
    {
        dst_indirect_y[i].v = i;
    }

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, N0, K0, 0, BUFFER, dst, xo, (OUTPUT_STEP_X * sizeof(DATA_TYPE)), false, in_tr, dst_indirect_y);

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}

#endif // defined(RESHAPE_RHS_T)