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
#include "helpers.h"
#include "tile_helpers.h"
#include "gemm_helpers.h"
#include "repeat.h"

#if defined(M0) && defined(K0) && defined(V0) && defined(DATA_TYPE) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(PARTIAL_LOAD_M0) && defined(PARTIAL_LOAD_K0)
#define INC2 (VEC_DATA_TYPE(uint, 2))(0, 1)
#define INC3 (VEC_DATA_TYPE(uint, 3))(0, 1, 2)
#define INC4 (VEC_DATA_TYPE(uint, 4))(0, 1, 2, 3)
#define INC8 (VEC_DATA_TYPE(uint, 8))(0, 1, 2, 3, 4, 5, 6, 7)
#define INC16 (VEC_DATA_TYPE(uint, 16))(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
#define CONCAT_INC(K0) INC##K0
#define INC(K0) CONCAT_INC(K0)

#if(SRC_WIDTH % K0)
#define BOUNDARY_CONDITION_X(x, a)                                                                                                                   \
    ({                                                                                                                                               \
        a = select(0, a, CONVERT(((x * (VEC_DATA_TYPE(uint, K0))K0 + INC(K0)) < (VEC_DATA_TYPE(uint, K0))SRC_WIDTH), VEC_DATA_TYPE(DATA_TYPE, K0))); \
    })
#else // (SRC_WIDTH % K0)
#define BOUNDARY_CONDITION_X(x, a) \
    ({})
#endif // (SRC_WIDTH % K0)

#define LOAD_TENSOR_BOUNDARY_AWARE_M0XK0(M0, K0, DATA_TYPE, a, input_ptr, src_stride_y, zin)                     \
    ({                                                                                                           \
        if(y * M0 + M0 >= SRC_HEIGHT && PARTIAL_LOAD_M0 != 0)                                                    \
        {                                                                                                        \
            if(x * K0 + K0 >= SRC_WIDTH && (PARTIAL_LOAD_K0 != 0))                                               \
            {                                                                                                    \
                LOAD_TENSOR_M0XN0(PARTIAL_LOAD_M0, PARTIAL_LOAD_K0, DATA_TYPE, a, input_ptr, src_stride_y, zin); \
            }                                                                                                    \
            else                                                                                                 \
            {                                                                                                    \
                LOAD_TENSOR_M0XN0(PARTIAL_LOAD_M0, K0, DATA_TYPE, a, input_ptr, src_stride_y, zin);              \
            }                                                                                                    \
        }                                                                                                        \
        else                                                                                                     \
        {                                                                                                        \
            if(x * K0 + K0 >= SRC_WIDTH && (PARTIAL_LOAD_K0 != 0))                                               \
            {                                                                                                    \
                LOAD_TENSOR_M0XN0(M0, PARTIAL_LOAD_K0, DATA_TYPE, a, input_ptr, src_stride_y, zin);              \
            }                                                                                                    \
            else                                                                                                 \
            {                                                                                                    \
                LOAD_TENSOR_M0XN0(M0, K0, DATA_TYPE, a, input_ptr, src_stride_y, zin);                           \
            }                                                                                                    \
        }                                                                                                        \
    })

/** This OpenCL kernel reshapes the lhs input matrix. The kernel splits the input matrix in blocks of size M0xK0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The width of the input tensor must be passed at compile time using -DSRC_WIDTH (e.g. -DSRC_WIDTH=16)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (e.g. -DSRC_HEIGHT=16)
 * @note The block's dimensions (M0 and K0) must be passed at compile time using -DM0 and -DK0 (e.g. -DM0=2, -DK0=2).
 * @note The number of M0xK0 vertical blocks to store on the same output row must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note The size of the partial load block in y must be passed at compile time using -DPARTIAL_LOAD_M0 (e.g. -DPARTIAL_LOAD_M0=1)
 * @note The size of the partial load block in x must be passed at compile time using -DPARTIAL_LOAD_K0 (e.g. -DPARTIAL_LOAD_K0=1)
 * @note Only the following values for M0, K0 and V0 are supported:
 *                                      M0: 2,3,4,5,6,7,8
 *                                      K0: 2,3,4,8,16
 *                                      V0: greater than 0
 * @note In case the input has to be reinterpreted as a 3D tensor (e.g. input of convolution layer 1x1), the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# HEIGHT_GEMM3D: The height of the input in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the input in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 * @note If the M0xK0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 *
 * @param[in]  src_ptr                           Pointer to the source LHS tensor. Supported data types: All
 * @param[in]  src_stride_x                      Stride of the source LHS tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source LHS tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source LHS tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source LHS tensor
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  cross_plane_pad                   (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 */
__kernel void gemm_reshape_lhs_matrix_nt(TENSOR3D_DECLARATION(src),
                                         TENSOR3D_DECLARATION(dst)
#if defined(REINTERPRET_INPUT_AS_3D)
                                         ,
                                         uint cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
                                        )
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

    // Compute source and destination addresses
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // ------------------ Compute input/output addresses ---------------------------

    // Compute the input address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + x * (uint)K0 * sizeof(DATA_TYPE) + y * (uint)M0 * src_stride_y;

    // Compute the output address
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)BLOCK_SIZE * (uint)V0 * sizeof(DATA_TYPE)) + ((y / (uint)V0) * (uint)dst_stride_y) + ((y % V0) *
                                 (uint)OUTPUT_OFFSET_X * sizeof(DATA_TYPE));

    // Create variables: uint zin0=0, zin1=0, zin2=0...zin(M0-1)=0;
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zin, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src_stride_z by DEPTH_GEMM3D

    input_ptr += z * (uint)src_stride_z * DEPTH_GEMM3D;

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zin, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, cross_plane_pad, src_stride_y);

#else // defined(REINTERPRET_INPUT_AS_3D)

    input_ptr += z * (uint)src_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    output_ptr += z * (uint)dst_stride_z;

    // ---------------------------Load input values --------------------------------
    // Load values from the LHS matrix
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, K0), a, 0);

    LOAD_TENSOR_BOUNDARY_AWARE_M0XK0(M0, K0, DATA_TYPE, a, input_ptr, src_stride_y, zin);

    // ---------------------------Store output values ------------------------------
    REPEAT_VAR_INIT_TO_CONST(16, uint, zout, 0);
    STORE_BLOCK(M0, K0, DATA_TYPE, a, output_ptr, OUTPUT_STEP_X * sizeof(DATA_TYPE), zout);

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}

#if M0 == 2
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                  \
    ({                                                                                            \
        VEC_DATA_TYPE(DATA_TYPE, M0)                                                              \
        res = (VEC_DATA_TYPE(DATA_TYPE, M0))(a0.s##i, a1.s##i);                                   \
        VSTORE(M0)                                                                                \
        (res, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE))); \
    })
#elif M0 == 3 // M0 == 3
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                  \
    ({                                                                                            \
        VEC_DATA_TYPE(DATA_TYPE, M0)                                                              \
        res = (VEC_DATA_TYPE(DATA_TYPE, M0))(a0.s##i, a1.s##i, a2.s##i);                          \
        VSTORE(M0)                                                                                \
        (res, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE))); \
    })
#elif M0 == 4 // M0 == 4
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                  \
    ({                                                                                            \
        VEC_DATA_TYPE(DATA_TYPE, M0)                                                              \
        res = (VEC_DATA_TYPE(DATA_TYPE, M0))(a0.s##i, a1.s##i, a2.s##i, a3.s##i);                 \
        VSTORE(M0)                                                                                \
        (res, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE))); \
    })
#elif M0 == 5 // M0 == 5
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                      \
    ({                                                                                                \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                   \
        res0           = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s##i, a1.s##i, a2.s##i, a3.s##i);           \
        DATA_TYPE res1 = a4.s##i;                                                                     \
        VSTORE(4)                                                                                     \
        (res0, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)));    \
        *((__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)) + 4) = res1; \
    })
#elif M0 == 6 // M0 == 6
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                       \
    ({                                                                                                 \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                    \
        res0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s##i, a1.s##i, a2.s##i, a3.s##i);                      \
        VEC_DATA_TYPE(DATA_TYPE, 2)                                                                    \
        res1 = (VEC_DATA_TYPE(DATA_TYPE, 2))(a4.s##i, a5.s##i);                                        \
        VSTORE(4)                                                                                      \
        (res0, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)));     \
        VSTORE(2)                                                                                      \
        (res1, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)) + 4); \
    })
#elif M0 == 7 // M0 == 7
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                       \
    ({                                                                                                 \
        VEC_DATA_TYPE(DATA_TYPE, 4)                                                                    \
        res0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s##i, a1.s##i, a2.s##i, a3.s##i);                      \
        VEC_DATA_TYPE(DATA_TYPE, 3)                                                                    \
        res1 = (VEC_DATA_TYPE(DATA_TYPE, 3))(a4.s##i, a5.s##i, a6.s##i);                               \
        VSTORE(4)                                                                                      \
        (res0, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)));     \
        VSTORE(3)                                                                                      \
        (res1, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)) + 4); \
    })
#elif M0 == 8 // M0 == 8
#define TRANSPOSE_COLUMN_AND_STORE(output_ptr, output_step_x, i)                                                      \
    ({                                                                                                                \
        VEC_DATA_TYPE(DATA_TYPE, M0)                                                                                  \
        res = (VEC_DATA_TYPE(DATA_TYPE, M0))(a0.s##i, a1.s##i, a2.s##i, a3.s##i, a4.s##i, a5.s##i, a6.s##i, a7.s##i); \
        VSTORE(M0)                                                                                                    \
        (res, 0, (__global DATA_TYPE *)(output_ptr + 0x##i * output_step_x * sizeof(DATA_TYPE)));                     \
    })
#else // M0 not supported
#error "M0 value not supported"
#endif // N0 conditions

/** This OpenCL kernel reshapes the lhs input matrix. The kernel splits the input matrix in blocks of size M0xK0 and stores each one (transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The width of the input tensor must be passed at compile time using -DSRC_WIDTH (e.g. -DSRC_WIDTH=16)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (e.g. -DSRC_HEIGHT=16)
 * @note The block's dimensions (M0 and K0) must be passed at compile time using -DM0 and -DK0 (e.g. -DM0=2, -DK0=2).
 * @note The number of M0xK0 vertical blocks to store on the same output row must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note The size of the partial load block in y must be passed at compile time using -DPARTIAL_LOAD_M0 (e.g. -DPARTIAL_LOAD_M0=1)
 * @note The size of the partial load block in x must be passed at compile time using -DPARTIAL_LOAD_K0 (e.g. -DPARTIAL_LOAD_K0=1)
 * @note Only the following values for M0, K0 and V0 are supported:
 *                                      M0: 2,3,4,5,6,7,8
 *                                      K0: 2,3,4,8,16
 *                                      V0: greater than 0
 * @note In case the input has to be reinterpreted as a 3D tensor (e.g. input of convolution layer 1x1), the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# HEIGHT_GEMM3D: The height of the input in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the input in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 * @note If the M0xK0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 *
 * @param[in]  src_ptr                           Pointer to the source LHS tensor. Supported data types: All
 * @param[in]  src_stride_x                      Stride of the source LHS tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source LHS tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source LHS tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source LHS tensor
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  cross_plane_pad                   (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 */
__kernel void gemm_reshape_lhs_matrix_t(TENSOR3D_DECLARATION(src),
                                        TENSOR3D_DECLARATION(dst)
#if defined(REINTERPRET_INPUT_AS_3D)
                                        ,
                                        uint cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
                                       )
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

    // Compute source and destination addresses
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // ------------------ Compute input/output addresses ---------------------------

    // Compute the input address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + x * (uint)K0 * sizeof(DATA_TYPE) + y * (uint)M0 * src_stride_y;

    // Compute the output address
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)BLOCK_SIZE * (uint)V0 * sizeof(DATA_TYPE)) + ((y / (uint)V0) * (uint)dst_stride_y) + ((y % V0) *
                                 (uint)OUTPUT_OFFSET_X * sizeof(DATA_TYPE));

    // Create variables: uint zin0=0, zin1=0, zin2=0...zin(M0-1)=0;
    REPEAT_VAR_INIT_TO_CONST(M0, uint, zin, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src_stride_z by DEPTH_GEMM3D

    input_ptr += z * (uint)src_stride_z * DEPTH_GEMM3D;

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zin, y, HEIGHT_GEMM3D, DEPTH_GEMM3D, cross_plane_pad, src_stride_y);

#else // defined(REINTERPRET_INPUT_AS_3D)

    input_ptr += z * (uint)src_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    output_ptr += z * (uint)dst_stride_z;

    // ---------------------------Load input values --------------------------------
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, K0), a, 0);

    LOAD_TENSOR_BOUNDARY_AWARE_M0XK0(M0, K0, DATA_TYPE, a, input_ptr, src_stride_y, zin);

    // ---------------------------Transpose and store block -----------------------

    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 0);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 1);
#if K0 > 2
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 2);
#endif // K0 > 2
#if K0 > 3
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 3);
#endif // K0 > 3
#if K0 > 4
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 4);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 5);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 6);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 7);
#endif // K0 > 4
#if K0 > 8
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 8);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, 9);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, A);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, B);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, C);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, D);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, E);
    TRANSPOSE_COLUMN_AND_STORE(output_ptr, OUTPUT_STEP_X, F);
#endif // K0 > 8

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}
#endif // defined(M0) && defined(K0) && defined(V0) && defined(DATA_TYPE) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(PARTIAL_LOAD_M0) && defined(PARTIAL_LOAD_K0)

#if defined(RESHAPE_RHS_NT)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (e.g. -DK0=2, -DN0=2).
 * @note The number of K0xN0 vertical blocks to store on the same output row must be passed at compile time using -DH0 (e.g. -DH0=2)
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
 * @param[in] H0                                The number of blocks to place on the same row. It must be greater than 0.
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
 * @note The number of K0xN0 vertical blocks to store on the same output row must be passed at compile time using -DH0 (e.g. -DH0=2)
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