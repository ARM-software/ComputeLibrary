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

#if defined(K0) && defined(N0) && defined(H0) && defined(DATA_TYPE) && defined(SRC_HEIGHT)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (e.g. -DSRC_HEIGHT=16)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (e.g. -DK0=2, -DN0=2).
 * @note The number of K0xN0 vertical blocks to store on the same output row must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 * @note Only the following values for K0, N0 and H0 are supported:
 *                                      N0: 2,3,4,8,16
 *                                      K0: 1,2,3,4,8,16
 *                                      H0: greater than 0
 *
 * @param[in]  src_ptr                           Pointer to the source RHS tensor. Supported data types: All
 * @param[in]  src_stride_x                      Stride of the source RHS tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source RHS tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source RHS tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source RHS tensor
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_reshape_rhs_matrix_nt(TENSOR3D_DECLARATION(src),
                                         TENSOR3D_DECLARATION(dst))
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

    // Compute source and destination addresses
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // ------------------ Compute input/output addresses ---------------------------

    // Compute the input address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + x * (uint)N0 * sizeof(DATA_TYPE) + y * (uint)K0 * src_stride_y + z * (uint)src_stride_z;

    // Compute the output address
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + (y * (uint)BLOCK_SIZE * (uint)H0 * sizeof(DATA_TYPE)) + ((x % (uint)H0) * (uint)OUTPUT_OFFSET_X * sizeof(DATA_TYPE)) + ((
                                     x / (uint)H0)
                                 * (uint)dst_stride_y)
                                 + z * (uint)dst_stride_z;

    // ---------------------------Load input values --------------------------------

    REPEAT_VAR_INIT_TO_CONST(K0, VEC_DATA_TYPE(DATA_TYPE, N0), a, 0); ////uint a0=0, a1=0, a2=0...a(M0-1)=0;

    // Load values from the RHS matrix
    a0 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y));
#if K0 > 1
    if(y * (uint)K0 + 1 < SRC_HEIGHT)
    {
        a1 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y));
    }
#endif // K0 > 1
#if K0 > 2
    if(y * (uint)K0 + 2 < SRC_HEIGHT)
    {
        a2 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y));
    }
#endif // K0 > 2
#if K0 > 3
    if(y * (uint)K0 + 3 < SRC_HEIGHT)
    {
        a3 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 3 * src_stride_y));
    }
#endif // K0 > 3
#if K0 > 4
    if(y * (uint)K0 + 4 < SRC_HEIGHT)
    {
        a4 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 4 * src_stride_y));
    }
    if(y * (uint)K0 + 5 < SRC_HEIGHT)
    {
        a5 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 5 * src_stride_y));
    }
    if(y * (uint)K0 + 6 < SRC_HEIGHT)
    {
        a6 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 6 * src_stride_y));
    }
    if(y * (uint)K0 + 7 < SRC_HEIGHT)
    {
        a7 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 7 * src_stride_y));
    }
#endif // K0 > 4
#if K0 > 8
    if(y * (uint)K0 + 8 < SRC_HEIGHT)
    {
        a8 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 8 * src_stride_y));
    }
    if(y * (uint)K0 + 9 < SRC_HEIGHT)
    {
        a9 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 9 * src_stride_y));
    }
    if(y * (uint)K0 + 10 < SRC_HEIGHT)
    {
        aA = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 10 * src_stride_y));
    }
    if(y * (uint)K0 + 11 < SRC_HEIGHT)
    {
        aB = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 11 * src_stride_y));
    }
    if(y * (uint)K0 + 12 < SRC_HEIGHT)
    {
        aC = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 12 * src_stride_y));
    }
    if(y * (uint)K0 + 13 < SRC_HEIGHT)
    {
        aD = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 13 * src_stride_y));
    }
    if(y * (uint)K0 + 14 < SRC_HEIGHT)
    {
        aE = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 14 * src_stride_y));
    }
    if(y * (uint)K0 + 15 < SRC_HEIGHT)
    {
        aF = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 15 * src_stride_y));
    }
#endif // K0 > 8

    // ---------------------------Store output values ------------------------------
    REPEAT_VAR_INIT_TO_CONST(16, uint, zout, 0);
    STORE_BLOCK(K0, N0, DATA_TYPE, a, output_ptr, OUTPUT_STEP_X * sizeof(DATA_TYPE), zout);

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}

#if defined(TRANSPOSE)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (e.g. -DSRC_HEIGHT=16)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (e.g. -DK0=2, -DN0=2).
 * @note The number of K0xN0 vertical blocks to store on the same output row must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 * @note The option -DTRANSPOSE must passed at compile time.
 * @note Only the following values for K0, N0 and H0 are supported:
 *                                      N0: 2,3,4,8,16
 *                                      K0: 2,3,4,8,16
 *                                      H0: greater than 0
 *
 * @param[in]  src_ptr                           Pointer to the source RHS tensor. Supported data types: All
 * @param[in]  src_stride_x                      Stride of the source RHS tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source RHS tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source RHS tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source RHS tensor
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_reshape_rhs_matrix_t(TENSOR3D_DECLARATION(src),
                                        TENSOR3D_DECLARATION(dst))
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

    // Compute source and destination addresses
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // ------------------ Compute input/output addresses ---------------------------

    // Compute the input address
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + x * (uint)N0 * sizeof(DATA_TYPE) + y * (uint)K0 * src_stride_y + z * (uint)src_stride_z;

    // Compute the output address
    __global uchar *output_ptr = dst_ptr + dst_offset_first_element_in_bytes + (y * (uint)BLOCK_SIZE * (uint)H0 * sizeof(DATA_TYPE)) + ((x % H0) * (uint)OUTPUT_OFFSET_X * sizeof(DATA_TYPE)) + ((x /
                                 (uint)H0) * (uint)dst_stride_y) + z * (uint)dst_stride_z;

    // ---------------------------Load input values --------------------------------
    REPEAT_VAR_INIT_TO_CONST(K0, VEC_DATA_TYPE(DATA_TYPE, N0), a, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    a0=0, a1=0, ... a(K0-1)=0;

    // Load values from the RHS matrix
    a0 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y));
    if(y * (uint)K0 + 1 < SRC_HEIGHT)
    {
        a1 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y));
    }
#if K0 > 2
    if(y * (uint)K0 + 2 < SRC_HEIGHT)
    {
        a2 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y));
    }
#endif // K0 > 2
#if K0 > 3
    if(y * (uint)K0 + 3 < SRC_HEIGHT)
    {
        a3 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 3 * src_stride_y));
    }
#endif // K0 > 3
#if K0 > 4
    if(y * (uint)K0 + 4 < SRC_HEIGHT)
    {
        a4 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 4 * src_stride_y));
    }
    if(y * (uint)K0 + 5 < SRC_HEIGHT)
    {
        a5 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 5 * src_stride_y));
    }
    if(y * (uint)K0 + 6 < SRC_HEIGHT)
    {
        a6 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 6 * src_stride_y));
    }
    if(y * (uint)K0 + 7 < SRC_HEIGHT)
    {
        a7 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 7 * src_stride_y));
    }
#endif // K0 > 4
#if K0 > 8
    if(y * (uint)K0 + 8 < SRC_HEIGHT)
    {
        a8 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 8 * src_stride_y));
    }
    if(y * (uint)K0 + 9 < SRC_HEIGHT)
    {
        a9 = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 9 * src_stride_y));
    }
    if(y * (uint)K0 + 10 < SRC_HEIGHT)
    {
        aA = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 10 * src_stride_y));
    }
    if(y * (uint)K0 + 11 < SRC_HEIGHT)
    {
        aB = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 11 * src_stride_y));
    }
    if(y * (uint)K0 + 12 < SRC_HEIGHT)
    {
        aC = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 12 * src_stride_y));
    }
    if(y * (uint)K0 + 13 < SRC_HEIGHT)
    {
        aD = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 13 * src_stride_y));
    }
    if(y * (uint)K0 + 14 < SRC_HEIGHT)
    {
        aE = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 14 * src_stride_y));
    }
    if(y * (uint)K0 + 15 < SRC_HEIGHT)
    {
        aF = VLOAD(N0)(0, (__global DATA_TYPE *)(input_ptr + 15 * src_stride_y));
    }
#endif // K0 > 8

    // ---------------------------Transpose the block ------------------------------
    REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), res, 0); //VEC_DATA_TYPE(DATA_TYPE, K0)    res0=0, res1=0, res2=0,... res(N0-1)=0;

#if K0 == 2
    // This part computes the following transpositions:
    // 2x2 -> 2x2
    // 2x4 -> 4x2
    // 2x8 -> 8x2
    // 2x16 -> 16x2
    res0 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s0, a1.s0);
    res1 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s1, a1.s1);
#if N0 > 2
    res2 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s2, a1.s2);
#endif // N0 > 2
#if N0 > 3
    res3 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s3, a1.s3);
#endif // N0 > 3
#if N0 > 4
    res4 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s4, a1.s4);
    res5 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s5, a1.s5);
    res6 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s6, a1.s6);
    res7 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s7, a1.s7);
#endif // N0 > 4
#if N0 > 8
    res8 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s8, a1.s8);
    res9 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s9, a1.s9);
    resA = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sA, a1.sA);
    resB = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sB, a1.sB);
    resC = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sC, a1.sC);
    resD = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sD, a1.sD);
    resE = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sE, a1.sE);
    resF = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sF, a1.sF);
#endif // N0 > 8

#elif K0 == 3 // K0 == 2
    // This part computes the following transpositions:
    // 3x2 -> 2x3
    // 3x4 -> 4x3
    // 3x8 -> 8x3
    // 3x16 -> 16x3
    res0                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s0, a1.s0, a2.s0);
    res1                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s1, a1.s1, a2.s1);
#if N0 > 2
    res2                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s2, a1.s2, a2.s2);
#endif // N0 > 2
#if N0 > 3
    res3                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s3, a1.s3, a2.s3);
#endif // N0 > 3
#if N0 > 4
    res4                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s4, a1.s4, a2.s4);
    res5                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s5, a1.s5, a2.s5);
    res6                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s6, a1.s6, a2.s6);
    res7                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s7, a1.s7, a2.s7);
#endif // N0 > 4
#if N0 > 8
    res8                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s8, a1.s8, a2.s8);
    res9                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s9, a1.s9, a2.s9);
    resA                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sA, a1.sA, a2.sA);
    resB                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sB, a1.sB, a2.sB);
    resC                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sC, a1.sC, a2.sC);
    resD                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sD, a1.sD, a2.sD);
    resE                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sE, a1.sE, a2.sE);
    resF                      = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sF, a1.sF, a2.sF);
#endif // N0 > 8

#elif K0 == 4 // K0 == 4
    // This part computes the following transpositions:
    // 4x2 -> 2x4
    // 4x4 -> 4x4
    // 4x8 -> 8x4
    // 4x16 -> 16x4
    res0 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s0, a1.s0, a2.s0, a3.s0);
    res1 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s1, a1.s1, a2.s1, a3.s1);
#if N0 > 2
    res2 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s2, a1.s2, a2.s2, a3.s2);
#endif // N0 > 2
#if N0 > 3
    res3 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s3, a1.s3, a2.s3, a3.s3);
#endif // N0 > 3
#if N0 > 4
    res4 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s4, a1.s4, a2.s4, a3.s4);
    res5 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s5, a1.s5, a2.s5, a3.s5);
    res6 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s6, a1.s6, a2.s6, a3.s6);
    res7 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s7, a1.s7, a2.s7, a3.s7);
#endif // N0 > 4
#if N0 > 8
    res8 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s8, a1.s8, a2.s8, a3.s8);
    res9 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s9, a1.s9, a2.s9, a3.s9);
    resA = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sA, a1.sA, a2.sA, a3.sA);
    resB = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sB, a1.sB, a2.sB, a3.sB);
    resC = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sC, a1.sC, a2.sC, a3.sC);
    resD = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sD, a1.sD, a2.sD, a3.sD);
    resE = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sE, a1.sE, a2.sE, a3.sE);
    resF = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sF, a1.sF, a2.sF, a3.sF);
#endif // N0 > 8

#elif K0 == 8 // K0 == 8
    // This part computes the following transpositions:
    // 8x2 -> 2x8
    // 8x4 -> 4x8
    // 8x8 -> 8x8
    // 8x16 -> 16x8
    res0 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s0, a1.s0, a2.s0, a3.s0, a4.s0, a5.s0, a6.s0, a7.s0);
    res1 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s1, a1.s1, a2.s1, a3.s1, a4.s1, a5.s1, a6.s1, a7.s1);
#if N0 > 2
    res2 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s2, a1.s2, a2.s2, a3.s2, a4.s2, a5.s2, a6.s2, a7.s2);
#endif // N0 > 2
#if N0 > 3
    res3 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s3, a1.s3, a2.s3, a3.s3, a4.s3, a5.s3, a6.s3, a7.s3);
#endif // N0 > 3
#if N0 > 4
    res4 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s4, a1.s4, a2.s4, a3.s4, a4.s4, a5.s4, a6.s4, a7.s4);
    res5 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s5, a1.s5, a2.s5, a3.s5, a4.s5, a5.s5, a6.s5, a7.s5);
    res6 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s6, a1.s6, a2.s6, a3.s6, a4.s6, a5.s6, a6.s6, a7.s6);
    res7 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s7, a1.s7, a2.s7, a3.s7, a4.s7, a5.s7, a6.s7, a7.s7);
#endif // N0 > 4
#if N0 > 8
    res8 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s8, a1.s8, a2.s8, a3.s8, a4.s8, a5.s8, a6.s8, a7.s8);
    res9 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s9, a1.s9, a2.s9, a3.s9, a4.s9, a5.s9, a6.s9, a7.s9);
    resA = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sA, a1.sA, a2.sA, a3.sA, a4.sA, a5.sA, a6.sA, a7.sA);
    resB = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sB, a1.sB, a2.sB, a3.sB, a4.sB, a5.sB, a6.sB, a7.sB);
    resC = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sC, a1.sC, a2.sC, a3.sC, a4.sC, a5.sC, a6.sC, a7.sC);
    resD = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sD, a1.sD, a2.sD, a3.sD, a4.sD, a5.sD, a6.sD, a7.sD);
    resE = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sE, a1.sE, a2.sE, a3.sE, a4.sE, a5.sE, a6.sE, a7.sE);
    resF = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sF, a1.sF, a2.sF, a3.sF, a4.sF, a5.sF, a6.sF, a7.sF);
#endif // N0 > 8

#elif K0 == 16 // K0 == 16

    // This part computes the following transpositions:
    // 16x2 -> 2x16
    // 16x4 -> 4x16
    // 16x8 -> 8x16
    // 16x16 -> 16x16
    res0 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s0, a1.s0, a2.s0, a3.s0, a4.s0, a5.s0, a6.s0, a7.s0,
                                          a8.s0, a9.s0, aA.s0, aB.s0, aC.s0, aD.s0, aE.s0, aF.s0);
    res1 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s1, a1.s1, a2.s1, a3.s1, a4.s1, a5.s1, a6.s1, a7.s1,
                                          a8.s1, a9.s1, aA.s1, aB.s1, aC.s1, aD.s1, aE.s1, aF.s1);
#if N0 > 2
    res2 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s2, a1.s2, a2.s2, a3.s2, a4.s2, a5.s2, a6.s2, a7.s2,
                                          a8.s2, a9.s2, aA.s2, aB.s2, aC.s2, aD.s2, aE.s2, aF.s2);
#endif // N0 > 2
#if N0 > 3
    res3 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s3, a1.s3, a2.s3, a3.s3, a4.s3, a5.s3, a6.s3, a7.s3,
                                          a8.s3, a9.s3, aA.s3, aB.s3, aC.s3, aD.s3, aE.s3, aF.s3);
#endif // N0 > 3
#if N0 > 4
    res4 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s4, a1.s4, a2.s4, a3.s4, a4.s4, a5.s4, a6.s4, a7.s4,
                                          a8.s4, a9.s4, aA.s4, aB.s4, aC.s4, aD.s4, aE.s4, aF.s4);
    res5 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s5, a1.s5, a2.s5, a3.s5, a4.s5, a5.s5, a6.s5, a7.s5,
                                          a8.s5, a9.s5, aA.s5, aB.s5, aC.s5, aD.s5, aE.s5, aF.s5);
    res6 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s6, a1.s6, a2.s6, a3.s6, a4.s6, a5.s6, a6.s6, a7.s6,
                                          a8.s6, a9.s6, aA.s6, aB.s6, aC.s6, aD.s6, aE.s6, aF.s6);
    res7 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s7, a1.s7, a2.s7, a3.s7, a4.s7, a5.s7, a6.s7, a7.s7,
                                          a8.s7, a9.s7, aA.s7, aB.s7, aC.s7, aD.s7, aE.s7, aF.s7);
#endif // N0 > 4
#if N0 > 8
    res8 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s8, a1.s8, a2.s8, a3.s8, a4.s8, a5.s8, a6.s8, a7.s8,
                                          a8.s8, a9.s8, aA.s8, aB.s8, aC.s8, aD.s8, aE.s8, aF.s8);
    res9 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s9, a1.s9, a2.s9, a3.s9, a4.s9, a5.s9, a6.s9, a7.s9,
                                          a8.s9, a9.s9, aA.s9, aB.s9, aC.s9, aD.s9, aE.s9, aF.s9);
    resA = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sA, a1.sA, a2.sA, a3.sA, a4.sA, a5.sA, a6.sA, a7.sA,
                                          a8.sA, a9.sA, aA.sA, aB.sA, aC.sA, aD.sA, aE.sA, aF.sA);
    resB = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sB, a1.sB, a2.sB, a3.sB, a4.sB, a5.sB, a6.sB, a7.sB,
                                          a8.sB, a9.sB, aA.sB, aB.sB, aC.sB, aD.sB, aE.sB, aF.sB);
    resC = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sC, a1.sC, a2.sC, a3.sC, a4.sC, a5.sC, a6.sC, a7.sC,
                                          a8.sC, a9.sC, aA.sC, aB.sC, aC.sC, aD.sC, aE.sC, aF.sC);
    resD = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sD, a1.sD, a2.sD, a3.sD, a4.sD, a5.sD, a6.sD, a7.sD,
                                          a8.sD, a9.sD, aA.sD, aB.sD, aC.sD, aD.sD, aE.sD, aF.sD);
    resE = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sE, a1.sE, a2.sE, a3.sE, a4.sE, a5.sE, a6.sE, a7.sE,
                                          a8.sE, a9.sE, aA.sE, aB.sE, aC.sE, aD.sE, aE.sE, aF.sE);
    resF = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sF, a1.sF, a2.sF, a3.sF, a4.sF, a5.sF, a6.sF, a7.sF,
                                          a8.sF, a9.sF, aA.sF, aB.sF, aC.sF, aD.sF, aE.sF, aF.sF);
#endif // N0 > 8

#else // N0 == 16
#error "Not supported N0 value"
#endif // N0 > 2

    // ---------------------------Store the output values ------------------------------
    REPEAT_VAR_INIT_TO_CONST(16, uint, zout, 0);
    STORE_BLOCK(N0, K0, DATA_TYPE, res, output_ptr, OUTPUT_STEP_X * sizeof(DATA_TYPE), zout);

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}
#endif // defined(TRANSPOSE)
#endif // defined(K0) && defined(N0) && defined(H0) && defined(DATA_TYPE) && defined(SRC_HEIGHT)

#if defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE) && defined(M) && defined(N) && defined(K)

#define CONCAT(a, b) a##b

#define ARM_DOT1(a, b, c) \
    ({                    \
        c = fma(a, b, c); \
    })
#define ARM_DOT2(a, b, c)       \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
    })
#define ARM_DOT3(a, b, c)           \
    ({                              \
        ARM_DOT2(a, b, c);          \
        c = fma((a.s2), (b.s2), c); \
    })
#define ARM_DOT4(a, b, c)           \
    ({                              \
        ARM_DOT3(a, b, c);          \
        c = fma((a.s3), (b.s3), c); \
    })
#define ARM_DOT8(a, b, c)            \
    ({                               \
        ARM_DOT4((a.lo), (b.lo), c); \
        ARM_DOT4((a.hi), (b.hi), c); \
    })
#define ARM_DOT16(a, b, c)           \
    ({                               \
        ARM_DOT8((a.lo), (b.lo), c); \
        ARM_DOT8((a.hi), (b.hi), c); \
    })

#if N0 == 2
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(k0, a, b, c) \
    ({                             \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##0), (c.s0));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##1), (c.s1));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##2), (c.s2));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##3), (c.s3));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##4), (c.s4));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##5), (c.s5));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##6), (c.s6));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##7), (c.s7));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##8), (c.s8));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##9), (c.s9));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##A), (c.sA));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##B), (c.sB));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##C), (c.sC));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##D), (c.sD));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##E), (c.sE));     \
        CONCAT(ARM_DOT, k0)        \
        ((a), (b##F), (c.sF));     \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the block K0xN0 is transposed
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions (M,N and K) must be passed at compile time using -DM, -DN and and -DK (e.g. -DM=52, -DN=30 and -DK=90)
 * @note The number of columns of LHS matrix must be passed at compile time using -DK (e.g. -DK=64)
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
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                       Stride of the LHS matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS matrix
 * @param[in]  rhs_ptr                            Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                       Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                       Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the RHS reshaped matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                       Stride of the LHS matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad                (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_only_rhs_t(IMAGE_DECLARATION(lhs),
                                          IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                          IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                          IMAGE_DECLARATION(dst),
                                          uint lhs_stride_z,
                                          uint rhs_stride_z,
#if defined(BETA)
                                          uint bias_stride_z,
#endif //defined(BETA)
                                          uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                          ,
                                          uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                          ,
                                          uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
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

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

    // Compute RHS reshaped matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zlhs, 0); //uint zlhs0=0,zlhs1=0,zlhs2=0,... zlhs7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS reshaped matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_ptr, rhs_offset, RHS_STEP_X * sizeof(DATA_TYPE), zero);

        // Accumulate
        ARM_DOT_K0XN0(K0, a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(K0, a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(K0, a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(K0, a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(K0, a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(K0, a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(K0, a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(K0, a7, b, c7);
#endif // M0 > 7

        lhs_offset += K0 * sizeof(DATA_TYPE);
        rhs_offset += (N0 * RHS_STEP_X * RHS_STEP_LOOP) * sizeof(DATA_TYPE);
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, 1, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS reshaped matrix
        LOAD_BLOCK(N0, 1, DATA_TYPE, b, rhs_ptr, rhs_offset, RHS_STEP_X * sizeof(DATA_TYPE), zero);

        // Accumulate
        ARM_DOT_K0XN0(1, a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(1, a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(1, a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(1, a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(1, a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(1, a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(1, a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(1, a7, b, c7);
#endif // M0 > 7

        lhs_offset += sizeof(DATA_TYPE);
        rhs_offset += sizeof(DATA_TYPE);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}

#if defined(OPENCL_IMAGE_SUPPORT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices. The RHS matrix is stored in OpenCL image
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the block K0xN0 is transposed
 *
 * @note -DOPENCL_IMAGE_SUPPORT must be passed at compile time in order to compile this OpenCL kernel
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions (M,N and K) must be passed at compile time using -DM, -DN and and -DK (e.g. -DM=52, -DN=30 and -DK=90)
 * @note The height of the RHS matrix, defined before creating the OpenCL image object from the OpenCL buffer, should be passed at compile time using -DRHS_HEIGHT=<value> (e.g. -DRHS_HEIGHT=32)
 *       Since we cannot create a 3d image from a buffer, the third dimension could be collapsed with the second dimension so RHS_HEIGHT
 *       could be different from the value returned by get_image_height(rhs_img).
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 4, 8, 16
 *  - K0 = 4, 8, 16
 *  - H0 >= 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS matrix. Supported data type: F32
 * @param[in]  lhs_stride_x                       Stride of the LHS matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS matrix
 * @param[in]  rhs_img                            The RHS reshaped matrix as OpenCL image object. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                       Stride of the LHS matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad                (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_only_rhs_t_texture(IMAGE_DECLARATION(lhs),
                                                  __read_only image2d_t rhs_img,
#if defined(BETA)
                                                  IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                  IMAGE_DECLARATION(dst),
                                                  uint lhs_stride_z,
                                                  uint rhs_stride_z,
#if defined(BETA)
                                                  uint bias_stride_z,
#endif //defined(BETA)
                                                  uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                  ,
                                                  uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                  ,
                                                  uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                 )
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(K0)

#define LEFTOVER_K (K % K0)

    // Block size
#define RHS_BLOCK_SIZE (PIXEL_UNIT * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X (PIXEL_UNIT * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X PIXEL_UNIT
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (get_global_id(2) % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = get_global_id(2);
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (get_global_id(0) % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (get_global_id(0) / (uint)H0) + z_rhs * RHS_HEIGHT;

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0);

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix stored in a cl_image
        REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), b, 0);
        LOAD_TEXTURE2D(N0, PIXEL_UNIT, DATA_TYPE, b, rhs_img, x_rhs, y_rhs, RHS_STEP_X, 0);

        // Accumulate
        ARM_DOT_K0XN0(K0, a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(K0, a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(K0, a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(K0, a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(K0, a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(K0, a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(K0, a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(K0, a7, b, c7);
#endif // M0 > 7

        lhs_offset += K0 * sizeof(DATA_TYPE);
        x_rhs += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

#if LEFTOVER_K != 0
    // Note: We cannot read out-of-bound elements from the RHS matrix because
    // the RHS width is always multiple of K0. This is not be true for the LHS matrix

    union UNION_VEC_TYPE
    {
        DATA_TYPE s[K0];
        VEC_DATA_TYPE(DATA_TYPE, K0)
        v;
    };

    union UNION_VEC_TYPE a0 = {.v = 0 };
#if M0 > 1
    union UNION_VEC_TYPE a1 = {.v = 0 };
#endif // M0 > 1
#if M0 > 2
    union UNION_VEC_TYPE a2 = {.v = 0 };
#endif // M0 > 2
#if M0 > 3
    union UNION_VEC_TYPE a3 = {.v = 0 };
#endif // M0 > 3
#if M0 > 4
    union UNION_VEC_TYPE a4 = {.v = 0 };
#endif // M0 > 4
#if M0 > 5
    union UNION_VEC_TYPE a5 = {.v = 0 };
#endif // M0 > 5
#if M0 > 6
    union UNION_VEC_TYPE a6 = {.v = 0 };
#endif // M0 > 6
#if M0 > 7
    union UNION_VEC_TYPE a7 = {.v = 0 };
#endif // M0 > 7

    REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), b, 0);

    // Load from RHS matrix
    LOAD_TEXTURE2D(N0, PIXEL_UNIT, DATA_TYPE, b, rhs_img, x_rhs, y_rhs, RHS_STEP_X, 0);

    // Load from LHS matrix
    for(int k = 0; k < LEFTOVER_K; ++k)
    {
        a0.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zlhs0);
#if M0 > 1
        a1.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zlhs1);
#endif // M0 > 1
#if M0 > 2
        a2.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zlhs2);
#endif // M0 > 2
#if M0 > 3
        a3.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zlhs3);
#endif // M0 > 3
#if M0 > 4
        a4.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zlhs4);
#endif // M0 > 4
#if M0 > 5
        a5.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zlhs5);
#endif // M0 > 5
#if M0 > 6
        a6.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zlhs6);
#endif // M0 > 6
#if M0 > 7
        a7.s[k] = *(__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zlhs7);
#endif // M0 > 7

        lhs_offset += sizeof(DATA_TYPE);
    }

    // Accumulate
    ARM_DOT_K0XN0(K0, a0.v, b, c0);
#if M0 > 1
    ARM_DOT_K0XN0(K0, a1.v, b, c1);
#endif // M0 > 1
#if M0 > 2
    ARM_DOT_K0XN0(K0, a2.v, b, c2);
#endif // M0 > 2
#if M0 > 3
    ARM_DOT_K0XN0(K0, a3.v, b, c3);
#endif // M0 > 3
#if M0 > 4
    ARM_DOT_K0XN0(K0, a4.v, b, c4);
#endif // M0 > 4
#if M0 > 5
    ARM_DOT_K0XN0(K0, a5.v, b, c5);
#endif // M0 > 5
#if M0 > 6
    ARM_DOT_K0XN0(K0, a6.v, b, c6);
#endif // M0 > 6
#if M0 > 7
    ARM_DOT_K0XN0(K0, a7.v, b, c7);
#endif // M0 > 7

#endif // LEFTOVER_K != 0

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef LEFTOVER_K
#undef PIXEL_UNIT
}
#endif // defined(OPENCL_IMAGE_SUPPORT)

#define VFMA(a, b, c)     \
    ({                    \
        c = fma(a, b, c); \
    })

#if M0 == 1
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
    })
#elif M0 == 2 // M0 == 2
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
    })
#elif M0 == 3 // M0 == 3
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
    })
#elif M0 == 4 // M0 == 4
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
    })
#elif M0 == 5 // M0 == 5
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
    })
#elif M0 == 6 // M0 == 6
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
    })
#elif M0 == 7 // M0 == 7
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
    })
#elif M0 == 8 // M0 == 8
#define VFMA_M0xN0(i, a, b, c)                                        \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##7).s##i), b, (c##7)); \
    })
#else // M0 not supported
#error "M0 not supported"
#endif // M0 not supported

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the block K0xN0 is NOT transposed
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions (M,N and K) must be passed at compile time using -DM, -DN and and -DK (e.g. -DM=52, -DN=30 and -DK=90).
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
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                       Stride of the LHS matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS matrix
 * @param[in]  rhs_ptr                            Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                       Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                       Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the RHS reshaped matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                       Stride of the LHS matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad                (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_only_rhs_nt(IMAGE_DECLARATION(lhs),
                                           IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                           IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                           IMAGE_DECLARATION(dst),
                                           uint lhs_stride_z,
                                           uint rhs_stride_z,
#if defined(BETA)
                                           uint bias_stride_z,
#endif //defined(BETA)
                                           uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                           ,
                                           uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                           ,
                                           uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
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

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

    // Compute RHS reshaped matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(8, uint, zin, 0);   //uint zin0=0,zin1=0,zin2=0,... zin7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0); //uint zero0=0,zero1=0,zero2=0,... zero7=0;

#if defined(REINTERPRET_INPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zin, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(N0-1)=0;

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zin);

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 0 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(0, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 1 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(1, a, b0, c);
#if K0 > 2
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 2 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(2, a, b0, c);
#endif // K0 > 2
#if K0 > 3
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 3 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(3, a, b0, c);
#endif // K0 > 3
#if K0 > 4
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 4 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(4, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 5 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(5, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 6 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(6, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 7 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(7, a, b0, c);
#endif // K0 > 4
#if K0 > 8
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 8 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(8, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 9 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(9, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 10 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(A, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 11 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(B, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 12 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(C, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 13 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(D, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 14 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(E, a, b0, c);
        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 15 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(F, a, b0, c);
#endif // K0 > 8

        lhs_offset += K0 * sizeof(DATA_TYPE);
        rhs_offset += K0 * RHS_STEP_X * RHS_STEP_LOOP * sizeof(DATA_TYPE);
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zin0));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zin1));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zin2));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zin3));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a4 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zin4));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a5 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zin5));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a6 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zin6));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a7 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zin7));
#endif // M0 > 7

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        b0 = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 0 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VFMA_M0xN0(0, a, b0, c);

        lhs_offset += sizeof(DATA_TYPE);
        rhs_offset += RHS_STEP_X * sizeof(DATA_TYPE);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}

#if defined(OPENCL_IMAGE_SUPPORT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS is reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the block K0xN0 is NOT transposed
 *
 * @note -DOPENCL_IMAGE_SUPPORT must be passed at compile time in order to compile this OpenCL kernel
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions (M,N and K) must be passed at compile time using -DM, -DN and and -DK (e.g. -DM=52, -DN=30 and -DK=90).
 * @note The height of the RHS matrix, defined before creating the OpenCL image object from the OpenCL buffer, should be passed at compile time using -DRHS_HEIGHT=<value> (e.g. -DRHS_HEIGHT=32)
 *       Since we cannot create a 3d image from a buffer, the third dimension could be collapsed with the second dimension so RHS_HEIGHT
 *       could be different from the value returned by get_image_height(rhs_img).
 * @note The block's dimensions used for reshaping the RHS matrix (N0 and K0) must be passed at compile time using -DN0 and -DK0 (e.g. -DN0=8, -DK0=4).
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 4, 8, 16
 *  - K0 = 4, 8, 16
 *  - H0 >= 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS matrix. Supported data type: F32
 * @param[in]  lhs_stride_x                       Stride of the LHS matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS matrix
 * @param[in]  rhs_img                            The RHS reshaped matrix as OpenCL image object. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                       Stride of the LHS matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad                (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_only_rhs_nt_texture(IMAGE_DECLARATION(lhs),
                                                   __read_only image2d_t rhs_img,
#if defined(BETA)
                                                   IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                   IMAGE_DECLARATION(dst),
                                                   uint lhs_stride_z,
                                                   uint rhs_stride_z,
#if defined(BETA)
                                                   uint bias_stride_z,
#endif //defined(BETA)
                                                   uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                   ,
                                                   uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                   ,
                                                   uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                  )
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(N0)

    // Block size
#define RHS_BLOCK_SIZE ((K0) * (PIXEL_UNIT))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X ((PIXEL_UNIT) * (H0))
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (PIXEL_UNIT)
#endif // defined(RHS_INTERLEAVE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (z % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = z;
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (x % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (x / (uint)H0) + z_rhs * RHS_HEIGHT;

    REPEAT_VAR_INIT_TO_CONST(8, uint, zin, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zin, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0);

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zin);

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 0 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(0, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 1 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(1, a, b0, c);
#if K0 > 2
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 2 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(2, a, b0, c);
#endif // K0 > 2
#if K0 > 3
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 3 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(3, a, b0, c);
#endif // K0 > 3
#if K0 > 4
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 4 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(4, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 5 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(5, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 6 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(6, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 7 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(7, a, b0, c);
#endif // K0 > 4
#if K0 > 8
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 8 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(8, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 9 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(9, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 10 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(A, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 11 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(B, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 12 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(C, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 13 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(D, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 14 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(E, a, b0, c);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 15 * RHS_STEP_X), (y_rhs));
        VFMA_M0xN0(F, a, b0, c);
#endif // K0 > 8

        lhs_offset += K0 * sizeof(DATA_TYPE);
        x_rhs += K0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zin0));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zin1));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zin2));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zin3));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a4 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zin4));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a5 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zin5));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a6 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zin6));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a7 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zin7));
#endif // M0 > 7

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 0 * RHS_STEP_X), (y_rhs));

        VFMA_M0xN0(0, a, b0, c);

        lhs_offset += sizeof(DATA_TYPE);
        x_rhs += RHS_STEP_X;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(OPENCL_IMAGE_SUPPORT)
#endif // defined(M0) && defined(N0) && defined(K0) && defined(H0) && defined(DATA_TYPE) && defined(M) && defined(N) && defined(K)

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(DATA_TYPE) && defined(DATA_TYPE_ACCUMULATOR) && defined(M) && defined(N)

#if defined(MIXED_PRECISION)
#if K0 == 2
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
        c += a.s4 * b.s4;   \
        c += a.s5 * b.s5;   \
        c += a.s6 * b.s6;   \
        c += a.s7 * b.s7;   \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c) \
    ({                      \
        c += a.s0 * b.s0;   \
        c += a.s1 * b.s1;   \
        c += a.s2 * b.s2;   \
        c += a.s3 * b.s3;   \
        c += a.s4 * b.s4;   \
        c += a.s5 * b.s5;   \
        c += a.s6 * b.s6;   \
        c += a.s7 * b.s7;   \
        c += a.s8 * b.s8;   \
        c += a.s9 * b.s9;   \
        c += a.sA * b.sA;   \
        c += a.sB * b.sB;   \
        c += a.sC * b.sC;   \
        c += a.sD * b.sD;   \
        c += a.sE * b.sE;   \
        c += a.sF * b.sF;   \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0 conditions
#else  // defined(MIXED_PRECISION)
#if K0 == 2
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
    })
#elif K0 == 3 // K0 == 3
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
    })
#elif K0 == 4 // K0 == 4
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
    })
#elif K0 == 8 // K0 == 8
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
        c = fma(a.s4, b.s4, c); \
        c = fma(a.s5, b.s5, c); \
        c = fma(a.s6, b.s6, c); \
        c = fma(a.s7, b.s7, c); \
    })
#elif K0 == 16 // K0 == 16
#define ARM_DOT_K0(a, b, c)     \
    ({                          \
        c = fma(a.s0, b.s0, c); \
        c = fma(a.s1, b.s1, c); \
        c = fma(a.s2, b.s2, c); \
        c = fma(a.s3, b.s3, c); \
        c = fma(a.s4, b.s4, c); \
        c = fma(a.s5, b.s5, c); \
        c = fma(a.s6, b.s6, c); \
        c = fma(a.s7, b.s7, c); \
        c = fma(a.s8, b.s8, c); \
        c = fma(a.s9, b.s9, c); \
        c = fma(a.sA, b.sA, c); \
        c = fma(a.sB, b.sB, c); \
        c = fma(a.sC, b.sC, c); \
        c = fma(a.sD, b.sD, c); \
        c = fma(a.sE, b.sE, c); \
        c = fma(a.sF, b.sF, c); \
    })
#else // K0 not supported
#error "K0 value not supported"
#endif // K0 conditions
#endif // defined(MIXED_PRECISION)

#if defined(ARM_DOT_K0XN0)
#undef ARM_DOT_K0XN0
#endif // defined(ARM_DOT_K0XN0)

#if N0 == 2
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
    })
#elif N0 == 3 // N0 == 3
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
    })
#elif N0 == 4 // N0 == 4
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
    })
#elif N0 == 8 // N0 == 8
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
        ARM_DOT_K0((a), (b##4), (c.s4)); \
        ARM_DOT_K0((a), (b##5), (c.s5)); \
        ARM_DOT_K0((a), (b##6), (c.s6)); \
        ARM_DOT_K0((a), (b##7), (c.s7)); \
    })
#elif N0 == 16 // N0 == 16
#define ARM_DOT_K0XN0(a, b, c)           \
    ({                                   \
        ARM_DOT_K0((a), (b##0), (c.s0)); \
        ARM_DOT_K0((a), (b##1), (c.s1)); \
        ARM_DOT_K0((a), (b##2), (c.s2)); \
        ARM_DOT_K0((a), (b##3), (c.s3)); \
        ARM_DOT_K0((a), (b##4), (c.s4)); \
        ARM_DOT_K0((a), (b##5), (c.s5)); \
        ARM_DOT_K0((a), (b##6), (c.s6)); \
        ARM_DOT_K0((a), (b##7), (c.s7)); \
        ARM_DOT_K0((a), (b##8), (c.s8)); \
        ARM_DOT_K0((a), (b##9), (c.s9)); \
        ARM_DOT_K0((a), (b##A), (c.sA)); \
        ARM_DOT_K0((a), (b##B), (c.sB)); \
        ARM_DOT_K0((a), (b##C), (c.sC)); \
        ARM_DOT_K0((a), (b##D), (c.sD)); \
        ARM_DOT_K0((a), (b##E), (c.sE)); \
        ARM_DOT_K0((a), (b##F), (c.sF)); \
    })
#else // N0 not supported
#error "N0 value not supported"
#endif // N0 conditions

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be NOT transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be transposed
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The data type used for the accumulators must be passed at compile time using -DDATA_TYPE_ACCUMULATOR (e.g. -DDATA_TYPE_ACCUMULATOR=float)
 * @note The F16 computation also supports mixed precision through the option -DMIXED_PRECISION passed at compile time. If enabled, DATA_TYPE_ACCUMULATOR should be set to float
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions M, N and K must be passed at compile time using -DM, -DN and -DK (e.g. -DM=52, -DN=90 and -DK=24).
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (e.g. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - V0 >= 1
 *  - H0 >= 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix NOT reshaped
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS reshaped matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                       Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                            Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                       Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                       Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the RHS reshaped matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  k                                  Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_stride_z                       Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_lhs_nt_rhs_t(IMAGE_DECLARATION(lhs),
                                            IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                            IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                            IMAGE_DECLARATION(dst),
                                            uint k,
                                            uint lhs_stride_z,
                                            uint rhs_stride_z,
#if defined(BETA)
                                            uint bias_stride_z,
#endif //defined(BETA)
                                            uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                            ,
                                            uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                           )
{
    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (K0)
#define LHS_STEP_X ((K0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (K0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

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

#if defined(DUMMY_WORK_ITEMS)
    if((get_global_id(0) * N0 >= N) || (get_global_id(1) * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (get_global_id(1) % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (get_global_id(1) / V0) * (uint)lhs_stride_y +
                               (get_global_id(2) * lhs_stride_z);

    // Compute RHS matrix address
    __global uchar *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (get_global_id(0) % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (get_global_id(0) / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (get_global_id(2) % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += get_global_id(2) * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0); //uint zlhs0=0,zlhs1=0,zlhs2=0,... zlhs7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

    for(int i = 0; i < k; i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_addr, 0, LHS_STEP_X * sizeof(DATA_TYPE), zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(N0, K0, DATA_TYPE, b, rhs_addr, 0, RHS_STEP_X * sizeof(DATA_TYPE), zero);

        // Accumulate
        ARM_DOT_K0XN0(a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(a7, b, c7);
#endif // M0 > 7

        lhs_addr += (M0 * LHS_STEP_X * LHS_STEP_LOOP) * sizeof(DATA_TYPE);
        rhs_addr += (N0 * RHS_STEP_X * RHS_STEP_LOOP) * sizeof(DATA_TYPE);
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, get_global_id(1) * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += get_global_id(2) * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(1, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK_BROADCAST(M0, c, bias_hp0);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK_BROADCAST(M0, c, bias0);
#endif // defined(MIXED_PRECISION)

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * bias_stride_y) + get_global_id(
                                    2) * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK(M0, c, bias_hp);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK(M0, c, bias);
#endif // defined(MIXED_PRECISION)

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
#if defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, c, A_VAL, B_VAL);
#else  // defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(MIXED_PRECISION)
#endif // defined(ACTIVATION_TYPE)

    // Store output block
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE, c, c_lp);
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c_lp, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#else  // defined(MIXED_PRECISION)
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#endif // defined(MIXED_PRECISION)

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef LHS_STEP_LOOP
#undef RHS_STEP_LOOP
}

#if defined(OPENCL_IMAGE_SUPPORT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices. The RHS matrix is stored in OpenCL image object.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be NOT transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be transposed
 *
 * @note -DOPENCL_IMAGE_SUPPORT must be passed at compile time in order to compile this OpenCL kernel
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The data type used for the accumulators must be passed at compile time using -DDATA_TYPE_ACCUMULATOR (e.g. -DDATA_TYPE_ACCUMULATOR=float)
 * @note The F16 computation also supports mixed precision through the option -DMIXED_PRECISION passed at compile time. If enabled, DATA_TYPE_ACCUMULATOR should be set to float
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions M, N and K must be passed at compile time using -DM, -DN and -DK (e.g. -DM=52, -DN=90 and -DK=24).
 * @note The height of the RHS matrix, defined before creating the OpenCL image object from the OpenCL buffer, should be passed at compile time using -DRHS_HEIGHT=<value> (e.g. -DRHS_HEIGHT=32)
 *       Since we cannot create a 3d image from a buffer, the third dimension could be collapsed with the second dimension so RHS_HEIGHT
 *       could be different from the value returned by get_image_height(rhs_img).
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (e.g. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 4, 8, 16
 *  - K0 = 4, 8, 16
 *  - V0 >= 1
 *  - H0 >= 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix NOT reshaped
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS reshaped matrix. Supported data type: F32
 * @param[in]  lhs_stride_x                       Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_img                            The RHS reshaped matrix as OpenCL image object. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  k                                  Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_stride_z                       Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_lhs_nt_rhs_t_texture(IMAGE_DECLARATION(lhs),
                                                    __read_only image2d_t rhs_img,
#if defined(BETA)
                                                    IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                    IMAGE_DECLARATION(dst),
                                                    uint k,
                                                    uint lhs_stride_z,
                                                    uint rhs_stride_z,
#if defined(BETA)
                                                    uint bias_stride_z,
#endif //defined(BETA)
                                                    uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                    ,
                                                    uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                   )
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(K0)

    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (K0)
#define LHS_STEP_X ((K0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (K0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

    // Block size
#define RHS_BLOCK_SIZE (PIXEL_UNIT * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X (PIXEL_UNIT * (H0))
#define RHS_STEP_LOOP (1)
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X PIXEL_UNIT
#define RHS_STEP_LOOP (H0)
#endif // defined(RHS_INTERLEAVE)

#if defined(DUMMY_WORK_ITEMS)
    if((get_global_id(0) * N0 >= N) || (get_global_id(1) * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (get_global_id(1) % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (get_global_id(1) / V0) * (uint)lhs_stride_y +
                               (get_global_id(2) * lhs_stride_z);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (get_global_id(2) % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = get_global_id(2);
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (get_global_id(0) % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (get_global_id(0) / (uint)H0) + z_rhs * RHS_HEIGHT;

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0); //uint zlhs0=0,zlhs1=0,zlhs2=0,... zlhs7=0;
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

    for(int i = 0; i < K; i += K0)
    {
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_addr, 0, LHS_STEP_X * sizeof(DATA_TYPE), zlhs);

        // Load values from RHS matrix stored in a cl_image
        REPEAT_VAR_INIT_TO_CONST(N0, VEC_DATA_TYPE(DATA_TYPE, K0), b, 0);
        LOAD_TEXTURE2D(N0, PIXEL_UNIT, DATA_TYPE, b, rhs_img, x_rhs, y_rhs, RHS_STEP_X, 0);

        // Accumulate
        ARM_DOT_K0XN0(a0, b, c0);
#if M0 > 1
        ARM_DOT_K0XN0(a1, b, c1);
#endif // M0 > 1
#if M0 > 2
        ARM_DOT_K0XN0(a2, b, c2);
#endif // M0 > 2
#if M0 > 3
        ARM_DOT_K0XN0(a3, b, c3);
#endif // M0 > 3
#if M0 > 4
        ARM_DOT_K0XN0(a4, b, c4);
#endif // M0 > 4
#if M0 > 5
        ARM_DOT_K0XN0(a5, b, c5);
#endif // M0 > 5
#if M0 > 6
        ARM_DOT_K0XN0(a6, b, c6);
#endif // M0 > 6
#if M0 > 7
        ARM_DOT_K0XN0(a7, b, c7);
#endif // M0 > 7

        lhs_addr += (M0 * LHS_STEP_X * LHS_STEP_LOOP) * sizeof(DATA_TYPE);

        x_rhs += N0 * RHS_STEP_X * RHS_STEP_LOOP;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, get_global_id(1) * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += get_global_id(2) * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(1, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK_BROADCAST(M0, c, bias_hp0);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK_BROADCAST(M0, c, bias0);
#endif // defined(MIXED_PRECISION)

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * bias_stride_y) + get_global_id(
                                    2) * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK(M0, c, bias_hp);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK(M0, c, bias);
#endif // defined(MIXED_PRECISION)

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
#if defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, c, A_VAL, B_VAL);
#else  // defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(MIXED_PRECISION)
#endif // defined(ACTIVATION_TYPE)

    // Store output block
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE, c, c_lp);
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c_lp, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#else  // defined(MIXED_PRECISION)
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#endif // defined(MIXED_PRECISION)

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef PIXEL_UNIT
#undef LHS_STEP_LOOP
#undef RHS_STEP_LOOP
}
#endif // defined(OPENCL_IMAGE_SUPPORT)

#if defined(LHS_TRANSPOSE)

#define VTYPE(TYPE, SIZE) VEC_DATA_TYPE(TYPE, SIZE)

#if defined(MIXED_PRECISION)

#if(GPU_ARCH == GPU_ARCH_MIDGARD)
#define ARM_VFMA(N0, a, b, c) c += (CONVERT(a, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))) * (CONVERT(b, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0)));
#else // GPU_ARCH == GPU_ARCH_MIDGARD
#define ARM_VFMA(N0, a, b, c) c = fma((CONVERT(a, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))), (CONVERT(b, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0))), (c));
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

#else // defined(MIXED_PRECISION

#if(GPU_ARCH == GPU_ARCH_MIDGARD)
#define ARM_VFMA(N0, a, b, c) c += (a) * (b);
#else // GPU_ARCH == GPU_ARCH_MIDGARD
#define ARM_VFMA(N0, a, b, c) c = fma((a), (b), (c));
#endif // GPU_ARCH == GPU_ARCH_MIDGARD

#endif // defined(MIXED_PRECISION)

#define ARM_VVM_T_NT_1xN0x1(N0, TYPE, a, b, C)         \
    ({                                                 \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a), b, (C##0)); \
    })
#define ARM_VVM_T_NT_2xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s0), b, (C##0)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s1), b, (C##1)); \
    })
#define ARM_VVM_T_NT_3xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_2xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s2), b, (C##2)); \
    })
#define ARM_VVM_T_NT_4xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_3xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s3), b, (C##3)); \
    })
#define ARM_VVM_T_NT_8xN0x1(N0, TYPE, a, b, C)            \
    ({                                                    \
        ARM_VVM_T_NT_4xN0x1(N0, TYPE, a, b, C);           \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s4), b, (C##4)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s5), b, (C##5)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s6), b, (C##6)); \
        ARM_VFMA(N0, (VTYPE(TYPE, N0))(a.s7), b, (C##7)); \
    })

// Factory macro for the column-vector (transposed) by row-vector (not transposed) multiplication. K0 = 1
// a is the column-vector (transposed)
// b is the row-vector (not transposed)
// C is the output matrix
// Lower case is a vector (a, b)
// Upper case is a matrix (C)
#define ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, a, b, C) ARM_VVM_T_NT_##M0##xN0x1(N0, TYPE, a, b, C)

#define ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##0), (B##0), C); \
    })
#define ARM_MM_T_NT_M0xN0x2(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##1), (B##1), C); \
    })
#define ARM_MM_T_NT_M0xN0x3(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x2(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##2), (B##2), C); \
    })
#define ARM_MM_T_NT_M0xN0x4(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x3(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##3), (B##3), C); \
    })
#define ARM_MM_T_NT_M0xN0x8(M0, N0, TYPE, A, B, C)             \
    ({                                                         \
        ARM_MM_T_NT_M0xN0x4(M0, N0, TYPE, A, B, C);            \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##4), (B##4), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##5), (B##5), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##6), (B##6), C); \
        ARM_VVM_T_NT_M0xN0x1(M0, N0, TYPE, (A##7), (B##7), C); \
    })
#define ARM_MM_T_NT_M0xN0x16(M0, N0, TYPE, A, B, C)           \
    ({                                                        \
        ARM_MM_T_NT_M0xN0x8(M0, N0, TYPE, A, B, C);           \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##8), (B##8), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##9), (B##9), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##A), (B##A), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##B), (B##B), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##C), (B##C), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##D), (B##D), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##E), (B##E), C); \
        ARM_MM_T_NT_M0xN0x1(M0, N0, TYPE, (A##F), (B##F), C); \
    })

// Factory macro for the matrix (transposed) by matrix (not transposed) multiplication.
// The dimensions for this matrix multiplications are defined through M0, N0 and K0
// The dimensions supported are:
// M0: 1, 2, 3, 4, 8
// N0: 1, 2, 3, 4, 8, 16
// K0: 1, 2, 3, 4, 8, 16
// This macro calls the vector-by-matrix macro K0 times
// A, B and C are matrices
#define ARM_MM_T_NT(M0, N0, K0, TYPE, A, B, C) \
    CONCAT(ARM_MM_T_NT_M0xN0x, K0)             \
    (M0, N0, TYPE, A, B, C)

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be NOT transposed
 *
 * @note LHS_TRANSPOSE should be passed at compile time in order to compile this OpenCL kernel (e.g. -DLHS_TRANSPOSE).
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions M, N and K must be passed at compile time using -DM, -DN and -DK (e.g. -DM=52, -DN=90 and -DK=24).
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (e.g. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *  - V0 >= 1
 *  - H0 >= 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix NOT reshaped
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS reshaped matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                       Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                            Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                       Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                       Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the RHS reshaped matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  k                                  Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_stride_z                       Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_lhs_t_rhs_nt(IMAGE_DECLARATION(lhs),
                                            IMAGE_DECLARATION(rhs),
#if defined(BETA)
                                            IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                            IMAGE_DECLARATION(dst),
                                            uint k,
                                            uint lhs_stride_z,
                                            uint rhs_stride_z,
#if defined(BETA)
                                            uint bias_stride_z,
#endif //defined(BETA)
                                            uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                            ,
                                            uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                           )
{
    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (M0)
#define LHS_STEP_X ((M0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (M0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (N0)
#define RHS_STEP_X ((N0) * (H0))
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (N0)
#endif // defined(RHS_INTERLEAVE)

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint z = get_global_id(2);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (y % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (y / V0) * (uint)lhs_stride_y + (z * lhs_stride_z);

    // Compute RHS matrix address
    __global uchar *rhs_addr = rhs_ptr + rhs_offset_first_element_in_bytes + (x % H0) * (uint)RHS_OFFSET_X * sizeof(DATA_TYPE) + (x / (uint)H0) * rhs_stride_y;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_addr += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_addr += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

    __global DATA_TYPE *lhs = (__global DATA_TYPE *)(lhs_addr);
    __global DATA_TYPE *rhs = (__global DATA_TYPE *)(rhs_addr);

    for(int i = 0; i < k; i += K0)
    {
        VEC_DATA_TYPE(DATA_TYPE, M0)
        a0;
        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

#if K0 > 1
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 1

#if K0 > 2
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 2

#if K0 > 3
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 3

#if K0 > 4
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 4

#if K0 > 8
        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = VLOAD(N0)(0, rhs);

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
        rhs += RHS_STEP_X;
#endif // K0 > 8

#ifndef LHS_INTERLEAVE
        lhs += (M0 * K0 * (V0 - 1));
#endif // LHS_INTERLEAVE

#ifndef RHS_INTERLEAVE
        rhs += (N0 * K0 * (H0 - 1));
#endif // RHS_INTERLEAVE
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(1, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK_BROADCAST(M0, c, bias_hp0);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK_BROADCAST(M0, c, bias0);
#endif // defined(MIXED_PRECISION)

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE)) + (get_global_id(1) * (uint)M0 * bias_stride_y) + get_global_id(
                                    2) * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK(M0, c, bias_hp);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK(M0, c, bias);
#endif // defined(MIXED_PRECISION)

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
#if defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, c, A_VAL, B_VAL);
#else  // defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(MIXED_PRECISION)
#endif // defined(ACTIVATION_TYPE)

    // Store output block
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE, c, c_lp);
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c_lp, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#else  // defined(MIXED_PRECISION)
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#endif // defined(MIXED_PRECISION)

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}

#if defined(OPENCL_IMAGE_SUPPORT)
/** This OpenCL kernel computes the matrix multiplication between 2 matrices. The RHS matrix is stored in OpenCL image object.
 *  The LHS matrix must be reshaped with @ref CLGEMMReshapeLHSMatrixKernel and the M0xK0 must be transposed
 *  The RHS matrix must be reshaped with @ref CLGEMMReshapeRHSMatrixKernel and the K0xN0 must be NOT transposed
 *
 * @note -DOPENCL_IMAGE_SUPPORT must be passed at compile time in order to compile this OpenCL kernel
 * @note LHS_TRANSPOSE should be passed at compile time in order to compile this OpenCL kernel (e.g. -DLHS_TRANSPOSE).
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions M, N and K must be passed at compile time using -DM, -DN and -DK (e.g. -DM=52, -DN=90 and -DK=24).
 * @note The height of the RHS matrix, defined before creating the OpenCL image object from the OpenCL buffer, should be passed at compile time using -DRHS_HEIGHT=<value> (e.g. -DRHS_HEIGHT=32)
 *       Since we cannot create a 3d image from a buffer, the third dimension could be collapsed with the second dimension so RHS_HEIGHT
 *       could be different from the value returned by get_image_height(rhs_img).
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (e.g. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (e.g. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (e.g. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 8
 *  - N0 = 4, 8, 16
 *  - K0 = 4, 8, 16
 *  - V0 >= 1
 *  - H0 >= 1
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the output has to be reinterpreted as a 3D tensor (e.g. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix NOT reshaped
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS reshaped matrix. Supported data type: F32
 * @param[in]  lhs_stride_x                       Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_img                            The RHS reshaped matrix as cl_image 2d. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  k                                  Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_stride_z                       Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_lhs_t_rhs_nt_texture(IMAGE_DECLARATION(lhs),
                                                    __read_only image2d_t rhs_img,
#if defined(BETA)
                                                    IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                                                    IMAGE_DECLARATION(dst),
                                                    uint k,
                                                    uint lhs_stride_z,
                                                    uint rhs_stride_z,
#if defined(BETA)
                                                    uint bias_stride_z,
#endif //defined(BETA)
                                                    uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                    ,
                                                    uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                   )
{
    // Pixel unit
#define PIXEL_UNIT CONVERT_VECTOR_SIZE_TO_PIXEL_UNIT(N0)

    // Block size
#define LHS_BLOCK_SIZE ((K0) * (M0))

#if defined(LHS_INTERLEAVE)
#define LHS_OFFSET_X (M0)
#define LHS_STEP_X ((M0) * (V0))
#define LHS_STEP_LOOP (1)
#else // defined(INTERLEAVE)
#define LHS_OFFSET_X (LHS_BLOCK_SIZE)
#define LHS_STEP_X (M0)
#define LHS_STEP_LOOP (V0)
#endif // defined(INTERLEAVE)

    // Block size
#define RHS_BLOCK_SIZE ((K0) * (PIXEL_UNIT))

    // RHS offset and step X
#if defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (PIXEL_UNIT)
#define RHS_STEP_X ((PIXEL_UNIT) * (H0))
#else // defined(RHS_INTERLEAVE)
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)
#define RHS_STEP_X (PIXEL_UNIT)
#endif // defined(RHS_INTERLEAVE)

    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    __global uchar *lhs_addr = lhs_ptr + lhs_offset_first_element_in_bytes + (y % V0) * (uint)LHS_OFFSET_X * sizeof(DATA_TYPE) + (y / V0) * (uint)lhs_stride_y + (z * lhs_stride_z);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    const uint z_rhs = (z % MATRIX_B_DEPTH);
#else  // defined(MATRIX_B_DEPTH)
    const uint z_rhs = z;
#endif // defined(MATRIX_B_DEPTH)

    // Compute RHS matrix coordinates
    uint       x_rhs = (x % H0) * (uint)RHS_OFFSET_X;
    const uint y_rhs = (x / (uint)H0) + z_rhs * RHS_HEIGHT;

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE_ACCUMULATOR, N0), c, 0);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zero, 0);

    __global DATA_TYPE *lhs = (__global DATA_TYPE *)(lhs_addr);

    for(int i = 0; i < K; i += K0)
    {
        VEC_DATA_TYPE(DATA_TYPE, M0)
        a0;
        VEC_DATA_TYPE(DATA_TYPE, N0)
        b0;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 0 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

#if K0 > 1
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 1 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 1

#if K0 > 2
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 2 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 2

#if K0 > 3
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 3 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 3

#if K0 > 4
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 4 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 5 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 6 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 7 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 4

#if K0 > 8
        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 8 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 9 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 10 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 11 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 12 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 13 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 14 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;

        a0 = VLOAD(M0)(0, lhs);
        b0 = READ_IMAGE2D(DATA_TYPE, PIXEL_UNIT, rhs_img, (x_rhs + 15 * RHS_STEP_X), (y_rhs));

        ARM_MM_T_NT(M0, N0, 1, DATA_TYPE, a, b, c);

        lhs += LHS_STEP_X;
#endif // K0 > 8

#ifndef LHS_INTERLEAVE
        lhs += (M0 * K0 * (V0 - 1));
#endif // LHS_INTERLEAVE

        x_rhs += K0 * RHS_STEP_X;
#ifndef RHS_INTERLEAVE
        x_rhs += (PIXEL_UNIT * K0 * (H0 - 1));
#endif // RHS_INTERLEAVE
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (y * (uint)M0 * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

    const bool cond_y = ((get_global_id(1) + 1) * M0 >= M);
    const bool cond_x = ((get_global_id(0) + 1) * N0 >= N);

#if defined(REINTERPRET_OUTPUT_AS_3D)

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, y * (uint)M0, HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);
    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK_BOUNDARY_AWARE(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, 1, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(1, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK_BROADCAST(M0, c, bias_hp0);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK_BROADCAST(M0, c, bias0);
#endif // defined(MIXED_PRECISION)

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (y * (uint)M0 * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE_ACCUMULATOR, bias, bias_hp);
    ADD_BLOCK(M0, c, bias_hp);
#else  // defined(MIXED_PRECISION)
    ADD_BLOCK(M0, c, bias);
#endif // defined(MIXED_PRECISION)

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
#if defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE_ACCUMULATOR, VEC_SIZE, c, A_VAL, B_VAL);
#else  // defined(MIXED_PRECISION)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(MIXED_PRECISION)
#endif // defined(ACTIVATION_TYPE)

    // Store output block
#if defined(MIXED_PRECISION)
    CONVERT_BLOCK(M0, N0, DATA_TYPE, c, c_lp);
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c_lp, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#else  // defined(MIXED_PRECISION)
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);
#endif // defined(MIXED_PRECISION)

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
#undef PIXEL_UNIT
#undef LHS_STEP_LOOP
#undef RHS_STEP_LOOP
}
#endif // defined(OPENCL_IMAGE_SUPPORT)

#endif // defined(LHS_TRANSPOSE)

#endif // defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(K) && defined(DATA_TYPE)

#if defined(M0) && defined(N0) && defined(K0) && defined(K) && defined(DATA_TYPE)

#define VFMA(a, b, c)     \
    ({                    \
        c = fma(a, b, c); \
    })

#if M0 == 1
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
    })
#elif M0 == 2 // M0 == 2
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
    })
#elif M0 == 3 // M0 == 3
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
    })
#elif M0 == 4 // M0 == 4
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
    })
#elif M0 == 5 // M0 == 5
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
    })
#elif M0 == 6 // M0 == 6
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
    })
#elif M0 == 7 // M0 == 7
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
    })
#elif M0 == 8 // M0 == 8
#define RHS_VFMA_M0xN0(i, a, b, c)                                    \
    ({                                                                \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##0).s##i), b, (c##0)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##1).s##i), b, (c##1)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##2).s##i), b, (c##2)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##3).s##i), b, (c##3)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##4).s##i), b, (c##4)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##5).s##i), b, (c##5)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##6).s##i), b, (c##6)); \
        VFMA((VEC_DATA_TYPE(DATA_TYPE, N0))((a##7).s##i), b, (c##7)); \
    })
#else // M0 not supported
#error "M0 not supported"
#endif // M0 not supported

/** This OpenCL kernel computes the matrix multiplication between 2 matrices.
 *  The LHS matrix is NOT reshaped
 *  The RHS matrix is NOT reshaped
 *
 * @note If the first two dimensions of NDRange have been dispatched with "dummy_work_items" support, the option -DDUMMY_WORK_ITEMS must be passed at compile time.
 * @note The GEMM's dimensions (M,N and K) must be passed at compile time using -DM, -DN and and -DK (e.g. -DM=52, -DN=30 and -DK=90)
 * @note The number of columns of LHS matrix must be passed at compile time using -DK (e.g. -DK=64)
 * @note The number of M0 rows to process must be passed at compile time using -DM0 (e.g. -DM0=2)
 * @note The number of K0 partial accumulations must be passed at compile time using -DK0 (e.g., -DK0=2)
 * @note The number of N0 columns to process must be passed at compile time using -DN0 (e.g. -DN0=2)
 * @note The size of the partial store block in y must be passed at compile time using -DPARTIAL_STORE_M0 (e.g. -DPARTIAL_STORE_M0=1)
 * @note The size of the partial store block in x must be passed at compile time using -DPARTIAL_STORE_N0 (e.g. -DPARTIAL_STORE_N0=1)
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 1, 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *
 * @note If the activation type were passed at compile time through -DACTIVATION_TYPE (e.g. -DACTIVATION_TYPE=RELU), A, B variables, required by some activation functions, should be passed at compile time as well using -DA_VAL= and -DB_VAL= respectively.
 *       The activation function is performed after the bias addition
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix
 *
 * @param[in]  lhs_ptr                            Pointer to the LHS matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                       Stride of the LHS matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                         lhs_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                       Stride of the LHS matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                         lhs_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes  The offset of the first element in the LHS matrix
 * @param[in]  rhs_ptr                            Pointer to the RHS matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                       Stride of the RHS matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                         rhs_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                       Stride of the RHS matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                         rhs_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes  The offset of the first element in the RHS matrix
 * @param[in]  bias_ptr                           (Optional) Pointer to the bias matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  bias_stride_x                      (Optional) Stride of the bias matrix in X dimension (in bytes)
 * @param[in]  bias_step_x                        (Optional) bias_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  bias_stride_y                      (Optional) Stride of the bias matrix in Y dimension (in bytes)
 * @param[in]  bias_step_y                        (Optional) bias_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  bias_offset_first_element_in_bytes (Optional) The offset of the first element in the bias matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  lhs_stride_z                       Stride of the LHS matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                       Stride of the RHS matrix in Z dimension (in bytes)
 * @param[in]  bias_stride_z                      (Optional) Stride of the bias matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  lhs_cross_plane_pad                (Optional) Bottom paddings for LHS matrix in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings for the output matrix in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_native(IMAGE_DECLARATION(lhs),
                             IMAGE_DECLARATION(rhs),
#if defined(BETA)
                             IMAGE_DECLARATION(bias),
#endif // defined(BETA)
                             IMAGE_DECLARATION(dst),
                             uint lhs_stride_z,
                             uint rhs_stride_z,
#if defined(BETA)
                             uint bias_stride_z,
#endif //defined(BETA)
                             uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                             ,
                             uint lhs_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                             ,
                             uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                            )
{
    // Block size
#define RHS_BLOCK_SIZE ((K0) * (N0))

    // RHS offset and step X
#define RHS_OFFSET_X (RHS_BLOCK_SIZE)

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

#if defined(DUMMY_WORK_ITEMS)
    if((x * N0 >= N) || (y * M0 >= M))
    {
        return;
    }
#endif // defined(DUMMY_WORK_ITEMS)

    // Compute LHS matrix address
    uint lhs_offset = lhs_offset_first_element_in_bytes + COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * (uint)lhs_stride_y;

    // Compute RHS matrix address
    uint rhs_offset = rhs_offset_first_element_in_bytes + x * N0 * sizeof(DATA_TYPE);

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    rhs_offset += (z % MATRIX_B_DEPTH) * rhs_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    rhs_offset += z * rhs_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zlhs, 0);
    REPEAT_VAR_INIT_TO_CONST(16, uint, zero, 0);

#if defined(REINTERPRET_INPUT_AS_3D)
    // The plane (zlhs) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zlhs, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, lhs_cross_plane_pad, lhs_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply lhs_stride_z by DEPTH_GEMM3D
    lhs_offset += z * lhs_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    lhs_offset += z * lhs_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Initialize the accumulators
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    int i = 0;
    for(; i <= (K - K0); i += K0)
    {
        // Supported cases (M0, K0):
        // 1,2 - 1,3 - 1,4 - 1,8 - 1,16
        // 2,2 - 2,3 - 2,4 - 2,8 - 2,16
        // 3,2 - 3,3 - 3,4 - 3,8 - 3,16
        // 4,2 - 4,3 - 4,4 - 4,8 - 4,16
        // 5,2 - 5,3 - 5,4 - 5,8 - 5,16
        // 6,2 - 6,3 - 6,4 - 6,8 - 6,16
        // 7,2 - 7,3 - 7,4 - 7,8 - 7,16
        // 8,2 - 8,3 - 8,4 - 8,8 - 8,16
        // Load values from LHS matrix
        LOAD_BLOCK(M0, K0, DATA_TYPE, a, lhs_ptr, lhs_offset, lhs_stride_y, zlhs);

        // Load values from RHS matrix
        LOAD_BLOCK(K0, N0, DATA_TYPE, b, rhs_ptr, rhs_offset, rhs_stride_y, zero);

        RHS_VFMA_M0xN0(0, a, b0, c);
        RHS_VFMA_M0xN0(1, a, b1, c);
#if K0 > 2
        RHS_VFMA_M0xN0(2, a, b2, c);
#endif // K0 > 2
#if K0 > 3
        RHS_VFMA_M0xN0(3, a, b3, c);
#endif // K0 > 3
#if K0 > 4
        RHS_VFMA_M0xN0(4, a, b4, c);
        RHS_VFMA_M0xN0(5, a, b5, c);
        RHS_VFMA_M0xN0(6, a, b6, c);
        RHS_VFMA_M0xN0(7, a, b7, c);
#endif // K0 > 4
#if K0 > 8
        RHS_VFMA_M0xN0(8, a, b8, c);
        RHS_VFMA_M0xN0(9, a, b9, c);
        RHS_VFMA_M0xN0(A, a, bA, c);
        RHS_VFMA_M0xN0(B, a, bB, c);
        RHS_VFMA_M0xN0(C, a, bC, c);
        RHS_VFMA_M0xN0(D, a, bD, c);
        RHS_VFMA_M0xN0(E, a, bE, c);
        RHS_VFMA_M0xN0(F, a, bF, c);
#endif // K0 > 8

        lhs_offset += K0 * sizeof(DATA_TYPE);
        rhs_offset += K0 * rhs_stride_y;
    }

    // Left-over accumulations
    for(; i < K; ++i)
    {
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 0 * lhs_stride_y + zlhs0));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 1 * lhs_stride_y + zlhs1));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 2 * lhs_stride_y + zlhs2));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 3 * lhs_stride_y + zlhs3));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a4 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 4 * lhs_stride_y + zlhs4));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a5 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 5 * lhs_stride_y + zlhs5));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a6 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 6 * lhs_stride_y + zlhs6));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a7 = *((__global DATA_TYPE *)(lhs_ptr + lhs_offset + 7 * lhs_stride_y + zlhs7));
#endif // M0 > 7

        VEC_DATA_TYPE(DATA_TYPE, N0)
        b = VLOAD(N0)(0, (__global DATA_TYPE *)(rhs_ptr + rhs_offset + 0 * rhs_stride_y));
        RHS_VFMA_M0xN0(0, a, b, c);

        lhs_offset += sizeof(DATA_TYPE);
        rhs_offset += rhs_stride_y;
    }

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * dst_stride_y);

    REPEAT_VAR_INIT_TO_CONST(M0, uint, zout, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // The plane (zout) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    CALCULATE_Z_OFFSET(M0, uint, zout, COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0), HEIGHT_GEMM3D, DEPTH_GEMM3D, dst_cross_plane_pad, dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    SCALE_BLOCK(M0, DATA_TYPE, c, ALPHA);
#endif // defined(ALPHA)

    // Add beta*bias
#if defined(BETA)
#if defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (get_global_id(0) * (uint)N0 * sizeof(DATA_TYPE));

    LOAD_BLOCK(1, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(1, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias[broadcasted]
    ADD_BLOCK_BROADCAST(M0, c, bias0);

#else // defined(BROADCAST_BIAS)
    __global uchar *bias_addr = bias_ptr + bias_offset_first_element_in_bytes + (x * (uint)N0 * sizeof(DATA_TYPE)) + (COMPUTE_M0_START_ROW(y, M0, PARTIAL_STORE_M0) * bias_stride_y) + z * bias_stride_z;

    LOAD_BLOCK(M0, N0, DATA_TYPE, bias, bias_addr, 0, bias_stride_y, zero);

#ifndef UNIT_BETA
    SCALE_BLOCK(M0, DATA_TYPE, bias, BETA);
#endif // UNIT_BIAS

    // c = c + bias
    ADD_BLOCK(M0, c, bias);

#endif // defined(BROADCAST_BIAS)
#endif // defined(BETA)

#if defined(ACTIVATION_TYPE)
    ACTIVATION_BLOCK(M0, ACTIVATION_TYPE, DATA_TYPE, VEC_SIZE, c, A_VAL, B_VAL);
#endif // defined(ACTIVATION_TYPE)

    const bool cond_y = y == 0;
    const bool cond_x = ((x + 1) * N0 >= N);

    // Store output block
    STORE_BLOCK_BOUNDARY_AWARE(M0, N0, DATA_TYPE, c, dst_addr, dst_stride_y, zout, PARTIAL_STORE_M0, PARTIAL_STORE_N0, cond_y, cond_x);

#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(M0) && defined(N0) && defined(K0) && defined(K) && defined(DATA_TYPE)

#if defined(BETA)
/** This OpenCL kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @note The beta's value need to be passed at compile time using -DBETA
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_ma_f32(TENSOR3D_DECLARATION(src),
                          TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    // Load values from A x B
    float4 alpha_ab = vload4(0, (__global float *)dst.ptr);

    // Load values from Matrix C
    float4 c = vload4(0, (__global float *)src.ptr);

    // Computes alpha * axb + beta * c
    float4 out = alpha_ab + (float4)BETA * c;

    // Store final result in axb matrix
    vstore4(out, 0, (__global float *)dst.ptr);
}

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
/** This OpenCL kernel performs the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @note The beta's value need to be passed at compile time using -DBETA
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_ma_f16(TENSOR3D_DECLARATION(src),
                          TENSOR3D_DECLARATION(dst))
{
    // Compute source and destination addresses
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    // Load values from A x B
    half8 alpha_ab = vload8(0, (__global half *)dst.ptr);

    // Load values from Matrix C
    half8 c = vload8(0, (__global half *)src.ptr);

    // Computes alpha * axb + beta * c
    half8 out = alpha_ab + (half8)BETA * c;

    // Store final result in axb matrix
    vstore8(out, 0, (__global half *)dst.ptr);
}
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
#endif // defined(BETA)