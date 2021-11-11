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
    res0 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s0, a1.s0, a2.s0);
    res1 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s1, a1.s1, a2.s1);
#if N0 > 2
    res2 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s2, a1.s2, a2.s2);
#endif // N0 > 2
#if N0 > 3
    res3 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s3, a1.s3, a2.s3);
#endif // N0 > 3
#if N0 > 4
    res4 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s4, a1.s4, a2.s4);
    res5 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s5, a1.s5, a2.s5);
    res6 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s6, a1.s6, a2.s6);
    res7 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s7, a1.s7, a2.s7);
#endif // N0 > 4
#if N0 > 8
    res8 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s8, a1.s8, a2.s8);
    res9 = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.s9, a1.s9, a2.s9);
    resA = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sA, a1.sA, a2.sA);
    resB = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sB, a1.sB, a2.sB);
    resC = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sC, a1.sC, a2.sC);
    resD = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sD, a1.sD, a2.sD);
    resE = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sE, a1.sE, a2.sE);
    resF = (VEC_DATA_TYPE(DATA_TYPE, K0))(a0.sF, a1.sF, a2.sF);
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
