/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "repeat.h"

#if defined(M0) && defined(K0) && defined(V0) && defined(DATA_TYPE) && defined(SRC_WIDTH)
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

/** This OpenCL kernel reshapes the lhs input matrix. The kernel splits the input matrix in blocks of size M0xK0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=float)
 * @note The width of the input tensor must be passed at compile time using -DSRC_WIDTH (i.e. -DSRC_WIDTH=16)
 * @note The block's dimensions (M0 and K0) must be passed at compile time using -DM0 and -DK0 (i.e. -DM0=2, -DK0=2).
 * @note The number of M0xK0 vertical blocks to store on the same output row must be passed at compile time using -DV0 (i.e. -DV0=2)
 * @note Only the following values for M0, K0 and V0 are supported:
 *                                      M0: 2,3,4,5,6,7,8
 *                                      K0: 2,3,4,8,16
 *                                      V0: greater than 0
 * @note In case the input has to be reinterpreted as a 3D tensor (i.e. input of convolution layer 1x1), the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# HEIGHT_GEMM3D: The height of the input in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the input in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 * @note If the M0xK0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 *
 * @param[in]  src_ptr                           Pointer to the source LHS tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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

    // Note for the REINTERPRET_INPUT_AS_3D case
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    input_ptr += z * (uint)src_stride_z * DEPTH_GEMM3D;

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    zin0 = (0 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin0 = min((uint)(DEPTH_GEMM3D - 1), zin0);
    zin0 *= (cross_plane_pad * src_stride_y);
#if M0 > 1
    zin1 = (1 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin1 = min((uint)(DEPTH_GEMM3D - 1), zin1);
    zin1 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 1
#if M0 > 2
    zin2 = (2 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin2 = min((uint)(DEPTH_GEMM3D - 1), zin2);
    zin2 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 2
#if M0 > 3
    zin3 = (3 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin3 = min((uint)(DEPTH_GEMM3D - 1), zin3);
    zin3 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 3
#if M0 > 4
    zin4 = (4 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin4 = min((uint)(DEPTH_GEMM3D - 1), zin4);
    zin4 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 4
#if M0 > 5
    zin5 = (5 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin5 = min((uint)(DEPTH_GEMM3D - 1), zin5);
    zin5 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 5
#if M0 > 6
    zin6 = (6 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin6 = min((uint)(DEPTH_GEMM3D - 1), zin6);
    zin6 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 6
#if M0 > 7
    zin7 = (7 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin7 = min((uint)(DEPTH_GEMM3D - 1), zin7);
    zin7 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 7

#else // defined(REINTERPRET_INPUT_AS_3D)

    input_ptr += z * (uint)src_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    output_ptr += z * (uint)dst_stride_z;

    // ---------------------------Load input values --------------------------------

    // Load values from the LHS matrix
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a0 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y + zin0));
    BOUNDARY_CONDITION_X(x, a0);
#if M0 > 1
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a1 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y + zin1));
    BOUNDARY_CONDITION_X(x, a1);
#endif // M0 > 1
#if M0 > 2
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a2 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y + zin2));
    BOUNDARY_CONDITION_X(x, a2);
#endif // M0 > 2
#if M0 > 3
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a3 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 3 * src_stride_y + zin3));
    BOUNDARY_CONDITION_X(x, a3);
#endif // M0 > 3
#if M0 > 4
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a4 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 4 * src_stride_y + zin4));
    BOUNDARY_CONDITION_X(x, a4);
#endif // M0 > 4
#if M0 > 5
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a5 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 5 * src_stride_y + zin5));
    BOUNDARY_CONDITION_X(x, a5);
#endif // M0 > 5
#if M0 > 6
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a6 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 6 * src_stride_y + zin6));
    BOUNDARY_CONDITION_X(x, a6);
#endif // M0 > 6
#if M0 > 7
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a7 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 7 * src_stride_y + zin7));
    BOUNDARY_CONDITION_X(x, a7);
#endif // M0 > 7

    // ---------------------------Store output values ------------------------------

    VSTORE(K0)
    (a0, 0, (__global DATA_TYPE *)(output_ptr + 0 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#if M0 > 1
    VSTORE(K0)
    (a1, 0, (__global DATA_TYPE *)(output_ptr + 1 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 1
#if M0 > 2
    VSTORE(K0)
    (a2, 0, (__global DATA_TYPE *)(output_ptr + 2 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 2
#if M0 > 3
    VSTORE(K0)
    (a3, 0, (__global DATA_TYPE *)(output_ptr + 3 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 3
#if M0 > 4
    VSTORE(K0)
    (a4, 0, (__global DATA_TYPE *)(output_ptr + 4 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 4
#if M0 > 5
    VSTORE(K0)
    (a5, 0, (__global DATA_TYPE *)(output_ptr + 5 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 5
#if M0 > 6
    VSTORE(K0)
    (a6, 0, (__global DATA_TYPE *)(output_ptr + 6 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 6
#if M0 > 7
    VSTORE(K0)
    (a7, 0, (__global DATA_TYPE *)(output_ptr + 7 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 7

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
 * @note The data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=float)
 * @note The width of the input tensor must be passed at compile time using -DSRC_WIDTH (i.e. -DSRC_WIDTH=16)
 * @note The block's dimensions (M0 and K0) must be passed at compile time using -DM0 and -DK0 (i.e. -DM0=2, -DK0=2).
 * @note The number of M0xK0 vertical blocks to store on the same output row must be passed at compile time using -DV0 (i.e. -DV0=2)
 * @note Only the following values for M0, K0 and V0 are supported:
 *                                      M0: 2,3,4,5,6,7,8
 *                                      K0: 2,3,4,8,16
 *                                      V0: greater than 0
 * @note In case the input has to be reinterpreted as a 3D tensor (i.e. input of convolution layer 1x1), the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# HEIGHT_GEMM3D: The height of the input in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the input in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 * @note If the M0xK0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 *
 * @param[in]  src_ptr                           Pointer to the source LHS tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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

    // Note for the REINTERPRET_INPUT_AS_3D case
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    input_ptr += z * (uint)src_stride_z * DEPTH_GEMM3D;

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    zin0 = (0 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin0 = min((uint)(DEPTH_GEMM3D - 1), zin0);
    zin0 *= (cross_plane_pad * src_stride_y);
#if M0 > 1
    zin1 = (1 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin1 = min((uint)(DEPTH_GEMM3D - 1), zin1);
    zin1 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 1
#if M0 > 2
    zin2 = (2 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin2 = min((uint)(DEPTH_GEMM3D - 1), zin2);
    zin2 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 2
#if M0 > 3
    zin3 = (3 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin3 = min((uint)(DEPTH_GEMM3D - 1), zin3);
    zin3 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 3
#if M0 > 4
    zin4 = (4 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin4 = min((uint)(DEPTH_GEMM3D - 1), zin4);
    zin4 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 4
#if M0 > 5
    zin5 = (5 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin5 = min((uint)(DEPTH_GEMM3D - 1), zin5);
    zin5 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 5
#if M0 > 6
    zin6 = (6 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin6 = min((uint)(DEPTH_GEMM3D - 1), zin6);
    zin6 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 6
#if M0 > 7
    zin7 = (7 + (uint)(y * M0)) / (uint)HEIGHT_GEMM3D;
    zin7 = min((uint)(DEPTH_GEMM3D - 1), zin7);
    zin7 *= (cross_plane_pad * src_stride_y);
#endif // M0 > 7

#else // defined(REINTERPRET_INPUT_AS_3D)

    input_ptr += z * (uint)src_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    output_ptr += z * (uint)dst_stride_z;

    // ---------------------------Load input values --------------------------------

    // Load values from the LHS matrix
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a0 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y + zin0));
    BOUNDARY_CONDITION_X(x, a0);
#if M0 > 1
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a1 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y + zin1));
    BOUNDARY_CONDITION_X(x, a1);
#endif // M0 > 1
#if M0 > 2
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a2 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y + zin2));
    BOUNDARY_CONDITION_X(x, a2);
#endif // M0 > 2
#if M0 > 3
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a3 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 3 * src_stride_y + zin3));
    BOUNDARY_CONDITION_X(x, a3);
#endif // M0 > 3
#if M0 > 4
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a4 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 4 * src_stride_y + zin4));
    BOUNDARY_CONDITION_X(x, a4);
#endif // M0 > 4
#if M0 > 5
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a5 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 5 * src_stride_y + zin5));
    BOUNDARY_CONDITION_X(x, a5);
#endif // M0 > 5
#if M0 > 6
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a6 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 6 * src_stride_y + zin6));
    BOUNDARY_CONDITION_X(x, a6);
#endif // M0 > 6
#if M0 > 7
    VEC_DATA_TYPE(DATA_TYPE, K0)
    a7 = VLOAD(K0)(0, (__global DATA_TYPE *)(input_ptr + 7 * src_stride_y + zin7));
    BOUNDARY_CONDITION_X(x, a7);
#endif // M0 > 7

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
#endif // defined(M0) && defined(K0) && defined(V0) && defined(DATA_TYPE) && defined(SRC_WIDTH)

#if defined(K0) && defined(N0) && defined(H0) && defined(DATA_TYPE) && defined(SRC_HEIGHT)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (not transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=float)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (i.e. -DSRC_HEIGHT=16)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (i.e. -DK0=2, -DN0=2).
 * @note The number of K0xN0 vertical blocks to store on the same output row must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the K0xN0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 * @note Only the following values for K0, N0 and H0 are supported:
 *                                      N0: 2,3,4,8,16
 *                                      K0: 1,2,3,4,8,16
 *                                      H0: greater than 0
 *
 * @param[in]  src_ptr                           Pointer to the source RHS tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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
    VSTORE(N0)
    (a0, 0, (__global DATA_TYPE *)(output_ptr + 0 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#if K0 > 1
    VSTORE(N0)
    (a1, 0, (__global DATA_TYPE *)(output_ptr + 1 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // K0 > 1
#if K0 > 2
    VSTORE(N0)
    (a2, 0, (__global DATA_TYPE *)(output_ptr + 2 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // K0 > 2
#if K0 > 3
    VSTORE(N0)
    (a3, 0, (__global DATA_TYPE *)(output_ptr + 3 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // K0 > 3
#if K0 > 4
    VSTORE(N0)
    (a4, 0, (__global DATA_TYPE *)(output_ptr + 4 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (a5, 0, (__global DATA_TYPE *)(output_ptr + 5 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (a6, 0, (__global DATA_TYPE *)(output_ptr + 6 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (a7, 0, (__global DATA_TYPE *)(output_ptr + 7 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 4
#if K0 > 8
    VSTORE(N0)
    (a8, 0, (__global DATA_TYPE *)(output_ptr + 8 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (a9, 0, (__global DATA_TYPE *)(output_ptr + 9 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (aA, 0, (__global DATA_TYPE *)(output_ptr + 10 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (aB, 0, (__global DATA_TYPE *)(output_ptr + 11 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (aC, 0, (__global DATA_TYPE *)(output_ptr + 12 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (aD, 0, (__global DATA_TYPE *)(output_ptr + 13 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (aE, 0, (__global DATA_TYPE *)(output_ptr + 14 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(N0)
    (aF, 0, (__global DATA_TYPE *)(output_ptr + 15 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 8

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}

#if defined(TRANSPOSE)
/** This OpenCL kernel reshapes the rhs input matrix. The kernel splits the input matrix in blocks of size K0xN0 and stores each one (transposed) in
 *  the output matrix unrolling the values.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=float)
 * @note The height of the input tensor must be passed at compile time using -DSRC_HEIGHT (i.e. -DSRC_HEIGHT=16)
 * @note The block's dimensions (K0 and N0) must be passed at compile time using -DK0 and -DN0 (i.e. -DK0=2, -DN0=2).
 * @note The number of K0xN0 vertical blocks to store on the same output row must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the K0xN0 blocks have to be interleaved, the option -DINTERLEAVE must passed at compile time.
 * @note The option -DTRANSPOSE must passed at compile time.
 * @note Only the following values for K0, N0 and H0 are supported:
 *                                      N0: 2,3,4,8,16
 *                                      K0: 2,3,4,8,16
 *                                      H0: greater than 0
 *
 * @param[in]  src_ptr                           Pointer to the source RHS tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
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

    VSTORE(K0)
    (res0, 0, (__global DATA_TYPE *)(output_ptr + 0 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (res1, 0, (__global DATA_TYPE *)(output_ptr + 1 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#if N0 > 2
    VSTORE(K0)
    (res2, 0, (__global DATA_TYPE *)(output_ptr + 2 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 2
#if N0 > 3
    VSTORE(K0)
    (res3, 0, (__global DATA_TYPE *)(output_ptr + 3 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 3
#if N0 > 4
    VSTORE(K0)
    (res4, 0, (__global DATA_TYPE *)(output_ptr + 4 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (res5, 0, (__global DATA_TYPE *)(output_ptr + 5 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (res6, 0, (__global DATA_TYPE *)(output_ptr + 6 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (res7, 0, (__global DATA_TYPE *)(output_ptr + 7 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 4
#if N0 > 8
    VSTORE(K0)
    (res8, 0, (__global DATA_TYPE *)(output_ptr + 8 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (res9, 0, (__global DATA_TYPE *)(output_ptr + 9 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (resA, 0, (__global DATA_TYPE *)(output_ptr + 10 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (resB, 0, (__global DATA_TYPE *)(output_ptr + 11 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (resC, 0, (__global DATA_TYPE *)(output_ptr + 12 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (resD, 0, (__global DATA_TYPE *)(output_ptr + 13 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (resE, 0, (__global DATA_TYPE *)(output_ptr + 14 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
    VSTORE(K0)
    (resF, 0, (__global DATA_TYPE *)(output_ptr + 15 * OUTPUT_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 8

#undef BLOCK_SIZE
#undef OUTPUT_OFFSET_X
#undef OUTPUT_STEP_X
}
#endif // defined(TRANSPOSE)
#endif // defined(K0) && defined(N0) && defined(H0) && defined(DATA_TYPE) && defined(SRC_HEIGHT)

#if defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(DATA_TYPE)

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
 * @note The block's dimensions used for reshaping the LHS matrix and the RHS matrix (M0, N0 and K0) must be passed at compile time using -DM0, -DN0 and -DK0 (i.e. -DM0=4, -DN0=8, -DK0=4).
 * @note The number of M0xK0 vertical blocks stored on the same output row of the reshaped LHS matrix must be passed at compile time using -DV0 (i.e. -DV0=2)
 * @note The number of K0xN0 horizontal blocks stored on the same output row of the reshaped RHS matrix must be passed at compile time using -DH0 (i.e. -DH0=2)
 * @note If the M0xK0 blocks in the reshaped LHS matrix have been interleaved, the option -DLHS_INTERLEAVE must passed at compile time.
 * @note If the K0xN0 blocks in the reshaped RHS matrix have been interleaved, the option -DRHS_INTERLEAVE must passed at compile time.
 * @note Only the following configurations of M0, N0 and K0 are currently supported:
 *  - M0 = 2, 3, 4, 5, 6, 7, 8
 *  - N0 = 2, 3, 4, 8, 16
 *  - K0 = 2, 3, 4, 8, 16
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns LHS matrix NOT reshaped
 *
 * @param[in]  lhs_ptr                           Pointer to the LHS reshaped matrix. Supported data type: F16/F32
 * @param[in]  lhs_stride_x                      Stride of the LHS reshaped matrix in X dimension (in bytes)
 * @param[in]  lhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  lhs_stride_y                      Stride of the LHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  lhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  lhs_offset_first_element_in_bytes The offset of the first element in the LHS reshaped matrix
 * @param[in]  rhs_ptr                           Pointer to the RHS reshaped matrix. Supported data type: same as @p lhs_ptr
 * @param[in]  rhs_stride_x                      Stride of the RHS reshaped matrix in X dimension (in bytes)
 * @param[in]  rhs_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rhs_stride_y                      Stride of the RHS reshaped matrix in Y dimension (in bytes)
 * @param[in]  rhs_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rhs_offset_first_element_in_bytes The offset of the first element in the RHS reshaped matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data type: same as @p lhs_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  k                                 Number of columns in LHS matrix and rows in RHS matrix not reshaped.
 * @param[in]  lhs_stride_z                      Stride of the LHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  rhs_stride_z                      Stride of the RHS reshaped matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_cross_plane_pad               (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_reshaped_lhs_nt_rhs_t(IMAGE_DECLARATION(lhs),
                                            IMAGE_DECLARATION(rhs),
                                            IMAGE_DECLARATION(dst),
                                            uint k,
                                            uint lhs_stride_z,
                                            uint rhs_stride_z,
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
    REPEAT_VAR_INIT_TO_CONST(M0, VEC_DATA_TYPE(DATA_TYPE, N0), c, 0); //VEC_DATA_TYPE(DATA_TYPE, N0)    c0=0,c1=0,c2=0,... c(M0-1)=0;

    for(int i = 0; i < k; i += K0)
    {
        // Supported cases (M0, K0):
        // 2,4 - 2,8 - 2,16
        // 3,4 - 3,8 - 3,16
        // 4,4 - 4,8 - 4,16
        // 5,4 - 5,8 - 5,16
        // 6,4 - 6,8 - 6,16
        // Load values from LHS matrix
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a0 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 0 * LHS_STEP_X * sizeof(DATA_TYPE)));
#if M0 > 1
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a1 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 1 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 1
#if M0 > 2
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a2 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 2 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 2
#if M0 > 3
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a3 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 3 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 3
#if M0 > 4
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a4 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 4 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 4
#if M0 > 5
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a5 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 5 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 5
#if M0 > 6
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a6 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 6 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 6
#if M0 > 7
        VEC_DATA_TYPE(DATA_TYPE, K0)
        a7 = VLOAD(K0)(0, (__global DATA_TYPE *)(lhs_addr + 7 * LHS_STEP_X * sizeof(DATA_TYPE)));
#endif // M0 > 7

        // Load values from RHS matrix
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b0 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 0 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b1 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 1 * RHS_STEP_X * sizeof(DATA_TYPE)));
#if N0 > 2
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b2 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 2 * RHS_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 2
#if N0 > 3
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b3 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 3 * RHS_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 3
#if N0 > 4
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b4 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 4 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b5 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 5 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b6 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 6 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b7 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 7 * RHS_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 4
#if N0 > 8
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b8 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 8 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        b9 = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 9 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        bA = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 10 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        bB = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 11 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        bC = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 12 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        bD = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 13 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        bE = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 14 * RHS_STEP_X * sizeof(DATA_TYPE)));
        VEC_DATA_TYPE(DATA_TYPE, K0)
        bF = VLOAD(K0)(0, (__global DATA_TYPE *)(rhs_addr + 15 * RHS_STEP_X * sizeof(DATA_TYPE)));
#endif // N0 > 8

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

    REPEAT_VAR_INIT_TO_CONST(8, uint, zout, 0); //uint zout0=0,zout1=0,zout2=0,... zout7=0;

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (y * M0) by HEIGHT_GEMM3D
    zout0 = (0 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout0 = min((uint)(DEPTH_GEMM3D - 1), zout0);
    zout0 *= (dst_cross_plane_pad * dst_stride_y);
#if M0 > 1
    zout1 = (1 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout1 = min((uint)(DEPTH_GEMM3D - 1), zout1);
    zout1 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 1
#if M0 > 2
    zout2 = (2 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout2 = min((uint)(DEPTH_GEMM3D - 1), zout2);
    zout2 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 2
#if M0 > 3
    zout3 = (3 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout3 = min((uint)(DEPTH_GEMM3D - 1), zout3);
    zout3 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 3
#if M0 > 4
    zout4 = (4 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout4 = min((uint)(DEPTH_GEMM3D - 1), zout4);
    zout4 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 4
#if M0 > 5
    zout5 = (5 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout5 = min((uint)(DEPTH_GEMM3D - 1), zout5);
    zout5 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 5
#if M0 > 6
    zout6 = (6 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout6 = min((uint)(DEPTH_GEMM3D - 1), zout6);
    zout6 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 6
#if M0 > 7
    zout7 = (7 + (uint)(get_global_id(1) * (uint)M0)) / (uint)HEIGHT_GEMM3D;
    zout7 = min((uint)(DEPTH_GEMM3D - 1), zout7);
    zout7 *= (dst_cross_plane_pad * dst_stride_y);
#endif // M0 > 7

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += get_global_id(2) * dst_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_OUTPUT_AS_3D)

    // Add offset for batched GEMM
    dst_addr += get_global_id(2) * dst_stride_z;

#endif // defined(REINTERPRET_OUTPUT_AS_3D)

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    c0 = c0 * (DATA_TYPE)ALPHA;
#if M0 > 1
    c1 = c1 * (DATA_TYPE)ALPHA;
#endif // M0 > 1
#if M0 > 2
    c2 = c2 * (DATA_TYPE)ALPHA;
#endif // M0 > 2
#if M0 > 3
    c3 = c3 * (DATA_TYPE)ALPHA;
#endif // M0 > 3
#if M0 > 4
    c4 = c4 * (DATA_TYPE)ALPHA;
#endif // M0 > 4
#if M0 > 5
    c5 = c5 * (DATA_TYPE)ALPHA;
#endif // M0 > 5
#if M0 > 6
    c6 = c6 * (DATA_TYPE)ALPHA;
#endif // M0 > 5
#if M0 > 7
    c7 = c7 * (DATA_TYPE)ALPHA;
#endif // M0 > 7
#endif // defined(ALPHA)

    // Store output block
    VSTORE(N0)
    (c0, 0, (__global DATA_TYPE *)(dst_addr + 0 * dst_stride_y + zout0));
#if M0 > 1
    VSTORE(N0)
    (c1, 0, (__global DATA_TYPE *)(dst_addr + 1 * dst_stride_y + zout1));
#endif // M0 > 1
#if M0 > 2
    VSTORE(N0)
    (c2, 0, (__global DATA_TYPE *)(dst_addr + 2 * dst_stride_y + zout2));
#endif // M0 > 2
#if M0 > 3
    VSTORE(N0)
    (c3, 0, (__global DATA_TYPE *)(dst_addr + 3 * dst_stride_y + zout3));
#endif // M0 > 3
#if M0 > 4
    VSTORE(N0)
    (c4, 0, (__global DATA_TYPE *)(dst_addr + 4 * dst_stride_y + zout4));
#endif // M0 > 4
#if M0 > 5
    VSTORE(N0)
    (c5, 0, (__global DATA_TYPE *)(dst_addr + 5 * dst_stride_y + zout5));
#endif // M0 > 5
#if M0 > 6
    VSTORE(N0)
    (c6, 0, (__global DATA_TYPE *)(dst_addr + 6 * dst_stride_y + zout6));
#endif // M0 > 6
#if M0 > 7
    VSTORE(N0)
    (c7, 0, (__global DATA_TYPE *)(dst_addr + 7 * dst_stride_y + zout7));
#endif // M0 > 7

#undef LHS_BLOCK_SIZE
#undef LHS_OFFSET_X
#undef LHS_STEP_X
#undef RHS_BLOCK_SIZE
#undef RHS_OFFSET_X
#undef RHS_STEP_X
}
#endif // defined(M0) && defined(N0) && defined(K0) && defined(V0) && defined(H0) && defined(K) && defined(DATA_TYPE)

#if defined(TRANSPOSE_W) && defined(MULT_TRANSPOSE1XW_WIDTH)

#if ELEMENT_SIZE == 1
#define DATA_TYPE uchar
#elif ELEMENT_SIZE == 2
#define DATA_TYPE ushort
#elif ELEMENT_SIZE == 4
#define DATA_TYPE uint
#else // ELEMENT_SIZE == 1
#error "Element size not supported"
#endif // ELEMENT_SIZE

/** This OpenCL kernel computes the "vector" 1xW transposition of input matrix
 *
 * @note The transposition width must be passed at compile time using -DTRANSPOSE_W (i.e. -DTRANSPOSE_W)
 * @note The multiplication factor for the transposition width (mult_transpose1xW_width) must be passed at compile time using -DMULT_TRANSPOSE1XW_WIDTH (i.e. -DMULT_TRANSPOSE1XW_WIDTH=2)
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 */
__kernel void gemm_transpose1xW(TENSOR3D_DECLARATION(src),
                                TENSOR3D_DECLARATION(dst))
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // Compute address for Matrix B - source
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    // Compute address for Matrix B transposed - destination. X and Y are swapped
    uint dst_addr_in_bytes = dst_offset_first_element_in_bytes + y * TRANSPOSE_W * sizeof(DATA_TYPE) * MULT_TRANSPOSE1XW_WIDTH + (x / MULT_TRANSPOSE1XW_WIDTH) * dst_stride_y +
                             (x % MULT_TRANSPOSE1XW_WIDTH) * TRANSPOSE_W * sizeof(DATA_TYPE);

    // Add offset for batched GEMM
    dst_addr_in_bytes += z * dst_stride_z;

    VEC_DATA_TYPE(DATA_TYPE, TRANSPOSE_W)
    b0 = VLOAD(TRANSPOSE_W)(0, (__global DATA_TYPE *)src.ptr);

    VSTORE(TRANSPOSE_W)
    (b0, 0, (__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes));
}
#endif // defined(TRANSPOSE_W) && defined(MULT_TRANSPOSE1XW_WIDTH)

#if defined(MULT_INTERLEAVE4X4_HEIGHT) && defined(DATA_TYPE)

/** This OpenCL kernel reshapes the input matrix transposing each 4x4 block. If -DUNROLL_BLOCK is passed at compile time, the 4x4 block
 * will be simply unrolled.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE (i.e. -DDATA_TYPE=float)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note In case the input has to be reinterpreted as a 3D tensor (i.e. input of convolution layer 1x1), the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# HEIGHT_GEMM3D: The height of the input in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the input in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @param[in]  src_ptr                           Pointer to the source matrix. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  src_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                           Pointer to the destination matrix Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination matrix
 * @param[in]  cross_plane_pad                   (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_INPUT_AS_3D)
 */
__kernel void gemm_interleave4x4(TENSOR3D_DECLARATION(src),
                                 TENSOR3D_DECLARATION(dst)
#if defined(REINTERPRET_INPUT_AS_3D)
                                 ,
                                 uint cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
                                )
{
    // Compute source and destination addresses
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);

    // Compute address for source tensor
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    // Compute address for Matrix B transposed - destination. X and Y are swapped
    uint dst_addr_in_bytes = dst_offset_first_element_in_bytes + x * sizeof(DATA_TYPE) * 16 * MULT_INTERLEAVE4X4_HEIGHT + (y / MULT_INTERLEAVE4X4_HEIGHT) * dst_stride_y +
                             (y % MULT_INTERLEAVE4X4_HEIGHT) * 4 * sizeof(DATA_TYPE);

    // Add offset for batched GEMM
    dst_addr_in_bytes += z * dst_stride_z;

#if defined(REINTERPRET_INPUT_AS_3D)
    __global uchar *input_ptr = src_ptr + src_offset_first_element_in_bytes + x * 4 * sizeof(DATA_TYPE) + y * 4 * src_stride_y;

    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (y * 4) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(y * 4)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (cross_plane_pad * src_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src_stride_z by DEPTH_GEMM3D
    input_ptr += z * src_stride_z * DEPTH_GEMM3D;

    // Load values from Matrix A
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a0 = vload4(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y + zin.s0));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a1 = vload4(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y + zin.s1));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a2 = vload4(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y + zin.s2));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a3 = vload4(0, (__global DATA_TYPE *)(input_ptr + 3 * src_stride_y + zin.s3));
#else  // defined(REINTERPRET_INPUT_AS_3D)
    __global uchar *input_ptr = src.ptr;

    // Load values from Matrix A
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a0 = vload4(0, (__global DATA_TYPE *)(input_ptr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a1 = vload4(0, (__global DATA_TYPE *)(input_ptr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a2 = vload4(0, (__global DATA_TYPE *)(input_ptr + 2 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    a3 = vload4(0, (__global DATA_TYPE *)(input_ptr + 3 * src_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(UNROLL_BLOCK)
    vstore4(a0, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 0 * MULT_INTERLEAVE4X4_HEIGHT));
    vstore4(a1, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 4 * MULT_INTERLEAVE4X4_HEIGHT));
    vstore4(a2, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 8 * MULT_INTERLEAVE4X4_HEIGHT));
    vstore4(a3, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 12 * MULT_INTERLEAVE4X4_HEIGHT));
#else  // defined(UNROLL_BLOCK)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    val0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s0, a1.s0, a2.s0, a3.s0);
    vstore4(val0, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 0 * MULT_INTERLEAVE4X4_HEIGHT));

    val0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s1, a1.s1, a2.s1, a3.s1);
    vstore4(val0, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 4 * MULT_INTERLEAVE4X4_HEIGHT));

    val0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s2, a1.s2, a2.s2, a3.s2);
    vstore4(val0, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 8 * MULT_INTERLEAVE4X4_HEIGHT));

    val0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(a0.s3, a1.s3, a2.s3, a3.s3);
    vstore4(val0, 0, ((__global DATA_TYPE *)(dst_ptr + dst_addr_in_bytes) + 12 * MULT_INTERLEAVE4X4_HEIGHT));
#endif // defined(UNROLL_BLOCK)
}
#endif // defined(MULT_INTERLEAVE4X4_HEIGHT) && defined(DATA_TYPE)

#if defined(COLS_B) && defined(MULT_TRANSPOSE1XW_WIDTH) && defined(MULT_INTERLEAVE4X4_HEIGHT)
/** This OpenCL kernel is optimised for Midgard. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note The number of columns of matrix B and the optional alpha's value need to be passed at compile time using -DCOLS_B and -DALPHA
 * @note The multiplication factor for the transposition width (mult_transpose1xW_width) must be passed at compile time using -DMULT_TRANSPOSE1XW_WIDTH (i.e. -DMULT_TRANSPOSE1XW_WIDTH=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f32(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                 VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
                                                 uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                 ,
                                                 uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                )
{
    int x = get_global_id(0) / MULT_TRANSPOSE1XW_WIDTH;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % MULT_TRANSPOSE1XW_WIDTH) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global float *src_addr_a = (__global float *)(src0_ptr + src0_addr_in_bytes);
    __global float *src_addr_b = (__global float *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global float *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    float4 c00 = 0.0f;
    float4 c10 = 0.0f;
    float4 c20 = 0.0f;
    float4 c30 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - (int)(8 * MULT_TRANSPOSE1XW_WIDTH)); src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        c00 += (float4)a0.s0 * b0;
        c10 += (float4)a0.s1 * b0;
        c20 += (float4)a0.s2 * b0;
        c30 += (float4)a0.s3 * b0;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 4 * MULT_INTERLEAVE4X4_HEIGHT);
        b0 = vload4(0, src_addr_b + 4 * MULT_TRANSPOSE1XW_WIDTH);

        c00 += (float4)a0.s0 * b0;
        c10 += (float4)a0.s1 * b0;
        c20 += (float4)a0.s2 * b0;
        c30 += (float4)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 4 * MULT_TRANSPOSE1XW_WIDTH)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        c00 += (float4)a0.s0 * b0;
        c10 += (float4)a0.s1 * b0;
        c20 += (float4)a0.s2 * b0;
        c30 += (float4)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(ALPHA)
    // Multiply by the weight of matrix product
    c00 = c00 * (float4)ALPHA;
    c10 = c10 * (float4)ALPHA;
    c20 = c20 * (float4)ALPHA;
    c30 = c30 * (float4)ALPHA;
#endif // defined(ALPHA)

#if defined(ADD_VEC_C)
    __global float *src2_addr = (__global float *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    float4          c0        = vload4(0, src2_addr);

    c00 += c0;
    c10 += c0;
    c20 += c0;
    c30 += c0;
#endif /* defined(ADD_VEC_C) */

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x4 block
    vstore4(c00, 0, (__global float *)(dst_addr + 0 * dst_stride_y + zout.s0));
    vstore4(c10, 0, (__global float *)(dst_addr + 1 * dst_stride_y + zout.s1));
    vstore4(c20, 0, (__global float *)(dst_addr + 2 * dst_stride_y + zout.s2));
    vstore4(c30, 0, (__global float *)(dst_addr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store 4x4 block
    vstore4(c00, 0, (__global float *)(dst_addr + 0 * dst_stride_y));
    vstore4(c10, 0, (__global float *)(dst_addr + 1 * dst_stride_y));
    vstore4(c20, 0, (__global float *)(dst_addr + 2 * dst_stride_y));
    vstore4(c30, 0, (__global float *)(dst_addr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

/** This OpenCL kernel is optimized for Bifrost. It computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_32bit and @ref gemm_transpose1x4 before running the matrix multiplication.
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note The number of columns of matrix B and the optional alpha's value need to be passed at compile time using -DCOLS_B and -DALPHA
 * @note The multiplication factor for the transposition width (mult_transpose1xW_width) must be passed at compile time using -DMULT_TRANSPOSE1XW_WIDTH (i.e. -DMULT_TRANSPOSE1XW_WIDTH=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f32_bifrost(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                         VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                         IMAGE_DECLARATION(dst),
                                                         uint src0_stride_z,
                                                         uint src1_stride_z,
                                                         uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                         ,
                                                         uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                        )
{
    int x = get_global_id(0) / MULT_TRANSPOSE1XW_WIDTH;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % MULT_TRANSPOSE1XW_WIDTH) * 4;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global float *src_addr_a = (__global float *)(src0_ptr + src0_addr_in_bytes);
    __global float *src_addr_b = (__global float *)(src1_ptr + src1_addr_in_bytes);

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c02 = 0.0f;
    float c03 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;
    float c12 = 0.0f;
    float c13 = 0.0f;
    float c20 = 0.0f;
    float c21 = 0.0f;
    float c22 = 0.0f;
    float c23 = 0.0f;
    float c30 = 0.0f;
    float c31 = 0.0f;
    float c32 = 0.0f;
    float c33 = 0.0f;

#define COLS_MTX_B (COLS_B / (4 * MULT_TRANSPOSE1XW_WIDTH))

    int i = 0;
    for(; i <= (int)(COLS_MTX_B - 4); i += 4)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 4 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 4 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 4 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 4 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);
    }

    for(; i < (int)(COLS_MTX_B); ++i)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = vload4(0, src_addr_a);
        float4 b0 = vload4(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 4 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma(a0.s0, b0.s0, c00);
        c01 = fma(a0.s0, b0.s1, c01);
        c02 = fma(a0.s0, b0.s2, c02);
        c03 = fma(a0.s0, b0.s3, c03);

        c10 = fma(a0.s1, b0.s0, c10);
        c11 = fma(a0.s1, b0.s1, c11);
        c12 = fma(a0.s1, b0.s2, c12);
        c13 = fma(a0.s1, b0.s3, c13);

        c20 = fma(a0.s2, b0.s0, c20);
        c21 = fma(a0.s2, b0.s1, c21);
        c22 = fma(a0.s2, b0.s2, c22);
        c23 = fma(a0.s2, b0.s3, c23);

        c30 = fma(a0.s3, b0.s0, c30);
        c31 = fma(a0.s3, b0.s1, c31);
        c32 = fma(a0.s3, b0.s2, c32);
        c33 = fma(a0.s3, b0.s3, c33);
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(ALPHA)
    // Multiply by the weight of matrix product
    c00 = c00 * ALPHA;
    c01 = c01 * ALPHA;
    c02 = c02 * ALPHA;
    c03 = c03 * ALPHA;
    c10 = c10 * ALPHA;
    c11 = c11 * ALPHA;
    c12 = c12 * ALPHA;
    c13 = c13 * ALPHA;
    c20 = c20 * ALPHA;
    c21 = c21 * ALPHA;
    c22 = c22 * ALPHA;
    c23 = c23 * ALPHA;
    c30 = c30 * ALPHA;
    c31 = c31 * ALPHA;
    c32 = c32 * ALPHA;
    c33 = c33 * ALPHA;
#endif // defined(ALPHA)

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(ADD_VEC_C)
    __global float *src2_addr = (__global float *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    float4          c0        = vload4(0, src2_addr);

    c00 += c0.s0;
    c01 += c0.s1;
    c02 += c0.s2;
    c03 += c0.s3;
    c10 += c0.s0;
    c11 += c0.s1;
    c12 += c0.s2;
    c13 += c0.s3;
    c20 += c0.s0;
    c21 += c0.s1;
    c22 += c0.s2;
    c23 += c0.s3;
    c30 += c0.s0;
    c31 += c0.s1;
    c32 += c0.s2;
    c33 += c0.s3;
#endif /* defined(ADD_VEC_C) */

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x4 block
    vstore4((float4)(c00, c01, c02, c03), 0, (__global float *)(dst_addr + 0 * dst_stride_y + zout.s0));
    vstore4((float4)(c10, c11, c12, c13), 0, (__global float *)(dst_addr + 1 * dst_stride_y + zout.s1));
    vstore4((float4)(c20, c21, c22, c23), 0, (__global float *)(dst_addr + 2 * dst_stride_y + zout.s2));
    vstore4((float4)(c30, c31, c32, c33), 0, (__global float *)(dst_addr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store 4x4 block
    vstore4((float4)(c00, c01, c02, c03), 0, (__global float *)(dst_addr + 0 * dst_stride_y));
    vstore4((float4)(c10, c11, c12, c13), 0, (__global float *)(dst_addr + 1 * dst_stride_y));
    vstore4((float4)(c20, c21, c22, c23), 0, (__global float *)(dst_addr + 2 * dst_stride_y));
    vstore4((float4)(c30, c31, c32, c33), 0, (__global float *)(dst_addr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

// Undefine local defines
#undef COLS_MTX_B

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_16bit and @ref gemm_transpose1x8 before running the matrix multiplication
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note The number of columns of matrix B and the optional alpha's value need to be passed at compile time using -DCOLS_B and -DALPHA
 * @note The multiplication factor for the transposition width (mult_transpose1xW_width) must be passed at compile time using -DMULT_TRANSPOSE1XW_WIDTH (i.e. -DMULT_TRANSPOSE1XW_WIDTH=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f16(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                 VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
                                                 uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                 ,
                                                 uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                )
{
    int x = get_global_id(0) / MULT_TRANSPOSE1XW_WIDTH;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % MULT_TRANSPOSE1XW_WIDTH) * 8;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global half *src_addr_a = (__global half *)(src0_ptr + src0_addr_in_bytes);
    __global half *src_addr_b = (__global half *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global half *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    half8 c00 = 0.0f;
    half8 c10 = 0.0f;
    half8 c20 = 0.0f;
    half8 c30 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - (int)(16 * MULT_TRANSPOSE1XW_WIDTH)); src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 16 * MULT_TRANSPOSE1XW_WIDTH)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        c00 += (half8)a0.s0 * b0;
        c10 += (half8)a0.s1 * b0;
        c20 += (half8)a0.s2 * b0;
        c30 += (half8)a0.s3 * b0;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a + 4 * MULT_INTERLEAVE4X4_HEIGHT);
        b0 = vload8(0, src_addr_b + 8 * MULT_TRANSPOSE1XW_WIDTH);

        c00 += (half8)a0.s0 * b0;
        c10 += (half8)a0.s1 * b0;
        c20 += (half8)a0.s2 * b0;
        c30 += (half8)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        c00 += (half8)a0.s0 * b0;
        c10 += (half8)a0.s1 * b0;
        c20 += (half8)a0.s2 * b0;
        c30 += (half8)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(ALPHA)
    // Multiply by the weight of matrix product
    c00 = c00 * (half8)ALPHA;
    c10 = c10 * (half8)ALPHA;
    c20 = c20 * (half8)ALPHA;
    c30 = c30 * (half8)ALPHA;
#endif // defined(ALPHA)

#if defined(ADD_VEC_C)
    // *INDENT-OFF*
    // clang-format off
    __global half *src2_addr = (__global half *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    half8          c0        = vload8(0, src2_addr);
    // clang-format on
    // *INDENT-ON*

    c00 += c0;
    c10 += c0;
    c20 += c0;
    c30 += c0;
#endif /* defined(ADD_VEC_C) */

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x8 block
    vstore8(c00, 0, (__global half *)(dst_addr + 0 * dst_stride_y + zout.s0));
    vstore8(c10, 0, (__global half *)(dst_addr + 1 * dst_stride_y + zout.s1));
    vstore8(c20, 0, (__global half *)(dst_addr + 2 * dst_stride_y + zout.s2));
    vstore8(c30, 0, (__global half *)(dst_addr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store 4x8 block
    vstore8(c00, 0, (__global half *)(dst_addr + 0 * dst_stride_y));
    vstore8(c10, 0, (__global half *)(dst_addr + 1 * dst_stride_y));
    vstore8(c20, 0, (__global half *)(dst_addr + 2 * dst_stride_y));
    vstore8(c30, 0, (__global half *)(dst_addr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

/** This OpenCL kernel computes the matrix multiplication between matrix A (src0) and matrix B (src1) while accumulating the result in a 32 floating point variable.
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_16bit and @ref gemm_transpose1x8 before running the matrix multiplication
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note The number of columns of matrix B and the optional alpha's value need to be passed at compile time using -DCOLS_B and -DALPHA
 * @note The multiplication factor for the transposition width (mult_transpose1xW_width) must be passed at compile time using -DMULT_TRANSPOSE1XW_WIDTH (i.e. -DMULT_TRANSPOSE1XW_WIDTH=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f16_acc32(IMAGE_DECLARATION(src0),
                                                       IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                       VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                       IMAGE_DECLARATION(dst),
                                                       uint src0_stride_z,
                                                       uint src1_stride_z,
                                                       uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                       ,
                                                       uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                      )
{
    int x = get_global_id(0) / MULT_TRANSPOSE1XW_WIDTH;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % MULT_TRANSPOSE1XW_WIDTH) * 8;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global half *src_addr_a = (__global half *)(src0_ptr + src0_addr_in_bytes);
    __global half *src_addr_b = (__global half *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global half *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    float8 c00 = 0.0f;
    float8 c10 = 0.0f;
    float8 c20 = 0.0f;
    float8 c30 = 0.0f;

    for(; src_addr_b <= (src_end_addr_b - (int)(16 * MULT_TRANSPOSE1XW_WIDTH)); src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 16 * MULT_TRANSPOSE1XW_WIDTH)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = convert_float4(vload4(0, src_addr_a));
        float8 b0 = convert_float8(vload8(0, src_addr_b));

        c00 += (float8)a0.s0 * b0;
        c10 += (float8)a0.s1 * b0;
        c20 += (float8)a0.s2 * b0;
        c30 += (float8)a0.s3 * b0;

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = convert_float4(vload4(0, src_addr_a + 4 * MULT_INTERLEAVE4X4_HEIGHT));
        b0 = convert_float8(vload8(0, src_addr_b + 8 * MULT_TRANSPOSE1XW_WIDTH));

        c00 += (float8)a0.s0 * b0;
        c10 += (float8)a0.s1 * b0;
        c20 += (float8)a0.s2 * b0;
        c30 += (float8)a0.s3 * b0;
    }

    for(; src_addr_b < src_end_addr_b; src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT, src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        float4 a0 = convert_float4(vload4(0, src_addr_a));
        float8 b0 = convert_float8(vload8(0, src_addr_b));

        c00 += (float8)a0.s0 * b0;
        c10 += (float8)a0.s1 * b0;
        c20 += (float8)a0.s2 * b0;
        c30 += (float8)a0.s3 * b0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(ALPHA)
    // Multiply by the weight of matrix product
    c00 = c00 * (float8)ALPHA;
    c10 = c10 * (float8)ALPHA;
    c20 = c20 * (float8)ALPHA;
    c30 = c30 * (float8)ALPHA;
#endif // defined(ALPHA)

#if defined(ADD_VEC_C)
    // *INDENT-OFF*
    // clang-format off
    __global half *src2_addr = (__global half *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    float8         c0        = convert_float8(vload8(0, src2_addr));
    // clang-format on
    // *INDENT-ON*

    c00 += c0;
    c10 += c0;
    c20 += c0;
    c30 += c0;
#endif /* defined(ADD_VEC_C) */

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x8 block
    vstore8(convert_half8(c00), 0, (__global half *)(dst_addr + 0 * dst_stride_y + zout.s0));
    vstore8(convert_half8(c10), 0, (__global half *)(dst_addr + 1 * dst_stride_y + zout.s1));
    vstore8(convert_half8(c20), 0, (__global half *)(dst_addr + 2 * dst_stride_y + zout.s2));
    vstore8(convert_half8(c30), 0, (__global half *)(dst_addr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store 4x8 block
    vstore8(convert_half8(c00), 0, (__global half *)(dst_addr + 0 * dst_stride_y));
    vstore8(convert_half8(c10), 0, (__global half *)(dst_addr + 1 * dst_stride_y));
    vstore8(convert_half8(c20), 0, (__global half *)(dst_addr + 2 * dst_stride_y));
    vstore8(convert_half8(c30), 0, (__global half *)(dst_addr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

/** This OpenCL kernel optimized for Bifrost architectures computes the matrix multiplication between matrix A (src0) and matrix B (src1)
 *  Matrix A and matrix B must be reshaped respectively with @ref gemm_interleave4x4_16bit and @ref gemm_transpose1x8 before running the matrix multiplication
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note The number of columns of matrix B and the optional alpha's value need to be passed at compile time using -DCOLS_B and -DALPHA
 * @note The multiplication factor for the transposition width (mult_transpose1xW_width) must be passed at compile time using -DMULT_TRANSPOSE1XW_WIDTH (i.e. -DMULT_TRANSPOSE1XW_WIDTH=2)
 * @note The multiplication factor for the height of the 4x4 interleaved block must be passed at compile time using -DMULT_INTERLEAVE4X4_HEIGHT (i.e. -DMULT_INTERLEAVE4X4_HEIGHT=2)
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the output has to be reinterpreted as a 3D tensor (i.e. output of convolution layer), the following information must be passed at compile time:
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  cross_plane_pad                    (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_interleaved_transposed_f16_bifrost(IMAGE_DECLARATION(src0),
                                                         IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                         VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                         IMAGE_DECLARATION(dst),
                                                         uint src0_stride_z,
                                                         uint src1_stride_z,
                                                         uint dst_stride_z
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                         ,
                                                         uint cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                        )
{
    int x = get_global_id(0) / MULT_TRANSPOSE1XW_WIDTH;
    int y = get_global_id(1) / MULT_INTERLEAVE4X4_HEIGHT;
    int z = get_global_id(2);

    // Offset
    const int offset_row_a = (get_global_id(1) % MULT_INTERLEAVE4X4_HEIGHT) * 4;
    const int offset_row_b = (get_global_id(0) % MULT_TRANSPOSE1XW_WIDTH) * 8;

    // src_addr_a = address of matrix A
    // src_addr_b = address of matrix B
    int src0_addr_in_bytes = z * src0_stride_z + y * src0_stride_y + src0_offset_first_element_in_bytes;
    int src1_addr_in_bytes = x * src1_stride_y + src1_offset_first_element_in_bytes;

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src1_addr_in_bytes += (z % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src1_addr_in_bytes += z * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    __global half *src_addr_a = (__global half *)(src0_ptr + src0_addr_in_bytes);
    __global half *src_addr_b = (__global half *)(src1_ptr + src1_addr_in_bytes);

    // Compute end row address for matrix B
    __global half *src_end_addr_b = src_addr_b + COLS_B;

    src_addr_a += offset_row_a;
    src_addr_b += offset_row_b;

    // Reset accumulators
    half8 c00 = 0.0f;
    half8 c10 = 0.0f;
    half8 c20 = 0.0f;
    half8 c30 = 0.0f;

#define COLS_MTX_B (COLS_B / (8 * MULT_TRANSPOSE1XW_WIDTH))

    int i = 0;
    for(; i <= (int)(COLS_MTX_B - 4); i += 4)
    {
#if MULT_INTERLEAVE4X4_HEIGHT == 1
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half8 a0 = vload8(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);

        // Load values from matrix B (transposed)
        b0 = vload8(0, src_addr_b);

        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s4, b0, c00);
        c10 = fma((half8)a0.s5, b0, c10);
        c20 = fma((half8)a0.s6, b0, c20);
        c30 = fma((half8)a0.s7, b0, c30);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload8(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 8 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);

        // Load values from matrix B (transposed)
        b0 = vload8(0, src_addr_b);

        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s4, b0, c00);
        c10 = fma((half8)a0.s5, b0, c10);
        c20 = fma((half8)a0.s6, b0, c20);
        c30 = fma((half8)a0.s7, b0, c30);
#else  // MULT_INTERLEAVE4X4_HEIGHT == 1
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);

        // Load values from matrix A (interleaved) and matrix B (transposed)
        a0 = vload4(0, src_addr_a);
        b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);
#endif // MULT_INTERLEAVE4X4_HEIGHT == 1
    }

    for(; i < (int)(COLS_MTX_B); ++i)
    {
        // Load values from matrix A (interleaved) and matrix B (transposed)
        half4 a0 = vload4(0, src_addr_a);
        half8 b0 = vload8(0, src_addr_b);

        src_addr_a += 4 * MULT_INTERLEAVE4X4_HEIGHT;
        src_addr_b += 8 * MULT_TRANSPOSE1XW_WIDTH;

        c00 = fma((half8)a0.s0, b0, c00);
        c10 = fma((half8)a0.s1, b0, c10);
        c20 = fma((half8)a0.s2, b0, c20);
        c30 = fma((half8)a0.s3, b0, c30);
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

#if defined(ALPHA)
    // Multiply by the weight of matrix product
    c00 = c00 * (half8)ALPHA;
    c10 = c10 * (half8)ALPHA;
    c20 = c20 * (half8)ALPHA;
    c30 = c30 * (half8)ALPHA;
#endif // defined(ALPHA)

#if defined(ADD_VEC_C)
    // *INDENT-OFF*
    // clang-format off
    __global half *src2_addr = (__global half *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    half8          c0        = vload8(0, src2_addr);
    // clang-format on
    // *INDENT-ON*

    c00 += c0;
    c10 += c0;
    c20 += c0;
    c30 += c0;
#endif /* defined(ADD_VEC_C) */

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * 4) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * 4)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store 4x8 block
    vstore8(c00, 0, (__global half *)(dst_addr + 0 * dst_stride_y + zout.s0));
    vstore8(c10, 0, (__global half *)(dst_addr + 1 * dst_stride_y + zout.s1));
    vstore8(c20, 0, (__global half *)(dst_addr + 2 * dst_stride_y + zout.s2));
    vstore8(c30, 0, (__global half *)(dst_addr + 3 * dst_stride_y + zout.s3));

#else  // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store 4x8 block
    vstore8(c00, 0, (__global half *)(dst_addr + 0 * dst_stride_y));
    vstore8(c10, 0, (__global half *)(dst_addr + 1 * dst_stride_y));
    vstore8(c20, 0, (__global half *)(dst_addr + 2 * dst_stride_y));
    vstore8(c30, 0, (__global half *)(dst_addr + 3 * dst_stride_y));
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

// Undefine local defines
#undef COLS_MTX_B

#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)

#endif // defined(COLS_B) && defined(MULT_TRANSPOSE1XW_WIDTH) && defined(MULT_INTERLEAVE4X4_HEIGHT)

#if defined(COLS_A) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && (NUM_ELEMS_PROCESSED_PER_THREAD_Y)
#if defined(DATA_TYPE)
#define VECTOR_TYPE VEC_DATA_TYPE(DATA_TYPE, NUM_ELEMS_PROCESSED_PER_THREAD_X)
/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not been reshaped.
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note This OpenCL kernel works with floating point data types (F16/F32)
 * @note The floating point data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=float)
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DNUM_ELEMS_PROCESSED_PER_THREAD_X and -DNUM_ELEMS_PROCESSED_PER_THREAD_Y
 * @note The number of matrix A columns and the optional alpha's value need to be passed at compile time using -DCOLS_A and -DALPHA
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16/F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the output tensor (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point(IMAGE_DECLARATION(src0),
                                     IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                     VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                     IMAGE_DECLARATION(dst),
                                     uint src0_stride_z,
                                     uint src1_stride_z,
                                     uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                     ,
                                     uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                     ,
                                     uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                    )
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(DATA_TYPE);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (src_cross_plane_pad * src0_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src0_stride_z by DEPTH_GEMM3D
    src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    int end_row_vec_a = src_addr.s0 + (COLS_A * sizeof(DATA_TYPE));

    VECTOR_TYPE acc0 = 0.0f;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VECTOR_TYPE acc1 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VECTOR_TYPE acc2 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VECTOR_TYPE acc3 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    for(; src_addr.s0 <= (end_row_vec_a - 2 * (int)sizeof(DATA_TYPE)); src_addr += (int2)(2 * sizeof(DATA_TYPE), 2 * src1_stride_y))
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a0 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a1 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a2 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        VEC_DATA_TYPE(DATA_TYPE, 2)
        a3 = vload2(0, (__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        VECTOR_TYPE b0 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, (__global DATA_TYPE *)(src1_ptr + src_addr.s1));
        VECTOR_TYPE b1 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, (__global DATA_TYPE *)(src1_ptr + src_addr.s1 + src1_stride_y));

        // Accumulate
        acc0 += b0 * (VECTOR_TYPE)a0.s0;
        acc0 += b1 * (VECTOR_TYPE)a0.s1;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += b0 * (VECTOR_TYPE)a1.s0;
        acc1 += b1 * (VECTOR_TYPE)a1.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += b0 * (VECTOR_TYPE)a2.s0;
        acc2 += b1 * (VECTOR_TYPE)a2.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += b0 * (VECTOR_TYPE)a3.s0;
        acc3 += b1 * (VECTOR_TYPE)a3.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(sizeof(DATA_TYPE), src1_stride_y))
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        DATA_TYPE a0 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        DATA_TYPE a1 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        DATA_TYPE a2 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        DATA_TYPE a3 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        DATA_TYPE a0 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        DATA_TYPE a1 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        DATA_TYPE a2 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        DATA_TYPE a3 = *((__global DATA_TYPE *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        VECTOR_TYPE b0 = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, (__global DATA_TYPE *)(src1_ptr + src_addr.s1));

        // Accumulate
        acc0 += b0 * (VECTOR_TYPE)a0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 += b0 * (VECTOR_TYPE)a1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 += b0 * (VECTOR_TYPE)a2;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 += b0 * (VECTOR_TYPE)a3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    acc0 = acc0 * (VECTOR_TYPE)ALPHA;
#endif // defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
    acc1 = acc1 * (VECTOR_TYPE)ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
    acc2 = acc2 * (VECTOR_TYPE)ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)
    acc3 = acc3 * (VECTOR_TYPE)ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)

#if defined(ADD_VEC_C)
    // *INDENT-OFF*
    // clang-format off
    __global DATA_TYPE *src2_addr = (__global DATA_TYPE *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    VECTOR_TYPE         c0        = VLOAD(NUM_ELEMS_PROCESSED_PER_THREAD_X)(0, src2_addr);
    // clang-format on
    // *INDENT-ON*

    acc0 += c0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc1 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc2 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc3 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif /* defined(ADD_VEC_C) */

    int z = get_global_id(2);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store output block
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc0, 0, (__global DATA_TYPE *)(dst_addr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc1, 0, (__global DATA_TYPE *)(dst_addr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc2, 0, (__global DATA_TYPE *)(dst_addr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc3, 0, (__global DATA_TYPE *)(dst_addr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store output block
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc0, 0, (__global DATA_TYPE *)(dst_addr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc1, 0, (__global DATA_TYPE *)(dst_addr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc2, 0, (__global DATA_TYPE *)(dst_addr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    VSTORE(NUM_ELEMS_PROCESSED_PER_THREAD_X)
    (acc3, 0, (__global DATA_TYPE *)(dst_addr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}
#endif // defined(DATA_TYPE)

/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not been reshaped
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note This OpenCL kernel works with the 32-bit floating point data type (float) and uses the fma units.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DNUM_ELEMS_PROCESSED_PER_THREAD_X and -DNUM_ELEMS_PROCESSED_PER_THREAD_Y.
 * This kernel optimally uses -DNUM_ELEMS_PROCESSED_PER_THREAD_X=4.
 * @note The number of matrix A columns must be passed at compile time using -DCOLS_A.
 * @note The optional value of scalar alpha is passed at compile time using -DALPHA=alpha
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16/F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f32_bifrost(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                 VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
                                                 uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                 ,
                                                 uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                 ,
                                                 uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                )
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for matrix B
    src_addr.s1 += idx * sizeof(float);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (src_cross_plane_pad * src0_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src0_stride_z by DEPTH_GEMM3D
    src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize accumulators
    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc02 = 0.0f;
    float acc03 = 0.0f;

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    float acc12 = 0.0f;
    float acc13 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    float acc20 = 0.0f;
    float acc21 = 0.0f;
    float acc22 = 0.0f;
    float acc23 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    float acc30 = 0.0f;
    float acc31 = 0.0f;
    float acc32 = 0.0f;
    float acc33 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    // A and B src indices get incremented at the same time.
    int i = 0;
    for(; i <= ((int)COLS_A - 4); i += 4)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A and matrix B
        float4 a0 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float4 a1 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float4 a2 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float4 a3 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A and matrix B
        float4 a0 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float4 a1 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float4 a2 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float4 a3 = vload4(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0.s0, b0.s0, acc00);
        acc01 = fma(a0.s0, b0.s1, acc01);
        acc02 = fma(a0.s0, b0.s2, acc02);
        acc03 = fma(a0.s0, b0.s3, acc03);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1

        acc10 = fma(a1.s0, b0.s0, acc10);
        acc11 = fma(a1.s0, b0.s1, acc11);
        acc12 = fma(a1.s0, b0.s2, acc12);
        acc13 = fma(a1.s0, b0.s3, acc13);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2

        acc20 = fma(a2.s0, b0.s0, acc20);
        acc21 = fma(a2.s0, b0.s1, acc21);
        acc22 = fma(a2.s0, b0.s2, acc22);
        acc23 = fma(a2.s0, b0.s3, acc23);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        acc30 = fma(a3.s0, b0.s0, acc30);
        acc31 = fma(a3.s0, b0.s1, acc31);
        acc32 = fma(a3.s0, b0.s2, acc32);
        acc33 = fma(a3.s0, b0.s3, acc33);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        // Load values from matrix A and matrix B
        b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0.s1, b0.s0, acc00);
        acc01 = fma(a0.s1, b0.s1, acc01);
        acc02 = fma(a0.s1, b0.s2, acc02);
        acc03 = fma(a0.s1, b0.s3, acc03);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1

        acc10 = fma(a1.s1, b0.s0, acc10);
        acc11 = fma(a1.s1, b0.s1, acc11);
        acc12 = fma(a1.s1, b0.s2, acc12);
        acc13 = fma(a1.s1, b0.s3, acc13);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2

        acc20 = fma(a2.s1, b0.s0, acc20);
        acc21 = fma(a2.s1, b0.s1, acc21);
        acc22 = fma(a2.s1, b0.s2, acc22);
        acc23 = fma(a2.s1, b0.s3, acc23);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        acc30 = fma(a3.s1, b0.s0, acc30);
        acc31 = fma(a3.s1, b0.s1, acc31);
        acc32 = fma(a3.s1, b0.s2, acc32);
        acc33 = fma(a3.s1, b0.s3, acc33);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        // Load values from matrix A and matrix B
        b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0.s2, b0.s0, acc00);
        acc01 = fma(a0.s2, b0.s1, acc01);
        acc02 = fma(a0.s2, b0.s2, acc02);
        acc03 = fma(a0.s2, b0.s3, acc03);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1

        acc10 = fma(a1.s2, b0.s0, acc10);
        acc11 = fma(a1.s2, b0.s1, acc11);
        acc12 = fma(a1.s2, b0.s2, acc12);
        acc13 = fma(a1.s2, b0.s3, acc13);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2

        acc20 = fma(a2.s2, b0.s0, acc20);
        acc21 = fma(a2.s2, b0.s1, acc21);
        acc22 = fma(a2.s2, b0.s2, acc22);
        acc23 = fma(a2.s2, b0.s3, acc23);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        acc30 = fma(a3.s2, b0.s0, acc30);
        acc31 = fma(a3.s2, b0.s1, acc31);
        acc32 = fma(a3.s2, b0.s2, acc32);
        acc33 = fma(a3.s2, b0.s3, acc33);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        // Load values from matrix A and matrix B
        b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0.s3, b0.s0, acc00);
        acc01 = fma(a0.s3, b0.s1, acc01);
        acc02 = fma(a0.s3, b0.s2, acc02);
        acc03 = fma(a0.s3, b0.s3, acc03);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1

        acc10 = fma(a1.s3, b0.s0, acc10);
        acc11 = fma(a1.s3, b0.s1, acc11);
        acc12 = fma(a1.s3, b0.s2, acc12);
        acc13 = fma(a1.s3, b0.s3, acc13);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2

        acc20 = fma(a2.s3, b0.s0, acc20);
        acc21 = fma(a2.s3, b0.s1, acc21);
        acc22 = fma(a2.s3, b0.s2, acc22);
        acc23 = fma(a2.s3, b0.s3, acc23);

#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        acc30 = fma(a3.s3, b0.s0, acc30);
        acc31 = fma(a3.s3, b0.s1, acc31);
        acc32 = fma(a3.s3, b0.s2, acc32);
        acc33 = fma(a3.s3, b0.s3, acc33);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += 4 * sizeof(float);
    }

    for(; i < (int)COLS_A; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0, b0.s0, acc00);
        acc01 = fma(a0, b0.s1, acc01);
        acc02 = fma(a0, b0.s2, acc02);
        acc03 = fma(a0, b0.s3, acc03);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc10 = fma(a1, b0.s0, acc10);
        acc11 = fma(a1, b0.s1, acc11);
        acc12 = fma(a1, b0.s2, acc12);
        acc13 = fma(a1, b0.s3, acc13);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc20 = fma(a2, b0.s0, acc20);
        acc21 = fma(a2, b0.s1, acc21);
        acc22 = fma(a2, b0.s2, acc22);
        acc23 = fma(a2, b0.s3, acc23);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc30 = fma(a3, b0.s0, acc30);
        acc31 = fma(a3, b0.s1, acc31);
        acc32 = fma(a3, b0.s2, acc32);
        acc33 = fma(a3, b0.s3, acc33);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += sizeof(float);
    }

    int z = get_global_id(2);

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    acc00 = acc00 * ALPHA;
    acc01 = acc01 * ALPHA;
    acc02 = acc02 * ALPHA;
    acc03 = acc03 * ALPHA;
#endif // defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
    acc10 = acc10 * ALPHA;
    acc11 = acc11 * ALPHA;
    acc12 = acc12 * ALPHA;
    acc13 = acc13 * ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
    acc20 = acc20 * ALPHA;
    acc21 = acc21 * ALPHA;
    acc22 = acc22 * ALPHA;
    acc23 = acc23 * ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)
    acc30 = acc30 * ALPHA;
    acc31 = acc31 * ALPHA;
    acc32 = acc32 * ALPHA;
    acc33 = acc33 * ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(ADD_VEC_C)
    __global float *src2_addr = (__global float *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    float4          c0        = vload4(0, src2_addr);

    acc00 += c0.s0;
    acc01 += c0.s1;
    acc02 += c0.s2;
    acc03 += c0.s3;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc10 += c0.s0;
    acc11 += c0.s1;
    acc12 += c0.s2;
    acc13 += c0.s3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc20 += c0.s0;
    acc21 += c0.s1;
    acc22 += c0.s2;
    acc23 += c0.s3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc30 += c0.s0;
    acc31 += c0.s1;
    acc32 += c0.s2;
    acc33 += c0.s3;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif /* defined(ADD_VEC_C) */

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store the output block
    vstore4((float4)(acc00, acc01, acc02, acc03), 0, (__global float *)(dst_addr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((float4)(acc10, acc11, acc12, acc13), 0, (__global float *)(dst_addr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((float4)(acc20, acc21, acc22, acc23), 0, (__global float *)(dst_addr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((float4)(acc30, acc31, acc32, acc33), 0, (__global float *)(dst_addr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store the output block
    vstore4((float4)(acc00, acc01, acc02, acc03), 0, (__global float *)(dst_addr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore4((float4)(acc10, acc11, acc12, acc13), 0, (__global float *)(dst_addr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore4((float4)(acc20, acc21, acc22, acc23), 0, (__global float *)(dst_addr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore4((float4)(acc30, acc31, acc32, acc33), 0, (__global float *)(dst_addr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not been reshaped
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note This OpenCL kernel works with the 32-bit floating point data type (float) and uses the fma units.
 * This OpenCL kernel is optimized for Bifrost when the number of matrix B columns is less or equal to 1000.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DNUM_ELEMS_PROCESSED_PER_THREAD_X and -DNUM_ELEMS_PROCESSED_PER_THREAD_Y.
 * This kernel optimally uses -DNUM_ELEMS_PROCESSED_PER_THREAD_X=2.
 * @note The number of matrix A columns must be passed at compile time using -DCOLS_A.
 * @note The optional value of scalar alpha is passed at compile time using -DALPHA=alpha if alpha!=1.0f.
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16/F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f32_bifrost_1000(IMAGE_DECLARATION(src0),
                                                      IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                      VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                      IMAGE_DECLARATION(dst),
                                                      uint src0_stride_z,
                                                      uint src1_stride_z,
                                                      uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                      ,
                                                      uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                      ,
                                                      uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                     )
{
    // Requires 2 NUM_ELEMS_PROCESSED_PER_THREAD_X, C vect2, A vect4, B (2 vload2) // to fix for NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(float);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (src_cross_plane_pad * src0_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src0_stride_z by DEPTH_GEMM3D
    src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    // Initialize accumulators
    float acc00 = 0.0f;
    float acc01 = 0.0f;

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    float acc10 = 0.0f;
    float acc11 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    float acc20 = 0.0f;
    float acc21 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    float acc30 = 0.0f;
    float acc31 = 0.0f;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    // A and B src indices get incremented at the same time.
    int i = 0;
    for(; i <= ((int)COLS_A - 8); i += 8)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float8 a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + zin.s0));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float8 a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0));
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float2 b0 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b1 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b2 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b3 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b4 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b5 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b6 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        float2 b7 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0.s0, b0.s0, acc00);
        acc00 = fma(a0.s1, b1.s0, acc00);
        acc00 = fma(a0.s2, b2.s0, acc00);
        acc00 = fma(a0.s3, b3.s0, acc00);
        acc00 = fma(a0.s4, b4.s0, acc00);
        acc00 = fma(a0.s5, b5.s0, acc00);
        acc00 = fma(a0.s6, b6.s0, acc00);
        acc00 = fma(a0.s7, b7.s0, acc00);

        acc01 = fma(a0.s0, b0.s1, acc01);
        acc01 = fma(a0.s1, b1.s1, acc01);
        acc01 = fma(a0.s2, b2.s1, acc01);
        acc01 = fma(a0.s3, b3.s1, acc01);
        acc01 = fma(a0.s4, b4.s1, acc01);
        acc01 = fma(a0.s5, b5.s1, acc01);
        acc01 = fma(a0.s6, b6.s1, acc01);
        acc01 = fma(a0.s7, b7.s1, acc01);

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)
        acc10 = fma(a0.s0, b0.s0, acc10);
        acc10 = fma(a0.s1, b1.s0, acc10);
        acc10 = fma(a0.s2, b2.s0, acc10);
        acc10 = fma(a0.s3, b3.s0, acc10);
        acc10 = fma(a0.s4, b4.s0, acc10);
        acc10 = fma(a0.s5, b5.s0, acc10);
        acc10 = fma(a0.s6, b6.s0, acc10);
        acc10 = fma(a0.s7, b7.s0, acc10);

        acc11 = fma(a0.s0, b0.s1, acc11);
        acc11 = fma(a0.s1, b1.s1, acc11);
        acc11 = fma(a0.s2, b2.s1, acc11);
        acc11 = fma(a0.s3, b3.s1, acc11);
        acc11 = fma(a0.s4, b4.s1, acc11);
        acc11 = fma(a0.s5, b5.s1, acc11);
        acc11 = fma(a0.s6, b6.s1, acc11);
        acc11 = fma(a0.s7, b7.s1, acc11);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)
        acc20 = fma(a0.s0, b0.s0, acc20);
        acc20 = fma(a0.s1, b1.s0, acc20);
        acc20 = fma(a0.s2, b2.s0, acc20);
        acc20 = fma(a0.s3, b3.s0, acc20);
        acc20 = fma(a0.s4, b4.s0, acc20);
        acc20 = fma(a0.s5, b5.s0, acc20);
        acc20 = fma(a0.s6, b6.s0, acc20);
        acc20 = fma(a0.s7, b7.s0, acc20);

        acc21 = fma(a0.s0, b0.s1, acc21);
        acc21 = fma(a0.s1, b1.s1, acc21);
        acc21 = fma(a0.s2, b2.s1, acc21);
        acc21 = fma(a0.s3, b3.s1, acc21);
        acc21 = fma(a0.s4, b4.s1, acc21);
        acc21 = fma(a0.s5, b5.s1, acc21);
        acc21 = fma(a0.s6, b6.s1, acc21);
        acc21 = fma(a0.s7, b7.s1, acc21);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#else  // defined(REINTERPRET_INPUT_AS_3D)
        a0 = vload8(0, (__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // defined(REINTERPRET_INPUT_AS_3D)
        acc30 = fma(a0.s0, b0.s0, acc30);
        acc30 = fma(a0.s1, b1.s0, acc30);
        acc30 = fma(a0.s2, b2.s0, acc30);
        acc30 = fma(a0.s3, b3.s0, acc30);
        acc30 = fma(a0.s4, b4.s0, acc30);
        acc30 = fma(a0.s5, b5.s0, acc30);
        acc30 = fma(a0.s6, b6.s0, acc30);
        acc30 = fma(a0.s7, b7.s0, acc30);

        acc31 = fma(a0.s0, b0.s1, acc31);
        acc31 = fma(a0.s1, b1.s1, acc31);
        acc31 = fma(a0.s2, b2.s1, acc31);
        acc31 = fma(a0.s3, b3.s1, acc31);
        acc31 = fma(a0.s4, b4.s1, acc31);
        acc31 = fma(a0.s5, b5.s1, acc31);
        acc31 = fma(a0.s6, b6.s1, acc31);
        acc31 = fma(a0.s7, b7.s1, acc31);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += sizeof(float) * 8;
    }
    // float size increment
    for(; i < (int)COLS_A; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        float a0 = *((__global float *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        float a1 = *((__global float *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        float a2 = *((__global float *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        float a3 = *((__global float *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float2 b0 = vload2(0, (__global float *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Multiply and accumulate
        acc00 = fma(a0, b0.s0, acc00);
        acc01 = fma(a0, b0.s1, acc01);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc10 = fma(a1, b0.s0, acc10);
        acc11 = fma(a1, b0.s1, acc11);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc20 = fma(a2, b0.s0, acc20);
        acc21 = fma(a2, b0.s1, acc21);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc30 = fma(a3, b0.s0, acc30);
        acc31 = fma(a3, b0.s1, acc31);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += sizeof(float);
    }

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    acc00 = acc00 * ALPHA;
    acc01 = acc01 * ALPHA;
#endif // defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
    acc10 = acc10 * ALPHA;
    acc11 = acc11 * ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
    acc20 = acc20 * ALPHA;
    acc21 = acc21 * ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)
    acc30 = acc30 * ALPHA;
    acc31 = acc31 * ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)

    int z = get_global_id(2);

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(ADD_VEC_C)
    __global float *src2_addr = (__global float *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    float2          c0        = vload2(0, src2_addr);

    acc00 += c0.s0;
    acc01 += c0.s1;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc10 += c0.s0;
    acc11 += c0.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc20 += c0.s0;
    acc21 += c0.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc30 += c0.s0;
    acc31 += c0.s1;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif /* defined(ADD_VEC_C) */

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store the output block
    vstore2((float2)(acc00, acc01), 0, (__global float *)(dst_addr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore2((float2)(acc10, acc11), 0, (__global float *)(dst_addr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore2((float2)(acc20, acc21), 0, (__global float *)(dst_addr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore2((float2)(acc30, acc31), 0, (__global float *)(dst_addr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store the output block
    vstore2((float2)(acc00, acc01), 0, (__global float *)(dst_addr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore2((float2)(acc10, acc11), 0, (__global float *)(dst_addr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore2((float2)(acc20, acc21), 0, (__global float *)(dst_addr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore2((float2)(acc30, acc31), 0, (__global float *)(dst_addr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_OUTPUT_AS_3D)
}

#if defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)
/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note This OpenCL kernel works with the 16-bit floating point data type (half) and accumulating the result in a 32 floating point variable.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DNUM_ELEMS_PROCESSED_PER_THREAD_X and -DNUM_ELEMS_PROCESSED_PER_THREAD_Y.
 * This kernel optimally uses -DNUM_ELEMS_PROCESSED_PER_THREAD_X=4.
 * @note The number of matrix A columns must be passed at compile time using -DCOLS_A.
 * @note The optional value of scalar alpha is passed at compile time using -DALPHA=alpha
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f16_bifrost_acc32(IMAGE_DECLARATION(src0),
                                                       IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                       VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                       IMAGE_DECLARATION(dst),
                                                       uint src0_stride_z,
                                                       uint src1_stride_z,
                                                       uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                       ,
                                                       uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                       ,
                                                       uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                      )
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(half);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (src_cross_plane_pad * src0_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src0_stride_z by DEPTH_GEMM3D
    src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    float8 acc0 = 0.0h;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    float8 acc1 = 0.0h;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    float8 acc2 = 0.0h;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    float8 acc3 = 0.0h;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    int i = 0;
    for(; i <= ((int)COLS_A - 4); i += 4)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half4 a1 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half4 a2 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half4 a3 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half4 a1 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half4 a2 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half4 a3 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float8 b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;

        // Accumulate
        acc0 = fma(b0, (float8)a0.s0, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (float8)a1.s0, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (float8)a2.s0, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (float8)a3.s0, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (float8)a0.s1, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (float8)a1.s1, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (float8)a2.s1, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (float8)a3.s1, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (float8)a0.s2, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (float8)a1.s2, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (float8)a2.s2, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (float8)a3.s2, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (float8)a0.s3, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (float8)a1.s3, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (float8)a2.s3, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (float8)a3.s3, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += 4 * sizeof(half);
    }

    for(; i < (int)COLS_A; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        float8 b0 = convert_float8(vload8(0, (__global half *)(src1_ptr + src_addr.s1)));

        src_addr += (int2)(sizeof(half), src1_stride_y);

        // Accumulate
        acc0 = fma(b0, (float8)a0, acc0); // b0 * (half8)a0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (float8)a1, acc1); // b0 * (half8)a1;
#endif                                    // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (float8)a2, acc2); // b0 * (half8)a2;
#endif                                    // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (float8)a3, acc3); // b0 * (half8)a3;
#endif                                    // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    half8 hacc0 = convert_half8(acc0) * (half8)ALPHA;
#else  //defined(ALPHA)
    half8 hacc0 = convert_half8(acc0);
#endif // defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if defined(ALPHA)
    half8 hacc1 = convert_half8(acc1) * (half8)ALPHA;
#else  //defined(ALPHA)
    half8 hacc1 = convert_half8(acc1);
#endif //defined(ALPHA)
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if defined(ALPHA)
    half8 hacc2 = convert_half8(acc2) * (half8)ALPHA;
#else  //defined(ALPHA)
    half8 hacc2 = convert_half8(acc2);
#endif //defined(ALPHA)
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2

#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#if defined(ALPHA)
    half8 hacc3 = convert_half8(acc3) * (half8)ALPHA;
#else  //defined(ALPHA)
    half8 hacc3 = convert_half8(acc3);
#endif // defined(ALPHA)
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#if defined(ADD_VEC_C)
    // *INDENT-OFF*
    // clang-format off
    __global half *src2_addr = (__global half *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    half8          c0        = vload8(0, src2_addr);
    // clang-format on
    // *INDENT-ON*

    hacc0 += c0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    hacc1 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    hacc2 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    hacc3 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif /* defined(ADD_VEC_C) */

    int z = get_global_id(2);

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store the output block
    vstore8(hacc0, 0, (__global half *)(dst_addr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore8(hacc1, 0, (__global half *)(dst_addr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore8(hacc2, 0, (__global half *)(dst_addr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore8(hacc3, 0, (__global half *)(dst_addr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store the output block
    vstore8(hacc0, 0, (__global half *)(dst_addr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore8(hacc1, 0, (__global half *)(dst_addr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore8(hacc2, 0, (__global half *)(dst_addr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore8(hacc3, 0, (__global half *)(dst_addr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // REINTERPRET_OUTPUT_AS_3D
}

/** This OpenCL kernel computes the matrix by matrix multiplication between the matrix A (src0) and matrix B (src1) in case both matrices have not beed reshaped
 *
 * Moreover, it can add a vector (src2) if the ADD_VEC_C parameter is passed at compile time.
 *
 * @note This OpenCL kernel works with the 16-bit floating point data type (half) and uses the fma units.
 * @note The number of elements processed along the x and y directions must be passed at compile time using -DNUM_ELEMS_PROCESSED_PER_THREAD_X and -DNUM_ELEMS_PROCESSED_PER_THREAD_Y.
 * This kernel optimally uses -DNUM_ELEMS_PROCESSED_PER_THREAD_X=4.
 * @note The number of matrix A columns must be passed at compile time using -DCOLS_A.
 * @note The optional value of scalar alpha is passed at compile time using -DALPHA=alpha
 * @note In case the matrix B has 3 dimensions and the matrix A more than 3, in order to avoid out-of-bounds reads, the number of channels of matrix B must be passed at compile time using MATRIX_B_DEPTH (i.e. -DMATRIX_B_DEPTH=16)
 *       This case can happen when GEMM is used to perform the element-wise multiplication through a batched matrix multiplication (2D Winograd) and we have multiple inputs (i.e. a = [K, M, 16, Batches], b = [N, K, 16])
 *
 * @note In case the input or output have to be reinterpreted as a 3D tensor, the following information must be passed at compile time:
 *       -# REINTERPRET_INPUT_AS_3D: To reinterpret the input as 3D
 *       -# REINTERPRET_OUTPUT_AS_3D: To reinterpret the output as 3D
 *       -# HEIGHT_GEMM3D: The height of the output in case it has to be reinterpreted as a 3D tensor.
 *       -# DEPTH_GEMM3D: The depth of the output in case it has to be reinterpreted as a 3D tensor
 *          (HEIGHT_GEMM3D * DEPTH_GEMM3D) = columns matrix A NOT reshaped
 *
 * @note In case a 3rd input (src2) needs to be added, the ADD_VEC_C parameter has to be passed at compile time as -DADD_VEC_C
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F16
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src2_ptr                           (Optional) Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src2_stride_x                      (Optional) Stride of the source vector in X dimension (in bytes)
 * @param[in]  src2_step_x                        (Optional) src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src2_offset_first_element_in_bytes (Optional) The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 * @param[in]  src0_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  src_cross_plane_pad                (Optional) Bottom paddings in unit of elements for the input tensor (only if defined REINTERPRET_INPUT_AS_3D)
 * @param[in]  dst_cross_plane_pad                (Optional) Bottom paddings in unit of elements (only if defined REINTERPRET_OUTPUT_AS_3D)
 */
__kernel void gemm_mm_floating_point_f16_bifrost(IMAGE_DECLARATION(src0),
                                                 IMAGE_DECLARATION(src1),
#if defined(ADD_VEC_C)
                                                 VECTOR_DECLARATION(src2),
#endif /* defined(ADD_VEC_C) */
                                                 IMAGE_DECLARATION(dst),
                                                 uint src0_stride_z,
                                                 uint src1_stride_z,
                                                 uint dst_stride_z
#if defined(REINTERPRET_INPUT_AS_3D)
                                                 ,
                                                 uint src_cross_plane_pad
#endif // REINTERPRET_INPUT_AS_3D
#if defined(REINTERPRET_OUTPUT_AS_3D)
                                                 ,
                                                 uint dst_cross_plane_pad
#endif // REINTERPRET_OUTPUT_AS_3D
                                                )
{
    int idx = get_global_id(0) * NUM_ELEMS_PROCESSED_PER_THREAD_X;

    // Compute starting address for matrix A and Matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes, src1_offset_first_element_in_bytes));

    // Update address for the matrix A
    src_addr.s0 += get_global_id(1) * src0_stride_y * NUM_ELEMS_PROCESSED_PER_THREAD_Y;

    // Update address for the matrix B
    src_addr.s1 += idx * sizeof(half);

#if defined(REINTERPRET_INPUT_AS_3D)
    // Since we load a 2D input tile from a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zin) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zin = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zin       = min(DEPTH_GEMM3D - 1, zin);

    // Add offset due to the cross plane paddings
    zin *= (src_cross_plane_pad * src0_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply src0_stride_z by DEPTH_GEMM3D
    src_addr.s0 += get_global_id(2) * src0_stride_z * DEPTH_GEMM3D;

#else // defined(REINTERPRET_INPUT_AS_3D)

    // Add offset for batched GEMM
    src_addr.s0 += get_global_id(2) * src0_stride_z;

#endif // defined(REINTERPRET_INPUT_AS_3D)

#if defined(MATRIX_B_DEPTH)
    // Do not slide matrix B if the matrix B has 3 dimensions and matrix A more than 3
    src_addr.s1 += (get_global_id(2) % MATRIX_B_DEPTH) * src1_stride_z;
#else  // defined(MATRIX_B_DEPTH)
    src_addr.s1 += get_global_id(2) * src1_stride_z;
#endif // defined(MATRIX_B_DEPTH)

    half8 acc0 = 0.0h;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    half8 acc1 = 0.0h;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    half8 acc2 = 0.0h;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    half8 acc3 = 0.0h;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

    int i = 0;
    for(; i <= ((int)COLS_A - 4); i += 4)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half4 a1 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half4 a2 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half4 a3 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half4 a0 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half4 a1 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half4 a2 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half4 a3 = vload4(0, (__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        half8 b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;

        // Accumulate
        acc0 = fma(b0, (half8)a0.s0, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (half8)a1.s0, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (half8)a2.s0, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (half8)a3.s0, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (half8)a0.s1, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (half8)a1.s1, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (half8)a2.s1, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (half8)a3.s1, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (half8)a0.s2, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (half8)a1.s2, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (half8)a2.s2, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (half8)a3.s2, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));
        src_addr.s1 += src1_stride_y;
        acc0 = fma(b0, (half8)a0.s3, acc0);
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (half8)a1.s3, acc1);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (half8)a2.s3, acc2);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (half8)a3.s3, acc3);
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

        src_addr.s0 += 4 * sizeof(half);
    }

    for(; i < (int)COLS_A; ++i)
    {
#if defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y + zin.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y + zin.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y + zin.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y + zin.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#else  // defined(REINTERPRET_INPUT_AS_3D)
        // Load values from matrix A
        half a0 = *((__global half *)(src0_ptr + src_addr.s0 + 0 * src0_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        half a1 = *((__global half *)(src0_ptr + src_addr.s0 + 1 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        half a2 = *((__global half *)(src0_ptr + src_addr.s0 + 2 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        half a3 = *((__global half *)(src0_ptr + src_addr.s0 + 3 * src0_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // defined(REINTERPRET_INPUT_AS_3D)

        // Load values from matrix B
        half8 b0 = vload8(0, (__global half *)(src1_ptr + src_addr.s1));

        src_addr += (int2)(sizeof(half), src1_stride_y);

        // Accumulate
        acc0 = fma(b0, (half8)a0, acc0); // b0 * (half8)a0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
        acc1 = fma(b0, (half8)a1, acc1); // b0 * (half8)a1;
#endif                                   // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
        acc2 = fma(b0, (half8)a2, acc2); // b0 * (half8)a2;
#endif                                   // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
        acc3 = fma(b0, (half8)a3, acc3); // b0 * (half8)a3;
#endif                                   // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    }

    // Multiply by the weight of matrix-matrix product and store the result
#if defined(ALPHA)
    acc0 = acc0 * (half8)ALPHA;
#endif // defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
    acc1 = acc1 * (half8)ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
    acc2 = acc2 * (half8)ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2 && defined(ALPHA)
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)
    acc3 = acc3 * (half8)ALPHA;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3 && defined(ALPHA)

#if defined(ADD_VEC_C)
    // *INDENT-OFF*
    // clang-format off
    __global half *src2_addr = (__global half *)(src2_ptr + src2_offset_first_element_in_bytes + get_global_id(0) * src2_step_x);
    half8          c0        = vload8(0, src2_addr);
    // clang-format on
    // *INDENT-ON*

    acc0 += c0;
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    acc1 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    acc2 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    acc3 += c0;
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif /* defined(ADD_VEC_C) */

    int z = get_global_id(2);

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Compute dst address
    __global uchar *dst_addr = offset(&dst, 0, 0);

#if defined(REINTERPRET_OUTPUT_AS_3D)
    // Since we store a 2D output tile in a 3D tensor, we need to check when the plane changes across the z dimension
    // in order to take into account the presence of possible cross plane paddings
    //
    //  |                  |
    //  |      plane0      |
    //  |                  |
    //  |__________________|
    //  |******************|
    //  |  cross_plane_pad |
    //  |******************|
    //  |                  |
    //  |      plane1      |
    //  |                  |
    //  |__________________|

    // The plane (zout) is calculated dividing M (get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y) by HEIGHT_GEMM3D
    uint4 zout = ((uint4)(0, 1, 2, 3) + (uint4)(get_global_id(1) * NUM_ELEMS_PROCESSED_PER_THREAD_Y)) / (uint4)HEIGHT_GEMM3D;
    zout       = min(DEPTH_GEMM3D - 1, zout);

    // Add offset due to the cross plane paddings
    zout *= (dst_cross_plane_pad * dst_stride_y);

    // Add offset for batched GEMM. The batches will be in the fourth dimension and for this reason we
    // multiply dst_stride_z by DEPTH_GEMM3D
    dst_addr += z * dst_stride_z * DEPTH_GEMM3D;

    // Store the output block
    vstore8(acc0, 0, (__global half *)(dst_addr + 0 * dst_stride_y + zout.s0));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore8(acc1, 0, (__global half *)(dst_addr + 1 * dst_stride_y + zout.s1));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore8(acc2, 0, (__global half *)(dst_addr + 2 * dst_stride_y + zout.s2));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore8(acc3, 0, (__global half *)(dst_addr + 3 * dst_stride_y + zout.s3));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3

#else // defined(REINTERPRET_OUTPUT_AS_3D)
    // Add offset for batched GEMM
    dst_addr += z * dst_stride_z;

    // Store the output block
    vstore8(acc0, 0, (__global half *)(dst_addr + 0 * dst_stride_y));
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
    vstore8(acc1, 0, (__global half *)(dst_addr + 1 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 1
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
    vstore8(acc2, 0, (__global half *)(dst_addr + 2 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 2
#if NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
    vstore8(acc3, 0, (__global half *)(dst_addr + 3 * dst_stride_y));
#endif // NUM_ELEMS_PROCESSED_PER_THREAD_Y > 3
#endif // REINTERPRET_OUTPUT_AS_3D
}
#endif // defined(ARM_COMPUTE_OPENCL_FP16_ENABLED)

#endif // defined(COLS_A) && defined(NUM_ELEMS_PROCESSED_PER_THREAD_X) && (NUM_ELEMS_PROCESSED_PER_THREAD_Y)

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

#if defined(WIDTH_VECTOR_A)
/** This OpenCL kernel computes the vector by matrix multiplication between each row of A (src0) and matrix B (src1) used for locally connected layer
 *
 * @note The width of A need to be passed at compile time using -DWIDTH_VECTOR_A
 *
 * @note The input A and matrix B must not be reshaped
 *
 * @param[in]  src0_ptr                           Pointer to the source matrix. Supported data types: F32
 * @param[in]  src0_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src0_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src0_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src0_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src0_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[in]  src1_ptr                           Pointer to the source matrix. Supported data types: same as @p src0_ptr
 * @param[in]  src1_stride_x                      Stride of the source matrix in X dimension (in bytes)
 * @param[in]  src1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src1_stride_y                      Stride of the source matrix in Y dimension (in bytes)
 * @param[in]  src1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src1_stride_z                      Stride of the source matrix in Z dimension (in bytes)
 * @param[in]  src1_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src1_offset_first_element_in_bytes The offset of the first element in the source matrix
 * @param[out] dst_ptr                            Pointer to the destination matrix Supported data types: same as @p src0_ptr
 * @param[in]  dst_stride_x                       Stride of the destination matrix in X dimension (in bytes)
 * @param[in]  dst_step_x                         dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination matrix in Y dimension (in bytes)
 * @param[in]  dst_step_y                         dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination matrix
 */
__kernel void gemm_lc_vm_f32(IMAGE_DECLARATION(src0),
                             TENSOR3D_DECLARATION(src1),
                             IMAGE_DECLARATION(dst))
{
    int idx = get_global_id(0) * 4;
    int idy = get_global_id(1);

    // Compute the address for the vector A and matrix B
    int2 src_addr = ((int2)(src0_offset_first_element_in_bytes + src0_stride_y * idy, src1_offset_first_element_in_bytes + src1_stride_z * idy));
    src_addr.s1 += idx * sizeof(float);

    int end_row_vec_a = src_addr.s0 + (WIDTH_VECTOR_A * sizeof(float));

    float4 acc = 0.0f;

    for(; src_addr.s0 <= (end_row_vec_a - 2 * (int)sizeof(float)); src_addr += (int2)(2 * sizeof(float), 2 * src1_stride_y))
    {
        float2 a0 = vload2(0, (__global float *)(src0_ptr + src_addr.s0));
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));
        float4 b1 = vload4(0, (__global float *)(src1_ptr + src_addr.s1 + src1_stride_y));

        acc += b0 * (float4)a0.s0;
        acc += b1 * (float4)a0.s1;
    }

    for(; src_addr.s0 < end_row_vec_a; src_addr += (int2)(sizeof(float), src1_stride_y))
    {
        float  a0 = *((__global float *)(src0_ptr + src_addr.s0));
        float4 b0 = vload4(0, (__global float *)(src1_ptr + src_addr.s1));

        acc += b0 * (float4)a0;
    }

    // Compute destination address
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    vstore4(acc, 0, (__global float *)(offset(&dst, 0, 0)));
}
#endif // defined(WIDTH_VECTOR_A)

/** This kernel accumulates each row with the biases vector.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=short.
 * @note The vector size must be passed at compile time using -DVECTOR_SIZE e.g. -DVECTOR_SIZE=16.
 *
 * @param[in, out] accum_ptr                            Pointer to the accumulate tensor. Supported data type: U8/S8/U16/S16/F16/U32/S32/F32
 * @param[in]      accum_stride_x                       Stride of the accmulate tensor in X dimension (in bytes)
 * @param[in]      accum_step_x                         accum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      accum_stride_y                       Stride of the accumlulate tensor in Y dimension (in bytes)
 * @param[in]      accum_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]      accum_offset_first_element_in_bytes  The offset of the first element in the accumulate tensor
 * @param[in]      biases_ptr                           Pointer to the biases vector. Same as @p accum_ptr
 * @param[in]      biases_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]      biases_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]      biases_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
#if defined(DATA_TYPE) && defined(VECTOR_SIZE)
__kernel void gemm_accumulate_biases(
    IMAGE_DECLARATION(accum),
    VECTOR_DECLARATION(biases))
{
    Image  accum  = CONVERT_TO_IMAGE_STRUCT(accum);
    Vector biases = CONVERT_TO_VECTOR_STRUCT(biases);

    // Vector size, i.e. number of vector elements.
    VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
    accum_value = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)accum.ptr);
    VEC_DATA_TYPE(DATA_TYPE, VECTOR_SIZE)
    biases_value = VLOAD(VECTOR_SIZE)(0, (__global DATA_TYPE *)biases.ptr);
    accum_value  = biases_value + accum_value;
    // Store result in the accumulate buffer
    VSTORE(VECTOR_SIZE)
    (accum_value, 0, (__global DATA_TYPE *)accum.ptr);
}
#endif // defined(DATA_TYPE) && defined(VECTOR_SIZE)
