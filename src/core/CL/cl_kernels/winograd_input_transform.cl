/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#define OUTPUT_ROW_4x4_5x5(out, tmp, comm_fact)                     \
    ({                                                              \
        comm_fact.s0 = tmp.s2 - 4.25f * tmp.s4 + tmp.s6;            \
        comm_fact.s1 = tmp.s1 - 4.25f * tmp.s3 + tmp.s5;            \
        comm_fact.s2 = 2.5f * tmp.s3;                               \
        comm_fact.s3 = 0.5f * tmp.s1 + 2.f * tmp.s5 - comm_fact.s2; \
        comm_fact.s4 = 0.25f * tmp.s2 - 1.25f * tmp.s4 + tmp.s6;    \
        comm_fact.s5 = 4.f * tmp.s2 + tmp.s6 - 5.f * tmp.s4;        \
        comm_fact.s6 = 2.f * tmp.s1 + 0.5f * tmp.s5 - comm_fact.s2; \
        \
        out.s0 = tmp.s0 - tmp.s6 + 5.25f * tmp.s4 - 5.25f * tmp.s2; \
        out.s1 = comm_fact.s0 + comm_fact.s1;                       \
        out.s2 = comm_fact.s0 - comm_fact.s1;                       \
        out.s3 = comm_fact.s3 + comm_fact.s4;                       \
        out.s4 = comm_fact.s4 - comm_fact.s3;                       \
        out.s5 = comm_fact.s5 + comm_fact.s6;                       \
        out.s6 = comm_fact.s5 - comm_fact.s6;                       \
        out.s7 = tmp.s7 - tmp.s1 + 5.25f * tmp.s3 - 5.25f * tmp.s5; \
    })

#define OUTPUT_ROW_2x2_7x7(out, tmp, comm_fact)                                                    \
    ({                                                                                             \
        comm_fact.s0 = 36.0f * tmp.s2 - 13.0f * tmp.s4 + tmp.s6;                                   \
        comm_fact.s1 = 36.0f * tmp.s1 - 13.0f * tmp.s3 + 1.0f * tmp.s5;                            \
        comm_fact.s2 = 9.0f * tmp.s2 - 10.0f * tmp.s4 + tmp.s6;                                    \
        comm_fact.s3 = 18.0f * tmp.s1 - 20.0f * tmp.s3 + 2.0f * tmp.s5;                            \
        comm_fact.s4 = 4.0f * tmp.s2 - 5.0f * tmp.s4 + tmp.s6;                                     \
        comm_fact.s5 = 12.0f * tmp.s1 - 15.0f * tmp.s3 + 3.0f * tmp.s5;                            \
        out.s0       = -36.0f * tmp.s0 + 49.0f * tmp.s2 + -14.0f * tmp.s4 + tmp.s6;                \
        out.s1       = comm_fact.s0 - comm_fact.s1;                                                \
        out.s2       = comm_fact.s0 + comm_fact.s1;                                                \
        out.s3       = comm_fact.s2 - comm_fact.s3;                                                \
        out.s4       = comm_fact.s2 + comm_fact.s3;                                                \
        out.s5       = comm_fact.s4 - comm_fact.s5;                                                \
        out.s6       = comm_fact.s4 + comm_fact.s5;                                                \
        out.s7       = -36.0f * tmp.s1 + 0.0f * tmp.s2 + 49.0f * tmp.s3 - 14.0f * tmp.s5 + tmp.s7; \
    })

#if defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)
/** This OpenCL kernel computes the input transform when the kernel size is 3x3/3x1 or 1x3 and the output tile is 2x2/2x1 or 1x2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_2x2_3x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
#if defined(SRC_DEPTH)
    const int z = get_global_id(2) % SRC_DEPTH;
    const int b = get_global_id(2) / SRC_DEPTH;
#else  /* defined(SRC_DEPTH) */
    const int z              = get_global_id(2);
#endif /* defined(SRC_DEPTH) */

    // Compute input address
#if defined(SRC_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z + b * src_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;
#endif /* defined(SRC_DEPTH) */

    src_addr = src_addr - ((int)PAD_LEFT * sizeof(DATA_TYPE)) - ((int)PAD_TOP * src_stride_y);

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row0 = vload4(0, (__global DATA_TYPE *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(*((__global DATA_TYPE *)(src_addr + 0 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 1 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 2 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 3 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row0 = vload4(0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row1 = vload4(0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row2 = vload4(0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row3 = vload4(0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp0 = in_row0;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    tmp0 -= in_row2;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    DATA_TYPE out00 = tmp0.s0 - tmp0.s2;
    DATA_TYPE out01 = tmp0.s1 + tmp0.s2;
    DATA_TYPE out02 = tmp0.s2 - tmp0.s1;
    DATA_TYPE out03 = tmp0.s1 - tmp0.s3;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp1 = in_row1 + in_row2;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp2 = in_row2 - in_row1;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp3 = in_row1 - in_row3;

    DATA_TYPE out10 = tmp1.s0 - tmp1.s2;
    DATA_TYPE out11 = tmp1.s1 + tmp1.s2;
    DATA_TYPE out12 = tmp1.s2 - tmp1.s1;
    DATA_TYPE out13 = tmp1.s1 - tmp1.s3;

    DATA_TYPE out20 = tmp2.s0 - tmp2.s2;
    DATA_TYPE out21 = tmp2.s1 + tmp2.s2;
    DATA_TYPE out22 = tmp2.s2 - tmp2.s1;
    DATA_TYPE out23 = tmp2.s1 - tmp2.s3;

    DATA_TYPE out30 = tmp3.s0 - tmp3.s2;
    DATA_TYPE out31 = tmp3.s1 + tmp3.s2;
    DATA_TYPE out32 = tmp3.s2 - tmp3.s1;
    DATA_TYPE out33 = tmp3.s1 - tmp3.s3;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

#if defined(SRC_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y + b * dst_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y;
#endif /* defined(SRC_DEPTH) */

    *((__global DATA_TYPE *)(dst_addr + 0 * dst_stride_z)) = out00; // in_row0.s0; out00;
    *((__global DATA_TYPE *)(dst_addr + 1 * dst_stride_z)) = out01; // in_row0.s1; out01;
    *((__global DATA_TYPE *)(dst_addr + 2 * dst_stride_z)) = out02; // in_row0.s2; out02;
    *((__global DATA_TYPE *)(dst_addr + 3 * dst_stride_z)) = out03; // in_row0.s3; out03;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    *((__global DATA_TYPE *)(dst_addr + 4 * dst_stride_z))  = out10;
    *((__global DATA_TYPE *)(dst_addr + 5 * dst_stride_z))  = out11;
    *((__global DATA_TYPE *)(dst_addr + 6 * dst_stride_z))  = out12;
    *((__global DATA_TYPE *)(dst_addr + 7 * dst_stride_z))  = out13;
    *((__global DATA_TYPE *)(dst_addr + 8 * dst_stride_z))  = out20;
    *((__global DATA_TYPE *)(dst_addr + 9 * dst_stride_z))  = out21;
    *((__global DATA_TYPE *)(dst_addr + 10 * dst_stride_z)) = out22;
    *((__global DATA_TYPE *)(dst_addr + 11 * dst_stride_z)) = out23;
    *((__global DATA_TYPE *)(dst_addr + 12 * dst_stride_z)) = out30;
    *((__global DATA_TYPE *)(dst_addr + 13 * dst_stride_z)) = out31;
    *((__global DATA_TYPE *)(dst_addr + 14 * dst_stride_z)) = out32;
    *((__global DATA_TYPE *)(dst_addr + 15 * dst_stride_z)) = out33;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

/** This OpenCL kernel computes the input transform when the kernel size is 3x3/3x1 or 1x3, the output tile is 2x2/2x1 or 1x2 and the number of channels is multiple of 2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_2x2_3x3_stepz2_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
#if defined(SRC_DEPTH)
    const int z = (get_global_id(2) * 2) % SRC_DEPTH;
    const int b = (get_global_id(2) * 2) / SRC_DEPTH;
#else  /* defined(SRC_DEPTH) */
    const int       z        = get_global_id(2) * 2;
#endif /* defined(SRC_DEPTH) */

    // Compute input address
#if defined(SRC_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z + b * src_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;
#endif /* defined(SRC_DEPTH) */
    src_addr = src_addr - ((int)PAD_LEFT * sizeof(DATA_TYPE)) - ((int)PAD_TOP * src_stride_y);

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row0 = vload4(0, (__global DATA_TYPE *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row0 = (VEC_DATA_TYPE(DATA_TYPE, 4))(*((__global DATA_TYPE *)(src_addr + 0 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 1 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 2 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 3 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row0 = vload4(0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row1 = vload4(0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row2 = vload4(0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row3 = vload4(0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    src_addr += src_stride_z;
#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row4 = vload4(0, (__global DATA_TYPE *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row4 = (VEC_DATA_TYPE(DATA_TYPE, 4))(*((__global DATA_TYPE *)(src_addr + 0 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 1 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 2 * src_stride_y)),
                                            *((__global DATA_TYPE *)(src_addr + 3 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row4 = vload4(0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row5 = vload4(0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row6 = vload4(0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 4)
    in_row7 = vload4(0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp0 = in_row0;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp4 = in_row4;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    tmp0 -= in_row2;
    tmp4 -= in_row6;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    VEC_DATA_TYPE(DATA_TYPE, 2)
    out00 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp0.s0 - tmp0.s2, tmp4.s0 - tmp4.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out01 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp0.s1 + tmp0.s2, tmp4.s1 + tmp4.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out02 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp0.s2 - tmp0.s1, tmp4.s2 - tmp4.s1);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out03 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp0.s1 - tmp0.s3, tmp4.s1 - tmp4.s3);

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp1 = in_row1 + in_row2;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp2 = in_row2 - in_row1;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp3 = in_row1 - in_row3;

    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp5 = in_row5 + in_row6;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp6 = in_row6 - in_row5;
    VEC_DATA_TYPE(DATA_TYPE, 4)
    tmp7 = in_row5 - in_row7;

    VEC_DATA_TYPE(DATA_TYPE, 2)
    out10 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp1.s0 - tmp1.s2, tmp5.s0 - tmp5.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out11 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp1.s1 + tmp1.s2, tmp5.s1 + tmp5.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out12 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp1.s2 - tmp1.s1, tmp5.s2 - tmp5.s1);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out13 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp1.s1 - tmp1.s3, tmp5.s1 - tmp5.s3);

    VEC_DATA_TYPE(DATA_TYPE, 2)
    out20 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp2.s0 - tmp2.s2, tmp6.s0 - tmp6.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out21 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp2.s1 + tmp2.s2, tmp6.s1 + tmp6.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out22 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp2.s2 - tmp2.s1, tmp6.s2 - tmp6.s1);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out23 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp2.s1 - tmp2.s3, tmp6.s1 - tmp6.s3);

    VEC_DATA_TYPE(DATA_TYPE, 2)
    out30 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp3.s0 - tmp3.s2, tmp7.s0 - tmp7.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out31 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp3.s1 + tmp3.s2, tmp7.s1 + tmp7.s2);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out32 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp3.s2 - tmp3.s1, tmp7.s2 - tmp7.s1);
    VEC_DATA_TYPE(DATA_TYPE, 2)
    out33 = (VEC_DATA_TYPE(DATA_TYPE, 2))(tmp3.s1 - tmp3.s3, tmp7.s1 - tmp7.s3);
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

#if defined(SRC_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y + b * dst_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y;
#endif /* defined(SRC_DEPTH) */

    vstore2(out00, 0, (__global DATA_TYPE *)(dst_addr + 0 * dst_stride_z));
    vstore2(out01, 0, (__global DATA_TYPE *)(dst_addr + 1 * dst_stride_z));
    vstore2(out02, 0, (__global DATA_TYPE *)(dst_addr + 2 * dst_stride_z));
    vstore2(out03, 0, (__global DATA_TYPE *)(dst_addr + 3 * dst_stride_z));

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    vstore2(out10, 0, (__global DATA_TYPE *)(dst_addr + 4 * dst_stride_z));
    vstore2(out11, 0, (__global DATA_TYPE *)(dst_addr + 5 * dst_stride_z));
    vstore2(out12, 0, (__global DATA_TYPE *)(dst_addr + 6 * dst_stride_z));
    vstore2(out13, 0, (__global DATA_TYPE *)(dst_addr + 7 * dst_stride_z));
    vstore2(out20, 0, (__global DATA_TYPE *)(dst_addr + 8 * dst_stride_z));
    vstore2(out21, 0, (__global DATA_TYPE *)(dst_addr + 9 * dst_stride_z));
    vstore2(out22, 0, (__global DATA_TYPE *)(dst_addr + 10 * dst_stride_z));
    vstore2(out23, 0, (__global DATA_TYPE *)(dst_addr + 11 * dst_stride_z));
    vstore2(out30, 0, (__global DATA_TYPE *)(dst_addr + 12 * dst_stride_z));
    vstore2(out31, 0, (__global DATA_TYPE *)(dst_addr + 13 * dst_stride_z));
    vstore2(out32, 0, (__global DATA_TYPE *)(dst_addr + 14 * dst_stride_z));
    vstore2(out33, 0, (__global DATA_TYPE *)(dst_addr + 15 * dst_stride_z));
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

/** This OpenCL kernel computes the input transform when the output tile is 4x4/4x1 or 1x4, the filter size 3x3/3x1 or 1x3 and the data layout is NCHW
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_4x4_3x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
#if defined(SRC_DEPTH)
    const int z = get_global_id(2) % SRC_DEPTH;
    const int b = get_global_id(2) / SRC_DEPTH;
#else  /* defined(SRC_DEPTH) */
    const int       z        = get_global_id(2);
#endif /* defined(SRC_DEPTH) */

    // Compute input address
#if defined(SRC_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z + b * src_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;
#endif /* defined(SRC_DEPTH) */

    src_addr = src_addr - ((int)PAD_LEFT * sizeof(DATA_TYPE)) - ((int)PAD_TOP * src_stride_y);

#if defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row0
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d00 = (VEC_DATA_TYPE(DATA_TYPE, 4))(*((__global DATA_TYPE *)(src_addr + 0 * src_stride_y)),
                                        *((__global DATA_TYPE *)(src_addr + 1 * src_stride_y)),
                                        *((__global DATA_TYPE *)(src_addr + 2 * src_stride_y)),
                                        *((__global DATA_TYPE *)(src_addr + 3 * src_stride_y)));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d01 = (VEC_DATA_TYPE(DATA_TYPE, 2))(*((__global DATA_TYPE *)(src_addr + 4 * src_stride_y)),
                                        *((__global DATA_TYPE *)(src_addr + 5 * src_stride_y)));
#else  // defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row0
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d00 = vload4(0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d01                                        = vload2(2, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y));
#endif // defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    DATA_TYPE out0 = 0.0f;
    DATA_TYPE out1 = 0.0f;
    DATA_TYPE out2 = 0.0f;
    DATA_TYPE out3 = 0.0f;
    DATA_TYPE out4 = 0.0f;
    DATA_TYPE out5 = 0.0f;

    // Channels [0, 5]: [out00, out01, out02, out03, out04, out05]
    out0 += 16.0f * d00.s0 - 20.0f * d00.s2 + 4.0f * d01.s0;
    out1 += -16.0f * d00.s1 - 16.0f * d00.s2 + 4.0f * d00.s3 + 4.0f * d01.s0;
    out2 += 16.0f * d00.s1 - 16.0f * d00.s2 - 4.0f * d00.s3 + 4.0f * d01.s0;
    out3 += -8.0f * d00.s1 - 4.0f * d00.s2 + 8.0f * d00.s3 + 4.0f * d01.s0;
    out4 += 8.0f * d00.s1 - 4.0f * d00.s2 - 8.0f * d00.s3 + 4.0f * d01.s0;
    out5 += 16.0f * d00.s1 - 20.0f * d00.s3 + 4.0f * d01.s1;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row4
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d40 = vload4(0, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d41 = vload2(2, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y));

    // k0, k1, k2, k3, k4, k5 are common terms for row0, row1, row2, row3 and row4
    DATA_TYPE k0 = d41.s0;
    DATA_TYPE k1 = d41.s0;
    DATA_TYPE k2 = d41.s0;
    DATA_TYPE k3 = d41.s0;
    DATA_TYPE k4 = d41.s0;
    DATA_TYPE k5 = 0.0f;

    k0 += 4.0f * d40.s0 - 5.0f * d40.s2;
    k1 += -4.0f * d40.s1 - 4.0f * d40.s2 + d40.s3;
    k2 += 4.0f * d40.s1 - 4.0f * d40.s2 - d40.s3;
    k3 += -2.0f * d40.s1 + 2.0f * d40.s3 - d40.s2;
    k4 += 2.0f * d40.s1 - 2.0f * d40.s3 - d40.s2;
    k5 += 4.0f * d40.s1 - 5.0f * d40.s3 + d41.s1;

    out0 += k0;
    out1 += k1;
    out2 += k2;
    out3 += k3;
    out4 += k4;
    out5 += k5;

    // Row2
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d20 = vload4(0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d21 = vload2(2, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));

    out0 += -20.0f * d20.s0 + 25.0f * d20.s2 - 5.0f * d21.s0;
    out1 += +20.0f * d20.s1 + 20.0f * d20.s2 - 5.0f * d20.s3 - 5.0f * d21.s0;
    out2 += -20.0f * d20.s1 + 20.0f * d20.s2 + 5.0f * d20.s3 - 5.0f * d21.s0;
    out3 += +10.0f * d20.s1 + 5.0f * d20.s2 - 10.0f * d20.s3 - 5.0f * d21.s0;
    out4 += -10.0f * d20.s1 + 5.0f * d20.s2 + 10.0f * d20.s3 - 5.0f * d21.s0;
    out5 += -20.0f * d20.s1 + 25.0f * d20.s3 - 5.0f * d21.s1;
#endif // #if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Compute destination address
#if defined(SRC_DEPTH)
    __global DATA_TYPE *dst_addr = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y + b * dst_stride_w);
#else  /* defined(SRC_DEPTH) */
    __global DATA_TYPE *dst_addr               = (__global DATA_TYPE *)(dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y);
#endif /* defined(SRC_DEPTH) */

    uint dst_plane_stride = dst_stride_z / sizeof(DATA_TYPE);

    *(dst_addr) = out0;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out1;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out2;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out3;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out4;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out5;
    dst_addr += dst_plane_stride;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    DATA_TYPE out6  = k0;
    DATA_TYPE out7  = k1;
    DATA_TYPE out8  = k2;
    DATA_TYPE out9  = k3;
    DATA_TYPE out10 = k4;
    DATA_TYPE out11 = k5;
    DATA_TYPE out12 = k0;
    DATA_TYPE out13 = k1;
    DATA_TYPE out14 = k2;
    DATA_TYPE out15 = k3;
    DATA_TYPE out16 = k4;
    DATA_TYPE out17 = k5;
    DATA_TYPE out18 = k0;
    DATA_TYPE out19 = k1;
    DATA_TYPE out20 = k2;
    DATA_TYPE out21 = k3;
    DATA_TYPE out22 = k4;
    DATA_TYPE out23 = k5;
    DATA_TYPE out24 = k0;
    DATA_TYPE out25 = k1;
    DATA_TYPE out26 = k2;
    DATA_TYPE out27 = k3;
    DATA_TYPE out28 = k4;
    DATA_TYPE out29 = k5;

    // Row1
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d10 = vload4(0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d11 = vload2(2, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y));

    // Row3
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d30 = vload4(0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d31 = vload2(2, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));

    // Compute common parts for the channels between [6, 29]
    // Channels [6, 11]:  [out10, out11, out12, out13, out14, out15]
    // Channels [12, 17]: [out20, out21, out22, out23, out24, out25]
    DATA_TYPE part0  = -16.0f * d20.s0 + 20.0f * d20.s2 - 4.0f * d21.s0;
    DATA_TYPE part1  = 16.0f * d10.s0 - 20.0f * d10.s2 + 4.0f * d11.s0 - 4.0f * d30.s0 + 5.0f * d30.s2 - d31.s0;
    DATA_TYPE part2  = 16.0f * d20.s2 - 4.0f * d21.s0;
    DATA_TYPE part3  = 16.0f * d20.s1 - 4.0f * d20.s3;
    DATA_TYPE part4  = 16.0f * d10.s2 - 4.0f * d11.s0 - 4.0f * d30.s2 + d31.s0;
    DATA_TYPE part5  = 16.0f * d10.s1 - 4.0f * d10.s3 - 4.0f * d30.s1 + d30.s3;
    DATA_TYPE part6  = 4.0f * d20.s2 - 4.0f * d21.s0;
    DATA_TYPE part7  = 8.0f * d10.s1 - 8.0f * d10.s3 - 2.0f * d30.s1 + 2.0f * d30.s3;
    DATA_TYPE part8  = 4.0f * d10.s2 - 4.0f * d11.s0 - d30.s2 + d31.s0;
    DATA_TYPE part9  = 8.0f * d20.s1 - 8.0f * d20.s3;
    DATA_TYPE part10 = -16.0f * d20.s1 + 20.0f * d20.s3 - 4.0f * d21.s1;
    DATA_TYPE part11 = -16.0f * d10.s1 + 20.0f * d10.s3 - 4.0f * d11.s1 + 4.0f * d30.s1 - 5.0f * d30.s3 + d31.s1;

    // Channels [18, 23]: [out30, out31, out32, out33, out34, out35]
    // Channels [24, 29]: [out40, out41, out42, out43, out44, out45]
    DATA_TYPE part12 = 8.0f * d10.s0 - 10.0f * d10.s2 + 2.0f * d11.s0 - 8.0f * d30.s0 + 10.0f * d30.s2 - 2.0f * d31.s0;
    DATA_TYPE part13 = part0 * 0.25f; // -4.0f * d20.s0 + 5.0f * d20.s2 - d21.s0
    DATA_TYPE part14 = part2 * 0.25f; // 4.0f * d20.s2 - d21.s0
    DATA_TYPE part15 = 8.0f * d10.s1 - 2.0f * d10.s3 - 8.0f * d30.s1 + 2.0f * d30.s3;
    DATA_TYPE part16 = 8.0f * d10.s2 - 2.0f * d11.s0 - 8.0f * d30.s2 + 2.0f * d31.s0;
    DATA_TYPE part17 = part3 * 0.25f; // 4.0f * d20.s1 - d20.s3
    DATA_TYPE part18 = part6 * 0.25f; // d20.s2 - d21.s0
    DATA_TYPE part19 = 4.0f * d10.s1 - 4.0f * d10.s3 - 4.0f * d30.s1 + 4.0f * d30.s3;
    DATA_TYPE part20 = 2.0f * d10.s2 - 2.0f * d11.s0 - 2.0f * d30.s2 + 2.0f * d31.s0;
    DATA_TYPE part21 = part9 * 0.25f;                                                 // 2.0f * (d20.s1 - d20.s3)
    DATA_TYPE part22 = part10 * 0.25f;                                                // - 4.0f * d20.s1 + 5.0f * d20.s3 - d21.s1
    DATA_TYPE part23 = part11 * 0.5f + 6.0f * d30.s1 - 7.5f * d30.s3 + 1.5f * d31.s1; // - 8.0f * d10.s1 + 10.0f * d10.s3 - 2.0f * d11.s1 + 8.0f * d30.s1 - 10.0f * d30.s3 + 2.0f * d31.s1;

    out6 += part0 - part1;
    out12 += part0 + part1;
    out7 += part2 + part3 + part4 + part5;
    out8 += part2 - part3 + part4 - part5;
    out13 += part2 + part3 - part4 - part5;
    out14 += part2 - part3 - part4 + part5;
    out9 += part6 + part7 + part8 + part9;
    out10 += part6 - part7 + part8 - part9;
    out15 += part6 - part7 - part8 + part9;
    out16 += part6 + part7 - part8 - part9;
    out11 += part10 + part11;
    out17 += part10 - part11;

    out18 += part13 - part12;
    out24 += part13 + part12;
    out19 += part14 + part15 + part16 + part17;
    out20 += part14 - part15 + part16 - part17;
    out25 += part14 - part15 - part16 + part17;
    out26 += part14 + part15 - part16 - part17;
    out21 += part18 + part19 + part20 + part21;
    out22 += part18 - part19 + part20 - part21;
    out27 += part18 - part19 - part20 + part21;
    out28 += part18 + part19 - part20 - part21;
    out23 += part22 + part23;
    out29 += part22 - part23;

    *(dst_addr) = out6;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out7;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out8;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out9;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out10;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out11;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out12;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out13;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out14;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out15;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out16;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out17;
    dst_addr += dst_plane_stride;

    *(dst_addr) = out18;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out19;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out20;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out21;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out22;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out23;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out24;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out25;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out26;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out27;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out28;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out29;
    dst_addr += dst_plane_stride;

    // Row5
    VEC_DATA_TYPE(DATA_TYPE, 4)
    d50 = vload4(0, (__global DATA_TYPE *)(src_addr + 5 * src_stride_y));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    d51 = vload2(2, (__global DATA_TYPE *)(src_addr + 5 * src_stride_y));

    // Channels [30, 35]
    out0 = 16.0f * d10.s0 - 20.0f * d10.s2 - 20.0f * d30.s0 + 25.0f * d30.s2 + 4.0f * d50.s0 - 5.0f * d50.s2 + d51.s0 + 4.0f * d11.s0 - 5.0f * d31.s0;
    out1 = -16.0f * d10.s1 - 16.0f * d10.s2 + 4.0f * d10.s3 + 20.0f * d30.s1 + 20.0f * d30.s2 - 5.0f * d30.s3 - 4.0f * d50.s1 - 4.0f * d50.s2 + d50.s3 + d51.s0 + 4.0f * d11.s0 - 5.0f * d31.s0;
    out2 = 16.0f * d10.s1 - 16.0f * d10.s2 - 4.0f * d10.s3 - 20.0f * d30.s1 + 20.0f * d30.s2 + 5.0f * d30.s3 + 4.0f * d50.s1 - 4.0f * d50.s2 - d50.s3 + d51.s0 + 4.0f * d11.s0 - 5.0f * d31.s0;
    out3 = -8.0f * d10.s1 - 4.0f * d10.s2 + 8.0f * d10.s3 + 10.0f * d30.s1 - 10.0f * d30.s3 + 5.0f * d30.s2 - 2.0f * d50.s1 + 2.0f * d50.s3 - d50.s2 + d51.s0 + 4.0f * d11.s0 - 5.0f * d31.s0;
    out4 = 8.0f * d10.s1 - 4.0f * d10.s2 - 8.0f * d10.s3 - 10.0f * d30.s1 + 5.0f * d30.s2 + 10.0f * d30.s3 + 2.0f * d50.s1 - 2.0f * d50.s3 - d50.s2 + d51.s0 + 4.0f * d11.s0 - 5.0f * d31.s0;
    out5 = 16.0f * d10.s1 - 20.0f * d10.s3 + 4.0f * d11.s1 - 20.0f * d30.s1 + 25.0f * d30.s3 - 5.0f * d31.s1 + 4.0f * d50.s1 - 5.0f * d50.s3 + d51.s1;

    *(dst_addr) = out0;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out1;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out2;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out3;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out4;
    dst_addr += dst_plane_stride;
    *(dst_addr) = out5;
    dst_addr += dst_plane_stride;
#endif // #if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

/** This OpenCL kernel computes the input transform when the kernel size is 5x5/5x1 or 1x5 and the output tile is 4x4/4x1 or 1x4 when the data layout is NCHW
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd input transform 5x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x5, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_4x4_5x5_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
#if defined(SRC_DEPTH)
    const int z = get_global_id(2) % SRC_DEPTH;
    const int b = get_global_id(2) / SRC_DEPTH;
#else  /* defined(SRC_DEPTH) */
    const int                                z = get_global_id(2);
#endif /* defined(SRC_DEPTH) */

    // Compute input address
#if defined(SRC_DEPTH)
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z + b * src_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *src_addr                   = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(DATA_TYPE) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;
#endif /* defined(SRC_DEPTH) */
    src_addr = src_addr - ((int)PAD_LEFT * sizeof(DATA_TYPE)) - ((int)PAD_TOP * src_stride_y);

    // Load input tile
#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row0 = vload8(0, (__global DATA_TYPE *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row0 = (VEC_DATA_TYPE(DATA_TYPE, 8))(*((__global DATA_TYPE *)(src_addr + 0 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 1 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 2 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 3 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 4 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 5 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 6 * src_stride_y)),
                                                                              *((__global DATA_TYPE *)(src_addr + 7 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row0 = vload8(0, (__global DATA_TYPE *)(src_addr + 0 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row1 = vload8(0, (__global DATA_TYPE *)(src_addr + 1 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row2 = vload8(0, (__global DATA_TYPE *)(src_addr + 2 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row3 = vload8(0, (__global DATA_TYPE *)(src_addr + 3 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row4 = vload8(0, (__global DATA_TYPE *)(src_addr + 4 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row5 = vload8(0, (__global DATA_TYPE *)(src_addr + 5 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row6 = vload8(0, (__global DATA_TYPE *)(src_addr + 6 * src_stride_y));
    const VEC_DATA_TYPE(DATA_TYPE, 8) in_row7 = vload8(0, (__global DATA_TYPE *)(src_addr + 7 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Calculate common factors for intermediate tensor
    VEC_DATA_TYPE(DATA_TYPE, 8)
    tmp0 = in_row0;
    VEC_DATA_TYPE(DATA_TYPE, 8)
    comm_fact0 = 0.0f;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    comm_fact0 += in_row2 + in_row6 - (DATA_TYPE)4.25f * in_row4;
    tmp0 += -in_row6 + (DATA_TYPE)5.25f * in_row4 - (DATA_TYPE)5.25f * in_row2;

    VEC_DATA_TYPE(DATA_TYPE, 8)
    comm_fact1 = in_row1 + in_row5 - (DATA_TYPE)4.25f * in_row3;
    VEC_DATA_TYPE(DATA_TYPE, 8)
    comm_fact2 = (DATA_TYPE)0.25f * in_row2 - (DATA_TYPE)1.25f * in_row4 + in_row6;

    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp1 = comm_fact0 + comm_fact1;
    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp2 = comm_fact0 - comm_fact1;

    comm_fact0 = (DATA_TYPE)2.5f * in_row3;
    comm_fact1 = (DATA_TYPE)0.5f * in_row1 - comm_fact0 + (DATA_TYPE)2.0f * in_row5;

    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp3 = comm_fact1 + comm_fact2;
    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp4 = comm_fact2 - comm_fact1;

    comm_fact1 = (DATA_TYPE)2.0f * in_row1 - comm_fact0 + (DATA_TYPE)0.5f * in_row5;
    comm_fact2 = (DATA_TYPE)4.0f * in_row2 - (DATA_TYPE)5.0f * in_row4 + in_row6;

    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp5 = comm_fact1 + comm_fact2;
    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp6 = comm_fact2 - comm_fact1;
    const VEC_DATA_TYPE(DATA_TYPE, 8) tmp7 = in_row7 - in_row1 + (DATA_TYPE)5.25f * in_row3 - (DATA_TYPE)5.25f * in_row5;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Calculate output rows (reuse comm_fact0 vector)
    VEC_DATA_TYPE(DATA_TYPE, 8)
    out0;

    OUTPUT_ROW_4x4_5x5(out0, tmp0, comm_fact0);

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    VEC_DATA_TYPE(DATA_TYPE, 8)
    out1, out2, out3, out4, out5, out6, out7;

    OUTPUT_ROW_4x4_5x5(out1, tmp1, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out2, tmp2, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out3, tmp3, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out4, tmp4, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out5, tmp5, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out6, tmp6, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out7, tmp7, comm_fact0);
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Store values across the channels
#if defined(SRC_DEPTH)
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y + b * dst_stride_w;
#else  /* defined(SRC_DEPTH) */
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(DATA_TYPE) + (x + y * (int)NUM_TILES_X) * dst_stride_y;
#endif /* defined(SRC_DEPTH) */

    *((__global DATA_TYPE *)(dst_addr + 0 * dst_stride_z)) = out0.s0;
    *((__global DATA_TYPE *)(dst_addr + 1 * dst_stride_z)) = out0.s1;
    *((__global DATA_TYPE *)(dst_addr + 2 * dst_stride_z)) = out0.s2;
    *((__global DATA_TYPE *)(dst_addr + 3 * dst_stride_z)) = out0.s3;
    *((__global DATA_TYPE *)(dst_addr + 4 * dst_stride_z)) = out0.s4;
    *((__global DATA_TYPE *)(dst_addr + 5 * dst_stride_z)) = out0.s5;
    *((__global DATA_TYPE *)(dst_addr + 6 * dst_stride_z)) = out0.s6;
    *((__global DATA_TYPE *)(dst_addr + 7 * dst_stride_z)) = out0.s7;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    *((__global DATA_TYPE *)(dst_addr + 8 * dst_stride_z))  = out1.s0;
    *((__global DATA_TYPE *)(dst_addr + 9 * dst_stride_z))  = out1.s1;
    *((__global DATA_TYPE *)(dst_addr + 10 * dst_stride_z)) = out1.s2;
    *((__global DATA_TYPE *)(dst_addr + 11 * dst_stride_z)) = out1.s3;
    *((__global DATA_TYPE *)(dst_addr + 12 * dst_stride_z)) = out1.s4;
    *((__global DATA_TYPE *)(dst_addr + 13 * dst_stride_z)) = out1.s5;
    *((__global DATA_TYPE *)(dst_addr + 14 * dst_stride_z)) = out1.s6;
    *((__global DATA_TYPE *)(dst_addr + 15 * dst_stride_z)) = out1.s7;
    *((__global DATA_TYPE *)(dst_addr + 16 * dst_stride_z)) = out2.s0;
    *((__global DATA_TYPE *)(dst_addr + 17 * dst_stride_z)) = out2.s1;
    *((__global DATA_TYPE *)(dst_addr + 18 * dst_stride_z)) = out2.s2;
    *((__global DATA_TYPE *)(dst_addr + 19 * dst_stride_z)) = out2.s3;
    *((__global DATA_TYPE *)(dst_addr + 20 * dst_stride_z)) = out2.s4;
    *((__global DATA_TYPE *)(dst_addr + 21 * dst_stride_z)) = out2.s5;
    *((__global DATA_TYPE *)(dst_addr + 22 * dst_stride_z)) = out2.s6;
    *((__global DATA_TYPE *)(dst_addr + 23 * dst_stride_z)) = out2.s7;
    *((__global DATA_TYPE *)(dst_addr + 24 * dst_stride_z)) = out3.s0;
    *((__global DATA_TYPE *)(dst_addr + 25 * dst_stride_z)) = out3.s1;
    *((__global DATA_TYPE *)(dst_addr + 26 * dst_stride_z)) = out3.s2;
    *((__global DATA_TYPE *)(dst_addr + 27 * dst_stride_z)) = out3.s3;
    *((__global DATA_TYPE *)(dst_addr + 28 * dst_stride_z)) = out3.s4;
    *((__global DATA_TYPE *)(dst_addr + 29 * dst_stride_z)) = out3.s5;
    *((__global DATA_TYPE *)(dst_addr + 30 * dst_stride_z)) = out3.s6;
    *((__global DATA_TYPE *)(dst_addr + 31 * dst_stride_z)) = out3.s7;
    *((__global DATA_TYPE *)(dst_addr + 32 * dst_stride_z)) = out4.s0;
    *((__global DATA_TYPE *)(dst_addr + 33 * dst_stride_z)) = out4.s1;
    *((__global DATA_TYPE *)(dst_addr + 34 * dst_stride_z)) = out4.s2;
    *((__global DATA_TYPE *)(dst_addr + 35 * dst_stride_z)) = out4.s3;
    *((__global DATA_TYPE *)(dst_addr + 36 * dst_stride_z)) = out4.s4;
    *((__global DATA_TYPE *)(dst_addr + 37 * dst_stride_z)) = out4.s5;
    *((__global DATA_TYPE *)(dst_addr + 38 * dst_stride_z)) = out4.s6;
    *((__global DATA_TYPE *)(dst_addr + 39 * dst_stride_z)) = out4.s7;
    *((__global DATA_TYPE *)(dst_addr + 40 * dst_stride_z)) = out5.s0;
    *((__global DATA_TYPE *)(dst_addr + 41 * dst_stride_z)) = out5.s1;
    *((__global DATA_TYPE *)(dst_addr + 42 * dst_stride_z)) = out5.s2;
    *((__global DATA_TYPE *)(dst_addr + 43 * dst_stride_z)) = out5.s3;
    *((__global DATA_TYPE *)(dst_addr + 44 * dst_stride_z)) = out5.s4;
    *((__global DATA_TYPE *)(dst_addr + 45 * dst_stride_z)) = out5.s5;
    *((__global DATA_TYPE *)(dst_addr + 46 * dst_stride_z)) = out5.s6;
    *((__global DATA_TYPE *)(dst_addr + 47 * dst_stride_z)) = out5.s7;
    *((__global DATA_TYPE *)(dst_addr + 48 * dst_stride_z)) = out6.s0;
    *((__global DATA_TYPE *)(dst_addr + 49 * dst_stride_z)) = out6.s1;
    *((__global DATA_TYPE *)(dst_addr + 50 * dst_stride_z)) = out6.s2;
    *((__global DATA_TYPE *)(dst_addr + 51 * dst_stride_z)) = out6.s3;
    *((__global DATA_TYPE *)(dst_addr + 52 * dst_stride_z)) = out6.s4;
    *((__global DATA_TYPE *)(dst_addr + 53 * dst_stride_z)) = out6.s5;
    *((__global DATA_TYPE *)(dst_addr + 54 * dst_stride_z)) = out6.s6;
    *((__global DATA_TYPE *)(dst_addr + 55 * dst_stride_z)) = out6.s7;
    *((__global DATA_TYPE *)(dst_addr + 56 * dst_stride_z)) = out7.s0;
    *((__global DATA_TYPE *)(dst_addr + 57 * dst_stride_z)) = out7.s1;
    *((__global DATA_TYPE *)(dst_addr + 58 * dst_stride_z)) = out7.s2;
    *((__global DATA_TYPE *)(dst_addr + 59 * dst_stride_z)) = out7.s3;
    *((__global DATA_TYPE *)(dst_addr + 60 * dst_stride_z)) = out7.s4;
    *((__global DATA_TYPE *)(dst_addr + 61 * dst_stride_z)) = out7.s5;
    *((__global DATA_TYPE *)(dst_addr + 62 * dst_stride_z)) = out7.s6;
    *((__global DATA_TYPE *)(dst_addr + 63 * dst_stride_z)) = out7.s7;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

#if defined(NHWC) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(NUM_TILES_X) && defined(NUM_TILES_Y)
//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the output tile is 4x4, 4x1 or 1x4, the filter size 3x3, 3x1 or 1x3 and the data layout is NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x4_3x3_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    const int cout = GET_SPATIAL_IDX(0, 1, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0); // NUM_TILES_X x NUM_TILES_Y
    const int bout = GET_SPATIAL_IDX(2, 1, 0); // BATCH SIZE IDX

    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _INUM_TILES_X NUM_TILES_X
#define _INUM_TILES_Y NUM_TILES_Y

    int x = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;
    x -= PAD_LEFT;
    y -= PAD_TOP;

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 6, 1, in);
    TILE(DATA_TYPE, 6, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        in[i].v = 0;
    })

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 1, 6, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 6, 1, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

    TILE(DATA_TYPE, 6, 1, com);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        in[i].v *= 4.0f;
    })

    com[0].v = in[2].v - 4.f * in[0].v;
    com[1].v = in[3].v - 4.f * in[1].v;
    com[2].v = in[4].v - 4.f * in[2].v;
    com[3].v = in[5].v - 4.f * in[3].v;
    com[4].v = in[3].v - in[1].v;
    com[4].v = com[4].v + com[4].v;
    com[5].v = in[4].v - in[2].v;

    out[0].v = com[2].v - com[0].v;
    out[1].v = com[2].v + com[1].v;
    out[2].v = com[2].v - com[1].v;
    out[3].v = com[5].v + com[4].v;
    out[4].v = com[5].v - com[4].v;
    out[5].v = com[3].v - com[1].v;

    TILE(uint, 6, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        dst_indirect_y[i].v = mout + i * _INUM_TILES_X * _INUM_TILES_Y;
        dst_indirect_y[i].v += bout * _INUM_TILES_X * _INUM_TILES_Y * 6;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 6, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 36, 1, in);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 36,
    {
        in[i].v = 0;
    })

    // Load the tile from a NHWC tensor
    T_LOAD_NHWC(DATA_TYPE, 6, 6, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);

    TILE(DATA_TYPE, 6, 1, com);
    TILE(DATA_TYPE, 36, 1, tmp);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        com[0].v         = in[2 * 6 + i].v - (DATA_TYPE)4.0f * in[0 * 6 + i].v;
        com[1].v         = in[3 * 6 + i].v - (DATA_TYPE)4.0f * in[1 * 6 + i].v;
        com[2].v         = in[4 * 6 + i].v - (DATA_TYPE)4.0f * in[2 * 6 + i].v;
        com[3].v         = in[5 * 6 + i].v - (DATA_TYPE)4.0f * in[3 * 6 + i].v;
        com[4].v         = in[3 * 6 + i].v - in[1 * 6 + i].v;
        com[4].v         = com[4].v + com[4].v;
        com[5].v         = in[4 * 6 + i].v - in[2 * 6 + i].v;
        tmp[i + 0 * 6].v = com[2].v - com[0].v;
        tmp[i + 1 * 6].v = com[2].v + com[1].v;
        tmp[i + 2 * 6].v = com[2].v - com[1].v;
        tmp[i + 3 * 6].v = com[5].v + com[4].v;
        tmp[i + 4 * 6].v = com[5].v - com[4].v;
        tmp[i + 5 * 6].v = com[3].v - com[1].v;
    })

    TILE(DATA_TYPE, 36, 1, out);

    LOOP_UNROLLING(int, i, 0, 1, 6,
    {
        com[0].v         = tmp[i * 6 + 2].v - 4.f * tmp[i * 6 + 0].v;
        com[1].v         = tmp[i * 6 + 3].v - 4.f * tmp[i * 6 + 1].v;
        com[2].v         = tmp[i * 6 + 4].v - 4.f * tmp[i * 6 + 2].v;
        com[3].v         = tmp[i * 6 + 5].v - 4.f * tmp[i * 6 + 3].v;
        com[4].v         = tmp[i * 6 + 3].v - tmp[i * 6 + 1].v;
        com[4].v         = com[4].v + com[4].v;
        com[5].v         = tmp[i * 6 + 4].v - tmp[i * 6 + 2].v;
        out[i * 6 + 0].v = com[2].v - com[0].v;
        out[i * 6 + 1].v = com[2].v + com[1].v;
        out[i * 6 + 2].v = com[2].v - com[1].v;
        out[i * 6 + 3].v = com[5].v + com[4].v;
        out[i * 6 + 4].v = com[5].v - com[4].v;
        out[i * 6 + 5].v = com[3].v - com[1].v;
    })

    // Compute destination address
    TILE(uint, 36, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 36,
    {
        dst_indirect_y[i].v = mout + i * _INUM_TILES_X * _INUM_TILES_Y;
        dst_indirect_y[i].v += bout * _INUM_TILES_X * _INUM_TILES_Y * 36;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 36, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 5x5/5x1 or 1x5 and the output tile is 4x4/4x1 or 1x4 when the data layout is NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x4_5x5_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    const int cout = GET_SPATIAL_IDX(0, 1, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0); // NUM_TILES_X x NUM_TILES_Y
    const int bout = GET_SPATIAL_IDX(2, 1, 0); // BATCH SIZE IDX

    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _INUM_TILES_X NUM_TILES_X
#define _INUM_TILES_Y NUM_TILES_Y

    int x = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;
    x -= PAD_LEFT;
    y -= PAD_TOP;

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 8, 1, in);
    TILE(DATA_TYPE, 8, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v = 0;
    })

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 1, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 8, 1, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

    TILE(DATA_TYPE, 1, 8, com);

    com[0].s[0] = in[2].v - 4.25f * in[4].v + in[6].v;
    com[0].s[1] = in[1].v - 4.25f * in[3].v + in[5].v;
    com[0].s[2] = 0.5f * in[1].v - 2.5f * in[3].v + 2.0f * in[5].v;
    com[0].s[3] = 0.25f * in[2].v - 1.25f * in[4].v + in[6].v;
    com[0].s[4] = 4.0f * in[2].v - 5.0f * in[4].v + in[6].v;
    com[0].s[5] = 2.0f * in[1].v - 2.5f * in[3].v + 0.5f * in[5].v;
    out[0].s[0] = in[0].v - 5.25f * in[2].v + 5.25f * in[4].v - in[6].v;
    out[1].s[0] = com[0].s[0] + com[0].s[1];
    out[2].s[0] = com[0].s[0] - com[0].s[1];
    out[3].s[0] = com[0].s[3] + com[0].s[2];
    out[4].s[0] = com[0].s[3] - com[0].s[2];
    out[5].s[0] = com[0].s[4] + com[0].s[5];
    out[6].s[0] = com[0].s[4] - com[0].s[5];
    out[7].s[0] = -in[1].v + 5.25f * in[3].v - 5.25f * in[5].v + in[7].v;

    TILE(uint, 8, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        dst_indirect_y[i].v = mout + i * _INUM_TILES_X * _INUM_TILES_Y;
        dst_indirect_y[i].v += bout * _INUM_TILES_X * _INUM_TILES_Y * 8;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 8, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 64, 1, in);
    TILE(DATA_TYPE, 64, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        in[i].v = 0;
    })

    // Load the tile from a NHWC tensor
    T_LOAD_NHWC(DATA_TYPE, 8, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);

    TILE(DATA_TYPE, 8, 8, com);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[i] = in[2 * 8 + i].s[0] - (DATA_TYPE)4.25f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];                                    // x
        com[1].s[i] = in[1 * 8 + i].s[0] - (DATA_TYPE)4.25f * in[3 * 8 + i].s[0] + in[5 * 8 + i].s[0];                                    // x
        com[2].s[i] = (DATA_TYPE)0.25f * in[2 * 8 + i].s[0] - (DATA_TYPE)1.25f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];                 // x
        com[3].s[i] = (DATA_TYPE)0.5f * in[1 * 8 + i].s[0] - (DATA_TYPE)2.5f * in[3 * 8 + i].s[0] + (DATA_TYPE)2.0f * in[5 * 8 + i].s[0]; // x
        com[4].s[i] = (DATA_TYPE)4.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)5.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[5].s[i] = (DATA_TYPE)2.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)2.5f * in[3 * 8 + i].s[0] + (DATA_TYPE)0.5f * in[5 * 8 + i].s[0];
        com[6].s[i] = in[0 * 8 + i].s[0] - (DATA_TYPE)5.25f * in[2 * 8 + i].s[0] + (DATA_TYPE)5.25f * in[4 * 8 + i].s[0] - in[6 * 8 + i].s[0];
        com[7].s[i] = -in[1 * 8 + i].s[0] + (DATA_TYPE)5.25f * in[3 * 8 + i].s[0] - (DATA_TYPE)5.25f * in[5 * 8 + i].s[0] + in[7 * 8 + i].s[0];
    })

    TILE(DATA_TYPE, 8, 8, tmp);
    tmp[0].v = com[6].v;
    tmp[1].v = com[0].v + com[1].v;
    tmp[2].v = com[0].v - com[1].v;
    tmp[3].v = com[2].v + com[3].v;
    tmp[4].v = com[2].v - com[3].v;
    tmp[5].v = com[4].v + com[5].v;
    tmp[6].v = com[4].v - com[5].v;
    tmp[7].v = com[7].v;

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[0]         = tmp[i].s[2] - 4.25f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[1]         = tmp[i].s[1] - 4.25f * tmp[i].s[3] + tmp[i].s[5];
        com[0].s[2]         = 0.5f * tmp[i].s[1] - 2.5f * tmp[i].s[3] + 2.0f * tmp[i].s[5];
        com[0].s[3]         = 0.25f * tmp[i].s[2] - 1.25f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[4]         = 4.0f * tmp[i].s[2] - 5.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[5]         = 2.0f * tmp[i].s[1] - 2.5f * tmp[i].s[3] + 0.5f * tmp[i].s[5];
        out[i * 8 + 0].s[0] = tmp[i].s[0] - 5.25f * tmp[i].s[2] + 5.25f * tmp[i].s[4] - tmp[i].s[6];
        out[i * 8 + 1].s[0] = com[0].s[0] + com[0].s[1];
        out[i * 8 + 2].s[0] = com[0].s[0] - com[0].s[1];
        out[i * 8 + 3].s[0] = com[0].s[3] + com[0].s[2];
        out[i * 8 + 4].s[0] = com[0].s[3] - com[0].s[2];
        out[i * 8 + 5].s[0] = com[0].s[4] + com[0].s[5];
        out[i * 8 + 6].s[0] = com[0].s[4] - com[0].s[5];
        out[i * 8 + 7].s[0] = -tmp[i].s[1] + 5.25f * tmp[i].s[3] - 5.25f * tmp[i].s[5] + tmp[i].s[7];
    })

    TILE(uint, 64, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        dst_indirect_y[i].v = mout + i * _INUM_TILES_X * _INUM_TILES_Y;
        dst_indirect_y[i].v += bout * _INUM_TILES_X * _INUM_TILES_Y * 64;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 64, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 7x7/7x1/1x7 and the output tile is 2x2/7x1/1x7 when the data layout is NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_2x2_7x7_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    const int cout = GET_SPATIAL_IDX(0, 1, 0); // OFM
    const int mout = GET_SPATIAL_IDX(1, 1, 0); // NUM_TILES_X x NUM_TILES_Y
    const int bout = GET_SPATIAL_IDX(2, 1, 0); // BATCH SIZE IDX

    // All the tensor dimensions are passed at compile time.
    // In case of dynamic tensor support, the following dimensions should be passed as function argument.
#define _ISRC_WIDTH SRC_WIDTH
#define _ISRC_HEIGHT SRC_HEIGHT
#define _INUM_TILES_X NUM_TILES_X
#define _INUM_TILES_Y NUM_TILES_Y

    int x = (mout % _INUM_TILES_X) * OUTPUT_TILE_W;
    int y = (mout / _INUM_TILES_X) * OUTPUT_TILE_H;
    x -= PAD_LEFT;
    y -= PAD_TOP;

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 8, 1, in);
    TILE(DATA_TYPE, 8, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v = 0;
    })

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 1, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#else  // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    T_LOAD_NHWC(DATA_TYPE, 8, 1, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        in[i].v *= (DATA_TYPE) - 36.0f;
    })

    TILE(DATA_TYPE, 1, 8, com) = { { { 0 } } };

    com[0].s[0] = 36.0f * in[2].v - 13.0f * in[4].v + in[6].v;
    com[0].s[1] = 36.0f * in[1].v - 13.0f * in[3].v + 1.0f * in[5].v;
    com[0].s[2] = 9.0f * in[2].v - 10.0f * in[4].v + in[6].v;
    com[0].s[3] = 18.0f * in[1].v - 20.0f * in[3].v + 2.0f * in[5].v;
    com[0].s[4] = 4.0f * in[2].v - 5.0f * in[4].v + in[6].v;
    com[0].s[5] = 12.0f * in[1].v - 15.0f * in[3].v + 3.0f * in[5].v;
    out[0].s[0] = -36.0f * in[0].v + 49.0f * in[2].v + -14.0f * in[4].v + in[6].v;
    out[1].s[0] = com[0].s[0] - com[0].s[1];
    out[2].s[0] = com[0].s[0] + com[0].s[1];
    out[3].s[0] = com[0].s[2] - com[0].s[3];
    out[4].s[0] = com[0].s[2] + com[0].s[3];
    out[5].s[0] = com[0].s[4] - com[0].s[5];
    out[6].s[0] = com[0].s[4] + com[0].s[5];
    out[7].s[0] = -36.0f * in[1].v + 0.0f * in[2].v + 49.0f * in[3].v - 14.0f * in[5].v + in[7].v;

    TILE(uint, 8, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        dst_indirect_y[i].v = mout + i * _INUM_TILES_X * _INUM_TILES_Y;
        dst_indirect_y[i].v += bout * _INUM_TILES_X * _INUM_TILES_Y * 8;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 8, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#else // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    TILE(DATA_TYPE, 64, 1, in);
    TILE(DATA_TYPE, 64, 1, out);

    // Initialize the input tile
    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        in[i].v = 0;
    })

    // Load the tile from a NHWC tensor
    T_LOAD_NHWC(DATA_TYPE, 8, 8, 1, BUFFER, src, bout, y, x, cout, _ISRC_WIDTH, _ISRC_HEIGHT, src_stride_y, in);

    TILE(DATA_TYPE, 8, 8, com);

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[i] = (DATA_TYPE)36.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)13.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[1].s[i] = (DATA_TYPE)36.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)13.0f * in[3 * 8 + i].s[0] + in[5 * 8 + i].s[0];
        com[2].s[i] = (DATA_TYPE)9.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)10.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[3].s[i] = (DATA_TYPE)18.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)20.0f * in[3 * 8 + i].s[0] + (DATA_TYPE)2.0f * in[5 * 8 + i].s[0];
        com[4].s[i] = (DATA_TYPE)4.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)5.0f * in[4 * 8 + i].s[0] + in[6 * 8 + i].s[0];
        com[5].s[i] = (DATA_TYPE)12.0f * in[1 * 8 + i].s[0] - (DATA_TYPE)15.0f * in[3 * 8 + i].s[0] + (DATA_TYPE)3.0f * in[5 * 8 + i].s[0];
        com[6].s[i] = (DATA_TYPE)49.0f * in[2 * 8 + i].s[0] - (DATA_TYPE)36.0f * in[0 * 8 + i].s[0] + in[6 * 8 + i].s[0] - (DATA_TYPE)14.0f * in[4 * 8 + i].s[0];
        com[7].s[i] = (DATA_TYPE)49.0f * in[3 * 8 + i].s[0] - (DATA_TYPE)36.0f * in[1 * 8 + i].s[0] + in[7 * 8 + i].s[0] - (DATA_TYPE)14.0f * in[5 * 8 + i].s[0];
    })

    TILE(DATA_TYPE, 8, 8, tmp);
    tmp[0].v = com[6].v;
    tmp[1].v = com[0].v - com[1].v;
    tmp[2].v = com[0].v + com[1].v;
    tmp[3].v = com[2].v - com[3].v;
    tmp[4].v = com[2].v + com[3].v;
    tmp[5].v = com[4].v - com[5].v;
    tmp[6].v = com[4].v + com[5].v;
    tmp[7].v = com[7].v;

    LOOP_UNROLLING(int, i, 0, 1, 8,
    {
        com[0].s[0]         = 36.0f * tmp[i].s[2] - 13.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[1]         = 36.0f * tmp[i].s[1] - 13.0f * tmp[i].s[3] + 1.0f * tmp[i].s[5];
        com[0].s[2]         = 9.0f * tmp[i].s[2] - 10.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[3]         = 18.0f * tmp[i].s[1] - 20.0f * tmp[i].s[3] + 2.0f * tmp[i].s[5];
        com[0].s[4]         = 4.0f * tmp[i].s[2] - 5.0f * tmp[i].s[4] + tmp[i].s[6];
        com[0].s[5]         = 12.0f * tmp[i].s[1] - 15.0f * tmp[i].s[3] + 3.0f * tmp[i].s[5];
        out[i * 8 + 0].s[0] = -36.0f * tmp[i].s[0] + 49.0f * tmp[i].s[2] + -14.0f * tmp[i].s[4] + tmp[i].s[6];
        out[i * 8 + 1].s[0] = com[0].s[0] - com[0].s[1];
        out[i * 8 + 2].s[0] = com[0].s[0] + com[0].s[1];
        out[i * 8 + 3].s[0] = com[0].s[2] - com[0].s[3];
        out[i * 8 + 4].s[0] = com[0].s[2] + com[0].s[3];
        out[i * 8 + 5].s[0] = com[0].s[4] - com[0].s[5];
        out[i * 8 + 6].s[0] = com[0].s[4] + com[0].s[5];
        out[i * 8 + 7].s[0] = -36.0f * tmp[i].s[1] + 0.0f * tmp[i].s[2] + 49.0f * tmp[i].s[3] - 14.0f * tmp[i].s[5] + tmp[i].s[7];
    })

    TILE(uint, 64, 1, dst_indirect_y);

    LOOP_UNROLLING(int, i, 0, 1, 64,
    {
        dst_indirect_y[i].v = mout + i * _INUM_TILES_X * _INUM_TILES_Y;
        dst_indirect_y[i].v += bout * _INUM_TILES_X * _INUM_TILES_Y * 64;
    })

    T_STORE_INDIRECT_WIDTH_SELECT(DATA_TYPE, 64, 1, 0, BUFFER, dst, cout, dst_stride_y, false, out, dst_indirect_y);

#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) || defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 4x1 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x1_3x1_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_3x3_stepz1_nhwc(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_stride_w,
                                                 src_step_w,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_stride_w,
                                                 dst_step_w,
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 5x1 and the output tile is 4x1 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_4x1_5x1_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_5x5_stepz1_nhwc(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_stride_w,
                                                 src_step_w,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_stride_w,
                                                 dst_step_w,
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 7x1 and the output tile is 2x1 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_2x1_7x1_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_2x2_7x7_stepz1_nhwc(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_stride_w,
                                                 src_step_w,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_stride_w,
                                                 dst_step_w,
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x4 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_1x4_1x3_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_3x3_stepz1_nhwc(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_stride_w,
                                                 src_step_w,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_stride_w,
                                                 dst_step_w,
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 1x5 and the output tile is 1x4 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_1x4_1x5_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_4x4_5x5_stepz1_nhwc(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_stride_w,
                                                 src_step_w,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_stride_w,
                                                 dst_step_w,
                                                 dst_offset_first_element_in_bytes);
}

//! @cond Doxygen_Suppress
/** This OpenCL kernel computes the input transform when the kernel size is 1x7 and the output tile is 1x2 for data layout NHWC
 *
 * @note Data layout supported: NHWC
 * @note Data type supported: F32/F16
 * @note The data type must be passed at compile time using -DDATA_TYPE (e.g. -DDATA_TYPE=half)
 * @note The number of tiles in the X and Y axes must be passed at compile time using -DNUM_TILES_X and -DNUM_TILES_Y (i.e.-DNUM_TILES_X=5, -DNUM_TILES_Y=3).
 * @note The convolution padding (left and top) must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (e.g. -DPAD_LEFT=2, -DPAD_TOP=2)
 * @note The spatial dimensions of the source tensor must be passed at compile time using -DSRC_WIDTH and -DSRC_HEIGHT (e.g. -DSRC_WIDTH=96, -DSRC_HEIGHT=64)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 * @param[in] dst_step_w                        dst_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
//! @endcond
__kernel void winograd_input_transform_1x2_1x7_stepz1_nhwc(
    TENSOR4D(src, BUFFER),
    TENSOR4D(dst, BUFFER))
{
    winograd_input_transform_2x2_7x7_stepz1_nhwc(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_stride_w,
                                                 src_step_w,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_stride_w,
                                                 dst_step_w,
                                                 dst_offset_first_element_in_bytes);
}
#endif // defined(NHWC) && defined(SRC_WIDTH) && defined(SRC_HEIGHT) && defined(NUM_TILES_X) && defined(NUM_TILES_Y)

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 2x1
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_2x1_3x1_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_2x2_3x3_stepz1_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}

/** This OpenCL kernel computes the input transform when the kernel size is 3x1, the output tile is 2x1 and the number of channels is multiple of 2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_2x1_3x1_stepz2_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_2x2_3x3_stepz2_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}

/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 4x1
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_4x1_3x1_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_4x4_3x3_stepz1_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}

/** This OpenCL kernel computes the input transform when the kernel size is 5x1 and the output tile is 4x1 when the data layout is NCHW
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_4x1_5x1_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_4x4_5x5_stepz1_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

#if defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_1x2_1x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_2x2_3x3_stepz1_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x3, the output tile is 1x2 and the number of channels is multiple of 2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_1x2_1x3_stepz2_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_2x2_3x3_stepz2_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x4
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_1x4_1x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_4x4_3x3_stepz1_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x5 and the output tile is 1x4
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 * @note The data type must be passed at compile time using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types: float/half.
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32/F16
 * @param[in] src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in] src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in] src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in] src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in] src_step_z                        src_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_ptr                           Pointer to the destination tensor. Supported data types: as @p src_ptr
 * @param[in] dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in] dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in] dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in] dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in] src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in] dst_stride_w                      Stride of the destination tensor in W dimension (in bytes)
 */
__kernel void winograd_input_transform_1x4_1x5_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    uint src_stride_w,
    uint dst_stride_w)
{
    winograd_input_transform_4x4_5x5_stepz1_nchw(src_ptr,
                                                 src_stride_x,
                                                 src_step_x,
                                                 src_stride_y,
                                                 src_step_y,
                                                 src_stride_z,
                                                 src_step_z,
                                                 src_offset_first_element_in_bytes,
                                                 dst_ptr,
                                                 dst_stride_x,
                                                 dst_step_x,
                                                 dst_stride_y,
                                                 dst_step_y,
                                                 dst_stride_z,
                                                 dst_step_z,
                                                 dst_offset_first_element_in_bytes,
                                                 src_stride_w,
                                                 dst_stride_w);
}
#endif // defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
#endif // defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)
