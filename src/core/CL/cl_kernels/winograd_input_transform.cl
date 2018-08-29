/*
 * Copyright (c) 2018 ARM Limited.
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

#if defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)
/** This OpenCL kernel computes the input transform when the kernel size is 3x3/3x1 or 1x3 and the output tile is 2x2/2x1 or 1x2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_2x2_3x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(float) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * sizeof(float)) - ((int)PAD_TOP * src_stride_y);

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    float4 in_row0 = vload4(0, (__global float *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    float4 in_row0 = (float4)(*((__global float *)(src_addr + 0 * src_stride_y)),
                              *((__global float *)(src_addr + 1 * src_stride_y)),
                              *((__global float *)(src_addr + 2 * src_stride_y)),
                              *((__global float *)(src_addr + 3 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    float4       in_row0 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float4       in_row1 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float4       in_row2 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float4       in_row3 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    float4 tmp0 = in_row0;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    tmp0 -= in_row2;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    float out00 = tmp0.s0 - tmp0.s2;
    float out01 = tmp0.s1 + tmp0.s2;
    float out02 = tmp0.s2 - tmp0.s1;
    float out03 = tmp0.s1 - tmp0.s3;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    float4 tmp1 = in_row1 + in_row2;
    float4 tmp2 = in_row2 - in_row1;
    float4 tmp3 = in_row1 - in_row3;

    float out10 = tmp1.s0 - tmp1.s2;
    float out11 = tmp1.s1 + tmp1.s2;
    float out12 = tmp1.s2 - tmp1.s1;
    float out13 = tmp1.s1 - tmp1.s3;

    float out20 = tmp2.s0 - tmp2.s2;
    float out21 = tmp2.s1 + tmp2.s2;
    float out22 = tmp2.s2 - tmp2.s1;
    float out23 = tmp2.s1 - tmp2.s3;

    float out30 = tmp3.s0 - tmp3.s2;
    float out31 = tmp3.s1 + tmp3.s2;
    float out32 = tmp3.s2 - tmp3.s1;
    float out33 = tmp3.s1 - tmp3.s3;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(float) + (x + y * (int)NUM_TILES_X) * dst_stride_y;

    *((__global float *)(dst_addr + 0 * dst_stride_z)) = out00; // in_row0.s0; out00;
    *((__global float *)(dst_addr + 1 * dst_stride_z)) = out01; // in_row0.s1; out01;
    *((__global float *)(dst_addr + 2 * dst_stride_z)) = out02; // in_row0.s2; out02;
    *((__global float *)(dst_addr + 3 * dst_stride_z)) = out03; // in_row0.s3; out03;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    *((__global float *)(dst_addr + 4 * dst_stride_z))  = out10;
    *((__global float *)(dst_addr + 5 * dst_stride_z))  = out11;
    *((__global float *)(dst_addr + 6 * dst_stride_z))  = out12;
    *((__global float *)(dst_addr + 7 * dst_stride_z))  = out13;
    *((__global float *)(dst_addr + 8 * dst_stride_z))  = out20;
    *((__global float *)(dst_addr + 9 * dst_stride_z))  = out21;
    *((__global float *)(dst_addr + 10 * dst_stride_z)) = out22;
    *((__global float *)(dst_addr + 11 * dst_stride_z)) = out23;
    *((__global float *)(dst_addr + 12 * dst_stride_z)) = out30;
    *((__global float *)(dst_addr + 13 * dst_stride_z)) = out31;
    *((__global float *)(dst_addr + 14 * dst_stride_z)) = out32;
    *((__global float *)(dst_addr + 15 * dst_stride_z)) = out33;
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
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_2x2_3x3_stepz2_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2) * 2;

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(float) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * sizeof(float)) - ((int)PAD_TOP * src_stride_y);

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    float4 in_row0 = vload4(0, (__global float *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    float4 in_row0 = (float4)(*((__global float *)(src_addr + 0 * src_stride_y)),
                              *((__global float *)(src_addr + 1 * src_stride_y)),
                              *((__global float *)(src_addr + 2 * src_stride_y)),
                              *((__global float *)(src_addr + 3 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    float4       in_row0 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float4       in_row1 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float4       in_row2 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float4       in_row3 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    src_addr += src_stride_z;
#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    float4 in_row4 = vload4(0, (__global float *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    float4 in_row4 = (float4)(*((__global float *)(src_addr + 0 * src_stride_y)),
                              *((__global float *)(src_addr + 1 * src_stride_y)),
                              *((__global float *)(src_addr + 2 * src_stride_y)),
                              *((__global float *)(src_addr + 3 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    float4       in_row4 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float4       in_row5 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float4       in_row6 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float4       in_row7 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    float4 tmp0 = in_row0;
    float4 tmp4 = in_row4;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    tmp0 -= in_row2;
    tmp4 -= in_row6;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    float2 out00 = (float2)(tmp0.s0 - tmp0.s2, tmp4.s0 - tmp4.s2);
    float2 out01 = (float2)(tmp0.s1 + tmp0.s2, tmp4.s1 + tmp4.s2);
    float2 out02 = (float2)(tmp0.s2 - tmp0.s1, tmp4.s2 - tmp4.s1);
    float2 out03 = (float2)(tmp0.s1 - tmp0.s3, tmp4.s1 - tmp4.s3);

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    float4 tmp1 = in_row1 + in_row2;
    float4 tmp2 = in_row2 - in_row1;
    float4 tmp3 = in_row1 - in_row3;

    float4 tmp5 = in_row5 + in_row6;
    float4 tmp6 = in_row6 - in_row5;
    float4 tmp7 = in_row5 - in_row7;

    float2 out10 = (float2)(tmp1.s0 - tmp1.s2, tmp5.s0 - tmp5.s2);
    float2 out11 = (float2)(tmp1.s1 + tmp1.s2, tmp5.s1 + tmp5.s2);
    float2 out12 = (float2)(tmp1.s2 - tmp1.s1, tmp5.s2 - tmp5.s1);
    float2 out13 = (float2)(tmp1.s1 - tmp1.s3, tmp5.s1 - tmp5.s3);

    float2 out20 = (float2)(tmp2.s0 - tmp2.s2, tmp6.s0 - tmp6.s2);
    float2 out21 = (float2)(tmp2.s1 + tmp2.s2, tmp6.s1 + tmp6.s2);
    float2 out22 = (float2)(tmp2.s2 - tmp2.s1, tmp6.s2 - tmp6.s1);
    float2 out23 = (float2)(tmp2.s1 - tmp2.s3, tmp6.s1 - tmp6.s3);

    float2 out30 = (float2)(tmp3.s0 - tmp3.s2, tmp7.s0 - tmp7.s2);
    float2 out31 = (float2)(tmp3.s1 + tmp3.s2, tmp7.s1 + tmp7.s2);
    float2 out32 = (float2)(tmp3.s2 - tmp3.s1, tmp7.s2 - tmp7.s1);
    float2 out33 = (float2)(tmp3.s1 - tmp3.s3, tmp7.s1 - tmp7.s3);
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(float) + (x + y * (int)NUM_TILES_X) * dst_stride_y;

    vstore2(out00, 0, (__global float *)(dst_addr + 0 * dst_stride_z));
    vstore2(out01, 0, (__global float *)(dst_addr + 1 * dst_stride_z));
    vstore2(out02, 0, (__global float *)(dst_addr + 2 * dst_stride_z));
    vstore2(out03, 0, (__global float *)(dst_addr + 3 * dst_stride_z));

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    vstore2(out10, 0, (__global float *)(dst_addr + 4 * dst_stride_z));
    vstore2(out11, 0, (__global float *)(dst_addr + 5 * dst_stride_z));
    vstore2(out12, 0, (__global float *)(dst_addr + 6 * dst_stride_z));
    vstore2(out13, 0, (__global float *)(dst_addr + 7 * dst_stride_z));
    vstore2(out20, 0, (__global float *)(dst_addr + 8 * dst_stride_z));
    vstore2(out21, 0, (__global float *)(dst_addr + 9 * dst_stride_z));
    vstore2(out22, 0, (__global float *)(dst_addr + 10 * dst_stride_z));
    vstore2(out23, 0, (__global float *)(dst_addr + 11 * dst_stride_z));
    vstore2(out30, 0, (__global float *)(dst_addr + 12 * dst_stride_z));
    vstore2(out31, 0, (__global float *)(dst_addr + 13 * dst_stride_z));
    vstore2(out32, 0, (__global float *)(dst_addr + 14 * dst_stride_z));
    vstore2(out33, 0, (__global float *)(dst_addr + 15 * dst_stride_z));
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
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x4_3x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(float) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * sizeof(float)) - ((int)PAD_TOP * src_stride_y);

#if defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row0
    float4 d00 = (float4)(*((__global float *)(src_addr + 0 * src_stride_y)),
                          *((__global float *)(src_addr + 1 * src_stride_y)),
                          *((__global float *)(src_addr + 2 * src_stride_y)),
                          *((__global float *)(src_addr + 3 * src_stride_y)));
    float2 d01 = (float2)(*((__global float *)(src_addr + 4 * src_stride_y)),
                          *((__global float *)(src_addr + 5 * src_stride_y)));
#else  // defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row0
    float4       d00     = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float2       d01     = vload2(2, (__global float *)(src_addr + 0 * src_stride_y));
#endif // defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;
    float out4 = 0.0f;
    float out5 = 0.0f;

    // Channels [0, 5]: [out00, out01, out02, out03, out04, out05]
    out0 += 16.0f * d00.s0 - 20.0f * d00.s2 + 4.0f * d01.s0;
    out1 += -16.0f * d00.s1 - 16.0f * d00.s2 + 4.0f * d00.s3 + 4.0f * d01.s0;
    out2 += 16.0f * d00.s1 - 16.0f * d00.s2 - 4.0f * d00.s3 + 4.0f * d01.s0;
    out3 += -8.0f * d00.s1 - 4.0f * d00.s2 + 8.0f * d00.s3 + 4.0f * d01.s0;
    out4 += 8.0f * d00.s1 - 4.0f * d00.s2 - 8.0f * d00.s3 + 4.0f * d01.s0;
    out5 += 16.0f * d00.s1 - 20.0f * d00.s3 + 4.0f * d01.s1;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row4
    float4 d40 = vload4(0, (__global float *)(src_addr + 4 * src_stride_y));
    float2 d41 = vload2(2, (__global float *)(src_addr + 4 * src_stride_y));

    // k0, k1, k2, k3, k4, k5 are common terms for row0, row1, row2, row3 and row4
    float k0 = d41.s0;
    float k1 = d41.s0;
    float k2 = d41.s0;
    float k3 = d41.s0;
    float k4 = d41.s0;
    float k5 = 0.0f;

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
    float4 d20 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float2 d21 = vload2(2, (__global float *)(src_addr + 2 * src_stride_y));

    out0 += -20.0f * d20.s0 + 25.0f * d20.s2 - 5.0f * d21.s0;
    out1 += +20.0f * d20.s1 + 20.0f * d20.s2 - 5.0f * d20.s3 - 5.0f * d21.s0;
    out2 += -20.0f * d20.s1 + 20.0f * d20.s2 + 5.0f * d20.s3 - 5.0f * d21.s0;
    out3 += +10.0f * d20.s1 + 5.0f * d20.s2 - 10.0f * d20.s3 - 5.0f * d21.s0;
    out4 += -10.0f * d20.s1 + 5.0f * d20.s2 + 10.0f * d20.s3 - 5.0f * d21.s0;
    out5 += -20.0f * d20.s1 + 25.0f * d20.s3 - 5.0f * d21.s1;
#endif // #if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Compute destination address
    __global float *dst_addr = (__global float *)(dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(float) + (x + y * (int)NUM_TILES_X) * dst_stride_y);

    uint dst_plane_stride = dst_stride_z / sizeof(float);

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
    float out6  = k0;
    float out7  = k1;
    float out8  = k2;
    float out9  = k3;
    float out10 = k4;
    float out11 = k5;
    float out12 = k0;
    float out13 = k1;
    float out14 = k2;
    float out15 = k3;
    float out16 = k4;
    float out17 = k5;
    float out18 = k0;
    float out19 = k1;
    float out20 = k2;
    float out21 = k3;
    float out22 = k4;
    float out23 = k5;
    float out24 = k0;
    float out25 = k1;
    float out26 = k2;
    float out27 = k3;
    float out28 = k4;
    float out29 = k5;

    // Row1
    float4 d10 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float2 d11 = vload2(2, (__global float *)(src_addr + 1 * src_stride_y));

    // Row3
    float4 d30 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));
    float2 d31 = vload2(2, (__global float *)(src_addr + 3 * src_stride_y));

    // Compute common parts for the channels between [6, 29]
    // Channels [6, 11]:  [out10, out11, out12, out13, out14, out15]
    // Channels [12, 17]: [out20, out21, out22, out23, out24, out25]
    float part0  = -16.0f * d20.s0 + 20.0f * d20.s2 - 4.0f * d21.s0;
    float part1  = 16.0f * d10.s0 - 20.0f * d10.s2 + 4.0f * d11.s0 - 4.0f * d30.s0 + 5.0f * d30.s2 - d31.s0;
    float part2  = 16.0f * d20.s2 - 4.0f * d21.s0;
    float part3  = 16.0f * d20.s1 - 4.0f * d20.s3;
    float part4  = 16.0f * d10.s2 - 4.0f * d11.s0 - 4.0f * d30.s2 + d31.s0;
    float part5  = 16.0f * d10.s1 - 4.0f * d10.s3 - 4.0f * d30.s1 + d30.s3;
    float part6  = 4.0f * d20.s2 - 4.0f * d21.s0;
    float part7  = 8.0f * d10.s1 - 8.0f * d10.s3 - 2.0f * d30.s1 + 2.0f * d30.s3;
    float part8  = 4.0f * d10.s2 - 4.0f * d11.s0 - d30.s2 + d31.s0;
    float part9  = 8.0f * d20.s1 - 8.0f * d20.s3;
    float part10 = -16.0f * d20.s1 + 20.0f * d20.s3 - 4.0f * d21.s1;
    float part11 = -16.0f * d10.s1 + 20.0f * d10.s3 - 4.0f * d11.s1 + 4.0f * d30.s1 - 5.0f * d30.s3 + d31.s1;

    // Channels [18, 23]: [out30, out31, out32, out33, out34, out35]
    // Channels [24, 29]: [out40, out41, out42, out43, out44, out45]
    float part12 = 8.0f * d10.s0 - 10.0f * d10.s2 + 2.0f * d11.s0 - 8.0f * d30.s0 + 10.0f * d30.s2 - 2.0f * d31.s0;
    float part13 = part0 * 0.25f; // -4.0f * d20.s0 + 5.0f * d20.s2 - d21.s0
    float part14 = part2 * 0.25f; // 4.0f * d20.s2 - d21.s0
    float part15 = 8.0f * d10.s1 - 2.0f * d10.s3 - 8.0f * d30.s1 + 2.0f * d30.s3;
    float part16 = 8.0f * d10.s2 - 2.0f * d11.s0 - 8.0f * d30.s2 + 2.0f * d31.s0;
    float part17 = part3 * 0.25f; // 4.0f * d20.s1 - d20.s3
    float part18 = part6 * 0.25f; // d20.s2 - d21.s0
    float part19 = 4.0f * d10.s1 - 4.0f * d10.s3 - 4.0f * d30.s1 + 4.0f * d30.s3;
    float part20 = 2.0f * d10.s2 - 2.0f * d11.s0 - 2.0f * d30.s2 + 2.0f * d31.s0;
    float part21 = part9 * 0.25f;                                                 // 2.0f * (d20.s1 - d20.s3)
    float part22 = part10 * 0.25f;                                                // - 4.0f * d20.s1 + 5.0f * d20.s3 - d21.s1
    float part23 = part11 * 0.5f + 6.0f * d30.s1 - 7.5f * d30.s3 + 1.5f * d31.s1; // - 8.0f * d10.s1 + 10.0f * d10.s3 - 2.0f * d11.s1 + 8.0f * d30.s1 - 10.0f * d30.s3 + 2.0f * d31.s1;

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
    float4 d50 = vload4(0, (__global float *)(src_addr + 5 * src_stride_y));
    float2 d51 = vload2(2, (__global float *)(src_addr + 5 * src_stride_y));

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

#if defined(SRC_DIM_1) && defined(SRC_DIM_2)
/** This OpenCL kernel computes the input transform when the output tile is 4x4, 4x1 or 1x4, the filter size 3x3, 3x1 or 1x3 and the data layout is NHWC
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note Dimension one of the input tensor (width for NHWC data layout) must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM_1=112)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 3x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x3, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x4_3x3_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(float);

    // Clamp coordinates. This clamp is valid for all rows
    int4 y_coord0 = (int4)(y * OUTPUT_TILE_W) + (int4)(0, 1, 2, 3) - (int4)PAD_LEFT;
    int2 y_coord1 = (int2)(y * OUTPUT_TILE_W) + (int2)(4, 5) - (int2)PAD_LEFT;
    y_coord0      = clamp(y_coord0, (int4) - 1, (int4)SRC_DIM_1);
    y_coord1      = clamp(y_coord1, (int2) - 1, (int2)SRC_DIM_1);

    int  z_coord;
    int4 valid_y0;
    int2 valid_y1;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row4
    z_coord = (z * 4) - (int)PAD_TOP + 4;

    // If z < 0, set y to -1
    valid_y0 = select(y_coord0, (int4) - 1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, (int2) - 1, (int2)z_coord < 0);
    // If z >= SRC_DIM_2, set y to SRC_DIM_2
    valid_y0 = select(valid_y0, (int4)SRC_DIM_1, (int4)z_coord >= (int)SRC_DIM_2);
    valid_y1 = select(valid_y1, (int2)SRC_DIM_1, (int2)z_coord >= (int)SRC_DIM_2);

    // Clamp z coordinate
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    float d40 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d41 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d42 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d43 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d44 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d45 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);

    float k0 = d44;
    float k1 = d44;
    float k2 = d44;
    float k3 = d44;
    float k4 = d44;
    float k5 = (float)0.0f;

    k0 += 4.0f * d40 - 5.0f * d42;
    k1 += -4.0f * d41 - 4.0f * d42 + d43;
    k2 += 4.0f * d41 - 4.0f * d42 - d43;
    k3 += -2.0f * d41 + 2.0f * d43 - d42;
    k4 += 2.0f * d41 - 2.0f * d43 - d42;
    k5 += 4.0f * d41 - 5.0f * d43 + d45;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

#if !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row0
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 0;

#if PAD_TOP != 0
    valid_y0 = select(y_coord0, (int4) - 1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, (int2) - 1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, (int)SRC_DIM_1, (int4)z_coord >= (int)SRC_DIM_2);
    valid_y1 = select(valid_y1, (int)SRC_DIM_1, (int2)z_coord >= (int)SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);
#else  // PAD_TOP != 0
    valid_y0 = y_coord0;
    valid_y1 = y_coord1;
#endif // if PAD_TOP == 0, we cannot read out of bound

    float d00 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d01 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d02 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d03 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d04 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d05 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);
#else  // !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    int4 z_coords0 = (int4)(z * OUTPUT_TILE_H) + (int4)(0, 1, 2, 3) - (int4)PAD_TOP;
    int2 z_coords1 = (int2)(z * OUTPUT_TILE_H) + (int2)(4, 5) - (int2)PAD_TOP;

    valid_y0 = select((int4)y_coord0.s0, (int4) - 1, z_coords0 < (int4)0);
    valid_y1 = select((int2)y_coord0.s0, (int2) - 1, z_coords1 < (int2)0);
    valid_y0 = select(valid_y0, (int4)SRC_DIM_1, z_coords0 >= (int4)SRC_DIM_2);
    valid_y1 = select(valid_y1, (int2)SRC_DIM_1, z_coords1 >= (int2)SRC_DIM_2);

    z_coords0 = clamp((int4)z_coords0, (int4)0, (int4)((int)SRC_DIM_2 - 1));
    z_coords1 = clamp((int2)z_coords1, (int2)0, (int2)((int)SRC_DIM_2 - 1));

    float d00 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coords0.s0 * src_stride_z);
    float d01 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coords0.s1 * src_stride_z);
    float d02 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coords0.s2 * src_stride_z);
    float d03 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coords0.s3 * src_stride_z);
    float d04 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coords1.s0 * src_stride_z);
    float d05 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coords1.s1 * src_stride_z);
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    float out0 = 16.0f * d00 - 20.0f * d02 + 4.0f * d04;
    float out1 = -16.0f * d01 - 16.0f * d02 + 4.0f * d03 + 4.0f * d04;
    float out2 = 16.0f * d01 - 16.0f * d02 - 4.0f * d03 + 4.0f * d04;
    float out3 = -8.0f * d01 - 4.0f * d02 + 8.0f * d03 + 4.0f * d04;
    float out4 = 8.0f * d01 - 4.0f * d02 - 8.0f * d03 + 4.0f * d04;
    float out5 = 16.0f * d01 - 20.0f * d03 + 4.0f * d05;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row2
    z_coord  = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 2;
    valid_y0 = select(y_coord0, (int4) - 1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, (int2) - 1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, (int4)SRC_DIM_1, (int4)z_coord >= (int)SRC_DIM_2);
    valid_y1 = select(valid_y1, (int2)SRC_DIM_1, (int2)z_coord >= (int)SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    float d20 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d21 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d22 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d23 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d24 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d25 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);

    out0 += k0;
    out1 += k1;
    out2 += k2;
    out3 += k3;
    out4 += k4;
    out5 += k5;
    float out6  = k0;
    float out7  = k1;
    float out8  = k2;
    float out9  = k3;
    float out10 = k4;
    float out11 = k5;
    float out12 = k0;
    float out13 = k1;
    float out14 = k2;
    float out15 = k3;
    float out16 = k4;
    float out17 = k5;
    float out18 = k0;
    float out19 = k1;
    float out20 = k2;
    float out21 = k3;
    float out22 = k4;
    float out23 = k5;
    float out24 = k0;
    float out25 = k1;
    float out26 = k2;
    float out27 = k3;
    float out28 = k4;
    float out29 = k5;

    // Channels [0, 5]: [out00, out01, out02, out03, out04, out05]
    out0 += -20.0f * d20 + 25.0f * d22 - 5.0f * d24;
    out1 += 20.0f * d21 + 20.0f * d22 - 5.0f * d23 - 5.0f * d24;
    out2 += -20.0f * d21 + 20.0f * d22 + 5.0f * d23 - 5.0f * d24;
    out3 += 10.0f * d21 + 5.0f * d22 - 10.0f * d23 - 5.0f * d24;
    out4 += -10.0f * d21 + 5.0f * d22 + 10.0f * d23 - 5.0f * d24;
    out5 += -20.0f * d21 + 25.0f * d23 - 5.0f * d25;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Compute destination address
    __global float *dst_addr         = (__global float *)(dst_ptr + dst_offset_first_element_in_bytes + x * sizeof(float) + (y + z * (int)NUM_TILES_X) * dst_stride_y);
    uint            dst_plane_stride = dst_stride_z / sizeof(float);

    *((__global float *)dst_addr) = out0;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out1;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out2;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out3;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out4;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out5;
    dst_addr += dst_plane_stride;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    // Row1
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 1;
    // Row1 can never be out of bounds
    valid_y0 = y_coord0;
    valid_y1 = y_coord1;

    float d10 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d11 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d12 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d13 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d14 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d15 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);

    // Row3
    z_coord  = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 3;
    valid_y0 = select(y_coord0, (int4) - 1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, (int2) - 1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, (int4)SRC_DIM_1, (int4)z_coord >= (int)SRC_DIM_2);
    valid_y1 = select(valid_y1, (int2)SRC_DIM_1, (int2)z_coord >= (int)SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);
    z_coord  = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    float d30 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d31 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d32 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d33 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d34 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d35 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);

    // Compute common parts for the channels between [6, 29]
    // Channels [6, 11]:  [out10, out11, out12, out13, out14, out15]
    // Channels [12, 17]: [out20, out21, out22, out23, out24, out25]
    float part0  = -16.0f * d20 + 20.0f * d22 - 4.0f * d24;
    float part1  = 16.0f * d10 - 20.0f * d12 + 4.0f * d14 - 4.0f * d30 + 5.0f * d32 - d34;
    float part2  = 16.0f * d22 - 4.0f * d24;
    float part3  = 16.0f * d21 - 4.0f * d23;
    float part4  = 16.0f * d12 - 4.0f * d14 - 4.0f * d32 + d34;
    float part5  = 16.0f * d11 - 4.0f * d13 - 4.0f * d31 + d33;
    float part6  = 4.0f * d22 - 4.0f * d24;
    float part7  = 8.0f * d11 - 8.0f * d13 - 2.0f * d31 + 2.0f * d33;
    float part8  = 4.0f * d12 - 4.0f * d14 - d32 + d34;
    float part9  = 8.0f * d21 - 8.0f * d23;
    float part10 = -16.0f * d21 + 20.0f * d23 - 4.0f * d25;
    float part11 = -16.0f * d11 + 20.0f * d13 - 4.0f * d15 + 4.0f * d31 - 5.0f * d33 + d35;

    // Channels [18, 23]: [out30, out31, out32, out33, out34, out35]
    // Channels [24, 29]: [out40, out41, out42, out43, out44, out45]
    float part12 = 8.0f * d10 - 10.0f * d12 + 2.0f * d14 - 8.0f * d30 + 10.0f * d32 - 2.0f * d34;
    float part13 = part0 * 0.25f; // -4.0f * d20 + 5.0f * d22 - d24
    float part14 = part2 * 0.25f; // 4.0f * d22 - d24
    float part15 = 8.0f * d11 - 2.0f * d13 - 8.0f * d31 + 2.0f * d33;
    float part16 = 8.0f * d12 - 2.0f * d14 - 8.0f * d32 + 2.0f * d34;
    float part17 = part3 * 0.25f; // 4.0f * d21 - d23
    float part18 = part6 * 0.25f; // d22 - d24
    float part19 = 4.0f * d11 - 4.0f * d13 - 4.0f * d31 + 4.0f * d33;
    float part20 = 2.0f * d12 - 2.0f * d14 - 2.0f * d32 + 2.0f * d34;
    float part21 = part9 * 0.25f;                                        // 2.0f * (d21 - d23)
    float part22 = part10 * 0.25f;                                       // - 4.0f * d21 + 5.0f * d23 - d25
    float part23 = part11 * 0.5f + 6.0f * d31 - 7.5f * d33 + 1.5f * d35; // - 8.0f * d11 + 10.0f * d13 - 2.0f * d15 + 8.0f * d31 - 10.0f * d33 + 2.0f * d35;

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

    *((__global float *)dst_addr) = out6;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out7;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out8;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out9;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out10;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out11;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out12;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out13;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out14;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out15;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out16;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out17;
    dst_addr += dst_plane_stride;

    *((__global float *)dst_addr) = out18;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out19;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out20;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out21;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out22;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out23;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out24;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out25;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out26;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out27;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out28;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out29;
    dst_addr += dst_plane_stride;

    // Row5
    z_coord  = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 5;
    valid_y0 = select(y_coord0, (int4) - 1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, (int2) - 1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, (int4)SRC_DIM_1, (int4)z_coord >= (int)SRC_DIM_2);
    valid_y1 = select(valid_y1, (int2)SRC_DIM_1, (int2)z_coord >= (int)SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);
    z_coord  = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    float d50 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d51 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d52 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d53 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d54 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d55 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);

    // Channels [30, 35]
    out0 = 16.0f * d10 - 20.0f * d12 - 20.0f * d30 + 25.0f * d32 + 4.0f * d50 - 5.0f * d52 + d54 + 4.0f * d14 - 5.0f * d34;
    out1 = -16.0f * d11 - 16.0f * d12 + 4.0f * d13 + 20.0f * d31 + 20.0f * d32 - 5.0f * d33 - 4.0f * d51 - 4.0f * d52 + d53 + d54 + 4.0f * d14 - 5.0f * d34;
    out2 = 16.0f * d11 - 16.0f * d12 - 4.0f * d13 - 20.0f * d31 + 20.0f * d32 + 5.0f * d33 + 4.0f * d51 - 4.0f * d52 - d53 + d54 + 4.0f * d14 - 5.0f * d34;
    out3 = -8.0f * d11 - 4.0f * d12 + 8.0f * d13 + 10.0f * d31 - 10.0f * d33 + 5.0f * d32 - 2.0f * d51 + 2.0f * d53 - d52 + d54 + 4.0f * d14 - 5.0f * d34;
    out4 = 8.0f * d11 - 4.0f * d12 - 8.0f * d13 - 10.0f * d31 + 5.0f * d32 + 10.0f * d33 + 2.0f * d51 - 2.0f * d53 - d52 + d54 + 4.0f * d14 - 5.0f * d34;
    out5 = 16.0f * d11 - 20.0f * d13 + 4.0f * d15 - 20.0f * d31 + 25.0f * d33 - 5.0f * d35 + 4.0f * d51 - 5.0f * d53 + d55;

    *((__global float *)dst_addr) = out0;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out1;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out2;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out3;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out4;
    dst_addr += dst_plane_stride;
    *((__global float *)dst_addr) = out5;
    dst_addr += dst_plane_stride;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

/** This OpenCL kernel computes the input transform when the kernel size is 5x5/5x1 or 1x5 and the output tile is 4x4/4x1 or 1x4 when the data layout is NHWC
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note Dimension one of the input tensor (width for NHWC data layout) must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM_1=112)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note If this kernel is used to perform Winograd input transform 5x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x5, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x4_5x5_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * sizeof(float);

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    // Clamp coordinates. This clamp is valid for all rows
    int8 y_coord = (int8)(y * OUTPUT_TILE_W) + (int8)(0, 1, 2, 3, 4, 5, 6, 7) - (int8)PAD_LEFT;
    y_coord      = clamp(y_coord, (int8) - 1, (int8)SRC_DIM_1);

    // Row0
    // We can skip the border clamping along the z dimension as we cannot read out-of-bound in case of 5x1 kernels
    int z_coord = z * OUTPUT_TILE_H;

    // Load the input tile
    float8 in_row0;
    in_row0.s0 = *(__global float *)(src_addr + y_coord.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s1 = *(__global float *)(src_addr + y_coord.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s2 = *(__global float *)(src_addr + y_coord.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s3 = *(__global float *)(src_addr + y_coord.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s4 = *(__global float *)(src_addr + y_coord.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s5 = *(__global float *)(src_addr + y_coord.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s6 = *(__global float *)(src_addr + y_coord.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s7 = *(__global float *)(src_addr + y_coord.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Calculate common factors for intermediate tensor
    float8 comm_fact0 = 0.0f;
    float8 tmp0       = in_row0;

    float8 out0 = (float8)0.0f;

    OUTPUT_ROW_4x4_5x5(out0, tmp0, comm_fact0);

#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    // We can skip the border clamping along the y dimension as we cannot read out-of-bound in case of 1x5 kernels
    int y_coord = y * (int)OUTPUT_TILE_W;

    // Row0
    // We can skip the border clamping along the z dimension as we cannot read out-of-bound in case of 5x1 kernels
    int8 z_coord = (int8)(z * OUTPUT_TILE_H) + (int8)(0, 1, 2, 3, 4, 5, 6, 7) - (int8)PAD_TOP;
    int8 valid_y = select((int8)y_coord, (int8) - 1, z_coord < (int8)0);         // If z < 0, set y to -1
    valid_y      = select(valid_y, (int8)SRC_DIM_1, z_coord >= (int8)SRC_DIM_2); // If z >= SRC_DIM_2, set y to SRC_DIM_2
    z_coord      = clamp(z_coord, (int8)0, (int8)SRC_DIM_2 - 1);                 // Clamp z coordinate

    // Load the input tile
    float8 in_row0;
    in_row0.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord.s0 * src_stride_z);
    in_row0.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord.s1 * src_stride_z);
    in_row0.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord.s2 * src_stride_z);
    in_row0.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord.s3 * src_stride_z);
    in_row0.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord.s4 * src_stride_z);
    in_row0.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord.s5 * src_stride_z);
    in_row0.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord.s6 * src_stride_z);
    in_row0.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord.s7 * src_stride_z);

    // Calculate common factors for intermediate tensor
    float8 comm_fact0 = 0.0f;
    float8 tmp0       = in_row0;

    float8 out0 = (float8)0.0f;

    OUTPUT_ROW_4x4_5x5(out0, tmp0, comm_fact0);
#else                                            // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    float8 in_row0, in_row1, in_row2, in_row3, in_row4, in_row5, in_row6, in_row7;

    // Clamp coordinates. This clamp is valid for all rows
    int8 y_coord = (int8)(y * OUTPUT_TILE_W) + (int8)(0, 1, 2, 3, 4, 5, 6, 7) - (int8)PAD_LEFT;
    y_coord      = clamp(y_coord, (int8) - 1, (int8)SRC_DIM_1);

    // Row0
    int  z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 0;
    int8 valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);                    // If z < 0, set y to -1
    valid_y      = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2); // If z >= SRC_DIM_2, set y to SRC_DIM_2
    z_coord      = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);                             // Clamp z coordinate

    // Load the input tile
    in_row0.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row0.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row1
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 1;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row1.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row1.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row2
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 2;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row2.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row2.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row3
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 3;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row3.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row3.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row4
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 4;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row4.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row4.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row5
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 5;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row5.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row5.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row6
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 6;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row6.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row6.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    // Row7
    z_coord = (z * (int)OUTPUT_TILE_H) - (int)PAD_TOP + 7;
    valid_y = select(y_coord, (int8) - 1, (int8)z_coord < 0);
    valid_y = select(valid_y, (int8)SRC_DIM_1, (int8)z_coord >= (int)SRC_DIM_2);
    z_coord = clamp(z_coord, 0, (int)SRC_DIM_2 - 1);

    in_row7.s0 = *(__global float *)(src_addr + valid_y.s0 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s1 = *(__global float *)(src_addr + valid_y.s1 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s2 = *(__global float *)(src_addr + valid_y.s2 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s3 = *(__global float *)(src_addr + valid_y.s3 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s4 = *(__global float *)(src_addr + valid_y.s4 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s5 = *(__global float *)(src_addr + valid_y.s5 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s6 = *(__global float *)(src_addr + valid_y.s6 * (int)src_stride_y + z_coord * src_stride_z);
    in_row7.s7 = *(__global float *)(src_addr + valid_y.s7 * (int)src_stride_y + z_coord * src_stride_z);

    float8 comm_fact0 = in_row2 + in_row6 - 4.25f * in_row4;
    float8 comm_fact1 = in_row1 + in_row5 - 4.25f * in_row3;
    float8 comm_fact2 = 0.25f * in_row2 - 1.25f * in_row4 + in_row6;

    // Calculate intermediate tensor and reuse common factor vectors
    const float8 tmp0 = in_row0 - in_row6 + 5.25f * in_row4 - 5.25f * in_row2;
    const float8 tmp1 = comm_fact0 + comm_fact1;
    const float8 tmp2 = comm_fact0 - comm_fact1;

    comm_fact0 = 2.5f * in_row3;
    comm_fact1 = 0.5f * in_row1 - comm_fact0 + 2.f * in_row5;

    const float8 tmp3 = comm_fact1 + comm_fact2;
    const float8 tmp4 = comm_fact2 - comm_fact1;

    comm_fact1 = 2.f * in_row1 - comm_fact0 + 0.5f * in_row5;
    comm_fact2 = 4.f * in_row2 - 5.f * in_row4 + in_row6;

    const float8 tmp5 = comm_fact1 + comm_fact2;
    const float8 tmp6 = comm_fact2 - comm_fact1;
    const float8 tmp7 = in_row7 - in_row1 + 5.25f * in_row3 - 5.25f * in_row5;

    // Calculate output rows (reuse comm_fact0 vector)
    float8 out0, out1, out2, out3, out4, out5, out6, out7;
    OUTPUT_ROW_4x4_5x5(out0, tmp0, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out1, tmp1, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out2, tmp2, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out3, tmp3, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out4, tmp4, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out5, tmp5, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out6, tmp6, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out7, tmp7, comm_fact0);
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Store values across the channels
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x * sizeof(float) + (y + z * (int)NUM_TILES_X) * dst_stride_y;

    *((__global float *)(dst_addr + 0 * dst_stride_z)) = out0.s0;
    *((__global float *)(dst_addr + 1 * dst_stride_z)) = out0.s1;
    *((__global float *)(dst_addr + 2 * dst_stride_z)) = out0.s2;
    *((__global float *)(dst_addr + 3 * dst_stride_z)) = out0.s3;
    *((__global float *)(dst_addr + 4 * dst_stride_z)) = out0.s4;
    *((__global float *)(dst_addr + 5 * dst_stride_z)) = out0.s5;
    *((__global float *)(dst_addr + 6 * dst_stride_z)) = out0.s6;
    *((__global float *)(dst_addr + 7 * dst_stride_z)) = out0.s7;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    *((__global float *)(dst_addr + 8 * dst_stride_z))  = out1.s0;
    *((__global float *)(dst_addr + 9 * dst_stride_z))  = out1.s1;
    *((__global float *)(dst_addr + 10 * dst_stride_z)) = out1.s2;
    *((__global float *)(dst_addr + 11 * dst_stride_z)) = out1.s3;
    *((__global float *)(dst_addr + 12 * dst_stride_z)) = out1.s4;
    *((__global float *)(dst_addr + 13 * dst_stride_z)) = out1.s5;
    *((__global float *)(dst_addr + 14 * dst_stride_z)) = out1.s6;
    *((__global float *)(dst_addr + 15 * dst_stride_z)) = out1.s7;
    *((__global float *)(dst_addr + 16 * dst_stride_z)) = out2.s0;
    *((__global float *)(dst_addr + 17 * dst_stride_z)) = out2.s1;
    *((__global float *)(dst_addr + 18 * dst_stride_z)) = out2.s2;
    *((__global float *)(dst_addr + 19 * dst_stride_z)) = out2.s3;
    *((__global float *)(dst_addr + 20 * dst_stride_z)) = out2.s4;
    *((__global float *)(dst_addr + 21 * dst_stride_z)) = out2.s5;
    *((__global float *)(dst_addr + 22 * dst_stride_z)) = out2.s6;
    *((__global float *)(dst_addr + 23 * dst_stride_z)) = out2.s7;
    *((__global float *)(dst_addr + 24 * dst_stride_z)) = out3.s0;
    *((__global float *)(dst_addr + 25 * dst_stride_z)) = out3.s1;
    *((__global float *)(dst_addr + 26 * dst_stride_z)) = out3.s2;
    *((__global float *)(dst_addr + 27 * dst_stride_z)) = out3.s3;
    *((__global float *)(dst_addr + 28 * dst_stride_z)) = out3.s4;
    *((__global float *)(dst_addr + 29 * dst_stride_z)) = out3.s5;
    *((__global float *)(dst_addr + 30 * dst_stride_z)) = out3.s6;
    *((__global float *)(dst_addr + 31 * dst_stride_z)) = out3.s7;
    *((__global float *)(dst_addr + 32 * dst_stride_z)) = out4.s0;
    *((__global float *)(dst_addr + 33 * dst_stride_z)) = out4.s1;
    *((__global float *)(dst_addr + 34 * dst_stride_z)) = out4.s2;
    *((__global float *)(dst_addr + 35 * dst_stride_z)) = out4.s3;
    *((__global float *)(dst_addr + 36 * dst_stride_z)) = out4.s4;
    *((__global float *)(dst_addr + 37 * dst_stride_z)) = out4.s5;
    *((__global float *)(dst_addr + 38 * dst_stride_z)) = out4.s6;
    *((__global float *)(dst_addr + 39 * dst_stride_z)) = out4.s7;
    *((__global float *)(dst_addr + 40 * dst_stride_z)) = out5.s0;
    *((__global float *)(dst_addr + 41 * dst_stride_z)) = out5.s1;
    *((__global float *)(dst_addr + 42 * dst_stride_z)) = out5.s2;
    *((__global float *)(dst_addr + 43 * dst_stride_z)) = out5.s3;
    *((__global float *)(dst_addr + 44 * dst_stride_z)) = out5.s4;
    *((__global float *)(dst_addr + 45 * dst_stride_z)) = out5.s5;
    *((__global float *)(dst_addr + 46 * dst_stride_z)) = out5.s6;
    *((__global float *)(dst_addr + 47 * dst_stride_z)) = out5.s7;
    *((__global float *)(dst_addr + 48 * dst_stride_z)) = out6.s0;
    *((__global float *)(dst_addr + 49 * dst_stride_z)) = out6.s1;
    *((__global float *)(dst_addr + 50 * dst_stride_z)) = out6.s2;
    *((__global float *)(dst_addr + 51 * dst_stride_z)) = out6.s3;
    *((__global float *)(dst_addr + 52 * dst_stride_z)) = out6.s4;
    *((__global float *)(dst_addr + 53 * dst_stride_z)) = out6.s5;
    *((__global float *)(dst_addr + 54 * dst_stride_z)) = out6.s6;
    *((__global float *)(dst_addr + 55 * dst_stride_z)) = out6.s7;
    *((__global float *)(dst_addr + 56 * dst_stride_z)) = out7.s0;
    *((__global float *)(dst_addr + 57 * dst_stride_z)) = out7.s1;
    *((__global float *)(dst_addr + 58 * dst_stride_z)) = out7.s2;
    *((__global float *)(dst_addr + 59 * dst_stride_z)) = out7.s3;
    *((__global float *)(dst_addr + 60 * dst_stride_z)) = out7.s4;
    *((__global float *)(dst_addr + 61 * dst_stride_z)) = out7.s5;
    *((__global float *)(dst_addr + 62 * dst_stride_z)) = out7.s6;
    *((__global float *)(dst_addr + 63 * dst_stride_z)) = out7.s7;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}
#endif // defined(SRC_DIM_1) && defined(SRC_DIM_2)

/** This OpenCL kernel computes the input transform when the kernel size is 5x5/5x1 or 1x5 and the output tile is 4x4/4x1 or 1x4 when the data layout is NCHW
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note If this kernel is used to perform Winograd input transform 5x1, -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 * @note If this kernel is used to perform Winograd input transform 1x5, -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x4_5x5_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * OUTPUT_TILE_W * sizeof(float) + y * OUTPUT_TILE_H * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * sizeof(float)) - ((int)PAD_TOP * src_stride_y);

    // Load input tile
#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
    const float8 in_row0 = vload8(0, (__global float *)(src_addr));
#elif defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL) // !defined(WINOGRAD_FILTER_TRANSFORM_HORIZONTAL)
    const float8 in_row0 = (float8)(*((__global float *)(src_addr + 0 * src_stride_y)),
                                    *((__global float *)(src_addr + 1 * src_stride_y)),
                                    *((__global float *)(src_addr + 2 * src_stride_y)),
                                    *((__global float *)(src_addr + 3 * src_stride_y)),
                                    *((__global float *)(src_addr + 4 * src_stride_y)),
                                    *((__global float *)(src_addr + 5 * src_stride_y)),
                                    *((__global float *)(src_addr + 6 * src_stride_y)),
                                    *((__global float *)(src_addr + 7 * src_stride_y)));
#else                                            // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    const float8 in_row0 = vload8(0, (__global float *)(src_addr + 0 * src_stride_y));
    const float8 in_row1 = vload8(0, (__global float *)(src_addr + 1 * src_stride_y));
    const float8 in_row2 = vload8(0, (__global float *)(src_addr + 2 * src_stride_y));
    const float8 in_row3 = vload8(0, (__global float *)(src_addr + 3 * src_stride_y));
    const float8 in_row4 = vload8(0, (__global float *)(src_addr + 4 * src_stride_y));
    const float8 in_row5 = vload8(0, (__global float *)(src_addr + 5 * src_stride_y));
    const float8 in_row6 = vload8(0, (__global float *)(src_addr + 6 * src_stride_y));
    const float8 in_row7 = vload8(0, (__global float *)(src_addr + 7 * src_stride_y));
#endif                                           // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Calculate common factors for intermediate tensor
    float8 tmp0       = in_row0;
    float8 comm_fact0 = 0.0f;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    comm_fact0 += in_row2 + in_row6 - 4.25f * in_row4;
    tmp0 += -in_row6 + 5.25f * in_row4 - 5.25f * in_row2;

    float8 comm_fact1 = in_row1 + in_row5 - 4.25f * in_row3;
    float8 comm_fact2 = 0.25f * in_row2 - 1.25f * in_row4 + in_row6;

    const float8 tmp1 = comm_fact0 + comm_fact1;
    const float8 tmp2 = comm_fact0 - comm_fact1;

    comm_fact0 = 2.5f * in_row3;
    comm_fact1 = 0.5f * in_row1 - comm_fact0 + 2.f * in_row5;

    const float8 tmp3 = comm_fact1 + comm_fact2;
    const float8 tmp4 = comm_fact2 - comm_fact1;

    comm_fact1 = 2.f * in_row1 - comm_fact0 + 0.5f * in_row5;
    comm_fact2 = 4.f * in_row2 - 5.f * in_row4 + in_row6;

    const float8 tmp5 = comm_fact1 + comm_fact2;
    const float8 tmp6 = comm_fact2 - comm_fact1;
    const float8 tmp7 = in_row7 - in_row1 + 5.25f * in_row3 - 5.25f * in_row5;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Calculate output rows (reuse comm_fact0 vector)
    float8 out0;

    OUTPUT_ROW_4x4_5x5(out0, tmp0, comm_fact0);

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    float8 out1, out2, out3, out4, out5, out6, out7;

    OUTPUT_ROW_4x4_5x5(out1, tmp1, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out2, tmp2, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out3, tmp3, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out4, tmp4, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out5, tmp5, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out6, tmp6, comm_fact0);
    OUTPUT_ROW_4x4_5x5(out7, tmp7, comm_fact0);
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)

    // Store values across the channels
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * sizeof(float) + (x + y * (int)NUM_TILES_X) * dst_stride_y;

    *((__global float *)(dst_addr + 0 * dst_stride_z)) = out0.s0;
    *((__global float *)(dst_addr + 1 * dst_stride_z)) = out0.s1;
    *((__global float *)(dst_addr + 2 * dst_stride_z)) = out0.s2;
    *((__global float *)(dst_addr + 3 * dst_stride_z)) = out0.s3;
    *((__global float *)(dst_addr + 4 * dst_stride_z)) = out0.s4;
    *((__global float *)(dst_addr + 5 * dst_stride_z)) = out0.s5;
    *((__global float *)(dst_addr + 6 * dst_stride_z)) = out0.s6;
    *((__global float *)(dst_addr + 7 * dst_stride_z)) = out0.s7;

#if !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
    *((__global float *)(dst_addr + 8 * dst_stride_z))  = out1.s0;
    *((__global float *)(dst_addr + 9 * dst_stride_z))  = out1.s1;
    *((__global float *)(dst_addr + 10 * dst_stride_z)) = out1.s2;
    *((__global float *)(dst_addr + 11 * dst_stride_z)) = out1.s3;
    *((__global float *)(dst_addr + 12 * dst_stride_z)) = out1.s4;
    *((__global float *)(dst_addr + 13 * dst_stride_z)) = out1.s5;
    *((__global float *)(dst_addr + 14 * dst_stride_z)) = out1.s6;
    *((__global float *)(dst_addr + 15 * dst_stride_z)) = out1.s7;
    *((__global float *)(dst_addr + 16 * dst_stride_z)) = out2.s0;
    *((__global float *)(dst_addr + 17 * dst_stride_z)) = out2.s1;
    *((__global float *)(dst_addr + 18 * dst_stride_z)) = out2.s2;
    *((__global float *)(dst_addr + 19 * dst_stride_z)) = out2.s3;
    *((__global float *)(dst_addr + 20 * dst_stride_z)) = out2.s4;
    *((__global float *)(dst_addr + 21 * dst_stride_z)) = out2.s5;
    *((__global float *)(dst_addr + 22 * dst_stride_z)) = out2.s6;
    *((__global float *)(dst_addr + 23 * dst_stride_z)) = out2.s7;
    *((__global float *)(dst_addr + 24 * dst_stride_z)) = out3.s0;
    *((__global float *)(dst_addr + 25 * dst_stride_z)) = out3.s1;
    *((__global float *)(dst_addr + 26 * dst_stride_z)) = out3.s2;
    *((__global float *)(dst_addr + 27 * dst_stride_z)) = out3.s3;
    *((__global float *)(dst_addr + 28 * dst_stride_z)) = out3.s4;
    *((__global float *)(dst_addr + 29 * dst_stride_z)) = out3.s5;
    *((__global float *)(dst_addr + 30 * dst_stride_z)) = out3.s6;
    *((__global float *)(dst_addr + 31 * dst_stride_z)) = out3.s7;
    *((__global float *)(dst_addr + 32 * dst_stride_z)) = out4.s0;
    *((__global float *)(dst_addr + 33 * dst_stride_z)) = out4.s1;
    *((__global float *)(dst_addr + 34 * dst_stride_z)) = out4.s2;
    *((__global float *)(dst_addr + 35 * dst_stride_z)) = out4.s3;
    *((__global float *)(dst_addr + 36 * dst_stride_z)) = out4.s4;
    *((__global float *)(dst_addr + 37 * dst_stride_z)) = out4.s5;
    *((__global float *)(dst_addr + 38 * dst_stride_z)) = out4.s6;
    *((__global float *)(dst_addr + 39 * dst_stride_z)) = out4.s7;
    *((__global float *)(dst_addr + 40 * dst_stride_z)) = out5.s0;
    *((__global float *)(dst_addr + 41 * dst_stride_z)) = out5.s1;
    *((__global float *)(dst_addr + 42 * dst_stride_z)) = out5.s2;
    *((__global float *)(dst_addr + 43 * dst_stride_z)) = out5.s3;
    *((__global float *)(dst_addr + 44 * dst_stride_z)) = out5.s4;
    *((__global float *)(dst_addr + 45 * dst_stride_z)) = out5.s5;
    *((__global float *)(dst_addr + 46 * dst_stride_z)) = out5.s6;
    *((__global float *)(dst_addr + 47 * dst_stride_z)) = out5.s7;
    *((__global float *)(dst_addr + 48 * dst_stride_z)) = out6.s0;
    *((__global float *)(dst_addr + 49 * dst_stride_z)) = out6.s1;
    *((__global float *)(dst_addr + 50 * dst_stride_z)) = out6.s2;
    *((__global float *)(dst_addr + 51 * dst_stride_z)) = out6.s3;
    *((__global float *)(dst_addr + 52 * dst_stride_z)) = out6.s4;
    *((__global float *)(dst_addr + 53 * dst_stride_z)) = out6.s5;
    *((__global float *)(dst_addr + 54 * dst_stride_z)) = out6.s6;
    *((__global float *)(dst_addr + 55 * dst_stride_z)) = out6.s7;
    *((__global float *)(dst_addr + 56 * dst_stride_z)) = out7.s0;
    *((__global float *)(dst_addr + 57 * dst_stride_z)) = out7.s1;
    *((__global float *)(dst_addr + 58 * dst_stride_z)) = out7.s2;
    *((__global float *)(dst_addr + 59 * dst_stride_z)) = out7.s3;
    *((__global float *)(dst_addr + 60 * dst_stride_z)) = out7.s4;
    *((__global float *)(dst_addr + 61 * dst_stride_z)) = out7.s5;
    *((__global float *)(dst_addr + 62 * dst_stride_z)) = out7.s6;
    *((__global float *)(dst_addr + 63 * dst_stride_z)) = out7.s7;
#endif // !defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL) && !defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
}

#if defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)
/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 2x1
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_2x1_3x1_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 3x1, the output tile is 2x1 and the number of channels is multiple of 2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_2x1_3x1_stepz2_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 4x1
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x1_3x1_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 5x1 and the output tile is 4x1 when the data layout is NCHW
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=2
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x1_5x1_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

#if defined(SRC_DIM_1) && defined(SRC_DIM_2)
/** This OpenCL kernel computes the input transform when the kernel size is 3x1 and the output tile is 4x1 for data layout NHWC
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note Dimension one of the input tensor (width for NHWC data layout) must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM_1=112)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x1_3x1_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    winograd_input_transform_4x4_3x3_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 5x1 and the output tile is 4x1 for data layout NHWC
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note Dimension one of the input tensor (width for NHWC data layout) must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM_1=112)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=4
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=1
 * @note -DWINOGRAD_INPUT_TRANSFORM_HORIZONTAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_4x1_5x1_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    winograd_input_transform_4x4_5x5_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}
#endif // defined(SRC_DIM_1) && defined(SRC_DIM_2)
#endif // defined(WINOGRAD_INPUT_TRANSFORM_HORIZONTAL)

#if defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_1x2_1x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x3, the output tile is 1x2 and the number of channels is multiple of 2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=2
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_1x2_1x3_stepz2_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x4
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_1x4_1x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x5 and the output tile is 1x4
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_1x4_1x5_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
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
                                                 dst_offset_first_element_in_bytes);
}

#if defined(SRC_DIM_1) && defined(SRC_DIM_2)
/** This OpenCL kernel computes the input transform when the kernel size is 1x3 and the output tile is 1x4 for data layout NHWC
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note Dimension one of the input tensor (width for NHWC data layout) must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM_1=112)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_1x4_1x3_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    winograd_input_transform_4x4_3x3_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}

/** This OpenCL kernel computes the input transform when the kernel size is 1x5 and the output tile is 1x4 for data layout NHWC
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note Dimension one of the input tensor (width for NHWC data layout) must be passed at compile time using -DSRC_DIM1 (e.g. -DSRC_DIM_1=112)
 * @note Dimension two of the input tensor (height for NHWC data layout) must be passed at compile time using -DSRC_DIM2 (e.g. -DSRC_DIM_2=112)
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
 * @note The width of the output tile must be passed at compile time using -DOUTPUT_TILE_W: e.g. -DOUTPUT_TILE_W=1
 * @note The height of the output tile must be passed at compile time using -DOUTPUT_TILE_H: e.g. -DOUTPUT_TILE_H=4
 * @note -DWINOGRAD_INPUT_TRANSFORM_VERTICAL has to be passed at compile time
 *
 * @param[in] src_ptr                           Pointer to the source image. Supported data types: F32
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
 */
__kernel void winograd_input_transform_1x4_1x5_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    winograd_input_transform_4x4_5x5_stepz1_nhwc(src_ptr,
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
                                                 dst_offset_first_element_in_bytes);
}
#endif // defined(SRC_DIM_1) && defined(SRC_DIM_2)
#endif // defined(WINOGRAD_INPUT_TRANSFORM_VERTICAL)
#endif // defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP) && defined(OUTPUT_TILE_W) && defined(OUTPUT_TILE_H)