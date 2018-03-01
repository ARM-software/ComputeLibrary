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

#if defined(NUM_TILES_X)

/** This OpenCL kernel computes the input transform when the kernel size is 3x3 and the output tile is 2x2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
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
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * 2 * src_stride_x + y * 2 * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * src_stride_x) - ((int)PAD_TOP * src_stride_y);

    float4 in_row0 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float4 in_row1 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float4 in_row2 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float4 in_row3 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));

    float4 tmp0 = in_row0 - in_row2;
    float4 tmp1 = in_row1 + in_row2;
    float4 tmp2 = in_row2 - in_row1;
    float4 tmp3 = in_row1 - in_row3;

    float out00 = tmp0.s0 - tmp0.s2;
    float out01 = tmp0.s1 + tmp0.s2;
    float out02 = tmp0.s2 - tmp0.s1;
    float out03 = tmp0.s1 - tmp0.s3;

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

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * dst_stride_x + (x + y * (int)NUM_TILES_X) * dst_stride_y;

    *((__global float *)(dst_addr + 0 * dst_stride_z))  = out00;
    *((__global float *)(dst_addr + 1 * dst_stride_z))  = out01;
    *((__global float *)(dst_addr + 2 * dst_stride_z))  = out02;
    *((__global float *)(dst_addr + 3 * dst_stride_z))  = out03;
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
}

/** This OpenCL kernel computes the input transform when the kernel size is 3x3, the output tile is 2x2 and the number of channels is multiple of 2
 *
 * @note The number of tiles in the x axis must be passed at compile time using -DNUM_TILES_X (i.e.-DNUM_TILES_X=5).
 * @note The pad left and pad top must be passed at compile time using -DPAD_LEFT and -DPAD_TOP (i.e.-DPAD_LEFT=1 and -DPAD_TOP=0).
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
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * 2 * src_stride_x + y * 2 * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * src_stride_x) - ((int)PAD_TOP * src_stride_y);

    float4 in_row0 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float4 in_row1 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float4 in_row2 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float4 in_row3 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));

    src_addr += src_stride_z;
    float4 in_row4 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float4 in_row5 = vload4(0, (__global float *)(src_addr + 1 * src_stride_y));
    float4 in_row6 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float4 in_row7 = vload4(0, (__global float *)(src_addr + 3 * src_stride_y));

    float4 tmp0 = in_row0 - in_row2;
    float4 tmp1 = in_row1 + in_row2;
    float4 tmp2 = in_row2 - in_row1;
    float4 tmp3 = in_row1 - in_row3;

    float4 tmp4 = in_row4 - in_row6;
    float4 tmp5 = in_row5 + in_row6;
    float4 tmp6 = in_row6 - in_row5;
    float4 tmp7 = in_row5 - in_row7;

    float2 out00 = (float2)(tmp0.s0 - tmp0.s2, tmp4.s0 - tmp4.s2);
    float2 out01 = (float2)(tmp0.s1 + tmp0.s2, tmp4.s1 + tmp4.s2);
    float2 out02 = (float2)(tmp0.s2 - tmp0.s1, tmp4.s2 - tmp4.s1);
    float2 out03 = (float2)(tmp0.s1 - tmp0.s3, tmp4.s1 - tmp4.s3);

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

    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * dst_stride_x + (x + y * (int)NUM_TILES_X) * dst_stride_y;

    vstore2(out00, 0, (__global float *)(dst_addr + 0 * dst_stride_z));
    vstore2(out01, 0, (__global float *)(dst_addr + 1 * dst_stride_z));
    vstore2(out02, 0, (__global float *)(dst_addr + 2 * dst_stride_z));
    vstore2(out03, 0, (__global float *)(dst_addr + 3 * dst_stride_z));
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
}
#endif //defined(NUM_TILES_X)