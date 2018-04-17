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

#if defined(NUM_CHANNELS)

/** This OpenCL kernel performs Winograd filter transform 3x3 when the data format is NCHW and the output tile is 2x2
 *
 * @note The number of channels must be passed at compile time using -DNUM_CHANNELS: e.g. -DNUM_CHANNELS=64
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void winograd_filter_transform_2x2_3x3_nchw(
    TENSOR4D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, NUM_CHANNELS);

    const __global uchar *src_addr = tensor4D_offset(&src, 0, 0, 0, 0);

    // Load the values from the input tensor
    float3 w0 = vload3(0, (__global float *)(src_addr + 0 * src_stride_y));
    float3 w1 = vload3(0, (__global float *)(src_addr + 1 * src_stride_y));
    float3 w2 = vload3(0, (__global float *)(src_addr + 2 * src_stride_y));

    // Transform the 3x3 tile in a 4x4 tile
    float4 out0 = 0.0f;
    float4 out1 = 0.0f;
    float4 out2 = 0.0f;
    float4 out3 = 0.0f;

    // Row 0
    out0.s0 = (w0.s0);
    out0.s1 = (w0.s0 + w0.s1 + w0.s2) * 0.5f;
    out0.s2 = (w0.s0 + w0.s2 - w0.s1) * 0.5f;
    out0.s3 = (w0.s2);

    // Row 1
    out1.s0 = (w0.s0 + w1.s0 + w2.s0) * 0.5f;
    out1.s1 = (w0.s0 + w1.s0 + w2.s0 + w0.s1 + w1.s1 + w2.s1 + w0.s2 + w1.s2 + w2.s2) * 0.25f;
    out1.s2 = (w0.s0 + w1.s0 + w2.s0 + w0.s2 + w1.s2 + w2.s2 - w0.s1 - w1.s1 - w2.s1) * 0.25f;
    out1.s3 = (w0.s2 + w1.s2 + w2.s2) * 0.5f;

    // Row 2
    out2.s0 = (w0.s0 + w2.s0 - w1.s0) * 0.5f;
    out2.s1 = (w0.s0 + w2.s0 + w0.s1 + w2.s1 + w0.s2 + w2.s2 - w1.s0 - w1.s1 - w1.s2) * 0.25f;
    out2.s2 = (w0.s0 + w2.s0 + w1.s1 + w0.s2 + w2.s2 - w1.s0 - w0.s1 - w2.s1 - w1.s2) * 0.25f;
    out2.s3 = (w0.s2 + w2.s2 - w1.s2) * 0.5f;

    // Row 3
    out3.s0 = (w2.s0);
    out3.s1 = (w2.s0 + w2.s1 + w2.s2) * 0.5f;
    out3.s2 = (w2.s0 + w2.s2 - w2.s1) * 0.5f;
    out3.s3 = (w2.s2);

    int z  = get_global_id(2);
    int x0 = z / NUM_CHANNELS; // idx filter
    int y0 = z % NUM_CHANNELS; // idx channel

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x0 * dst_stride_x + y0 * dst_stride_y;

    // Store the 16 values across the 16 channels
    *(__global float *)(dst_addr + 0 * dst_stride_z)  = out0.s0;
    *(__global float *)(dst_addr + 1 * dst_stride_z)  = out0.s1;
    *(__global float *)(dst_addr + 2 * dst_stride_z)  = out0.s2;
    *(__global float *)(dst_addr + 3 * dst_stride_z)  = out0.s3;
    *(__global float *)(dst_addr + 4 * dst_stride_z)  = out1.s0;
    *(__global float *)(dst_addr + 5 * dst_stride_z)  = out1.s1;
    *(__global float *)(dst_addr + 6 * dst_stride_z)  = out1.s2;
    *(__global float *)(dst_addr + 7 * dst_stride_z)  = out1.s3;
    *(__global float *)(dst_addr + 8 * dst_stride_z)  = out2.s0;
    *(__global float *)(dst_addr + 9 * dst_stride_z)  = out2.s1;
    *(__global float *)(dst_addr + 10 * dst_stride_z) = out2.s2;
    *(__global float *)(dst_addr + 11 * dst_stride_z) = out2.s3;
    *(__global float *)(dst_addr + 12 * dst_stride_z) = out3.s0;
    *(__global float *)(dst_addr + 13 * dst_stride_z) = out3.s1;
    *(__global float *)(dst_addr + 14 * dst_stride_z) = out3.s2;
    *(__global float *)(dst_addr + 15 * dst_stride_z) = out3.s3;
}

/** This OpenCL kernel performs Winograd filter transform 3x3 when the data format is NCHW and the output tile is 4x4
 *
 * @note The number of channels must be passed at compile time using -DNUM_CHANNELS: e.g. -DNUM_CHANNELS=64
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void winograd_filter_transform_4x4_3x3_nchw(
    TENSOR4D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, NUM_CHANNELS);

    const __global uchar *src_addr = tensor4D_offset(&src, 0, 0, 0, 0);

    // Load the values from the input tensor
    float3 w0 = vload3(0, (__global float *)(src_addr + 0 * src_stride_y));
    float3 w1 = vload3(0, (__global float *)(src_addr + 1 * src_stride_y));
    float3 w2 = vload3(0, (__global float *)(src_addr + 2 * src_stride_y));

    // Transform the 3x3 tile in a 6x6 tile
    float8 out0 = 0.0f;
    float8 out1 = 0.0f;
    float8 out2 = 0.0f;
    float8 out3 = 0.0f;
    float8 out4 = 0.0f;
    float8 out5 = 0.0f;

    // Row 0
    out0.s0 = (w0.s0) / 16.f;
    out0.s1 = (-w0.s0 - w0.s1 - w0.s2) / 24.f;
    out0.s2 = (-w0.s0 + w0.s1 - w0.s2) / 24.f;
    out0.s3 = (w0.s0 + 2.f * w0.s1 + 4.f * w0.s2) / 96.f;
    out0.s4 = (w0.s0 - 2.f * w0.s1 + 4.f * w0.s2) / 96.f;
    out0.s5 = (w0.s2) / 4.f;

    // Row 1
    out1.s0 = (-w0.s0 - w1.s0 - w2.s0) / 24.f;
    out1.s1 = (w0.s0 + w1.s0 + w2.s0 + w0.s1 + w1.s1 + w2.s1 + w0.s2 + w1.s2 + w2.s2) / 36.f;
    out1.s2 = (w0.s0 + w1.s0 + w2.s0 - w0.s1 - w1.s1 - w2.s1 + w0.s2 + w1.s2 + w2.s2) / 36.f;
    out1.s3 = (-w0.s0 - w1.s0 - w2.s0 + 2.f * (-w0.s1 - w1.s1 - w2.s1) + 4.f * (-w0.s2 - w1.s2 - w2.s2)) / 144.f;
    out1.s4 = (-w0.s0 - w1.s0 - w2.s0 + 2.f * (w0.s1 + w1.s1 + w2.s1) + 4.f * (-w0.s2 - w1.s2 - w2.s2)) / 144.f;
    out1.s5 = (-w0.s2 - w1.s2 - w2.s2) / 6.f;

    // Row 2
    out2.s0 = (-w0.s0 + w1.s0 - w2.s0) / 24.f;
    out2.s1 = (w0.s0 - w1.s0 + w2.s0 + w0.s1 - w1.s1 + w2.s1 + w0.s2 - w1.s2 + w2.s2) / 36.f;
    out2.s2 = (w0.s0 - w1.s0 + w2.s0 - w0.s1 + w1.s1 - w2.s1 + w0.s2 - w1.s2 + w2.s2) / 36.f;
    out2.s3 = (-w0.s0 + w1.s0 - w2.s0 + 2.f * (-w0.s1 + w1.s1 - w2.s1) + 4.f * (-w0.s2 + w1.s2 - w2.s2)) / 144.f;
    out2.s4 = (-w0.s0 + w1.s0 - w2.s0 + 2.f * (w0.s1 - w1.s1 + w2.s1) + 4.f * (-w0.s2 + w1.s2 - w2.s2)) / 144.f;
    out2.s5 = (-w0.s2 + w1.s2 - w2.s2) / 6.f;

    // Row 3
    out3.s0 = (w0.s0 + 2.f * w1.s0 + 4.f * w2.s0) / 96.f;
    out3.s1 = (-w0.s0 - 2.f * w1.s0 - 4.f * w2.s0 - w0.s1 - 2.f * w1.s1 - 4.f * w2.s1 - w0.s2 - 2.f * w1.s2 - 4.f * w2.s2) / 144.f;
    out3.s2 = (-w0.s0 - 2.f * w1.s0 - 4.f * w2.s0 + w0.s1 + 2.f * w1.s1 + 4.f * w2.s1 - w0.s2 - 2.f * w1.s2 - 4.f * w2.s2) / 144.f;
    out3.s3 = ((w0.s0 + 2.f * w1.s0 + 4.f * w2.s0) + 2.f * (w0.s1 + 2.f * w1.s1 + 4.f * w2.s1) + 4.f * (w0.s2 + 2.f * w1.s2 + 4.f * w2.s2)) / 576.f;
    out3.s4 = ((w0.s0 + 2.f * w1.s0 + 4.f * w2.s0) + 2.f * (-w0.s1 - 2.f * w1.s1 - 4.f * w2.s1) + 4.f * (w0.s2 + 2.f * w1.s2 + 4.f * w2.s2)) / 576.f;
    out3.s5 = (w0.s2 + 2.f * w1.s2 + 4.f * w2.s2) / 24.f;

    // Row 4
    out4.s0 = (w0.s0 - 2.f * w1.s0 + 4.f * w2.s0) / 96.f;
    out4.s1 = (-w0.s0 + 2.f * w1.s0 - 4.f * w2.s0 - w0.s1 + 2.f * w1.s1 - 4.f * w2.s1 - w0.s2 + 2.f * w1.s2 - 4.f * w2.s2) / 144.f;
    out4.s2 = (-w0.s0 + 2.f * w1.s0 - 4.f * w2.s0 + w0.s1 - 2.f * w1.s1 + 4.f * w2.s1 - w0.s2 + 2.f * w1.s2 - 4.f * w2.s2) / 144.f;
    out4.s3 = ((w0.s0 - 2.f * w1.s0 + 4.f * w2.s0) + 2.f * (w0.s1 - 2.f * w1.s1 + 4.f * w2.s1) + 4.f * (w0.s2 - 2.f * w1.s2 + 4.f * w2.s2)) / 576.f;
    out4.s4 = ((w0.s0 - 2.f * w1.s0 + 4.f * w2.s0) + 2.f * (-w0.s1 + 2.f * w1.s1 - 4.f * w2.s1) + 4.f * (w0.s2 - 2.f * w1.s2 + 4.f * w2.s2)) / 576.f;
    out4.s5 = (w0.s2 - 2.f * w1.s2 + 4.f * w2.s2) / 24.f;

    // Row 5
    out5.s0 = (w2.s0) / 4.f;
    out5.s1 = (-w2.s0 - w2.s1 - w2.s2) / 6.f;
    out5.s2 = (-w2.s0 + w2.s1 - w2.s2) / 6.f;
    out5.s3 = (w2.s0 + 2.f * w2.s1 + 4.f * w2.s2) / 24.f;
    out5.s4 = (w2.s0 - 2.f * w2.s1 + 4.f * w2.s2) / 24.f;
    out5.s5 = (w2.s2);

    int z  = get_global_id(2);
    int x0 = z / NUM_CHANNELS; // idx filter
    int y0 = z % NUM_CHANNELS; // idx channel

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x0 * dst_stride_x + y0 * dst_stride_y;

    // Store the 36 values across the 36 channels
    *(__global float *)(dst_addr + 0 * dst_stride_z)  = out0.s0;
    *(__global float *)(dst_addr + 1 * dst_stride_z)  = out0.s1;
    *(__global float *)(dst_addr + 2 * dst_stride_z)  = out0.s2;
    *(__global float *)(dst_addr + 3 * dst_stride_z)  = out0.s3;
    *(__global float *)(dst_addr + 4 * dst_stride_z)  = out0.s4;
    *(__global float *)(dst_addr + 5 * dst_stride_z)  = out0.s5;
    *(__global float *)(dst_addr + 6 * dst_stride_z)  = out1.s0;
    *(__global float *)(dst_addr + 7 * dst_stride_z)  = out1.s1;
    *(__global float *)(dst_addr + 8 * dst_stride_z)  = out1.s2;
    *(__global float *)(dst_addr + 9 * dst_stride_z)  = out1.s3;
    *(__global float *)(dst_addr + 10 * dst_stride_z) = out1.s4;
    *(__global float *)(dst_addr + 11 * dst_stride_z) = out1.s5;
    *(__global float *)(dst_addr + 12 * dst_stride_z) = out2.s0;
    *(__global float *)(dst_addr + 13 * dst_stride_z) = out2.s1;
    *(__global float *)(dst_addr + 14 * dst_stride_z) = out2.s2;
    *(__global float *)(dst_addr + 15 * dst_stride_z) = out2.s3;
    *(__global float *)(dst_addr + 16 * dst_stride_z) = out2.s4;
    *(__global float *)(dst_addr + 17 * dst_stride_z) = out2.s5;
    *(__global float *)(dst_addr + 18 * dst_stride_z) = out3.s0;
    *(__global float *)(dst_addr + 19 * dst_stride_z) = out3.s1;
    *(__global float *)(dst_addr + 20 * dst_stride_z) = out3.s2;
    *(__global float *)(dst_addr + 21 * dst_stride_z) = out3.s3;
    *(__global float *)(dst_addr + 22 * dst_stride_z) = out3.s4;
    *(__global float *)(dst_addr + 23 * dst_stride_z) = out3.s5;
    *(__global float *)(dst_addr + 24 * dst_stride_z) = out4.s0;
    *(__global float *)(dst_addr + 25 * dst_stride_z) = out4.s1;
    *(__global float *)(dst_addr + 26 * dst_stride_z) = out4.s2;
    *(__global float *)(dst_addr + 27 * dst_stride_z) = out4.s3;
    *(__global float *)(dst_addr + 28 * dst_stride_z) = out4.s4;
    *(__global float *)(dst_addr + 29 * dst_stride_z) = out4.s5;
    *(__global float *)(dst_addr + 30 * dst_stride_z) = out5.s0;
    *(__global float *)(dst_addr + 31 * dst_stride_z) = out5.s1;
    *(__global float *)(dst_addr + 32 * dst_stride_z) = out5.s2;
    *(__global float *)(dst_addr + 33 * dst_stride_z) = out5.s3;
    *(__global float *)(dst_addr + 34 * dst_stride_z) = out5.s4;
    *(__global float *)(dst_addr + 35 * dst_stride_z) = out5.s5;
}

/** This OpenCL kernel performs Winograd filter transform 5x5 when the data format is NCHW and the output tile is 4x4
 *
 * @note The number of channels must be passed at compile time using -DNUM_CHANNELS: e.g. -DNUM_CHANNELS=64
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_stride_w                      Stride of the source tensor in W dimension (in bytes)
 * @param[in]  src_step_w                        src_stride_w * number of elements along W processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void winograd_filter_transform_4x4_5x5_nchw(
    TENSOR4D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, NUM_CHANNELS);

    const __global uchar *src_addr = tensor4D_offset(&src, 0, 0, 0, 0);

    // Load the values from the input tensor
    const char   stride_x = 4 * sizeof(float); // Used for accessing the last value in each row
    const uchar8 stride_y = (uchar8)(0, 1, 2, 3, 4, 0, 0, 0) * (uchar8)src_stride_y;

    float4 w00 = vload4(0, (__global float *)(src_addr + stride_y.s0));
    float  w01 = *((__global float *)(src_addr + stride_y.s0 + stride_x));
    float4 w10 = vload4(0, (__global float *)(src_addr + stride_y.s1));
    float  w11 = *((__global float *)(src_addr + stride_y.s1 + stride_x));
    float4 w20 = vload4(0, (__global float *)(src_addr + stride_y.s2));
    float  w21 = *((__global float *)(src_addr + stride_y.s2 + stride_x));
    float4 w30 = vload4(0, (__global float *)(src_addr + stride_y.s3));
    float  w31 = *((__global float *)(src_addr + stride_y.s3 + stride_x));
    float4 w40 = vload4(0, (__global float *)(src_addr + stride_y.s4));
    float  w41 = *((__global float *)(src_addr + stride_y.s4 + stride_x));

    // Transform the 3x3 tile in a 8x8 tile
    float8 out0 = 0.0f;
    float8 out1 = 0.0f;
    float8 out2 = 0.0f;
    float8 out3 = 0.0f;
    float8 out4 = 0.0f;
    float8 out5 = 0.0f;
    float8 out6 = 0.0f;
    float8 out7 = 0.0f;

    // Row 0
    out0.s0 = w00.s0;
    out0.s1 = -2.f * (w00.s0 + w00.s1 + w00.s2 + w00.s3 + w01) / 9.f;
    out0.s2 = -2.f * (w00.s0 - w00.s1 + w00.s2 - w00.s3 + w01) / 9.f;
    out0.s3 = (w00.s0 + 2.f * w00.s1 + 4.f * w00.s2 + 8.f * w00.s3 + 16.f * w01) / 90.f;
    out0.s4 = (w00.s0 - 2.f * w00.s1 + 4.f * w00.s2 - 8.f * w00.s3 + 16.f * w01) / 90.f;
    out0.s5 = (16.f * w00.s0 + 8.f * w00.s1 + 4.f * w00.s2 + 2.f * w00.s3 + w01) / 180.f;
    out0.s6 = (16.f * w00.s0 - 8.f * w00.s1 + 4.f * w00.s2 - 2.f * w00.s3 + w01) / 180.f;
    out0.s7 = w01;

    // Row 1
    out1.s0 = -2.f * (w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) / 9.f;
    out1.s1 = 4.f * ((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) + (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) + (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) +
                     (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) + (w01 + w11 + w21 + w31 + w41)) / 81.f;
    out1.s2 = 4.f * ((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) - (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) + (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) -
                     (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) + (w01 + w11 + w21 + w31 + w41)) / 81.f;
    out1.s3 = -((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) + 2.f * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) + 4.f * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) + 8.f *
                (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) + 16.f * (w01 + w11 + w21 + w31 + w41)) / 405.f;
    out1.s4 = -((w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) - 2.f * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) + 4.f * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) - 8.f *
                (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) + 16.f * (w01 + w11 + w21 + w31 + w41)) / 405.f;
    out1.s5 = -(16.f * (w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) + 8.f * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) + 4.f * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) + 2.f *
                (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) + (w01 + w11 + w21 + w31 + w41)) / 810.f;
    out1.s6 = -(16.f * (w00.s0 + w10.s0 + w20.s0 + w30.s0 + w40.s0) - 8.f * (w00.s1 + w10.s1 + w20.s1 + w30.s1 + w40.s1) + 4.f * (w00.s2 + w10.s2 + w20.s2 + w30.s2 + w40.s2) - 2.f *
                (w00.s3 + w10.s3 + w20.s3 + w30.s3 + w40.s3) + (w01 + w11 + w21 + w31 + w41)) / 810.f;
    out1.s7 = -2.f * (w01 + w11 + w21 + w31 + w41) / 9.f;

    // Row 2
    out2.s0 = -2.f * (w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) / 9.f;
    out2.s1 = 4.f * ((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) + (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) + (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) +
                     (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) + (w01 - w11 + w21 - w31 + w41)) / 81.f;
    out2.s2 = 4.f * ((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) - (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) + (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) -
                     (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) + (w01 - w11 + w21 - w31 + w41)) / 81.f;
    out2.s3 = -((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) + 2.f * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) + 4.f * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) + 8.f *
                (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) + 16.f * (w01 - w11 + w21 - w31 + w41)) / 405.f;
    out2.s4 = -((w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) - 2.f * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) + 4.f * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) - 8.f *
                (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) + 16.f * (w01 - w11 + w21 - w31 + w41)) / 405.f;
    out2.s5 = -(16.f * (w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) + 8.f * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) + 4.f * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) + 2.f *
                (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) + (w01 - w11 + w21 - w31 + w41)) / 810.f;
    out2.s6 = -(16.f * (w00.s0 - w10.s0 + w20.s0 - w30.s0 + w40.s0) - 8.f * (w00.s1 - w10.s1 + w20.s1 - w30.s1 + w40.s1) + 4.f * (w00.s2 - w10.s2 + w20.s2 - w30.s2 + w40.s2) - 2.f *
                (w00.s3 - w10.s3 + w20.s3 - w30.s3 + w40.s3) + (w01 - w11 + w21 - w31 + w41)) / 810.f;
    out2.s7 = -2.f * (w01 - w11 + w21 - w31 + w41) / 9.f;

    // Row 3
    out3.s0 = (w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) / 90.f;
    out3.s1 = -((w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) + (w00.s1 + 2.f * w10.s1 + 4.f * w20.s1 + 8.f * w30.s1 + 16.f * w40.s1) +
                (w00.s2 + 2.f * w10.s2 + 4.f * w20.s2 + 8.f * w30.s2 + 16.f * w40.s2) + (w00.s3 + 2.f * w10.s3 + 4.f * w20.s3 + 8.f * w30.s3 + 16.f * w40.s3) +
                (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41)) / 405.f;
    out3.s2 = -((w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) - (w00.s1 + 2.f * w10.s1 + 4.f * w20.s1 + 8.f * w30.s1 + 16.f * w40.s1) +
                (w00.s2 + 2.f * w10.s2 + 4.f * w20.s2 + 8.f * w30.s2 + 16.f * w40.s2) - (w00.s3 + 2.f * w10.s3 + 4.f * w20.s3 + 8.f * w30.s3 + 16.f * w40.s3) +
                (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41)) / 405.f;
    out3.s3 = ((w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) + 2.f * (w00.s1 + 2.f * w10.s1 + 4.f * w20.s1 + 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 + 2.f * w10.s2 + 4.f * w20.s2 + 8.f * w30.s2 + 16.f * w40.s2) + 8.f * (w00.s3 + 2.f * w10.s3 + 4.f * w20.s3 + 8.f * w30.s3 + 16.f * w40.s3) + 16.f *
               (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41)) / 8100.f;
    out3.s4 = ((w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) - 2.f * (w00.s1 + 2.f * w10.s1 + 4.f * w20.s1 + 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 + 2.f * w10.s2 + 4.f * w20.s2 + 8.f * w30.s2 + 16.f * w40.s2) - 8.f * (w00.s3 + 2.f * w10.s3 + 4.f * w20.s3 + 8.f * w30.s3 + 16.f * w40.s3) + 16.f *
               (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41)) / 8100.f;
    out3.s5 = (16.f * (w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) + 8.f * (w00.s1 + 2.f * w10.s1 + 4.f * w20.s1 + 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 + 2.f * w10.s2 + 4.f * w20.s2 + 8.f * w30.s2 + 16.f * w40.s2) + 2.f * (w00.s3 + 2.f * w10.s3 + 4.f * w20.s3 + 8.f * w30.s3 + 16.f * w40.s3) +
               (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41)) / 16200.f;
    out3.s6 = (16.f * (w00.s0 + 2.f * w10.s0 + 4.f * w20.s0 + 8.f * w30.s0 + 16.f * w40.s0) - 8.f * (w00.s1 + 2.f * w10.s1 + 4.f * w20.s1 + 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 + 2.f * w10.s2 + 4.f * w20.s2 + 8.f * w30.s2 + 16.f * w40.s2) - 2.f * (w00.s3 + 2.f * w10.s3 + 4.f * w20.s3 + 8.f * w30.s3 + 16.f * w40.s3) +
               (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41)) / 16200.f;
    out3.s7 = (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) / 90.f;

    // Row 4
    out4.s0 = (w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) / 90.f;
    out4.s1 = -((w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) + (w00.s1 - 2.f * w10.s1 + 4.f * w20.s1 - 8.f * w30.s1 + 16.f * w40.s1) +
                (w00.s2 - 2.f * w10.s2 + 4.f * w20.s2 - 8.f * w30.s2 + 16.f * w40.s2) + (w00.s3 - 2.f * w10.s3 + 4.f * w20.s3 - 8.f * w30.s3 + 16.f * w40.s3) +
                (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41)) / 405.f;
    out4.s2 = -((w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) - (w00.s1 - 2.f * w10.s1 + 4.f * w20.s1 - 8.f * w30.s1 + 16.f * w40.s1) +
                (w00.s2 - 2.f * w10.s2 + 4.f * w20.s2 - 8.f * w30.s2 + 16.f * w40.s2) - (w00.s3 - 2.f * w10.s3 + 4.f * w20.s3 - 8.f * w30.s3 + 16.f * w40.s3) +
                (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41)) / 405.f;
    out4.s3 = ((w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) + 2.f * (w00.s1 - 2.f * w10.s1 + 4.f * w20.s1 - 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 - 2.f * w10.s2 + 4.f * w20.s2 - 8.f * w30.s2 + 16.f * w40.s2) + 8.f * (w00.s3 - 2.f * w10.s3 + 4.f * w20.s3 - 8.f * w30.s3 + 16.f * w40.s3) + 16.f *
               (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41)) / 8100.f;
    out4.s4 = ((w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) - 2.f * (w00.s1 - 2.f * w10.s1 + 4.f * w20.s1 - 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 - 2.f * w10.s2 + 4.f * w20.s2 - 8.f * w30.s2 + 16.f * w40.s2) - 8.f * (w00.s3 - 2.f * w10.s3 + 4.f * w20.s3 - 8.f * w30.s3 + 16.f * w40.s3) + 16.f *
               (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41)) / 8100.f;
    out4.s5 = (16.f * (w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) + 8.f * (w00.s1 - 2.f * w10.s1 + 4.f * w20.s1 - 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 - 2.f * w10.s2 + 4.f * w20.s2 - 8.f * w30.s2 + 16.f * w40.s2) + 2.f * (w00.s3 - 2.f * w10.s3 + 4.f * w20.s3 - 8.f * w30.s3 + 16.f * w40.s3) +
               (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41)) / 16200.f;
    out4.s6 = (16.f * (w00.s0 - 2.f * w10.s0 + 4.f * w20.s0 - 8.f * w30.s0 + 16.f * w40.s0) - 8.f * (w00.s1 - 2.f * w10.s1 + 4.f * w20.s1 - 8.f * w30.s1 + 16.f * w40.s1) + 4.f *
               (w00.s2 - 2.f * w10.s2 + 4.f * w20.s2 - 8.f * w30.s2 + 16.f * w40.s2) - 2.f * (w00.s3 - 2.f * w10.s3 + 4.f * w20.s3 - 8.f * w30.s3 + 16.f * w40.s3) +
               (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41)) / 16200.f;
    out4.s7 = (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) / 90.f;

    // Row 5
    out5.s0 = (16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) / 180.f;
    out5.s1 = -((16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) + (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s1 + w40.s1) +
                (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) + (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) +
                (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 810.f;
    out5.s2 = -((16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) - (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s1 + w40.s1) +
                (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) - (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) +
                (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 810.f;
    out5.s3 = ((16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) + 2.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) + 8.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 16200.f;
    out5.s4 = ((16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) - 2.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) - 8.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 16200.f;
    out5.s5 = (16.f * (16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) + 8.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) + 2.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) +
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 32400.f;
    out5.s6 = (16.f * (16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) - 8.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) - 2.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) +
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 32400.f;
    out5.s7 = (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) / 180.f;

    // Row 6
    out6.s0 = (16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) / 180.f;
    out6.s1 = -((16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) + (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s1 + w40.s1) +
                (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) + (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) +
                (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 810.f;
    out6.s2 = -((16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) - (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s1 + w40.s1) +
                (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) - (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) +
                (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 810.f;
    out6.s3 = ((16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) + 2.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) + 8.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 16200.f;
    out6.s4 = ((16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) - 2.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) - 8.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 16200.f;
    out6.s5 = (16.f * (16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) + 8.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) + 2.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) +
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 32400.f;
    out6.s6 = (16.f * (16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) - 8.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s0 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) - 2.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) +
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 32400.f;
    out6.s7 = (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) / 180.f;

    // Row 7
    out7.s0 = w40.s0;
    out7.s1 = -2.f * (w40.s0 + w40.s1 + w40.s2 + w40.s3 + w41) / 9.f;
    out7.s2 = -2.f * (w40.s0 - w40.s1 + w40.s2 - w40.s3 + w41) / 9.f;
    out7.s3 = (w40.s0 + 2.f * w40.s1 + 4.f * w40.s2 + 8.f * w40.s3 + 16.f * w41) / 90.f;
    out7.s4 = (w40.s0 - 2.f * w40.s1 + 4.f * w40.s2 - 8.f * w40.s3 + 16.f * w41) / 90.f;
    out7.s5 = (16.f * w40.s0 + 8.f * w40.s1 + 4.f * w40.s2 + 2.f * w40.s3 + w41) / 180.f;
    out7.s6 = (16.f * w40.s0 - 8.f * w40.s1 + 4.f * w40.s2 - 2.f * w40.s3 + w41) / 180.f;
    out7.s7 = w41;

    int z  = get_global_id(2);
    int x0 = z / NUM_CHANNELS; // idx filter
    int y0 = z % NUM_CHANNELS; // idx channel

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x0 * dst_stride_x + y0 * dst_stride_y;

    // Store the 64 values across the 64 channels
    *(__global float *)(dst_addr + 0 * dst_stride_z)  = out0.s0;
    *(__global float *)(dst_addr + 1 * dst_stride_z)  = out0.s1;
    *(__global float *)(dst_addr + 2 * dst_stride_z)  = out0.s2;
    *(__global float *)(dst_addr + 3 * dst_stride_z)  = out0.s3;
    *(__global float *)(dst_addr + 4 * dst_stride_z)  = out0.s4;
    *(__global float *)(dst_addr + 5 * dst_stride_z)  = out0.s5;
    *(__global float *)(dst_addr + 6 * dst_stride_z)  = out0.s6;
    *(__global float *)(dst_addr + 7 * dst_stride_z)  = out0.s7;
    *(__global float *)(dst_addr + 8 * dst_stride_z)  = out1.s0;
    *(__global float *)(dst_addr + 9 * dst_stride_z)  = out1.s1;
    *(__global float *)(dst_addr + 10 * dst_stride_z) = out1.s2;
    *(__global float *)(dst_addr + 11 * dst_stride_z) = out1.s3;
    *(__global float *)(dst_addr + 12 * dst_stride_z) = out1.s4;
    *(__global float *)(dst_addr + 13 * dst_stride_z) = out1.s5;
    *(__global float *)(dst_addr + 14 * dst_stride_z) = out1.s6;
    *(__global float *)(dst_addr + 15 * dst_stride_z) = out1.s7;
    *(__global float *)(dst_addr + 16 * dst_stride_z) = out2.s0;
    *(__global float *)(dst_addr + 17 * dst_stride_z) = out2.s1;
    *(__global float *)(dst_addr + 18 * dst_stride_z) = out2.s2;
    *(__global float *)(dst_addr + 19 * dst_stride_z) = out2.s3;
    *(__global float *)(dst_addr + 20 * dst_stride_z) = out2.s4;
    *(__global float *)(dst_addr + 21 * dst_stride_z) = out2.s5;
    *(__global float *)(dst_addr + 22 * dst_stride_z) = out2.s6;
    *(__global float *)(dst_addr + 23 * dst_stride_z) = out2.s7;
    *(__global float *)(dst_addr + 24 * dst_stride_z) = out3.s0;
    *(__global float *)(dst_addr + 25 * dst_stride_z) = out3.s1;
    *(__global float *)(dst_addr + 26 * dst_stride_z) = out3.s2;
    *(__global float *)(dst_addr + 27 * dst_stride_z) = out3.s3;
    *(__global float *)(dst_addr + 28 * dst_stride_z) = out3.s4;
    *(__global float *)(dst_addr + 29 * dst_stride_z) = out3.s5;
    *(__global float *)(dst_addr + 30 * dst_stride_z) = out3.s6;
    *(__global float *)(dst_addr + 31 * dst_stride_z) = out3.s7;
    *(__global float *)(dst_addr + 32 * dst_stride_z) = out4.s0;
    *(__global float *)(dst_addr + 33 * dst_stride_z) = out4.s1;
    *(__global float *)(dst_addr + 34 * dst_stride_z) = out4.s2;
    *(__global float *)(dst_addr + 35 * dst_stride_z) = out4.s3;
    *(__global float *)(dst_addr + 36 * dst_stride_z) = out4.s4;
    *(__global float *)(dst_addr + 37 * dst_stride_z) = out4.s5;
    *(__global float *)(dst_addr + 38 * dst_stride_z) = out4.s6;
    *(__global float *)(dst_addr + 39 * dst_stride_z) = out4.s7;
    *(__global float *)(dst_addr + 40 * dst_stride_z) = out5.s0;
    *(__global float *)(dst_addr + 41 * dst_stride_z) = out5.s1;
    *(__global float *)(dst_addr + 42 * dst_stride_z) = out5.s2;
    *(__global float *)(dst_addr + 43 * dst_stride_z) = out5.s3;
    *(__global float *)(dst_addr + 44 * dst_stride_z) = out5.s4;
    *(__global float *)(dst_addr + 45 * dst_stride_z) = out5.s5;
    *(__global float *)(dst_addr + 46 * dst_stride_z) = out5.s6;
    *(__global float *)(dst_addr + 47 * dst_stride_z) = out5.s7;
    *(__global float *)(dst_addr + 48 * dst_stride_z) = out6.s0;
    *(__global float *)(dst_addr + 49 * dst_stride_z) = out6.s1;
    *(__global float *)(dst_addr + 50 * dst_stride_z) = out6.s2;
    *(__global float *)(dst_addr + 51 * dst_stride_z) = out6.s3;
    *(__global float *)(dst_addr + 52 * dst_stride_z) = out6.s4;
    *(__global float *)(dst_addr + 53 * dst_stride_z) = out6.s5;
    *(__global float *)(dst_addr + 54 * dst_stride_z) = out6.s6;
    *(__global float *)(dst_addr + 55 * dst_stride_z) = out6.s7;
    *(__global float *)(dst_addr + 56 * dst_stride_z) = out7.s0;
    *(__global float *)(dst_addr + 57 * dst_stride_z) = out7.s1;
    *(__global float *)(dst_addr + 58 * dst_stride_z) = out7.s2;
    *(__global float *)(dst_addr + 59 * dst_stride_z) = out7.s3;
    *(__global float *)(dst_addr + 60 * dst_stride_z) = out7.s4;
    *(__global float *)(dst_addr + 61 * dst_stride_z) = out7.s5;
    *(__global float *)(dst_addr + 62 * dst_stride_z) = out7.s6;
    *(__global float *)(dst_addr + 63 * dst_stride_z) = out7.s7;
}
#endif // defined(NUM_CHANNELS)

#if defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP)
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

/** This OpenCL kernel computes the input transform when the kernel size is 5x5 and the output tile is 4x4
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
__kernel void winograd_input_transform_4x4_5x5_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * 4 * src_stride_x + y * 4 * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * src_stride_x) - ((int)PAD_TOP * src_stride_y);

    // Load 8x8 input tile
    const float8 in_row0 = vload8(0, (__global float *)(src_addr + 0 * src_stride_y));
    const float8 in_row1 = vload8(0, (__global float *)(src_addr + 1 * src_stride_y));
    const float8 in_row2 = vload8(0, (__global float *)(src_addr + 2 * src_stride_y));
    const float8 in_row3 = vload8(0, (__global float *)(src_addr + 3 * src_stride_y));
    const float8 in_row4 = vload8(0, (__global float *)(src_addr + 4 * src_stride_y));
    const float8 in_row5 = vload8(0, (__global float *)(src_addr + 5 * src_stride_y));
    const float8 in_row6 = vload8(0, (__global float *)(src_addr + 6 * src_stride_y));
    const float8 in_row7 = vload8(0, (__global float *)(src_addr + 7 * src_stride_y));

    // Calculate common factors for intermediate tensor
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

    // Store values across the 64 channels
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + z * dst_stride_x + (x + y * (int)NUM_TILES_X) * dst_stride_y;

    *((__global float *)(dst_addr + 0 * dst_stride_z))  = out0.s0;
    *((__global float *)(dst_addr + 1 * dst_stride_z))  = out0.s1;
    *((__global float *)(dst_addr + 2 * dst_stride_z))  = out0.s2;
    *((__global float *)(dst_addr + 3 * dst_stride_z))  = out0.s3;
    *((__global float *)(dst_addr + 4 * dst_stride_z))  = out0.s4;
    *((__global float *)(dst_addr + 5 * dst_stride_z))  = out0.s5;
    *((__global float *)(dst_addr + 6 * dst_stride_z))  = out0.s6;
    *((__global float *)(dst_addr + 7 * dst_stride_z))  = out0.s7;
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
}
#endif // defined(NUM_TILES_X) && defined(PAD_LEFT) && defined(PAD_TOP)

#if defined(NUM_TILES_X)
/** This OpenCL kernel performs Winograd output transform when the output tile is 2x2, the filter size 3x3 and the data format is NCHW
 *
 * @note The number of tiles along the X direction must be passed at compile time using -DNUM_TILES_X: e.g. -DNUM_TILES_X=16
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void winograd_output_transform_2x2_3x3_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(bias)
#endif // defined(HAS_BIAS)
)
{
    // Each thread stores a 2x2 tile
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const __global uchar *src_addr = tensor3D_offset(&src, 0, 0, 0);

    // Load the values across the 16 channels to compose the 4x4 tile
    float d00 = *((__global float *)(src_addr + 0 * src_stride_z));
    float d01 = *((__global float *)(src_addr + 1 * src_stride_z));
    float d02 = *((__global float *)(src_addr + 2 * src_stride_z));
    float d03 = *((__global float *)(src_addr + 3 * src_stride_z));

    float d10 = *((__global float *)(src_addr + 4 * src_stride_z));
    float d11 = *((__global float *)(src_addr + 5 * src_stride_z));
    float d12 = *((__global float *)(src_addr + 6 * src_stride_z));
    float d13 = *((__global float *)(src_addr + 7 * src_stride_z));

    float d20 = *((__global float *)(src_addr + 8 * src_stride_z));
    float d21 = *((__global float *)(src_addr + 9 * src_stride_z));
    float d22 = *((__global float *)(src_addr + 10 * src_stride_z));
    float d23 = *((__global float *)(src_addr + 11 * src_stride_z));

    float d30 = *((__global float *)(src_addr + 12 * src_stride_z));
    float d31 = *((__global float *)(src_addr + 13 * src_stride_z));
    float d32 = *((__global float *)(src_addr + 14 * src_stride_z));
    float d33 = *((__global float *)(src_addr + 15 * src_stride_z));

    // Compute the 2x2 output tile
    float k0 = d01 + d11 + d21;
    float k1 = d02 + d12 + d22;
    float k2 = d11 - d21 - d31;
    float k3 = d12 - d22 - d32;

    // out00 = d00 + d10 + d20 + d01 + d11 + d21 + d02 + d12 + d22
    // out01 = d01 + d11 + d21 - (d02 + d12 + d22) - (d03 + d13 + d23)
    // out10 = d10 - d20 - d30 + (d11 - d21 - d31) + (d12 - d22 - d32)
    // out11 = d11 - d21 - d31 - (d12 - d22 - d32) - (d13 - d23 - d33)

    float out00 = d10;
    float out01 = -d13;
    float out10 = d10;
    float out11 = -d13;

    out00 += d00 + d20 + k0 + k1;
    out01 += k0 - k1 - (d03 + d23);
    out10 += -d20 - d30 + k2 + k3;
    out11 += k2 - k3 + d23 + d33;

    int y_in  = get_global_id(1);
    int x_out = (y_in % NUM_TILES_X) * 2;
    int y_out = (y_in / NUM_TILES_X) * 2;
    int z_out = get_global_id(0);

#if defined(HAS_BIAS)
    // Add bias
    Vector bias = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bias);

    float b = (float) * ((__global float *)(vector_offset(&bias, z_out)));

    out00 += (float)b;
    out01 += (float)b;
    out10 += (float)b;
    out11 += (float)b;
#endif // defined(HAS_BIAS)

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_out * dst_stride_x + y_out * dst_stride_y + z_out * dst_stride_z;

    // Store the 2x2 output tile
    vstore2((float2)(out00, out01), 0, (__global float *)(dst_addr + 0 * dst_stride_y));
    vstore2((float2)(out10, out11), 0, (__global float *)(dst_addr + 1 * dst_stride_y));
}
#endif // defined(NUM_TILES_X)
