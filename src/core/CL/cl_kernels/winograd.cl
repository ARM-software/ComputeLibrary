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

#if defined(SRC_DIM_Z)

/** This OpenCL kernel performs Winograd filter transform 3x3 when the data layout is NCHW and the output tile is 2x2
 *
 * @note In order to correctly split the input tensor in batches, its dimension across the Z axis (channels for NCHW, height for NHWC) must be passed at compile time using -DSRC_DIM_Z: e.g. -DSRC_DIM_Z=64
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
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, SRC_DIM_Z);

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
    int x0 = z / SRC_DIM_Z; // idx filter
    int y0 = z % SRC_DIM_Z; // idx channel

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

/** This OpenCL kernel performs Winograd filter transform 3x3 when the data layout is NCHW and the output tile is 4x4
 *
 * @note In order to correctly split the input tensor in batches, its dimension across the Z axis (channels for NCHW, height for NHWC) must be passed at compile time using -DSRC_DIM_Z: e.g. -DSRC_DIM_Z=64
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
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, SRC_DIM_Z);

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
    int x0 = z / SRC_DIM_Z; // idx filter
    int y0 = z % SRC_DIM_Z; // idx channel

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

/** This OpenCL kernel performs Winograd filter transform 3x3 when the data layout is NHWC and the output tile is 4x4
 *
 * @note In order to correctly split the input tensor in batches, its dimension across the Z axis (channels for NCHW, height for NHWC) must be passed at compile time using -DSRC_DIM_Z: e.g. -DSRC_DIM_Z=64
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
__kernel void winograd_filter_transform_4x4_3x3_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, SRC_DIM_Z);

    const __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + get_global_id(0) * src_step_x + get_global_id(1) * src_step_y + get_global_id(2) * src_step_w;

    // Load the values from the input tensor
    float w00 = *((__global float *)(src_addr + 0 * src_stride_z + 0 * src_stride_y));
    float w01 = *((__global float *)(src_addr + 0 * src_stride_z + 1 * src_stride_y));
    float w02 = *((__global float *)(src_addr + 0 * src_stride_z + 2 * src_stride_y));
    float w10 = *((__global float *)(src_addr + 1 * src_stride_z + 0 * src_stride_y));
    float w11 = *((__global float *)(src_addr + 1 * src_stride_z + 1 * src_stride_y));
    float w12 = *((__global float *)(src_addr + 1 * src_stride_z + 2 * src_stride_y));
    float w20 = *((__global float *)(src_addr + 2 * src_stride_z + 0 * src_stride_y));
    float w21 = *((__global float *)(src_addr + 2 * src_stride_z + 1 * src_stride_y));
    float w22 = *((__global float *)(src_addr + 2 * src_stride_z + 2 * src_stride_y));

    // Transform the 3x3 tile in a 6x6 tile
    float out00, out01, out02, out03, out04, out05;
    float out10, out11, out12, out13, out14, out15;
    float out20, out21, out22, out23, out24, out25;
    float out30, out31, out32, out33, out34, out35;
    float out40, out41, out42, out43, out44, out45;
    float out50, out51, out52, out53, out54, out55;

    out00 = out01 = out02 = out03 = out04 = out05 = 0.f;
    out10 = out11 = out12 = out13 = out14 = out15 = 0.f;
    out20 = out21 = out22 = out23 = out24 = out25 = 0.f;
    out30 = out31 = out32 = out33 = out34 = out35 = 0.f;
    out40 = out41 = out42 = out43 = out44 = out45 = 0.f;
    out50 = out51 = out52 = out53 = out54 = out55 = 0.f;

    // Row 0
    out00 = (w00) / 16.f;
    out01 = (-w00 - w01 - w02) / 24.f;
    out02 = (-w00 + w01 - w02) / 24.f;
    out03 = (w00 + 2.f * w01 + 4.f * w02) / 96.f;
    out04 = (w00 - 2.f * w01 + 4.f * w02) / 96.f;
    out05 = (w02) / 4.f;

    // Row 1
    out10 = (-w00 - w10 - w20) / 24.f;
    out11 = (w00 + w10 + w20 + w01 + w11 + w21 + w02 + w12 + w22) / 36.f;
    out12 = (w00 + w10 + w20 - w01 - w11 - w21 + w02 + w12 + w22) / 36.f;
    out13 = (-w00 - w10 - w20 + 2.f * (-w01 - w11 - w21) + 4.f * (-w02 - w12 - w22)) / 144.f;
    out14 = (-w00 - w10 - w20 + 2.f * (w01 + w11 + w21) + 4.f * (-w02 - w12 - w22)) / 144.f;
    out15 = (-w02 - w12 - w22) / 6.f;

    // Row 2
    out20 = (-w00 + w10 - w20) / 24.f;
    out21 = (w00 - w10 + w20 + w01 - w11 + w21 + w02 - w12 + w22) / 36.f;
    out22 = (w00 - w10 + w20 - w01 + w11 - w21 + w02 - w12 + w22) / 36.f;
    out23 = (-w00 + w10 - w20 + 2.f * (-w01 + w11 - w21) + 4.f * (-w02 + w12 - w22)) / 144.f;
    out24 = (-w00 + w10 - w20 + 2.f * (w01 - w11 + w21) + 4.f * (-w02 + w12 - w22)) / 144.f;
    out25 = (-w02 + w12 - w22) / 6.f;

    // Row 3
    out30 = (w00 + 2.f * w10 + 4.f * w20) / 96.f;
    out31 = (-w00 - 2.f * w10 - 4.f * w20 - w01 - 2.f * w11 - 4.f * w21 - w02 - 2.f * w12 - 4.f * w22) / 144.f;
    out32 = (-w00 - 2.f * w10 - 4.f * w20 + w01 + 2.f * w11 + 4.f * w21 - w02 - 2.f * w12 - 4.f * w22) / 144.f;
    out33 = ((w00 + 2.f * w10 + 4.f * w20) + 2.f * (w01 + 2.f * w11 + 4.f * w21) + 4.f * (w02 + 2.f * w12 + 4.f * w22)) / 576.f;
    out34 = ((w00 + 2.f * w10 + 4.f * w20) + 2.f * (-w01 - 2.f * w11 - 4.f * w21) + 4.f * (w02 + 2.f * w12 + 4.f * w22)) / 576.f;
    out35 = (w02 + 2.f * w12 + 4.f * w22) / 24.f;

    // Row 4
    out40 = (w00 - 2.f * w10 + 4.f * w20) / 96.f;
    out41 = (-w00 + 2.f * w10 - 4.f * w20 - w01 + 2.f * w11 - 4.f * w21 - w02 + 2.f * w12 - 4.f * w22) / 144.f;
    out42 = (-w00 + 2.f * w10 - 4.f * w20 + w01 - 2.f * w11 + 4.f * w21 - w02 + 2.f * w12 - 4.f * w22) / 144.f;
    out43 = ((w00 - 2.f * w10 + 4.f * w20) + 2.f * (w01 - 2.f * w11 + 4.f * w21) + 4.f * (w02 - 2.f * w12 + 4.f * w22)) / 576.f;
    out44 = ((w00 - 2.f * w10 + 4.f * w20) + 2.f * (-w01 + 2.f * w11 - 4.f * w21) + 4.f * (w02 - 2.f * w12 + 4.f * w22)) / 576.f;
    out45 = (w02 - 2.f * w12 + 4.f * w22) / 24.f;

    // Row 5
    out50 = (w20) / 4.f;
    out51 = (-w20 - w21 - w22) / 6.f;
    out52 = (-w20 + w21 - w22) / 6.f;
    out53 = (w20 + 2.f * w21 + 4.f * w22) / 24.f;
    out54 = (w20 - 2.f * w21 + 4.f * w22) / 24.f;
    out55 = (w22);

    int x0 = get_global_id(2); // idx filter
    int y0 = get_global_id(0); // idx channel

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x0 * dst_stride_x + y0 * dst_stride_y;

    // Store the values across the channels
    *(__global float *)(dst_addr + 0 * dst_stride_z)  = out00;
    *(__global float *)(dst_addr + 1 * dst_stride_z)  = out01;
    *(__global float *)(dst_addr + 2 * dst_stride_z)  = out02;
    *(__global float *)(dst_addr + 3 * dst_stride_z)  = out03;
    *(__global float *)(dst_addr + 4 * dst_stride_z)  = out04;
    *(__global float *)(dst_addr + 5 * dst_stride_z)  = out05;
    *(__global float *)(dst_addr + 6 * dst_stride_z)  = out10;
    *(__global float *)(dst_addr + 7 * dst_stride_z)  = out11;
    *(__global float *)(dst_addr + 8 * dst_stride_z)  = out12;
    *(__global float *)(dst_addr + 9 * dst_stride_z)  = out13;
    *(__global float *)(dst_addr + 10 * dst_stride_z) = out14;
    *(__global float *)(dst_addr + 11 * dst_stride_z) = out15;
    *(__global float *)(dst_addr + 12 * dst_stride_z) = out20;
    *(__global float *)(dst_addr + 13 * dst_stride_z) = out21;
    *(__global float *)(dst_addr + 14 * dst_stride_z) = out22;
    *(__global float *)(dst_addr + 15 * dst_stride_z) = out23;
    *(__global float *)(dst_addr + 16 * dst_stride_z) = out24;
    *(__global float *)(dst_addr + 17 * dst_stride_z) = out25;
    *(__global float *)(dst_addr + 18 * dst_stride_z) = out30;
    *(__global float *)(dst_addr + 19 * dst_stride_z) = out31;
    *(__global float *)(dst_addr + 20 * dst_stride_z) = out32;
    *(__global float *)(dst_addr + 21 * dst_stride_z) = out33;
    *(__global float *)(dst_addr + 22 * dst_stride_z) = out34;
    *(__global float *)(dst_addr + 23 * dst_stride_z) = out35;
    *(__global float *)(dst_addr + 24 * dst_stride_z) = out40;
    *(__global float *)(dst_addr + 25 * dst_stride_z) = out41;
    *(__global float *)(dst_addr + 26 * dst_stride_z) = out42;
    *(__global float *)(dst_addr + 27 * dst_stride_z) = out43;
    *(__global float *)(dst_addr + 28 * dst_stride_z) = out44;
    *(__global float *)(dst_addr + 29 * dst_stride_z) = out45;
    *(__global float *)(dst_addr + 30 * dst_stride_z) = out50;
    *(__global float *)(dst_addr + 31 * dst_stride_z) = out51;
    *(__global float *)(dst_addr + 32 * dst_stride_z) = out52;
    *(__global float *)(dst_addr + 33 * dst_stride_z) = out53;
    *(__global float *)(dst_addr + 34 * dst_stride_z) = out54;
    *(__global float *)(dst_addr + 35 * dst_stride_z) = out55;
}
/** This OpenCL kernel performs Winograd filter transform 5x5 when the data layout is NCHW and the output tile is 4x4
 *
 * @note In order to correctly split the input tensor in batches, its dimension across the Z axis (channels for NCHW, height for NHWC) must be passed at compile time using -DSRC_DIM_Z: e.g. -DSRC_DIM_Z=64
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
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, SRC_DIM_Z);

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
    out5.s3 = ((16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) + 2.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s1 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) + 8.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 16200.f;
    out5.s4 = ((16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) - 2.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s1 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) - 8.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 16200.f;
    out5.s5 = (16.f * (16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) + 8.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s1 + w40.s1) + 4.f *
               (16.f * w00.s2 + 8.f * w10.s2 + 4.f * w20.s2 + 2.f * w30.s2 + w40.s2) + 2.f * (16.f * w00.s3 + 8.f * w10.s3 + 4.f * w20.s3 + 2.f * w30.s3 + w40.s3) +
               (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41)) / 32400.f;
    out5.s6 = (16.f * (16.f * w00.s0 + 8.f * w10.s0 + 4.f * w20.s0 + 2.f * w30.s0 + w40.s0) - 8.f * (16.f * w00.s1 + 8.f * w10.s1 + 4.f * w20.s1 + 2.f * w30.s1 + w40.s1) + 4.f *
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
    out6.s3 = ((16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) + 2.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s1 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) + 8.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 16200.f;
    out6.s4 = ((16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) - 2.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s1 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) - 8.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) + 16.f *
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 16200.f;
    out6.s5 = (16.f * (16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) + 8.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s1 + w40.s1) + 4.f *
               (16.f * w00.s2 - 8.f * w10.s2 + 4.f * w20.s2 - 2.f * w30.s2 + w40.s2) + 2.f * (16.f * w00.s3 - 8.f * w10.s3 + 4.f * w20.s3 - 2.f * w30.s3 + w40.s3) +
               (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41)) / 32400.f;
    out6.s6 = (16.f * (16.f * w00.s0 - 8.f * w10.s0 + 4.f * w20.s0 - 2.f * w30.s0 + w40.s0) - 8.f * (16.f * w00.s1 - 8.f * w10.s1 + 4.f * w20.s1 - 2.f * w30.s1 + w40.s1) + 4.f *
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
    int x0 = z / SRC_DIM_Z; // idx filter
    int y0 = z % SRC_DIM_Z; // idx channel

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

/** This OpenCL kernel performs Winograd filter transform 5x5 when the data layout is NHWC and the output tile is 4x4
 *
 * @note In order to correctly split the input tensor in batches, its dimension across the Z axis (channels for NCHW, height for NHWC) must be passed at compile time using -DSRC_DIM_Z: e.g. -DSRC_DIM_Z=64
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
__kernel void winograd_filter_transform_4x4_5x5_nhwc(
    TENSOR4D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    Tensor4D src = CONVERT_TO_TENSOR4D_STRUCT(src, SRC_DIM_Z);

    const __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + get_global_id(0) * sizeof(float) + get_global_id(1) * src_step_y + get_global_id(2) * src_step_w;

    // Load the values from the input tensor
    float w00 = *((__global float *)(src_addr + 0 * src_stride_z + 0 * src_stride_y));
    float w01 = *((__global float *)(src_addr + 0 * src_stride_z + 1 * src_stride_y));
    float w02 = *((__global float *)(src_addr + 0 * src_stride_z + 2 * src_stride_y));
    float w03 = *((__global float *)(src_addr + 0 * src_stride_z + 3 * src_stride_y));
    float w04 = *((__global float *)(src_addr + 0 * src_stride_z + 4 * src_stride_y));
    float w10 = *((__global float *)(src_addr + 1 * src_stride_z + 0 * src_stride_y));
    float w11 = *((__global float *)(src_addr + 1 * src_stride_z + 1 * src_stride_y));
    float w12 = *((__global float *)(src_addr + 1 * src_stride_z + 2 * src_stride_y));
    float w13 = *((__global float *)(src_addr + 1 * src_stride_z + 3 * src_stride_y));
    float w14 = *((__global float *)(src_addr + 1 * src_stride_z + 4 * src_stride_y));
    float w20 = *((__global float *)(src_addr + 2 * src_stride_z + 0 * src_stride_y));
    float w21 = *((__global float *)(src_addr + 2 * src_stride_z + 1 * src_stride_y));
    float w22 = *((__global float *)(src_addr + 2 * src_stride_z + 2 * src_stride_y));
    float w23 = *((__global float *)(src_addr + 2 * src_stride_z + 3 * src_stride_y));
    float w24 = *((__global float *)(src_addr + 2 * src_stride_z + 4 * src_stride_y));
    float w30 = *((__global float *)(src_addr + 3 * src_stride_z + 0 * src_stride_y));
    float w31 = *((__global float *)(src_addr + 3 * src_stride_z + 1 * src_stride_y));
    float w32 = *((__global float *)(src_addr + 3 * src_stride_z + 2 * src_stride_y));
    float w33 = *((__global float *)(src_addr + 3 * src_stride_z + 3 * src_stride_y));
    float w34 = *((__global float *)(src_addr + 3 * src_stride_z + 4 * src_stride_y));
    float w40 = *((__global float *)(src_addr + 4 * src_stride_z + 0 * src_stride_y));
    float w41 = *((__global float *)(src_addr + 4 * src_stride_z + 1 * src_stride_y));
    float w42 = *((__global float *)(src_addr + 4 * src_stride_z + 2 * src_stride_y));
    float w43 = *((__global float *)(src_addr + 4 * src_stride_z + 3 * src_stride_y));
    float w44 = *((__global float *)(src_addr + 4 * src_stride_z + 4 * src_stride_y));

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
    out0.s0 = w00;
    out0.s1 = -2.f * (w00 + w01 + w02 + w03 + w04) / 9.f;
    out0.s2 = -2.f * (w00 - w01 + w02 - w03 + w04) / 9.f;
    out0.s3 = (w00 + 2.f * w01 + 4.f * w02 + 8.f * w03 + 16.f * w04) / 90.f;
    out0.s4 = (w00 - 2.f * w01 + 4.f * w02 - 8.f * w03 + 16.f * w04) / 90.f;
    out0.s5 = (16.f * w00 + 8.f * w01 + 4.f * w02 + 2.f * w03 + w04) / 180.f;
    out0.s6 = (16.f * w00 - 8.f * w01 + 4.f * w02 - 2.f * w03 + w04) / 180.f;
    out0.s7 = w04;

    // Row 1
    out1.s0 = -2.f * (w00 + w10 + w20 + w30 + w40) / 9.f;
    out1.s1 = 4.f * ((w00 + w10 + w20 + w30 + w40) + (w01 + w11 + w21 + w31 + w41) + (w02 + w12 + w22 + w32 + w42) + (w03 + w13 + w23 + w33 + w43) + (w04 + w14 + w24 + w34 + w44)) / 81.f;
    out1.s2 = 4.f * ((w00 + w10 + w20 + w30 + w40) - (w01 + w11 + w21 + w31 + w41) + (w02 + w12 + w22 + w32 + w42) - (w03 + w13 + w23 + w33 + w43) + (w04 + w14 + w24 + w34 + w44)) / 81.f;
    out1.s3 = -((w00 + w10 + w20 + w30 + w40) + 2.f * (w01 + w11 + w21 + w31 + w41) + 4.f * (w02 + w12 + w22 + w32 + w42) + 8.f * (w03 + w13 + w23 + w33 + w43) + 16.f *
                (w04 + w14 + w24 + w34 + w44)) / 405.f;
    out1.s4 = -((w00 + w10 + w20 + w30 + w40) - 2.f * (w01 + w11 + w21 + w31 + w41) + 4.f * (w02 + w12 + w22 + w32 + w42) - 8.f * (w03 + w13 + w23 + w33 + w43) + 16.f *
                (w04 + w14 + w24 + w34 + w44)) / 405.f;
    out1.s5 = -(16.f * (w00 + w10 + w20 + w30 + w40) + 8.f * (w01 + w11 + w21 + w31 + w41) + 4.f * (w02 + w12 + w22 + w32 + w42) + 2.f * (w03 + w13 + w23 + w33 + w43) +
                (w04 + w14 + w24 + w34 + w44)) / 810.f;
    out1.s6 = -(16.f * (w00 + w10 + w20 + w30 + w40) - 8.f * (w01 + w11 + w21 + w31 + w41) + 4.f * (w02 + w12 + w22 + w32 + w42) - 2.f * (w03 + w13 + w23 + w33 + w43) +
                (w04 + w14 + w24 + w34 + w44)) / 810.f;
    out1.s7 = -2.f * (w04 + w14 + w24 + w34 + w44) / 9.f;

    // Row 2
    out2.s0 = -2.f * (w00 - w10 + w20 - w30 + w40) / 9.f;
    out2.s1 = 4.f * ((w00 - w10 + w20 - w30 + w40) + (w01 - w11 + w21 - w31 + w41) + (w02 - w12 + w22 - w32 + w42) + (w03 - w13 + w23 - w33 + w43) + (w04 - w14 + w24 - w34 + w44)) / 81.f;
    out2.s2 = 4.f * ((w00 - w10 + w20 - w30 + w40) - (w01 - w11 + w21 - w31 + w41) + (w02 - w12 + w22 - w32 + w42) - (w03 - w13 + w23 - w33 + w43) + (w04 - w14 + w24 - w34 + w44)) / 81.f;
    out2.s3 = -((w00 - w10 + w20 - w30 + w40) + 2.f * (w01 - w11 + w21 - w31 + w41) + 4.f * (w02 - w12 + w22 - w32 + w42) + 8.f * (w03 - w13 + w23 - w33 + w43) + 16.f *
                (w04 - w14 + w24 - w34 + w44)) / 405.f;
    out2.s4 = -((w00 - w10 + w20 - w30 + w40) - 2.f * (w01 - w11 + w21 - w31 + w41) + 4.f * (w02 - w12 + w22 - w32 + w42) - 8.f * (w03 - w13 + w23 - w33 + w43) + 16.f *
                (w04 - w14 + w24 - w34 + w44)) / 405.f;
    out2.s5 = -(16.f * (w00 - w10 + w20 - w30 + w40) + 8.f * (w01 - w11 + w21 - w31 + w41) + 4.f * (w02 - w12 + w22 - w32 + w42) + 2.f * (w03 - w13 + w23 - w33 + w43) +
                (w04 - w14 + w24 - w34 + w44)) / 810.f;
    out2.s6 = -(16.f * (w00 - w10 + w20 - w30 + w40) - 8.f * (w01 - w11 + w21 - w31 + w41) + 4.f * (w02 - w12 + w22 - w32 + w42) - 2.f * (w03 - w13 + w23 - w33 + w43) +
                (w04 - w14 + w24 - w34 + w44)) / 810.f;
    out2.s7 = -2.f * (w04 - w14 + w24 - w34 + w44) / 9.f;

    // Row 3
    out3.s0 = (w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) / 90.f;
    out3.s1 = -((w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) + (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) + (w02 + 2.f * w12 + 4.f * w22 + 8.f * w32 + 16.f * w42) +
                (w03 + 2.f * w13 + 4.f * w23 + 8.f * w33 + 16.f * w43) + (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44)) / 405.f;
    out3.s2 = -((w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) - (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) + (w02 + 2.f * w12 + 4.f * w22 + 8.f * w32 + 16.f * w42) -
                (w03 + 2.f * w13 + 4.f * w23 + 8.f * w33 + 16.f * w43) + (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44)) / 405.f;
    out3.s3 = ((w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) + 2.f * (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) + 4.f * (w02 + 2.f * w12 + 4.f * w22 + 8.f * w32 + 16.f * w42) + 8.f
               * (w03 + 2.f * w13 + 4.f * w23 + 8.f * w33 + 16.f * w43) + 16.f * (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44)) / 8100.f;
    out3.s4 = ((w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) - 2.f * (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) + 4.f * (w02 + 2.f * w12 + 4.f * w22 + 8.f * w32 + 16.f * w42) - 8.f
               * (w03 + 2.f * w13 + 4.f * w23 + 8.f * w33 + 16.f * w43) + 16.f * (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44)) / 8100.f;
    out3.s5 = (16.f * (w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) + 8.f * (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) + 4.f *
               (w02 + 2.f * w12 + 4.f * w22 + 8.f * w32 + 16.f * w42) + 2.f * (w03 + 2.f * w13 + 4.f * w23 + 8.f * w33 + 16.f * w43) + (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44)) / 16200.f;
    out3.s6 = (16.f * (w00 + 2.f * w10 + 4.f * w20 + 8.f * w30 + 16.f * w40) - 8.f * (w01 + 2.f * w11 + 4.f * w21 + 8.f * w31 + 16.f * w41) + 4.f *
               (w02 + 2.f * w12 + 4.f * w22 + 8.f * w32 + 16.f * w42) - 2.f * (w03 + 2.f * w13 + 4.f * w23 + 8.f * w33 + 16.f * w43) + (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44)) / 16200.f;
    out3.s7 = (w04 + 2.f * w14 + 4.f * w24 + 8.f * w34 + 16.f * w44) / 90.f;

    // Row 4
    out4.s0 = (w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) / 90.f;
    out4.s1 = -((w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) + (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) + (w02 - 2.f * w12 + 4.f * w22 - 8.f * w32 + 16.f * w42) +
                (w03 - 2.f * w13 + 4.f * w23 - 8.f * w33 + 16.f * w43) + (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44)) / 405.f;
    out4.s2 = -((w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) - (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) + (w02 - 2.f * w12 + 4.f * w22 - 8.f * w32 + 16.f * w42) -
                (w03 - 2.f * w13 + 4.f * w23 - 8.f * w33 + 16.f * w43) + (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44)) / 405.f;
    out4.s3 = ((w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) + 2.f * (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) + 4.f * (w02 - 2.f * w12 + 4.f * w22 - 8.f * w32 + 16.f * w42) + 8.f
               * (w03 - 2.f * w13 + 4.f * w23 - 8.f * w33 + 16.f * w43) + 16.f * (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44)) / 8100.f;
    out4.s4 = ((w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) - 2.f * (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) + 4.f * (w02 - 2.f * w12 + 4.f * w22 - 8.f * w32 + 16.f * w42) - 8.f
               * (w03 - 2.f * w13 + 4.f * w23 - 8.f * w33 + 16.f * w43) + 16.f * (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44)) / 8100.f;
    out4.s5 = (16.f * (w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) + 8.f * (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) + 4.f *
               (w02 - 2.f * w12 + 4.f * w22 - 8.f * w32 + 16.f * w42) + 2.f * (w03 - 2.f * w13 + 4.f * w23 - 8.f * w33 + 16.f * w43) + (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44)) / 16200.f;
    out4.s6 = (16.f * (w00 - 2.f * w10 + 4.f * w20 - 8.f * w30 + 16.f * w40) - 8.f * (w01 - 2.f * w11 + 4.f * w21 - 8.f * w31 + 16.f * w41) + 4.f *
               (w02 - 2.f * w12 + 4.f * w22 - 8.f * w32 + 16.f * w42) - 2.f * (w03 - 2.f * w13 + 4.f * w23 - 8.f * w33 + 16.f * w43) + (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44)) / 16200.f;
    out4.s7 = (w04 - 2.f * w14 + 4.f * w24 - 8.f * w34 + 16.f * w44) / 90.f;

    // Row 5
    out5.s0 = (16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) / 180.f;
    out5.s1 = -((16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) + (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) + (16.f * w02 + 8.f * w12 + 4.f * w22 + 2.f * w32 + w42) +
                (16.f * w03 + 8.f * w13 + 4.f * w23 + 2.f * w33 + w43) + (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44)) / 810.f;
    out5.s2 = -((16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) - (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) + (16.f * w02 + 8.f * w12 + 4.f * w22 + 2.f * w32 + w42) -
                (16.f * w03 + 8.f * w13 + 4.f * w23 + 2.f * w33 + w43) + (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44)) / 810.f;
    out5.s3 = ((16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) + 2.f * (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) + 4.f * (16.f * w02 + 8.f * w12 + 4.f * w22 + 2.f * w32 + w42) + 8.f
               * (16.f * w03 + 8.f * w13 + 4.f * w23 + 2.f * w33 + w43) + 16.f * (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44)) / 16200.f;
    out5.s4 = ((16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) - 2.f * (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) + 4.f * (16.f * w02 + 8.f * w12 + 4.f * w22 + 2.f * w32 + w42) - 8.f
               * (16.f * w03 + 8.f * w13 + 4.f * w23 + 2.f * w33 + w43) + 16.f * (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44)) / 16200.f;
    out5.s5 = (16.f * (16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) + 8.f * (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) + 4.f *
               (16.f * w02 + 8.f * w12 + 4.f * w22 + 2.f * w32 + w42) + 2.f * (16.f * w03 + 8.f * w13 + 4.f * w23 + 2.f * w33 + w43) + (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44)) / 32400.f;
    out5.s6 = (16.f * (16.f * w00 + 8.f * w10 + 4.f * w20 + 2.f * w30 + w40) - 8.f * (16.f * w01 + 8.f * w11 + 4.f * w21 + 2.f * w31 + w41) + 4.f *
               (16.f * w02 + 8.f * w12 + 4.f * w22 + 2.f * w32 + w42) - 2.f * (16.f * w03 + 8.f * w13 + 4.f * w23 + 2.f * w33 + w43) + (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44)) / 32400.f;
    out5.s7 = (16.f * w04 + 8.f * w14 + 4.f * w24 + 2.f * w34 + w44) / 180.f;

    // Row 6
    out6.s0 = (16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) / 180.f;
    out6.s1 = -((16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) + (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) + (16.f * w02 - 8.f * w12 + 4.f * w22 - 2.f * w32 + w42) +
                (16.f * w03 - 8.f * w13 + 4.f * w23 - 2.f * w33 + w43) + (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44)) / 810.f;
    out6.s2 = -((16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) - (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) + (16.f * w02 - 8.f * w12 + 4.f * w22 - 2.f * w32 + w42) -
                (16.f * w03 - 8.f * w13 + 4.f * w23 - 2.f * w33 + w43) + (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44)) / 810.f;
    out6.s3 = ((16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) + 2.f * (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) + 4.f * (16.f * w02 - 8.f * w12 + 4.f * w22 - 2.f * w32 + w42) + 8.f
               * (16.f * w03 - 8.f * w13 + 4.f * w23 - 2.f * w33 + w43) + 16.f * (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44)) / 16200.f;
    out6.s4 = ((16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) - 2.f * (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) + 4.f * (16.f * w02 - 8.f * w12 + 4.f * w22 - 2.f * w32 + w42) - 8.f
               * (16.f * w03 - 8.f * w13 + 4.f * w23 - 2.f * w33 + w43) + 16.f * (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44)) / 16200.f;
    out6.s5 = (16.f * (16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) + 8.f * (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) + 4.f *
               (16.f * w02 - 8.f * w12 + 4.f * w22 - 2.f * w32 + w42) + 2.f * (16.f * w03 - 8.f * w13 + 4.f * w23 - 2.f * w33 + w43) + (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44)) / 32400.f;
    out6.s6 = (16.f * (16.f * w00 - 8.f * w10 + 4.f * w20 - 2.f * w30 + w40) - 8.f * (16.f * w01 - 8.f * w11 + 4.f * w21 - 2.f * w31 + w41) + 4.f *
               (16.f * w02 - 8.f * w12 + 4.f * w22 - 2.f * w32 + w42) - 2.f * (16.f * w03 - 8.f * w13 + 4.f * w23 - 2.f * w33 + w43) + (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44)) / 32400.f;
    out6.s7 = (16.f * w04 - 8.f * w14 + 4.f * w24 - 2.f * w34 + w44) / 180.f;

    // Row 7
    out7.s0 = w40;
    out7.s1 = -2.f * (w40 + w41 + w42 + w43 + w44) / 9.f;
    out7.s2 = -2.f * (w40 - w41 + w42 - w43 + w44) / 9.f;
    out7.s3 = (w40 + 2.f * w41 + 4.f * w42 + 8.f * w43 + 16.f * w44) / 90.f;
    out7.s4 = (w40 - 2.f * w41 + 4.f * w42 - 8.f * w43 + 16.f * w44) / 90.f;
    out7.s5 = (16.f * w40 + 8.f * w41 + 4.f * w42 + 2.f * w43 + w44) / 180.f;
    out7.s6 = (16.f * w40 - 8.f * w41 + 4.f * w42 - 2.f * w43 + w44) / 180.f;
    out7.s7 = w44;

    int x0 = get_global_id(2); // idx filter
    int y0 = get_global_id(0); // idx channel

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x0 * sizeof(float) + y0 * dst_stride_y;

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
#endif // defined(SRC_DIM_Z)

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

/** This OpenCL kernel computes the input transform when the output tile is 4x4, the filter size 3x3 and the data layout is NCHW
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
__kernel void winograd_input_transform_4x4_3x3_stepz1_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    // Compute input address
    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * 4 * src_stride_x + y * 4 * src_stride_y + z * src_stride_z;

    src_addr = src_addr - ((int)PAD_LEFT * src_stride_x) - ((int)PAD_TOP * src_stride_y);

    // Row4
    float4 d40 = vload4(0, (__global float *)(src_addr + 4 * src_stride_y));
    float2 d41 = vload2(2, (__global float *)(src_addr + 4 * src_stride_y));

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

    // Row0
    float4 d00 = vload4(0, (__global float *)(src_addr + 0 * src_stride_y));
    float2 d01 = vload2(2, (__global float *)(src_addr + 0 * src_stride_y));

    // Row2
    float4 d20 = vload4(0, (__global float *)(src_addr + 2 * src_stride_y));
    float2 d21 = vload2(2, (__global float *)(src_addr + 2 * src_stride_y));

    // Compute destination address
    __global float *dst_addr = (__global float *)(dst_ptr + dst_offset_first_element_in_bytes + z * dst_stride_x + (x + y * (int)NUM_TILES_X) * dst_stride_y);

    uint dst_plane_stride = dst_stride_z / sizeof(float);

    float out0  = k0;
    float out1  = k1;
    float out2  = k2;
    float out3  = k3;
    float out4  = k4;
    float out5  = k5;
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
    out0 += 16.0f * d00.s0 - 20.0f * d00.s2 - 20.0f * d20.s0 + 25.0f * d20.s2 + 4.0f * d01.s0 - 5.0f * d21.s0;
    out1 += -16.0f * d00.s1 - 16.0f * d00.s2 + 4.0f * d00.s3 + 20.0f * d20.s1 + 20.0f * d20.s2 - 5.0f * d20.s3 + 4.0f * d01.s0 - 5.0f * d21.s0;
    out2 += 16.0f * d00.s1 - 16.0f * d00.s2 - 4.0f * d00.s3 - 20.0f * d20.s1 + 20.0f * d20.s2 + 5.0f * d20.s3 + 4.0f * d01.s0 - 5.0f * d21.s0;
    out3 += -8.0f * d00.s1 - 4.0f * d00.s2 + 8.0f * d00.s3 + 10.0f * d20.s1 + 5.0f * d20.s2 - 10.0f * d20.s3 + 4.0f * d01.s0 - 5.0f * d21.s0;
    out4 += 8.0f * d00.s1 - 4.0f * d00.s2 - 8.0f * d00.s3 - 10.0f * d20.s1 + 5.0f * d20.s2 + 10.0f * d20.s3 + 4.0f * d01.s0 - 5.0f * d21.s0;
    out5 += 16.0f * d00.s1 - 20.0f * d00.s3 - 20.0f * d20.s1 + 4.0f * d01.s1 + 25.0f * d20.s3 - 5.0f * d21.s1;

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
}

#if defined(SRC_DIM_1) && defined(SRC_DIM_2)
/** This OpenCL kernel computes the input transform when the output tile is 4x4, the filter size 3x3 and the data layout is NHWC
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
__kernel void winograd_input_transform_4x4_3x3_stepz1_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst))
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);

    __global uchar *src_addr = src_ptr + src_offset_first_element_in_bytes + x * src_stride_x;

    // Clamp coordinates. This clamp is valid for all rows
    int4 y_coord0 = (int4)(y * 4) + (int4)(0, 1, 2, 3) - (int4)PAD_LEFT;
    int2 y_coord1 = (int2)(y * 4) + (int2)(4, 5) - (int2)PAD_LEFT;
    y_coord0      = clamp(y_coord0, -1, SRC_DIM_1);
    y_coord1      = clamp(y_coord1, -1, SRC_DIM_1);

    // Row4
    int z_coord = (z * 4) - PAD_TOP + 4;

    // If z < 0, set y to -1
    int4 valid_y0 = select(y_coord0, -1, (int4)z_coord < 0);
    int2 valid_y1 = select(y_coord1, -1, (int2)z_coord < 0);
    // If z >= SRC_DIM_2, set y to SRC_DIM_2
    valid_y0 = select(valid_y0, SRC_DIM_1, (int4)z_coord >= SRC_DIM_2);
    valid_y1 = select(valid_y1, SRC_DIM_1, (int2)z_coord >= SRC_DIM_2);

    // Clamp z coordinate
    z_coord = clamp(z_coord, 0, SRC_DIM_2 - 1);

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

    // Row0
    z_coord = (z * 4) - PAD_TOP + 0;

#if PAD_TOP != 0
    valid_y0 = select(y_coord0, -1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, -1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, SRC_DIM_1, (int4)z_coord >= SRC_DIM_2);
    valid_y1 = select(valid_y1, SRC_DIM_1, (int2)z_coord >= SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, SRC_DIM_2 - 1);
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

    // Row2
    z_coord  = (z * 4) - PAD_TOP + 2;
    valid_y0 = select(y_coord0, -1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, -1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, SRC_DIM_1, (int4)z_coord >= SRC_DIM_2);
    valid_y1 = select(valid_y1, SRC_DIM_1, (int2)z_coord >= SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, SRC_DIM_2 - 1);

    float d20 = *(__global float *)(src_addr + valid_y0.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d21 = *(__global float *)(src_addr + valid_y0.s1 * (int)src_stride_y + z_coord * src_stride_z);
    float d22 = *(__global float *)(src_addr + valid_y0.s2 * (int)src_stride_y + z_coord * src_stride_z);
    float d23 = *(__global float *)(src_addr + valid_y0.s3 * (int)src_stride_y + z_coord * src_stride_z);
    float d24 = *(__global float *)(src_addr + valid_y1.s0 * (int)src_stride_y + z_coord * src_stride_z);
    float d25 = *(__global float *)(src_addr + valid_y1.s1 * (int)src_stride_y + z_coord * src_stride_z);

    // Compute destination address
    __global float *dst_addr = (__global float *)(dst_ptr + dst_offset_first_element_in_bytes + x * dst_stride_x + (y + z * (int)NUM_TILES_X) * dst_stride_y);

    uint dst_plane_stride = dst_stride_z / sizeof(float);

    float out0  = k0;
    float out1  = k1;
    float out2  = k2;
    float out3  = k3;
    float out4  = k4;
    float out5  = k5;
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
    out0 += 16.0f * d00 - 20.0f * d02 - 20.0f * d20 + 25.0f * d22 + 4.0f * d04 - 5.0f * d24;
    out1 += -16.0f * d01 - 16.0f * d02 + 4.0f * d03 + 20.0f * d21 + 20.0f * d22 - 5.0f * d23 + 4.0f * d04 - 5.0f * d24;
    out2 += 16.0f * d01 - 16.0f * d02 - 4.0f * d03 - 20.0f * d21 + 20.0f * d22 + 5.0f * d23 + 4.0f * d04 - 5.0f * d24;
    out3 += -8.0f * d01 - 4.0f * d02 + 8.0f * d03 + 10.0f * d21 + 5.0f * d22 - 10.0f * d23 + 4.0f * d04 - 5.0f * d24;
    out4 += 8.0f * d01 - 4.0f * d02 - 8.0f * d03 - 10.0f * d21 + 5.0f * d22 + 10.0f * d23 + 4.0f * d04 - 5.0f * d24;
    out5 += 16.0f * d01 - 20.0f * d03 - 20.0f * d21 + 4.0f * d05 + 25.0f * d23 - 5.0f * d25;

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

    // Row1
    z_coord = (z * 4) - PAD_TOP + 1;
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
    z_coord  = (z * 4) - PAD_TOP + 3;
    valid_y0 = select(y_coord0, -1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, -1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, SRC_DIM_1, (int4)z_coord >= SRC_DIM_2);
    valid_y1 = select(valid_y1, SRC_DIM_1, (int2)z_coord >= SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, SRC_DIM_2 - 1);
    z_coord  = clamp(z_coord, 0, SRC_DIM_2 - 1);

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
    z_coord  = (z * 4) - PAD_TOP + 5;
    valid_y0 = select(y_coord0, -1, (int4)z_coord < 0);
    valid_y1 = select(y_coord1, -1, (int2)z_coord < 0);
    valid_y0 = select(valid_y0, SRC_DIM_1, (int4)z_coord >= SRC_DIM_2);
    valid_y1 = select(valid_y1, SRC_DIM_1, (int2)z_coord >= SRC_DIM_2);
    z_coord  = clamp(z_coord, 0, SRC_DIM_2 - 1);
    z_coord  = clamp(z_coord, 0, SRC_DIM_2 - 1);

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
}

#endif /* defined(SRC_DIM_1) && defined(SRC_DIM_2) */

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
/** This OpenCL kernel performs Winograd output transform when the output tile is 2x2, the filter size 3x3 and the data layout is NCHW
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

/** This OpenCL kernel performs Winograd output transform when the output tile is 4x4, the filter size 3x3 and the data layout is NCHW
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
__kernel void winograd_output_transform_4x4_3x3_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(bias)
#endif // defined(HAS_BIAS)
)
{
    // Each thread stores a 4x4 tile
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const __global uchar *src_addr = tensor3D_offset(&src, 0, 0, 0);

    // Load the values across the 36 channels to compose the 6x6 tile
    float d00 = *((__global float *)(src_addr + 0 * src_stride_z));
    float d01 = *((__global float *)(src_addr + 1 * src_stride_z));
    float d02 = *((__global float *)(src_addr + 2 * src_stride_z));
    float d03 = *((__global float *)(src_addr + 3 * src_stride_z));
    float d04 = *((__global float *)(src_addr + 4 * src_stride_z));
    float d05 = *((__global float *)(src_addr + 5 * src_stride_z));

    float d10 = *((__global float *)(src_addr + 6 * src_stride_z));
    float d11 = *((__global float *)(src_addr + 7 * src_stride_z));
    float d12 = *((__global float *)(src_addr + 8 * src_stride_z));
    float d13 = *((__global float *)(src_addr + 9 * src_stride_z));
    float d14 = *((__global float *)(src_addr + 10 * src_stride_z));
    float d15 = *((__global float *)(src_addr + 11 * src_stride_z));

    float d20 = *((__global float *)(src_addr + 12 * src_stride_z));
    float d21 = *((__global float *)(src_addr + 13 * src_stride_z));
    float d22 = *((__global float *)(src_addr + 14 * src_stride_z));
    float d23 = *((__global float *)(src_addr + 15 * src_stride_z));
    float d24 = *((__global float *)(src_addr + 16 * src_stride_z));
    float d25 = *((__global float *)(src_addr + 17 * src_stride_z));

    float d30 = *((__global float *)(src_addr + 18 * src_stride_z));
    float d31 = *((__global float *)(src_addr + 19 * src_stride_z));
    float d32 = *((__global float *)(src_addr + 20 * src_stride_z));
    float d33 = *((__global float *)(src_addr + 21 * src_stride_z));
    float d34 = *((__global float *)(src_addr + 22 * src_stride_z));
    float d35 = *((__global float *)(src_addr + 23 * src_stride_z));

    float d40 = *((__global float *)(src_addr + 24 * src_stride_z));
    float d41 = *((__global float *)(src_addr + 25 * src_stride_z));
    float d42 = *((__global float *)(src_addr + 26 * src_stride_z));
    float d43 = *((__global float *)(src_addr + 27 * src_stride_z));
    float d44 = *((__global float *)(src_addr + 28 * src_stride_z));
    float d45 = *((__global float *)(src_addr + 29 * src_stride_z));

    float d50 = *((__global float *)(src_addr + 30 * src_stride_z));
    float d51 = *((__global float *)(src_addr + 31 * src_stride_z));
    float d52 = *((__global float *)(src_addr + 32 * src_stride_z));
    float d53 = *((__global float *)(src_addr + 33 * src_stride_z));
    float d54 = *((__global float *)(src_addr + 34 * src_stride_z));
    float d55 = *((__global float *)(src_addr + 35 * src_stride_z));

    // Compute out00, out01, out02 and out03
    float out00 = d01 + d21 + d41 + d11 + d31;
    float out01 = d01 + d21 + d41 + d11 + d31;
    float out02 = d01 + d21 + d41 + d11 + d31;
    float out03 = d01 + d21 + d41 + d11 + d31;

    float k0 = d03 + d04 + d13 + d14 + d23 + d24 + d33 + d34 + d43 + d44;
    float k1 = 2.0f * d03 - 2.0f * d04 + 2.0f * d13 - 2.0f * d14 + 2.0f * d23 - 2.0f * d24 + 2.0f * d33 - 2.0f * d34 + 2.0f * d43 - 2.0f * d44;

    out00 += k0 + d00 + d02 + d10 + d12 + d20 + d22 + d30 + d32 + d40 + d42;
    out01 += k1 - d02 - d12 - d22 - d32 - d42;
    out02 += 4.0f * k0 + d02 + d12 + d22 + d32 + d42;
    out03 += 4.0f * k1 - d02 - d12 - d22 - d32 - d42 + d05 + d15 + d25 + d35 + d45;

    // Compute out10, out11, out12 and out13
    float out10 = d11 - d21 + 2.0f * d31 - 2.0f * d41;
    float out11 = d11 - d21 + 2.0f * d31 - 2.0f * d41;
    float out12 = d11 - d21 + 2.0f * d31 - 2.0f * d41;
    float out13 = d11 - d21 + 2.0f * d31 - 2.0f * d41;

    k0 = d13 + d14 - d23 - d24 + 2.0f * d33 + 2.0f * d34 - 2.0f * d43 - 2.0f * d44;
    k1 = 2.0f * d13 - 2.0f * d14 - 2.0f * d23 + 2.0f * d24 + 4.0f * d33 - 4.0f * d34 - 4.0f * d43 + 4.0f * d44;

    out10 += k0 + d10 + d12 - d20 - d22 + 2.0f * d30 + 2.0f * d32 - 2.0f * d40 - 2.0f * d42;
    out11 += k1 - d12 + d22 - 2.0f * d32 + 2.0f * d42;
    out12 += 4.0f * k0 + d12 - d22 + 2.0f * d32 - 2.0f * d42;
    out13 += 4.0f * k1 - d12 + d15 + d22 - d25 - 2.0f * d32 + 2.0f * d35 + 2.0f * d42 - 2.0f * d45;

    // Compute out20, out21, out22 and out23
    float out20 = d11 + d21 + 4.0f * d31 + 4.0f * d41;
    float out21 = d11 + d21 + 4.0f * d31 + 4.0f * d41;
    float out22 = d11 + d21 + 4.0f * d31 + 4.0f * d41;
    float out23 = d11 + d21 + 4.0f * d31 + 4.0f * d41;

    k0 = d13 + d14 + d23 + d24 + 4.0f * d33 + 4.0f * d34 + 4.0f * d43 + 4.0f * d44;
    k1 = 2.0f * d13 - 2.0f * d14 + 2.0f * d23 - 2.0f * d24 + 8.0f * d33 - 8.0f * d34 + 8.0f * d43 - 8.0f * d44;

    out20 += k0 + d10 + d12 + d20 + d22 + 4.0f * d30 + 4.0f * d32 + 4.0f * d40 + 4.0f * d42;
    out21 += k1 - d12 - d22 - 4.0f * d32 - 4.0f * d42;
    out22 += 4.0f * k0 + d12 + d22 + 4.0f * d32 + 4.0f * d42;
    out23 += 4.0f * k1 - d12 + d15 - d22 + d25 - 4.0f * d32 + 4.0f * d35 - 4.0f * d42 + 4.0f * d45;

    // Compute out30, out31, out32 and out33
    float out30 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;
    float out31 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;
    float out32 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;
    float out33 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;

    k0 = d13 + d14 - d23 - d24 + 8.0f * d33 + 8.0f * d34 - 8.0f * d43 - 8.0f * d44 + d53 + d54;
    k1 = 2.0f * d13 - 2.0f * d14 - 2.0f * d23 + 2.0f * d24 + 16.0f * d33 - 16.0f * d34 - 16.0f * d43 + 16.0f * d44 + 2.0f * d53 - 2.0f * d54;

    out30 += k0 + d10 + d12 - d20 - d22 + 8.0f * d30 + 8.0f * d32 - 8.0f * d40 - 8.0f * d42 + d50 + d52;
    out31 += k1 - d12 + d22 - 8.0f * d32 + 8.0f * d42 - d52;
    out32 += 4.0f * k0 + d12 - d22 + 8.0f * d32 - 8.0f * d42 + d52;
    out33 += 4.0f * k1 - d12 + d15 + d22 - d25 - 8.0f * d32 + 8.0f * d35 + 8.0f * d42 - 8.0f * d45 - d52 + d55;

    int y_in  = get_global_id(1);
    int x_out = (y_in % NUM_TILES_X) * 4;
    int y_out = (y_in / NUM_TILES_X) * 4;
    int z_out = get_global_id(0);

#if defined(HAS_BIAS)
    // Add bias
    Vector bias = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bias);

    float b = (float) * ((__global float *)(vector_offset(&bias, z_out)));

    out00 += (float)b;
    out01 += (float)b;
    out02 += (float)b;
    out03 += (float)b;

    out10 += (float)b;
    out11 += (float)b;
    out12 += (float)b;
    out13 += (float)b;

    out20 += (float)b;
    out21 += (float)b;
    out22 += (float)b;
    out23 += (float)b;

    out30 += (float)b;
    out31 += (float)b;
    out32 += (float)b;
    out33 += (float)b;

#endif // defined(HAS_BIAS)

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_out * dst_stride_x + y_out * dst_stride_y + z_out * dst_stride_z;

    // Store the 4x4 output tile
    vstore4((float4)(out00, out01, out02, out03), 0, (__global float *)(dst_addr + 0 * dst_stride_y));
    vstore4((float4)(out10, out11, out12, out13), 0, (__global float *)(dst_addr + 1 * dst_stride_y));
    vstore4((float4)(out20, out21, out22, out23), 0, (__global float *)(dst_addr + 2 * dst_stride_y));
    vstore4((float4)(out30, out31, out32, out33), 0, (__global float *)(dst_addr + 3 * dst_stride_y));
}

/** This OpenCL kernel performs Winograd output transform when the output tile is 4x4, the filter size 3x3 and the data layout is NHWC
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
 * @param[in]  dst_size                          Size of the destination tensor, minus the last padding
 */
__kernel void winograd_output_transform_4x4_3x3_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int dst_size)
{
    // Each thread stores a 4x4 tile
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const __global uchar *src_addr = tensor3D_offset(&src, 0, 0, 0);

    // Load the values across the 36 channels to compose the 6x6 tile
    float d00 = *((__global float *)(src_addr + 0 * src_stride_z));
    float d01 = *((__global float *)(src_addr + 1 * src_stride_z));
    float d02 = *((__global float *)(src_addr + 2 * src_stride_z));
    float d03 = *((__global float *)(src_addr + 3 * src_stride_z));
    float d04 = *((__global float *)(src_addr + 4 * src_stride_z));
    float d05 = *((__global float *)(src_addr + 5 * src_stride_z));

    float d10 = *((__global float *)(src_addr + 6 * src_stride_z));
    float d11 = *((__global float *)(src_addr + 7 * src_stride_z));
    float d12 = *((__global float *)(src_addr + 8 * src_stride_z));
    float d13 = *((__global float *)(src_addr + 9 * src_stride_z));
    float d14 = *((__global float *)(src_addr + 10 * src_stride_z));
    float d15 = *((__global float *)(src_addr + 11 * src_stride_z));

    float d20 = *((__global float *)(src_addr + 12 * src_stride_z));
    float d21 = *((__global float *)(src_addr + 13 * src_stride_z));
    float d22 = *((__global float *)(src_addr + 14 * src_stride_z));
    float d23 = *((__global float *)(src_addr + 15 * src_stride_z));
    float d24 = *((__global float *)(src_addr + 16 * src_stride_z));
    float d25 = *((__global float *)(src_addr + 17 * src_stride_z));

    float d30 = *((__global float *)(src_addr + 18 * src_stride_z));
    float d31 = *((__global float *)(src_addr + 19 * src_stride_z));
    float d32 = *((__global float *)(src_addr + 20 * src_stride_z));
    float d33 = *((__global float *)(src_addr + 21 * src_stride_z));
    float d34 = *((__global float *)(src_addr + 22 * src_stride_z));
    float d35 = *((__global float *)(src_addr + 23 * src_stride_z));

    float d40 = *((__global float *)(src_addr + 24 * src_stride_z));
    float d41 = *((__global float *)(src_addr + 25 * src_stride_z));
    float d42 = *((__global float *)(src_addr + 26 * src_stride_z));
    float d43 = *((__global float *)(src_addr + 27 * src_stride_z));
    float d44 = *((__global float *)(src_addr + 28 * src_stride_z));
    float d45 = *((__global float *)(src_addr + 29 * src_stride_z));

    float d50 = *((__global float *)(src_addr + 30 * src_stride_z));
    float d51 = *((__global float *)(src_addr + 31 * src_stride_z));
    float d52 = *((__global float *)(src_addr + 32 * src_stride_z));
    float d53 = *((__global float *)(src_addr + 33 * src_stride_z));
    float d54 = *((__global float *)(src_addr + 34 * src_stride_z));
    float d55 = *((__global float *)(src_addr + 35 * src_stride_z));

    // Compute out00, out01, out02 and out03
    float out00 = d01 + d21 + d41 + d11 + d31;
    float out01 = d01 + d21 + d41 + d11 + d31;
    float out02 = d01 + d21 + d41 + d11 + d31;
    float out03 = d01 + d21 + d41 + d11 + d31;

    float k0 = d03 + d04 + d13 + d14 + d23 + d24 + d33 + d34 + d43 + d44;
    float k1 = 2.0f * d03 - 2.0f * d04 + 2.0f * d13 - 2.0f * d14 + 2.0f * d23 - 2.0f * d24 + 2.0f * d33 - 2.0f * d34 + 2.0f * d43 - 2.0f * d44;

    out00 += k0 + d00 + d02 + d10 + d12 + d20 + d22 + d30 + d32 + d40 + d42;
    out01 += k1 - d02 - d12 - d22 - d32 - d42;
    out02 += 4.0f * k0 + d02 + d12 + d22 + d32 + d42;
    out03 += 4.0f * k1 - d02 - d12 - d22 - d32 - d42 + d05 + d15 + d25 + d35 + d45;

    // Compute out10, out11, out12 and out13
    float out10 = d11 - d21 + 2.0f * d31 - 2.0f * d41;
    float out11 = d11 - d21 + 2.0f * d31 - 2.0f * d41;
    float out12 = d11 - d21 + 2.0f * d31 - 2.0f * d41;
    float out13 = d11 - d21 + 2.0f * d31 - 2.0f * d41;

    k0 = d13 + d14 - d23 - d24 + 2.0f * d33 + 2.0f * d34 - 2.0f * d43 - 2.0f * d44;
    k1 = 2.0f * d13 - 2.0f * d14 - 2.0f * d23 + 2.0f * d24 + 4.0f * d33 - 4.0f * d34 - 4.0f * d43 + 4.0f * d44;

    out10 += k0 + d10 + d12 - d20 - d22 + 2.0f * d30 + 2.0f * d32 - 2.0f * d40 - 2.0f * d42;
    out11 += k1 - d12 + d22 - 2.0f * d32 + 2.0f * d42;
    out12 += 4.0f * k0 + d12 - d22 + 2.0f * d32 - 2.0f * d42;
    out13 += 4.0f * k1 - d12 + d15 + d22 - d25 - 2.0f * d32 + 2.0f * d35 + 2.0f * d42 - 2.0f * d45;

    // Compute out20, out21, out22 and out23
    float out20 = d11 + d21 + 4.0f * d31 + 4.0f * d41;
    float out21 = d11 + d21 + 4.0f * d31 + 4.0f * d41;
    float out22 = d11 + d21 + 4.0f * d31 + 4.0f * d41;
    float out23 = d11 + d21 + 4.0f * d31 + 4.0f * d41;

    k0 = d13 + d14 + d23 + d24 + 4.0f * d33 + 4.0f * d34 + 4.0f * d43 + 4.0f * d44;
    k1 = 2.0f * d13 - 2.0f * d14 + 2.0f * d23 - 2.0f * d24 + 8.0f * d33 - 8.0f * d34 + 8.0f * d43 - 8.0f * d44;

    out20 += k0 + d10 + d12 + d20 + d22 + 4.0f * d30 + 4.0f * d32 + 4.0f * d40 + 4.0f * d42;
    out21 += k1 - d12 - d22 - 4.0f * d32 - 4.0f * d42;
    out22 += 4.0f * k0 + d12 + d22 + 4.0f * d32 + 4.0f * d42;
    out23 += 4.0f * k1 - d12 + d15 - d22 + d25 - 4.0f * d32 + 4.0f * d35 - 4.0f * d42 + 4.0f * d45;

    // Compute out30, out31, out32 and out33
    float out30 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;
    float out31 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;
    float out32 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;
    float out33 = d11 - d21 + 8.0f * d31 - 8.0f * d41 + d51;

    k0 = d13 + d14 - d23 - d24 + 8.0f * d33 + 8.0f * d34 - 8.0f * d43 - 8.0f * d44 + d53 + d54;
    k1 = 2.0f * d13 - 2.0f * d14 - 2.0f * d23 + 2.0f * d24 + 16.0f * d33 - 16.0f * d34 - 16.0f * d43 + 16.0f * d44 + 2.0f * d53 - 2.0f * d54;

    out30 += k0 + d10 + d12 - d20 - d22 + 8.0f * d30 + 8.0f * d32 - 8.0f * d40 - 8.0f * d42 + d50 + d52;
    out31 += k1 - d12 + d22 - 8.0f * d32 + 8.0f * d42 - d52;
    out32 += 4.0f * k0 + d12 - d22 + 8.0f * d32 - 8.0f * d42 + d52;
    out33 += 4.0f * k1 - d12 + d15 + d22 - d25 - 8.0f * d32 + 8.0f * d35 + 8.0f * d42 - 8.0f * d45 - d52 + d55;

    int y_in  = get_global_id(1);
    int x_out = get_global_id(0);
    int y_out = (y_in % NUM_TILES_X) * 4;
    int z_out = (y_in / NUM_TILES_X) * 4;

#if defined(HAS_BIAS)
    // Add bias
    Vector bias = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bias);

    float b = (float) * ((__global float *)(vector_offset(&bias, z_out)));

    out00 += (float)b;
    out01 += (float)b;
    out02 += (float)b;
    out03 += (float)b;

    out10 += (float)b;
    out11 += (float)b;
    out12 += (float)b;
    out13 += (float)b;

    out20 += (float)b;
    out21 += (float)b;
    out22 += (float)b;
    out23 += (float)b;

    out30 += (float)b;
    out31 += (float)b;
    out32 += (float)b;
    out33 += (float)b;

#endif // defined(HAS_BIAS)

    // Get output address
    int4 offset = (int4)(dst_offset_first_element_in_bytes + x_out * sizeof(float) + y_out * dst_stride_y + z_out * dst_stride_z);
    offset      = min(offset + (int4)(0, 1, 2, 3) * (int4)dst_stride_z, dst_size); // If address is beyond the last plane, clamp it to dst_size (which points to the last padding).
    int4 mult_y = min(dst_size - offset, 1);                                       // If out of bound, we don't want to increase dst_stride_y, so we set the multiplier to 0. It will be 1 otherwise.

    // Store the 4x4 output tile
    *((__global float *)(dst_ptr + mult_y.s0 * 0 * dst_stride_y + offset.s0)) = out00;
    *((__global float *)(dst_ptr + mult_y.s0 * 1 * dst_stride_y + offset.s0)) = out01;
    *((__global float *)(dst_ptr + mult_y.s0 * 2 * dst_stride_y + offset.s0)) = out02;
    *((__global float *)(dst_ptr + mult_y.s0 * 3 * dst_stride_y + offset.s0)) = out03;
    *((__global float *)(dst_ptr + mult_y.s1 * 0 * dst_stride_y + offset.s1)) = out10;
    *((__global float *)(dst_ptr + mult_y.s1 * 1 * dst_stride_y + offset.s1)) = out11;
    *((__global float *)(dst_ptr + mult_y.s1 * 2 * dst_stride_y + offset.s1)) = out12;
    *((__global float *)(dst_ptr + mult_y.s1 * 3 * dst_stride_y + offset.s1)) = out13;
    *((__global float *)(dst_ptr + mult_y.s2 * 0 * dst_stride_y + offset.s2)) = out20;
    *((__global float *)(dst_ptr + mult_y.s2 * 1 * dst_stride_y + offset.s2)) = out21;
    *((__global float *)(dst_ptr + mult_y.s2 * 2 * dst_stride_y + offset.s2)) = out22;
    *((__global float *)(dst_ptr + mult_y.s2 * 3 * dst_stride_y + offset.s2)) = out23;
    *((__global float *)(dst_ptr + mult_y.s3 * 0 * dst_stride_y + offset.s3)) = out30;
    *((__global float *)(dst_ptr + mult_y.s3 * 1 * dst_stride_y + offset.s3)) = out31;
    *((__global float *)(dst_ptr + mult_y.s3 * 2 * dst_stride_y + offset.s3)) = out32;
    *((__global float *)(dst_ptr + mult_y.s3 * 3 * dst_stride_y + offset.s3)) = out33;
}

#define COMPUTE_TMP_COL(col, d0, d1, d2, d3, d4, d5, d6, d7, comm_fact)  \
    ({                                                                   \
        comm_fact.s0 = d1 + d2;                                          \
        comm_fact.s1 = d3 + d4;                                          \
        comm_fact.s2 = d5 + d6;                                          \
        \
        col.s0 = comm_fact.s0 + comm_fact.s1 + 8.f * comm_fact.s2 + d0;  \
        col.s2 = comm_fact.s0 + 4.f * comm_fact.s1 + 2.f * comm_fact.s2; \
        \
        comm_fact.s0 = d1 - d2;                                          \
        comm_fact.s1 = d3 - d4;                                          \
        comm_fact.s2 = d5 - d6;                                          \
        \
        col.s1 = comm_fact.s0 + 2.f * comm_fact.s1 + 4.f * comm_fact.s2; \
        col.s3 = comm_fact.s0 + 8.f * comm_fact.s1 + comm_fact.s2 + d7;  \
    })

/** This OpenCL kernel performs Winograd output transform when the output tile is 4x4, the filter size 5x5 and the data layout is NCHW
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
__kernel void winograd_output_transform_4x4_5x5_nchw(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst)
#if defined(HAS_BIAS)
    ,
    VECTOR_DECLARATION(bias)
#endif // defined(HAS_BIAS)
)
{
    // Each thread stores a 4x4 tile
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const __global uchar *src_addr = tensor3D_offset(&src, 0, 0, 0);

    // Load the values across the 64 channels to compose the 8x8 input tile
    float d00 = *((__global float *)(src_addr + 0 * src_stride_z));
    float d01 = *((__global float *)(src_addr + 1 * src_stride_z));
    float d02 = *((__global float *)(src_addr + 2 * src_stride_z));
    float d03 = *((__global float *)(src_addr + 3 * src_stride_z));
    float d04 = *((__global float *)(src_addr + 4 * src_stride_z));
    float d05 = *((__global float *)(src_addr + 5 * src_stride_z));
    float d06 = *((__global float *)(src_addr + 6 * src_stride_z));
    float d07 = *((__global float *)(src_addr + 7 * src_stride_z));

    float d10 = *((__global float *)(src_addr + 8 * src_stride_z));
    float d11 = *((__global float *)(src_addr + 9 * src_stride_z));
    float d12 = *((__global float *)(src_addr + 10 * src_stride_z));
    float d13 = *((__global float *)(src_addr + 11 * src_stride_z));
    float d14 = *((__global float *)(src_addr + 12 * src_stride_z));
    float d15 = *((__global float *)(src_addr + 13 * src_stride_z));
    float d16 = *((__global float *)(src_addr + 14 * src_stride_z));
    float d17 = *((__global float *)(src_addr + 15 * src_stride_z));

    float d20 = *((__global float *)(src_addr + 16 * src_stride_z));
    float d21 = *((__global float *)(src_addr + 17 * src_stride_z));
    float d22 = *((__global float *)(src_addr + 18 * src_stride_z));
    float d23 = *((__global float *)(src_addr + 19 * src_stride_z));
    float d24 = *((__global float *)(src_addr + 20 * src_stride_z));
    float d25 = *((__global float *)(src_addr + 21 * src_stride_z));
    float d26 = *((__global float *)(src_addr + 22 * src_stride_z));
    float d27 = *((__global float *)(src_addr + 23 * src_stride_z));

    float d30 = *((__global float *)(src_addr + 24 * src_stride_z));
    float d31 = *((__global float *)(src_addr + 25 * src_stride_z));
    float d32 = *((__global float *)(src_addr + 26 * src_stride_z));
    float d33 = *((__global float *)(src_addr + 27 * src_stride_z));
    float d34 = *((__global float *)(src_addr + 28 * src_stride_z));
    float d35 = *((__global float *)(src_addr + 29 * src_stride_z));
    float d36 = *((__global float *)(src_addr + 30 * src_stride_z));
    float d37 = *((__global float *)(src_addr + 31 * src_stride_z));

    float d40 = *((__global float *)(src_addr + 32 * src_stride_z));
    float d41 = *((__global float *)(src_addr + 33 * src_stride_z));
    float d42 = *((__global float *)(src_addr + 34 * src_stride_z));
    float d43 = *((__global float *)(src_addr + 35 * src_stride_z));
    float d44 = *((__global float *)(src_addr + 36 * src_stride_z));
    float d45 = *((__global float *)(src_addr + 37 * src_stride_z));
    float d46 = *((__global float *)(src_addr + 38 * src_stride_z));
    float d47 = *((__global float *)(src_addr + 39 * src_stride_z));

    float d50 = *((__global float *)(src_addr + 40 * src_stride_z));
    float d51 = *((__global float *)(src_addr + 41 * src_stride_z));
    float d52 = *((__global float *)(src_addr + 42 * src_stride_z));
    float d53 = *((__global float *)(src_addr + 43 * src_stride_z));
    float d54 = *((__global float *)(src_addr + 44 * src_stride_z));
    float d55 = *((__global float *)(src_addr + 45 * src_stride_z));
    float d56 = *((__global float *)(src_addr + 46 * src_stride_z));
    float d57 = *((__global float *)(src_addr + 47 * src_stride_z));

    float d60 = *((__global float *)(src_addr + 48 * src_stride_z));
    float d61 = *((__global float *)(src_addr + 49 * src_stride_z));
    float d62 = *((__global float *)(src_addr + 50 * src_stride_z));
    float d63 = *((__global float *)(src_addr + 51 * src_stride_z));
    float d64 = *((__global float *)(src_addr + 52 * src_stride_z));
    float d65 = *((__global float *)(src_addr + 53 * src_stride_z));
    float d66 = *((__global float *)(src_addr + 54 * src_stride_z));
    float d67 = *((__global float *)(src_addr + 55 * src_stride_z));

    float d70 = *((__global float *)(src_addr + 56 * src_stride_z));
    float d71 = *((__global float *)(src_addr + 57 * src_stride_z));
    float d72 = *((__global float *)(src_addr + 58 * src_stride_z));
    float d73 = *((__global float *)(src_addr + 59 * src_stride_z));
    float d74 = *((__global float *)(src_addr + 60 * src_stride_z));
    float d75 = *((__global float *)(src_addr + 61 * src_stride_z));
    float d76 = *((__global float *)(src_addr + 62 * src_stride_z));
    float d77 = *((__global float *)(src_addr + 63 * src_stride_z));

    // Compute the 8x4 intermediate tensor
    float4 comm_fact0, comm_fact1, comm_fact2;
    float4 tmp_col0, tmp_col1, tmp_col2, tmp_col3, tmp_col4, tmp_col5, tmp_col6, tmp_col7;

    COMPUTE_TMP_COL(tmp_col0, d00, d10, d20, d30, d40, d50, d60, d70, comm_fact0);
    COMPUTE_TMP_COL(tmp_col1, d01, d11, d21, d31, d41, d51, d61, d71, comm_fact0);
    COMPUTE_TMP_COL(tmp_col2, d02, d12, d22, d32, d42, d52, d62, d72, comm_fact0);
    COMPUTE_TMP_COL(tmp_col3, d03, d13, d23, d33, d43, d53, d63, d73, comm_fact0);
    COMPUTE_TMP_COL(tmp_col4, d04, d14, d24, d34, d44, d54, d64, d74, comm_fact0);
    COMPUTE_TMP_COL(tmp_col5, d05, d15, d25, d35, d45, d55, d65, d75, comm_fact0);
    COMPUTE_TMP_COL(tmp_col6, d06, d16, d26, d36, d46, d56, d66, d76, comm_fact0);
    COMPUTE_TMP_COL(tmp_col7, d07, d17, d27, d37, d47, d57, d67, d77, comm_fact0);

    // Compute the 4x4 output tile
    comm_fact0 = tmp_col1 + tmp_col2;
    comm_fact1 = tmp_col3 + tmp_col4;
    comm_fact2 = tmp_col5 + tmp_col6;

    float4 out_col0 = comm_fact0 + comm_fact1 + 8.f * comm_fact2 + tmp_col0;
    float4 out_col2 = comm_fact0 + 4.f * comm_fact1 + 2.f * comm_fact2;

    comm_fact0 = tmp_col1 - tmp_col2;
    comm_fact1 = tmp_col3 - tmp_col4;
    comm_fact2 = tmp_col5 - tmp_col6;

    float4 out_col1 = comm_fact0 + 2.f * comm_fact1 + 4.f * comm_fact2;
    float4 out_col3 = comm_fact0 + 8.f * comm_fact1 + comm_fact2 + tmp_col7;

    int y_in  = get_global_id(1);
    int x_out = (y_in % NUM_TILES_X) * 4;
    int y_out = (y_in / NUM_TILES_X) * 4;
    int z_out = get_global_id(0);

#if defined(HAS_BIAS)
    // Add bias
    Vector bias = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bias);

    float b = (float) * ((__global float *)(vector_offset(&bias, z_out)));

    out_col0 += (float4)b;
    out_col1 += (float4)b;
    out_col2 += (float4)b;
    out_col3 += (float4)b;
#endif // defined(HAS_BIAS)

    // Get output address
    __global uchar *dst_addr = dst_ptr + dst_offset_first_element_in_bytes + x_out * dst_stride_x + y_out * dst_stride_y + z_out * dst_stride_z;

    // Store the 4x4 output tile
    *(__global float *)(dst_addr + 0 * dst_stride_x + 0 * dst_stride_y) = out_col0.s0;
    *(__global float *)(dst_addr + 1 * dst_stride_x + 0 * dst_stride_y) = out_col1.s0;
    *(__global float *)(dst_addr + 2 * dst_stride_x + 0 * dst_stride_y) = out_col2.s0;
    *(__global float *)(dst_addr + 3 * dst_stride_x + 0 * dst_stride_y) = out_col3.s0;
    *(__global float *)(dst_addr + 0 * dst_stride_x + 1 * dst_stride_y) = out_col0.s1;
    *(__global float *)(dst_addr + 1 * dst_stride_x + 1 * dst_stride_y) = out_col1.s1;
    *(__global float *)(dst_addr + 2 * dst_stride_x + 1 * dst_stride_y) = out_col2.s1;
    *(__global float *)(dst_addr + 3 * dst_stride_x + 1 * dst_stride_y) = out_col3.s1;
    *(__global float *)(dst_addr + 0 * dst_stride_x + 2 * dst_stride_y) = out_col0.s2;
    *(__global float *)(dst_addr + 1 * dst_stride_x + 2 * dst_stride_y) = out_col1.s2;
    *(__global float *)(dst_addr + 2 * dst_stride_x + 2 * dst_stride_y) = out_col2.s2;
    *(__global float *)(dst_addr + 3 * dst_stride_x + 2 * dst_stride_y) = out_col3.s2;
    *(__global float *)(dst_addr + 0 * dst_stride_x + 3 * dst_stride_y) = out_col0.s3;
    *(__global float *)(dst_addr + 1 * dst_stride_x + 3 * dst_stride_y) = out_col1.s3;
    *(__global float *)(dst_addr + 2 * dst_stride_x + 3 * dst_stride_y) = out_col2.s3;
    *(__global float *)(dst_addr + 3 * dst_stride_x + 3 * dst_stride_y) = out_col3.s3;
}

/** This OpenCL kernel performs Winograd output transform when the output tile is 4x4, the filter size 5x5 and the data format is NHWC
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
__kernel void winograd_output_transform_4x4_5x5_nhwc(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
#if defined(HAS_BIAS)
    VECTOR_DECLARATION(bias),
#endif // defined(HAS_BIAS)
    int dst_size)
{
    // Each thread stores a 4x4 tile
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);

    const __global uchar *src_addr = tensor3D_offset(&src, 0, 0, 0);

    // Load the values across the 64 channels to compose the 8x8 input tile
    float d00 = *((__global float *)(src_addr + 0 * src_stride_z));
    float d01 = *((__global float *)(src_addr + 1 * src_stride_z));
    float d02 = *((__global float *)(src_addr + 2 * src_stride_z));
    float d03 = *((__global float *)(src_addr + 3 * src_stride_z));
    float d04 = *((__global float *)(src_addr + 4 * src_stride_z));
    float d05 = *((__global float *)(src_addr + 5 * src_stride_z));
    float d06 = *((__global float *)(src_addr + 6 * src_stride_z));
    float d07 = *((__global float *)(src_addr + 7 * src_stride_z));

    float d10 = *((__global float *)(src_addr + 8 * src_stride_z));
    float d11 = *((__global float *)(src_addr + 9 * src_stride_z));
    float d12 = *((__global float *)(src_addr + 10 * src_stride_z));
    float d13 = *((__global float *)(src_addr + 11 * src_stride_z));
    float d14 = *((__global float *)(src_addr + 12 * src_stride_z));
    float d15 = *((__global float *)(src_addr + 13 * src_stride_z));
    float d16 = *((__global float *)(src_addr + 14 * src_stride_z));
    float d17 = *((__global float *)(src_addr + 15 * src_stride_z));

    float d20 = *((__global float *)(src_addr + 16 * src_stride_z));
    float d21 = *((__global float *)(src_addr + 17 * src_stride_z));
    float d22 = *((__global float *)(src_addr + 18 * src_stride_z));
    float d23 = *((__global float *)(src_addr + 19 * src_stride_z));
    float d24 = *((__global float *)(src_addr + 20 * src_stride_z));
    float d25 = *((__global float *)(src_addr + 21 * src_stride_z));
    float d26 = *((__global float *)(src_addr + 22 * src_stride_z));
    float d27 = *((__global float *)(src_addr + 23 * src_stride_z));

    float d30 = *((__global float *)(src_addr + 24 * src_stride_z));
    float d31 = *((__global float *)(src_addr + 25 * src_stride_z));
    float d32 = *((__global float *)(src_addr + 26 * src_stride_z));
    float d33 = *((__global float *)(src_addr + 27 * src_stride_z));
    float d34 = *((__global float *)(src_addr + 28 * src_stride_z));
    float d35 = *((__global float *)(src_addr + 29 * src_stride_z));
    float d36 = *((__global float *)(src_addr + 30 * src_stride_z));
    float d37 = *((__global float *)(src_addr + 31 * src_stride_z));

    float d40 = *((__global float *)(src_addr + 32 * src_stride_z));
    float d41 = *((__global float *)(src_addr + 33 * src_stride_z));
    float d42 = *((__global float *)(src_addr + 34 * src_stride_z));
    float d43 = *((__global float *)(src_addr + 35 * src_stride_z));
    float d44 = *((__global float *)(src_addr + 36 * src_stride_z));
    float d45 = *((__global float *)(src_addr + 37 * src_stride_z));
    float d46 = *((__global float *)(src_addr + 38 * src_stride_z));
    float d47 = *((__global float *)(src_addr + 39 * src_stride_z));

    float d50 = *((__global float *)(src_addr + 40 * src_stride_z));
    float d51 = *((__global float *)(src_addr + 41 * src_stride_z));
    float d52 = *((__global float *)(src_addr + 42 * src_stride_z));
    float d53 = *((__global float *)(src_addr + 43 * src_stride_z));
    float d54 = *((__global float *)(src_addr + 44 * src_stride_z));
    float d55 = *((__global float *)(src_addr + 45 * src_stride_z));
    float d56 = *((__global float *)(src_addr + 46 * src_stride_z));
    float d57 = *((__global float *)(src_addr + 47 * src_stride_z));

    float d60 = *((__global float *)(src_addr + 48 * src_stride_z));
    float d61 = *((__global float *)(src_addr + 49 * src_stride_z));
    float d62 = *((__global float *)(src_addr + 50 * src_stride_z));
    float d63 = *((__global float *)(src_addr + 51 * src_stride_z));
    float d64 = *((__global float *)(src_addr + 52 * src_stride_z));
    float d65 = *((__global float *)(src_addr + 53 * src_stride_z));
    float d66 = *((__global float *)(src_addr + 54 * src_stride_z));
    float d67 = *((__global float *)(src_addr + 55 * src_stride_z));

    float d70 = *((__global float *)(src_addr + 56 * src_stride_z));
    float d71 = *((__global float *)(src_addr + 57 * src_stride_z));
    float d72 = *((__global float *)(src_addr + 58 * src_stride_z));
    float d73 = *((__global float *)(src_addr + 59 * src_stride_z));
    float d74 = *((__global float *)(src_addr + 60 * src_stride_z));
    float d75 = *((__global float *)(src_addr + 61 * src_stride_z));
    float d76 = *((__global float *)(src_addr + 62 * src_stride_z));
    float d77 = *((__global float *)(src_addr + 63 * src_stride_z));

    // Compute the 8x4 intermediate tensor
    float4 comm_fact0, comm_fact1, comm_fact2;
    float4 tmp_col0, tmp_col1, tmp_col2, tmp_col3, tmp_col4, tmp_col5, tmp_col6, tmp_col7;

    COMPUTE_TMP_COL(tmp_col0, d00, d10, d20, d30, d40, d50, d60, d70, comm_fact0);
    COMPUTE_TMP_COL(tmp_col1, d01, d11, d21, d31, d41, d51, d61, d71, comm_fact0);
    COMPUTE_TMP_COL(tmp_col2, d02, d12, d22, d32, d42, d52, d62, d72, comm_fact0);
    COMPUTE_TMP_COL(tmp_col3, d03, d13, d23, d33, d43, d53, d63, d73, comm_fact0);
    COMPUTE_TMP_COL(tmp_col4, d04, d14, d24, d34, d44, d54, d64, d74, comm_fact0);
    COMPUTE_TMP_COL(tmp_col5, d05, d15, d25, d35, d45, d55, d65, d75, comm_fact0);
    COMPUTE_TMP_COL(tmp_col6, d06, d16, d26, d36, d46, d56, d66, d76, comm_fact0);
    COMPUTE_TMP_COL(tmp_col7, d07, d17, d27, d37, d47, d57, d67, d77, comm_fact0);

    // Compute the 4x4 output tile
    comm_fact0 = tmp_col1 + tmp_col2;
    comm_fact1 = tmp_col3 + tmp_col4;
    comm_fact2 = tmp_col5 + tmp_col6;

    float4 out_col0 = comm_fact0 + comm_fact1 + 8.f * comm_fact2 + tmp_col0;
    float4 out_col2 = comm_fact0 + 4.f * comm_fact1 + 2.f * comm_fact2;

    comm_fact0 = tmp_col1 - tmp_col2;
    comm_fact1 = tmp_col3 - tmp_col4;
    comm_fact2 = tmp_col5 - tmp_col6;

    float4 out_col1 = comm_fact0 + 2.f * comm_fact1 + 4.f * comm_fact2;
    float4 out_col3 = comm_fact0 + 8.f * comm_fact1 + comm_fact2 + tmp_col7;

    int y_in  = get_global_id(1);
    int x_out = get_global_id(0);
    int y_out = (y_in % NUM_TILES_X) * 4;
    int z_out = (y_in / NUM_TILES_X) * 4;

#if defined(HAS_BIAS)
    // Add bias
    Vector bias = CONVERT_TO_VECTOR_STRUCT_NO_STEP(bias);

    float b = (float) * ((__global float *)(vector_offset(&bias, z_out)));

    out_col0 += (float4)b;
    out_col1 += (float4)b;
    out_col2 += (float4)b;
    out_col3 += (float4)b;
#endif // defined(HAS_BIAS)

    // Get output address
    int4 offset = (int4)(dst_offset_first_element_in_bytes + x_out * sizeof(float) + y_out * dst_stride_y + z_out * dst_stride_z);
    offset      = min(offset + (int4)(0, 1, 2, 3) * (int4)dst_stride_z, dst_size); // If address is beyond the last plane, clamp it to dst_size (which points to the last padding).
    int4 mult_y = min(dst_size - offset, 1);                                       // If out of bound, we don't want to increase dst_stride_y, so we set the multiplier to 0. It will be 1 otherwise.

    // Store the 4x4 output tile
    *(__global float *)(dst_ptr + mult_y.s0 * 0 * dst_stride_y + offset.s0) = out_col0.s0;
    *(__global float *)(dst_ptr + mult_y.s0 * 1 * dst_stride_y + offset.s0) = out_col1.s0;
    *(__global float *)(dst_ptr + mult_y.s0 * 2 * dst_stride_y + offset.s0) = out_col2.s0;
    *(__global float *)(dst_ptr + mult_y.s0 * 3 * dst_stride_y + offset.s0) = out_col3.s0;
    *(__global float *)(dst_ptr + mult_y.s0 * 0 * dst_stride_y + offset.s1) = out_col0.s1;
    *(__global float *)(dst_ptr + mult_y.s0 * 1 * dst_stride_y + offset.s1) = out_col1.s1;
    *(__global float *)(dst_ptr + mult_y.s0 * 2 * dst_stride_y + offset.s1) = out_col2.s1;
    *(__global float *)(dst_ptr + mult_y.s0 * 3 * dst_stride_y + offset.s1) = out_col3.s1;
    *(__global float *)(dst_ptr + mult_y.s0 * 0 * dst_stride_y + offset.s2) = out_col0.s2;
    *(__global float *)(dst_ptr + mult_y.s0 * 1 * dst_stride_y + offset.s2) = out_col1.s2;
    *(__global float *)(dst_ptr + mult_y.s0 * 2 * dst_stride_y + offset.s2) = out_col2.s2;
    *(__global float *)(dst_ptr + mult_y.s0 * 3 * dst_stride_y + offset.s2) = out_col3.s2;
    *(__global float *)(dst_ptr + mult_y.s0 * 0 * dst_stride_y + offset.s3) = out_col0.s3;
    *(__global float *)(dst_ptr + mult_y.s0 * 1 * dst_stride_y + offset.s3) = out_col1.s3;
    *(__global float *)(dst_ptr + mult_y.s0 * 2 * dst_stride_y + offset.s3) = out_col2.s3;
    *(__global float *)(dst_ptr + mult_y.s0 * 3 * dst_stride_y + offset.s3) = out_col3.s3;
}
#endif // defined(NUM_TILES_X)
