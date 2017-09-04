/*
 * Copyright (c) 2017 ARM Limited.
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

/** Computes the Gaussian Filter 1x5 + sub-sampling along the X direction
 *
 * @note Each thread computes 8 pixels
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: U16
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void gaussian1x5_sub_x(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values for the convolution (20 bytes needed)
    uchar16 temp0 = vload16(0, src.ptr);
    uchar4  temp1 = vload4(0, src.ptr + 16);

    // Convert to USHORT8
    ushort8 l2_data = convert_ushort8((uchar8)(temp0.s02468ACE));
    ushort8 l1_data = convert_ushort8((uchar8)(temp0.s13579BDF));
    ushort8 m_data  = convert_ushort8((uchar8)(temp0.s2468, temp0.sACE, temp1.s0));
    ushort8 r1_data = convert_ushort8((uchar8)(temp0.s3579, temp0.sBDF, temp1.s1));
    ushort8 r2_data = convert_ushort8((uchar8)(temp0.s468A, temp0.sCE, temp1.s02));

    // Compute convolution along the X direction
    ushort8 pixels = l2_data + r2_data;
    pixels += l1_data * (ushort8)4;
    pixels += m_data * (ushort8)6;
    pixels += r1_data * (ushort8)4;

    // Store result
    vstore8(pixels, 0, (__global ushort *)dst.ptr);
}

/** Computes the Gaussian Filter 5x1 + sub-sampling along the Y direction
 *
 * @note Each thread computes 8 pixels
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void gaussian5x1_sub_y(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    ushort8 u2_data = vload8(0, (__global ushort *)offset(&src, 0, 0));
    ushort8 u1_data = vload8(0, (__global ushort *)offset(&src, 0, 1));
    ushort8 m_data  = vload8(0, (__global ushort *)offset(&src, 0, 2));
    ushort8 d1_data = vload8(0, (__global ushort *)offset(&src, 0, 3));
    ushort8 d2_data = vload8(0, (__global ushort *)offset(&src, 0, 4));

    // Compute convolution along the Y direction
    ushort8 pixels = u2_data + d2_data;
    pixels += u1_data * (ushort8)4;
    pixels += m_data * (ushort8)6;
    pixels += d1_data * (ushort8)4;

    // Scale result
    pixels >>= (ushort8)8;

    // Store result
    vstore8(convert_uchar8_sat(pixels), 0, dst.ptr);
}
