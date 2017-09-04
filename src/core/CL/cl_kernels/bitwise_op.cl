/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

/** This function computes the bitwise OR of two input images.
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_or(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    Image in1 = CONVERT_TO_IMAGE_STRUCT(in1);
    Image in2 = CONVERT_TO_IMAGE_STRUCT(in2);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    uchar16 in_a = vload16(0, in1.ptr);
    uchar16 in_b = vload16(0, in2.ptr);

    vstore16(in_a | in_b, 0, out.ptr);
}

/** This function computes the bitwise AND of two input images.
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_and(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    Image in1 = CONVERT_TO_IMAGE_STRUCT(in1);
    Image in2 = CONVERT_TO_IMAGE_STRUCT(in2);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    uchar16 in_a = vload16(0, in1.ptr);
    uchar16 in_b = vload16(0, in2.ptr);

    vstore16(in_a & in_b, 0, out.ptr);
}

/** This function computes the bitwise XOR of two input images.
 *
 * @param[in]  in1_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in1_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        in1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        in1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  in2_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  in2_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        in2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        in2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_xor(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    Image in1 = CONVERT_TO_IMAGE_STRUCT(in1);
    Image in2 = CONVERT_TO_IMAGE_STRUCT(in2);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    uchar16 in_a = vload16(0, in1.ptr);
    uchar16 in_b = vload16(0, in2.ptr);

    vstore16(in_a ^ in_b, 0, out.ptr);
}

/** This function computes the bitwise NOT of an image.
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void bitwise_not(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    uchar16 in_data = vload16(0, in.ptr);

    vstore16(~in_data, 0, out.ptr);
}
