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

/** Perform binary thresholding on an image.
 *
 * @param[in]  in_ptr                            Pointer to the source image
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the first source image
 * @param[out] out_ptr                           Pointer to the destination image
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  false_val                         False value
 * @param[in]  true_val                          True value
 * @param[in]  threshold                         The thresold value
 */
__kernel void threshold_binary(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const uchar false_val,
    const uchar true_val,
    const uchar threshold)
{
    // Get pixels pointer
    Image in  = CONVERT_TO_IMAGE_STRUCT(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    // Load data
    uchar16 in_data = vload16(0, in.ptr);

    // Perform binary thresholding
    in_data = select((uchar16)false_val, (uchar16)true_val, in_data > (uchar16)threshold);

    // Store result
    vstore16(in_data, 0, out.ptr);
}

/** Perform range thresholding on an image.
 *
 * @param[in]  in_ptr                            Pointer to the source image
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the first source image
 * @param[out] out_ptr                           Pointer to the destination image
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  false_val                         False value
 * @param[in]  true_val                          True value
 * @param[in]  lower                             Lower threshold
 * @param[in]  upper                             Upper threshold
 */
__kernel void threshold_range(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const uchar false_val,
    const uchar true_val,
    const uchar lower,
    const uchar upper)
{
    // Get pixels pointer
    Image in  = CONVERT_TO_IMAGE_STRUCT(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    // Load data
    uchar16 in_data = vload16(0, in.ptr);

    // Perform range thresholding
    in_data = select((uchar16)true_val, (uchar16)false_val, in_data > (uchar16)upper || in_data < (uchar16)lower);

    // Store result
    vstore16(in_data, 0, out.ptr);
}
