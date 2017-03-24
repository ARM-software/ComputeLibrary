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

/** This function accumulates an input image into output image.
 *
 * @param[in]  input_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] accu_ptr                            Pointer to the destination image. Supported data types: S16
 * @param[in]  accu_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  accu_step_x                         accu_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  accu_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  accu_step_y                         accu_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  accu_offset_first_element_in_bytes  The offset of the first element in the destination image
 */
__kernel void accumulate(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(accu))
{
    // Get pixels pointer
    Image input = CONVERT_TO_IMAGE_STRUCT(input);
    Image accu  = CONVERT_TO_IMAGE_STRUCT(accu);

    // Load data
    uchar16 in_data   = vload16(0, input.ptr);
    short16 accu_data = vload16(0, (__global short *)accu.ptr);

    // Perform accumulation
    short16 res = add_sat(convert_short16(in_data), accu_data);

    // Store result
    vstore16(res, 0, (__global short *)accu.ptr);
}

/** This function accumulates a weighted value from an input image to an output image.
 *
 * @param[in]  input_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] accu_ptr                            Pointer to the destination image. Supported data types: S16
 * @param[in]  accu_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  accu_step_x                         accu_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  accu_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  accu_step_y                         accu_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  accu_offset_first_element_in_bytes  The offset of the first element in the destination image
 * @param[in]  alpha                               The float scalar value with a value in the range of 0 to 1
 */
__kernel void accumulate_weighted(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(accu),
    const float alpha)
{
    // Get pixels pointer
    Image input = CONVERT_TO_IMAGE_STRUCT(input);
    Image accu  = CONVERT_TO_IMAGE_STRUCT(accu);

    // Load data
    const float16 in_data   = convert_float16(vload16(0, input.ptr));
    const float16 accu_data = convert_float16(vload16(0, accu.ptr));

    // Calculate weighted accumulation
    const uchar16 res = convert_uchar16((1.0f - alpha) * accu_data + alpha * in_data);

    // Store result
    vstore16(res, 0, accu.ptr);
}

/** This function accumulates a squared value from an input image to an output image.
 *
 * @param[in]  input_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  input_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] accu_ptr                            Pointer to the destination image. Supported data types: S16
 * @param[in]  accu_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  accu_step_x                         accu_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  accu_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  accu_step_y                         accu_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  accu_offset_first_element_in_bytes  The offset of the first element in the destination image
 * @param[in]  shift                               The U32 scalar value with a value in the range of 0 to 15
 */
__kernel void accumulate_squared(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(accu),
    const uint shift)
{
    // Get pixels pointer
    Image input = CONVERT_TO_IMAGE_STRUCT(input);
    Image accu  = CONVERT_TO_IMAGE_STRUCT(accu);

    // Load data
    ushort16 in_data   = convert_ushort16(vload16(0, input.ptr));
    uint16   accu_data = convert_uint16(vload16(0, (__global short *)accu.ptr));

    // Calculate squared accumulation
    short16 res = convert_short16_sat(accu_data + convert_uint16((in_data * in_data) >> shift));

    // Store result
    vstore16(res, 0, (__global short *)accu.ptr);
}
