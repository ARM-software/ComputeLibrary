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

/** Calculate the absolute difference of two input images.
 *
 * @attention The input and output data types need to be passed at compile time using -DDATA_TYPE_IN1, -DDATA_TYPE_IN2 and -DDATA_TYPE_OUT:\n
 * e.g. -DDATA_TYPE_IN1=uchar -DDATA_TYPE_IN2=uchar -DDATA_TYPE_OUT=short
 *
 * @param[in]  in1_ptr                           Pointer to the first source image. Supported data types: U8, S16
 * @param[in]  in1_stride_x                      Stride of the first source image in X dimension (in bytes)
 * @param[in]  in1_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in1_stride_y                      Stride of the first source image in Y dimension (in bytes)
 * @param[in]  in1_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in1_offset_first_element_in_bytes The offset of the first element in the first source image
 * @param[in]  in2_ptr                           Pointer to the second source image. Supported data types: U8, S16
 * @param[in]  in2_stride_x                      Stride of the second source image in X dimension (in bytes)
 * @param[in]  in2_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in2_stride_y                      Stride of the second source image in Y dimension (in bytes)
 * @param[in]  in2_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in2_offset_first_element_in_bytes The offset of the first element in the second source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8, S16
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void absdiff(
    IMAGE_DECLARATION(in1),
    IMAGE_DECLARATION(in2),
    IMAGE_DECLARATION(out))
{
    Image in1 = CONVERT_TO_IMAGE_STRUCT(in1);
    Image in2 = CONVERT_TO_IMAGE_STRUCT(in2);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    VEC_DATA_TYPE(DATA_TYPE_OUT, 16)
    in_a = CONVERT(vload16(0, (__global DATA_TYPE_IN1 *)in1.ptr), VEC_DATA_TYPE(DATA_TYPE_OUT, 16));
    VEC_DATA_TYPE(DATA_TYPE_OUT, 16)
    in_b = CONVERT(vload16(0, (__global DATA_TYPE_IN2 *)in2.ptr), VEC_DATA_TYPE(DATA_TYPE_OUT, 16));

    vstore16(CONVERT_SAT(abs_diff(in_a, in_b), VEC_DATA_TYPE(DATA_TYPE_OUT, 16)), 0, (__global DATA_TYPE_OUT *)out.ptr);
}
