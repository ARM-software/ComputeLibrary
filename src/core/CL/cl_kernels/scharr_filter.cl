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

/** This OpenCL kernel computes Scharr3x3.
 *
 * @attention To enable computation of the X gradient -DGRAD_X must be passed at compile time, while computation of the Y gradient
 * is performed when -DGRAD_Y is used. You can use both when computation of both gradients is required.
 *
 * @param[in]  src_ptr                              Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                         Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                           src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                           src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] dst_gx_ptr                           Pointer to the destination image Supported data types: S16
 * @param[in]  dst_gx_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gx_step_x                        dst_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gx_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gx_step_y                        dst_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gx_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] dst_gy_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_gy_step_x                        dst_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_gy_step_y                        dst_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void scharr3x3(
    IMAGE_DECLARATION(src)
#ifdef GRAD_X
    ,
    IMAGE_DECLARATION(dst_gx)
#endif /* GRAD_X */
#ifdef GRAD_Y
    ,
    IMAGE_DECLARATION(dst_gy)
#endif /* GRAD_Y */
)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
#ifdef GRAD_X
    Image dst_gx = CONVERT_TO_IMAGE_STRUCT(dst_gx);
#endif /* GRAD_X */
#ifdef GRAD_Y
    Image dst_gy = CONVERT_TO_IMAGE_STRUCT(dst_gy);
#endif /* GRAD_Y */

    // Output pixels
#ifdef GRAD_X
    short8 gx = (short8)0;
#endif /* GRAD_X */
#ifdef GRAD_Y
    short8 gy = (short8)0;
#endif /* GRAD_Y */

    // Row0
    uchar16 temp   = vload16(0, offset(&src, -1, -1));
    short8  left   = convert_short8(temp.s01234567);
    short8  middle = convert_short8(temp.s12345678);
    short8  right  = convert_short8(temp.s23456789);
#ifdef GRAD_X
    gx += left * (short8)(-3);
    gx += right * (short8)(+3);
#endif /* GRAD_X */
#ifdef GRAD_Y
    gy += left * (short8)(-3);
    gy += middle * (short8)(-10);
    gy += right * (short8)(-3);
#endif /* GRAD_Y */

    // Row1
    temp  = vload16(0, offset(&src, -1, 0));
    left  = convert_short8(temp.s01234567);
    right = convert_short8(temp.s23456789);
#ifdef GRAD_X
    gx += left * (short8)(-10);
    gx += right * (short8)(+10);
#endif /* GRAD_X */

    // Row2
    temp   = vload16(0, offset(&src, -1, 1));
    left   = convert_short8(temp.s01234567);
    middle = convert_short8(temp.s12345678);
    right  = convert_short8(temp.s23456789);
#ifdef GRAD_X
    gx += left * (short8)(-3);
    gx += right * (short8)(+3);
#endif /* GRAD_X */
#ifdef GRAD_Y
    gy += left * (short8)(+3);
    gy += middle * (short8)(+10);
    gy += right * (short8)(+3);
#endif /* GRAD_Y */

    // Store results
#ifdef GRAD_X
    vstore8(gx, 0, ((__global short *)dst_gx.ptr));
#endif /* GRAD_X */
#ifdef GRAD_Y
    vstore8(gy, 0, ((__global short *)dst_gy.ptr));
#endif /* GRAD_Y */
}
