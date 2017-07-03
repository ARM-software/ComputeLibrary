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
#include "non_linear_filter_helpers.h"

/** This function applies a non linear filter on a 3x3 box basis on an input image.
 *
 * @note The needed filter operation is defined through the preprocessor by passing either -DMIN, -DMAX or -DMEDIAN.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
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
__kernel void non_linear_filter_box3x3(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    uchar16 top    = vload16(0, offset(&src, -1, -1));
    uchar16 middle = vload16(0, offset(&src, -1, 0));
    uchar16 bottom = vload16(0, offset(&src, -1, 1));

    // Apply respective filter
#ifdef MIN
    uchar16 tmp = min(top, min(middle, bottom));
    uchar8  out = row_reduce_min_3(tmp);
#elif defined(MAX)
    uchar16 tmp = max(top, max(middle, bottom));
    uchar8  out = row_reduce_max_3(tmp);
#elif defined(MEDIAN)
    uchar8 p0  = top.s01234567;
    uchar8 p1  = top.s12345678;
    uchar8 p2  = top.s23456789;
    uchar8 p3  = middle.s01234567;
    uchar8 p4  = middle.s12345678;
    uchar8 p5  = middle.s23456789;
    uchar8 p6  = bottom.s01234567;
    uchar8 p7  = bottom.s12345678;
    uchar8 p8  = bottom.s23456789;
    uchar8 out = sort9(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#else /* MIN or MAX or MEDIAN */
#error "Unsupported filter function"
#endif /* MIN or MAX or MEDIAN */

    // Store result
    vstore8(out, 0, dst.ptr);
}

/** This function applies a non linear filter on a 3x3 cross basis on an input image.
 *
 * @note The needed filter operation is defined through the preprocessor by passing either -DMIN, -DMAX or -DMEDIAN.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
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
__kernel void non_linear_filter_cross3x3(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    uchar8  top    = vload8(0, offset(&src, 0, -1));
    uchar16 middle = vload16(0, offset(&src, -1, 0));
    uchar8  bottom = vload8(0, offset(&src, 0, 1));

    // Apply respective filter
#ifdef MIN
    uchar8 tmp_middle = row_reduce_min_3(middle);
    uchar8 out        = min(tmp_middle, min(top, bottom));
#elif defined(MAX)
    uchar8  tmp_middle = row_reduce_max_3(middle);
    uchar8  out        = max(tmp_middle, max(top, bottom));
#elif defined(MEDIAN)
    uchar8 p0  = top.s01234567;
    uchar8 p1  = middle.s01234567;
    uchar8 p2  = middle.s12345678;
    uchar8 p3  = middle.s23456789;
    uchar8 p4  = bottom.s01234567;
    uchar8 out = sort5(p0, p1, p2, p3, p4);
#else /* MIN or MAX or MEDIAN */
#error "Unsupported filter function"
#endif /* MIN or MAX or MEDIAN */

    // Store result
    vstore8(out, 0, dst.ptr);
}

/** This function applies a non linear filter on a 3x3 disk basis on an input image.
 *
 * @note The needed filter operation is defined through the preprocessor by passing either -DMIN, -DMAX or -DMEDIAN.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
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
__kernel void non_linear_filter_disk3x3(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    uchar16 top    = vload16(0, offset(&src, -1, -1));
    uchar16 middle = vload16(0, offset(&src, -1, 0));
    uchar16 bottom = vload16(0, offset(&src, -1, 1));

    // Apply respective filter
#ifdef MIN
    uchar16 tmp = min(top, min(middle, bottom));
    uchar8  out = row_reduce_min_3(tmp);
#elif defined(MAX)
    uchar16 tmp        = max(top, max(middle, bottom));
    uchar8  out        = row_reduce_max_3(tmp);
#elif defined(MEDIAN)
    uchar8 p0  = top.s01234567;
    uchar8 p1  = top.s12345678;
    uchar8 p2  = top.s23456789;
    uchar8 p3  = middle.s01234567;
    uchar8 p4  = middle.s12345678;
    uchar8 p5  = middle.s23456789;
    uchar8 p6  = bottom.s01234567;
    uchar8 p7  = bottom.s12345678;
    uchar8 p8  = bottom.s23456789;
    uchar8 out = sort9(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#else /* MIN or MAX or MEDIAN */
#error "Unsupported filter function"
#endif /* MIN or MAX or MEDIAN */

    // Store result
    vstore8(out, 0, dst.ptr);
}
