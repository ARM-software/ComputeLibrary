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
#include "non_linear_filter_helpers.h"

// Sorting networks below were generated using http://pages.ripco.net/~jgamble/nw.html

/** Sorting network to sort 8 disks of diameter 5 and return their median.
 *
 * @param[in] top2    Values of elements two rows above.
 * @param[in] top     Values of elements one row above.
 * @param[in] middle  Values of middle elements.
 * @param[in] bottom  Values of elements one row below.
 * @param[in] bottom2 Values of elements two rows below.
 *
 * @return Median values for 8 elements.
 */
inline uchar8 median_disk5x5(uchar16 top2, uchar16 top, uchar16 middle, uchar16 bottom, uchar16 bottom2)
{
    uchar8 p0  = top2.s01234567;
    uchar8 p1  = top2.s12345678;
    uchar8 p2  = top2.s23456789;
    uchar8 p3  = top.s01234567;
    uchar8 p4  = top.s12345678;
    uchar8 p5  = top.s23456789;
    uchar8 p6  = top.s3456789A;
    uchar8 p7  = top.s456789AB;
    uchar8 p8  = middle.s01234567;
    uchar8 p9  = middle.s12345678;
    uchar8 p10 = middle.s23456789;
    uchar8 p11 = middle.s3456789A;
    uchar8 p12 = middle.s456789AB;
    uchar8 p13 = bottom.s01234567;
    uchar8 p14 = bottom.s12345678;
    uchar8 p15 = bottom.s23456789;
    uchar8 p16 = bottom.s3456789A;
    uchar8 p17 = bottom.s456789AB;
    uchar8 p18 = bottom2.s01234567;
    uchar8 p19 = bottom2.s12345678;
    uchar8 p20 = bottom2.s23456789;

    SORT(p0, p1);
    SORT(p2, p3);
    SORT(p4, p5);
    SORT(p6, p7);
    SORT(p8, p9);
    SORT(p10, p11);
    SORT(p12, p13);
    SORT(p14, p15);
    SORT(p16, p17);
    SORT(p18, p19);
    SORT(p0, p2);
    SORT(p1, p3);
    SORT(p4, p6);
    SORT(p5, p7);
    SORT(p8, p10);
    SORT(p9, p11);
    SORT(p12, p14);
    SORT(p13, p15);
    SORT(p16, p18);
    SORT(p17, p19);
    SORT(p1, p2);
    SORT(p5, p6);
    SORT(p0, p4);
    SORT(p3, p7);
    SORT(p9, p10);
    SORT(p13, p14);
    SORT(p8, p12);
    SORT(p11, p15);
    SORT(p17, p18);
    SORT(p16, p20);
    SORT(p1, p5);
    SORT(p2, p6);
    SORT(p9, p13);
    SORT(p10, p14);
    SORT(p0, p8);
    SORT(p7, p15);
    SORT(p17, p20);
    SORT(p1, p4);
    SORT(p3, p6);
    SORT(p9, p12);
    SORT(p11, p14);
    SORT(p18, p20);
    SORT(p0, p16);
    SORT(p2, p4);
    SORT(p3, p5);
    SORT(p10, p12);
    SORT(p11, p13);
    SORT(p1, p9);
    SORT(p6, p14);
    SORT(p19, p20);
    SORT(p3, p4);
    SORT(p11, p12);
    SORT(p1, p8);
    SORT(p2, p10);
    SORT(p5, p13);
    SORT(p7, p14);
    SORT(p3, p11);
    SORT(p2, p8);
    SORT(p4, p12);
    SORT(p7, p13);
    SORT(p1, p17);
    SORT(p3, p10);
    SORT(p5, p12);
    SORT(p1, p16);
    SORT(p2, p18);
    SORT(p3, p9);
    SORT(p6, p12);
    SORT(p2, p16);
    SORT(p3, p8);
    SORT(p7, p12);
    SORT(p5, p9);
    SORT(p6, p10);
    SORT(p4, p8);
    SORT(p7, p11);
    SORT(p3, p19);
    SORT(p5, p8);
    SORT(p7, p10);
    SORT(p3, p18);
    SORT(p4, p20);
    SORT(p6, p8);
    SORT(p7, p9);
    SORT(p3, p17);
    SORT(p5, p20);
    SORT(p7, p8);
    SORT(p3, p16);
    SORT(p6, p20);
    SORT(p5, p17);
    SORT(p7, p20);
    SORT(p4, p16);
    SORT(p6, p18);
    SORT(p5, p16);
    SORT(p7, p19);
    SORT(p7, p18);
    SORT(p6, p16);
    SORT(p7, p17);
    SORT(p10, p18);
    SORT(p7, p16);
    SORT(p9, p17);
    SORT(p8, p16);
    SORT(p9, p16);
    SORT(p10, p16);

    return p10;
}

/** Sorting network to sort 8 boxes of size 5 and return their median.
 *
 * @param[in] top2    Values of elements two rows above.
 * @param[in] top     Values of elements one row above.
 * @param[in] middle  Values of middle elements.
 * @param[in] bottom  Values of elements one row below.
 * @param[in] bottom2 Values of elements two rows below.
 *
 * @return Median values for 8 elements.
 */
inline uchar8 median_box5x5(uchar16 top2, uchar16 top, uchar16 middle, uchar16 bottom, uchar16 bottom2)
{
    uchar8 p0  = top2.s01234567;
    uchar8 p1  = top2.s12345678;
    uchar8 p2  = top2.s23456789;
    uchar8 p3  = top2.s3456789A;
    uchar8 p4  = top2.s456789AB;
    uchar8 p5  = top.s01234567;
    uchar8 p6  = top.s12345678;
    uchar8 p7  = top.s23456789;
    uchar8 p8  = top.s3456789A;
    uchar8 p9  = top.s456789AB;
    uchar8 p10 = middle.s01234567;
    uchar8 p11 = middle.s12345678;
    uchar8 p12 = middle.s23456789;
    uchar8 p13 = middle.s3456789A;
    uchar8 p14 = middle.s456789AB;
    uchar8 p15 = bottom.s01234567;
    uchar8 p16 = bottom.s12345678;
    uchar8 p17 = bottom.s23456789;
    uchar8 p18 = bottom.s3456789A;
    uchar8 p19 = bottom.s456789AB;
    uchar8 p20 = bottom2.s01234567;
    uchar8 p21 = bottom2.s12345678;
    uchar8 p22 = bottom2.s23456789;
    uchar8 p23 = bottom2.s3456789A;
    uchar8 p24 = bottom2.s456789AB;

    SORT(p1, p2);
    SORT(p0, p1);
    SORT(p1, p2);
    SORT(p4, p5);
    SORT(p3, p4);
    SORT(p4, p5);
    SORT(p0, p3);
    SORT(p2, p5);
    SORT(p2, p3);
    SORT(p1, p4);
    SORT(p1, p2);
    SORT(p3, p4);
    SORT(p7, p8);
    SORT(p6, p7);
    SORT(p7, p8);
    SORT(p10, p11);
    SORT(p9, p10);
    SORT(p10, p11);
    SORT(p6, p9);
    SORT(p8, p11);
    SORT(p8, p9);
    SORT(p7, p10);
    SORT(p7, p8);
    SORT(p9, p10);
    SORT(p0, p6);
    SORT(p4, p10);
    SORT(p4, p6);
    SORT(p2, p8);
    SORT(p2, p4);
    SORT(p6, p8);
    SORT(p1, p7);
    SORT(p5, p11);
    SORT(p5, p7);
    SORT(p3, p9);
    SORT(p3, p5);
    SORT(p7, p9);
    SORT(p1, p2);
    SORT(p3, p4);
    SORT(p5, p6);
    SORT(p7, p8);
    SORT(p9, p10);
    SORT(p13, p14);
    SORT(p12, p13);
    SORT(p13, p14);
    SORT(p16, p17);
    SORT(p15, p16);
    SORT(p16, p17);
    SORT(p12, p15);
    SORT(p14, p17);
    SORT(p14, p15);
    SORT(p13, p16);
    SORT(p13, p14);
    SORT(p15, p16);
    SORT(p19, p20);
    SORT(p18, p19);
    SORT(p19, p20);
    SORT(p21, p22);
    SORT(p23, p24);
    SORT(p21, p23);
    SORT(p22, p24);
    SORT(p22, p23);
    SORT(p18, p21);
    SORT(p20, p23);
    SORT(p20, p21);
    SORT(p19, p22);
    SORT(p22, p24);
    SORT(p19, p20);
    SORT(p21, p22);
    SORT(p23, p24);
    SORT(p12, p18);
    SORT(p16, p22);
    SORT(p16, p18);
    SORT(p14, p20);
    SORT(p20, p24);
    SORT(p14, p16);
    SORT(p18, p20);
    SORT(p22, p24);
    SORT(p13, p19);
    SORT(p17, p23);
    SORT(p17, p19);
    SORT(p15, p21);
    SORT(p15, p17);
    SORT(p19, p21);
    SORT(p13, p14);
    SORT(p15, p16);
    SORT(p17, p18);
    SORT(p19, p20);
    SORT(p21, p22);
    SORT(p23, p24);
    SORT(p0, p12);
    SORT(p8, p20);
    SORT(p8, p12);
    SORT(p4, p16);
    SORT(p16, p24);
    SORT(p12, p16);
    SORT(p2, p14);
    SORT(p10, p22);
    SORT(p10, p14);
    SORT(p6, p18);
    SORT(p6, p10);
    SORT(p10, p12);
    SORT(p1, p13);
    SORT(p9, p21);
    SORT(p9, p13);
    SORT(p5, p17);
    SORT(p13, p17);
    SORT(p3, p15);
    SORT(p11, p23);
    SORT(p11, p15);
    SORT(p7, p19);
    SORT(p7, p11);
    SORT(p11, p13);
    SORT(p11, p12);
    return p12;
}

/** This function applies a non linear filter on a 5x5 box basis on an input image.
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
__kernel void non_linear_filter_box5x5(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    uchar16 top2    = vload16(0, offset(&src, -2, -2));
    uchar16 top     = vload16(0, offset(&src, -2, -1));
    uchar16 middle  = vload16(0, offset(&src, -2, 0));
    uchar16 bottom  = vload16(0, offset(&src, -2, 1));
    uchar16 bottom2 = vload16(0, offset(&src, -2, 2));

    // Apply respective filter
#ifdef MIN
    uchar16 tmp = min(middle, min(min(top2, top), min(bottom, bottom2)));
    uchar8  out = row_reduce_min_5(tmp);
#elif defined(MAX)
    uchar16 tmp = max(middle, max(max(top2, top), max(bottom, bottom2)));
    uchar8  out = row_reduce_max_5(tmp);
#elif defined(MEDIAN)
    uchar8 out = median_box5x5(top2, top, middle, bottom, bottom2);
#else /* MIN or MAX or MEDIAN */
#error "Unsupported filter function"
#endif /* MIN or MAX or MEDIAN */

    // Store result
    vstore8(out, 0, dst.ptr);
}

/** This function applies a non linear filter on a 5x5 cross basis on an input image.
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
__kernel void non_linear_filter_cross5x5(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    uchar8  top2    = vload8(0, offset(&src, 0, -2));
    uchar8  top     = vload8(0, offset(&src, 0, -1));
    uchar16 middle  = vload16(0, offset(&src, -2, 0));
    uchar8  bottom  = vload8(0, offset(&src, 0, 1));
    uchar8  bottom2 = vload8(0, offset(&src, 0, 2));

    // Apply respective filter
#ifdef MIN
    uchar8 tmp_middle = row_reduce_min_5(middle);
    uchar8 out        = min(tmp_middle, min(min(top2, top), min(bottom, bottom2)));
#elif defined(MAX)
    uchar8  tmp_middle = row_reduce_max_5(middle);
    uchar8  out        = max(tmp_middle, max(max(top2, top.s01234567), max(bottom, bottom2)));
#elif defined(MEDIAN)
    uchar8 p0  = top2;
    uchar8 p1  = top;
    uchar8 p2  = middle.s01234567;
    uchar8 p3  = middle.s12345678;
    uchar8 p4  = middle.s23456789;
    uchar8 p5  = middle.s3456789A;
    uchar8 p6  = middle.s456789AB;
    uchar8 p7  = bottom;
    uchar8 p8  = bottom2;
    uchar8 out = sort9(p0, p1, p2, p3, p4, p5, p6, p7, p8);
#else /* MIN or MAX or MEDIAN */
#error "Unsupported filter function"
#endif /* MIN or MAX or MEDIAN */

    // Store result
    vstore8(out, 0, dst.ptr);
}

/** This function applies a non linear filter on a 5x5 disk basis on an input image.
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
__kernel void non_linear_filter_disk5x5(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Load values
    uchar16 top2    = vload16(0, offset(&src, -2, -2));
    uchar16 top     = vload16(0, offset(&src, -2, -1));
    uchar16 middle  = vload16(0, offset(&src, -2, 0));
    uchar16 bottom  = vload16(0, offset(&src, -2, 1));
    uchar16 bottom2 = vload16(0, offset(&src, -2, 2));

    // Shift top2 and bottom2 values
    top2    = top2.s123456789ABCDEFF;
    bottom2 = bottom2.s123456789ABCDEFF;

    // Apply respective filter
#ifdef MIN
    uchar16 tmp_3     = min(top2, bottom2);
    uchar16 tmp_5     = min(middle, min(top, bottom));
    uchar8  tmp_3_red = row_reduce_min_3(tmp_3);
    uchar8  tmp_5_red = row_reduce_min_5(tmp_5);
    uchar8  out       = min(tmp_3_red, tmp_5_red);
#elif defined(MAX)
    uchar16 tmp_3      = max(top2, bottom2);
    uchar16 tmp_5      = max(middle, max(top, bottom));
    uchar8  tmp_3_red  = row_reduce_max_3(tmp_3);
    uchar8  tmp_5_red  = row_reduce_max_5(tmp_5);
    uchar8  out        = max(tmp_3_red, tmp_5_red);
#elif defined(MEDIAN)
    uchar8 out = median_disk5x5(top2, top, middle, bottom, bottom2);
#else /* MIN or MAX or MEDIAN */
#error "Unsupported filter function"
#endif /* MIN or MAX or MEDIAN */

    // Store result
    vstore8(out, 0, dst.ptr);
}
