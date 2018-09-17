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

/** This function computes the horizontal integral of the image.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: U32
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void integral_horizontal(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uint prev = 0;

    for(uint j = 0; j < src_step_x; j += 16)
    {
        barrier(CLK_GLOBAL_MEM_FENCE);
        uint16 res = convert_uint16(vload16(0, offset(&src, j, 0)));
        res.s0 += prev;
        res.s1 += res.s0;
        res.s2 += res.s1;
        res.s3 += res.s2;
        res.s4 += res.s3;
        res.s5 += res.s4;
        res.s6 += res.s5;
        res.s7 += res.s6;
        res.s8 += res.s7;
        res.s9 += res.s8;
        res.sA += res.s9;
        res.sB += res.sA;
        res.sC += res.sB;
        res.sD += res.sC;
        res.sE += res.sD;
        res.sF += res.sE;
        prev = res.sF;
        vstore16(res, 0, (__global uint *)offset(&dst, j, 0));
    }
}

/** This function computes the vertical integral of the image.
 *
 * @param[in,out] src_ptr                           Pointer to the source image. Supported data types: U32
 * @param[in]     src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]     src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]     src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]     src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]     src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]     height                            Image height.
 */
__kernel void integral_vertical(
    IMAGE_DECLARATION(src),
    uint height)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    uint8 prev = vload8(0, (__global uint *)offset(&src, 0, 0));
    for(uint j = 1; j < height; ++j)
    {
        barrier(CLK_GLOBAL_MEM_FENCE);
        uint8 res = vload8(0, (__global uint *)offset(&src, 0, j));
        res += prev;
        vstore8(res, 0, (__global uint *)offset(&src, 0, j));
        prev = res;
    }
}
