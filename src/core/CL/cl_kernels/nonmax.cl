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

/** This function performs Non maxima suppression over a 3x3 window on a given image.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: F32
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void non_max_suppression(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    VEC_DATA_TYPE(DATA_TYPE, 8)
    vc = vload8(0, (__global DATA_TYPE *)src.ptr);

    if(all(vc == (DATA_TYPE)0))
    {
        vstore8(0, 0, (__global DATA_TYPE *)dst.ptr);

        return;
    }

    VEC_DATA_TYPE(DATA_TYPE, 16)
    nc = vload16(0, (__global DATA_TYPE *)offset(&src, -1, -1));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    out = select((DATA_TYPE)0, vc, (vc >= nc.s01234567) && (vc >= nc.s12345678) && (vc >= nc.s23456789));

    nc  = vload16(0, (__global DATA_TYPE *)offset(&src, -1, 0));
    out = select((DATA_TYPE)0, out, (vc >= nc.s01234567) && (vc > nc.s23456789));

    nc  = vload16(0, (__global DATA_TYPE *)offset(&src, -1, +1));
    out = select((DATA_TYPE)0, out, (vc > nc.s01234567) && (vc > nc.s12345678) && (vc > nc.s23456789));

    vstore8(out, 0, (__global DATA_TYPE *)dst.ptr);
}
