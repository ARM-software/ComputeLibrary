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

/** This function performs table lookup on U8 input/output images.
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 *
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
 * @param[in]  lut                               LUT table. Supported data types: U8
 */
__kernel void tablelookup_U8(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    __global uchar *lut)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load input data */
    uchar8 data = vload8(0, src.ptr);

    /* Load lut data */
    uchar8 lut_data = (uchar8)(lut[data.s0], lut[data.s1], lut[data.s2], lut[data.s3],
                               lut[data.s4], lut[data.s5], lut[data.s6], lut[data.s7]);

    /* Store result */
    vstore8(lut_data, 0, dst.ptr);
}

/** This function performs table lookup on S16 input/output images.
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: S16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: S16
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  lut                               LUT table. Supported data types: S16
 * @param[in]  offset                            LUT offset
 * @param[in]  count                             Number of elements in the LUT
 */
__kernel void tablelookup_S16(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst),
    __global short *lut,
    uint            offset,
    uint            count)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    /* Load input data */
    short8 data = vload8(0, (__global short *)src.ptr);

    /* Load output data */
    int8 out_data = convert_int8(vload8(0, (__global short *)dst.ptr));

    /* Calculate index */
    int8 index = convert_int8(data) + (int8)(offset);
    int8 cond  = (index >= 0 && index < (int8)count);
    index      = select(0, index, cond);

    /* Load lut data */
    int8 lut_data = (int8)(lut[index.s0], lut[index.s1], lut[index.s2], lut[index.s3],
                           lut[index.s4], lut[index.s5], lut[index.s6], lut[index.s7]);

    /* Select output data depending on condition */
    lut_data = select(out_data, lut_data, cond);

    /* Store result */
    vstore8(convert_short8(lut_data), 0, (__global short *)dst.ptr);
}
