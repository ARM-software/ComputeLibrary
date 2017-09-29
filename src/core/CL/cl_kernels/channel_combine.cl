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

/** This function combines three planes to a single RGB image.
 *
 * @param[in] plane0_ptr                           Pointer to the first plane. Supported Format: U8
 * @param[in] plane0_stride_x                      Stride of the first plane in X dimension (in bytes)
 * @param[in] plane0_step_x                        plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane0_stride_y                      Stride of the first plane in Y dimension (in bytes)
 * @param[in] plane0_step_y                        plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane0_offset_first_element_in_bytes The offset of the first element in the first plane
 * @param[in] plane1_ptr                           Pointer to the second plane. Supported Format: U8
 * @param[in] plane1_stride_x                      Stride of the second plane in X dimension (in bytes)
 * @param[in] plane1_step_x                        plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane1_stride_y                      Stride of the second plane in Y dimension (in bytes)
 * @param[in] plane1_step_y                        plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane1_offset_first_element_in_bytes The offset of the first element in the second plane
 * @param[in] plane2_ptr                           Pointer to the third plane. Supported Format: U8
 * @param[in] plane2_stride_x                      Stride of the third plane in X dimension (in bytes)
 * @param[in] plane2_step_x                        plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane2_stride_y                      Stride of the third plane in Y dimension (in bytes)
 * @param[in] plane2_step_y                        plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane2_offset_first_element_in_bytes The offset of the first element in the third plane
 * @param[in] dst_ptr                              Pointer to the destination image. Supported Format: RGB
 * @param[in] dst_stride_x                         Stride of the destination image in X dimension (in bytes)
 * @param[in] dst_step_x                           dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                         Stride of the destination image in Y dimension (in bytes)
 * @param[in] dst_step_y                           dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes    The offset of the first element in the destination image
 */
__kernel void channel_combine_RGB888(
    IMAGE_DECLARATION(plane0),
    IMAGE_DECLARATION(plane1),
    IMAGE_DECLARATION(plane2),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image plane0 = CONVERT_TO_IMAGE_STRUCT(plane0);
    Image plane1 = CONVERT_TO_IMAGE_STRUCT(plane1);
    Image plane2 = CONVERT_TO_IMAGE_STRUCT(plane2);
    Image dst    = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data0 = vload16(0, plane0.ptr);
    uchar16 data1 = vload16(0, plane1.ptr);
    uchar16 data2 = vload16(0, plane2.ptr);

    uchar16 out0 = (uchar16)(data0.s0, data1.s0, data2.s0,
                             data0.s1, data1.s1, data2.s1,
                             data0.s2, data1.s2, data2.s2,
                             data0.s3, data1.s3, data2.s3,
                             data0.s4, data1.s4, data2.s4,
                             data0.s5);
    vstore16(out0, 0, dst.ptr);

    uchar16 out1 = (uchar16)(data1.s5, data2.s5, data0.s6,
                             data1.s6, data2.s6, data0.s7,
                             data1.s7, data2.s7, data0.s8,
                             data1.s8, data2.s8, data0.s9,
                             data1.s9, data2.s9, data0.sA,
                             data1.sA);
    vstore16(out1, 0, dst.ptr + 16);

    uchar16 out2 = (uchar16)(data2.sA, data0.sB, data1.sB,
                             data2.sB, data0.sC, data1.sC,
                             data2.sC, data0.sD, data1.sD,
                             data2.sD, data0.sE, data1.sE,
                             data2.sE, data0.sF, data1.sF,
                             data2.sF);
    vstore16(out2, 0, dst.ptr + 32);
}

/** This function combines three planes to a single RGBA image.
 *
 * @param[in] plane0_ptr                           Pointer to the first plane. Supported Format: U8
 * @param[in] plane0_stride_x                      Stride of the first plane in X dimension (in bytes)
 * @param[in] plane0_step_x                        plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane0_stride_y                      Stride of the first plane in Y dimension (in bytes)
 * @param[in] plane0_step_y                        plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane0_offset_first_element_in_bytes The offset of the first element in the first plane
 * @param[in] plane1_ptr                           Pointer to the second plane. Supported Format: U8
 * @param[in] plane1_stride_x                      Stride of the second plane in X dimension (in bytes)
 * @param[in] plane1_step_x                        plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane1_stride_y                      Stride of the second plane in Y dimension (in bytes)
 * @param[in] plane1_step_y                        plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane1_offset_first_element_in_bytes The offset of the first element in the second plane
 * @param[in] plane2_ptr                           Pointer to the third plane. Supported Format: U8
 * @param[in] plane2_stride_x                      Stride of the third plane in X dimension (in bytes)
 * @param[in] plane2_step_x                        plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane2_stride_y                      Stride of the third plane in Y dimension (in bytes)
 * @param[in] plane2_step_y                        plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane2_offset_first_element_in_bytes The offset of the first element in the third plane
 * @param[in] plane3_ptr                           Pointer to the fourth plane. Supported Format: U8
 * @param[in] plane3_stride_x                      Stride of the fourth plane in X dimension (in bytes)
 * @param[in] plane3_step_x                        plane3_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane3_stride_y                      Stride of the fourth plane in Y dimension (in bytes)
 * @param[in] plane3_step_y                        plane3_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane3_offset_first_element_in_bytes The offset of the first element in the fourth plane
 * @param[in] dst_ptr                              Pointer to the destination image. Supported Format: RGBA
 * @param[in] dst_stride_x                         Stride of the destination image in X dimension (in bytes)
 * @param[in] dst_step_x                           dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                         Stride of the destination image in Y dimension (in bytes)
 * @param[in] dst_step_y                           dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes    The offset of the first element in the destination image
 */
__kernel void channel_combine_RGBA8888(
    IMAGE_DECLARATION(plane0),
    IMAGE_DECLARATION(plane1),
    IMAGE_DECLARATION(plane2),
    IMAGE_DECLARATION(plane3),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image plane0 = CONVERT_TO_IMAGE_STRUCT(plane0);
    Image plane1 = CONVERT_TO_IMAGE_STRUCT(plane1);
    Image plane2 = CONVERT_TO_IMAGE_STRUCT(plane2);
    Image plane3 = CONVERT_TO_IMAGE_STRUCT(plane3);
    Image dst    = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data0 = vload16(0, plane0.ptr);
    uchar16 data1 = vload16(0, plane1.ptr);
    uchar16 data2 = vload16(0, plane2.ptr);
    uchar16 data3 = vload16(0, plane3.ptr);

    uchar16 out0 = (uchar16)(data0.s0, data1.s0, data2.s0, data3.s0,
                             data0.s1, data1.s1, data2.s1, data3.s1,
                             data0.s2, data1.s2, data2.s2, data3.s2,
                             data0.s3, data1.s3, data2.s3, data3.s3);
    vstore16(out0, 0, dst.ptr);

    uchar16 out1 = (uchar16)(data0.s4, data1.s4, data2.s4, data3.s4,
                             data0.s5, data1.s5, data2.s5, data3.s5,
                             data0.s6, data1.s6, data2.s6, data3.s6,
                             data0.s7, data1.s7, data2.s7, data3.s7);
    vstore16(out1, 0, dst.ptr + 16);

    uchar16 out2 = (uchar16)(data0.s8, data1.s8, data2.s8, data3.s8,
                             data0.s9, data1.s9, data2.s9, data3.s9,
                             data0.sA, data1.sA, data2.sA, data3.sA,
                             data0.sB, data1.sB, data2.sB, data3.sB);
    vstore16(out2, 0, dst.ptr + 32);

    uchar16 out3 = (uchar16)(data0.sC, data1.sC, data2.sC, data3.sC,
                             data0.sD, data1.sD, data2.sD, data3.sD,
                             data0.sE, data1.sE, data2.sE, data3.sE,
                             data0.sF, data1.sF, data2.sF, data3.sF);
    vstore16(out3, 0, dst.ptr + 48);
}

/** This function combines three planes to a single YUYV image.
 *
 * @param[in] plane0_ptr                           Pointer to the first plane. Supported Format: U8
 * @param[in] plane0_stride_x                      Stride of the first plane in X dimension (in bytes)
 * @param[in] plane0_step_x                        plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane0_stride_y                      Stride of the first plane in Y dimension (in bytes)
 * @param[in] plane0_step_y                        plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane0_offset_first_element_in_bytes The offset of the first element in the first plane
 * @param[in] plane1_ptr                           Pointer to the second plane. Supported Format: U8
 * @param[in] plane1_stride_x                      Stride of the second plane in X dimension (in bytes)
 * @param[in] plane1_step_x                        plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane1_stride_y                      Stride of the second plane in Y dimension (in bytes)
 * @param[in] plane1_step_y                        plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane1_offset_first_element_in_bytes The offset of the first element in the second plane
 * @param[in] plane2_ptr                           Pointer to the third plane. Supported Format: U8
 * @param[in] plane2_stride_x                      Stride of the third plane in X dimension (in bytes)
 * @param[in] plane2_step_x                        plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane2_stride_y                      Stride of the third plane in Y dimension (in bytes)
 * @param[in] plane2_step_y                        plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane2_offset_first_element_in_bytes The offset of the first element in the third plane
 * @param[in] dst_ptr                              Pointer to the destination image. Supported Format: YUYV
 * @param[in] dst_stride_x                         Stride of the destination image in X dimension (in bytes)
 * @param[in] dst_step_x                           dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                         Stride of the destination image in Y dimension (in bytes)
 * @param[in] dst_step_y                           dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes    The offset of the first element in the destination image
 */
__kernel void channel_combine_YUYV422(
    IMAGE_DECLARATION(plane0),
    IMAGE_DECLARATION(plane1),
    IMAGE_DECLARATION(plane2),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image plane0 = CONVERT_TO_IMAGE_STRUCT(plane0);
    Image plane1 = CONVERT_TO_IMAGE_STRUCT(plane1);
    Image plane2 = CONVERT_TO_IMAGE_STRUCT(plane2);
    Image dst    = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data0 = vload16(0, plane0.ptr);
    uchar8  data1 = vload8(0, plane1.ptr);
    uchar8  data2 = vload8(0, plane2.ptr);

    uchar16 out0 = (uchar16)(data0.s0, data1.s0, data0.s1, data2.s0,
                             data0.s2, data1.s1, data0.s3, data2.s1,
                             data0.s4, data1.s2, data0.s5, data2.s2,
                             data0.s6, data1.s3, data0.s7, data2.s3);
    vstore16(out0, 0, dst.ptr);
    uchar16 out1 = (uchar16)(data0.s8, data1.s4, data0.s9, data2.s4,
                             data0.sA, data1.s5, data0.sB, data2.s5,
                             data0.sC, data1.s6, data0.sD, data2.s6,
                             data0.sE, data1.s7, data0.sF, data2.s7);
    vstore16(out1, 0, dst.ptr + 16);
}

/** This function combines three planes to a single UYUV image.
 *
 * @param[in] plane0_ptr                           Pointer to the first plane. Supported Format: U8
 * @param[in] plane0_stride_x                      Stride of the first plane in X dimension (in bytes)
 * @param[in] plane0_step_x                        plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane0_stride_y                      Stride of the first plane in Y dimension (in bytes)
 * @param[in] plane0_step_y                        plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane0_offset_first_element_in_bytes The offset of the first element in the first plane
 * @param[in] plane1_ptr                           Pointer to the second plane. Supported Format: U8
 * @param[in] plane1_stride_x                      Stride of the second plane in X dimension (in bytes)
 * @param[in] plane1_step_x                        plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane1_stride_y                      Stride of the second plane in Y dimension (in bytes)
 * @param[in] plane1_step_y                        plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane1_offset_first_element_in_bytes The offset of the first element in the second plane
 * @param[in] plane2_ptr                           Pointer to the third plane. Supported Format: U8
 * @param[in] plane2_stride_x                      Stride of the third plane in X dimension (in bytes)
 * @param[in] plane2_step_x                        plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] plane2_stride_y                      Stride of the third plane in Y dimension (in bytes)
 * @param[in] plane2_step_y                        plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] plane2_offset_first_element_in_bytes The offset of the first element in the third plane
 * @param[in] dst_ptr                              Pointer to the destination image. Supported Format: UYUV
 * @param[in] dst_stride_x                         Stride of the destination image in X dimension (in bytes)
 * @param[in] dst_step_x                           dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_stride_y                         Stride of the destination image in Y dimension (in bytes)
 * @param[in] dst_step_y                           dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_offset_first_element_in_bytes    The offset of the first element in the destination image
 */
__kernel void channel_combine_UYVY422(
    IMAGE_DECLARATION(plane0),
    IMAGE_DECLARATION(plane1),
    IMAGE_DECLARATION(plane2),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image plane0 = CONVERT_TO_IMAGE_STRUCT(plane0);
    Image plane1 = CONVERT_TO_IMAGE_STRUCT(plane1);
    Image plane2 = CONVERT_TO_IMAGE_STRUCT(plane2);
    Image dst    = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data0 = vload16(0, plane0.ptr);
    uchar8  data1 = vload8(0, plane1.ptr);
    uchar8  data2 = vload8(0, plane2.ptr);

    uchar16 out0 = (uchar16)(data1.s0, data0.s0, data2.s0, data0.s1,
                             data1.s1, data0.s2, data2.s1, data0.s3,
                             data1.s2, data0.s4, data2.s2, data0.s5,
                             data1.s3, data0.s6, data2.s3, data0.s7);
    vstore16(out0, 0, dst.ptr);
    uchar16 out1 = (uchar16)(data1.s4, data0.s8, data2.s4, data0.s9,
                             data1.s5, data0.sA, data2.s5, data0.sB,
                             data1.s6, data0.sC, data2.s6, data0.sD,
                             data1.s7, data0.sE, data2.s7, data0.sF);
    vstore16(out1, 0, dst.ptr + 16);
}

/** This function combines three planes to a single NV12/NV21 image.
 *
 * @note NV12 or NV21 has to be specified through preprocessor macro. eg. -DNV12 performs NV12 channel combine.
 *
 * @param[in] src_plane0_ptr                           Pointer to the first plane. Supported Format: U8
 * @param[in] src_plane0_stride_x                      Stride of the first plane in X dimension (in bytes)
 * @param[in] src_plane0_step_x                        src_plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_plane0_stride_y                      Stride of the first plane in Y dimension (in bytes)
 * @param[in] src_plane0_step_y                        src_plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_plane0_offset_first_element_in_bytes The offset of the first element in the first plane
 * @param[in] src_plane1_ptr                           Pointer to the second plane. Supported Format: U8
 * @param[in] src_plane1_stride_x                      Stride of the second plane in X dimension (in bytes)
 * @param[in] src_plane1_step_x                        src_plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_plane1_stride_y                      Stride of the second plane in Y dimension (in bytes)
 * @param[in] src_plane1_step_y                        src_plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_plane1_offset_first_element_in_bytes The offset of the first element in the second plane
 * @param[in] src_plane2_ptr                           Pointer to the third plane. Supported Format: U8
 * @param[in] src_plane2_stride_x                      Stride of the third plane in X dimension (in bytes)
 * @param[in] src_plane2_step_x                        src_plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_plane2_stride_y                      Stride of the third plane in Y dimension (in bytes)
 * @param[in] src_plane2_step_y                        src_plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_plane2_offset_first_element_in_bytes The offset of the first element in the third plane
 * @param[in] dst_plane0_ptr                           Pointer to the first plane of the destination image. Supported Format: U8
 * @param[in] dst_plane0_stride_x                      Stride of the first plane of the destination image in X dimension (in bytes)
 * @param[in] dst_plane0_step_x                        dst_plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_plane0_stride_y                      Stride of the first plane of the destination image in Y dimension (in bytes)
 * @param[in] dst_plane0_step_y                        dst_plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_plane0_offset_first_element_in_bytes The offset of the first element in the first plane of the destination image
 * @param[in] dst_plane1_ptr                           Pointer to the second plane of the destination image. Supported Format: UV88
 * @param[in] dst_plane1_stride_x                      Stride of the second plane of the destination image in X dimension (in bytes)
 * @param[in] dst_plane1_step_x                        dst_plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_plane1_stride_y                      Stride of the second plane of the destination image in Y dimension (in bytes)
 * @param[in] dst_plane1_step_y                        dst_plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_plane1_offset_first_element_in_bytes The offset of the first element in the second plane of the destination image
 * @param[in] height                                   Sub-sampled height
 */
__kernel void channel_combine_NV(
    IMAGE_DECLARATION(src_plane0),
    IMAGE_DECLARATION(src_plane1),
    IMAGE_DECLARATION(src_plane2),
    IMAGE_DECLARATION(dst_plane0),
    IMAGE_DECLARATION(dst_plane1),
    uint height)
{
    // Get pixels pointer
    Image src_plane0 = CONVERT_TO_IMAGE_STRUCT(src_plane0);
    Image src_plane1 = CONVERT_TO_IMAGE_STRUCT(src_plane1);
    Image src_plane2 = CONVERT_TO_IMAGE_STRUCT(src_plane2);
    Image dst_plane0 = CONVERT_TO_IMAGE_STRUCT(dst_plane0);
    Image dst_plane1 = CONVERT_TO_IMAGE_STRUCT(dst_plane1);

    // Copy plane data
    vstore16(vload16(0, src_plane0.ptr), 0, dst_plane0.ptr);
    vstore16(vload16(0, offset(&src_plane0, 0, height)), 0, (__global uchar *)offset(&dst_plane0, 0, height));

    // Create UV place
    uchar8 data1 = vload8(0, src_plane1.ptr);
    uchar8 data2 = vload8(0, src_plane2.ptr);

#ifdef NV12
    vstore16(shuffle2(data1, data2, (uchar16)(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)), 0, dst_plane1.ptr);
#elif defined(NV21)
    vstore16(shuffle2(data2, data1, (uchar16)(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)), 0, dst_plane1.ptr);
#endif /* NV12 or NV21 */
}

/** This function combines three planes to a single YUV444 or IYUV image.
 *
 * @note YUV444 or IYUV has to be specified through preprocessor macro. eg. -DIYUV performs IYUV channel combine.
 *
 * @param[in] src_plane0_ptr                           Pointer to the first plane. Supported Format: U8
 * @param[in] src_plane0_stride_x                      Stride of the first plane in X dimension (in bytes)
 * @param[in] src_plane0_step_x                        src_plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_plane0_stride_y                      Stride of the first plane in Y dimension (in bytes)
 * @param[in] src_plane0_step_y                        src_plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_plane0_offset_first_element_in_bytes The offset of the first element in the first plane
 * @param[in] src_plane1_ptr                           Pointer to the second plane. Supported Format: U8
 * @param[in] src_plane1_stride_x                      Stride of the second plane in X dimension (in bytes)
 * @param[in] src_plane1_step_x                        src_plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_plane1_stride_y                      Stride of the second plane in Y dimension (in bytes)
 * @param[in] src_plane1_step_y                        src_plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_plane1_offset_first_element_in_bytes The offset of the first element in the second plane
 * @param[in] src_plane2_ptr                           Pointer to the third plane. Supported Format: U8
 * @param[in] src_plane2_stride_x                      Stride of the third plane in X dimension (in bytes)
 * @param[in] src_plane2_step_x                        src_plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_plane2_stride_y                      Stride of the third plane in Y dimension (in bytes)
 * @param[in] src_plane2_step_y                        src_plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] src_plane2_offset_first_element_in_bytes The offset of the first element in the third plane
 * @param[in] dst_plane0_ptr                           Pointer to the first plane of the destination image. Supported Format: U8
 * @param[in] dst_plane0_stride_x                      Stride of the first plane of the destination image in X dimension (in bytes)
 * @param[in] dst_plane0_step_x                        dst_plane0_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_plane0_stride_y                      Stride of the first plane of the destination image in Y dimension (in bytes)
 * @param[in] dst_plane0_step_y                        dst_plane0_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_plane0_offset_first_element_in_bytes The offset of the first element in the first plane of the destination image
 * @param[in] dst_plane1_ptr                           Pointer to the second plane of the destination image. Supported Format: U8
 * @param[in] dst_plane1_stride_x                      Stride of the second plane of the destination image in X dimension (in bytes)
 * @param[in] dst_plane1_step_x                        dst_plane1_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_plane1_stride_y                      Stride of the second plane of the destination image in Y dimension (in bytes)
 * @param[in] dst_plane1_step_y                        dst_plane1_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_plane1_offset_first_element_in_bytes The offset of the first element in the second plane of the destination image
 * @param[in] dst_plane2_ptr                           Pointer to the third plane of the destination image. Supported Format: U8
 * @param[in] dst_plane2_stride_x                      Stride of the third plane of the destination image in X dimension (in bytes)
 * @param[in] dst_plane2_step_x                        dst_plane2_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] dst_plane2_stride_y                      Stride of the third plane of the destination image in Y dimension (in bytes)
 * @param[in] dst_plane2_step_y                        dst_plane2_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in] dst_plane2_offset_first_element_in_bytes The offset of the first element in the third plane of the destination image
 * @param[in] height                                   Sub-sampled height
 */
__kernel void copy_planes_3p(
    IMAGE_DECLARATION(src_plane0),
    IMAGE_DECLARATION(src_plane1),
    IMAGE_DECLARATION(src_plane2),
    IMAGE_DECLARATION(dst_plane0),
    IMAGE_DECLARATION(dst_plane1),
    IMAGE_DECLARATION(dst_plane2),
    uint height)
{
    // Get pixels pointer
    Image src_plane0 = CONVERT_TO_IMAGE_STRUCT(src_plane0);
    Image src_plane1 = CONVERT_TO_IMAGE_STRUCT(src_plane1);
    Image src_plane2 = CONVERT_TO_IMAGE_STRUCT(src_plane2);
    Image dst_plane0 = CONVERT_TO_IMAGE_STRUCT(dst_plane0);
    Image dst_plane1 = CONVERT_TO_IMAGE_STRUCT(dst_plane1);
    Image dst_plane2 = CONVERT_TO_IMAGE_STRUCT(dst_plane2);

    // Copy plane data
    vstore16(vload16(0, src_plane0.ptr), 0, dst_plane0.ptr);
#ifdef YUV444
    vstore16(vload16(0, src_plane1.ptr), 0, dst_plane1.ptr);
    vstore16(vload16(0, src_plane2.ptr), 0, dst_plane2.ptr);
#elif defined(IYUV)
    vstore16(vload16(0, offset(&src_plane0, 0, height)), 0, (__global uchar *)offset(&dst_plane0, 0, height));
    vstore8(vload8(0, src_plane1.ptr), 0, dst_plane1.ptr);
    vstore8(vload8(0, src_plane2.ptr), 0, dst_plane2.ptr);
#endif /* YUV444 or IYUV */
}
