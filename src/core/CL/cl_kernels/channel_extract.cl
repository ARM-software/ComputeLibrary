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

/** This function extracts a given channel from an RGB image.
 *
 * @note Channel to be extracted should be passed as a pre-processor argument, e.g. -DCHANNEL_B will extract the B channel.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: RGB
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void channel_extract_RGB888(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data  = vload16(0, src.ptr);
    uchar8  data2 = vload8(0, src.ptr + 16);

#ifdef CHANNEL_R
    vstore4(data.s0369, 0, dst.ptr);
    vstore4((uchar4)(data.sCF, data2.s25), 0, dst.ptr + 4);
#elif defined(CHANNEL_G)
    vstore4(data.s147A, 0, dst.ptr);
    vstore4((uchar4)(data.sD, data2.s036), 0, dst.ptr + 4);
#elif defined(CHANNEL_B)
    vstore4(data.s258B, 0, dst.ptr);
    vstore4((uchar4)(data.sE, data2.s147), 0, dst.ptr + 4);
#endif /* CHANNEL_R or CHANNEL_G or CHANNEL_B */
}

/** This function extracts a given channel from an RGBA image.
 *
 * @note Channel to be extracted should be passed as a pre-processor argument, e.g. -DCHANNEL_B will extract the B channel.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: RGBA
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void channel_extract_RGBA8888(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data  = vload16(0, src.ptr);
    uchar16 data2 = vload16(0, src.ptr + 16);

#ifdef CHANNEL_R
    vstore8((uchar8)(data.s048C, data2.s048C), 0, dst.ptr);
#elif defined(CHANNEL_G)
    vstore8((uchar8)(data.s159D, data2.s159D), 0, dst.ptr);
#elif defined(CHANNEL_B)
    vstore8((uchar8)(data.s26AE, data2.s26AE), 0, dst.ptr);
#elif defined(CHANNEL_A)
    vstore8((uchar8)(data.s37BF, data2.s37BF), 0, dst.ptr);
#endif /* CHANNEL_R or CHANNEL_G or CHANNEL_B or CHANNEL_A */
}

/** This function extracts a given channel from an YUYV image.
 *
 * @note Channel to be extracted should be passed as a pre-processor argument, e.g. -DCHANNEL_U will extract the U channel.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: YUYV
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void channel_extract_YUYV422(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data = vload16(0, src.ptr);

#ifdef CHANNEL_Y
    vstore8(data.s02468ACE, 0, dst.ptr);
#elif defined(CHANNEL_U)
    vstore4(data.s159D, 0, dst.ptr);
#elif defined(CHANNEL_V)
    vstore4(data.s37BF, 0, dst.ptr);
#endif /* CHANNEL_Y or CHANNEL_U or CHANNEL_V */
}

/** This function extracts a given channel from an UYUV image.
 *
 * @note Channel to be extracted should be passed as a pre-processor argument, e.g. -DCHANNEL_U will extract the U channel.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: UYUV
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void channel_extract_UYVY422(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data = vload16(0, src.ptr);

#ifdef CHANNEL_Y
    vstore8(data.s13579BDF, 0, dst.ptr);
#elif defined(CHANNEL_U)
    vstore4(data.s048C, 0, dst.ptr);
#elif defined(CHANNEL_V)
    vstore4(data.s26AE, 0, dst.ptr);
#endif /* CHANNEL_Y or CHANNEL_U or CHANNEL_V */
}

/** This function extracts a given channel from an NV12 image.
 *
 * @note Channel to be extracted should be passed as a pre-processor argument, e.g. -DCHANNEL_U will extract the U channel.
 * @warning Only channels UV can be extracted using this kernel.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: NV12 (UV88)
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void channel_extract_NV12(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data = vload16(0, src.ptr);

#ifdef CHANNEL_U
    vstore8(data.s02468ACE, 0, dst.ptr);
#elif defined(CHANNEL_V)
    vstore8(data.s13579BDF, 0, dst.ptr);
#endif /* CHANNEL_U or CHANNEL_V */
}

/** This function extracts a given channel from an NV21 image.
 *
 * @note Channel to be extracted should be passed as a pre-processor argument, e.g. -DCHANNEL_U will extract the U channel.
 * @warning Only channels UV can be extracted using this kernel.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: NV21 (UV88)
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void channel_extract_NV21(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    uchar16 data = vload16(0, src.ptr);

#ifdef CHANNEL_U
    vstore8(data.s13579BDF, 0, dst.ptr);
#elif defined(CHANNEL_V)
    vstore8(data.s02468ACE, 0, dst.ptr);
#endif /* CHANNEL_U or CHANNEL_V */
}

/** This function extracts a given plane from an multi-planar image.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported Format: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void copy_plane(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(dst))
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    // Copy plane data
    vstore16(vload16(0, src.ptr), 0, dst.ptr);
}
