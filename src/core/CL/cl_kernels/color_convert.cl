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

/** Convert an RGB888 image to RGBX8888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void RGB888_to_RGBA8888_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    // handle 16 pixels every time
    uchar16 rgb_0 = vload16(0, in.ptr);
    uchar16 rgb_1 = vload16(0, in.ptr + 16);
    uchar16 rgb_2 = vload16(0, in.ptr + 32);

    uchar16 rgba_0 = (uchar16)(rgb_0.s012, 255, rgb_0.s345, 255, rgb_0.s678, 255, rgb_0.s9ab, 255);
    uchar16 rgba_1 = (uchar16)(rgb_0.scde, 255, rgb_0.sf, rgb_1.s01, 255, rgb_1.s234, 255, rgb_1.s567, 255);
    uchar16 rgba_2 = (uchar16)(rgb_1.s89a, 255, rgb_1.sbcd, 255, rgb_1.sef, rgb_2.s0, 255, rgb_2.s123, 255);
    uchar16 rgba_3 = (uchar16)(rgb_2.s456, 255, rgb_2.s789, 255, rgb_2.sabc, 255, rgb_2.sdef, 255);

    vstore16(rgba_0, 0, out.ptr);
    vstore16(rgba_1, 0, out.ptr + 16);
    vstore16(rgba_2, 0, out.ptr + 32);
    vstore16(rgba_3, 0, out.ptr + 48);
}

/** Convert an RGB888 image to RGBX8888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void RGBA8888_to_RGB888_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);
    // handle 16 pixels every time
    uchar16 rgba_0 = vload16(0, in.ptr);
    uchar16 rgba_1 = vload16(0, in.ptr + 16);
    uchar16 rgba_2 = vload16(0, in.ptr + 32);
    uchar16 rgba_3 = vload16(0, in.ptr + 48);

    uchar16 rgb_0 = (uchar16)(rgba_0.s01245689, rgba_0.sacde, rgba_1.s0124);
    uchar16 rgb_1 = (uchar16)(rgba_1.s5689acde, rgba_2.s01245689);
    uchar16 rgb_2 = (uchar16)(rgba_2.sacde, rgba_3.s01245689, rgba_3.sacde);

    vstore16(rgb_0, 0, out.ptr);
    vstore16(rgb_1, 0, out.ptr + 16);
    vstore16(rgb_2, 0, out.ptr + 32);
}

/** Convert a UYVY422 image to RGB888 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void UYVY422_to_RGB888_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    // handle 8 pixels every time
    uchar16 uyvy = vload16(0, in.ptr);

    uchar8 luma = (uchar8)(uyvy.s1, uyvy.s3, uyvy.s5, uyvy.s7, uyvy.s9, uyvy.sb, uyvy.sd, uyvy.sf);
    char8  cb   = (char8)(uyvy.s0, uyvy.s0, uyvy.s4, uyvy.s4, uyvy.s8, uyvy.s8, uyvy.sc, uyvy.sc) - (char8)(128);
    char8  cr   = (char8)(uyvy.s2, uyvy.s2, uyvy.s6, uyvy.s6, uyvy.sa, uyvy.sa, uyvy.se, uyvy.se) - (char8)(128);

    float8 f_r = convert_float8(luma) + (float8)(0.0000f) * convert_float8(cb) + (float8)(1.5748f) * convert_float8(cr);
    float8 f_g = convert_float8(luma) - (float8)(0.1873f) * convert_float8(cb) - (float8)(0.4681f) * convert_float8(cr);
    float8 f_b = convert_float8(luma) + (float8)(1.8556f) * convert_float8(cb) + (float8)(0.0000f) * convert_float8(cr);

    uchar8 r_0 = convert_uchar8_rtz(f_r);
    uchar8 g_0 = convert_uchar8_rtz(f_g);
    uchar8 b_0 = convert_uchar8_rtz(f_b);

    uchar16 rgb_0 = (uchar16)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2, b_0.s2,
                              r_0.s3, g_0.s3, b_0.s3, r_0.s4, g_0.s4, b_0.s4, r_0.s5);
    uchar8 rgb_1 = (uchar8)(g_0.s5, b_0.s5, r_0.s6, g_0.s6, b_0.s6, r_0.s7, g_0.s7, b_0.s7);

    vstore16(rgb_0, 0, out.ptr);
    vstore8(rgb_1, 0, out.ptr + 16);
}

/** Convert a UYVY422 image to RGBX8888 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void UYVY422_to_RGBA8888_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    // handle 8 pixels every time
    uchar16 uyvy = vload16(0, in.ptr);

    uchar8 luma = (uchar8)(uyvy.s1, uyvy.s3, uyvy.s5, uyvy.s7, uyvy.s9, uyvy.sb, uyvy.sd, uyvy.sf);
    char8  cb   = (char8)(uyvy.s0, uyvy.s0, uyvy.s4, uyvy.s4, uyvy.s8, uyvy.s8, uyvy.sc, uyvy.sc) - (char8)(128);
    char8  cr   = (char8)(uyvy.s2, uyvy.s2, uyvy.s6, uyvy.s6, uyvy.sa, uyvy.sa, uyvy.se, uyvy.se) - (char8)(128);

    float8 f_r = convert_float8(luma) + (float8)(0.0000f) * convert_float8(cb) + (float8)(1.5748f) * convert_float8(cr);
    float8 f_g = convert_float8(luma) - (float8)(0.1873f) * convert_float8(cb) - (float8)(0.4681f) * convert_float8(cr);
    float8 f_b = convert_float8(luma) + (float8)(1.8556f) * convert_float8(cb) + (float8)(0.0000f) * convert_float8(cr);

    uchar8 r_0 = convert_uchar8_rtz(f_r);
    uchar8 g_0 = convert_uchar8_rtz(f_g);
    uchar8 b_0 = convert_uchar8_rtz(f_b);

    uchar16 rgba_0 = (uchar16)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255,
                               r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    uchar16 rgba_1 = (uchar16)(r_0.s4, g_0.s4, b_0.s4, 255, r_0.s5, g_0.s5, b_0.s5, 255,
                               r_0.s6, g_0.s6, b_0.s6, 255, r_0.s7, g_0.s7, b_0.s7, 255);

    vstore16(rgba_0, 0, out.ptr);
    vstore16(rgba_1, 0, out.ptr + 16);
}

/** Convert a YUYV422 image to RGB888 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void YUYV422_to_RGB888_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    // handle 8 pixels every time
    uchar16 uyvy = vload16(0, in.ptr);

    uchar8 luma = (uchar8)(uyvy.s0, uyvy.s2, uyvy.s4, uyvy.s6, uyvy.s8, uyvy.sa, uyvy.sc, uyvy.se);
    char8  cb   = (char8)(uyvy.s1, uyvy.s1, uyvy.s5, uyvy.s5, uyvy.s9, uyvy.s9, uyvy.sd, uyvy.sd) - (char8)(128);
    char8  cr   = (char8)(uyvy.s3, uyvy.s3, uyvy.s7, uyvy.s7, uyvy.sb, uyvy.sb, uyvy.sf, uyvy.sf) - (char8)(128);

    float8 f_r = convert_float8(luma) + (float8)(0.0000f) * convert_float8(cb) + (float8)(1.5748f) * convert_float8(cr);
    float8 f_g = convert_float8(luma) - (float8)(0.1873f) * convert_float8(cb) - (float8)(0.4681f) * convert_float8(cr);
    float8 f_b = convert_float8(luma) + (float8)(1.8556f) * convert_float8(cb) + (float8)(0.0000f) * convert_float8(cr);

    uchar8 r_0 = convert_uchar8_rtz(f_r);
    uchar8 g_0 = convert_uchar8_rtz(f_g);
    uchar8 b_0 = convert_uchar8_rtz(f_b);

    uchar16 rgb_0 = (uchar16)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2, b_0.s2,
                              r_0.s3, g_0.s3, b_0.s3, r_0.s4, g_0.s4, b_0.s4, r_0.s5);
    uchar8 rgb_1 = (uchar8)(g_0.s5, b_0.s5, r_0.s6, g_0.s6, b_0.s6, r_0.s7, g_0.s7, b_0.s7);

    vstore16(rgb_0, 0, out.ptr);
    vstore8(rgb_1, 0, out.ptr + 16);
}

/** Convert a YUYV422 image to RGBX8888 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void YUYV422_to_RGBA8888_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(output))
{
    Image in  = CONVERT_TO_IMAGE_STRUCT(input);
    Image out = CONVERT_TO_IMAGE_STRUCT(output);

    // handle 8 pixels every time
    uchar16 uyvy = vload16(0, in.ptr);

    uchar8 luma = (uchar8)(uyvy.s0, uyvy.s2, uyvy.s4, uyvy.s6, uyvy.s8, uyvy.sa, uyvy.sc, uyvy.se);
    char8  cb   = (char8)(uyvy.s1, uyvy.s1, uyvy.s5, uyvy.s5, uyvy.s9, uyvy.s9, uyvy.sd, uyvy.sd) - (char8)(128);
    char8  cr   = (char8)(uyvy.s3, uyvy.s3, uyvy.s7, uyvy.s7, uyvy.sb, uyvy.sb, uyvy.sf, uyvy.sf) - (char8)(128);

    float8 f_r = convert_float8(luma) + (float8)(0.0000f) * convert_float8(cb) + (float8)(1.5748f) * convert_float8(cr);
    float8 f_g = convert_float8(luma) - (float8)(0.1873f) * convert_float8(cb) - (float8)(0.4681f) * convert_float8(cr);
    float8 f_b = convert_float8(luma) + (float8)(1.8556f) * convert_float8(cb) + (float8)(0.0000f) * convert_float8(cr);

    uchar8 r_0 = convert_uchar8_rtz(f_r);
    uchar8 g_0 = convert_uchar8_rtz(f_g);
    uchar8 b_0 = convert_uchar8_rtz(f_b);

    uchar16 rgba_0 = (uchar16)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255,
                               r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    uchar16 rgba_1 = (uchar16)(r_0.s4, g_0.s4, b_0.s4, 255, r_0.s5, g_0.s5, b_0.s5, 255,
                               r_0.s6, g_0.s6, b_0.s6, 255, r_0.s7, g_0.s7, b_0.s7, 255);

    vstore16(rgba_0, 0, out.ptr);
    vstore16(rgba_1, 0, out.ptr + 16);
}

/** Convert a RGB image to NV12 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  input_ptr                           Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] luma_ptr                            Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_stride_x                       Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_step_x                         luma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_stride_y                       Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_step_y                         luma_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_offset_first_element_in_bytes  The offset of the first element in the destination image luma channel
 * @param[out] uv_ptr                              Pointer to the destination uv channel. Supported Format: U8
 * @param[in]  uv_stride_x                         Stride of the destination uv channel in X dimension (in bytes)
 * @param[in]  uv_step_x                           uv_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_stride_y                         Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  uv_step_y                           uv_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_offset_first_element_in_bytes    The offset of the first element in the destination image uv channel
 *
 */
__kernel void RGB888_to_NV12_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(luma),
    IMAGE_DECLARATION(uv))
{
    Image in     = CONVERT_TO_IMAGE_STRUCT(input);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma);
    Image out_uv = CONVERT_TO_IMAGE_STRUCT(uv);

    // handle 4 pixels every time, two lines, each line for 2 pixels
    // Read 2 pixel of the first line
    uchar8 rgb_0 = vload8(0, in.ptr);
    uchar2 r_0   = (uchar2)(rgb_0.s0, rgb_0.s3);
    uchar2 g_0   = (uchar2)(rgb_0.s1, rgb_0.s4);
    uchar2 b_0   = (uchar2)(rgb_0.s2, rgb_0.s5);

    float2 f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_0) + (float2)(0.7152f) * convert_float2(g_0) + (float2)(0.0722f) * convert_float2(b_0);
    float2 f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_0) - (float2)(0.3854f) * convert_float2(g_0) + (float2)(0.5000f) * convert_float2(b_0);
    float2 f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_0) - (float2)(0.4542f) * convert_float2(g_0) - (float2)(0.0458f) * convert_float2(b_0);

    short2 i_y = convert_short2_rtz(f_y);
    short2 i_u = convert_short2_rtz(f_u) + (short2)(128);
    short2 i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_0 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_0, 0, out_y.ptr);

    uchar2 cb_0 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_0 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));

    // Read 2 pixel of the second line
    uchar8 rgb_1 = vload8(0, in.ptr + input_stride_y);
    uchar2 r_1   = (uchar2)(rgb_1.s0, rgb_1.s3);
    uchar2 g_1   = (uchar2)(rgb_1.s1, rgb_1.s4);
    uchar2 b_1   = (uchar2)(rgb_1.s2, rgb_1.s5);

    f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_1) + (float2)(0.7152f) * convert_float2(g_1) + (float2)(0.0722f) * convert_float2(b_1);
    f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_1) - (float2)(0.3854f) * convert_float2(g_1) + (float2)(0.5000f) * convert_float2(b_1);
    f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_1) - (float2)(0.4542f) * convert_float2(g_1) - (float2)(0.0458f) * convert_float2(b_1);

    i_y = convert_short2_rtz(f_y);
    i_u = convert_short2_rtz(f_u) + (short2)(128);
    i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_1 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_1, 0, out_y.ptr + luma_stride_y);

    uchar2 cb_1 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_1 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));
    uchar2 cbcr = (uchar2)(((cb_0.s0 + cb_0.s1 + cb_1.s0 + cb_1.s1) / 4),
                           ((cr_0.s0 + cr_0.s1 + cr_1.s0 + cr_1.s1) / 4));

    vstore2(cbcr, 0, out_uv.ptr);
}

/*
    R'= Y' + 0.0000*U + 1.5748*V
    G'= Y' - 0.1873*U - 0.4681*V
    B'= Y' + 1.8556*U + 0.0000*V
*/

/** Convert an NV12 image to RGB888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                           Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                      Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                        luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                      Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                        luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  uv_input_ptr                             Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                        Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                          uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                          uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes   The offset of the first element in the source image
 * @param[out] rgb_output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  rgb_output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  rgb_output_step_x                        rgb_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgb_output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  rgb_output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgb_output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void NV12_to_RGB888_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(rgb_output))
{
    Image in_luma = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv   = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_rgb = CONVERT_TO_IMAGE_STRUCT(rgb_output);

    // handle 8 pixels every time, two lines, each line for 4 pixels
    uchar4 luma_0 = vload4(0, in_luma.ptr);
    uchar4 luma_1 = vload4(0, in_luma.ptr + luma_input_stride_y);
    uchar4 cbcr   = vload4(0, in_uv.ptr);
    char4  cb     = (char4)(cbcr.s0, cbcr.s0, cbcr.s2, cbcr.s2) - (char4)(128);
    char4  cr     = (char4)(cbcr.s1, cbcr.s1, cbcr.s3, cbcr.s3) - (char4)(128);

    float4 temp0 = (float4)(0.0000f) + (float4)(0.0000f) * convert_float4(cb) + (float4)(1.5748f) * convert_float4(cr);
    float4 temp1 = (float4)(0.0000f) - (float4)(0.1873f) * convert_float4(cb) - (float4)(0.4681f) * convert_float4(cr);
    float4 temp2 = (float4)(0.0000f) + (float4)(1.8556f) * convert_float4(cb) + (float4)(0.0000f) * convert_float4(cr);

    float4 f_r = convert_float4(luma_0) + temp0;
    float4 f_g = convert_float4(luma_0) + temp1;
    float4 f_b = convert_float4(luma_0) + temp2;

    uchar4 r_0 = convert_uchar4_rtz(f_r);
    uchar4 g_0 = convert_uchar4_rtz(f_g);
    uchar4 b_0 = convert_uchar4_rtz(f_b);

    uchar8 rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2);
    uchar4 rgb_1 = (uchar4)(b_0.s2, r_0.s3, g_0.s3, b_0.s3);
    vstore8(rgb_0, 0, out_rgb.ptr);
    vstore4(rgb_1, 0, out_rgb.ptr + 8);

    f_r = convert_float4(luma_1) + temp0;
    f_g = convert_float4(luma_1) + temp1;
    f_b = convert_float4(luma_1) + temp2;

    r_0 = convert_uchar4_rtz(f_r);
    g_0 = convert_uchar4_rtz(f_g);
    b_0 = convert_uchar4_rtz(f_b);

    rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2);
    rgb_1 = (uchar4)(b_0.s2, r_0.s3, g_0.s3, b_0.s3);
    vstore8(rgb_0, 0, out_rgb.ptr + rgb_output_stride_y);
    vstore4(rgb_1, 0, out_rgb.ptr + rgb_output_stride_y + 8);
}

/** Convert a RGB image to YUV444 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  rgb_input_ptr                             Pointer to the source image. Supported Format: U8
 * @param[in]  rgb_input_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  rgb_input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgb_input_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  rgb_input_step_y                          rgb_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgb_input_offset_first_element_in_bytes   The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination image V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void RGB888_to_YUV444_bt709(
    IMAGE_DECLARATION(rgb_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    // handle 4 pixels every time
    Image in_rgb = CONVERT_TO_IMAGE_STRUCT(rgb_input);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u  = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v  = CONVERT_TO_IMAGE_STRUCT(v_output);

    // Read 4 pixel
    uchar16 rgb_0 = vload16(0, in_rgb.ptr);
    uchar4  r_0   = (uchar4)(rgb_0.s0, rgb_0.s3, rgb_0.s6, rgb_0.s9);
    uchar4  g_0   = (uchar4)(rgb_0.s1, rgb_0.s4, rgb_0.s7, rgb_0.sa);
    uchar4  b_0   = (uchar4)(rgb_0.s2, rgb_0.s5, rgb_0.s8, rgb_0.sb);

    float4 f_y = (float4)(0.0000f) + (float4)(0.2126f) * convert_float4(r_0) + (float4)(0.7152f) * convert_float4(g_0) + (float4)(0.0722f) * convert_float4(b_0);
    float4 f_u = (float4)(0.0000f) - (float4)(0.1146f) * convert_float4(r_0) - (float4)(0.3854f) * convert_float4(g_0) + (float4)(0.5000f) * convert_float4(b_0);
    float4 f_v = (float4)(0.0000f) + (float4)(0.5000f) * convert_float4(r_0) - (float4)(0.4542f) * convert_float4(g_0) - (float4)(0.0458f) * convert_float4(b_0);

    short4 i_y = convert_short4_rtz(f_y);
    short4 i_u = convert_short4_rtz(f_u) + (short4)(128);
    short4 i_v = convert_short4_rtz(f_v) + (short4)(128);

    uchar4 luma_0 = convert_uchar4(max((short4)(0), min(i_y, (short4)(255))));
    vstore4(luma_0, 0, out_y.ptr);

    uchar4 cb_0 = convert_uchar4(max((short4)(0), min(i_u, (short4)(255))));
    uchar4 cr_0 = convert_uchar4(max((short4)(0), min(i_v, (short4)(255))));
    vstore4(cb_0, 0, out_u.ptr);
    vstore4(cr_0, 0, out_v.ptr);
}

/** Convert a RGB image to IYUV using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 2), height ]
 * No offset.
 *
 * @param[in]  rgb_input_ptr                             Pointer to the source image. Supported Format: U8
 * @param[in]  rgb_input_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  rgb_input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgb_input_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  rgb_input_step_y                          rgb_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgb_input_offset_first_element_in_bytes   The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void RGB888_to_IYUV_bt709(
    IMAGE_DECLARATION(rgb_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    // handle 4 pixels every time, two lines, each line for 2 pixels
    Image in_rgb = CONVERT_TO_IMAGE_STRUCT(rgb_input);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u  = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v  = CONVERT_TO_IMAGE_STRUCT(v_output);

    // Read 2 pixel of the first line
    uchar8 rgb_0 = vload8(0, in_rgb.ptr);
    uchar2 r_0   = (uchar2)(rgb_0.s0, rgb_0.s3);
    uchar2 g_0   = (uchar2)(rgb_0.s1, rgb_0.s4);
    uchar2 b_0   = (uchar2)(rgb_0.s2, rgb_0.s5);

    float2 f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_0) + (float2)(0.7152f) * convert_float2(g_0) + (float2)(0.0722f) * convert_float2(b_0);
    float2 f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_0) - (float2)(0.3854f) * convert_float2(g_0) + (float2)(0.5000f) * convert_float2(b_0);
    float2 f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_0) - (float2)(0.4542f) * convert_float2(g_0) - (float2)(0.0458f) * convert_float2(b_0);

    short2 i_y = convert_short2_rtz(f_y);
    short2 i_u = convert_short2_rtz(f_u) + (short2)(128);
    short2 i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_0 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_0, 0, out_y.ptr);

    uchar2 cb_0 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_0 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));

    // Read 2 pixel of the second line
    uchar8 rgb_1 = vload8(0, in_rgb.ptr + rgb_input_stride_y);
    uchar2 r_1   = (uchar2)(rgb_1.s0, rgb_1.s3);
    uchar2 g_1   = (uchar2)(rgb_1.s1, rgb_1.s4);
    uchar2 b_1   = (uchar2)(rgb_1.s2, rgb_1.s5);

    f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_1) + (float2)(0.7152f) * convert_float2(g_1) + (float2)(0.0722f) * convert_float2(b_1);
    f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_1) - (float2)(0.3854f) * convert_float2(g_1) + (float2)(0.5000f) * convert_float2(b_1);
    f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_1) - (float2)(0.4542f) * convert_float2(g_1) - (float2)(0.0458f) * convert_float2(b_1);

    i_y = convert_short2_rtz(f_y);
    i_u = convert_short2_rtz(f_u) + (short2)(128);
    i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_1 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_1, 0, out_y.ptr + luma_output_stride_y);

    uchar2 cb_1 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_1 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));
    uchar2 cbcr = (uchar2)(((cb_0.s0 + cb_0.s1 + cb_1.s0 + cb_1.s1) / 4),
                           ((cr_0.s0 + cr_0.s1 + cr_1.s0 + cr_1.s1) / 4));
    *out_u.ptr = cbcr.x;
    *out_v.ptr = cbcr.y;
}

/** Convert a RGBA image to YUV444 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  rgba_input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  rgba_input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  rgba_input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgba_input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  rgba_input_step_y                         rgb_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgba_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination image V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void RGBA8888_to_YUV444_bt709(
    IMAGE_DECLARATION(rgba_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    // handle 4 pixels every time
    Image in_rgba = CONVERT_TO_IMAGE_STRUCT(rgba_input);
    Image out_y   = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u   = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v   = CONVERT_TO_IMAGE_STRUCT(v_output);

    // Read 4 pixel
    uchar16 rgb_0 = vload16(0, in_rgba.ptr);
    uchar4  r_0   = (uchar4)(rgb_0.s0, rgb_0.s4, rgb_0.s8, rgb_0.sc);
    uchar4  g_0   = (uchar4)(rgb_0.s1, rgb_0.s5, rgb_0.s9, rgb_0.sd);
    uchar4  b_0   = (uchar4)(rgb_0.s2, rgb_0.s6, rgb_0.sa, rgb_0.se);

    float4 f_y = (float4)(0.0000f) + (float4)(0.2126f) * convert_float4(r_0) + (float4)(0.7152f) * convert_float4(g_0) + (float4)(0.0722f) * convert_float4(b_0);
    float4 f_u = (float4)(0.0000f) - (float4)(0.1146f) * convert_float4(r_0) - (float4)(0.3854f) * convert_float4(g_0) + (float4)(0.5000f) * convert_float4(b_0);
    float4 f_v = (float4)(0.0000f) + (float4)(0.5000f) * convert_float4(r_0) - (float4)(0.4542f) * convert_float4(g_0) - (float4)(0.0458f) * convert_float4(b_0);

    short4 i_y = convert_short4(f_y);
    short4 i_u = convert_short4(f_u) + (short4)(128);
    short4 i_v = convert_short4(f_v) + (short4)(128);

    uchar4 luma_0 = convert_uchar4_sat(max((short4)(0), min(i_y, (short4)(255))));
    vstore4(luma_0, 0, out_y.ptr);

    uchar4 cb_0 = convert_uchar4_sat(max((short4)(0), min(i_u, (short4)(255))));
    uchar4 cr_0 = convert_uchar4_sat(max((short4)(0), min(i_v, (short4)(255))));
    vstore4(cb_0, 0, out_u.ptr);
    vstore4(cr_0, 0, out_v.ptr);
}

/** Convert a RGBA image to NV12 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 2), height ]
 * No offset.
 *
 * @param[in]  input_ptr                                 Pointer to the source image. Supported Format: U8
 * @param[in]  input_stride_x                            Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                              input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                            Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                              input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes       The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination image luma channel
 * @param[out] uv_output_ptr                             Pointer to the destination uv channel. Supported Format: U8
 * @param[in]  uv_output_stride_x                        Stride of the destination uv channel in X dimension (in bytes)
 * @param[in]  uv_output_step_x                          uv_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_output_stride_y                        Stride of the destination image uv channel in Y dimension (in bytes)
 * @param[in]  uv_output_step_y                          uv_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_output_offset_first_element_in_bytes   The offset of the first element in the destination image uv channel
 *
 */
__kernel void RGBA8888_to_NV12_bt709(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(uv_output))
{
    Image in     = CONVERT_TO_IMAGE_STRUCT(input);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_uv = CONVERT_TO_IMAGE_STRUCT(uv_output);

    // Read 2 pixel of the first line
    uchar8 rgb_0 = vload8(0, in.ptr);
    uchar2 r_0   = (uchar2)(rgb_0.s0, rgb_0.s4);
    uchar2 g_0   = (uchar2)(rgb_0.s1, rgb_0.s5);
    uchar2 b_0   = (uchar2)(rgb_0.s2, rgb_0.s6);

    float2 f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_0) + (float2)(0.7152f) * convert_float2(g_0) + (float2)(0.0722f) * convert_float2(b_0);
    float2 f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_0) - (float2)(0.3854f) * convert_float2(g_0) + (float2)(0.5000f) * convert_float2(b_0);
    float2 f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_0) - (float2)(0.4542f) * convert_float2(g_0) - (float2)(0.0458f) * convert_float2(b_0);

    short2 i_y = convert_short2_rtz(f_y);
    short2 i_u = convert_short2_rtz(f_u) + (short2)(128);
    short2 i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_0 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_0, 0, out_y.ptr);

    uchar2 cb_0 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_0 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));

    // Read 2 pixel of the second line
    uchar8 rgb_1 = vload8(0, in.ptr + input_stride_y);
    uchar2 r_1   = (uchar2)(rgb_1.s0, rgb_1.s4);
    uchar2 g_1   = (uchar2)(rgb_1.s1, rgb_1.s5);
    uchar2 b_1   = (uchar2)(rgb_1.s2, rgb_1.s6);

    f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_1) + (float2)(0.7152f) * convert_float2(g_1) + (float2)(0.0722f) * convert_float2(b_1);
    f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_1) - (float2)(0.3854f) * convert_float2(g_1) + (float2)(0.5000f) * convert_float2(b_1);
    f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_1) - (float2)(0.4542f) * convert_float2(g_1) - (float2)(0.0458f) * convert_float2(b_1);

    i_y = convert_short2_rtz(f_y);
    i_u = convert_short2_rtz(f_u) + (short2)(128);
    i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_1 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_1, 0, out_y.ptr + luma_output_stride_y);

    uchar2 cb_1 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_1 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));
    uchar2 cbcr = (uchar2)(((cb_0.s0 + cb_0.s1 + cb_1.s0 + cb_1.s1) / 4),
                           ((cr_0.s0 + cr_0.s1 + cr_1.s0 + cr_1.s1) / 4));
    vstore2(cbcr, 0, out_uv.ptr);
}

/** Convert a RGBA image to IYUV using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 2), height ]
 * No offset.
 *
 * @param[in]  rgba_input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  rgba_input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  rgba_input_step_x                         rgba_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgba_input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  rgba_input_step_y                         rgba_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgba_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void RGBA8888_to_IYUV_bt709(
    IMAGE_DECLARATION(rgba_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    // handle 4 pixels every time, two lines, each line for 2 pixels
    Image in_rgb = CONVERT_TO_IMAGE_STRUCT(rgba_input);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u  = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v  = CONVERT_TO_IMAGE_STRUCT(v_output);

    // Read 2 pixel of the first line
    uchar8 rgb_0 = vload8(0, in_rgb.ptr);
    uchar2 r_0   = (uchar2)(rgb_0.s0, rgb_0.s4);
    uchar2 g_0   = (uchar2)(rgb_0.s1, rgb_0.s5);
    uchar2 b_0   = (uchar2)(rgb_0.s2, rgb_0.s6);

    float2 f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_0) + (float2)(0.7152f) * convert_float2(g_0) + (float2)(0.0722f) * convert_float2(b_0);
    float2 f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_0) - (float2)(0.3854f) * convert_float2(g_0) + (float2)(0.5000f) * convert_float2(b_0);
    float2 f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_0) - (float2)(0.4542f) * convert_float2(g_0) - (float2)(0.0458f) * convert_float2(b_0);

    short2 i_y = convert_short2_rtz(f_y);
    short2 i_u = convert_short2_rtz(f_u) + (short2)(128);
    short2 i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_0 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_0, 0, out_y.ptr);

    uchar2 cb_0 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_0 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));

    // Read 2 pixel of the second line
    uchar8 rgb_1 = vload8(0, in_rgb.ptr + rgba_input_stride_y);
    uchar2 r_1   = (uchar2)(rgb_1.s0, rgb_1.s4);
    uchar2 g_1   = (uchar2)(rgb_1.s1, rgb_1.s5);
    uchar2 b_1   = (uchar2)(rgb_1.s2, rgb_1.s6);

    f_y = (float2)(0.0000f) + (float2)(0.2126f) * convert_float2(r_1) + (float2)(0.7152f) * convert_float2(g_1) + (float2)(0.0722f) * convert_float2(b_1);
    f_u = (float2)(0.0000f) - (float2)(0.1146f) * convert_float2(r_1) - (float2)(0.3854f) * convert_float2(g_1) + (float2)(0.5000f) * convert_float2(b_1);
    f_v = (float2)(0.0000f) + (float2)(0.5000f) * convert_float2(r_1) - (float2)(0.4542f) * convert_float2(g_1) - (float2)(0.0458f) * convert_float2(b_1);

    i_y = convert_short2_rtz(f_y);
    i_u = convert_short2_rtz(f_u) + (short2)(128);
    i_v = convert_short2_rtz(f_v) + (short2)(128);

    uchar2 luma_1 = convert_uchar2(max((short2)(0), min(i_y, (short2)(255))));
    vstore2(luma_1, 0, out_y.ptr + luma_output_stride_y);

    uchar2 cb_1 = convert_uchar2(max((short2)(0), min(i_u, (short2)(255))));
    uchar2 cr_1 = convert_uchar2(max((short2)(0), min(i_v, (short2)(255))));
    uchar2 cbcr = (uchar2)(((cb_0.s0 + cb_0.s1 + cb_1.s0 + cb_1.s1) / 4),
                           ((cr_0.s0 + cr_0.s1 + cr_1.s0 + cr_1.s1) / 4));
    *out_u.ptr = cbcr.x;
    *out_v.ptr = cbcr.y;
}

/** Convert an NV12 image to RGB8888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                           Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                      Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                        luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                      Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                        luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  uv_input_ptr                             Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                        Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                          uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                          uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes   The offset of the first element in the source image
 * @param[out] rgb_output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  rgb_output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  rgb_output_step_x                        rgb_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgb_output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  rgb_output_step_y                        rgb_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgb_output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void NV12_to_RGBA8888_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(rgb_output))
{
    Image in_luma = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv   = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_rgb = CONVERT_TO_IMAGE_STRUCT(rgb_output);

    uchar4 luma_0 = vload4(0, in_luma.ptr);
    uchar4 luma_1 = vload4(0, in_luma.ptr + luma_input_stride_y);
    uchar4 cbcr   = vload4(0, in_uv.ptr);
    char4  cb     = (char4)(cbcr.s0, cbcr.s0, cbcr.s2, cbcr.s2) - (char4)(128);
    char4  cr     = (char4)(cbcr.s1, cbcr.s1, cbcr.s3, cbcr.s3) - (char4)(128);

    float4 temp0 = (float4)(0.0000f) + (float4)(0.0000f) * convert_float4(cb) + (float4)(1.5748f) * convert_float4(cr);
    float4 temp1 = (float4)(0.0000f) - (float4)(0.1873f) * convert_float4(cb) - (float4)(0.4681f) * convert_float4(cr);
    float4 temp2 = (float4)(0.0000f) + (float4)(1.8556f) * convert_float4(cb) + (float4)(0.0000f) * convert_float4(cr);

    float4 f_r = convert_float4(luma_0) + temp0;
    float4 f_g = convert_float4(luma_0) + temp1;
    float4 f_b = convert_float4(luma_0) + temp2;

    uchar4 r_0 = convert_uchar4_rtz(f_r);
    uchar4 g_0 = convert_uchar4_rtz(f_g);
    uchar4 b_0 = convert_uchar4_rtz(f_b);

    uchar8 rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255);
    uchar8 rgb_1 = (uchar8)(r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    vstore8(rgb_0, 0, out_rgb.ptr);
    vstore8(rgb_1, 0, out_rgb.ptr + 8);

    f_r = convert_float4(luma_1) + temp0;
    f_g = convert_float4(luma_1) + temp1;
    f_b = convert_float4(luma_1) + temp2;

    r_0 = convert_uchar4_rtz(f_r);
    g_0 = convert_uchar4_rtz(f_g);
    b_0 = convert_uchar4_rtz(f_b);

    rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255);
    rgb_1 = (uchar8)(r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    vstore8(rgb_0, 0, out_rgb.ptr + rgb_output_stride_y);
    vstore8(rgb_1, 0, out_rgb.ptr + rgb_output_stride_y + 8);
}

/** Convert an NV12 image to IYUV
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  uv_input_ptr                              Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                         Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                           uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                           uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 */
__kernel void NV12_to_IYUV_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_y  = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_y = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v = CONVERT_TO_IMAGE_STRUCT(v_output);

    // handle 32 pixels every time, two lines, each line for 16 pixels
    uchar16 luma_0 = vload16(0, in_y.ptr);
    uchar16 luma_1 = vload16(0, in_y.ptr + luma_input_stride_y);
    uchar16 cbcr   = vload16(0, in_uv.ptr);
    uchar8  cb     = (uchar8)(cbcr.s0, cbcr.s2, cbcr.s4, cbcr.s6, cbcr.s8, cbcr.sa, cbcr.sc, cbcr.se);
    uchar8  cr     = (uchar8)(cbcr.s1, cbcr.s3, cbcr.s5, cbcr.s7, cbcr.s9, cbcr.sb, cbcr.sd, cbcr.sf);

    vstore16(luma_0, 0, out_y.ptr);
    vstore16(luma_1, 0, out_y.ptr + luma_output_stride_y);
    vstore8(cb, 0, out_u.ptr);
    vstore8(cr, 0, out_v.ptr);
}

/** Convert an NV12 image to YUV444
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  uv_input_ptr                              Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                         Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                           uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                           uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 */
__kernel void NV12_to_YUV444_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_y  = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_y = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v = CONVERT_TO_IMAGE_STRUCT(v_output);

    // handle 32 pixels every time, two lines, each line for 16 pixels
    uchar16 luma_0 = vload16(0, in_y.ptr);
    uchar16 luma_1 = vload16(0, in_y.ptr + luma_input_stride_y);
    uchar16 cbcr   = vload16(0, in_uv.ptr);
    uchar16 cb     = (uchar16)(cbcr.s0, cbcr.s0, cbcr.s2, cbcr.s2, cbcr.s4, cbcr.s4, cbcr.s6, cbcr.s6, cbcr.s8, cbcr.s8,
                               cbcr.sa, cbcr.sa, cbcr.sc, cbcr.sc, cbcr.se, cbcr.se);
    uchar16 cr = (uchar16)(cbcr.s1, cbcr.s1, cbcr.s3, cbcr.s3, cbcr.s5, cbcr.s5, cbcr.s7, cbcr.s7, cbcr.s9, cbcr.s9,
                           cbcr.sb, cbcr.sb, cbcr.sd, cbcr.sd, cbcr.sf, cbcr.sf);

    vstore16(luma_0, 0, out_y.ptr);
    vstore16(luma_1, 0, out_y.ptr + luma_output_stride_y);
    vstore16(cb, 0, out_u.ptr);
    vstore16(cb, 0, out_u.ptr + u_output_stride_y);
    vstore16(cr, 0, out_v.ptr);
    vstore16(cr, 0, out_v.ptr + v_output_stride_y);
}

/** Convert an NV21 image to RGB888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                           Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                      Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                        luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                      Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                        luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  uv_input_ptr                             Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                        Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                          uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                          uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes   The offset of the first element in the source image
 * @param[out] rgb_output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  rgb_output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  rgb_output_step_x                        rgb_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgb_output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  rgb_output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgb_output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void NV21_to_RGB888_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(rgb_output))
{
    Image in_y    = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv   = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_rgb = CONVERT_TO_IMAGE_STRUCT(rgb_output);

    // handle 8 pixels every time, two lines, each line for 4 pixels
    uchar4 luma_0 = vload4(0, in_y.ptr);
    uchar4 luma_1 = vload4(0, in_y.ptr + luma_input_stride_y);
    uchar4 cbcr   = vload4(0, in_uv.ptr);
    char4  cr     = (char4)(cbcr.s0, cbcr.s0, cbcr.s2, cbcr.s2) - (char4)(128);
    char4  cb     = (char4)(cbcr.s1, cbcr.s1, cbcr.s3, cbcr.s3) - (char4)(128);

    float4 temp0 = (float4)(0.0000f) + (float4)(0.0000f) * convert_float4(cb) + (float4)(1.5748f) * convert_float4(cr);
    float4 temp1 = (float4)(0.0000f) - (float4)(0.1873f) * convert_float4(cb) - (float4)(0.4681f) * convert_float4(cr);
    float4 temp2 = (float4)(0.0000f) + (float4)(1.8556f) * convert_float4(cb) + (float4)(0.0000f) * convert_float4(cr);

    float4 f_r = convert_float4(luma_0) + temp0;
    float4 f_g = convert_float4(luma_0) + temp1;
    float4 f_b = convert_float4(luma_0) + temp2;

    uchar4 r_0 = convert_uchar4_rtz(f_r);
    uchar4 g_0 = convert_uchar4_rtz(f_g);
    uchar4 b_0 = convert_uchar4_rtz(f_b);

    uchar8 rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2);
    uchar4 rgb_1 = (uchar4)(b_0.s2, r_0.s3, g_0.s3, b_0.s3);
    vstore8(rgb_0, 0, out_rgb.ptr);
    vstore4(rgb_1, 0, out_rgb.ptr + 8);

    f_r = convert_float4(luma_1) + temp0;
    f_g = convert_float4(luma_1) + temp1;
    f_b = convert_float4(luma_1) + temp2;

    r_0 = convert_uchar4_rtz(f_r);
    g_0 = convert_uchar4_rtz(f_g);
    b_0 = convert_uchar4_rtz(f_b);

    rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2);
    rgb_1 = (uchar4)(b_0.s2, r_0.s3, g_0.s3, b_0.s3);
    vstore8(rgb_0, 0, out_rgb.ptr + rgb_output_stride_y);
    vstore4(rgb_1, 0, out_rgb.ptr + rgb_output_stride_y + 8);
}

/** Convert an NV12 image to RGB8888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  uv_input_ptr                              Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                         Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                           uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                           uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] rgba_output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  rgba_output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  rgba_output_step_x                        rgba_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgba_output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  rgba_output_step_y                        rgba_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgba_output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void NV21_to_RGBA8888_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(rgba_output))
{
    Image in_luma = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv   = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_rgb = CONVERT_TO_IMAGE_STRUCT(rgba_output);

    // handle 8 pixels every time, two lines, each line for 4 pixels
    uchar4 luma_0 = vload4(0, in_luma.ptr);
    uchar4 luma_1 = vload4(0, in_luma.ptr + luma_input_stride_y);
    uchar4 cbcr   = vload4(0, in_uv.ptr);
    char4  cr     = (char4)(cbcr.s0, cbcr.s0, cbcr.s2, cbcr.s2) - (char4)(128);
    char4  cb     = (char4)(cbcr.s1, cbcr.s1, cbcr.s3, cbcr.s3) - (char4)(128);

    float4 temp0 = (float4)(0.0000f) + (float4)(0.0000f) * convert_float4(cb) + (float4)(1.5748f) * convert_float4(cr);
    float4 temp1 = (float4)(0.0000f) - (float4)(0.1873f) * convert_float4(cb) - (float4)(0.4681f) * convert_float4(cr);
    float4 temp2 = (float4)(0.0000f) + (float4)(1.8556f) * convert_float4(cb) + (float4)(0.0000f) * convert_float4(cr);

    float4 f_r = convert_float4(luma_0) + temp0;
    float4 f_g = convert_float4(luma_0) + temp1;
    float4 f_b = convert_float4(luma_0) + temp2;

    uchar4 r_0 = convert_uchar4_rtz(f_r);
    uchar4 g_0 = convert_uchar4_rtz(f_g);
    uchar4 b_0 = convert_uchar4_rtz(f_b);

    uchar8 rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255);
    uchar8 rgb_1 = (uchar8)(r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    vstore8(rgb_0, 0, out_rgb.ptr);
    vstore8(rgb_1, 0, out_rgb.ptr + 8);

    f_r = convert_float4(luma_1) + temp0;
    f_g = convert_float4(luma_1) + temp1;
    f_b = convert_float4(luma_1) + temp2;

    r_0 = convert_uchar4_rtz(f_r);
    g_0 = convert_uchar4_rtz(f_g);
    b_0 = convert_uchar4_rtz(f_b);

    rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255);
    rgb_1 = (uchar8)(r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    vstore8(rgb_0, 0, out_rgb.ptr + rgba_output_stride_y);
    vstore8(rgb_1, 0, out_rgb.ptr + rgba_output_stride_y + 8);
}

/** Convert an NV21 image to YUV444
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  uv_input_ptr                              Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                         Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                           uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                           uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 */
__kernel void NV21_to_YUV444_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_y  = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_y = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v = CONVERT_TO_IMAGE_STRUCT(v_output);

    // handle 32 pixels every time, two lines, each line for 16 pixels
    uchar16 luma_0 = vload16(0, in_y.ptr);
    uchar16 luma_1 = vload16(0, in_y.ptr + luma_input_stride_y);
    uchar16 cbcr   = vload16(0, in_uv.ptr);
    uchar16 cr     = (uchar16)(cbcr.s0, cbcr.s0, cbcr.s2, cbcr.s2, cbcr.s4, cbcr.s4, cbcr.s6, cbcr.s6, cbcr.s8, cbcr.s8,
                               cbcr.sa, cbcr.sa, cbcr.sc, cbcr.sc, cbcr.se, cbcr.se);
    uchar16 cb = (uchar16)(cbcr.s1, cbcr.s1, cbcr.s3, cbcr.s3, cbcr.s5, cbcr.s5, cbcr.s7, cbcr.s7, cbcr.s9, cbcr.s9,
                           cbcr.sb, cbcr.sb, cbcr.sd, cbcr.sd, cbcr.sf, cbcr.sf);

    vstore16(luma_0, 0, out_y.ptr);
    vstore16(luma_1, 0, out_y.ptr + luma_output_stride_y);
    vstore16(cb, 0, out_u.ptr);
    vstore16(cb, 0, out_u.ptr + u_output_stride_y);
    vstore16(cr, 0, out_v.ptr);
    vstore16(cr, 0, out_v.ptr + v_output_stride_y);
}

/** Convert an NV21 image to IYUV
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  uv_input_ptr                              Pointer to the source uv channel. Supported Format: U8
 * @param[in]  uv_input_stride_x                         Stride of the source image uv channel in X dimension (in bytes)
 * @param[in]  uv_input_step_x                           uv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_input_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  uv_input_step_y                           uv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_input_offset_first_element_in_bytes    The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 */
__kernel void NV21_to_IYUV_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(uv_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_y  = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_uv = CONVERT_TO_IMAGE_STRUCT(uv_input);
    Image out_y = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v = CONVERT_TO_IMAGE_STRUCT(v_output);

    uchar16 luma_0 = vload16(0, in_y.ptr);
    uchar16 luma_1 = vload16(0, in_y.ptr + luma_input_stride_y);
    uchar16 cbcr   = vload16(0, in_uv.ptr);
    uchar8  cr     = (uchar8)(cbcr.s0, cbcr.s2, cbcr.s4, cbcr.s6, cbcr.s8, cbcr.sa, cbcr.sc, cbcr.se);
    uchar8  cb     = (uchar8)(cbcr.s1, cbcr.s3, cbcr.s5, cbcr.s7, cbcr.s9, cbcr.sb, cbcr.sd, cbcr.sf);

    vstore16(luma_0, 0, out_y.ptr);
    vstore16(luma_1, 0, out_y.ptr + luma_output_stride_y);
    vstore8(cb, 0, out_u.ptr);
    vstore8(cr, 0, out_v.ptr);
}

/** Convert a UYVY image to IYUV using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  uyvy_input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  uyvy_input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  uyvy_input_step_x                         uyvy_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uyvy_input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  uyvy_input_step_y                         uyvy_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uyvy_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void UYVY422_to_IYUV_bt709(
    IMAGE_DECLARATION(uyvy_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_uyvy = CONVERT_TO_IMAGE_STRUCT(uyvy_input);
    Image out_y   = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u   = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v   = CONVERT_TO_IMAGE_STRUCT(v_output);

    // handle 16 pixels every time, each line 8 pixels
    uchar16 uyvy = vload16(0, in_uyvy.ptr);
    uchar8  luma = (uchar8)(uyvy.s1, uyvy.s3, uyvy.s5, uyvy.s7, uyvy.s9, uyvy.sb, uyvy.sd, uyvy.sf);
    ushort4 cb_0 = (ushort4)(uyvy.s0, uyvy.s4, uyvy.s8, uyvy.sc);
    ushort4 cr_0 = (ushort4)(uyvy.s2, uyvy.s6, uyvy.sa, uyvy.se);
    vstore8(luma, 0, out_y.ptr);

    uyvy         = vload16(0, in_uyvy.ptr + uyvy_input_stride_y);
    luma         = (uchar8)(uyvy.s1, uyvy.s3, uyvy.s5, uyvy.s7, uyvy.s9, uyvy.sb, uyvy.sd, uyvy.sf);
    ushort4 cb_1 = (ushort4)(uyvy.s0, uyvy.s4, uyvy.s8, uyvy.sc);
    ushort4 cr_1 = (ushort4)(uyvy.s2, uyvy.s6, uyvy.sa, uyvy.se);
    vstore8(luma, 0, out_y.ptr + luma_output_stride_y);

    uchar4 cb = convert_uchar4((cb_0 + cb_1) / (ushort4)(2));
    uchar4 cr = convert_uchar4((cr_0 + cr_1) / (ushort4)(2));
    vstore4(cb, 0, out_u.ptr);
    vstore4(cr, 0, out_v.ptr);
}

/** Convert a YUYV image to IYUV using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  yuyv_input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  yuyv_input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  yuyv_input_step_x                         yuyv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  yuyv_input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  yuyv_input_step_y                         yuyv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  yuyv_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void YUYV422_to_IYUV_bt709(
    IMAGE_DECLARATION(yuyv_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_yuyv = CONVERT_TO_IMAGE_STRUCT(yuyv_input);
    Image out_y   = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u   = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v   = CONVERT_TO_IMAGE_STRUCT(v_output);

    // handle 16 pixels every time, each line 8 pixels
    uchar16 yuyv = vload16(0, in_yuyv.ptr);
    uchar8  luma = (uchar8)(yuyv.s0, yuyv.s2, yuyv.s4, yuyv.s6, yuyv.s8, yuyv.sa, yuyv.sc, yuyv.se);
    ushort4 cb_0 = (ushort4)(yuyv.s1, yuyv.s5, yuyv.s9, yuyv.sd);
    ushort4 cr_0 = (ushort4)(yuyv.s3, yuyv.s7, yuyv.sb, yuyv.sf);
    vstore8(luma, 0, out_y.ptr);

    yuyv         = vload16(0, in_yuyv.ptr + yuyv_input_stride_y);
    luma         = (uchar8)(yuyv.s0, yuyv.s2, yuyv.s4, yuyv.s6, yuyv.s8, yuyv.sa, yuyv.sc, yuyv.se);
    ushort4 cb_1 = (ushort4)(yuyv.s1, yuyv.s5, yuyv.s9, yuyv.sd);
    ushort4 cr_1 = (ushort4)(yuyv.s3, yuyv.s7, yuyv.sb, yuyv.sf);
    vstore8(luma, 0, out_y.ptr + luma_output_stride_y);

    uchar4 cb = convert_uchar4((cb_0 + cb_1) / (ushort4)(2));
    uchar4 cr = convert_uchar4((cr_0 + cr_1) / (ushort4)(2));
    vstore4(cb, 0, out_u.ptr);
    vstore4(cr, 0, out_v.ptr);
}

/** Convert an IYUV image to RGB888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                           Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                      Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                        luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                      Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                        luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  u_input_ptr                              Pointer to the source U channel. Supported Format: U8
 * @param[in]  u_input_stride_x                         Stride of the source image U channel in X dimension (in bytes)
 * @param[in]  u_input_step_x                           u_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_input_stride_y                         Stride of the source image in Y dimension (in bytes)
 * @param[in]  u_input_step_y                           u_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_input_offset_first_element_in_bytes    The offset of the first element in the source U channel
 * @param[in]  v_input_ptr                              Pointer to the source V channel. Supported Format: U8
 * @param[in]  v_input_stride_x                         Stride of the source image V channel in X dimension (in bytes)
 * @param[in]  v_input_step_x                           v_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_input_stride_y                         Stride of the source image V channel in Y dimension (in bytes)
 * @param[in]  v_input_step_y                           v_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_input_offset_first_element_in_bytes    The offset of the first element in the source image V channel
 * @param[out] rgb_output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  rgb_output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  rgb_output_step_x                        rgb_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgb_output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  rgb_output_step_y                        rgb_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgb_output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void IYUV_to_RGB888_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(u_input),
    IMAGE_DECLARATION(v_input),
    IMAGE_DECLARATION(rgb_output))
{
    Image in_y    = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_u    = CONVERT_TO_IMAGE_STRUCT(u_input);
    Image in_v    = CONVERT_TO_IMAGE_STRUCT(v_input);
    Image out_rgb = CONVERT_TO_IMAGE_STRUCT(rgb_output);

    // handle 8 pixels every time, two lines, each line for 4 pixels
    uchar4 luma_0 = vload4(0, in_y.ptr);
    uchar4 luma_1 = vload4(0, in_y.ptr + luma_input_stride_y);
    uchar4 cbcr   = (uchar4)(vload2(0, in_u.ptr), vload2(0, in_v.ptr));
    char4  cb     = (char4)(cbcr.s0, cbcr.s0, cbcr.s1, cbcr.s1) - (char4)(128);
    char4  cr     = (char4)(cbcr.s2, cbcr.s2, cbcr.s3, cbcr.s3) - (char4)(128);

    float4 temp0 = (float4)(0.0000f) + (float4)(0.0000f) * convert_float4(cb) + (float4)(1.5748f) * convert_float4(cr);
    float4 temp1 = (float4)(0.0000f) - (float4)(0.1873f) * convert_float4(cb) - (float4)(0.4681f) * convert_float4(cr);
    float4 temp2 = (float4)(0.0000f) + (float4)(1.8556f) * convert_float4(cb) + (float4)(0.0000f) * convert_float4(cr);

    float4 f_r = convert_float4(luma_0) + temp0;
    float4 f_g = convert_float4(luma_0) + temp1;
    float4 f_b = convert_float4(luma_0) + temp2;

    uchar4 r_0 = convert_uchar4_rtz(f_r);
    uchar4 g_0 = convert_uchar4_rtz(f_g);
    uchar4 b_0 = convert_uchar4_rtz(f_b);

    uchar8 rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2);
    uchar4 rgb_1 = (uchar4)(b_0.s2, r_0.s3, g_0.s3, b_0.s3);
    vstore8(rgb_0, 0, out_rgb.ptr);
    vstore4(rgb_1, 0, out_rgb.ptr + 8);

    f_r = convert_float4(luma_1) + temp0;
    f_g = convert_float4(luma_1) + temp1;
    f_b = convert_float4(luma_1) + temp2;

    r_0 = convert_uchar4_rtz(f_r);
    g_0 = convert_uchar4_rtz(f_g);
    b_0 = convert_uchar4_rtz(f_b);

    rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, r_0.s1, g_0.s1, b_0.s1, r_0.s2, g_0.s2);
    rgb_1 = (uchar4)(b_0.s2, r_0.s3, g_0.s3, b_0.s3);
    vstore8(rgb_0, 0, out_rgb.ptr + rgb_output_stride_y);
    vstore4(rgb_1, 0, out_rgb.ptr + rgb_output_stride_y + 8);
}

/** Convert an IYUV image to RGB8888
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  u_input_ptr                               Pointer to the source U channel. Supported Format: U8
 * @param[in]  u_input_stride_x                          Stride of the source image U channel in X dimension (in bytes)
 * @param[in]  u_input_step_x                            u_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_input_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in]  u_input_step_y                            u_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_input_offset_first_element_in_bytes     The offset of the first element in the source U channel
 * @param[in]  v_input_ptr                               Pointer to the source V channel. Supported Format: U8
 * @param[in]  v_input_stride_x                          Stride of the source image V channel in X dimension (in bytes)
 * @param[in]  v_input_step_x                            v_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_input_stride_y                          Stride of the source image V channel in Y dimension (in bytes)
 * @param[in]  v_input_step_y                            v_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_input_offset_first_element_in_bytes     The offset of the first element in the source image V channel
 * @param[out] rgba_output_ptr                           Pointer to the destination image. Supported Format: U8
 * @param[in]  rgba_output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  rgba_output_step_x                        rgba_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  rgba_output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  rgba_output_step_y                        rgba_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  rgba_output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void IYUV_to_RGBA8888_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(u_input),
    IMAGE_DECLARATION(v_input),
    IMAGE_DECLARATION(rgba_output))
{
    Image in_y    = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_u    = CONVERT_TO_IMAGE_STRUCT(u_input);
    Image in_v    = CONVERT_TO_IMAGE_STRUCT(v_input);
    Image out_rgb = CONVERT_TO_IMAGE_STRUCT(rgba_output);

    // handle 8 pixels every time, two lines, each line for 4 pixels
    uchar4 luma_0 = vload4(0, in_y.ptr);
    uchar4 luma_1 = vload4(0, in_y.ptr + luma_input_stride_y);
    uchar4 cbcr   = (uchar4)(vload2(0, in_u.ptr), vload2(0, in_v.ptr));
    char4  cb     = (char4)(cbcr.s0, cbcr.s0, cbcr.s1, cbcr.s1) - (char4)(128);
    char4  cr     = (char4)(cbcr.s2, cbcr.s2, cbcr.s3, cbcr.s3) - (char4)(128);

    float4 temp0 = (float4)(0.0000f) + (float4)(0.0000f) * convert_float4(cb) + (float4)(1.5748f) * convert_float4(cr);
    float4 temp1 = (float4)(0.0000f) - (float4)(0.1873f) * convert_float4(cb) - (float4)(0.4681f) * convert_float4(cr);
    float4 temp2 = (float4)(0.0000f) + (float4)(1.8556f) * convert_float4(cb) + (float4)(0.0000f) * convert_float4(cr);

    float4 f_r = convert_float4(luma_0) + temp0;
    float4 f_g = convert_float4(luma_0) + temp1;
    float4 f_b = convert_float4(luma_0) + temp2;

    uchar4 r_0 = convert_uchar4_rtz(f_r);
    uchar4 g_0 = convert_uchar4_rtz(f_g);
    uchar4 b_0 = convert_uchar4_rtz(f_b);

    uchar8 rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255);
    uchar8 rgb_1 = (uchar8)(r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    vstore8(rgb_0, 0, out_rgb.ptr);
    vstore8(rgb_1, 0, out_rgb.ptr + 8);

    f_r = convert_float4(luma_1) + temp0;
    f_g = convert_float4(luma_1) + temp1;
    f_b = convert_float4(luma_1) + temp2;

    r_0 = convert_uchar4_rtz(f_r);
    g_0 = convert_uchar4_rtz(f_g);
    b_0 = convert_uchar4_rtz(f_b);

    rgb_0 = (uchar8)(r_0.s0, g_0.s0, b_0.s0, 255, r_0.s1, g_0.s1, b_0.s1, 255);
    rgb_1 = (uchar8)(r_0.s2, g_0.s2, b_0.s2, 255, r_0.s3, g_0.s3, b_0.s3, 255);
    vstore8(rgb_0, 0, out_rgb.ptr + rgba_output_stride_y);
    vstore8(rgb_1, 0, out_rgb.ptr + rgba_output_stride_y + 8);
}

/** Convert an IYUV image to YUV444
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  u_input_ptr                               Pointer to the source U channel. Supported Format: U8
 * @param[in]  u_input_stride_x                          Stride of the source image U channel in X dimension (in bytes)
 * @param[in]  u_input_step_x                            u_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_input_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in]  u_input_step_y                            u_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_input_offset_first_element_in_bytes     The offset of the first element in the source U channel
 * @param[in]  v_input_ptr                               Pointer to the source V channel. Supported Format: U8
 * @param[in]  v_input_stride_x                          Stride of the source image V channel in X dimension (in bytes)
 * @param[in]  v_input_step_x                            v_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_input_stride_y                          Stride of the source image V channel in Y dimension (in bytes)
 * @param[in]  v_input_step_y                            v_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_input_offset_first_element_in_bytes     The offset of the first element in the source image V channel
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] u_output_ptr                              Pointer to the destination U channel. Supported Format: U8
 * @param[in]  u_output_stride_x                         Stride of the destination U channel in X dimension (in bytes)
 * @param[in]  u_output_step_x                           u_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_output_stride_y                         Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  u_output_step_y                           u_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_output_offset_first_element_in_bytes    The offset of the first element in the destination U channel
 * @param[out] v_output_ptr                              Pointer to the destination V channel. Supported Format: U8
 * @param[in]  v_output_stride_x                         Stride of the destination V channel in X dimension (in bytes)
 * @param[in]  v_output_step_x                           v_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_output_stride_y                         Stride of the destination V channel in Y dimension (in bytes)
 * @param[in]  v_output_step_y                           v_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_output_offset_first_element_in_bytes    The offset of the first element in the destination V channel
 *
 */
__kernel void IYUV_to_YUV444_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(u_input),
    IMAGE_DECLARATION(v_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(u_output),
    IMAGE_DECLARATION(v_output))
{
    Image in_y  = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_u  = CONVERT_TO_IMAGE_STRUCT(u_input);
    Image in_v  = CONVERT_TO_IMAGE_STRUCT(v_input);
    Image out_y = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_u = CONVERT_TO_IMAGE_STRUCT(u_output);
    Image out_v = CONVERT_TO_IMAGE_STRUCT(v_output);

    // handle 32 pixels every time, two lines, each line for 16 pixels
    uchar16 luma_0 = vload16(0, in_y.ptr);
    uchar16 luma_1 = vload16(0, in_y.ptr + luma_input_stride_y);
    uchar8  cb_src = vload8(0, in_u.ptr);
    uchar8  cr_src = vload8(0, in_v.ptr);
    uchar16 cb     = (uchar16)(cb_src.s0, cb_src.s0, cb_src.s1, cb_src.s1, cb_src.s2, cb_src.s2, cb_src.s3, cb_src.s3,
                               cb_src.s4, cb_src.s4, cb_src.s5, cb_src.s5, cb_src.s6, cb_src.s6, cb_src.s7, cb_src.s7);
    uchar16 cr = (uchar16)(cr_src.s0, cr_src.s0, cr_src.s1, cr_src.s1, cr_src.s2, cr_src.s2, cr_src.s3, cr_src.s3,
                           cr_src.s4, cr_src.s4, cr_src.s5, cr_src.s5, cr_src.s6, cr_src.s6, cr_src.s7, cr_src.s7);

    vstore16(luma_0, 0, out_y.ptr);
    vstore16(luma_1, 0, out_y.ptr + luma_output_stride_y);
    vstore16(cb, 0, out_u.ptr);
    vstore16(cb, 0, out_u.ptr + u_output_stride_y);
    vstore16(cr, 0, out_v.ptr);
    vstore16(cr, 0, out_v.ptr + v_output_stride_y);
}

/** Convert an IYUV image to NV12
 *
 * Global Workgroup Size [ DIV_CEIL(width, 16), height ]
 * No offset.
 *
 * @param[in]  luma_input_ptr                            Pointer to the source luma channel. Supported Format: U8
 * @param[in]  luma_input_stride_x                       Stride of the luma image in X dimension (in bytes)
 * @param[in]  luma_input_step_x                         luma_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_input_stride_y                       Stride of the source luma channel in Y dimension (in bytes)
 * @param[in]  luma_input_step_y                         luma_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[in]  u_input_ptr                               Pointer to the source U channel. Supported Format: U8
 * @param[in]  u_input_stride_x                          Stride of the source image U channel in X dimension (in bytes)
 * @param[in]  u_input_step_x                            u_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  u_input_stride_y                          Stride of the source image in Y dimension (in bytes)
 * @param[in]  u_input_step_y                            u_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  u_input_offset_first_element_in_bytes     The offset of the first element in the source U channel
 * @param[in]  v_input_ptr                               Pointer to the source V channel. Supported Format: U8
 * @param[in]  v_input_stride_x                          Stride of the source image V channel in X dimension (in bytes)
 * @param[in]  v_input_step_x                            v_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  v_input_stride_y                          Stride of the source image V channel in Y dimension (in bytes)
 * @param[in]  v_input_step_y                            v_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  v_input_offset_first_element_in_bytes     The offset of the first element in the source image V channel
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] uv_output_ptr                             Pointer to the destination UV channel. Supported Format: U8
 * @param[in]  uv_output_stride_x                        Stride of the destination UV channel in X dimension (in bytes)
 * @param[in]  uv_output_step_x                          uv_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_output_stride_y                        Stride of the destination image U channel in Y dimension (in bytes)
 * @param[in]  uv_output_step_y                          uv_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_output_offset_first_element_in_bytes   The offset of the first element in the destination UV channel
 *
 */
__kernel void IYUV_to_NV12_bt709(
    IMAGE_DECLARATION(luma_input),
    IMAGE_DECLARATION(u_input),
    IMAGE_DECLARATION(v_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(uv_output))
{
    Image in_y   = CONVERT_TO_IMAGE_STRUCT(luma_input);
    Image in_u   = CONVERT_TO_IMAGE_STRUCT(u_input);
    Image in_v   = CONVERT_TO_IMAGE_STRUCT(v_input);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_uv = CONVERT_TO_IMAGE_STRUCT(uv_output);

    // handle 32 pixels every time, two lines, each line for 16 pixels
    uchar16 luma_0 = vload16(0, in_y.ptr);
    uchar16 luma_1 = vload16(0, in_y.ptr + luma_input_stride_y);
    uchar8  cb     = vload8(0, in_u.ptr);
    uchar8  cr     = vload8(0, in_v.ptr);
    uchar16 cbcr   = (uchar16)(cb.s0, cr.s0, cb.s1, cr.s1, cb.s2, cr.s2, cb.s3, cr.s3, cb.s4, cr.s4, cb.s5, cr.s5, cb.s6,
                               cr.s6, cb.s7, cr.s7);

    vstore16(luma_0, 0, out_y.ptr);
    vstore16(luma_1, 0, out_y.ptr + luma_output_stride_y);
    vstore16(cbcr, 0, out_uv.ptr);
}

/** Convert a YUYV image to NV12 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 8), height ]
 * No offset.
 *
 * @param[in]  yuyv_input_ptr                            Pointer to the source image. Supported Format: U8
 * @param[in]  yuyv_input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  yuyv_input_step_x                         yuyv_input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  yuyv_input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  yuyv_input_step_y                         yuyv_input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  yuyv_input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] luma_output_ptr                           Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_output_stride_x                      Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_output_step_x                        luma_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_output_stride_y                      Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_output_step_y                        luma_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_output_offset_first_element_in_bytes The offset of the first element in the destination luma channel
 * @param[out] uv_output_ptr                             Pointer to the destination UV channel. Supported Format: U8
 * @param[in]  uv_output_stride_x                        Stride of the destination UV channel in X dimension (in bytes)
 * @param[in]  uv_output_step_x                          uv_output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_output_stride_y                        Stride of the destination image UV channel in Y dimension (in bytes)
 * @param[in]  uv_output_step_y                          uv_output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_output_offset_first_element_in_bytes   The offset of the first element in the destination UV channel
 *
 */
__kernel void YUYV422_to_NV12_bt709(
    IMAGE_DECLARATION(yuyv_input),
    IMAGE_DECLARATION(luma_output),
    IMAGE_DECLARATION(uv_output))
{
    Image in_yuyv = CONVERT_TO_IMAGE_STRUCT(yuyv_input);
    Image out_y   = CONVERT_TO_IMAGE_STRUCT(luma_output);
    Image out_uv  = CONVERT_TO_IMAGE_STRUCT(uv_output);

    // handle 16 pixels every time, each line 8 pixels
    uchar16 yuyv   = vload16(0, in_yuyv.ptr);
    ushort8 cbcr_0 = (ushort8)(yuyv.s1, yuyv.s3, yuyv.s5, yuyv.s7, yuyv.s9, yuyv.sb, yuyv.sd, yuyv.sf);
    uchar8  luma   = (uchar8)(yuyv.s0, yuyv.s2, yuyv.s4, yuyv.s6, yuyv.s8, yuyv.sa, yuyv.sc, yuyv.se);
    vstore8(luma, 0, out_y.ptr);

    yuyv           = vload16(0, in_yuyv.ptr + yuyv_input_stride_y);
    ushort8 cbcr_1 = (ushort8)(yuyv.s1, yuyv.s3, yuyv.s5, yuyv.s7, yuyv.s9, yuyv.sb, yuyv.sd, yuyv.sf);
    luma           = (uchar8)(yuyv.s0, yuyv.s2, yuyv.s4, yuyv.s6, yuyv.s8, yuyv.sa, yuyv.sc, yuyv.se);
    vstore8(luma, 0, out_y.ptr + luma_output_stride_y);

    uchar8 cbcr = convert_uchar8((cbcr_0 + cbcr_1) / (ushort8)(2));
    vstore8(cbcr, 0, out_uv.ptr);
}

/** Convert a UYVY image to NV12 using BT709 color space
 *
 * Global Workgroup Size [ DIV_CEIL(width, 4), height ]
 * No offset.
 *
 * @param[in]  input_uyvy_ptr                           Pointer to the source image. Supported Format: U8
 * @param[in]  input_uyvy_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  input_uyvy_step_x                        input_uyvy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_uyvy_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_uyvy_step_y                        input_uyvy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_uyvy_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] luma_ptr                                 Pointer to the destination luma channel. Supported Format: U8
 * @param[in]  luma_stride_x                            Stride of the destination luma channel in X dimension (in bytes)
 * @param[in]  luma_step_x                              luma_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  luma_stride_y                            Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  luma_step_y                              luma_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  luma_offset_first_element_in_bytes       The offset of the first element in the destination image luma channel
 * @param[out] uv_ptr                                   Pointer to the destination uv channel. Supported Format: U8
 * @param[in]  uv_stride_x                              Stride of the destination uv channel in X dimension (in bytes)
 * @param[in]  uv_step_x                                uv_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  uv_stride_y                              Stride of the destination image luma channel in Y dimension (in bytes)
 * @param[in]  uv_step_y                                uv_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  uv_offset_first_element_in_bytes         The offset of the first element in the destination image uv channel
 *
 */
__kernel void UYVY422_to_NV12_bt709(
    IMAGE_DECLARATION(input_uyvy),
    IMAGE_DECLARATION(luma),
    IMAGE_DECLARATION(uv))
{
    Image in     = CONVERT_TO_IMAGE_STRUCT(input_uyvy);
    Image out_y  = CONVERT_TO_IMAGE_STRUCT(luma);
    Image out_uv = CONVERT_TO_IMAGE_STRUCT(uv);

    // handle 16 pixels every time, each line 8 pixels
    const uchar16 uyvy_t = vload16(0, in.ptr);
    vstore8(uyvy_t.s13579bdf, 0, out_y.ptr);

    const uchar16 uyvy_b = vload16(0, in.ptr + input_uyvy_stride_y);
    vstore8(uyvy_b.s13579bdf, 0, out_y.ptr + luma_stride_y);

    const ushort8 cbcr_t = (ushort8)(uyvy_t.s0, uyvy_t.s2, uyvy_t.s4, uyvy_t.s6, uyvy_t.s8, uyvy_t.sa, uyvy_t.sc, uyvy_t.se);
    const ushort8 cbcr_b = (ushort8)(uyvy_b.s0, uyvy_b.s2, uyvy_b.s4, uyvy_b.s6, uyvy_b.s8, uyvy_b.sa, uyvy_b.sc, uyvy_b.se);
    const uchar8  cbcr   = convert_uchar8((cbcr_t + cbcr_b) / (ushort8)(2));
    vstore8(cbcr, 0, out_uv.ptr);
}
