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

/** Function running harris score on 3x3 block size
 *
 * @attention: The input data type should be passed using a compile option -DDATA_TYPE. Supported types: short and int.
 *             e.g. -DDATA_TYPE=short.
 *
 * @param[in]  src_gx_ptr                           Pointer to the first source image. Supported data types: S16, S32
 * @param[in]  src_gx_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_gx_step_x                        src_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_gx_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_gx_step_y                        src_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_gx_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  src_gy_ptr                           Pointer to the second source image. Supported data types: S16, S32
 * @param[in]  src_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  src_gy_step_x                        src_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  src_gy_step_y                        src_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] vc_ptr                               Pointer to the destination image. Supported data types: F32
 * @param[in]  vc_stride_x                          Stride of the destination image in X dimension (in bytes)
 * @param[in]  vc_step_x                            vc_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  vc_stride_y                          Stride of the destination image in Y dimension (in bytes)
 * @param[in]  vc_step_y                            vc_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  vc_offset_first_element_in_bytes     The offset of the first element in the destination image
 * @param[in]  sensitivity                          Sensitivity threshold k from the Harris-Stephens equation
 * @param[in]  strength_thresh                      Minimum threshold with which to eliminate Harris Corner scores
 * @param[in]  pow4_normalization_factor            Normalization factor to apply harris score
 */
__kernel void harris_score_3x3(
    IMAGE_DECLARATION(src_gx),
    IMAGE_DECLARATION(src_gy),
    IMAGE_DECLARATION(vc),
    float sensitivity,
    float strength_thresh,
    float pow4_normalization_factor)
{
    Image src_gx = CONVERT_TO_IMAGE_STRUCT(src_gx);
    Image src_gy = CONVERT_TO_IMAGE_STRUCT(src_gy);
    Image vc     = CONVERT_TO_IMAGE_STRUCT(vc);

    /* Gx^2, Gy^2 and Gx*Gy */
    float4 gx2  = (float4)0.0f;
    float4 gy2  = (float4)0.0f;
    float4 gxgy = (float4)0.0f;

    /* Row0 */
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp_gx = vload8(0, (__global DATA_TYPE *)offset(&src_gx, -1, -1));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp_gy = vload8(0, (__global DATA_TYPE *)offset(&src_gy, -1, -1));

    float4 l_gx = convert_float4(temp_gx.s0123);
    float4 m_gx = convert_float4(temp_gx.s1234);
    float4 r_gx = convert_float4(temp_gx.s2345);

    float4 l_gy = convert_float4(temp_gy.s0123);
    float4 m_gy = convert_float4(temp_gy.s1234);
    float4 r_gy = convert_float4(temp_gy.s2345);

    gx2 += (l_gx * l_gx) + (m_gx * m_gx) + (r_gx * r_gx);
    gy2 += (l_gy * l_gy) + (m_gy * m_gy) + (r_gy * r_gy);
    gxgy += (l_gx * l_gy) + (m_gx * m_gy) + (r_gx * r_gy);

    /* Row1 */
    temp_gx = vload8(0, (__global DATA_TYPE *)offset(&src_gx, -1, 0));
    temp_gy = vload8(0, (__global DATA_TYPE *)offset(&src_gy, -1, 0));

    l_gx = convert_float4(temp_gx.s0123);
    m_gx = convert_float4(temp_gx.s1234);
    r_gx = convert_float4(temp_gx.s2345);

    l_gy = convert_float4(temp_gy.s0123);
    m_gy = convert_float4(temp_gy.s1234);
    r_gy = convert_float4(temp_gy.s2345);

    gx2 += (l_gx * l_gx) + (m_gx * m_gx) + (r_gx * r_gx);
    gy2 += (l_gy * l_gy) + (m_gy * m_gy) + (r_gy * r_gy);
    gxgy += (l_gx * l_gy) + (m_gx * m_gy) + (r_gx * r_gy);

    /* Row2 */
    temp_gx = vload8(0, (__global DATA_TYPE *)offset(&src_gx, -1, 1));
    temp_gy = vload8(0, (__global DATA_TYPE *)offset(&src_gy, -1, 1));

    l_gx = convert_float4(temp_gx.s0123);
    m_gx = convert_float4(temp_gx.s1234);
    r_gx = convert_float4(temp_gx.s2345);

    l_gy = convert_float4(temp_gy.s0123);
    m_gy = convert_float4(temp_gy.s1234);
    r_gy = convert_float4(temp_gy.s2345);

    gx2 += (l_gx * l_gx) + (m_gx * m_gx) + (r_gx * r_gx);
    gy2 += (l_gy * l_gy) + (m_gy * m_gy) + (r_gy * r_gy);
    gxgy += (l_gx * l_gy) + (m_gx * m_gy) + (r_gx * r_gy);

    /* Compute trace and determinant */
    float4 trace = gx2 + gy2;
    float4 det   = gx2 * gy2 - (gxgy * gxgy);

    /* Compute harris score */
    float4 mc = (det - (sensitivity * (trace * trace))) * pow4_normalization_factor;

    mc = select(0.0f, mc, mc > (float4)strength_thresh);

    vstore4(mc, 0, (__global float *)vc.ptr);
}

/** Function for calculating harris score 1x5.
 *
 * @param[in] src_gx Pointer to gx gradient image.
 * @param[in] src_gy Pointer to gy gradient image.
 * @param[in] row    Relative row.
 */
inline float16 harris_score_1x5(Image *src_gx, Image *src_gy, int row)
{
    float4 gx2  = 0.0f;
    float4 gy2  = 0.0f;
    float4 gxgy = 0.0f;

    /* Row */
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp_gx = vload8(0, (__global DATA_TYPE *)offset(src_gx, -2, row));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp_gy = vload8(0, (__global DATA_TYPE *)offset(src_gy, -2, row));

    float4 gx = convert_float4(temp_gx.s0123);
    float4 gy = convert_float4(temp_gy.s0123);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx.s1234);
    gy = convert_float4(temp_gy.s1234);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx.s2345);
    gy = convert_float4(temp_gy.s2345);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx.s3456);
    gy = convert_float4(temp_gy.s3456);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx.s4567);
    gy = convert_float4(temp_gy.s4567);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    return (float16)(gx2, gy2, gxgy, (float4)0);
}

/** Function running harris score on 5x5 block size
 *
 * @attention: The input data type should be passed using a compile option -DDATA_TYPE. Supported types: short and int.
 *             e.g. -DDATA_TYPE=short.
 *
 * @param[in]  src_gx_ptr                           Pointer to the first source image. Supported data types: S16, S32
 * @param[in]  src_gx_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_gx_step_x                        src_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_gx_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_gx_step_y                        src_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_gx_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  src_gy_ptr                           Pointer to the second source image. Supported data types: S16, S32
 * @param[in]  src_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  src_gy_step_x                        src_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  src_gy_step_y                        src_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] vc_ptr                               Pointer to the destination image. Supported data types: F32
 * @param[in]  vc_stride_x                          Stride of the destination image in X dimension (in bytes)
 * @param[in]  vc_step_x                            vc_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  vc_stride_y                          Stride of the destination image in Y dimension (in bytes)
 * @param[in]  vc_step_y                            vc_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  vc_offset_first_element_in_bytes     The offset of the first element in the destination image
 * @param[in]  sensitivity                          Sensitivity threshold k from the Harris-Stephens equation
 * @param[in]  strength_thresh                      Minimum threshold with which to eliminate Harris Corner scores
 * @param[in]  pow4_normalization_factor            Normalization factor to apply harris score
 */
__kernel void harris_score_5x5(
    IMAGE_DECLARATION(src_gx),
    IMAGE_DECLARATION(src_gy),
    IMAGE_DECLARATION(vc),
    float sensitivity,
    float strength_thresh,
    float pow4_normalization_factor)
{
    Image src_gx = CONVERT_TO_IMAGE_STRUCT(src_gx);
    Image src_gy = CONVERT_TO_IMAGE_STRUCT(src_gy);
    Image vc     = CONVERT_TO_IMAGE_STRUCT(vc);

    /* Gx^2, Gy^2 and Gx*Gy */
    float16 res = (float16)0.0f;

    /* Compute row */
    for(int i = -2; i < 3; i++)
    {
        res += harris_score_1x5(&src_gx, &src_gy, i);
    }

    float4 gx2  = res.s0123;
    float4 gy2  = res.s4567;
    float4 gxgy = res.s89AB;

    /* Compute trace and determinant */
    float4 trace = gx2 + gy2;
    float4 det   = gx2 * gy2 - (gxgy * gxgy);

    /* Compute harris score */
    float4 mc = (det - (sensitivity * (trace * trace))) * pow4_normalization_factor;

    mc = select(0.0f, mc, mc > (float4)strength_thresh);

    vstore4(mc, 0, (__global float *)vc.ptr);
}

/** Function for calculating harris score 1x7.
 *
 * @param[in] src_gx Pointer to gx gradient image.
 * @param[in] src_gy Pointer to gy gradient image.
 * @param[in] row    Relative row.
 */
inline float16 harris_score_1x7(Image *src_gx, Image *src_gy, int row)
{
    float4 gx2  = 0.0f;
    float4 gy2  = 0.0f;
    float4 gxgy = 0.0f;

    /* Row */
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp_gx0 = vload8(0, (__global DATA_TYPE *)offset(src_gx, -3, row));
    VEC_DATA_TYPE(DATA_TYPE, 8)
    temp_gy0 = vload8(0, (__global DATA_TYPE *)offset(src_gy, -3, row));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    temp_gx1 = vload2(0, (__global DATA_TYPE *)offset(src_gx, 5, row));
    VEC_DATA_TYPE(DATA_TYPE, 2)
    temp_gy1 = vload2(0, (__global DATA_TYPE *)offset(src_gy, 5, row));

    float4 gx = convert_float4(temp_gx0.s0123);
    float4 gy = convert_float4(temp_gy0.s0123);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx0.s1234);
    gy = convert_float4(temp_gy0.s1234);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx0.s2345);
    gy = convert_float4(temp_gy0.s2345);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx0.s3456);
    gy = convert_float4(temp_gy0.s3456);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4(temp_gx0.s4567);
    gy = convert_float4(temp_gy0.s4567);
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4((VEC_DATA_TYPE(DATA_TYPE, 4))(temp_gx0.s567, temp_gx1.s0));
    gy = convert_float4((VEC_DATA_TYPE(DATA_TYPE, 4))(temp_gy0.s567, temp_gy1.s0));
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    gx = convert_float4((VEC_DATA_TYPE(DATA_TYPE, 4))(temp_gx0.s67, temp_gx1.s01));
    gy = convert_float4((VEC_DATA_TYPE(DATA_TYPE, 4))(temp_gy0.s67, temp_gy1.s01));
    gx2 += (gx * gx);
    gy2 += (gy * gy);
    gxgy += (gx * gy);

    return (float16)(gx2, gy2, gxgy, (float4)0);
}

/** Function running harris score on 7x7 block size
 *
 * @attention: The input data type should be passed using a compile option -DDATA_TYPE. Supported types: short and int.
 *             e.g. -DDATA_TYPE=short.
 *
 * @param[in]  src_gx_ptr                           Pointer to the first source image. Supported data types: S16, S32
 * @param[in]  src_gx_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_gx_step_x                        src_gx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_gx_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_gx_step_y                        src_gx_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_gx_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  src_gy_ptr                           Pointer to the second source image. Supported data types: S16, S32
 * @param[in]  src_gy_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  src_gy_step_x                        src_gy_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_gy_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  src_gy_step_y                        src_gy_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_gy_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[out] vc_ptr                               Pointer to the destination image. Supported data types: F32
 * @param[in]  vc_stride_x                          Stride of the destination image in X dimension (in bytes)
 * @param[in]  vc_step_x                            vc_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  vc_stride_y                          Stride of the destination image in Y dimension (in bytes)
 * @param[in]  vc_step_y                            vc_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  vc_offset_first_element_in_bytes     The offset of the first element in the destination image
 * @param[in]  sensitivity                          Sensitivity threshold k from the Harris-Stephens equation
 * @param[in]  strength_thresh                      Minimum threshold with which to eliminate Harris Corner scores
 * @param[in]  pow4_normalization_factor            Normalization factor to apply harris score
 */
__kernel void harris_score_7x7(
    IMAGE_DECLARATION(src_gx),
    IMAGE_DECLARATION(src_gy),
    IMAGE_DECLARATION(vc),
    float sensitivity,
    float strength_thresh,
    float pow4_normalization_factor)
{
    Image src_gx = CONVERT_TO_IMAGE_STRUCT(src_gx);
    Image src_gy = CONVERT_TO_IMAGE_STRUCT(src_gy);
    Image vc     = CONVERT_TO_IMAGE_STRUCT(vc);

    /* Gx^2, Gy^2 and Gx*Gy */
    float16 res = (float16)0.0f;

    /* Compute row */
    for(int i = -3; i < 4; i++)
    {
        res += harris_score_1x7(&src_gx, &src_gy, i);
    }

    float4 gx2  = res.s0123;
    float4 gy2  = res.s4567;
    float4 gxgy = res.s89AB;

    /* Compute trace and determinant */
    float4 trace = gx2 + gy2;
    float4 det   = gx2 * gy2 - (gxgy * gxgy);

    /* Compute harris score */
    float4 mc = (det - (sensitivity * (trace * trace))) * pow4_normalization_factor;

    mc = select(0.0f, mc, mc > (float4)strength_thresh);

    vstore4(mc, 0, (__global float *)vc.ptr);
}
