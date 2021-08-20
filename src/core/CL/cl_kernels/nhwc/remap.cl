/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "warp_helpers.h"

#ifdef DEPTH_OUT
/** Performs a remapping of an input image to an output given two remapping image using nearest neighbor as interpolation.
 *  Also applies constant border value, "border_val", if "CONSTANT_BORDER" is set.
 *
 * This kernel performs remapping with this method of pixel coordinate translation:
 *     out(x,y) = in(mapx(x,y), mapy(x,y));
 *
 * @param[in]  in_ptr                             Pointer to the source image. Supported data types: U8,F16.
 * @param[in]  in_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                          in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                          in_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes   Offset of the first element in the source image
 * @param[out] out_ptr                            Pointer to the destination image. Supported data types: U8,F16.
 * @param[in]  out_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                         out_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  out_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                         out_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes  Offset of the first element in the destination image
 * @param[in]  mapx_ptr                           Pointer to the x remapping image. Supported data types: F32.
 * @param[in]  mapx_stride_x                      Stride of the remapping image in X dimension (in bytes)
 * @param[in]  mapx_step_x                        mapx_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  mapx_stride_y                      Stride of the remapping image in Y dimension (in bytes)
 * @param[in]  mapx_step_y                        mapy_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  mapx_offset_first_element_in_bytes Offset of the first element in the remapping image
 * @param[in]  mapy_ptr                           Pointer to the x remapping image. Supported data types: F32.
 * @param[in]  mapy_stride_x                      Stride of the remapping image in X dimension (in bytes)
 * @param[in]  mapy_step_x                        mapy_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  mapy_stride_y                      Stride of the remapping image in Y dimension (in bytes)
 * @param[in]  mapy_step_y                        mapy_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  mapy_offset_first_element_in_bytes Offset of the first element in the remapping image
 * @param[in]  width                              Width of the input image
 * @param[in]  height                             Height of the input image
 * @param[in]  border_val                         Value to use for border around input tensor when in CONSTANT border is selected
 */
__kernel void remap_nearest_neighbour_nhwc(
    TENSOR4D_DECLARATION(in),
    TENSOR4D_DECLARATION(out),
    TENSOR4D_DECLARATION(mapx),
    TENSOR4D_DECLARATION(mapy),
    const float width,
    const float height
#ifdef CONSTANT_BORDER
    ,
    const DATA_TYPE border_val
#endif // CONSTANT_BORDER
)
{
    Tensor4D in   = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(in, 0);
    Tensor4D out  = CONVERT_TO_TENSOR4D_STRUCT(out, DEPTH_OUT);
    Tensor4D mapx = CONVERT_TO_TENSOR4D_STRUCT(mapx, DEPTH_OUT);
    Tensor4D mapy = CONVERT_TO_TENSOR4D_STRUCT(mapy, DEPTH_OUT);

    float mapx_coord = (float) * (__global float *)mapx.ptr;
    float mapy_coord = (float) * (__global float *)mapy.ptr;

#ifdef CONSTANT_BORDER
    if(mapx_coord < 0 || mapx_coord > width - 1 || mapy_coord < 0 || mapy_coord > height - 1)
    {
        *((__global DATA_TYPE *)out.ptr) = border_val;
        return;
    }
#else  // CONSTANT_BORDER
    mapx_coord = clamp(mapx_coord, 0.0f, width - 1);
    mapy_coord = clamp(mapy_coord, 0.0f, height - 1);
#endif // CONSTANT_BORDER
    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(mapx_coord), convert_int(mapy_coord), (get_global_id(2) / DEPTH_OUT)));
}

/** Performs a remapping of an input image to an output given two remapping image using bilinear as interpolation.
 *  Also applies constant border value, "border_val", if "CONSTANT_BORDER" is set.
 *
 * This kernel performs remapping with this method of pixel coordinate translation:
 *     out(x,y) = in(mapx(x,y), mapy(x,y));
 *
 * @param[in]  in_ptr                             Pointer to the source image. Supported data types: U8,F16.
 * @param[in]  in_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                          in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                          in_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes   Offset of the first element in the source image
 * @param[out] out_ptr                            Pointer to the destination image. Supported data types: U8,F16.
 * @param[in]  out_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                         out_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  out_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                         out_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes  Offset of the first element in the destination image
 * @param[in]  mapx_ptr                           Pointer to the x remapping image. Supported data types: F32.
 * @param[in]  mapx_stride_x                      Stride of the remapping image in X dimension (in bytes)
 * @param[in]  mapx_step_x                        mapx_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  mapx_stride_y                      Stride of the remapping image in Y dimension (in bytes)
 * @param[in]  mapx_step_y                        mapy_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  mapx_offset_first_element_in_bytes Offset of the first element in the remapping image
 * @param[in]  mapy_ptr                           Pointer to the x remapping image. Supported data types: F32.
 * @param[in]  mapy_stride_x                      Stride of the remapping image in X dimension (in bytes)
 * @param[in]  mapy_step_x                        mapy_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  mapy_stride_y                      Stride of the remapping image in Y dimension (in bytes)
 * @param[in]  mapy_step_y                        mapy_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  mapy_offset_first_element_in_bytes Offset of the first element in the remapping image
 * @param[in]  width                              Width of the input image
 * @param[in]  height                             Height of the input image
 * @param[in]  border_val                         Value to use for border around input tensor when in CONSTANT border is selected
 */
__kernel void remap_bilinear_nhwc(
    TENSOR4D_DECLARATION(in),
    TENSOR4D_DECLARATION(out),
    TENSOR4D_DECLARATION(mapx),
    TENSOR4D_DECLARATION(mapy),
    const float width,
    const float height
#ifdef CONSTANT_BORDER
    ,
    const DATA_TYPE border_val
#endif // CONSTANT_BORDER
)
{
    Tensor4D in   = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(in, 0);
    Tensor4D out  = CONVERT_TO_TENSOR4D_STRUCT(out, DEPTH_OUT);
    Tensor4D mapx = CONVERT_TO_TENSOR4D_STRUCT(mapx, DEPTH_OUT);
    Tensor4D mapy = CONVERT_TO_TENSOR4D_STRUCT(mapy, DEPTH_OUT);

    float mapx_coord = (float) * (__global float *)mapx.ptr;
    float mapy_coord = (float) * (__global float *)mapy.ptr;

#ifdef CONSTANT_BORDER
    if(mapx_coord < 0 || mapx_coord > width - 1 || mapy_coord < 0 || mapy_coord > height - 1)
    {
        *((__global DATA_TYPE *)out.ptr) = border_val;
        return;
    }
#endif // CONSTANT_BORDER

    const float new_xf     = floor(mapx_coord);
    const float new_yf     = floor(mapy_coord);
    const float clamped_x  = clamp(new_xf, 0.0f, width - 1);
    const float clamped_x1 = clamp(new_xf + 1, 0.0f, width - 1);
    const float clamped_y  = clamp(new_yf, 0.0f, height - 1);
    const float clamped_y1 = clamp(new_yf + 1, 0.0f, height - 1);

    float4 ins = (float4)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                          *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                          *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))),
                          *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))));

    const float a  = mapx_coord - new_xf;
    const float b  = 1.f - a;
    const float a1 = mapy_coord - new_yf;
    const float b1 = 1.f - a1;
    const float fr = ((ins.s0 * b * b1) + (ins.s1 * a * b1) + (ins.s2 * b * a1) + (ins.s3 * a * a1));

    *((__global DATA_TYPE *)out.ptr) = CONVERT(fr, DATA_TYPE);
}

#endif // DEPTH_OUT