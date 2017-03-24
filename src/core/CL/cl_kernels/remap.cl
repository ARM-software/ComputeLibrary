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
#include "warp_helpers.h"

/** Performs a remapping of an input image to an output given two remapping image using nearest neighbor as interpolation.
 *
 * This kernel performs remapping with this method of pixel coordinate translation:
 *     out(x,y) = in(mapx(x,y), mapy(x,y));
 *
 * @param[in]  in_ptr                             Pointer to the source image. Supported data types: U8.
 * @param[in]  in_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                          in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                          in_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes   Offset of the first element in the source image
 * @param[out] out_ptr                            Pointer to the destination image. Supported data types: U8.
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
 */
__kernel void remap_nearest_neighbour(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    IMAGE_DECLARATION(mapx),
    IMAGE_DECLARATION(mapy),
    const float width,
    const float height)
{
    Image in   = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image out  = CONVERT_TO_IMAGE_STRUCT(out);
    Image mapx = CONVERT_TO_IMAGE_STRUCT(mapx);
    Image mapy = CONVERT_TO_IMAGE_STRUCT(mapy);

    float4 mapx_coords = vload4(0, (__global float *)mapx.ptr);
    float4 mapy_coords = vload4(0, (__global float *)mapy.ptr);
    float8 map_coords  = (float8)(mapx_coords.s0, mapy_coords.s0, mapx_coords.s1, mapy_coords.s1,
                                  mapx_coords.s2, mapy_coords.s2, mapx_coords.s3, mapy_coords.s3);
    map_coords += (float8)(0.5f);

    vstore4(read_texels4(&in, convert_int8(clamp_to_border(map_coords, width, height))), 0, out.ptr);
}

/** Performs a remapping of an input image to an output given two remapping image using bilinear as interpolation.
 *
 * This kernel performs remapping with this method of pixel coordinate translation:
 *     out(x,y) = in(mapx(x,y), mapy(x,y));
 *
 * @param[in]  in_ptr                             Pointer to the source image. Supported data types: U8.
 * @param[in]  in_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                          in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                          in_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes   Offset of the first element in the source image
 * @param[out] out_ptr                            Pointer to the destination image. Supported data types: U8.
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
 */
__kernel void remap_bilinear(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    IMAGE_DECLARATION(mapx),
    IMAGE_DECLARATION(mapy),
    const float width,
    const float height)
{
    Image in   = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image out  = CONVERT_TO_IMAGE_STRUCT(out);
    Image mapx = CONVERT_TO_IMAGE_STRUCT(mapx);
    Image mapy = CONVERT_TO_IMAGE_STRUCT(mapy);

    float4 mapx_coords = vload4(0, (__global float *)mapx.ptr);
    float4 mapy_coords = vload4(0, (__global float *)mapy.ptr);
    float8 map_coords  = (float8)(mapx_coords.s0, mapy_coords.s0, mapx_coords.s1, mapy_coords.s1,
                                  mapx_coords.s2, mapy_coords.s2, mapx_coords.s3, mapy_coords.s3);

    vstore4(bilinear_interpolate(&in, clamp_to_border(map_coords, width, height), width, height), 0, out.ptr);
}
