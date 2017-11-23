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
#include "warp_helpers.h"

/** Returns the perspective matrix */
inline const float16 build_perspective_mtx()
{
    return (float16)(MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, MAT6, MAT7, MAT8, 0, 0, 0, (float4)0);
}

/** Transforms four 2D coordinates using the formula:
 *
 *   x0 = M[1][1] * x + M[1][2] * y + M[1][3]
 *   y0 = M[2][1] * x + M[2][2] * y + M[2][3]
 *   z0 = M[3][1] * x + M[3][2] * y + M[3][3]
 *
 *   (x0/z0,y0/z0)
 *
 * @param[in] coord 2D coordinate to transform.
 * @param[in] mtx   perspective matrix
 *
 * @return a vector float8 containing four 2D transformed values.
 */
inline const float8 apply_perspective_transform(const float2 coord, const float16 mtx)
{
    const float4 in_x_coords = (float4)(coord.s0, 1 + coord.s0, 2 + coord.s0, 3 + coord.s0);
    // transform [z,z+1,z+2,z+3]
    const float4 z = (float4)mad(in_x_coords, (float4)(mtx.s2), mad((float4)(coord.s1), (float4)(mtx.s5), (float4)(mtx.s8)));
    // NOTE: Do not multiply x&y by 1.f/Z as this will result in loss of accuracy and mismatches with VX reference implementation
    // transform [x,x+1,x+2,x+3]
    const float4 new_x = (float4)mad(in_x_coords, (float4)(mtx.s0), mad((float4)(coord.s1), (float4)(mtx.s3), (float4)(mtx.s6))) / z;
    // transform [y,y+1,y+2,y+3]
    const float4 new_y = (float4)mad(in_x_coords, (float4)(mtx.s1), mad((float4)(coord.s1), (float4)(mtx.s4), (float4)(mtx.s7))) / z;
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
}

/** Performs perspective transformation on an image interpolating with the NEAREAST NEIGHBOUR method. Input and output are single channel U8.
 *
 * This kernel performs perspective transform with a 3x3 Matrix M with this method of pixel coordinate translation:
 *   x0 = M[1][1] * x + M[1][2] * y + M[1][3]
 *   y0 = M[2][1] * x + M[2][2] * y + M[2][3]
 *   z0 = M[3][1] * x + M[3][2] * y + M[3][3]
 *
 *   output(x,y) = input(x0/z0,y0/z0)
 *
 * @attention The matrix coefficients need to be passed at compile time:\n
 * const char build_options [] = "-DMAT0=1 -DMAT1=2 -DMAT2=3 -DMAT3=4 -DMAT4=5 -DMAT5=6 -DMAT6=7 -DMAT7=8 -DMAT8=9"\n
 * clBuildProgram( program, 0, NULL, build_options, NULL, NULL);
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes  Offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8.
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes Offset of the first element in the destination image
 * @param[in]  width                             Width of the destination image
 * @param[in]  height                            Height of the destination image
 */
__kernel void warp_perspective_nearest_neighbour(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const int width,
    const int height)
{
    Image in  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);
    vstore4(read_texels4(&in, convert_int8_rtn(clamp_to_border(apply_perspective_transform(get_current_coords(), build_perspective_mtx()), width, height))), 0, out.ptr);
}

/** Performs a perspective transform on an image interpolating with the BILINEAR method. Input and output are single channel U8.
 *
 * @attention The matrix coefficients need to be passed at compile time:\n
 * const char build_options [] = "-DMAT0=1 -DMAT1=2 -DMAT2=3 -DMAT3=4 -DMAT4=5 -DMAT5=6 -DMAT6=7 -DMAT7=8 -DMAT8=9"\n
 * clBuildProgram( program, 0, NULL, build_options, NULL, NULL);
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes  Offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8.
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes Offset of the first element in the destination image
 * @param[in]  width                             Width of the destination image
 * @param[in]  height                            Height of the destination image
 */
__kernel void warp_perspective_bilinear(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const int width,
    const int height)
{
    Image in  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);
    vstore4(bilinear_interpolate(&in, apply_perspective_transform(get_current_coords(), build_perspective_mtx()), width, height), 0, out.ptr);
}
