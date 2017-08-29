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

/** Returns a vector of floats contaning the matrix coefficients. */
inline const float8 build_affine_mtx()
{
    return (float8)(MAT0, MAT1, MAT2, MAT3, MAT4, MAT5, 0, 0);
}

/** Transforms 4 2D coordinates using the formula:
 *
 *   x0 = M[1][1] * x + M[1][2] * y + M[1][3]
 *   y0 = M[2][1] * x + M[2][2] * y + M[2][3]
 *
 * @param[in] coord 2D coordinate to transform.
 * @param[in] mtx   affine matrix
 *
 * @return a int8 containing 4 2D transformed values.
 */
inline const float8 apply_affine_transform(const float2 coord, const float8 mtx)
{
    const float4 in_x_coords = (float4)(coord.s0, 1 + coord.s0, 2 + coord.s0, 3 + coord.s0);
    // transform [x,x+1,x+2,x+3]
    const float4 new_x = mad(/*A*/ in_x_coords, (float4)(mtx.s0) /*B*/, mad((float4)(coord.s1), (float4)(mtx.s2), (float4)(mtx.s4)));
    // transform [y,y+1,y+2,y+3]
    const float4 new_y = mad(in_x_coords, (float4)(mtx.s1), mad((float4)(coord.s1), (float4)(mtx.s3), (float4)(mtx.s5)));
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
}

/** Performs an affine transform on an image interpolating with the NEAREAST NEIGHBOUR method. Input and output are single channel U8.
 *
 * This kernel performs an affine transform with a 2x3 Matrix M with this method of pixel coordinate translation:
 *   x0 = M[1][1] * x + M[1][2] * y + M[1][3]
 *   y0 = M[2][1] * x + M[2][2] * y + M[2][3]
 *   output(x,y) = input(x0,y0)
 *
 * @attention The matrix coefficients need to be passed at compile time:\n
 * const char build_options [] = "-DMAT0=1 -DMAT1=2 -DMAT2=1 -DMAT3=2 -DMAT4=4 -DMAT5=2 "\n
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
__kernel void warp_affine_nearest_neighbour(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const int width,
    const int height)
{
    Image in  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);
    vstore4(read_texels4(&in, convert_int8_rtn(clamp_to_border(apply_affine_transform(get_current_coords(), build_affine_mtx()), width, height))), 0, out.ptr);
}

/** Performs an affine transform on an image interpolating with the BILINEAR method. Input and output are single channel U8.
 *
 * @attention The matrix coefficients need to be passed at compile time:\n
 * const char build_options [] = "-DMAT0=1 -DMAT1=2 -DMAT2=1 -DMAT3=2 -DMAT4=4 -DMAT5=2 "\n
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
__kernel void warp_affine_bilinear(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const int width,
    const int height)
{
    Image in  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);
    vstore4(bilinear_interpolate(&in, apply_affine_transform(get_current_coords(), build_affine_mtx()), width, height), 0, out.ptr);
}
