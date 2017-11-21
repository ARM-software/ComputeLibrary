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

/** Transforms four 2D coordinates. This is used to map the output coordinates to the input coordinates.
 *
 * @param[in] coord 2D coordinates to transform.
 * @param[in] scale input/output scale ratio
 *
 * @return a float8 containing 4 2D transformed values in the input image.
 */
inline const float8 transform_nearest(const float2 coord, const float2 scale)
{
    const float4 in_x_coords = (float4)(coord.s0, 1 + coord.s0, 2 + coord.s0, 3 + coord.s0);
    const float4 new_x       = (in_x_coords + ((float4)(0.5f))) * (float4)(scale.s0);
    const float4 new_y       = (float4)((coord.s1 + 0.5f) * scale.s1);
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
}

/** Transforms four 2D coordinates. This is used to map the output coordinates to the input coordinates.
 *
 * @param[in] coord 2D coordinates to transform.
 * @param[in] scale input/output scale ratio
 *
 * @return a float8 containing 4 2D transformed values in the input image.
 */
inline const float8 transform_bilinear(const float2 coord, const float2 scale)
{
    const float4 in_x_coords = (float4)(coord.s0, 1 + coord.s0, 2 + coord.s0, 3 + coord.s0);
#ifdef SAMPLING_POLICY_TOP_LEFT
    const float4 new_x = in_x_coords * (float4)(scale.s0);
    const float4 new_y = (float4)(coord.s1 * scale.s1);
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
#elif SAMPLING_POLICY_CENTER
    const float4 new_x = (in_x_coords + ((float4)(0.5f))) * (float4)(scale.s0) - (float4)(0.5f);
    const float4 new_y = (float4)((coord.s1 + 0.5f) * scale.s1 - 0.5f);
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */
}

/** Performs an affine transformation on an image interpolating with the NEAREAST NEIGHBOUR method. Input and output are single channel U8 or S16.
 *
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8, S16.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8, S16. (Must be the same as the input)
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  input_width                       Input image width
 * @param[in]  input_height                      Input image height
 * @param[in]  scale_x                           The scale factor along x dimension
 * @param[in]  scale_y                           The scale factor along y dimension
 */
__kernel void scale_nearest_neighbour(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const float input_width,
    const float input_height,
    const float scale_x,
    const float scale_y)
{
    Image        in  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image        out = CONVERT_TO_IMAGE_STRUCT(out);
    const float2 r   = (float2)(scale_x, scale_y);
    const float8 tc  = clamp_to_border_with_size(transform_nearest(get_current_coords(), r), input_width, input_height, BORDER_SIZE);
    vstore4(read_texels4(&in, convert_int8(tc)), 0, (__global DATA_TYPE *)out.ptr);
}

/** Performs an affine transformation on an image interpolating with the BILINEAR method.
 *
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8, S16.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8, S16. (Must be the same as the input)
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  input_width                       Input image width
 * @param[in]  input_height                      Input image height
 * @param[in]  scale_x                           The scale factor along x dimension
 * @param[in]  scale_y                           The scale factor along y dimension
 */
__kernel void scale_bilinear(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const float input_width,
    const float input_height,
    const float scale_x,
    const float scale_y)
{
    Image        in  = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image        out = CONVERT_TO_IMAGE_STRUCT(out);
    const float2 r   = (float2)(scale_x, scale_y);
    const float8 tc  = transform_bilinear(get_current_coords(), r);
    vstore4(bilinear_interpolate_with_border(&in, tc, input_width, input_height, BORDER_SIZE), 0, (__global DATA_TYPE *)out.ptr);
}
