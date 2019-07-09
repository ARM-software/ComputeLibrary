/*
 * Copyright (c) 2016-2019 ARM Limited.
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
#ifdef SAMPLING_POLICY_TOP_LEFT
    const float4 in_x_coords = (float4)(coord.s0, 1 + coord.s0, 2 + coord.s0, 3 + coord.s0);
    const float4 new_x       = in_x_coords * (float4)(scale.s0);
    const float4 new_y       = (float4)(coord.s1 * scale.s1);
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
#elif SAMPLING_POLICY_CENTER
    const float4 in_x_coords = (float4)(coord.s0, 1 + coord.s0, 2 + coord.s0, 3 + coord.s0);
    const float4 new_x       = (in_x_coords + ((float4)(0.5f))) * (float4)(scale.s0);
    const float4 new_y       = (float4)((coord.s1 + 0.5f) * scale.s1);
    return (float8)(new_x.s0, new_y.s0, new_x.s1, new_y.s1, new_x.s2, new_y.s2, new_x.s3, new_y.s3);
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */
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
__kernel void scale_nearest_neighbour_nchw(
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
__kernel void scale_bilinear_nchw(
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

#if defined(DEPTH_OUT)
/** Performs scale on an image interpolating with the NEAREAST NEIGHBOUR method. Input and output are single channel F32. (NHWC)
 *
 * @note Sampling policy to used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note Output tensor's depth should be given as a preprocessor argument using -DDEPTH_OUT=size. e.g. -DDEPTH=16
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8/S16/F16/F32.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source image in Z dimension (in bytes)
 * @param[in]  in_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: same as @p in_ptr
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Z dimension (in bytes)
 * @param[in]  out_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  input_width                       Input image width
 * @param[in]  input_height                      Input image height
 * @param[in]  scale_x                           The scale factor along x dimension
 * @param[in]  scale_y                           The scale factor along y dimension
 */
__kernel void scale_nearest_neighbour_nhwc(
    TENSOR4D_DECLARATION(in),
    TENSOR4D_DECLARATION(out),
    const float input_width,
    const float input_height,
    const float scale_x,
    const float scale_y)
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(in, 0);
    Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(out, DEPTH_OUT);

#ifdef SAMPLING_POLICY_TOP_LEFT
    const float new_x = get_global_id(1) * scale_x;
    const float new_y = (get_global_id(2) % DEPTH_OUT) * scale_y;
#elif SAMPLING_POLICY_CENTER
    const float new_x = (get_global_id(1) + 0.5f) * scale_x;
    const float new_y = ((get_global_id(2) % DEPTH_OUT) + 0.5f) * scale_y;
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */
    const float clamped_x = clamp(new_x, 0.0f, input_width - 1);
    const float clamped_y = clamp(new_y, 0.0f, input_height - 1);

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT)));
}

/** Performs scale on an image interpolating with the BILINEAR method. (NHWC)
 *
 * @note Sampling policy to be used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note If border mode replicate is used, is should be passed as -DBORDER_MODE_REPLICATE
 * @note Output tensor's depth should be given as a preprocessor argument using -DDEPTH_OUT=size. e.g. -DDEPTH=16
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8/S16/F16/F32.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source image in Z dimension (in bytes)
 * @param[in]  in_step_z                         src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: same as @p in_ptr
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the destination image in Z dimension (in bytes)
 * @param[in]  out_step_z                        dst_stride_y * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  input_width                       Input image width
 * @param[in]  input_height                      Input image height
 * @param[in]  scale_x                           The scale factor along x dimension
 * @param[in]  scale_y                           The scale factor along y dimension
 */
__kernel void scale_bilinear_nhwc(
    TENSOR4D_DECLARATION(in),
    TENSOR4D_DECLARATION(out),
    const float input_width,
    const float input_height,
    const float scale_x,
    const float scale_y)
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(in, 0);
    Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(out, DEPTH_OUT);

#ifdef SAMPLING_POLICY_TOP_LEFT
    const float new_x = get_global_id(1) * scale_x;
    const float new_y = (get_global_id(2) % DEPTH_OUT) * scale_y;
#elif SAMPLING_POLICY_CENTER
    const float new_x = (get_global_id(1) + 0.5f) * scale_x - 0.5f;
    const float new_y = ((get_global_id(2) % DEPTH_OUT) + 0.5f) * scale_y - 0.5f;
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */

    const float new_xf      = floor(new_x);
    const float new_yf      = floor(new_y);
    float       clamped_x   = clamp(new_xf, 0.0f, input_width - 1);
    float       clamped_x1  = clamp(new_xf + 1, 0.0f, input_width - 1);
    float       clamped_x_  = clamped_x;
    float       clamped_x1_ = clamped_x1;
    const float clamped_y   = clamp(new_yf, 0.0f, input_height - 1);
    const float clamped_y1  = clamp(new_yf + 1, 0.0f, input_height - 1);

#ifndef BORDER_MODE_REPLICATE
    clamped_x1  = select(clamped_x1, 0.0f - BORDER_SIZE, new_yf + 1 < 0.f || new_yf + 1 > input_height - 1 || new_xf + 1 < 0.f || new_xf + 1 > input_width - 1);
    clamped_x_  = select(clamped_x_, 0.0f - BORDER_SIZE, new_yf + 1 > input_height - 1 || new_xf < 0.f || new_xf > input_width - 1);
    clamped_x   = select(clamped_x, 0.0f - BORDER_SIZE, new_yf < 0.f || new_yf > input_height - 1 || new_xf < 0.f || new_xf > input_width - 1);
    clamped_x1_ = select(clamped_x1_, 0.0f - BORDER_SIZE, new_xf + 1 < 0.f || new_xf + 1 > input_width - 1 || new_yf < 0.f || new_yf > input_height - 1);
#endif /* BORDER_MODE_REPLICATE */

    float4 ins = (float4)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                          *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1_), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                          *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x_), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))),
                          *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))));

    const float a  = new_x - new_xf;
    const float b  = 1.f - a;
    const float a1 = new_y - new_yf;
    const float b1 = 1.f - a1;
    const float fr = ((ins.s0 * b * b1) + (ins.s1 * a * b1) + (ins.s2 * b * a1) + (ins.s3 * a * a1));

    *((__global DATA_TYPE *)out.ptr) = CONVERT(fr, DATA_TYPE);
}
#endif /* defined(DEPTH_OUT) */