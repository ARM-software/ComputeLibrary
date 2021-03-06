/*
 * Copyright (c) 2016-2020 Arm Limited.
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
    Image        in          = CONVERT_TO_IMAGE_STRUCT_NO_STEP(in);
    Image        out         = CONVERT_TO_IMAGE_STRUCT(out);
    const float2 r           = (float2)(scale_x, scale_y);
    float8       transformed = transform_nearest(get_current_coords(), r);
#ifdef ALIGN_CORNERS
    transformed = round(transformed);
#endif // ALIGN_CORNERS
    const float8 tc = clamp_to_border_with_size(transformed, input_width, input_height, BORDER_SIZE);
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
    float new_x = get_global_id(1) * scale_x;
    float new_y = (get_global_id(2) % DEPTH_OUT) * scale_y;
#elif SAMPLING_POLICY_CENTER
    float       new_x = (get_global_id(1) + 0.5f) * scale_x;
    float       new_y = ((get_global_id(2) % DEPTH_OUT) + 0.5f) * scale_y;
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */
#ifdef ALIGN_CORNERS
    new_x = round(new_x);
    new_y = round(new_y);
#endif /* ALIGN_CORNERS */
    const float clamped_x = clamp(new_x, 0.0f, input_width - 1);
    const float clamped_y = clamp(new_y, 0.0f, input_height - 1);

    *((__global DATA_TYPE *)out.ptr) = *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT)));
}

/** Performs scale on an image interpolating with the BILINEAR method. (NHWC)
 *
 * @note Sampling policy to be used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note If border mode replicate is used, is should be passed as -DBORDER_MODE_REPLICATE
 * @note Output tensor's depth should be given as a preprocessor argument using -DDEPTH_OUT=size. e.g. -DDEPTH=16
 * @note The value to be used at the edges of the images shoud be given as a preprocessor argument using -DCONSTANT_VALUE=value.
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
 *
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

    const float new_xf     = floor(new_x);
    const float new_yf     = floor(new_y);
    const float clamped_x  = clamp(new_xf, 0.0f, input_width - 1);
    const float clamped_x1 = clamp(new_xf + 1, 0.0f, input_width - 1);
    const float clamped_y  = clamp(new_yf, 0.0f, input_height - 1);
    const float clamped_y1 = clamp(new_yf + 1, 0.0f, input_height - 1);

#ifndef BORDER_MODE_REPLICATE
    const bool  check_x = (0.f <= new_xf && new_xf < input_width);
    const bool  check_x1 = (-1.f <= new_xf && new_xf < input_width - 1);
    const bool  check_y = (0.f <= new_yf && new_yf < input_height);
    const bool  check_y1 = (-1.f <= new_yf && new_yf < input_height - 1);
    const float ins_0   = select((float)(CONSTANT_VALUE), (float)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y),
                                                                                                          (get_global_id(2) / DEPTH_OUT)))),
                                 check_x && check_y);
    const float ins_1 = select((float)(CONSTANT_VALUE), (float)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y),
                                                                                                        (get_global_id(2) / DEPTH_OUT)))),
                                 check_x1 && check_y);
    const float ins_2 = select((float)(CONSTANT_VALUE), (float)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1),
                                                                                                        (get_global_id(2) / DEPTH_OUT)))),
                                 check_x && check_y1);
    const float ins_3 = select((float)(CONSTANT_VALUE), (float)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1),
                                                                                                        (get_global_id(2) / DEPTH_OUT)))),
                                 check_x1 && check_y1);
    float4 ins = (float4)(ins_0, ins_1, ins_2, ins_3);
#else  /* BORDER_MODE_REPLICATE */
    float4 ins        = (float4)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                                 *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                                 *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))),
                                 *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))));
#endif /* BORDER_MODE_REPLICATE */

    const float a  = new_x - new_xf;
    const float b  = 1.f - a;
    const float a1 = new_y - new_yf;
    const float b1 = 1.f - a1;
    const float fr = ((ins.s0 * b * b1) + (ins.s1 * a * b1) + (ins.s2 * b * a1) + (ins.s3 * a * a1));

    *((__global DATA_TYPE *)out.ptr) = CONVERT(fr, DATA_TYPE);
}
#endif /* defined(DEPTH_OUT) */