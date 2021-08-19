/*
 * Copyright (c) 2016-2021 Arm Limited.
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
 */
__kernel void scale_nearest_neighbour_nhwc(
    TENSOR4D_DECLARATION(in),
    TENSOR4D_DECLARATION(out))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(in, 0);
    Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(out, DEPTH_OUT);

#ifdef SAMPLING_POLICY_TOP_LEFT
    float new_x = get_global_id(1) * SCALE_X;
    float new_y = (get_global_id(2) % DEPTH_OUT) * SCALE_Y;
#elif SAMPLING_POLICY_CENTER
    float       new_x = (get_global_id(1) + 0.5f) * SCALE_X;
    float       new_y = ((get_global_id(2) % DEPTH_OUT) + 0.5f) * SCALE_Y;
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */
#ifdef ALIGN_CORNERS
    new_x = round(new_x);
    new_y = round(new_y);
#endif /* ALIGN_CORNERS */
    const float clamped_x = clamp(new_x, 0.0f, (float)SRC_WIDTH - 1);
    const float clamped_y = clamp(new_y, 0.0f, (float)SRC_HEIGHT - 1);

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
 *
 */
__kernel void scale_bilinear_nhwc(
    TENSOR4D_DECLARATION(in),
    TENSOR4D_DECLARATION(out))
{
    Tensor4D in  = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(in, 0);
    Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT(out, DEPTH_OUT);

#ifdef SAMPLING_POLICY_TOP_LEFT
    const float new_x = get_global_id(1) * SCALE_X;
    const float new_y = (get_global_id(2) % DEPTH_OUT) * SCALE_Y;
#elif SAMPLING_POLICY_CENTER
    const float new_x = (get_global_id(1) + 0.5f) * SCALE_X - 0.5f;
    const float new_y = ((get_global_id(2) % DEPTH_OUT) + 0.5f) * SCALE_Y - 0.5f;
#else /* SAMPLING_POLICY */
#error("Unsupported sampling policy");
#endif /* SAMPLING_POLICY */

    const float new_xf     = floor(new_x);
    const float new_yf     = floor(new_y);
    const float clamped_x  = clamp(new_xf, 0.0f, SRC_WIDTH - 1.f);
    const float clamped_x1 = clamp(new_xf + 1, 0.0f, SRC_WIDTH - 1.f);
    const float clamped_y  = clamp(new_yf, 0.0f, SRC_HEIGHT - 1.f);
    const float clamped_y1 = clamp(new_yf + 1, 0.0f, SRC_HEIGHT - 1.f);

#if defined(OFFSET) && defined(SCALE)
#define IN_DATA_TYPE int
#else // defined(OFFSET) && defined(SCALE)
#define IN_DATA_TYPE float
#endif // defined(OFFSET) && defined(SCALE)

#ifndef BORDER_MODE_REPLICATE
    const bool check_x  = (0.f <= new_xf && new_xf < (float)SRC_WIDTH);
    const bool check_x1 = (-1.f <= new_xf && new_xf < SRC_WIDTH - 1.f);
    const bool check_y  = (0.f <= new_yf && new_yf < (float)SRC_HEIGHT);
    const bool check_y1 = (-1.f <= new_yf && new_yf < SRC_HEIGHT - 1.f);

    const IN_DATA_TYPE ins_0 = select((IN_DATA_TYPE)(CONSTANT_VALUE), (IN_DATA_TYPE)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y),
                                                                                       (get_global_id(2) / DEPTH_OUT)))),
                                      check_x && check_y);
    const IN_DATA_TYPE ins_1 = select((IN_DATA_TYPE)(CONSTANT_VALUE), (IN_DATA_TYPE)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y),
                                                                                       (get_global_id(2) / DEPTH_OUT)))),
                                      check_x1 && check_y);
    const IN_DATA_TYPE ins_2 = select((IN_DATA_TYPE)(CONSTANT_VALUE), (IN_DATA_TYPE)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1),
                                                                                       (get_global_id(2) / DEPTH_OUT)))),
                                      check_x && check_y1);
    const IN_DATA_TYPE ins_3 = select((IN_DATA_TYPE)(CONSTANT_VALUE), (IN_DATA_TYPE)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1),
                                                                                       (get_global_id(2) / DEPTH_OUT)))),
                                      check_x1 && check_y1);
    VEC_DATA_TYPE(IN_DATA_TYPE, 4)
    ins = (VEC_DATA_TYPE(IN_DATA_TYPE, 4))(ins_0, ins_1, ins_2, ins_3);
#else  /* BORDER_MODE_REPLICATE */
    VEC_DATA_TYPE(IN_DATA_TYPE, 4)
    ins = (VEC_DATA_TYPE(IN_DATA_TYPE, 4))(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                                           *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                                           *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))),
                                           *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))));
#endif /* BORDER_MODE_REPLICATE */

    const float a  = new_x - new_xf;
    const float b  = 1.f - a;
    const float a1 = new_y - new_yf;
    const float b1 = 1.f - a1;

#if defined(OFFSET) && defined(SCALE)
    const float4 insf32 = convert_float4(ins - (int4)OFFSET) * (float4)SCALE;
    const float  fr     = ((insf32.s0 * b * b1) + (insf32.s1 * a * b1) + (insf32.s2 * b * a1) + (insf32.s3 * a * a1));
    DATA_TYPE    res    = CONVERT_SAT(convert_int_sat_rtp(fr / SCALE) + OFFSET, DATA_TYPE);

    *((__global DATA_TYPE *)out.ptr) = res;
#else  // defined(OFFSET) && defined(SCALE)
    const float fr = ((ins.s0 * b * b1) + (ins.s1 * a * b1) + (ins.s2 * b * a1) + (ins.s3 * a * a1));

    *((__global DATA_TYPE *)out.ptr) = CONVERT(fr, DATA_TYPE);
#endif // defined(OFFSET) && defined(SCALE)
}
#endif /* defined(DEPTH_OUT) */