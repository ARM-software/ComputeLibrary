/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "helpers_asymm.h"
#include "warp_helpers_quantized.h"

#if defined(DEPTH_OUT)
/** Performs scale on an image interpolating with the BILINEAR method. (NHWC)
 *
 * @note Sampling policy to be used is passed as -DSAMPLING_POLICY_(TYPE) e.g. -DSAMPLING_POLICY_TOP_LEFT
 * @note Scale value for QASYMM8 data type to used is passed as -DSCALE=<VALUE> e.g. -DSCALE=0.5
 * @note Offset value for QASYMM8 data type to used is passed as -DOFFSET=<VALUE> e.g. -DOFFSET=1
 * @note If border mode replicate is used, is should be passed as -DBORDER_MODE_REPLICATE
 * @note Output tensor's depth should be given as a preprocessor argument using -DDEPTH_OUT=size. e.g. -DDEPTH=16
 * @note The value to be used at the edges of the images shoud be given as a preprocessor argument using -DCONSTANT_VALUE=value.
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: QASYMM8.
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
 * @param[in]  constant_border_value             Constant border value to use
 */
__kernel void scale_bilinear_quantized_nhwc(
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
    const bool check_x  = (0.f <= new_xf && new_xf < input_width);
    const bool check_x1 = (-1.f <= new_xf && new_xf < input_width - 1);
    const bool check_y  = (0.f <= new_yf && new_yf < input_height);
    const bool check_y1 = (-1.f <= new_yf && new_yf < input_height - 1);
    const int ins_0     = select((int)(CONSTANT_VALUE), (int)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y),
                                                                                                      (get_global_id(2) / DEPTH_OUT)))),
                                 check_x && check_y);
    const int ins_1 = select((int)(CONSTANT_VALUE), (int)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y),
                                                                                                  (get_global_id(2) / DEPTH_OUT)))),
                             check_x1 && check_y);
    const int ins_2 = select((int)(CONSTANT_VALUE), (int)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1),
                                                                                                  (get_global_id(2) / DEPTH_OUT)))),
                             check_x && check_y1);
    const int ins_3 = select((int)(CONSTANT_VALUE), (int)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1),
                                                                                                  (get_global_id(2) / DEPTH_OUT)))),
                             check_x1 && check_y1);
    int4 ins = (int4)(ins_0, ins_1, ins_2, ins_3);
#else  /* BORDER_MODE_REPLICATE */
    int4 ins          = (int4)(*((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                               *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y), (get_global_id(2) / DEPTH_OUT))),
                               *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))),
                               *((__global DATA_TYPE *)tensor4D_offset(&in, get_global_id(0), convert_int(clamped_x1), convert_int(clamped_y1), (get_global_id(2) / DEPTH_OUT))));
#endif /* BORDER_MODE_REPLICATE */

    const float  a      = new_x - new_xf;
    const float  b      = 1.f - a;
    const float  a1     = new_y - new_yf;
    const float  b1     = 1.f - a1;
    const float4 insf32 = convert_float4(ins - (int4)OFFSET) * (float4)SCALE;

    const float fr = ((insf32.s0 * b * b1) + (insf32.s1 * a * b1) + (insf32.s2 * b * a1) + (insf32.s3 * a * a1));

    DATA_TYPE res = CONVERT_SAT(convert_int_sat_rtp(fr / SCALE) + OFFSET, DATA_TYPE);

    *((__global DATA_TYPE *)out.ptr) = res;
}
#endif /* defined(DEPTH_OUT) */