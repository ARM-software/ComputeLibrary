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

#if defined(FIXED_POINT_POSITION)

#include "fixed_point.h"

#ifdef SATURATE
#define CONVERT_DOWN(x, in_type, out_type, fixed_point_position) CONVERT_DOWN1_SAT(x, in_type, out_type, fixed_point_position)
#define CONVERT_DOWN1_SAT(x, in_type, out_type, fixed_point_position) convert_##out_type##_##in_type##_sat(x, fixed_point_position)
#else /* SATURATE */
#define CONVERT_DOWN(x, in_type, out_type, fixed_point_position) CONVERT_DOWN1(x, in_type, out_type, fixed_point_position)
#define CONVERT_DOWN1(x, in_type, out_type, fixed_point_position) convert_##out_type##_##in_type(x, fixed_point_position)
#endif /* SATURATE */

#define CONVERT_UP(x, in_type, out_type, fixed_point_position) CONVERT_UP1(x, in_type, out_type, fixed_point_position)
#define CONVERT_UP1(x, in_type, out_type, fixed_point_position) convert_##out_type##_##in_type(x, fixed_point_position)

#else /* FIXED_POINT_POSITION */

#ifdef SATURATE
#define CONVERT_DOWN(x, type) CONVERT_SAT(x, type)
#else /* SATURATE */
#define CONVERT_DOWN(x, type) CONVERT(x, type)
#endif /* SATURATE */

#define CONVERT_UP(x, type) CONVERT(x, type)

#endif /* FIXED_POINT_POSITION */

/** This function performs a down-scaling depth conversion.
 *
 * @attention The input and output data_types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 *
 * @note In case of fixed-point operation -DFIXED_POINT_POSITION=fixed_point_position must be provided: e.g. -DFIXED_POINT_POSITION=3
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8, U16, S16, U32, S32, F16, F32
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: QS8, U8, QS16, U16, S16, U32, S32
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  shift                             The integer shift amount value. Supported data types: S32
 */
__kernel void convert_depth_down(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const int shift)
{
    // Get pixels pointer
    Image in  = CONVERT_TO_IMAGE_STRUCT(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE_IN, 16)
    in_data = vload16(0, (__global DATA_TYPE_IN *)in.ptr);

#if defined(FIXED_POINT_POSITION)
    vstore16(CONVERT_DOWN(in_data, VEC_DATA_TYPE(DATA_TYPE_IN, 16), VEC_DATA_TYPE(DATA_TYPE_OUT, 16), FIXED_POINT_POSITION), 0, (__global DATA_TYPE_OUT *)out.ptr);
#else  /* FIXED_POINT_POSITION */
    vstore16(CONVERT_DOWN(in_data >> shift, VEC_DATA_TYPE(DATA_TYPE_OUT, 16)), 0, (__global DATA_TYPE_OUT *)out.ptr);
#endif /* FIXED_POINT_POSITION */
}

/** This function performs a up-scaling depth conversion.
 *
 * @attention The input and output data_types need to be passed at compile time using -DDATA_TYPE_IN and -DDATA_TYPE_OUT:
 * e.g. -DDATA_TYPE_IN=uchar -DDATA_TYPE_OUT=short
 *
 * @note In case of fixed-point operation -DFIXED_POINT_POSITION=fixed_point_position must be provided: e.g. -DFIXED_POINT_POSITION=3
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8, QS8, U16, S16, QS16, U32 or S32
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  in_step_y                         in_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8, U16, S16, U32, S32, F16 or F32
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination image
 * @param[in]  shift                             The integer shift amount value. Supported data types: S32
 */
__kernel void convert_depth_up(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out),
    const int shift)
{
    // Get pixels pointer
    Image in  = CONVERT_TO_IMAGE_STRUCT(in);
    Image out = CONVERT_TO_IMAGE_STRUCT(out);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE_IN, 16)
    in_data = vload16(0, (__global DATA_TYPE_IN *)in.ptr);

#if defined(FIXED_POINT_POSITION)
    vstore16(CONVERT_UP(in_data, VEC_DATA_TYPE(DATA_TYPE_IN, 16), VEC_DATA_TYPE(DATA_TYPE_OUT, 16), FIXED_POINT_POSITION), 0, (__global DATA_TYPE_OUT *)out.ptr);
#else  /* FIXED_POINT_POSITION */
    vstore16(CONVERT_UP(in_data, VEC_DATA_TYPE(DATA_TYPE_OUT, 16)) << shift, 0, (__global DATA_TYPE_OUT *)out.ptr);
#endif /* FIXED_POINT_POSITION */
}
