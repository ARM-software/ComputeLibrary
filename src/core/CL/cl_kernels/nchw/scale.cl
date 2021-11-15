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
#include "tile_helpers.h"

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
 */
__kernel void scale_nearest_neighbour_nchw(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out))
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float8 transformed = transform_nearest((float2)(x * VEC_SIZE, y), (float2)(SCALE_X, SCALE_Y));
#ifdef ALIGN_CORNERS
    transformed = round(transformed);
#endif // ALIGN_CORNERS

    TILE(SELECT_DATA_TYPE(DATA_TYPE), 1, 4, cond);
    cond[0].v = CONVERT(((transformed.even < 0) || (transformed.even >= (int)SRC_WIDTH)) || ((transformed.odd < 0) || (transformed.odd >= (int)SRC_HEIGHT)), SELECT_VEC_DATA_TYPE(DATA_TYPE, 4));

    TILE(int, 1, 4, in_x);
    TILE(int, 1, 4, in_y);
    in_x[0].v = convert_int4(clamp(transformed.even, 0.f, SRC_WIDTH - 1.f));
    in_y[0].v = convert_int4(clamp(transformed.odd, 0.f, SRC_HEIGHT - 1.f));

    TILE(DATA_TYPE, 1, VEC_SIZE, out_vals);
    LOOP_UNROLLING(int, i, 0, 1, VEC_SIZE,
    {
        out_vals[0].s[i] = select(*((__global DATA_TYPE *)(in_ptr + in_offset_first_element_in_bytes + in_x[0].s[i] * sizeof(DATA_TYPE) + in_y[0].s[i] * in_stride_y)), (DATA_TYPE)CONSTANT_VALUE, cond[0].s[i]);
    })

    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x * out_step_x + y * out_stride_y;

    if(x == get_global_size(0) - 1)
    {
#if VEC_SIZE == 1
        VSTORE_PARTIAL(VEC_SIZE, VEC_SIZE_LEFTOVER)
        (out_vals[0].s[0], 0, (__global DATA_TYPE *)out_addr);
#else  // VEC_SIZE == 1
        VSTORE_PARTIAL(VEC_SIZE, VEC_SIZE_LEFTOVER)
        (out_vals[0].v, 0, (__global DATA_TYPE *)out_addr);
#endif // VEC_SIZE == 1
    }
    else
    {
#if VEC_SIZE == 1
        VSTORE(VEC_SIZE)
        (out_vals[0].s[0], 0, (__global DATA_TYPE *)out_addr);
#else  // VEC_SIZE == 1
        VSTORE(VEC_SIZE)
        (out_vals[0].v, 0, (__global DATA_TYPE *)out_addr);
#endif // VEC_SIZE == 1
    }
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
 */
__kernel void scale_bilinear_nchw(
    IMAGE_DECLARATION(in),
    IMAGE_DECLARATION(out))
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    TILE(float, 1, 8, trans_coords);
    TILE(float, 1, 8, floor_coords);
    TILE(int, 1, 16, in_x);
    TILE(int, 1, 16, in_y);

    trans_coords[0].v = transform_bilinear((float2)(x * VEC_SIZE, y), (float2)(SCALE_X, SCALE_Y));
    floor_coords[0].v = floor(trans_coords[0].v);

    LOOP_UNROLLING(int, i, 0, 1, 4,
    {
        LOOP_UNROLLING(int, j, 0, 1, 4,
        {
            in_x[0].s[i * 4 + j] = floor_coords[0].s[i * 2 + 0] + (j % 2);
            in_y[0].s[i * 4 + j] = floor_coords[0].s[i * 2 + 1] + (j > 1);
        })
    })

#if defined(BORDER_MODE_CONSTANT)
    TILE(SELECT_DATA_TYPE(DATA_TYPE), 1, 16, cond);
    cond[0].v = CONVERT(((in_x[0].v < 0) || (in_x[0].v >= (int)SRC_WIDTH)) || ((in_y[0].v < 0) || (in_y[0].v >= (int)SRC_HEIGHT)), SELECT_VEC_DATA_TYPE(DATA_TYPE, 16));
#endif // defined(BORDER_MODE_CONSTANT)

    in_x[0].v = clamp(in_x[0].v, 0, (int16)((int)SRC_WIDTH - 1));
    in_y[0].v = clamp(in_y[0].v, 0, (int16)((int)SRC_HEIGHT - 1));

    TILE(DATA_TYPE, 1, 16, in_vals);

    // Loads the values from the input image
#if defined(BORDER_MODE_CONSTANT)
    LOOP_UNROLLING(int, i, 0, 1, 16,
    {
        in_vals[0].s[i] = select(*((__global DATA_TYPE *)(in_ptr + in_offset_first_element_in_bytes + in_x[0].s[i] * sizeof(DATA_TYPE) + in_y[0].s[i] * (int)in_stride_y)), (DATA_TYPE)CONSTANT_VALUE, cond[0].s[i]);
    })
#else  // defined(BORDER_MODE_CONSTANT)
    LOOP_UNROLLING(int, i, 0, 1, 16,
    {
        in_vals[0].s[i] = *((__global DATA_TYPE *)(in_ptr + in_offset_first_element_in_bytes + in_x[0].s[i] * sizeof(DATA_TYPE) + in_y[0].s[i] * (int)in_stride_y));
    })
#endif // defined(BORDER_MODE_CONSTANT)

    TILE(float, 1, 8, a);
    TILE(float, 1, 8, b);

    a[0].v = trans_coords[0].v - floor_coords[0].v;
    b[0].v = ((float8)(1.f)) - a[0].v;

#if defined(OFFSET) && defined(SCALE)
    TILE(float, 1, 16, in_vals_f32);
    TILE(float, 1, 4, out_vals_f32);

    in_vals_f32[0].v = convert_float16(convert_int16(in_vals[0].v) - (int16)OFFSET) * (float16)SCALE;

    // Bilinear interpolation: (in0  * b0 * b1) + (in1  * a0 * b1) + (in2  * b0 * a1) + (in3  * a0 * a1)
    //                         (in4  * b2 * b3) + (in5  * a2 * b3) + (in6  * b2 * a3) + (in7  * a2 * a3)
    //                         (in8  * b4 * b5) + (in9  * a4 * b5) + (in10 * b4 * a5) + (in11 * a4 * a5)
    //                         (in12 * b6 * b7) + (in13 * a6 * b7) + (in14 * b6 * a7) + (in15 * a6 * a7)
    LOOP_UNROLLING(int, i, 0, 1, 4,
    {
        out_vals_f32[0].s[i] = (in_vals_f32[0].s[i * 4 + 0] * b[0].s[i * 2] * b[0].s[i * 2 + 1]) + (in_vals_f32[0].s[i * 4 + 1] * a[0].s[i * 2] * b[0].s[i * 2 + 1]) + (in_vals_f32[0].s[i * 4 + 2] * b[0].s[i * 2] * a[0].s[i * 2 + 1]) + (in_vals_f32[0].s[i * 4 + 3] * a[0].s[i * 2] * a[0].s[i * 2 + 1]);
    })

    TILE(DATA_TYPE, 1, 4, out_vals_4);
    TILE(DATA_TYPE, 1, VEC_SIZE, out_vals);

    out_vals_4[0].v = CONVERT_SAT(convert_int4_sat_rtp(out_vals_f32[0].v / (float)SCALE) + OFFSET, VEC_DATA_TYPE(DATA_TYPE, 4));

    LOOP_UNROLLING(int, i, 0, 1, VEC_SIZE,
    {
        out_vals[0].s[i] = out_vals_4[0].s[i];
    })
#else  // defined(OFFSET) && defined(SCALE)

    TILE(DATA_TYPE, 1, VEC_SIZE, out_vals);

    // Bilinear interpolation: (in0  * b0 * b1) + (in1  * a0 * b1) + (in2  * b0 * a1) + (in3  * a0 * a1)
    //                         (in4  * b2 * b3) + (in5  * a2 * b3) + (in6  * b2 * a3) + (in7  * a2 * a3)
    //                         (in8  * b4 * b5) + (in9  * a4 * b5) + (in10 * b4 * a5) + (in11 * a4 * a5)
    //                         (in12 * b6 * b7) + (in13 * a6 * b7) + (in14 * b6 * a7) + (in15 * a6 * a7)
    LOOP_UNROLLING(int, i, 0, 1, VEC_SIZE,
    {
        out_vals[0].s[i] = (in_vals[0].s[i * 4 + 0] * b[0].s[i * 2] * b[0].s[i * 2 + 1]) + (in_vals[0].s[i * 4 + 1] * a[0].s[i * 2] * b[0].s[i * 2 + 1]) + (in_vals[0].s[i * 4 + 2] * b[0].s[i * 2] * a[0].s[i * 2 + 1]) + (in_vals[0].s[i * 4 + 3] * a[0].s[i * 2] * a[0].s[i * 2 + 1]);
    })
#endif // defined(OFFSET) && defined(SCALE)

    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + x * out_step_x + y * out_stride_y;

    if(x == get_global_size(0) - 1)
    {
#if VEC_SIZE == 1
        VSTORE_PARTIAL(VEC_SIZE, VEC_SIZE_LEFTOVER)
        (out_vals[0].s[0], 0, (__global DATA_TYPE *)out_addr);
#else  // VEC_SIZE == 1
        VSTORE_PARTIAL(VEC_SIZE, VEC_SIZE_LEFTOVER)
        (out_vals[0].v, 0, (__global DATA_TYPE *)out_addr);
#endif // VEC_SIZE == 1
    }
    else
    {
#if VEC_SIZE == 1
        VSTORE(VEC_SIZE)
        (out_vals[0].s[0], 0, (__global DATA_TYPE *)out_addr);
#else  // VEC_SIZE == 1
        VSTORE(VEC_SIZE)
        (out_vals[0].v, 0, (__global DATA_TYPE *)out_addr);
#endif // VEC_SIZE == 1
    }
}