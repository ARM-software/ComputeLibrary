/*
 * Copyright (c) 2019-2020 Arm Limited.
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

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(INTERNAL_DATA_TYPE) && defined(GAMMA) && defined(BETA) && defined(EPSILON) && defined(DIM_X) && defined(DIM_Y) && defined(DIM_Z)
/** This function normalizes the input 2D tensor across the first dimension with respect to mean and standard deviation of the same dimension.
 *
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention Data type should be passed using the -DDATA_TYPE=data_type compile flag, e.g. -DDATA_TYPE=float
 * @attention The scale scalar value applied to the normalized tensor should be passed using the -DGAMMA=value compile flag, e.g. -DGAMMA=1.3
 * @attention The offset scalar value applied to the normalized tensor should be passed using the -DBETA=value compile flag, e.g. -DBETA=2.4
 * @attention Normalization epsilon parameter should be given as a preprocessor argument with -DEPSILON=value. e.g. -DEPSILON=0.001f
 * @attention Dimensions X, Y, and Z should be given as a preprocessor argument with -DDIM_X=value, -DDIM_Y=value, -DDIM_Z=value. e.g. -DDIM_X=6, -DDIM_Y=2, -DDIM_Z=7
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           (Optional) Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      (Optional) Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      (Optional) Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 */
__kernel void instance_normalization(
    TENSOR4D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR4D_DECLARATION(output)
#endif /* IN_PLACE */
)
{
    Tensor4D in = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(input, 0);
#ifndef IN_PLACE
    Tensor4D out = CONVERT_TO_TENSOR4D_STRUCT_NO_STEP(output, 0);
#endif /* IN_PLACE */

    INTERNAL_DATA_TYPE sum    = 0.f;
    INTERNAL_DATA_TYPE sum_sq = 0.f;

#if defined(NHWC)

    const int ch             = get_global_id(0); // Current channel
    const int batch          = get_global_id(2); // Current batch
    const int elements_plane = DIM_Y * DIM_Z;

    for(int i_w = 0; i_w < DIM_Y; ++i_w)
    {
        for(int i_h = 0; i_h < DIM_Z; ++i_h)
        {
            INTERNAL_DATA_TYPE data = (INTERNAL_DATA_TYPE) * ((__global DATA_TYPE *)tensor4D_offset(&in, ch, i_w, i_h, batch));
            sum += data;
            sum_sq += data * data;
        }
    }

#else // !defined(NHWC)
    const int ch             = get_global_id(2) % DIM_Z; // Current channel
    const int batch          = get_global_id(2) / DIM_Z; // Current batch
    const int elements_plane = DIM_X * DIM_Y;

    VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE)
    part_sum = 0.f;
    VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE)
    part_sum_sq = 0.f;
    // Calculate partial sum
    for(int y = 0; y < DIM_Y; ++y)
    {
        int x = 0;
        for(; x <= (DIM_X - VEC_SIZE); x += VEC_SIZE)
        {
            // Load data
            VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE)
            data = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch)), VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE));
            part_sum += data;
            part_sum_sq += data * data;
        }
        // Left-overs loop
        for(; x < DIM_X; ++x)
        {
            INTERNAL_DATA_TYPE data = (INTERNAL_DATA_TYPE)(*((__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch)));
            part_sum.s0 += data;
            part_sum_sq.s0 += data * data;
        }
    }
    // Perform reduction
#if VEC_SIZE > 8
    part_sum.s01234567 += part_sum.s89abcdef;
    part_sum_sq.s01234567 += part_sum_sq.s89abcdef;
#endif // VEC_SIZE > 8
#if VEC_SIZE > 4
    part_sum.s0123 += part_sum.s4567;
    part_sum_sq.s0123 += part_sum_sq.s4567;
#endif // VEC_SIZE > 4
#if VEC_SIZE > 2
    part_sum.s01 += part_sum.s23;
    part_sum_sq.s01 += part_sum_sq.s23;
#endif // VEC_SIZE > 2
    part_sum.s0 += part_sum.s1;
    part_sum_sq.s0 += part_sum_sq.s1;

    sum    = (INTERNAL_DATA_TYPE)part_sum.s0;
    sum_sq = (INTERNAL_DATA_TYPE)part_sum_sq.s0;

#endif // defined(NHWC)

    const INTERNAL_DATA_TYPE mean   = (sum / elements_plane);
    const INTERNAL_DATA_TYPE var    = (sum_sq / elements_plane) - (mean * mean);
    const INTERNAL_DATA_TYPE multip = GAMMA / sqrt(var + EPSILON);

#if defined(NHWC)

    for(int i_w = 0; i_w < DIM_Y; ++i_w)
    {
        for(int i_h = 0; i_h < DIM_Z; ++i_h)
        {
            __global DATA_TYPE *input_address = (__global DATA_TYPE *)tensor4D_offset(&in, ch, i_w, i_h, batch);
#ifdef IN_PLACE
            __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
            __global DATA_TYPE *output_address = (__global DATA_TYPE *)tensor4D_offset(&out, ch, i_w, i_h, batch);
#endif /* IN_PLACE */
            *(output_address) = (*(input_address) - mean) * multip + (INTERNAL_DATA_TYPE)BETA;
        }
    }

#else // !defined(NHWC)
    for(int y = 0; y < DIM_Y; ++y)
    {
        int x = 0;
        for(; x <= (DIM_X - VEC_SIZE); x += VEC_SIZE)
        {
            __global DATA_TYPE *input_address  = (__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch);
#ifdef IN_PLACE
            __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
            __global DATA_TYPE *output_address = (__global DATA_TYPE *)tensor4D_offset(&out, x, y, ch, batch);
#endif /* IN_PLACE */

            VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE)
            data = CONVERT(VLOAD(VEC_SIZE)(0, input_address), VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE));

            VEC_DATA_TYPE(INTERNAL_DATA_TYPE, VEC_SIZE)
            res = (data - mean) * multip + (INTERNAL_DATA_TYPE)BETA;
            VSTORE(VEC_SIZE)
            (CONVERT(res, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)), 0, output_address);
        }
        // Left-overs loop
        for(; x < DIM_X; ++x)
        {
            __global DATA_TYPE *input_address  = (__global DATA_TYPE *)tensor4D_offset(&in, x, y, ch, batch);
#ifdef IN_PLACE
            __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
            __global DATA_TYPE *output_address = (__global DATA_TYPE *)tensor4D_offset(&out, x, y, ch, batch);
#endif /* IN_PLACE */
            *(output_address)                  = (*(input_address) - mean) * multip + (INTERNAL_DATA_TYPE)BETA;
        }
    }
#endif // defined(NHWC)
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(INTERNAL_DATA_TYPE) && defined(GAMMA) && defined(BETA) && defined(EPSILON) && defined(DIM_X) && defined(DIM_Y) && defined(DIM_Z) */
