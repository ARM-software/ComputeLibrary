/*
 * Copyright (c) 2019 ARM Limited.
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

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(EPSILON) && defined(WIDTH)
/** This function normalizes the input 2D tensor across the first dimension with respect to mean and standard deviation of the same dimension.
 *
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention Data type should be passed using the -DDATA_TYPE compile flag, e.g. -DDATA_TYPE=float
 * @attention Width of the input tensor should be passed using the -DWIDTH compile flag, e.g. -DWIDTH=16
 * @attention Normalization epsilon parameter should be given as a preprocessor argument with -DEPSILON=value. e.g. -DEPSILON=0.001f
 *
 * @param[in]  input_ptr                            Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] output_ptr                           (Optional) Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      (Optional) Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 */
__kernel void mean_stddev_normalization(
    IMAGE_DECLARATION(input)
#ifndef IN_PLACE
    ,
    IMAGE_DECLARATION(output)
#endif /* IN_PLACE */
)
{
    // Get pixels pointer
    Image in = CONVERT_TO_IMAGE_STRUCT(input);
#ifdef IN_PLACE
    Image out = in;
#else  /* IN_PLACE */
    Image out = CONVERT_TO_IMAGE_STRUCT(output);
#endif /* IN_PLACE */

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    sum = 0.f;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    sum_sq = 0.f;
    // Calculate partial sum
    int i = 0;
    for(; i <= (WIDTH - VEC_SIZE); i += VEC_SIZE)
    {
        // Load data
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)offset(&in, i, 0));

        sum += data;
        sum_sq += data * data;
    }
    // Perform reduction
#if VEC_SIZE > 8
    sum.s01234567 += sum.s89abcdef;
    sum_sq.s01234567 += sum_sq.s89abcdef;
#endif // VEC_SIZE > 8
#if VEC_SIZE > 4
    sum.s0123 += sum.s4567;
    sum_sq.s0123 += sum_sq.s4567;
#endif // VEC_SIZE > 4
#if VEC_SIZE > 2
    sum.s01 += sum.s23;
    sum_sq.s01 += sum_sq.s23;
#endif // VEC_SIZE > 2
    sum.s0 += sum.s1;
    sum_sq.s0 += sum_sq.s1;
    // Left-overs loop
    for(; i < WIDTH; ++i)
    {
        DATA_TYPE data = *((__global DATA_TYPE *)offset(&in, i, 0));

        sum.s0 += data;
        sum_sq.s0 += data * data;
    }

    DATA_TYPE mean       = sum.s0 / WIDTH;
    DATA_TYPE var        = (sum_sq.s0 / WIDTH) - (mean * mean);
    DATA_TYPE stddev_inv = 1.f / sqrt(var + EPSILON);

    i = 0;
    for(; i <= (WIDTH - VEC_SIZE); i += VEC_SIZE)
    {
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)offset(&in, i, 0));

        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        res = (data - mean) * stddev_inv;
        VSTORE(VEC_SIZE)
        (res, 0, (__global DATA_TYPE *)offset(&out, i, 0));
    }
    for(; i < WIDTH; ++i)
    {
        DATA_TYPE data = *((__global DATA_TYPE *)offset(&in, i, 0));

        *((__global DATA_TYPE *)offset(&out, i, 0)) = (data - mean) * stddev_inv;
    }
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(EPSILON) && defined(WIDTH) */
