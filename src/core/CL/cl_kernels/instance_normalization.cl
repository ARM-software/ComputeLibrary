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

#if defined(VEC_SIZE) && defined(DATA_TYPE) && defined(GAMMA) && defined(BETA) && defined(EPSILON) && defined(DIM_X) && defined(DIM_Y) && defined(DIM_Z)
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
    float sum    = 0.f;
    float sum_sq = 0.f;

#if defined(NHWC)

    const int pc             = get_global_id(0);
    const int pn             = get_global_id(2);
    const int elements_plane = DIM_Y * DIM_Z;
    const int elements_x_y   = DIM_X * DIM_Y;
    const int elements_x_y_z = DIM_X * DIM_Y * DIM_Z;

    for(int i_w = 0; i_w < DIM_Y; ++i_w)
    {
        for(int i_h = 0; i_h < DIM_Z; ++i_h)
        {
            float data = (float)*((__global DATA_TYPE *)input_ptr + pc + i_w * DIM_X + i_h * elements_x_y + pn * elements_x_y_z);
            sum += data;
            sum_sq += data * data;
        }
    }

#else // !defined(NHWC)
    const int elements_plane = DIM_X * DIM_Y;
    const int plane_address  = get_global_id(2) * elements_plane;
    int       i              = 0;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    part_sum = 0.f;
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    part_sum_sq = 0.f;
    // Calculate partial sum
    for(; i <= (elements_plane - VEC_SIZE); i += VEC_SIZE)
    {
        // Load data
        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_ptr + i + plane_address);
        part_sum += data;
        part_sum_sq += data * data;
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
    // Left-overs loop
    for(; i < elements_plane; ++i)
    {
        DATA_TYPE data = *((__global DATA_TYPE *)input_ptr + i + plane_address);
        part_sum.s0 += data;
        part_sum_sq.s0 += data * data;
    }

    sum    = (float)part_sum.s0;
    sum_sq = (float)part_sum_sq.s0;

#endif // defined(NHWC)

    const float mean_float   = (sum / elements_plane);
    const DATA_TYPE mean         = (DATA_TYPE)mean_float;
    const float     var_float    = (sum_sq / elements_plane) - (mean_float * mean_float);
    const float     multip_float = GAMMA / sqrt(var_float + EPSILON);
    const DATA_TYPE multip       = (DATA_TYPE)multip_float;

#if defined(NHWC)

    for(int i_w = 0; i_w < DIM_Y; ++i_w)
    {
        for(int i_h = 0; i_h < DIM_Z; ++i_h)
        {
            __global DATA_TYPE *input_address = (__global DATA_TYPE *)input_ptr + pc + i_w * DIM_X + i_h * elements_x_y + pn * elements_x_y_z;
#ifdef IN_PLACE
            __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
            __global DATA_TYPE *output_address = (__global DATA_TYPE *)output_ptr + pc + i_w * DIM_X + i_h * elements_x_y + pn * elements_x_y_z;
#endif /* IN_PLACE */
            *(output_address) = (*(input_address) - mean) * multip + (DATA_TYPE)BETA;
        }
    }

#else // !defined(NHWC)
    i      = 0;
    for(; i <= (elements_plane - VEC_SIZE); i += VEC_SIZE)
    {
        __global DATA_TYPE *input_address  = (__global DATA_TYPE *)input_ptr + i + plane_address;
#ifdef IN_PLACE
        __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
        __global DATA_TYPE *output_address = (__global DATA_TYPE *)output_ptr + i + plane_address;
#endif /* IN_PLACE */

        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_address);

        VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
        res = (data - mean) * multip + (DATA_TYPE)BETA;
        VSTORE(VEC_SIZE)
        (res, 0, (__global DATA_TYPE *)output_address);
    }
    for(; i < elements_plane; ++i)
    {
        __global DATA_TYPE *input_address  = (__global DATA_TYPE *)input_ptr + i + plane_address;
#ifdef IN_PLACE
        __global DATA_TYPE *output_address = input_address;
#else  /* !IN_PLACE */
        __global DATA_TYPE *output_address = (__global DATA_TYPE *)output_ptr + i + plane_address;
#endif /* IN_PLACE */
        *(output_address)                  = (*(input_address) - mean) * multip + (DATA_TYPE)BETA;
    }
#endif // defined(NHWC)
}
#endif /* defined(VEC_SIZE) && defined(DATA_TYPE) && defined(GAMMA) && defined(BETA) && defined(EPSILON) && defined(DIM_X) && defined(DIM_Y) && defined(DIM_Z) */
