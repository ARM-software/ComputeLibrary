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

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

/** This function calculates the sum and sum of squares of a given input image.
 *
 * @note To enable calculation sum of squares -DSTDDEV should be passed as a preprocessor argument.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: U8
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[in]  height                            Height of the input image
 * @param[out] global_sum                        Global sum of all elements
 * @param[out] global_sum_sq                     Global sum of squares of all elements
 */
__kernel void mean_stddev_accumulate(
    IMAGE_DECLARATION(src),
    uint     height,
    __global ulong *global_sum
#ifdef STDDEV
    ,
    __global ulong *global_sum_sq
#endif /* STDDEV */
)
{
    // Get pixels pointer
    Image src = CONVERT_TO_IMAGE_STRUCT(src);

    uint8 tmp_sum = 0;
#ifdef STDDEV
    uint8 tmp_sum_sq = 0;
#endif /* STDDEV */
    // Calculate partial sum
    for(int i = 0; i < height; i++)
    {
        // Load data
        uint8 data = convert_uint8(vload8(0, offset(&src, 0, i)));

        tmp_sum += data;
#ifdef STDDEV
        tmp_sum_sq += data * data;
#endif /* STDDEV */
    }
    // Perform reduction
    tmp_sum.s0123 += tmp_sum.s4567;
    tmp_sum.s01 += tmp_sum.s23;
    atom_add(global_sum, tmp_sum.s0 + tmp_sum.s1);

#ifdef STDDEV
    tmp_sum_sq.s0123 += tmp_sum_sq.s4567;
    tmp_sum_sq.s01 += tmp_sum_sq.s23;
    atom_add(global_sum_sq, tmp_sum_sq.s0 + tmp_sum_sq.s1);
#endif /* STDDEV */
}

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : disable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : disable
