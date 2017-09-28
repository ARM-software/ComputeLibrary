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

/** Calculate square sum of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return square sum of vector.
 */
inline DATA_TYPE square_sum(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in *= in;

    in.s01234567 += in.s89ABCDEF;
    in.s0123 += in.s4567;
    in.s01 += in.s23;

    return (in.s0 + in.s1);
}

/** Calculate sum of a vector
 *
 * @param[in] input Pointer to the first pixel.
 *
 * @return sum of vector.
 */
inline DATA_TYPE sum(__global const DATA_TYPE *input)
{
    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, input);

    in.s01234567 += in.s89ABCDEF;
    in.s0123 += in.s4567;
    in.s01 += in.s23;

    return (in.s0 + in.s1);
}

/** This kernel performs reduction given an operation.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data size must be passed at compile time using -DDATA_SIZE e.g. -DDATA_SIZE=32
 * @note The operation we want to perform must be passed at compile time using -DOPERATION e.g. -DOPERATION=square_sum
 *
 * @param[in] src_ptr                                   Pointer to the source tensor. Supported data types: F32
 * @param[in] src_stride_x                              Stride of the source tensor in X dimension (in bytes)
 * @param[in] src_step_x                                src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] src_offset_first_element_in_bytes         The offset of the first element in the source tensor
 * @param[in] partial_sum_ptr                           The local buffer to hold sumed values. Supported data types: same as @p src_ptt
 * @param[in] partial_sum_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in] partial_sum_step_x                        partial_sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in] partial_sum_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in] local_sums                                Local buffer for storing the partioal sum
 */
__kernel void reduction_operation(
    VECTOR_DECLARATION(src),
    VECTOR_DECLARATION(partial_sum),
    __local DATA_TYPE *local_sums)
{
    Vector src         = CONVERT_TO_VECTOR_STRUCT(src);
    Vector partial_sum = CONVERT_TO_VECTOR_STRUCT(partial_sum);

    unsigned int lsize = get_local_size(0);
    unsigned int lid   = get_local_id(0);

    local_sums[lid] = OPERATION((__global DATA_TYPE *)src.ptr);
    barrier(CLK_LOCAL_MEM_FENCE);

    // Perform parallel reduction
    for(unsigned int i = lsize >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            local_sums[lid] += local_sums[lid + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(lid == 0)
    {
        ((__global DATA_TYPE *)partial_sum.ptr + get_group_id(0))[0] = local_sums[0];
    }
}