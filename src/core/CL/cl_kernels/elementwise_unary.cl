/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#if defined(DATA_TYPE) && defined(OPERATION)

// Calculate exponential
#define exp_op(input) exp(input)
// Calculate reverse square root
#define rsqrt_op(input) rsqrt(input)
// Calculate negative
#define neg_op(input) (-input)
// Calculate sine
#define sin_op(input) sin(input)
// Calculate abs for floating point values
#define fabs_op(input) fabs(input)
// Calculate natural_log
#define natural_log_op(input) log(input)
// Calculate round (Cannot use round function as it rounds halfway cases away from zero).
#if defined(VEC_SIZE)
#define round_op(input) CONVERT(CONVERT_SAT_ROUND(input, VEC_DATA_TYPE(int, VEC_SIZE), rte), VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))
#else // defined(VEC_SIZE
#define round_op(input) CONVERT(CONVERT_SAT_ROUND(input, int, rte), DATA_TYPE)
#endif // defined(VEC_SIZE

/** Applies element wise unary operator in a tensor.
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: F16/32.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes  Offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: F16/32.
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes Offset of the first element in the destination image
 */
__kernel void elementwise_unary(
    VECTOR_DECLARATION(in),
    VECTOR_DECLARATION(out))
{
    Vector in  = CONVERT_TO_VECTOR_STRUCT(in);
    Vector out = CONVERT_TO_VECTOR_STRUCT(out);

#if defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
    // Check if access on width gets out of bounds
    // If it does shift access vector to access elements within bounds
    const int xi = (int)(get_global_id(0) * VEC_SIZE);
    in.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * in_stride_x;
    out.ptr -= max(xi - (int)LAST_ACCESSED_X, 0) * out_stride_x;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);

    VSTORE(VEC_SIZE)
    (OPERATION(data), 0, (__global DATA_TYPE *)out.ptr);
#else  // !defined(VEC_SIZE) || !defined(LAST_ACCESSED_X)
    *((__global DATA_TYPE *)(out.ptr)) = (DATA_TYPE)(OPERATION(*((__global DATA_TYPE *)in.ptr)));
#endif // defined(VEC_SIZE) && defined(LAST_ACCESSED_X)
}
#endif // defined(DATA_TYPE) && defined(OPERATION)
