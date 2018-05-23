/*
 * Copyright (c) 2018 ARM Limited.
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

/** Performs a copy of input tensor to the output tensor.
 *
 * @param[in]  in_ptr                            Pointer to the source image. Supported data types: U8.
 * @param[in]  in_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  in_step_x                         in_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  in_offset_first_element_in_bytes  Offset of the first element in the source image
 * @param[out] out_ptr                           Pointer to the destination image. Supported data types: U8.
 * @param[in]  out_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per work item (in bytes)
 * @param[in]  out_offset_first_element_in_bytes Offset of the first element in the destination image
 */
__kernel void copy_tensor(
    VECTOR_DECLARATION(in),
    VECTOR_DECLARATION(out))
{
    Vector in  = CONVERT_TO_VECTOR_STRUCT(in);
    Vector out = CONVERT_TO_VECTOR_STRUCT(out);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    data = vload16(0, (__global DATA_TYPE *)in.ptr);

    vstore16(data, 0, (__global DATA_TYPE *)out.ptr);
}