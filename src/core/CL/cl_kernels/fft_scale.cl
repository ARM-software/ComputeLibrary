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

/** Computes the fft scale stage
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           (Optional) Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      (Optional) Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        (Optional) dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      (Optional) Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        (Optional) dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        (Optional) dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes (Optional) The offset of the first element in the destination tensor
 * @param[in]  scale                             Scale to apply to the complex value
 */
__kernel void fft_scale_conj(
    TENSOR3D_DECLARATION(src)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(dst)
#endif /* not IN_PLACE */
    ,
    float scale)
{
    // Get tensor pointers
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
#if defined(IN_PLACE)
    Tensor3D dst = src;
#else  /* IN_PLACE */
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
#endif /* IN_PLACE */

    // Store result
#if VEC_SIZE == 1
    *((__global float *)dst.ptr) = (*(__global float *)src.ptr) / scale;
#elif VEC_SIZE == 2
    // Load data
    float2 data = vload2(0, (__global float *)src.ptr);
    data /= scale;
#if defined(CONJ)
    vstore2((float2)(data.s0, -data.s1), 0, (__global float *)dst.ptr);
#else  // defined(CONJ)
    vstore2(data, 0, (__global float *)dst.ptr);
#endif // defined(CONJ)
#else  // VEC_SIZE == 1
#error "vec_size of 1 and 2 are supported"
#endif // VEC_SIZE == 1
}