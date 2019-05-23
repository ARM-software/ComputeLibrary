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

#if defined(VEC_SIZE)
/** Computes the digit reverse stage on axis X
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  idx_ptr                           Pointer to the index tensor. Supported data types: U32
 * @param[in]  idx_stride_x                      Stride of the index tensor in X dimension (in bytes)
 * @param[in]  idx_step_x                        idx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  idx_offset_first_element_in_bytes The offset of the first element in the index tensor
 */
__kernel void fft_digit_reverse_axis_0(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    VECTOR_DECLARATION(idx))
{
    // Get tensor pointers
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
    Vector   idx = CONVERT_TO_VECTOR_STRUCT(idx);

    const unsigned int iidx = *((__global uint *)(idx.ptr));

    // Load data
#if VEC_SIZE == 1
    float data = *((__global float *)tensor3D_offset(&src, iidx, get_global_id(1), get_global_id(2)));
#elif VEC_SIZE == 2
    float2 data = vload2(0, (__global float *)tensor3D_offset(&src, iidx, get_global_id(1), get_global_id(2)));
#else // VEC_SIZE == 1
#error "vec_size of 1 and 2 are supported"
#endif // VEC_SIZE == 1

    // Create result
#if VEC_SIZE == 1
    float2 res = { data, 0 };
#elif VEC_SIZE == 2
    float2 res  = data;
#else // VEC_SIZE == 1
#error "vec_size of 1 and 2 are supported"
#endif // VEC_SIZE == 1

    // Store result
#if defined(CONJ)
    vstore2((float2)(res.s0, -res.s1), 0, (__global float *)dst.ptr);
#else  // defined(CONJ)
    vstore2(res, 0, (__global float *)dst.ptr);
#endif // defined(CONJ)
}

/** Computes the digit reverse stage on axis Y
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  idx_ptr                           Pointer to the index tensor. Supported data types: U32
 * @param[in]  idx_stride_x                      Stride of the index tensor in X dimension (in bytes)
 * @param[in]  idx_step_x                        idx_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  idx_offset_first_element_in_bytes The offset of the first element in the index tensor
 */
__kernel void fft_digit_reverse_axis_1(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(dst),
    VECTOR_DECLARATION(idx))
{
    // Get tensor pointers
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_NO_STEP(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);
    Vector   idx = CONVERT_TO_VECTOR_STRUCT_NO_STEP(idx);

    const unsigned int iidx = *((__global uint *)vector_offset(&idx, (int)(get_global_id(1))));

    // Load data
#if VEC_SIZE == 1
    float data = *((__global float *)tensor3D_offset(&src, get_global_id(0), iidx, get_global_id(2)));
#elif VEC_SIZE == 2
    float2 data = vload2(0, (__global float *)tensor3D_offset(&src, get_global_id(0), iidx, get_global_id(2)));
#else // VEC_SIZE == 1
#error "vec_size of 1 and 2 are supported"
#endif // VEC_SIZE == 1

    // Create result
#if VEC_SIZE == 1
    float2 res = { data, 0 };
#elif VEC_SIZE == 2
    float2 res  = data;
#else // VEC_SIZE == 1
#error "vec_size of 1 and 2 are supported"
#endif // VEC_SIZE == 1

    // Store result
#if defined(CONJ)
    vstore2((float2)(res.s0, -res.s1), 0, (__global float *)dst.ptr);
#else  // defined(CONJ)
    vstore2(res, 0, (__global float *)dst.ptr);
#endif // defined(CONJ)
}
#endif // defined(VEC_SIZE)