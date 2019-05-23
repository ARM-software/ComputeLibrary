/*
 * Copyright (c) 2016-2019 ARM Limited.
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

/** This kernel performs l2 normalization on x-axis
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data size must be passed at compile time using -DDATA_SIZE e.g. -DDATA_SIZE=32
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_ptr                           Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                           Epsilon value
 */
__kernel void l2_normalize_x(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(sum),
    IMAGE_DECLARATION(dst),
    DATA_TYPE epsilon)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image sum = CONVERT_TO_IMAGE_STRUCT(sum);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, (__global DATA_TYPE *)src.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    normalize_value = (VEC_DATA_TYPE(DATA_TYPE, 16))rsqrt(fmax(((__global DATA_TYPE *)sum.ptr)[0], epsilon));

    vstore16(in * normalize_value, 0, (__global DATA_TYPE *)dst.ptr);
}

/** This kernel performs l2 normalization on y-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data size must be passed at compile time using -DDATA_SIZE e.g. -DDATA_SIZE=32
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_ptr                           Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                           Epsilon value
 */
__kernel void l2_normalize_y(
    IMAGE_DECLARATION(src),
    IMAGE_DECLARATION(sum),
    IMAGE_DECLARATION(dst),
    DATA_TYPE epsilon)
{
    Image src = CONVERT_TO_IMAGE_STRUCT(src);
    Image sum = CONVERT_TO_IMAGE_STRUCT(sum);
    Image dst = CONVERT_TO_IMAGE_STRUCT(dst);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, (__global DATA_TYPE *)src.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    sums = vload16(0, (__global DATA_TYPE *)sum.ptr);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    normalize_value = (VEC_DATA_TYPE(DATA_TYPE, 16))rsqrt(fmax(sums, epsilon));

    vstore16(in * normalize_value, 0, (__global DATA_TYPE *)dst.ptr);
}
/** This kernel performs l2 normalization on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note The data size must be passed at compile time using -DDATA_SIZE e.g. -DDATA_SIZE=32
 *
 * @param[in]  src_ptr                           Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[in]  sum_ptr                           Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                      Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                        sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                      Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                        sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                        sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes The offset of the first element in the source tensor
 * @param[out] dst_ptr                           Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                           Epsilon value
 */
__kernel void l2_normalize_z(
    TENSOR3D_DECLARATION(src),
    TENSOR3D_DECLARATION(sum),
    TENSOR3D_DECLARATION(dst),
    DATA_TYPE epsilon)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D sum = CONVERT_TO_TENSOR3D_STRUCT(sum);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    in = vload16(0, (__global DATA_TYPE *)src.ptr);
    VEC_DATA_TYPE(DATA_TYPE, 16)
    sums = vload16(0, (__global DATA_TYPE *)sum.ptr);

    VEC_DATA_TYPE(DATA_TYPE, 16)
    normalize_value = (VEC_DATA_TYPE(DATA_TYPE, 16))rsqrt(fmax(sums, epsilon));

    vstore16(in * normalize_value, 0, (__global DATA_TYPE *)dst.ptr);
}