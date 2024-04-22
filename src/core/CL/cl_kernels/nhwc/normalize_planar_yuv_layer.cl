/*
 * Copyright (c) 2018-2021 Arm Limited.
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

#if defined(DATA_TYPE) && defined(VEC_SIZE)

#define TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

/** Apply normalize_planar_yuv layer on tensors with NHWC data layout.
 *
 * @note Data type should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE e.g. -DVEC_SIZE=8
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 *
 * @param[in]  src_ptr                            Pointer to the first source tensor. Supported data types: F16/F32
 * @param[in]  src_stride_x                       Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  src_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                       Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  src_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                       Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes  The offset of the first element in the first source tensor
 * @param[out] dst_ptr                            Pointer to the destination tensor. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                       Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  dst_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                       Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  dst_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                       Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes  The offset of the first element in the destination tensor
 * @param[in]  mean_ptr                           Pointer to the mean source tensor. Supported data types: same as @p src_ptr
 * @param[in]  mean_stride_x                      Stride of the mean source tensor in X dimension (in bytes)
 * @param[in]  mean_step_x                        mean_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  mean_offset_first_element_in_bytes The offset of the first element in the mean source tensor
 * @param[in]  std_ptr                            Pointer to the std tensor. Supported data types: same as @p src_ptr
 * @param[in]  std_stride_x                       Stride of the std tensor in X dimension (in bytes)
 * @param[in]  std_step_x                         std_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  std_offset_first_element_in_bytes  The offset of the first element in the var source tensor
 */
__kernel void normalize_planar_yuv_layer_nhwc(TENSOR3D_DECLARATION(src),
                                              TENSOR3D_DECLARATION(dst),
                                              VECTOR_DECLARATION(mean),
                                              VECTOR_DECLARATION(std))
{
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE * sizeof(DATA_TYPE) - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE * sizeof(DATA_TYPE)), 0);

    __global uchar *src_addr  = src_ptr + src_offset_first_element_in_bytes + x_offs + get_global_id(1) * src_stride_y + get_global_id(2) * src_stride_z;
    __global uchar *dst_addr  = dst_ptr + dst_offset_first_element_in_bytes + x_offs + get_global_id(1) * dst_stride_y + get_global_id(2) * dst_stride_z;
    __global uchar *mean_addr = mean_ptr + mean_offset_first_element_in_bytes + x_offs;
    __global uchar *std_addr  = std_ptr + std_offset_first_element_in_bytes + x_offs;

    const TYPE curr_mean = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)mean_addr);
    const TYPE curr_std  = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)std_addr);

    TYPE data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)src_addr);
    TYPE res0 = (data - curr_mean) / curr_std;

    STORE_VECTOR_SELECT(res, DATA_TYPE, dst_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif // defined(DATA_TYPE) && defined(VEC_SIZE)