/*
 * Copyright (c) 2018-2020 Arm Limited.
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

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER)
/** This function perform a select operation between two tensors when condition tensor has the same rank.
 *
 * @attention The data_type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention Leftover size in the X dimension should be given as preprocessor argument using -DVEC_SIZE_LEFTOVER=value: e.g. x_dimension % VEC_SIZE
 *
 * @param[in]  c_ptr                             Pointer to the source tensor. Supported data types: U8
 * @param[in]  c_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  c_step_x                          c_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  c_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  c_step_y                          c_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  c_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  c_step_z                          c_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  c_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  x_ptr                             Pointer to the source tensor. Supported data types: All
 * @param[in]  x_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  x_step_x                          x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  x_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  x_step_y                          x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  x_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  x_step_z                          x_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  x_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  y_ptr                             Pointer to the source tensor. Supported data types: same as @p x_ptr
 * @param[in]  y_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  y_step_x                          y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  y_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  y_step_y                          y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  y_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  y_step_z                          y_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  y_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: same as @p x_ptr
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void select_same_rank(
    TENSOR3D_DECLARATION(c),
    TENSOR3D_DECLARATION(x),
    TENSOR3D_DECLARATION(y),
    TENSOR3D_DECLARATION(out))
{
    // Get pointers
    uint     offset          = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    __global uchar *c_addr   = c_ptr + c_offset_first_element_in_bytes + offset + get_global_id(1) * c_step_y + get_global_id(2) * c_step_z;
    __global uchar *x_addr   = x_ptr + x_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * x_step_y + get_global_id(2) * x_step_z;
    __global uchar *y_addr   = y_ptr + y_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * y_step_y + get_global_id(2) * y_step_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * out_step_y + get_global_id(2) * out_step_z;

    // Load values
    SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_c = CONVERT(VLOAD(VEC_SIZE)(0, c_addr), SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_x = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)x_addr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_y = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)y_addr);

    // Calculate result
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = select(in_y, in_x, in_c > (SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))0);

    // Boundary-aware store
    STORE_VECTOR_SELECT(res, DATA_TYPE, (__global DATA_TYPE *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}

/** This function perform a select operation between two tensors when condition tensor has a different rank.
 *
 * @attention The data_type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention Leftover size in the X dimension should be given as preprocessor argument using -DVEC_SIZE_LEFTOVER=value: e.g. x_dimension % VEC_SIZE
 *
 * @param[in]  c_ptr                             Pointer to the source tensor. Supported data types: U8
 * @param[in]  c_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  c_step_x                          c_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  c_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  x_ptr                             Pointer to the source tensor. Supported data types: All
 * @param[in]  x_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  x_step_x                          x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  x_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  x_step_y                          x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  x_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  x_step_z                          x_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  x_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  y_ptr                             Pointer to the source tensor. Supported data types: same as @p x_ptr
 * @param[in]  y_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  y_step_x                          y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  y_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  y_step_y                          y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  y_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  y_step_z                          y_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  y_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: same as @p x_ptr
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void select_different_rank_2(
    VECTOR_DECLARATION(c),
    TENSOR3D_DECLARATION(x),
    TENSOR3D_DECLARATION(y),
    TENSOR3D_DECLARATION(out))
{
    const int c_idx = get_global_id(1);

    // Get pointers
    uint     offset          = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    __global uchar *c_addr   = c_ptr + c_offset_first_element_in_bytes;
    __global uchar *x_addr   = x_ptr + x_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * x_step_y + get_global_id(2) * x_step_z;
    __global uchar *y_addr   = y_ptr + y_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * y_step_y + get_global_id(2) * y_step_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * out_step_y + get_global_id(2) * out_step_z;

    // Load values
    SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_c = *((__global uchar *)(c_addr + c_idx * c_stride_x));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_x = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)x_addr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_y = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)y_addr);

    // Calculate result
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = select(in_y, in_x, in_c > (SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))0);

    // Boundary-aware store
    STORE_VECTOR_SELECT(res, DATA_TYPE, (__global DATA_TYPE *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(DATA_TYPE) && defined(VEC_SIZE) && defined(VEC_SIZE_LEFTOVER) */

#if defined(DATA_TYPE) && defined(VEC_SIZE) && defined(DEPTH_SIZE) && defined(VEC_SIZE_LEFTOVER)
/** This function perform a select operation between two tensors when condition tensor has a different rank.
 *
 * @attention The data_type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @attention Leftover size in the X dimension should be given as preprocessor argument using -DVEC_SIZE_LEFTOVER=value: e.g. x_dimension % VEC_SIZE
 *
 * @param[in]  c_ptr                             Pointer to the source tensor. Supported data types: U8
 * @param[in]  c_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  c_step_x                          c_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  c_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  x_ptr                             Pointer to the source tensor. Supported data types: All
 * @param[in]  x_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  x_step_x                          x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  x_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  x_step_y                          x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  x_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  x_step_z                          x_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  x_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  y_ptr                             Pointer to the source tensor. Supported data types: same as @p x_ptr
 * @param[in]  y_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  y_step_x                          y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  y_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  y_step_y                          y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  y_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  y_step_z                          y_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  y_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: same as @p x_ptr
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        out_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        out_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        out_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void select_different_rank_n(
    VECTOR_DECLARATION(c),
    TENSOR3D_DECLARATION(x),
    TENSOR3D_DECLARATION(y),
    TENSOR3D_DECLARATION(out))
{
    const int c_idx = get_global_id(2) / DEPTH_SIZE;

    // Get pointers
    uint     offset          = max((int)(get_global_id(0) * VEC_SIZE - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE), 0);
    __global uchar *c_addr   = c_ptr + c_offset_first_element_in_bytes;
    __global uchar *x_addr   = x_ptr + x_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * x_step_y + get_global_id(2) * x_step_z;
    __global uchar *y_addr   = y_ptr + y_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * y_step_y + get_global_id(2) * y_step_z;
    __global uchar *out_addr = out_ptr + out_offset_first_element_in_bytes + offset * sizeof(DATA_TYPE) + get_global_id(1) * out_step_y + get_global_id(2) * out_step_z;

    // Load values
    SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_c = *((__global uchar *)(c_addr + c_idx * c_stride_x));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_x = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)x_addr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_y = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)y_addr);

    // Calculate result
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    res0 = select(in_y, in_x, in_c > (SELECT_VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE))0);

    // Boundary-aware store
    STORE_VECTOR_SELECT(res, DATA_TYPE, (__global DATA_TYPE *)out_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0);
}
#endif /* defined(DATA_TYPE) && defined(VEC_SIZE) && defined(DEPTH_SIZE) && defined(VEC_SIZE_LEFTOVER) */