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

#if defined(DATA_TYPE) && defined(SELECT_DATA_TYPE) && defined(VEC_SIZE)
/** This function perform a select operation between two tensors when condition tensor has the same rank.
 *
 * @attention The data_type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=uchar
 * @attention The select operation data_type need to be passed at compile time using -DSELECT_DATA_TYPE: e.g. -DSELECT_DATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 *
 * @param[in]  c_ptr                             Pointer to the source tensor. Supported data types: U8
 * @param[in]  c_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  c_step_x                          c_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  c_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  c_step_y                          c_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  c_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  c_step_z                          c_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  c_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  x_ptr                             Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  x_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  x_step_x                          x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  x_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  x_step_y                          x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  x_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  x_step_z                          x_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  x_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  y_ptr                             Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  y_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  y_step_x                          y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  y_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  y_step_y                          y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  y_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  y_step_z                          y_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  y_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
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
    // Get pixels pointer
    Tensor3D c_t   = CONVERT_TO_TENSOR3D_STRUCT(c);
    Tensor3D x_t   = CONVERT_TO_TENSOR3D_STRUCT(x);
    Tensor3D y_t   = CONVERT_TO_TENSOR3D_STRUCT(y);
    Tensor3D out_t = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load values
    VEC_DATA_TYPE(SELECT_DATA_TYPE, VEC_SIZE)
    in_c = CONVERT((VLOAD(VEC_SIZE)(0, (__global uchar *)c_t.ptr)), VEC_DATA_TYPE(SELECT_DATA_TYPE, VEC_SIZE));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_x = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)x_t.ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_y = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)y_t.ptr);

    // Calculate and store result
    VSTORE(VEC_SIZE)
    (select(in_y, in_x, in_c > (SELECT_DATA_TYPE)0), 0, (__global DATA_TYPE *)out_t.ptr);
}

/** This function perform a select operation between two tensors when condition tensor has a different rank.
 *
 * @attention The data_type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=uchar
 * @attention The select operation data_type need to be passed at compile time using -DSELECT_DATA_TYPE: e.g. -DSELECT_DATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 *
 * @param[in]  c_ptr                             Pointer to the source tensor. Supported data types: U8
 * @param[in]  c_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  c_step_x                          c_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  c_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  x_ptr                             Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  x_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  x_step_x                          x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  x_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  x_step_y                          x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  x_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  x_step_z                          x_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  x_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  y_ptr                             Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  y_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  y_step_x                          y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  y_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  y_step_y                          y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  y_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  y_step_z                          y_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  y_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
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

    // Get pixels pointer
    Vector   c_t   = CONVERT_TO_VECTOR_STRUCT_NO_STEP(c);
    Tensor3D x_t   = CONVERT_TO_TENSOR3D_STRUCT(x);
    Tensor3D y_t   = CONVERT_TO_TENSOR3D_STRUCT(y);
    Tensor3D out_t = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load values
    VEC_DATA_TYPE(SELECT_DATA_TYPE, VEC_SIZE)
    in_c = *((__global uchar *)(c_t.ptr + c_idx * c_t.stride_x));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_x = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)x_t.ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_y = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)y_t.ptr);

    // Calculate and store result
    VSTORE(VEC_SIZE)
    (select(in_y, in_x, in_c > (SELECT_DATA_TYPE)0), 0, (__global DATA_TYPE *)out_t.ptr);
}
#endif /* defined(DATA_TYPE) && defined(SELECT_DATA_TYPE) && defined(VEC_SIZE) */

#if defined(DATA_TYPE) && defined(SELECT_DATA_TYPE) && defined(VEC_SIZE) && defined(DEPTH_SIZE)
/** This function perform a select operation between two tensors when condition tensor has a different rank.
 *
 * @attention The data_type need to be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=uchar
 * @attention The select operation data_type need to be passed at compile time using -DSELECT_DATA_TYPE: e.g. -DSELECT_DATA_TYPE=uchar
 * @attention Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 *
 * @param[in]  c_ptr                             Pointer to the source tensor. Supported data types: U8
 * @param[in]  c_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  c_step_x                          c_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  c_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  x_ptr                             Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  x_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  x_step_x                          x_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  x_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  x_step_y                          x_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  x_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  x_step_z                          x_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  x_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[in]  y_ptr                             Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
 * @param[in]  y_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  y_step_x                          y_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  y_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  y_step_y                          y_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  y_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  y_step_z                          y_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  y_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: U8/S8/QASYMM8/U16/S16/U32/S32/F16/F32
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

    // Get pixels pointer
    Vector   c_t   = CONVERT_TO_VECTOR_STRUCT_NO_STEP(c);
    Tensor3D x_t   = CONVERT_TO_TENSOR3D_STRUCT(x);
    Tensor3D y_t   = CONVERT_TO_TENSOR3D_STRUCT(y);
    Tensor3D out_t = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load values
    VEC_DATA_TYPE(SELECT_DATA_TYPE, VEC_SIZE)
    in_c = *((__global uchar *)(c_t.ptr + c_idx * c_t.stride_x));
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_x = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)x_t.ptr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    in_y = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)y_t.ptr);

    // Calculate and store result
    VSTORE(VEC_SIZE)
    (select(in_y, in_x, in_c > (SELECT_DATA_TYPE)0), 0, (__global DATA_TYPE *)out_t.ptr);
}
#endif /* defined(DATA_TYPE) && defined(SELECT_DATA_TYPE) && defined(VEC_SIZE) && defined(DEPTH_SIZE) */