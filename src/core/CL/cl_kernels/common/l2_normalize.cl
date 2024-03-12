/*
 * Copyright (c) 2016-2021 Arm Limited.
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

#if defined(VEC_SIZE_X) && defined(VEC_SIZE_LEFTOVER_X)
/** This kernel performs l2 normalization on x-axis
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE_X=size. e.g. -DVEC_SIZE_X=16
 * @note The leftover size in the X dimension shoud be given as preprocessor argument using -DVEC_SIZE_LEFTOVER_X is; x_dimension % VEC_SIZE_X. e.g. -DVEC_SIZE_LEFTOVER_X=1
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in]  sum_ptr                              Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                           sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                           sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                              Epsilon value
 */
__kernel void l2_normalize_x(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(sum),
    IMAGE_DECLARATION(output),
    DATA_TYPE epsilon)
{
    // Offset computation
    const uint x_offs = max((int)(get_global_id(0) * VEC_SIZE_X - (VEC_SIZE_X - VEC_SIZE_LEFTOVER_X) % VEC_SIZE_X), 0);

    // Address computation
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y;
    __global uchar *sum_addr    = sum_ptr + sum_offset_first_element_in_bytes + get_global_id(1) * sum_stride_y;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * output_stride_y;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    in = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)input_addr);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    normalize_value = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X))rsqrt(fmax(*((__global DATA_TYPE *)sum_addr), epsilon));

    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    data0 = in * normalize_value;

    STORE_VECTOR_SELECT(data, DATA_TYPE, output_addr, VEC_SIZE_X, VEC_SIZE_LEFTOVER_X, VEC_SIZE_LEFTOVER_X != 0 && get_global_id(0) == 0);
}

/** This kernel performs l2 normalization on y-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE_X=size. e.g. -DVEC_SIZE_X=16
 * @note The leftover size in the X dimension shoud be given as preprocessor argument using -DVEC_SIZE_LEFTOVER_X is; x_dimension % VEC_SIZE_X. e.g. -DVEC_SIZE_LEFTOVER_X=1
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in]  sum_ptr                              Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                           sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                           sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                              Epsilon value
 */
__kernel void l2_normalize_y(
    IMAGE_DECLARATION(input),
    IMAGE_DECLARATION(sum),
    IMAGE_DECLARATION(output),
    DATA_TYPE epsilon)
{
    // Offset computation
    const uint x_offs = max((int)(get_global_id(0) * VEC_SIZE_X - (VEC_SIZE_X - VEC_SIZE_LEFTOVER_X) % VEC_SIZE_X), 0);

    // Address computation
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y;
    __global uchar *sum_addr    = sum_ptr + sum_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE);
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * output_stride_y;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    in = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)input_addr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    sums = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)sum_addr);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    normalize_value = (VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X))rsqrt(fmax(sums, epsilon));

    const VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    data0 = in * normalize_value;

    STORE_VECTOR_SELECT(data, DATA_TYPE, output_addr, VEC_SIZE_X, VEC_SIZE_LEFTOVER_X, VEC_SIZE_LEFTOVER_X != 0 && get_global_id(0) == 0);
}

/** This kernel performs l2 normalization on z-axis.
 *
 * @note The data type must be passed at compile time using -DDATA_TYPE: e.g. -DDATA_TYPE=float
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE_X=size. e.g. -DVEC_SIZE_X=16
 * @note The leftover size in the X dimension shoud be given as preprocessor argument using -DVEC_SIZE_LEFTOVER_X is; x_dimension % VEC_SIZE_X. e.g. -DVEC_SIZE_LEFTOVER_X=1
 *
 * @param[in]  input_ptr                            Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[in]  sum_ptr                              Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  sum_stride_x                         Stride of the source tensor in X dimension (in bytes)
 * @param[in]  sum_step_x                           sum_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  sum_stride_y                         Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  sum_step_y                           sum_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  sum_stride_z                         Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  sum_step_z                           sum_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  sum_offset_first_element_in_bytes    The offset of the first element in the source tensor
 * @param[out] output_ptr                           Pointer to the destination tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination tensor
 * @param[in]  epsilon                              Epsilon value
 */
__kernel void l2_normalize_z(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(sum),
    TENSOR3D_DECLARATION(output),
    DATA_TYPE epsilon)
{
    // Offset computation
    const uint x_offs = max((int)(get_global_id(0) * VEC_SIZE_X - (VEC_SIZE_X - VEC_SIZE_LEFTOVER_X) % VEC_SIZE_X), 0);

    // Address computation
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * input_stride_y + get_global_id(2) * input_stride_z;
    __global uchar *sum_addr    = sum_ptr + sum_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * sum_stride_y;
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs * sizeof(DATA_TYPE) + get_global_id(1) * output_stride_y + get_global_id(2) * output_stride_z;

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    in = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)input_addr);
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    sums = VLOAD(VEC_SIZE_X)(0, (__global DATA_TYPE *)sum_addr);

    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X)
    data0 = in * ((VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE_X))(rsqrt(fmax(sums, epsilon))));

    STORE_VECTOR_SELECT(data, DATA_TYPE, output_addr, VEC_SIZE_X, VEC_SIZE_LEFTOVER_X, VEC_SIZE_LEFTOVER_X != 0 && get_global_id(0) == 0);
}
#endif // defined(VEC_SIZE_X) && defined(VEC_SIZE_LEFTOVER_X)