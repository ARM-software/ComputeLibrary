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
 * @param[in]  in_ptr                            Pointer to the source tensor. Supported data types: U8/S8/QASYMM8/U16/S16/F16/U32/S32/F32
 * @param[in]  in_stride_x                       Stride of the source tensor in X dimension (in bytes)
 * @param[in]  in_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  in_stride_y                       Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  in_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  in_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  in_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  in_offset_first_element_in_bytes  The offset of the first element in the source tensor
 * @param[out] out_ptr                           Pointer to the destination tensor. Supported data types: same as @p in_ptr
 * @param[in]  out_stride_x                      Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  out_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  out_stride_y                      Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  out_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  out_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  out_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  out_offset_first_element_in_bytes The offset of the first element in the destination tensor
 */
__kernel void copy_tensor(
    TENSOR3D_DECLARATION(in),
    TENSOR3D_DECLARATION(out))
{
    Tensor3D in  = CONVERT_TO_TENSOR3D_STRUCT(in);
    Tensor3D out = CONVERT_TO_TENSOR3D_STRUCT(out);

    // Load data
    VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)
    data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)in.ptr);

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)out.ptr);
}