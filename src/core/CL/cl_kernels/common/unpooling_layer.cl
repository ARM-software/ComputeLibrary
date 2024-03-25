/*
 * Copyright (c) 2020-2021 Arm Limited.
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

/** Performs max unpooling function with pool size equal to 2.
 *
 * @note Datatype must be passed using -DDATA_TYPE e.g. -DDATA_TYPE=float. Supported data types are F16/F32
 * @note The width of the output tensor must be passed using -DWIDTH_DST e.g. -DWIDTH_DST=24
 * @note The height of the output tensor must be passed using -DHEIGHT_DST e.g. -DHEIGHT_DST=54
 * @note The depth of the output tensor must be passed using -DDEPTH_DST e.g. -DDEPTH_DST=32
 *
 * @param[in]  input_ptr                             Pointer to the source tensor. Supported data types: F16/F32
 * @param[in]  input_stride_x                        Stride of the source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source tensor
 * @param[out] output_ptr                            Pointer to the output tensor. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the output tensor in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the output tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the output tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the output tensor
 * @param[in]  indices_ptr                           Pointer to the indices tensor. Supported data types: U32
 * @param[in]  indices_stride_x                      Stride of the indices tensor in X dimension (in bytes)
 * @param[in]  indices_step_x                        indices_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  indices_stride_y                      Stride of the indices tensor in Y dimension (in bytes)
 * @param[in]  indices_step_y                        indices_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  indices_stride_z                      Stride of the indices tensor in Z dimension (in bytes)
 * @param[in]  indices_step_z                        indices_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  indices_offset_first_element_in_bytes The offset of the first element in the indices tensor
 */
__kernel void max_unpooling_layer_2(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output),
    TENSOR3D_DECLARATION(indices))
{
    Tensor3D input   = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output  = CONVERT_TO_TENSOR3D_STRUCT_NO_UPDATE_PTR(output);
    Tensor3D indices = CONVERT_TO_TENSOR3D_STRUCT(indices);

    unsigned int index = *((__global unsigned int *)indices.ptr);
    DATA_TYPE value    = *((__global DATA_TYPE *)input.ptr);

    *((__global DATA_TYPE *)tensor3D_index2ptr(&output, WIDTH_DST, HEIGHT_DST, DEPTH_DST, index)) = value;
}