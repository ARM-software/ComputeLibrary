/*
 * Copyright (c) 2017 ARM Limited.
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

/** This performs the dequantization of 8-bit unsigned integers to floating point.
 *
 * @param[in]  input_ptr                             Pointer to the source image. Supported data types: QS8/QS16/F16/F32
 * @param[in]  input_stride_x                        Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                          input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                        Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                          input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                        Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                          input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes   The offset of the first element in the source image
 * @param[out] output_ptr                            Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                       Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                         output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                       Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                         output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                         output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes  The offset of the first element in the destination image
 * @param[in]  min_max_ptr                           Pointer to the min/max vector. Minimum value in position 0, maximum value in position 1. Suppported data types: F32.
 * @param[in]  min_max_stride_x                      Stride of the min/max vector in X dimension (in bytes)
 * @param[in]  min_max_step_x                        min_max_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  min_max_offset_first_element_in_bytes The offset of the first element in the min/max vector
 */
__kernel void dequantization_layer(
    TENSOR3D_DECLARATION(input),
    TENSOR3D_DECLARATION(output),
    VECTOR_DECLARATION(min_max))
{
    // Get pixels pointer
    Tensor3D input   = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D output  = CONVERT_TO_TENSOR3D_STRUCT(output);
    Vector   min_max = CONVERT_TO_VECTOR_STRUCT(min_max);

    // min_max_value.s0 = min, min_max_value.s1 = max
    const float2 min_max_value = vload2(0, (__global float *)min_max.ptr);

    const float4 vmin  = (float4)min_max_value.s0;
    const float4 scale = (float4)((min_max_value.s1 - min_max_value.s0) / 255.0f);

    // Load data
    const uchar4 data = vload4(0, (__global uchar *)input.ptr);

    // Dequantize
    const float4 res = convert_float4(data) * scale + vmin;

    // Store result
    vstore4(res, 0, (__global float *)output.ptr);
}
