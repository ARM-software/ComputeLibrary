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

/** Apply cross map normalization.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  input_ptr                                   Pointer to the first source tensor. Supported data types: F16, F32
 * @param[in]  input_stride_x                              Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                                input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                              Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                                input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                              Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                                input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes         The offset of the first element in the first source tensor
 * @param[in]  squared_input_ptr                           Pointer to the second source tensor. Supported data types: F16, F32
 * @param[in]  squared_input_stride_x                      Stride of the second source tensor in X dimension (in bytes)
 * @param[in]  squared_input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  squared_input_stride_y                      Stride of the second source tensor in Y dimension (in bytes)
 * @param[in]  squared_input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  squared_input_stride_z                      Stride of the second source tensor in Z dimension (in bytes)
 * @param[in]  squared_input_step_z                        input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  squared_input_offset_first_element_in_bytes The offset of the second element in the second source tensor
 * @param[out] output_ptr                                  Pointer to the destination tensor. Supported data types: F16, F32
 * @param[in]  output_stride_x                             Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                               output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                             Stride of the destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                               output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                             Stride of the destination tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                               output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes        The offset of the first element in the destination tensor
 * @param[in]  coeff                                       Alpha parameter / norm_size
 * @param[in]  beta                                        Beta parameter in the normalization equation
 * @param[in]  kappa                                       Kappa parameter in the normalization equation
 * @param[in]  radius                                      Number of elements on the right or left side to normalize across
 */
__kernel void normalization_layer_cross_map(TENSOR3D_DECLARATION(input),
                                            TENSOR3D_DECLARATION(squared_input),
                                            TENSOR3D_DECLARATION(output),
                                            float coeff,
                                            float beta,
                                            float kappa,
                                            uint  radius)
{
    Tensor3D in         = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D squared_in = CONVERT_TO_TENSOR3D_STRUCT(squared_input);
    Tensor3D out        = CONVERT_TO_TENSOR3D_STRUCT(output);

    DATA_TYPE acc = 0;

    const int num_of_slices = get_global_size(2);
    const int current_slice = get_global_id(2);

    const int left_slice  = max(current_slice - (int)radius, (int)0);
    const int right_slice = min(current_slice + (int)radius, (int)(num_of_slices - 1));

    for(int i = left_slice; i <= right_slice; i++)
    {
        acc += *(__global DATA_TYPE *)tensor3D_offset(&squared_in, 0, 0, i - current_slice);
    }

    const float normalized = pow(kappa + coeff * (float)acc, beta);

    const float normalized_pixel = (float) * ((__global DATA_TYPE *)in.ptr) / normalized;

    *(__global DATA_TYPE *)out.ptr = CONVERT(normalized_pixel, DATA_TYPE);
}

/** Apply in map normalization.
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 *
 * @param[in]  input_ptr                                   Pointer to the first source tensor. Supported data types: F16, F32
 * @param[in]  input_stride_x                              Stride of the first source tensor in X dimension (in bytes)
 * @param[in]  input_step_x                                input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                              Stride of the first source tensor in Y dimension (in bytes)
 * @param[in]  input_step_y                                input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                              Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                                input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes         The offset of the first element in the first source tensor
 * @param[in]  squared_input_ptr                           Pointer to the second source tensor. Supported data types: F16, F32
 * @param[in]  squared_input_stride_x                      Stride of the second source tensor in X dimension (in bytes)
 * @param[in]  squared_input_step_x                        input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  squared_input_stride_y                      Stride of the second source tensor in Y dimension (in bytes)
 * @param[in]  squared_input_step_y                        input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  squared_input_stride_z                      Stride of the second source tensor in Z dimension (in bytes)
 * @param[in]  squared_input_step_z                        input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  squared_input_offset_first_element_in_bytes The offset of the second element in the second source tensor
 * @param[out] output_ptr                                  Pointer to the destination tensor. Supported data types: F16, F32
 * @param[in]  output_stride_x                             Stride of the destination tensor in X dimension (in bytes)
 * @param[in]  output_step_x                               output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                             Stride of the first destination tensor in Y dimension (in bytes)
 * @param[in]  output_step_y                               output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                             Stride of the first source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                               output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes        The offset of the first element in the destination tensor
 * @param[in]  coeff                                       Alpha parameter / norm_size
 * @param[in]  beta                                        Beta parameter in the normalization equation
 * @param[in]  kappa                                       Kappa parameter in the normalization equation
 * @param[in]  radius                                      Number of elements on the right or left side to normalize across
 */
__kernel void normalization_layer_in_map(TENSOR3D_DECLARATION(input),
                                         TENSOR3D_DECLARATION(squared_input),
                                         TENSOR3D_DECLARATION(output),
                                         float coeff,
                                         float beta,
                                         float kappa,
                                         uint  radius)
{
    Tensor3D in         = CONVERT_TO_TENSOR3D_STRUCT(input);
    Tensor3D squared_in = CONVERT_TO_TENSOR3D_STRUCT(squared_input);
    Tensor3D out        = CONVERT_TO_TENSOR3D_STRUCT(output);

    VEC_DATA_TYPE(DATA_TYPE, 4)
    acc_vec = 0;

    const int current_pos = get_global_id(0) << 2;

    const int left_pos  = max(current_pos - (int)radius, -3);
    const int right_pos = min(current_pos + (int)radius, (int)((get_global_size(0) << 2) + 3 - 1));

    for(int i = left_pos; i <= right_pos; i += 1)
    {
        acc_vec += vload4(0, (__global DATA_TYPE *)tensor3D_offset(&squared_in, i - current_pos, 0, 0));
    }

    const float4 normalized = pow((float4)kappa + coeff * (float4)acc_vec, beta);

    const float4 normalized_pixel = CONVERT(vload4(0, (__global DATA_TYPE *)in.ptr), float4) / normalized;

    vstore4(CONVERT(normalized_pixel, VEC_DATA_TYPE(DATA_TYPE, 4)), 0, (__global DATA_TYPE *)out.ptr);
}
