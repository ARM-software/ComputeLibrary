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
#if defined(ACT) && defined(DATA_TYPE) && defined(VEC_SIZE)

#define TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

#include "activation_float_helpers.h"

/** This performs an activation function floating point inputs.
 *
 * @note In order to perform the activation function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Activation function should be given as a preprocessor argument using -DACT=name. e.g. -DACT=TANH
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: F16/F32
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes The offset of the first element in the destination image
 */
__kernel void activation_layer(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get pixels pointer
    Tensor3D input = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load data
    TYPE data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input.ptr);

    // Perform activation
    data = ACTIVATION(ACT, DATA_TYPE, data, A_VAL, B_VAL);

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
}

#endif /* defined(ACT) */
