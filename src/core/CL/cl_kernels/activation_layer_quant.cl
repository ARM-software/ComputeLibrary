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
#include "activation_quant_helpers.h"

#define VEC_FLOAT VEC_DATA_TYPE(float, VEC_SIZE)

#if defined(FLOAT_DOMAIN)
// Activations performed in the float domain

#include "activation_float_helpers.h"

/** This performs an activation function on quantized inputs with float transformations.
 *
 * @note In order to perform the activation function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively.
 * @note Quantization scales of the input/output tensors are passed in with -DS1_VAL= and -DS2_VAL= respectively.
 * @note Quantization offsets of the input/output tensors are passed in only if asymmetric with -DO1_VAL= and -DO2_VAL= respectively.
 * @note Quantized value of constant zero should be given as a preprocessor argument using -DCONST_0=value. e.g. -DCONST_0=128.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8/QSYMM16
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void activation_layer_quant_f32(
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

    VEC_FLOAT data_flt = CONVERT(data, VEC_FLOAT);
#if defined(O1_VAL)
    data_flt = round(data_flt - (float)O1_VAL) * ((float)S1_VAL);
#else  // defined(O1_VAL)
    data_flt        = round(data_flt) * ((float)S1_VAL);
#endif // defined(O1_VAL)
    data_flt = ACTIVATION(ACT, float, data_flt, A_VAL, B_VAL);

#if defined(O2_VAL)
    data = CONVERT_SAT(round(data_flt / ((float)S2_VAL)) + (float)O2_VAL, TYPE);
#else  // defined(O2_VAL)
    data            = CONVERT_SAT(round(data_flt / ((float)S2_VAL)), TYPE);
#endif // defined(O2_VAL)

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
}

#else // defined(FLOAT_DOMAIN)
// Activations performed in the quantized domain

#if defined(ACT)
/** This performs an activation function on quantized inputs.
 *
 * @note In order to perform the activation function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Activation function should be given as a preprocessor argument using -DACT=name. e.g. -DACT=TANH
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively.
 * @note Quantization scales of the input/output tensors are passed in with -DS1_VAL= and -DS2_VAL= respectively.
 * @note Quantization offsets of the input/output tensors are passed in with -DO1_VAL= and -DO2_VAL= respectively.
 * @note Quantized value of constant zero should be given as a preprocessor argument using -DCONST_0=value. e.g. -DCONST_0=128.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8/QSYMM16
 * @param[in]  input_stride_x                       Stride of the source image in X dimension (in bytes)
 * @param[in]  input_step_x                         input_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  input_stride_y                       Stride of the source image in Y dimension (in bytes)
 * @param[in]  input_step_y                         input_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  input_stride_z                       Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  input_step_z                         input_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  input_offset_first_element_in_bytes  The offset of the first element in the source image
 * @param[out] output_ptr                           (Optional) Pointer to the destination image. Supported data types: same as @p input_ptr
 * @param[in]  output_stride_x                      (Optional) Stride of the destination image in X dimension (in bytes)
 * @param[in]  output_step_x                        (Optional) output_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  output_stride_y                      (Optional) Stride of the destination image in Y dimension (in bytes)
 * @param[in]  output_step_y                        (Optional) output_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  output_stride_z                      (Optional) Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  output_step_z                        (Optional) output_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  output_offset_first_element_in_bytes (Optional) The offset of the first element in the destination image
 */
__kernel void activation_layer_quant(
    TENSOR3D_DECLARATION(input)
#ifndef IN_PLACE
    ,
    TENSOR3D_DECLARATION(output)
#endif /* not IN_PLACE */
)
{
    // Get pixels pointer
    Tensor3D input  = CONVERT_TO_TENSOR3D_STRUCT(input);
#ifdef IN_PLACE
    Tensor3D output = input;
#else  /* IN_PLACE */
    Tensor3D output = CONVERT_TO_TENSOR3D_STRUCT(output);
#endif /* IN_PLACE */

    // Load data
    TYPE data = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input.ptr);

    data = PERFORM_ACTIVATION_QUANT(ACT, data);

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
}
#endif // defined(ACT)
#endif // defined(FLOAT_DOMAIN)
