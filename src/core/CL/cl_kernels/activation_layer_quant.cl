/*
 * Copyright (c) 2016-2020 Arm Limited.
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
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively.
 * @note Quantization scales of the input/output tensors are passed in with -DS1_VAL= and -DS2_VAL= respectively.
 * @note Quantization offsets of the input/output tensors are passed in only if asymmetric with -DO1_VAL= and -DO2_VAL= respectively.
 * @note Quantized value of constant zero should be given as a preprocessor argument using -DCONST_0=value. e.g. -DCONST_0=128.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8/QASYMM8_SIGNED/QSYMM16
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
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE * sizeof(DATA_TYPE) - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE * sizeof(DATA_TYPE)), 0);

    // Get pixels pointer
    __global uchar *input_addr = input_ptr + input_offset_first_element_in_bytes + x_offs + get_global_id(1) * input_stride_y + get_global_id(2) * input_stride_z;
#ifdef IN_PLACE
    __global uchar *output_addr = input_addr;
#else  /* IN_PLACE */
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs + get_global_id(1) * output_stride_y + get_global_id(2) * output_stride_z;
#endif /* IN_PLACE */

    // Load data
    TYPE data0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr);

    VEC_FLOAT data_flt = CONVERT(data0, VEC_FLOAT);
#if defined(O1_VAL)
    data_flt = round(data_flt - (float)O1_VAL) * ((float)S1_VAL);
#else  // defined(O1_VAL)
    data_flt                    = round(data_flt) * ((float)S1_VAL);
#endif // defined(O1_VAL)
    data_flt = ACTIVATION(ACT, float, VEC_SIZE, data_flt, A_VAL, B_VAL);

#if defined(O2_VAL)
    data0 = CONVERT_SAT(round(data_flt / ((float)S2_VAL)) + (float)O2_VAL, TYPE);
#else  // defined(O2_VAL)
    data0                       = CONVERT_SAT(round(data_flt / ((float)S2_VAL)), TYPE);
#endif // defined(O2_VAL)

    // Store result
    STORE_VECTOR_SELECT(data, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
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
 * @note Leftover vector size has to be passed at compile time using -DVEC_SIZE_LEFTOVER. e.g. -DVEC_SIZE_LEFTOVER=3. It is defined as the remainder between the input's first dimension and VEC_SIZE
 * @note Activation function should be given as a preprocessor argument using -DACT=name. e.g. -DACT=TANH
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively.
 * @note Quantization scales of the input/output tensors are passed in with -DS1_VAL= and -DS2_VAL= respectively.
 * @note Quantization offsets of the input/output tensors are passed in with -DO1_VAL= and -DO2_VAL= respectively.
 * @note Quantized value of constant zero should be given as a preprocessor argument using -DCONST_0=value. e.g. -DCONST_0=128.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8/QASYMM8_SIGNED/QSYMM16
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
    uint x_offs = max((int)(get_global_id(0) * VEC_SIZE * sizeof(DATA_TYPE) - (VEC_SIZE - VEC_SIZE_LEFTOVER) % VEC_SIZE * sizeof(DATA_TYPE)), 0);

    // Get pixels pointer
    __global uchar *input_addr  = input_ptr + input_offset_first_element_in_bytes + x_offs + get_global_id(1) * input_stride_y + get_global_id(2) * input_stride_z;
#ifdef IN_PLACE
    __global uchar *output_addr = input_addr;
#else  /* IN_PLACE */
    __global uchar *output_addr = output_ptr + output_offset_first_element_in_bytes + x_offs + get_global_id(1) * output_stride_y + get_global_id(2) * output_stride_z;
#endif /* IN_PLACE */

    // Load data
    TYPE data0 = VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input_addr);

    data0 = PERFORM_ACTIVATION_QUANT(ACT, data0);

    // Store result
    STORE_VECTOR_SELECT(data, DATA_TYPE, output_addr, VEC_SIZE, VEC_SIZE_LEFTOVER, VEC_SIZE_LEFTOVER != 0 && get_global_id(0) == 0)
}
#endif // defined(ACT)
#endif // defined(FLOAT_DOMAIN)
