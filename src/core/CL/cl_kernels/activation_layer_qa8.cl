/*
 * Copyright (c) 2016-2018 ARM Limited.
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
#include "helpers_asymm.h"

#define TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

// Logistic Activation
inline TYPE logistic_op(TYPE x)
{
    // This function is a temporary function that is not actually executed.
    // To keep the existing structure, it is added.
    return x;
}
// RELU Activation
inline TYPE relu_op(TYPE x)
{
    return max((TYPE)CONST_0, x);
}
// Bounded RELU Activation
inline TYPE brelu_op(TYPE x)
{
    return min((TYPE)A_VAL, max(CONST_0, x));
}
// Lower Upper Bounded RELU Activation
inline TYPE lu_brelu_op(TYPE x)
{
    return min(max(x, (TYPE)B_VAL), (TYPE)A_VAL);
}

#define ACTIVATION_OP2(op, x) op##_op(x)
#define ACTIVATION_OP(op, x) ACTIVATION_OP2(op, x)

#if defined(O1_VAL) && defined(O2_VAL) && defined(S1_VAL) && defined(S2_VAL)
#define PERFORM_ACTIVATION_QA8(act, data)                                                         \
    ({                                                                                            \
        data = ACTIVATION_OP(act, data);                                                          \
        \
        VEC_DATA_TYPE(float, VEC_SIZE)                                                            \
        fdata = CONVERT(data, VEC_DATA_TYPE(float, VEC_SIZE));                                    \
        \
        fdata = round((fdata - (float)O1_VAL) * ((float)S1_VAL / (float)S2_VAL) + (float)O2_VAL); \
        data  = CONVERT_SAT(fdata, VEC_DATA_TYPE(uchar, VEC_SIZE));                               \
    })
#else /* defined(O1_VAL) && defined(O2_VAL) && defined(S1_VAL) && defined(S2_VAL) */
#define PERFORM_ACTIVATION_QA8(act, data) \
    ({                                    \
        data = ACTIVATION_OP(act, data);  \
    })
#endif /* defined(O1_VAL) && defined(O2_VAL) && defined(S1_VAL) && defined(S2_VAL) */

#if defined(ACT)

/** This performs an activation function on QASYMM8 inputs.
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
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8
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
__kernel void activation_layer_qa8(
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

    data = PERFORM_ACTIVATION_QA8(ACT, data);

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
}

#endif /* defined(ACT) */

/** This performs a logistic activation function on QASYMM8 inputs.
 *
 * @note In order to perform the logistic activation function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Quantization scales of the input/output tensors are passed in with -DS1_VAL= and -DS2_VAL= respectively.
 * @note Quantization offsets of the input/output tensors are passed in with -DO1_VAL= and -DO2_VAL= respectively.
 * @note Quantized value of constant zero should be given as a preprocessor argument using -DCONST_0=value. e.g. -DCONST_0=128.
 * @note Quantized can be optionally passed at compile time using -DINPUT_MULTIPLIER and -DINPUT_LEFT_SHIFT (if undefined, assume that the original data is used and not scaled separately.
 * @note Number of integer bits should be given as a preprocessor argument using -DINPUT_INTEGER_BITS=value. e.g. -DINPUT_INTEGER_BITS=4.
 * @note Number of input range radius should be given at compile time using -DINPUT_RANGE_RADIUS.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QASYMM8
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
__kernel void activation_layer_logistic_qa8(
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
    VEC_DATA_TYPE(int, 16)
    data = CONVERT(VLOAD(VEC_SIZE)(0, (__global DATA_TYPE *)input.ptr), VEC_DATA_TYPE(int, 16));

    VEC_DATA_TYPE(int, 16)
    result = data;

#if defined(INPUT_INTEGER_BITS) && defined(INPUT_RANGE_RADIUS)
    const VEC_DATA_TYPE(int, 16) Q0_one      = INT_MAX;
    const VEC_DATA_TYPE(int, 16) Q0_one_half = (1 << 30);

    VEC_DATA_TYPE(int, 16)
    input_val_centered = data;
#ifdef O1_VAL
    input_val_centered = data - O1_VAL;
#endif /* O1_VAL */

    VEC_DATA_TYPE(int, 16) result_left  = ASYMM_SELECT_USING_MASK(input_val_centered <= -INPUT_RANGE_RADIUS, 1, 0, 16);
    VEC_DATA_TYPE(int, 16) result_right = ASYMM_SELECT_USING_MASK(input_val_centered >= INPUT_RANGE_RADIUS, 255, 0, 16);

    VEC_DATA_TYPE(int, 16) input_mask         = ASYMM_SELECT_USING_MASK(input_val_centered > -INPUT_RANGE_RADIUS && input_val_centered < INPUT_RANGE_RADIUS, 1, 0, 16);
    VEC_DATA_TYPE(int, 16) input_val_rescaled = input_val_centered * input_mask;
#if defined(INPUT_MULTIPLIER) && defined(INPUT_LEFT_SHIFT)
    if(INPUT_MULTIPLIER > 1)
    {
        input_val_rescaled   = ASYMM_MULT(input_val_rescaled * (1 << INPUT_LEFT_SHIFT), INPUT_MULTIPLIER, 16);
    }
#endif /* defined(INPUT_MULTIPLIER) && defined(INPUT_LEFT_SHIFT) */

    VEC_DATA_TYPE(int, 16) mask_if_positive   = ASYMM_MASK_IF_NON_ZERO(input_val_rescaled > CONST_0, 16);
    VEC_DATA_TYPE(int, 16) mask_if_zero       = ASYMM_MASK_IF_NON_ZERO(!input_val_rescaled, 16);
    VEC_DATA_TYPE(int, 16) abs_input          = ASYMM_SELECT_USING_MASK(mask_if_positive, input_val_rescaled, -input_val_rescaled, 16);
    VEC_DATA_TYPE(int, 16) result_exp         = ASYMM_EXP_ON_NEGATIVE_VALUES(-abs_input, INPUT_INTEGER_BITS, 16);
    VEC_DATA_TYPE(int, 16) result_if_positive = ASYMM_ONE_OVER_ONE_PLUS_X_FOR_X_IN_0_1(result_exp, 16);
    VEC_DATA_TYPE(int, 16) result_if_negative = Q0_one - result_if_positive;
    VEC_DATA_TYPE(int, 16) result_logistic    = ASYMM_SELECT_USING_MASK(mask_if_zero, Q0_one_half, ASYMM_SELECT_USING_MASK(mask_if_positive, result_if_positive, result_if_negative, 16), 16);

    result_logistic = ASYMM_ROUNDING_DIVIDE_BY_POW2(result_logistic, 23, 16);
    result_logistic = ASYMM_SELECT_USING_MASK(result_logistic == 256, 255, result_logistic, 16);
    result_logistic = result_logistic * input_mask;

    result = result_left + result_right + result_logistic;
#endif /* defined(INPUT_INTEGER_BITS) && defined(INPUT_RANGE_RADIUS) */

    // Store result
    TYPE tmp = CONVERT(result, TYPE);
    VSTORE(VEC_SIZE)
    (tmp, 0, (__global DATA_TYPE *)output.ptr);
}
