/*
 * Copyright (c) 2016, 2017 ARM Limited.
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

#define TYPE VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)

#if defined(FIXED_POINT_POSITION)
#include "fixed_point.h"

#define CONST_ONE (1 << FIXED_POINT_POSITION)
#define ABS_OP(a) ABS_SAT_OP_EXPAND((a), DATA_TYPE, VEC_SIZE)
#define ADD_OP(a, b) ADD_SAT_OP_EXPAND((a), (b), DATA_TYPE, VEC_SIZE)
#define SUB_OP(a, b) SUB_SAT_OP_EXPAND((a), (b), DATA_TYPE, VEC_SIZE)
#define MUL_OP(a, b) MUL_SAT_OP_EXPAND((a), (b), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define MLA_OP(a, b, c) MLA_SAT_OP_EXPAND((a), (b), (c), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define DIV_OP(a, b) DIV_SAT_OP_VEC_EXPAND((a), (b), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define EXP_OP(a) EXP_OP_EXPAND((a), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define LOG_OP(a) LOG_OP_EXPAND((a), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)
#define SQRT_OP(a) DIV_OP(CONST_ONE, INVSQRT_OP_EXPAND((a), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION))
#define TANH_OP(a) TANH_OP_EXPAND((a), DATA_TYPE, VEC_SIZE, FIXED_POINT_POSITION)

#else /* FIXED_POINT_POSITION */

#define CONST_ONE 1.f
#define ABS_OP(a) fabs((a))
#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define MLA_OP(a, b, c) ((b) * (c) + (a))
#define DIV_OP(a, b) ((a) / (b))
#define EXP_OP(a) exp((a))
#define LOG_OP(a) log((a))
#define SQRT_OP(a) sqrt((a))
#define TANH_OP(a) tanh((a))

#endif /* FIXED_POINT_POSITION */

// Logistic Activation
inline TYPE logistic_op(TYPE x)
{
    return DIV_OP((TYPE)CONST_ONE, ADD_OP((TYPE)CONST_ONE, EXP_OP(-x)));
}
// Hyperbolic Tangent Activation
inline TYPE tanh_op(TYPE x)
{
    return MUL_OP((TYPE)A_VAL, TANH_OP(MUL_OP((TYPE)B_VAL, x)));
}
// RELU Tangent Activation
inline TYPE relu_op(TYPE x)
{
    return max(0, x);
}
// Bounded RELU Activation
inline TYPE brelu_op(TYPE x)
{
    return min((TYPE)A_VAL, max(0, x));
}
// Lower Upper Bounded RELU Activation
inline TYPE lu_brelu_op(TYPE x)
{
    return min(max(x, (TYPE)B_VAL), (TYPE)A_VAL);
}
// Leaky RELU Activation
inline TYPE lrelu_op(TYPE x)
{
    return select(MUL_OP((TYPE)A_VAL, x), x, x > (TYPE)0);
}
// Soft RELU Activation
inline TYPE srelu_op(TYPE x)
{
    return LOG_OP(ADD_OP((TYPE)CONST_ONE, EXP_OP(x)));
}
// Absolute Activation
inline TYPE abs_op(TYPE x)
{
    return ABS_OP(x);
}
// Square Activation
inline TYPE square_op(TYPE x)
{
    return MUL_OP(x, x);
}
// Square-root Activation
inline TYPE sqrt_op(TYPE x)
{
    return SQRT_OP(x);
}
// Linear Activation
inline TYPE linear_op(TYPE x)
{
    return MLA_OP((TYPE)B_VAL, (TYPE)A_VAL, x);
}

#define ACTIVATION_OP2(op, x) op##_op(x)
#define ACTIVATION_OP(op, x) ACTIVATION_OP2(op, x)

/** This performs an activation function floating point inputs.
 *
 * @note In order to perform the activation function "in-place", the pre-processor -DIN_PLACE must be passed at compile time
 *
 * @note Datatype should be given as a preprocessor argument using -DDATA_TYPE=type. e.g. -DDATA_TYPE=short
 * @note Vector size should be given as a preprocessor argument using -DVEC_SIZE=size. e.g. -DVEC_SIZE=16
 * @note Activation function should be given as a preprocessor argument using -DACT=name. e.g. -DACT=TANH
 * @note A, B variables required by some activation functions are set using -DA_VAL= and -DB_VAL= respectively.
 * @note In case of fixed point calculations the fixed point position is passed using -DFIXED_POINT_POSITION=position. e.g. -DFIXED_POINT_POSITION=3.
 *
 * @param[in]  input_ptr                            Pointer to the source image. Supported data types: QS8/QS16/F16/F32
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
    data = ACTIVATION_OP(ACT, data);

    // Store result
    VSTORE(VEC_SIZE)
    (data, 0, (__global DATA_TYPE *)output.ptr);
}
