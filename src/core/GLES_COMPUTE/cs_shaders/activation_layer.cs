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
layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y, local_size_z = LOCAL_SIZE_Z) in;

#include "helpers.h"

#ifdef DATA_TYPE_FP32
precision highp float;
#elif defined(DATA_TYPE_FP16)
#if defined(LOGISTIC) || defined(TANH) || defined(SRELU) || defined(SQRT)
precision highp float;
#else  /*LOGISTIC_TANH_SRELU_SQRT*/
precision mediump float;
#endif /*LOGISTIC_TANH_SRELU_SQRT*/
#endif /*DATA_TYPE_FP32*/

#define ABS_OP(a) abs((a))
#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define MLA_OP(a, b, c) ((b) * (c) + (a))
#define DIV_OP(a, b) ((a) / (b))
#define EXP_OP(a) exp((a))
#define LOG_OP(a) log((a))
#define SQRT_OP(a) sqrt((a))
#define CONST_ONE (1.f)

// Logistic Activation
float logistic_op(float x)
{
    return DIV_OP(CONST_ONE, ADD_OP(CONST_ONE, EXP_OP(-x)));
}
// Hyperbolic Tangent Activation
float tanh_op(float x)
{
    float tmp = float(B_VAL) * x;
    if(tmp > 10.f)
    {
        return MUL_OP(float(A_VAL), 1.f);
    }
    else if(tmp < -10.f)
    {
        return MUL_OP(float(A_VAL), -1.f);
    }
    else
    {
        return MUL_OP(float(A_VAL), tanh(tmp + 0.000001f));
    }
}
// RELU Tangent Activation
float relu_op(float x)
{
    return max(0.f, x);
}
// Bounded RELU Activation
float brelu_op(float x)
{
    return min(float(A_VAL), max(float(0.0), x));
}
// Lower Upper Bounded RELU Activation
float lu_brelu_op(float x)
{
    return min(max(x, float(B_VAL)), float(A_VAL));
}
// Leaky RELU Activation
float lrelu_op(float x)
{
    return (x > float(0.0)) ? x : MUL_OP(float(A_VAL), x);
}
// Soft RELU Activation
float srelu_op(float x)
{
    return LOG_OP(ADD_OP(CONST_ONE, EXP_OP(x)));
}
// Absolute Activation
float abs_op(float x)
{
    return ABS_OP(x);
}
// Square Activation
float square_op(float x)
{
    return MUL_OP(x, x);
}
// Square-root Activation
float sqrt_op(float x)
{
    return SQRT_OP(x);
}
// Linear Activation
float linear_op(float x)
{
    return MLA_OP(float(B_VAL), float(A_VAL), x);
}

layout(std140) uniform shader_params
{
    TENSOR3D_PARAM_DECLARATION(src);
    TENSOR3D_PARAM_DECLARATION(dst);
};

#ifdef DATA_TYPE_FP32
BUFFER_DECLARATION(src, 1, float, readonly);
BUFFER_DECLARATION(dst, 2, float, writeonly);

/** This performs an activation function floating point inputs.
 *
 * @note Activation function should be given as a preprocessor argument using "#define act_name". e.g. "#define TANH"
 * @note A, B variables required by some activation functions are set using A_VAL= and B_VAL= respectively.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F32
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      ride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT(dst);

    float data     = src_ptr[src.current_offset];
    float data_out = 0.f;
    // Perform activation

#ifdef LOGISTIC
    data_out = logistic_op(data);
#elif defined(TANH)     /*LOGISTIC*/
    data_out = tanh_op(data);
#elif defined(RELU)     /*RELU*/
    data_out = relu_op(data);
#elif defined(BRELU)    /*BRELU*/
    data_out = brelu_op(data);
#elif defined(LU_BRELU) /*LU_BRELU*/
    data_out = lu_brelu_op(data);
#elif defined(LRELU)    /*LRELU*/
    data_out = lrelu_op(data);
#elif defined(SRELU)    /*SRELU*/
    data_out = srelu_op(data);
#elif defined(ABS)      /*ABS*/
    data_out = abs_op(data);
#elif defined(SQUARE)   /*SQUARE*/
    data_out = square_op(data);
#elif defined(SQRT)     /*SQRT*/
    data_out = sqrt_op(data);
#elif defined(LINEAR)   /*LINEAR*/
    data_out = linear_op(data);
#else                   /*LOGISTIC*/
#error Activation function not provided
#endif /*LOGISTIC*/

    dst_ptr[dst.current_offset] = data_out;
}

#elif defined(DATA_TYPE_FP16)
BUFFER_DECLARATION(src, 1, uint, readonly);
BUFFER_DECLARATION(dst, 2, uint, writeonly);

/** This performs an activation function floating point inputs.
 *
 * @note Activation function should be given as a preprocessor argument using "#define act_name". e.g. "#define TANH"
 * @note A, B variables required by some activation functions are set using A_VAL= and B_VAL= respectively.
 *
 * @param[in]  src_ptr                           Pointer to the source image. Supported data types: F16
 * @param[in]  src_stride_x                      Stride of the source image in X dimension (in bytes)
 * @param[in]  src_step_x                        src_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  src_stride_y                      Stride of the source image in Y dimension (in bytes)
 * @param[in]  src_step_y                        src_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  src_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  src_step_z                        src_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  src_offset_first_element_in_bytes The offset of the first element in the source image
 * @param[out] dst_ptr                           Pointer to the destination image. Supported data types: same as @p src_ptr
 * @param[in]  dst_stride_x                      Stride of the destination image in X dimension (in bytes)
 * @param[in]  dst_step_x                        dst_stride_x * number of elements along X processed per workitem(in bytes)
 * @param[in]  dst_stride_y                      ride of the destination image in Y dimension (in bytes)
 * @param[in]  dst_step_y                        dst_stride_y * number of elements along Y processed per workitem(in bytes)
 * @param[in]  dst_stride_z                      Stride of the source tensor in Z dimension (in bytes)
 * @param[in]  dst_step_z                        dst_stride_z * number of elements along Z processed per workitem(in bytes)
 * @param[in]  dst_offset_first_element_in_bytes The offset of the first element in the destination image
 */
void main(void)
{
    Tensor3D src = CONVERT_TO_TENSOR3D_STRUCT_FP16(src);
    Tensor3D dst = CONVERT_TO_TENSOR3D_STRUCT_FP16(dst);

    uint data = src_ptr[src.current_offset >> 2];
    // Perform activation
    float a = unpackHalf2x16(data).x;
    float b = unpackHalf2x16(data).y;
    vec2  data_out;
#ifdef LOGISTIC         /*LOGISTIC*/
    data_out.x = logistic_op(a);
    data_out.y = logistic_op(b);
#elif defined(TANH)     /*TANH*/
    data_out.x = tanh_op(a);
    data_out.y = tanh_op(b);
#elif defined(RELU)     /*RELU*/
    data_out.x = relu_op(a);
    data_out.y = relu_op(b);
#elif defined(BRELU)    /*BRELU*/
    data_out.x = brelu_op(a);
    data_out.y = brelu_op(b);
#elif defined(LU_BRELU) /*LU_BRELU*/
    data_out.x = lu_brelu_op(a);
    data_out.y = lu_brelu_op(b);
#elif defined(LRELU)    /*LRELU*/
    data_out.x = lrelu_op(a);
    data_out.y = lrelu_op(b);
#elif defined(SRELU)    /*SRELU*/
    data_out.x = srelu_op(a);
    data_out.y = srelu_op(b);
#elif defined(ABS)      /*ABS*/
    data_out.x = abs_op(a);
    data_out.y = abs_op(b);
#elif defined(SQUARE)   /*SQUARE*/
    data_out.x = square_op(a);
    data_out.y = square_op(b);
#elif defined(SQRT)     /*SQRT*/
    data_out.x = sqrt_op(a);
    data_out.y = sqrt_op(b);
#elif defined(LINEAR)   /*LINEAR*/
    data_out.x = linear_op(a);
    data_out.y = linear_op(b);
#else                   /*LOGISTIC*/
#error Activation function not provided
#endif /*LOGISTIC*/

    dst_ptr[dst.current_offset >> 2] = packHalf2x16(data_out);
}
#endif /*DATA_TYPE_FP32*/
