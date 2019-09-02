/*
 * Copyright (c) 2018-2019 ARM Limited.
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
vec4 logistic_op(vec4 x)
{
    return DIV_OP(vec4(CONST_ONE), ADD_OP(CONST_ONE, EXP_OP(-x)));
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
vec4 relu_op(vec4 x)
{
    return max(vec4(0.f), x);
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

// Linear Activation
float identity_op(float x)
{
    return x;
}
