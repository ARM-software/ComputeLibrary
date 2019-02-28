/*
 * Copyright (c) 2018 ARM Limited.
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

#if defined(TYPE) && defined(SELECT_TYPE)

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
    return max((TYPE)0, x);
}
// Bounded RELU Activation
inline TYPE brelu_op(TYPE x)
{
    return min((TYPE)A_VAL, max((TYPE)0, x));
}
// Lower Upper Bounded RELU Activation
inline TYPE lu_brelu_op(TYPE x)
{
    return min(max(x, (TYPE)B_VAL), (TYPE)A_VAL);
}
// Leaky RELU Activation
inline TYPE lrelu_op(TYPE x)
{
    return select(MUL_OP((TYPE)A_VAL, x), x, CONVERT(x > (TYPE)0, SELECT_TYPE));
}
// Soft RELU Activation
inline TYPE srelu_op(TYPE x)
{
    return CONVERT(LOG_OP(ADD_OP((VEC_DATA_TYPE(float, VEC_SIZE))CONST_ONE, EXP_OP(CONVERT(x, VEC_DATA_TYPE(float, VEC_SIZE))))), TYPE);
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

#endif // defined(TYPE) && defined(SELECT_TYPE)