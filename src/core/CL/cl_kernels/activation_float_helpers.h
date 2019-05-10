/*
 * Copyright (c) 2019 ARM Limited.
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

#if GPU_ARCH == GPU_ARCH_BIFROST
#define MLA(a, b, c) (fma(c, b, a))
#else // GPU_ARCH == GPU_ARCH_BIFROST
#define MLA(a, b, c) ((b) * (c) + (a))
#endif // GPU_ARCH == GPU_ARCH_BIFROST
// Logistic Activation
#define logistic_op(DATA_TYPE, x, A_VAL, B_VAL) ((DATA_TYPE)1.0 / ((DATA_TYPE)1.0 + exp(-x)))

// Hyperbolic Tangent Activation
#define tanh_op(DATA_TYPE, x, A_VAL, B_VAL) ((DATA_TYPE)A_VAL * tanh((DATA_TYPE)B_VAL * x))

// RELU Tangent Activation
#define relu_op(DATA_TYPE, x, A_VAL, B_VAL) (max((DATA_TYPE)0.0, x))

// Bounded RELU Activation
#define brelu_op(DATA_TYPE, x, A_VAL, B_VAL) (min((DATA_TYPE)A_VAL, max((DATA_TYPE)0.0, x)))

// Lower Upper Bounded RELU Activation
#define lu_brelu_op(DATA_TYPE, x, A_VAL, B_VAL) (min(max(x, (DATA_TYPE)B_VAL), (DATA_TYPE)A_VAL))

// Leaky RELU Activation
#define lrelu_op(DATA_TYPE, x, A_VAL, B_VAL) ((min(x, (DATA_TYPE)0.0) * (DATA_TYPE)A_VAL) + max(x, (DATA_TYPE)0.0))

// Soft RELU Activation
#define srelu_op(DATA_TYPE, x, A_VAL, B_VAL) (log((DATA_TYPE)1.0 + exp(x)))

// Absolute Activation
#define abs_op(DATA_TYPE, x, A_VAL, B_VAL) (fabs(x))

// Square Activation
#define square_op(DATA_TYPE, x, A_VAL, B_VAL) (x * x)

// Square-root Activation
#define sqrt_op(DATA_TYPE, x, A_VAL, B_VAL) (sqrt(x))

// Linear Activation
#define linear_op(DATA_TYPE, x, A_VAL, B_VAL) (MLA((DATA_TYPE)B_VAL, (DATA_TYPE)A_VAL, x))

// Identity Activation
#define identity_op(DATA_TYPE, x, A_VAL, B_VAL) (x)

#define OP(op, DATA_TYPE, x, A_VAL, B_VAL) op##_op(DATA_TYPE, x, A_VAL, B_VAL)

#define ACTIVATION(op, DATA_TYPE, x, A_VAL, B_VAL) OP(op, DATA_TYPE, x, A_VAL, B_VAL)
