/*
 * Copyright (c) 2019-2021 Arm Limited.
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

#if defined(S1_VAL) && !defined(S2_VAL)
#define S2_VAL S1_VAL
#endif // defined(S1_VAL) && !defined(S2_VAL)
#if defined(O1_VAL) && !defined(O2_VAL)
#define O2_VAL O1_VAL
#endif // defined(O1_VAL) && !defined(O2_VAL)

// RELU Activation
inline TYPE relu_op(TYPE x)
{
    return max((TYPE)CONST_0, x);
}
// Bounded RELU Activation
inline TYPE brelu_op(TYPE x)
{
    return min((TYPE)A_VAL, max((TYPE)CONST_0, x));
}
// Lower Upper Bounded RELU Activation
inline TYPE lu_brelu_op(TYPE x)
{
    return min(max(x, (TYPE)B_VAL), (TYPE)A_VAL);
}
// Hard Swish Activation
inline TYPE hard_swish_op(TYPE x)
{
    return (x * ((min(max((TYPE)(x + (TYPE)3.f), (TYPE)0.f), (TYPE)6.f)) * (TYPE)0.166666667f));
}

inline TYPE identiy_op(TYPE x)
{
    return x;
}

#define ACTIVATION_OP2(op, x) op##_op(x)
#define ACTIVATION_OP(op, x) ACTIVATION_OP2(op, x)

#if defined(S1_VAL) && defined(S2_VAL)
#if defined(O1_VAL) && defined(O2_VAL)
#define PERFORM_ACTIVATION_QUANT(act, data)                                                       \
    ({                                                                                            \
        data = ACTIVATION_OP(act, data);                                                          \
        \
        VEC_DATA_TYPE(float, VEC_SIZE)                                                            \
        fdata = CONVERT(data, VEC_DATA_TYPE(float, VEC_SIZE));                                    \
        \
        fdata = round((fdata - (float)O1_VAL) * ((float)S1_VAL / (float)S2_VAL) + (float)O2_VAL); \
        data  = CONVERT_SAT(fdata, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE));                           \
    })
#else // defined(O1_VAL) && defined(O2_VAL)
#define PERFORM_ACTIVATION_QUANT(act, data)                             \
    ({                                                                  \
        data = ACTIVATION_OP(act, data);                                \
        \
        VEC_DATA_TYPE(float, VEC_SIZE)                                  \
        fdata = CONVERT(data, VEC_DATA_TYPE(float, VEC_SIZE));          \
        \
        fdata = round((fdata) * ((float)S1_VAL / (float)S2_VAL));       \
        data  = CONVERT_SAT(fdata, VEC_DATA_TYPE(DATA_TYPE, VEC_SIZE)); \
    })
#endif /* defined(O1_VAL) && defined(O2_VAL) */
#else  /* defined(S1_VAL) && defined(S2_VAL) */
#define PERFORM_ACTIVATION_QUANT(act, data) \
    ({                                      \
        data = ACTIVATION_OP(act, data);    \
    })
#endif /* defined(S1_VAL) && defined(S2_VAL) */
