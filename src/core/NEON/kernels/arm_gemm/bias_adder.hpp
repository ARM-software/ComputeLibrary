/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#pragma once

namespace arm_gemm
{

// Fallback routine to add bias to a block
template<typename T>
inline void bias_adder(T *out, unsigned int stride, const T *bias, unsigned int rows, unsigned int cols) {
    for (unsigned int row=0; row<rows; row++) {
        for (unsigned int col=0; col<cols; col++) {
            out[row * stride + col] += bias[col];
        }
    }
}

template<bool DoBias, typename T>
inline void activator(T *out, unsigned int stride, const T *bias, Activation act, unsigned int rows, unsigned int cols) {
    if (act.type == Activation::Type::None) {
        if (DoBias) {
            bias_adder(out, stride, bias, rows, cols);
        }
        return;
    }

    if (act.type == Activation::Type::ReLU) {
        for (unsigned int row=0; row<rows; row++) {
            for (unsigned int col=0; col<cols; col++) {
                T &v = out[row * stride + col];
                if (DoBias) {
                    v += bias[col];
                }
                v = std::max(static_cast<T>(0), v);
            }
        }
    }

    if (act.type == Activation::Type::BoundedReLU) {
        const T max = static_cast<T>(act.param1);

        for (unsigned int row=0; row<rows; row++) {
            for (unsigned int col=0; col<cols; col++) {
                T &v = out[row * stride + col];
                if (DoBias) {
                    v += bias[col];
                }
                v = std::max(static_cast<T>(0), std::min(v, max));
            }
        }
    }
}

} // namespace arm_gemm
