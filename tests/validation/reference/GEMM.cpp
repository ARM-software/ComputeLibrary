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
#include "GEMM.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename std::enable_if<is_floating_point<T>::value, int>::type>
SimpleTensor<T> gemm(const SimpleTensor<T> &a, const SimpleTensor<T> &b, const SimpleTensor<T> &c, float alpha, float beta)
{
    // Create reference
    SimpleTensor<T> dst{ c.shape(), c.data_type(), 1, c.fixed_point_position() };

    // Compute reference
    const int M = dst.shape().y();
    const int N = dst.shape().x();
    const int K = a.shape().x();

    for(int row = 0; row < M; ++row)
    {
        for(int col = 0; col < N; ++col)
        {
            T acc(0);

            for(int k = 0; k < K; ++k)
            {
                acc += a[row * K + k] * b[k * N + col];
            }

            // Finalize the result: alpha * A * B + beta * C
            dst[col + row * N] = alpha * acc + beta * c[col + row * N];
        }
    }

    return dst;
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type>
SimpleTensor<T> gemm(const SimpleTensor<T> &a, const SimpleTensor<T> &b, const SimpleTensor<T> &c, float alpha, float beta)
{
    using namespace fixed_point_arithmetic;

    // Create reference
    SimpleTensor<T> dst{ c.shape(), c.data_type(), 1, c.fixed_point_position() };

    // Compute reference
    using promoted_type = fixed_point_arithmetic::traits::promote_t<T>;

    const int M                    = dst.shape().y();
    const int N                    = dst.shape().x();
    const int K                    = a.shape().x();
    const int fixed_point_position = a.fixed_point_position();

    const fixed_point<T> alpha_q(alpha, fixed_point_position);
    const fixed_point<T> beta_q(beta, fixed_point_position);

    for(int row = 0; row < M; ++row)
    {
        for(int col = 0; col < N; ++col)
        {
            fixed_point<promoted_type> acc_q(0, fixed_point_position);

            for(int k = 0; k < K; ++k)
            {
                const fixed_point<promoted_type> a0_q(a[row * K + k], fixed_point_position, true);
                const fixed_point<promoted_type> b0_q(b[k * N + col], fixed_point_position, true);

                acc_q = acc_q + (a0_q * b0_q);
            }

            // Finalize the result: alpha * A * B + beta * C
            const fixed_point<T> c0_q(c[col + row * N], fixed_point_position, true);

            fixed_point<T> res_q(acc_q);
            res_q = alpha_q * res_q;
            res_q = res_q + (beta_q * c0_q);

            // Store the result
            dst[col + row * N] = res_q.raw();
        }
    }

    return dst;
}

template SimpleTensor<float> gemm(const SimpleTensor<float> &a, const SimpleTensor<float> &b, const SimpleTensor<float> &c, float alpha, float beta);
template SimpleTensor<half> gemm(const SimpleTensor<half> &a, const SimpleTensor<half> &b, const SimpleTensor<half> &c, float alpha, float beta);
template SimpleTensor<qint8_t> gemm(const SimpleTensor<qint8_t> &a, const SimpleTensor<qint8_t> &b, const SimpleTensor<qint8_t> &c, float alpha, float beta);
template SimpleTensor<qint16_t> gemm(const SimpleTensor<qint16_t> &a, const SimpleTensor<qint16_t> &b, const SimpleTensor<qint16_t> &c, float alpha, float beta);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
