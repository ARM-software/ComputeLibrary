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
SimpleTensor<uint32_t> gemmlowp(const SimpleTensor<uint8_t> &a, const SimpleTensor<uint8_t> &b, SimpleTensor<uint32_t> &c)
{
    ARM_COMPUTE_UNUSED(a);
    ARM_COMPUTE_UNUSED(b);
    ARM_COMPUTE_UNUSED(c);
    const int            K       = a.shape().x();
    const int            b_width = b.shape().x();
    const int            rows    = c.shape().y(); //M
    const int            cols    = c.shape().x(); //N
    std::vector<int32_t> acc;
    acc.resize(cols);
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            acc[j] = 0;
        }
        for(int k = 0; k < K; ++k)
        {
            auto tmp_a = static_cast<int32_t>(a[k + i * K]);
            for(int j = 0; j < b_width; ++j)
            {
                auto          tmp_b       = static_cast<int32_t>(b[j + k * b_width]);
                const int32_t mult_as_int = tmp_a * tmp_b;
                acc[j] += mult_as_int;
            }
        }
        for(int j = 0; j < cols; ++j)
        {
            c[j + i * cols] = acc[j];
        }
    }

    return c;
}

template <typename T>
SimpleTensor<T> gemmlowp(const SimpleTensor<T> &a, const SimpleTensor<T> &b, SimpleTensor<T> &c,
                         int32_t a_offset, int32_t b_offset, int32_t c_offset, int32_t c_mult_int, int32_t out_shift)
{
    const int            K       = a.shape().x();
    const int            b_width = b.shape().x();
    const int            rows    = c.shape().y(); //M
    const int            cols    = c.shape().x(); //N
    std::vector<int32_t> acc;
    acc.resize(cols);
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            acc[j] = 0;
        }
        for(int k = 0; k < K; ++k)
        {
            const int32_t tmp_a = a_offset + static_cast<int32_t>(a[k + i * K]);
            for(int j = 0; j < b_width; ++j)
            {
                const int32_t tmp_b       = b_offset + static_cast<int32_t>(b[j + k * b_width]);
                const int32_t mult_as_int = tmp_a * tmp_b;
                acc[j] += mult_as_int;
            }
        }
        for(int j = 0; j < cols; ++j)
        {
            const int32_t result = ((c_offset + acc[j]) * c_mult_int) >> out_shift;
            c[j + i * cols]      = static_cast<uint8_t>(std::min(255, std::max(0, result)));
        }
    }

    return c;
}

template SimpleTensor<uint8_t> gemmlowp(const SimpleTensor<uint8_t> &a, const SimpleTensor<uint8_t> &b, SimpleTensor<uint8_t> &c,
                                        int32_t a_offset, int32_t b_offset, int32_t c_offset, int32_t c_mult_int, int32_t out_shift);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
