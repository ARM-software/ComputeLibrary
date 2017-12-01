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
template <typename T>
T safe_read(const SimpleTensor<T> &t, int y, int x)
{
    const int stride = t.shape().x();
    const int M      = t.shape().y();
    const int N      = t.shape().x();
    if((y < M) && (x < N))
    {
        return t[y * stride + x];
    }
    return 0;
}

template <typename T>
SimpleTensor<T> gemm_interleave_blocked(const SimpleTensor<T> &in, SimpleTensor<T> &out, int int_by, int block, bool transposed)
{
    const int M = out.shape().y();
    const int N = out.shape().x();
    for(int y = 0; y < M; y++)
    {
        T *out_ptr = &out[y * N];
        for(int x = 0; x < (N / int_by); x += block)
        {
            for(int z = 0; z < int_by; z++)
            {
                for(int a = 0; (out_ptr <= &out[y * N + (N - 1)]) && a < block; a++)
                {
                    if(!transposed)
                        *out_ptr++ = safe_read(in, (y * int_by) + z, x + a);
                    else
                    {
                        const T value = safe_read(in, x + a, (y * int_by) + z);
                        *out_ptr++    = value;
                    }
                }
            }
        }
    }
    return out;
}

template SimpleTensor<uint8_t> gemm_interleave_blocked(const SimpleTensor<uint8_t> &in, SimpleTensor<uint8_t> &out, int int_by, int block, bool transposed);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
