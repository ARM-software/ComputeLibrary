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
#include "ElementWiseUnary.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> elementwise_unary(const SimpleTensor<T> &src, ElementWiseUnary op)
{
    SimpleTensor<T> dst(src.shape(), src.data_type());

    for(int i = 0; i < src.num_elements(); ++i)
    {
        switch(op)
        {
            case ElementWiseUnary::RSQRT:
                dst[i] = 1.f / std::sqrt(src[i]);
                break;
            case ElementWiseUnary::EXP:
                dst[i] = std::exp(src[i]);
                break;
            default:
                ARM_COMPUTE_ERROR("Not implemented");
        }
    }

    return dst;
}

template SimpleTensor<float> elementwise_unary(const SimpleTensor<float> &src, ElementWiseUnary op);
template SimpleTensor<half> elementwise_unary(const SimpleTensor<half> &src, ElementWiseUnary op);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
