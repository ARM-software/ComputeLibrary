/*
 * Copyright (c) 2017, 2018 ARM Limited.
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
SimpleTensor<T> gemm_transpose_1xW(const SimpleTensor<T> &in)
{
    const int         W = 16 / sizeof(T);
    const TensorShape shape_out(static_cast<size_t>(in.shape().y() * W), static_cast<size_t>(std::ceil(in.shape().x() / static_cast<float>(W))));
    SimpleTensor<T>   out(shape_out, in.data_type());
    const int32_t     in_height     = in.shape().y();
    const int32_t     in_width      = in.shape().x();
    const int32_t     out_width     = out.shape().x();
    const T          *in_base_addr  = reinterpret_cast<const T *>(in.data());
    T                *out_base_addr = reinterpret_cast<T *>(out.data());
    int               x             = 0;
    for(; x < in_width; x += W)
    {
        for(int y = 0; y < in_height; y++)
        {
            const T *in_addr  = (in_base_addr + x + y * in_width);
            T       *out_addr = (out_base_addr + y * W + (x / W) * out_width);

            for(int k = 0; k < W; ++k)
            {
                // If the input width is not multiple of W, we fill the reference with 0s
                if((x + k) >= in_width)
                {
                    out_addr[k] = T(0);
                }
                else
                {
                    out_addr[k] = in_addr[k];
                }
            }
        }
    }
    return out;
}

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
