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
SimpleTensor<T> gemm_interleave_4x4(const SimpleTensor<T> &in, SimpleTensor<T> &out)
{
    const T      *mtx_in     = reinterpret_cast<const T *>(in.data());
    T            *mtx_ref    = reinterpret_cast<T *>(out.data());
    const int32_t in_rows    = in.shape().y();
    const int32_t in_cols    = in.shape().x();
    const int32_t out_stride = out.shape().x();
    int32_t       y          = 0;
    for(; y <= (in_rows - 4); y += 4)
    {
        const T *in_ptr = &mtx_in[y * in_cols];

        for(int32_t x = 0; x < in_cols; x++)
        {
            const T tmp[4] = { in_ptr[x + 0 * in_cols],
                               in_ptr[x + 1 * in_cols],
                               in_ptr[x + 2 * in_cols],
                               in_ptr[x + 3 * in_cols]
                             };

            T *dst = &mtx_ref[static_cast<size_t>(x * 4.f) + static_cast<size_t>(std::ceil(y / 4.f)) * out_stride];
            memcpy(dst, tmp, sizeof(T) * 4);
        }
    }

    // Leftover along the Y direction
    const int32_t leftover_y = in_rows - y;

    if(leftover_y != 0)
    {
        const T *in_ptr = &mtx_in[y * in_cols];

        for(int32_t x = 0; x < in_cols; x++)
        {
            T tmp[4] = { 0, 0, 0, 0 };

            for(int32_t k = 0; k < leftover_y; k++)
            {
                tmp[k] = in_ptr[k * in_cols + x];
            }
            T *dst = &mtx_ref[static_cast<size_t>(x * 4.f) + static_cast<size_t>(std::ceil(y / 4.f)) * out_stride];
            memcpy(dst, tmp, sizeof(T) * 4);
        }
    }

    return out;
}

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
