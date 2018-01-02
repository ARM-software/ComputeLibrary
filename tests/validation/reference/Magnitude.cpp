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
#include "Magnitude.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T>
SimpleTensor<T> magnitude(const SimpleTensor<T> &gx, const SimpleTensor<T> &gy, MagnitudeType magnitude_type)
{
    SimpleTensor<T> mag(gx.shape(), gx.data_type());

    using intermediate_type = typename common_promoted_unsigned_type<T>::intermediate_type;

    for(int i = 0; i < gx.num_elements(); ++i)
    {
        double val = 0.f;

        if(magnitude_type == MagnitudeType::L1NORM)
        {
            val = static_cast<intermediate_type>(std::abs(gx[i])) + static_cast<intermediate_type>(std::abs(gy[i]));
        }
        else // MagnitudeType::L2NORM
        {
            // Note: kernel saturates to uint32_t instead of intermediate_type for S32 format
            auto sum = static_cast<uint32_t>(gx[i] * gx[i]) + static_cast<uint32_t>(gy[i] * gy[i]);
            val      = std::sqrt(sum) + 0.5f;
        }

        mag[i] = saturate_cast<T>(val);
    }

    return mag;
}

template SimpleTensor<int16_t> magnitude(const SimpleTensor<int16_t> &gx, const SimpleTensor<int16_t> &gy, MagnitudeType magnitude_type);
template SimpleTensor<int32_t> magnitude(const SimpleTensor<int32_t> &gx, const SimpleTensor<int32_t> &gy, MagnitudeType magnitude_type);
template SimpleTensor<half_float::half> magnitude(const SimpleTensor<half_float::half> &gx, const SimpleTensor<half_float::half> &gy, MagnitudeType magnitude_type);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
