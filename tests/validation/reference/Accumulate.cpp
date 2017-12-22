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
#include "Accumulate.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/FixedPoint.h"
#include "tests/validation/Helpers.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T1, typename T2>
SimpleTensor<T2> accumulate(const SimpleTensor<T1> &src, DataType output_data_type)
{
    SimpleTensor<T2> dst{ src.shape(), output_data_type };

    library->fill_tensor_uniform(dst, 1, static_cast<T2>(0), static_cast<T2>(std::numeric_limits<T1>::max()));

    using intermediate_type = typename common_promoted_signed_type<T1, T2>::intermediate_type;

    for(int i = 0; i < src.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(src[i]) + static_cast<intermediate_type>(dst[i]);
        dst[i]                = saturate_cast<T2>(val);
    }

    return dst;
}

template <typename T1, typename T2>
SimpleTensor<T2> accumulate_weighted(const SimpleTensor<T1> &src, float alpha, DataType output_data_type)
{
    ARM_COMPUTE_ERROR_ON_MSG(alpha < 0.f || alpha > 1.f, "Weight (alpha) specified in accumulate_weighted must be within the range [0, 1]");

    SimpleTensor<T2> dst{ src.shape(), output_data_type };

    library->fill_tensor_uniform(dst, 1, static_cast<T2>(0), static_cast<T2>(std::numeric_limits<T1>::max()));

    using intermediate_type = typename common_promoted_signed_type<T1, T2>::intermediate_type;

    for(int i = 0; i < src.num_elements(); ++i)
    {
        double val = (1. - static_cast<double>(alpha)) * static_cast<intermediate_type>(dst[i]) + static_cast<double>(alpha) * static_cast<intermediate_type>(src[i]);
        dst[i]     = static_cast<T2>(val);
    }

    return dst;
}

template <typename T1, typename T2>
SimpleTensor<T2> accumulate_squared(const SimpleTensor<T1> &src, uint32_t shift, DataType output_data_type)
{
    ARM_COMPUTE_ERROR_ON_MSG(shift > 15, "Shift in accumulate_squared must be within the range [0, 15]");

    SimpleTensor<T2> dst{ src.shape(), output_data_type };

    library->fill_tensor_uniform(dst, 1, static_cast<T2>(0), static_cast<T2>(std::numeric_limits<T1>::max()));

    using intermediate_type = typename common_promoted_signed_type<T1, T2>::intermediate_type;
    intermediate_type denom = 1 << shift;

    for(int i = 0; i < src.num_elements(); ++i)
    {
        intermediate_type val = static_cast<intermediate_type>(dst[i]) + (static_cast<intermediate_type>(src[i]) * static_cast<intermediate_type>(src[i]) / denom);
        dst[i]                = saturate_cast<T2>(val);
    }

    return dst;
}

template SimpleTensor<int16_t> accumulate(const SimpleTensor<uint8_t> &src, DataType output_data_type);
template SimpleTensor<uint8_t> accumulate_weighted(const SimpleTensor<uint8_t> &src, float alpha, DataType output_data_type);
template SimpleTensor<int16_t> accumulate_squared(const SimpleTensor<uint8_t> &src, uint32_t shift, DataType output_data_type);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
