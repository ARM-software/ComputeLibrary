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
#include "LaplacianReconstruct.h"

#include "arm_compute/core/Types.h"
#include "tests/validation/reference/ArithmeticOperations.h"
#include "tests/validation/reference/DepthConvertLayer.h"
#include "tests/validation/reference/Scale.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace reference
{
template <typename T, typename U>
SimpleTensor<U> laplacian_reconstruct(const std::vector<SimpleTensor<T>> &pyramid, const SimpleTensor<T> &low_res, BorderMode border_mode, T constant_border_value)
{
    std::vector<SimpleTensor<T>> tmp_pyramid(pyramid);

    const size_t   last_level = pyramid.size() - 1;
    const DataType data_type  = low_res.data_type();

    // input + L(n-1)
    tmp_pyramid[last_level] = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, low_res, pyramid[last_level], data_type, ConvertPolicy::SATURATE);

    // Scale levels n-1 to 1, and add levels n-2 to 0
    for(size_t i = last_level; i-- > 0;)
    {
        const float scale_x = static_cast<float>(tmp_pyramid[i].shape().x()) / tmp_pyramid[i + 1].shape().x();
        const float scale_y = static_cast<float>(tmp_pyramid[i].shape().y()) / tmp_pyramid[i + 1].shape().y();

        tmp_pyramid[i] = reference::scale(tmp_pyramid[i + 1], scale_x, scale_y, InterpolationPolicy::NEAREST_NEIGHBOR,
                                          border_mode, constant_border_value, SamplingPolicy::CENTER, false);

        tmp_pyramid[i] = reference::arithmetic_operation(reference::ArithmeticOperation::ADD, tmp_pyramid[i], pyramid[i], data_type, ConvertPolicy::SATURATE);
    }

    return reference::depth_convert<T, U>(tmp_pyramid[0], DataType::U8, ConvertPolicy::SATURATE, 0);
}

template SimpleTensor<uint8_t> laplacian_reconstruct(const std::vector<SimpleTensor<int16_t>> &pyramid, const SimpleTensor<int16_t> &low_res, BorderMode border_mode, int16_t constant_border_value);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
