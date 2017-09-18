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
 * dst OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "FixedPointPixelWiseMultiplication.h"

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
SimpleTensor<T> fixed_point_pixel_wise_multiplication(const SimpleTensor<T> &src1, const SimpleTensor<T> &src2, float scale, ConvertPolicy convert_policy)
{
    using namespace fixed_point_arithmetic;

    SimpleTensor<T> dst(src2.shape(), src2.data_type(), 1, src2.fixed_point_position());

    const int fixed_point_position = src1.fixed_point_position();

    ARM_COMPUTE_ERROR_ON_MSG(src1.data_type() != src2.data_type() || src1.data_type() != dst.data_type(),
                             "Tensors must all have the same DataType");
    ARM_COMPUTE_ERROR_ON_MSG(fixed_point_position != src2.fixed_point_position() || fixed_point_position != dst.fixed_point_position(),
                             "Fixed-point position must be the same for both inputs and outputs");

    // Validate fixed_point_position
    ARM_COMPUTE_ERROR_ON((src1.data_type() == DataType::QS8) && (fixed_point_position == 0 || fixed_point_position > 7));
    ARM_COMPUTE_ERROR_ON((src1.data_type() == DataType::QS16) && (fixed_point_position == 0 || fixed_point_position > 15));

    const fixed_point<T> fp_scale(scale, fixed_point_position);
    const bool           is_sat = convert_policy == ConvertPolicy::SATURATE;

    for(int i = 0; i < src1.num_elements(); ++i)
    {
        const fixed_point<T> val1(src1[i], fixed_point_position, true);
        fixed_point<T>       res(src2[i], fixed_point_position, true);
        if(is_sat)
        {
            res = mul(mul(res, val1), fp_scale);
        }
        else
        {
            res = mul<OverflowPolicy::WRAP>(mul<OverflowPolicy::WRAP>(res, val1), fp_scale);
        }
        dst[i] = res.raw();
    }

    return dst;
}

// *INDENT-OFF*
// clang-format off
template SimpleTensor<qint8_t> fixed_point_pixel_wise_multiplication(const SimpleTensor<qint8_t> &src1, const SimpleTensor<qint8_t> &src2, float scale, ConvertPolicy convert_policy);
template SimpleTensor<qint16_t> fixed_point_pixel_wise_multiplication(const SimpleTensor<qint16_t> &src1, const SimpleTensor<qint16_t> &src2, float scale, ConvertPolicy convert_policy);
// *INDENT-ON*
// clang-format on

} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
