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
#include "FixedPoint.h"

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
template <typename T>
SimpleTensor<T> fixed_point_operation(const SimpleTensor<T> &src, FixedPointOp op)
{
    SimpleTensor<T> result(src.shape(), src.data_type());

    const int p = src.fixed_point_position();
    switch(op)
    {
        case FixedPointOp::EXP:
            for(int i = 0; i < src.num_elements(); ++i)
            {
                result[i] = fixed_point_arithmetic::exp(fixed_point_arithmetic::fixed_point<T>(src[i], p, true)).raw();
            }
            break;
        case FixedPointOp::LOG:
            for(int i = 0; i < src.num_elements(); ++i)
            {
                result[i] = fixed_point_arithmetic::log(fixed_point_arithmetic::fixed_point<T>(src[i], p, true)).raw();
            }
            break;
        case FixedPointOp::INV_SQRT:
            for(int i = 0; i < src.num_elements(); ++i)
            {
                result[i] = fixed_point_arithmetic::inv_sqrt(fixed_point_arithmetic::fixed_point<T>(src[i], p, true)).raw();
            }
            break;
        case FixedPointOp::RECIPROCAL:
            for(int i = 0; i < src.num_elements(); ++i)
            {
                result[i] = fixed_point_arithmetic::div(fixed_point_arithmetic::fixed_point<T>(1, p), fixed_point_arithmetic::fixed_point<T>(src[i], p, true)).raw();
            }
            break;
        default:
            ARM_COMPUTE_ERROR("Fixed point operation not supported");
            break;
    }

    return result;
}

template SimpleTensor<int8_t> fixed_point_operation(const SimpleTensor<int8_t> &src, FixedPointOp op);
template SimpleTensor<int16_t> fixed_point_operation(const SimpleTensor<int16_t> &src, FixedPointOp op);
} // namespace reference
} // namespace validation
} // namespace test
} // namespace arm_compute
