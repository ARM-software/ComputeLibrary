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
#ifndef ARM_COMPUTE_TEST_FIXED_POINT_FIXTURE
#define ARM_COMPUTE_TEST_FIXED_POINT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/FixedPoint.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename T>
class FixedPointValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt, FixedPointOp op, int fractional_bits)
    {
        _fractional_bits = fractional_bits;
        _target          = compute_target(shape, dt, op, fractional_bits);
        _reference       = compute_reference(shape, dt, op, fractional_bits);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int min, int max, int i)
    {
        std::uniform_int_distribution<> distribution(min, max);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape, DataType dt, FixedPointOp op, int fixed_point_position)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, dt, 1, fixed_point_position);
        TensorType dst = create_tensor<TensorType>(shape, dt, 1, fixed_point_position);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        compute_target_impl<TensorType, AccessorType, T>(shape, dt, op, fixed_point_position, src, dst);

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType dt, FixedPointOp op, int fixed_point_position)
    {
        // Create reference
        SimpleTensor<T> src{ shape, dt, 1, fixed_point_position };

        // Fill reference
        int min = 0;
        int max = 0;
        switch(op)
        {
            case FixedPointOp::EXP:
                min = -(1 << (fixed_point_position - 1));
                max = (1 << (fixed_point_position - 1));
                break;
            case FixedPointOp::INV_SQRT:
                min = 1;
                max = (dt == DataType::QS8) ? 0x7F : 0x7FFF;
                break;
            case FixedPointOp::LOG:
                min = (1 << (fixed_point_position - 1));
                max = (dt == DataType::QS8) ? 0x3F : 0x3FFF;
                break;
            case FixedPointOp::RECIPROCAL:
                min = 15;
                max = (dt == DataType::QS8) ? 0x7F : 0x7FFF;
                break;
            default:
                ARM_COMPUTE_ERROR("Fixed point operation not supported");
                break;
        }
        fill(src, min, max, 0);

        return reference::fixed_point_operation<T>(src, op);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    int             _fractional_bits{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FIXED_POINT_FIXTURE */
