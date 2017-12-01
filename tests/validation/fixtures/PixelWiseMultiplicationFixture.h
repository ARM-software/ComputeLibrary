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
#ifndef ARM_COMPUTE_TEST_PIXEL_WISE_MULTIPLICATION_FIXTURE
#define ARM_COMPUTE_TEST_PIXEL_WISE_MULTIPLICATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/PixelWiseMultiplication.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class PixelWiseMultiplicationValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape    shape,
               DataType       dt_in1,
               DataType       dt_in2,
               float          scale,
               ConvertPolicy  convert_policy,
               RoundingPolicy rounding_policy)
    {
        _target    = compute_target(shape, dt_in1, dt_in2, scale, convert_policy, rounding_policy);
        _reference = compute_reference(shape, dt_in1, dt_in2, scale, convert_policy, rounding_policy);
    }

protected:
    template <typename U>
    void fill(U &&tensor, unsigned int seed_offset)
    {
        library->fill_tensor_uniform(tensor, seed_offset);
    }

    TensorType compute_target(const TensorShape &shape, DataType dt_in1, DataType dt_in2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
    {
        // Create tensors
        TensorType src1 = create_tensor<TensorType>(shape, dt_in1);
        TensorType src2 = create_tensor<TensorType>(shape, dt_in2);
        TensorType dst  = create_tensor<TensorType>(shape, dt_in2);

        // Create and configure function
        FunctionType multiply;
        multiply.configure(&src1, &src2, &dst, scale, convert_policy, rounding_policy);

        ARM_COMPUTE_EXPECT(src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src1), 0);
        fill(AccessorType(src2), 1);

        // Compute function
        multiply.run();

        return dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape, DataType dt_in1, DataType dt_in2, float scale, ConvertPolicy convert_policy, RoundingPolicy rounding_policy)
    {
        // Create reference
        SimpleTensor<T1> src1{ shape, dt_in1 };
        SimpleTensor<T2> src2{ shape, dt_in2 };

        // Fill reference
        fill(src1, 0);
        fill(src2, 1);

        return reference::pixel_wise_multiplication<T1, T2>(src1, src2, scale, convert_policy, rounding_policy);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_PIXEL_WISE_MULTIPLICATION_FIXTURE */
