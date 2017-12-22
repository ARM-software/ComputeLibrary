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
#ifndef ARM_COMPUTE_TEST_ABSOLUTE_DIFFERENCE_FIXTURE
#define ARM_COMPUTE_TEST_ABSOLUTE_DIFFERENCE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/AbsoluteDifference.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class AbsoluteDifferenceValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type0, DataType data_type1, DataType output_data_type)
    {
        _target    = compute_target(shape, data_type0, data_type1, output_data_type);
        _reference = compute_reference(shape, data_type0, data_type1, output_data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type)
    {
        // Create tensors
        TensorType ref_src1 = create_tensor<TensorType>(shape, data_type0, 1);
        TensorType ref_src2 = create_tensor<TensorType>(shape, data_type1, 1);
        TensorType dst      = create_tensor<TensorType>(shape, output_data_type, 1);

        // Create and configure function
        FunctionType abs_diff;
        abs_diff.configure(&ref_src1, &ref_src2, &dst);

        ARM_COMPUTE_EXPECT(ref_src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(ref_src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!ref_src1.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!ref_src2.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(ref_src1), 0);
        fill(AccessorType(ref_src2), 1);

        // Compute function
        abs_diff.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type0, DataType data_type1, DataType output_data_type)
    {
        // Create reference
        SimpleTensor<T> ref_src1{ shape, data_type0, 1 };
        SimpleTensor<T> ref_src2{ shape, data_type1, 1 };

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        return reference::absolute_difference<T>(ref_src1, ref_src2, output_data_type);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ABSOLUTE_DIFFERENCE_FIXTURE */
