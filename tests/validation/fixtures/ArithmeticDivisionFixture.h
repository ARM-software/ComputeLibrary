/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_ARITHMETIC_DIVISION_FIXTURE
#define ARM_COMPUTE_TEST_ARITHMETIC_DIVISION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ArithmeticDivision.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionBroadcastValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const TensorShape &shape0, const TensorShape &shape1, DataType data_type)
    {
        _target    = compute_target(shape0, shape1, data_type);
        _reference = compute_reference(shape0, shape1, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(1.0f), T(5.0f) };
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1, DataType data_type)
    {
        // Create tensors
        TensorType ref_src1 = create_tensor<TensorType>(shape0, data_type, 1);
        TensorType ref_src2 = create_tensor<TensorType>(shape1, data_type, 1);
        TensorType dst      = create_tensor<TensorType>(TensorShape::broadcast_shape(shape0, shape1), data_type);

        // Create and configure function
        FunctionType add;
        add.configure(&ref_src1, &ref_src2, &dst);

        ARM_COMPUTE_ASSERT(ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(ref_src2.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        ref_src1.allocator()->allocate();
        ref_src2.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!ref_src1.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!ref_src2.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(ref_src1), 0);
        fill(AccessorType(ref_src2), 1);

        // Compute function
        add.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> ref_src1{ shape0, data_type, 1 };
        SimpleTensor<T> ref_src2{ shape1, data_type, 1 };

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        return reference::arithmetic_division<T>(ref_src1, ref_src2, data_type);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ArithmeticDivisionValidationFixture : public ArithmeticDivisionBroadcastValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        ArithmeticDivisionBroadcastValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, shape, data_type);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ARITHMETIC_DIVISION_FIXTURE */
