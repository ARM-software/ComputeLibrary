/*
 * Copyright (c) 2020-2021, 2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_LOGICAL_FIXTURE
#define ARM_COMPUTE_TEST_LOGICAL_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Logical.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class LogicalOperationValidationFixtureBase : public framework::Fixture
{
protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        constexpr auto zero              = static_cast<uint8_t>(0);
        constexpr auto one               = static_cast<uint8_t>(0x1);
        constexpr auto mixed             = static_cast<uint8_t>(0xAA);
        constexpr auto mixed_bitwise_not = static_cast<uint8_t>((~0xAA));

        library->fill_static_values(tensor, i == 0 ?
                                    std::vector<uint8_t> { zero, one, zero, one, mixed, zero, mixed } :
                                    std::vector<uint8_t> { zero, zero, one, one, zero, mixed, mixed_bitwise_not });
    }

    void allocate_tensor(std::initializer_list<TensorType *> tensors)
    {
        for(auto t : tensors)
        {
            ARM_COMPUTE_ASSERT(t->info()->is_resizable());
            t->allocator()->allocate();
            ARM_COMPUTE_ASSERT(!t->info()->is_resizable());
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename T>
using LogicalBinaryRefFunctionPtrType = SimpleTensor<T>(const SimpleTensor<T> &, const SimpleTensor<T> &);

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, LogicalBinaryRefFunctionPtrType<T> RefFunction>
class LogicalBinaryOperationValidationFixture : public LogicalOperationValidationFixtureBase<TensorType, AccessorType, FunctionType, T>
{
    using Parent = LogicalOperationValidationFixtureBase<TensorType, AccessorType, FunctionType, T>;

public:
    void setup(TensorShape shape0, TensorShape shape1)
    {
        Parent::_target    = compute_target(shape0, shape1);
        Parent::_reference = compute_reference(shape0, shape1);
    }

private:
    TensorType compute_target(const TensorShape &shape0, const TensorShape &shape1)
    {
        TensorType src0 = create_tensor<TensorType>(shape0, _data_type);
        TensorType src1 = create_tensor<TensorType>(shape1, _data_type);
        TensorType dst  = create_tensor<TensorType>(TensorShape::broadcast_shape(shape0, shape1), _data_type);

        FunctionType logical_binary_op;

        logical_binary_op.configure(&src0, &src1, &dst);

        Parent::allocate_tensor({ &src0, &src1, &dst });

        Parent::fill(AccessorType(src0), 0);
        Parent::fill(AccessorType(src1), 1);

        logical_binary_op.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape0, const TensorShape &shape1)
    {
        // Create reference
        SimpleTensor<T> src0{ shape0, _data_type };
        SimpleTensor<T> src1{ shape1, _data_type };

        // Fill reference
        Parent::fill(src0, 0);
        Parent::fill(src1, 1);

        return RefFunction(src0, src1);
    }

    static constexpr auto _data_type = DataType::U8;
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
using LogicalOrValidationFixture = LogicalBinaryOperationValidationFixture<TensorType, AccessorType, FunctionType, T, &reference::logical_or<T>>;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
using LogicalAndValidationFixture = LogicalBinaryOperationValidationFixture<TensorType, AccessorType, FunctionType, T, &reference::logical_and<T>>;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class LogicalNotValidationFixture : public LogicalOperationValidationFixtureBase<TensorType, AccessorType, FunctionType, T>
{
    using Parent = LogicalOperationValidationFixtureBase<TensorType, AccessorType, FunctionType, T>;

public:
    void setup(TensorShape shape, DataType data_type)
    {
        Parent::_target    = compute_target(shape, data_type);
        Parent::_reference = compute_reference(shape, data_type);
    }

private:
    TensorType compute_target(const TensorShape &shape, DataType data_type)
    {
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        FunctionType logical_not;

        logical_not.configure(&src, &dst);

        Parent::allocate_tensor({ &src, &dst });

        Parent::fill(AccessorType(src), 0);

        logical_not.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        Parent::fill(src, 0);

        return reference::logical_not<T>(src);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LOGICAL_FIXTURE */
