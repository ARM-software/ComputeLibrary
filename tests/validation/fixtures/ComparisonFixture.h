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
#ifndef ARM_COMPUTE_TEST_COMPARISON_FIXTURE
#define ARM_COMPUTE_TEST_COMPARISON_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Comparisons.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ComparisonValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(ComparisonOperation op, const TensorShape &shape0, const TensorShape &shape1, DataType data_type, QuantizationInfo qinfo0, QuantizationInfo qinfo1)
    {
        _target    = compute_target(op, shape0, shape1, data_type, qinfo0, qinfo1);
        _reference = compute_reference(op, shape0, shape1, data_type, qinfo0, qinfo1);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(ComparisonOperation op,
                              const TensorShape &shape0, const TensorShape &shape1, DataType data_type,
                              QuantizationInfo qinfo0, QuantizationInfo qinfo1)
    {
        // Create tensors
        TensorType ref_src1 = create_tensor<TensorType>(shape0, data_type, 1, qinfo0);
        TensorType ref_src2 = create_tensor<TensorType>(shape1, data_type, 1, qinfo1);
        TensorType dst      = create_tensor<TensorType>(TensorShape::broadcast_shape(shape0, shape1), DataType::U8);

        // Create and configure function
        FunctionType comp_op;
        comp_op.configure(&ref_src1, &ref_src2, &dst, op);

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
        comp_op.run();

        return dst;
    }

    SimpleTensor<uint8_t> compute_reference(ComparisonOperation op,
                                            const TensorShape &shape0, const TensorShape &shape1, DataType data_type,
                                            QuantizationInfo qinfo0, QuantizationInfo qinfo1)
    {
        // Create reference
        SimpleTensor<T> ref_src1{ shape0, data_type, 1, qinfo0 };
        SimpleTensor<T> ref_src2{ shape1, data_type, 1, qinfo1 };

        // Fill reference
        fill(ref_src1, 0);
        fill(ref_src2, 1);

        return reference::compare<T>(op, ref_src1, ref_src2);
    }

    TensorType            _target{};
    SimpleTensor<uint8_t> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ComparisonBroadcastValidationFixture : public ComparisonValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(ComparisonOperation op, const TensorShape &shape0, const TensorShape &shape1, DataType data_type)
    {
        ComparisonValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(op, shape0, shape1, data_type, QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ComparisonValidationFixture : public ComparisonValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(ComparisonOperation op, const TensorShape &shape, DataType data_type)
    {
        ComparisonValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(op, shape, shape, data_type, QuantizationInfo(), QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ComparisonValidationQuantizedFixture : public ComparisonValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(ComparisonOperation op, const TensorShape &shape, DataType data_type, QuantizationInfo qinfo0, QuantizationInfo qinfo1)

    {
        ComparisonValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(op, shape, shape, data_type, qinfo0, qinfo1);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COMPARISON_FIXTURE */
