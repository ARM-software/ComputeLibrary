/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_REDUCTION_OPERATION_FIXTURE
#define ARM_COMPUTE_TEST_REDUCTION_OPERATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ReductionOperation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReductionOperationValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info)
    {
        const TensorShape output_shape = get_output_shape(shape, axis);
        _target                        = compute_target(shape, output_shape, data_type, axis, op, quantization_info);
        _reference                     = compute_reference(shape, output_shape, data_type, axis, op, quantization_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            std::pair<int, int> bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
            std::uniform_int_distribution<uint8_t> distribution(bounds.first, bounds.second);

            library->fill(tensor, distribution, 0);
        }
    }

    TensorType compute_target(const TensorShape &src_shape, const TensorShape &dst_shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(src_shape, data_type, 1, quantization_info);
        TensorType dst = create_tensor<TensorType>(dst_shape, data_type, 1, quantization_info);

        // Create and configure function
        FunctionType reduction_func;
        reduction_func.configure(&src, &dst, axis, op);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        reduction_func.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &src_shape, const TensorShape &dst_shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info)
    {
        // Create reference
        SimpleTensor<T> src{ src_shape, data_type, 1, quantization_info };

        // Fill reference
        fill(src);

        return reference::reduction_operation<T, T>(src, dst_shape, axis, op);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};

private:
    TensorShape get_output_shape(TensorShape shape, unsigned int axis)
    {
        TensorShape output_shape(shape);
        output_shape.set(axis, 1);
        return output_shape;
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReductionOperationQuantizedFixture : public ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info = QuantizationInfo())
    {
        ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, axis, op, quantization_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReductionOperationFixture : public ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, unsigned int axis, ReductionOperation op)
    {
        ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, axis, op, QuantizationInfo());
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REDUCTION_OPERATION_FIXTURE */
