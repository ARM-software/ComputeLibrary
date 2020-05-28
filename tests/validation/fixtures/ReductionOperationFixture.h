/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
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
    void setup(TensorShape shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info, bool keep_dims = false)
    {
        const bool is_arg_min_max = (op == ReductionOperation::ARG_IDX_MAX) || (op == ReductionOperation::ARG_IDX_MIN);
        _keep_dims                = keep_dims && !is_arg_min_max;

        const TensorShape output_shape = arm_compute::misc::shape_calculator::compute_reduced_shape(shape, axis, _keep_dims);

        _target    = compute_target(shape, data_type, axis, op, quantization_info);
        _reference = compute_reference(shape, output_shape, data_type, axis, op, quantization_info);
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
            if(tensor.data_type() == DataType::QASYMM8)
            {
                std::pair<int, int> bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<uint8_t> distribution(bounds.first, bounds.second);

                library->fill(tensor, distribution, 0);
            }
            else if(tensor.data_type() == DataType::QASYMM8_SIGNED)
            {
                std::pair<int, int> bounds = get_quantized_qasymm8_signed_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<int8_t> distribution(bounds.first, bounds.second);

                library->fill(tensor, distribution, 0);
            }
            else
            {
                ARM_COMPUTE_ERROR("Not supported");
            }
        }
    }

    TensorType compute_target(const TensorShape &src_shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(src_shape, data_type, 1, quantization_info);
        TensorType dst;

        // Create and configure function
        FunctionType reduction_func;
        reduction_func.configure(&src, &dst, axis, op, _keep_dims);

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
    bool _keep_dims{ false };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReductionOperationQuantizedFixture : public ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, unsigned int axis, ReductionOperation op, QuantizationInfo quantization_info = QuantizationInfo(), bool keep_dims = false)
    {
        ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, axis, op, quantization_info, keep_dims);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReductionOperationFixture : public ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, unsigned int axis, ReductionOperation op, bool keep_dims = false)
    {
        ReductionOperationValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, axis, op, QuantizationInfo(), keep_dims);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REDUCTION_OPERATION_FIXTURE */
