/*
 * Copyright (c) 2018-2023 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_ARG_MIN_MAX_FIXTURE
#define ARM_COMPUTE_TEST_ARG_MIN_MAX_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ReductionOperation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class ArgMinMaxValidationBaseFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, DataType input_type, DataType output_type, int axis, ReductionOperation op, QuantizationInfo q_info)
    {
        _target    = compute_target(shape, input_type, output_type, axis, op, q_info);
        _reference = compute_reference(shape, input_type, output_type, axis, op, q_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, 0);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, 0);
                break;
            }
            case DataType::S32:
            {
                std::uniform_int_distribution<int32_t> distribution(-100, 100);
                library->fill(tensor, distribution, 0);
                break;
            }
            case DataType::QASYMM8:
            {
                std::pair<int, int> bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<uint32_t> distribution(bounds.first, bounds.second);

                library->fill(tensor, distribution, 0);
                break;
            }
            case DataType::QASYMM8_SIGNED:
            {
                std::pair<int, int> bounds = get_quantized_qasymm8_signed_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<int32_t> distribution(bounds.first, bounds.second);

                library->fill(tensor, distribution, 0);
                break;
            }
            default:
                ARM_COMPUTE_ERROR("DataType for Elementwise Negation Not implemented");
        }
    }

    TensorType compute_target(TensorShape &src_shape, DataType input_type, DataType output_type, int axis, ReductionOperation op, QuantizationInfo q_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(src_shape, input_type, 1, q_info);
        TensorType dst = create_tensor<TensorType>(compute_output_shape(src_shape, axis), output_type, 1, q_info);

        // Create and configure function
        FunctionType arg_min_max_layer;
        arg_min_max_layer.configure(&src, axis, &dst, op);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        arg_min_max_layer.run();

        return dst;
    }

    TensorShape compute_output_shape(const TensorShape &src_shape, int axis)
    {
        return arm_compute::misc::shape_calculator::compute_reduced_shape(src_shape, axis, false);
    }

    SimpleTensor<T2> compute_reference(TensorShape &src_shape, DataType input_type, DataType output_type, int axis, ReductionOperation op, QuantizationInfo q_info)
    {
        // Create reference
        SimpleTensor<T1> src{ src_shape, input_type, 1, q_info };

        // Fill reference
        fill(src);

        return reference::reduction_operation<T1, T2>(src, compute_output_shape(src_shape, axis), axis, op, output_type);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class ArgMinMaxValidationQuantizedFixture : public ArgMinMaxValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    void setup(const TensorShape &shape, DataType input_type, DataType output_type, int axis, ReductionOperation op, QuantizationInfo quantization_info)
    {
        ArgMinMaxValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, input_type, output_type, axis, op, quantization_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class ArgMinMaxValidationFixture : public ArgMinMaxValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    void setup(const TensorShape &shape, DataType input_type, DataType output_type, int axis, ReductionOperation op)
    {
        ArgMinMaxValidationBaseFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, input_type, output_type, axis, op, QuantizationInfo());
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_ARG_MIN_MAX_FIXTURE */
