/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_REDUCE_MEAN_FIXTURE
#define ARM_COMPUTE_TEST_REDUCE_MEAN_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ReductionOperation.h"
#include "tests/validation/reference/ReshapeLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReduceMeanValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, Coordinates axis, bool keep_dims, QuantizationInfo quantization_info)
    {
        _target    = compute_target(shape, data_type, axis, keep_dims, quantization_info);
        _reference = compute_reference(shape, data_type, axis, keep_dims, quantization_info);
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
            std::uniform_int_distribution<> distribution(bounds.first, bounds.second);

            library->fill(tensor, distribution, 0);
        }
    }

    TensorType compute_target(TensorShape &src_shape, DataType data_type, Coordinates axis, bool keep_dims, QuantizationInfo quantization_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(src_shape, data_type, 1, quantization_info);
        TensorType dst;

        // Create and configure function
        FunctionType reduction_mean;
        reduction_mean.configure(&src, axis, keep_dims, &dst);

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
        reduction_mean.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(TensorShape &src_shape, DataType data_type, Coordinates axis, bool keep_dims, QuantizationInfo quantization_info)
    {
        // Create reference
        SimpleTensor<T> src{ src_shape, data_type, 1, quantization_info };

        // Fill reference
        fill(src);

        SimpleTensor<T> out;
        for(unsigned int i = 0; i < axis.num_dimensions(); ++i)
        {
            TensorShape output_shape = i == 0 ? src_shape : out.shape();
            output_shape.set(axis[i], 1);
            out = reference::reduction_operation<T, T>(i == 0 ? src : out, output_shape, axis[i], ReductionOperation::MEAN_SUM);
        }

        if(!keep_dims)
        {
            TensorShape output_shape = src_shape;
            std::sort(axis.begin(), axis.begin() + axis.num_dimensions());
            for(unsigned int i = 0; i < axis.num_dimensions(); ++i)
            {
                output_shape.remove_dimension(axis[i] - i);
            }

            out = reference::reshape_layer(out, output_shape);
        }
        return out;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReduceMeanQuantizedFixture : public ReduceMeanValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, Coordinates axis, bool keep_dims, QuantizationInfo quantization_info = QuantizationInfo())
    {
        ReduceMeanValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, axis, keep_dims, quantization_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReduceMeanFixture : public ReduceMeanValidationFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, Coordinates axis, bool keep_dims)
    {
        ReduceMeanValidationFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, data_type, axis, keep_dims, QuantizationInfo());
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REDUCE_MEAN_FIXTURE */
