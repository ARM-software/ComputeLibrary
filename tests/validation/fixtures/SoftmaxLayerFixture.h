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
#ifndef ARM_COMPUTE_TEST_SOFTMAX_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_SOFTMAX_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/SoftmaxLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SoftmaxValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, int fractional_bits, QuantizationInfo quantization_info, float beta)
    {
        _fractional_bits   = fractional_bits;
        _quantization_info = quantization_info;

        _target    = compute_target(shape, data_type, fractional_bits, quantization_info, beta);
        _reference = compute_reference(shape, data_type, fractional_bits, quantization_info, beta);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_real_distribution<> distribution(-1000.f, 1000.f);
            library->fill(tensor, distribution, 0);
        }
        else if(is_data_type_quantized_asymmetric(tensor.data_type()))
        {
            std::uniform_int_distribution<> distribution(0, 100);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            const int                       one_fixed = 1 << _fractional_bits;
            std::uniform_int_distribution<> distribution(-one_fixed, one_fixed);
            library->fill(tensor, distribution, 0);
        }
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, int fixed_point_position,
                              QuantizationInfo quantization_info, float beta)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, fixed_point_position, quantization_info);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 1, fixed_point_position, QuantizationInfo(1.f / 256, 0));

        // Create and configure function
        FunctionType smx_layer;
        smx_layer.configure(&src, &dst, beta);

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
        smx_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, int fixed_point_position,
                                      QuantizationInfo quantization_info, float beta)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, fixed_point_position, quantization_info };

        // Fill reference
        fill(src);

        return reference::softmax_layer<T>(src, beta);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    int              _fractional_bits{};
    QuantizationInfo _quantization_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SoftmaxValidationFixture : public SoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, float beta)
    {
        SoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                          data_type,
                                                                                          0,
                                                                                          QuantizationInfo(),
                                                                                          beta);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SoftmaxValidationFixedPointFixture : public SoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, int fixed_point_position)
    {
        SoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                          data_type,
                                                                                          fixed_point_position,
                                                                                          QuantizationInfo(),
                                                                                          1.0f);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SoftmaxValidationQuantizedFixture : public SoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, QuantizationInfo quantization_info, float beta)
    {
        SoftmaxValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                          data_type,
                                                                                          0,
                                                                                          quantization_info,
                                                                                          beta);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SOFTMAX_LAYER_FIXTURE */
