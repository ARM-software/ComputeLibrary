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
#ifndef ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/PoolingLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingLayerInfo pool_info, DataType data_type, int fractional_bits, QuantizationInfo quantization_info)
    {
        _fractional_bits   = fractional_bits;
        _quantization_info = quantization_info;
        _pool_info         = pool_info;

        _target    = compute_target(shape, pool_info, data_type, fractional_bits, quantization_info);
        _reference = compute_reference(shape, pool_info, data_type, fractional_bits, quantization_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        if(!is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_real_distribution<> distribution(-1.f, 1.f);
            library->fill(tensor, distribution, 0);
        }
        else if(is_data_type_quantized_asymmetric(tensor.data_type()))
        {
            library->fill_tensor_uniform(tensor, 0);
        }
        else
        {
            const int                       one_fixed = 1 << _fractional_bits;
            std::uniform_int_distribution<> distribution(-one_fixed, one_fixed);
            library->fill(tensor, distribution, 0);
        }
    }

    TensorType compute_target(const TensorShape &shape, PoolingLayerInfo info,
                              DataType data_type, int fixed_point_position, QuantizationInfo quantization_info)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 1, fixed_point_position, quantization_info);
        TensorType dst;

        // Create and configure function
        FunctionType pool_layer;
        pool_layer.configure(&src, &dst, info);

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
        pool_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, PoolingLayerInfo info,
                                      DataType data_type, int fixed_point_position, QuantizationInfo quantization_info)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 1, fixed_point_position, quantization_info };

        // Fill reference
        fill(src);

        return reference::pooling_layer<T>(src, info);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    int              _fractional_bits{};
    QuantizationInfo _quantization_info{};
    PoolingLayerInfo _pool_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, int pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, pad_stride_info, exclude_padding),
                                                                                               data_type, 0, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationFixedPointFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, int pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, int fractional_bits)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, pad_stride_info, exclude_padding),
                                                                                               data_type, fractional_bits, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class PoolingLayerValidationQuantizedFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, int pool_size, PadStrideInfo pad_stride_info, bool exclude_padding, DataType data_type, QuantizationInfo quantization_info)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type, pool_size, pad_stride_info, exclude_padding),
                                                                                               data_type, 0, quantization_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GlobalPoolingLayerValidationFixture : public PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, PoolingType pool_type, DataType data_type)
    {
        PoolingLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape, PoolingLayerInfo(pool_type), data_type, 0, QuantizationInfo());
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_POOLING_LAYER_FIXTURE */
