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
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/UpsampleLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class UpsampleLayerFixtureBase : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType data_type, DataLayout data_layout,
               Size2D info, const InterpolationPolicy &policy, QuantizationInfo quantization_info)
    {
        _data_type = data_type;

        _target    = compute_target(input_shape, info, policy, data_type, data_layout, quantization_info);
        _reference = compute_reference(input_shape, info, policy, data_type, quantization_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        if(_data_type == DataType::QASYMM8)
        {
            const auto                             bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
            std::uniform_int_distribution<uint8_t> distribution(bounds.first, bounds.second);
            library->fill(tensor, distribution, i);
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(TensorShape input_shape, const Size2D &info, const InterpolationPolicy &policy,
                              DataType data_type, DataLayout data_layout, QuantizationInfo quantization_info)
    {
        TensorShape output_shape(input_shape);
        output_shape.set(0, info.x() * input_shape[0]);
        output_shape.set(1, info.y() * input_shape[1]);

        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type, 1, quantization_info, data_layout);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type, 1, quantization_info, data_layout);

        // Create and configure function
        FunctionType upsample;
        upsample.configure(&src, &dst, info, policy);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute DeconvolutionLayer function
        upsample.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const Size2D &info, const InterpolationPolicy &policy,
                                      DataType data_type, QuantizationInfo quantization_info)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, quantization_info };

        // Fill reference
        fill(src, 0);

        return reference::upsample_layer<T>(src, info, policy);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class UpsampleLayerFixture : public UpsampleLayerFixtureBase<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType data_type, DataLayout data_layout,
               Size2D info, const InterpolationPolicy &policy)
    {
        UpsampleLayerFixtureBase<TensorType, AccessorType, FunctionType, T>::setup(input_shape, data_type, data_layout,
                                                                                   info, policy, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class UpsampleLayerQuantizedFixture : public UpsampleLayerFixtureBase<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType data_type, DataLayout data_layout,
               Size2D info, const InterpolationPolicy &policy, QuantizationInfo quantization_info)
    {
        UpsampleLayerFixtureBase<TensorType, AccessorType, FunctionType, T>::setup(input_shape, data_type, data_layout,
                                                                                   info, policy, quantization_info);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
