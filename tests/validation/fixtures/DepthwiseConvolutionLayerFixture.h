/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE
#define ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/DepthwiseConvolutionLayer.h"

#include "utils/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional<std::is_same<typename std::decay<T>::type, uint8_t>::value, int32_t, T>::type;

public:
    template <typename...>
    void setup(TensorShape in_shape, TensorShape weights_shape, TensorShape out_shape, PadStrideInfo pad_stride_info, DataType data_type, QuantizationInfo quantization_info)
    {
        _quantization_info = quantization_info;
        _data_type         = data_type;
        const TensorShape biases_shape(weights_shape[2]);
        const DataType    bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

        _target    = compute_target(in_shape, weights_shape, biases_shape, out_shape, pad_stride_info, data_type, bias_data_type, quantization_info);
        _reference = compute_reference(in_shape, weights_shape, biases_shape, out_shape, pad_stride_info, data_type, bias_data_type, quantization_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                std::uniform_int_distribution<uint8_t> distribution(0, 10);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            case DataType::F16:
            {
                std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            {
                std::uniform_int_distribution<int32_t> distribution(-100, 100);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &biases_shape, const TensorShape &output_shape, PadStrideInfo &pad_stride_info,
                              const DataType data_type, const DataType bias_data_type, const QuantizationInfo quantization_info)
    {
        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, 0, quantization_info);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1, 0, quantization_info);
        TensorType biases  = create_tensor<TensorType>(biases_shape, bias_data_type, 1, 0, quantization_info);
        TensorType dst     = create_tensor<TensorType>(output_shape, data_type, 1, 0, quantization_info);

        // Create Depthwise Convolution configure function
        FunctionType dwc;
        dwc.configure(&src, &weights, &biases, &dst, pad_stride_info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(biases.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        biases.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!biases.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(weights), 1);
        fill(AccessorType(biases), 2);

        // Compute function
        dwc.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &in_shape, const TensorShape &weights_shape, const TensorShape &biases_shape, const TensorShape &out_shape, const PadStrideInfo &pad_stride_info,
                                      const DataType data_type, const DataType bias_data_type, QuantizationInfo quantization_info)
    {
        SimpleTensor<T>     src{ in_shape, data_type, 1, 0, quantization_info };
        SimpleTensor<T>     weights{ weights_shape, data_type, 1, 0, quantization_info };
        SimpleTensor<TBias> biases{ biases_shape, bias_data_type, 1, 0, quantization_info };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        return reference::depthwise_convolution(src, weights, biases, out_shape, pad_stride_info);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type{};
    QuantizationInfo _quantization_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, TensorShape weights_shape, TensorShape out_shape, PadStrideInfo pad_stride_info, DataType data_type)
    {
        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(in_shape, weights_shape, out_shape, pad_stride_info,
                                                                                                            data_type, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerValidationQuantizedFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, TensorShape weights_shape, TensorShape out_shape, PadStrideInfo pad_stride_info, DataType data_type, QuantizationInfo quantization_info)
    {
        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(in_shape, weights_shape, out_shape, pad_stride_info,
                                                                                                            data_type, quantization_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE */
