/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_FFT_FIXTURE
#define ARM_COMPUTE_TEST_FFT_FIXTURE

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/DFT.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename InfoType, typename T>
class FFTValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        _target    = compute_target(shape, data_type);
        _reference = compute_reference(shape, data_type);
        ARM_COMPUTE_ERROR_ON_MISMATCHING_DIMENSIONS(_target.info()->tensor_shape(), _reference.shape());
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        std::uniform_real_distribution<float> distribution(-5.f, 5.f);
        library->fill(tensor, distribution, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type, 2);
        TensorType dst = create_tensor<TensorType>(shape, data_type, 2);

        // Create and configure function
        FunctionType fft;
        fft.configure(&src, &dst, InfoType());

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
        fft.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type, 2 };

        // Fill reference
        fill(src);
        if(std::is_same<InfoType, FFT1DInfo>::value)
        {
            return reference::dft_1d(src, reference::FFTDirection::Forward);
        }
        else
        {
            return reference::dft_2d(src, reference::FFTDirection::Forward);
        }
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FFTConvolutionValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation,
               DataType data_type, DataLayout data_layout, ActivationLayerInfo act_info)
    {
        _data_type   = data_type;
        _data_layout = data_layout;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, dilation, act_info);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, dilation, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, const TensorShape &bias_shape, TensorShape output_shape, const PadStrideInfo &info,
                              const Size2D &dilation, const ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_UNUSED(dilation);
        ARM_COMPUTE_ERROR_ON((input_shape[2] % weights_shape[2]) != 0);

        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _data_type, 1, QuantizationInfo(), _data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, _data_type, 1, QuantizationInfo(), _data_layout);

        // Create and configure function
        FunctionType conv;
        conv.configure(&src, &weights, &bias, &dst, info, act_info);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!bias.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(weights), 1);
        fill(AccessorType(bias), 2);

        // Compute convolution function
        conv.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info,
                                      const Size2D &dilation, const ActivationLayerInfo act_info)
    {
        ARM_COMPUTE_ERROR_ON((input_shape[2] % weights_shape[2]) != 0);

        // Create reference
        SimpleTensor<T> src{ input_shape, _data_type, 1 };
        SimpleTensor<T> weights{ weights_shape, _data_type, 1 };
        SimpleTensor<T> bias{ bias_shape, _data_type, 1 };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        return (act_info.enabled()) ? reference::activation_layer<T>(reference::convolution_layer<T>(src, weights, bias, output_shape, info, dilation), act_info) : reference::convolution_layer<T>(src,
                weights, bias, output_shape, info, dilation);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    DataType        _data_type{};
    DataLayout      _data_layout{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FFTConvolutionValidationFixture : public FFTConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation,
               DataType data_type, DataLayout data_layout, ActivationLayerInfo act_info)
    {
        FFTConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation,
                                                                                                 data_type, data_layout, act_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FFT_FIXTURE */
