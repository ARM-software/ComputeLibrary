/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/DepthwiseConvolutionLayer.h"

#include "utils/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class DepthwiseConvolutionLayerValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int32_t, T >::type;

public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation,
               unsigned int depth_multiplier, DataType input_data_type, DataType weights_data_type,
               QuantizationInfo input_quantization_info, QuantizationInfo weights_quantization_info, QuantizationInfo output_quantization_info,
               DataLayout data_layout, ActivationLayerInfo act_info)
    {
        const DataType bias_data_type = is_data_type_quantized(input_data_type) ? DataType::S32 : input_data_type;

        TensorShape weights_shape(kernel_size.width, kernel_size.height);

        const TensorInfo  in_info(in_shape, 1, input_data_type);
        const TensorInfo  we_info(weights_shape, 1, weights_data_type);
        const TensorShape out_shape = compute_depthwise_convolution_shape(in_info, we_info, pad_stride_info, depth_multiplier, dilation);

        weights_shape.set(2, out_shape.z());
        const TensorShape biases_shape(weights_shape[2]);

        _target = compute_target(in_shape, weights_shape, biases_shape, out_shape, pad_stride_info, dilation, depth_multiplier,
                                 input_data_type, weights_data_type, bias_data_type, input_quantization_info, weights_quantization_info, output_quantization_info, data_layout, act_info);
        _reference = compute_reference(in_shape, weights_shape, biases_shape, out_shape, pad_stride_info, dilation, depth_multiplier,
                                       input_data_type, weights_data_type, bias_data_type, input_quantization_info, weights_quantization_info, output_quantization_info, act_info);
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
            case DataType::QASYMM8_SIGNED:
            case DataType::QSYMM8_PER_CHANNEL:
            {
                std::uniform_int_distribution<int8_t> distribution(-10, 10);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
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

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, TensorShape biases_shape, TensorShape output_shape, PadStrideInfo &pad_stride_info, Size2D dilation,
                              unsigned int depth_multiplier, const DataType input_data_type, const DataType weights_data_type, const DataType bias_data_type,
                              const QuantizationInfo &input_quantization_info, const QuantizationInfo &weights_quantization_info, const QuantizationInfo &output_quantization_info,
                              const DataLayout data_layout, const ActivationLayerInfo &act_info)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, input_data_type, 1, input_quantization_info, data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, weights_data_type, 1, weights_quantization_info, data_layout);
        TensorType biases  = create_tensor<TensorType>(biases_shape, bias_data_type, 1, input_quantization_info, data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, input_data_type, 1, output_quantization_info, data_layout);

        // Create Depthwise Convolution configure function
        FunctionType dwc;
        dwc.configure(&src, &weights, &biases, &dst, pad_stride_info, depth_multiplier, act_info, dilation);

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

    SimpleTensor<T> compute_reference(const TensorShape &in_shape, const TensorShape &weights_shape, const TensorShape &biases_shape, const TensorShape &out_shape,
                                      const PadStrideInfo &pad_stride_info, const Size2D &dilation, unsigned int depth_multiplier,
                                      const DataType input_data_type, const DataType weights_data_type, const DataType bias_data_type,
                                      const QuantizationInfo &input_quantization_info, const QuantizationInfo &weights_quantization_info, const QuantizationInfo &output_quantization_info,
                                      const ActivationLayerInfo &act_info)
    {
        SimpleTensor<T>     src{ in_shape, input_data_type, 1, input_quantization_info };
        SimpleTensor<TW>    weights{ weights_shape, weights_data_type, 1, weights_quantization_info };
        SimpleTensor<TBias> biases{ biases_shape, bias_data_type, 1, input_quantization_info };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        SimpleTensor<T> depth_out = reference::depthwise_convolution(src, weights, biases, out_shape, pad_stride_info, depth_multiplier, dilation, output_quantization_info);
        return (act_info.enabled()) ? reference::activation_layer<T>(depth_out, act_info) : depth_out;
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation, unsigned int depth_multiplier, DataType data_type, DataLayout data_layout,
               ActivationLayerInfo act_info)
    {
        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(in_shape, kernel_size, pad_stride_info, dilation, depth_multiplier,
                                                                                                               data_type, data_type, QuantizationInfo(), QuantizationInfo(), QuantizationInfo(),
                                                                                                               data_layout, act_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerNativeValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(size_t width, size_t height, size_t channel, size_t batch, Size2D kernel_size, size_t depth_multiplier, Size2D dilation, Size2D stride, bool padding_valid, DataType data_type,
               DataLayout data_layout)
    {
        const TensorShape src_shape(width, height, channel, batch);
        const TensorShape weights_shape(kernel_size.width, kernel_size.height, channel * depth_multiplier);
        const TensorShape biases_shape(weights_shape.z());

        PadStrideInfo conv_info;
        if(padding_valid)
        {
            conv_info = PadStrideInfo();
        }
        else
        {
            conv_info = calculate_same_pad(src_shape, weights_shape, PadStrideInfo(stride.width, stride.height), DataLayout::NCHW, dilation);
        }

        _target    = compute_target(src_shape, weights_shape, biases_shape, conv_info, dilation, depth_multiplier, data_type, data_layout);
        _reference = compute_reference(src_shape, weights_shape, biases_shape, conv_info, dilation, depth_multiplier, data_type);
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

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, TensorShape biases_shape, PadStrideInfo &conv_info, Size2D dilation,
                              unsigned int depth_multiplier, const DataType data_type, const DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType biases  = create_tensor<TensorType>(biases_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst     = create_tensor<TensorType>(TensorShape(), data_type, 1, QuantizationInfo(), data_layout);

        // Create Depthwise Convolution configure function
        FunctionType dwc;
        dwc.configure(&src, &weights, &biases, &dst, conv_info, depth_multiplier, dilation);

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

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &biases_shape, const PadStrideInfo &conv_info,
                                      const Size2D &dilation, unsigned int depth_multiplier, const DataType data_type)
    {
        SimpleTensor<T> src{ input_shape, data_type };
        SimpleTensor<T> weights{ weights_shape, data_type };
        SimpleTensor<T> biases{ biases_shape, data_type };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        const TensorShape dst_shape = compute_depthwise_convolution_shape(TensorInfo(input_shape, 1, data_type), TensorInfo(weights_shape, 1, data_type), conv_info,
                                                                          depth_multiplier, dilation);
        return reference::depthwise_convolution(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerNativeConfigurableValidationFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(size_t width, size_t height, size_t channel, size_t batch, Size2D kernel_size, size_t depth_multiplier, Size2D dilation, Size2D stride, bool padding_valid, DataType data_type,
               DataLayout data_layout, const ActivationLayerInfo &act_info, unsigned int n0)
    {
        const TensorShape src_shape(width, height, channel, batch);
        const TensorShape weights_shape(kernel_size.width, kernel_size.height, channel * depth_multiplier);
        const TensorShape biases_shape(weights_shape.z());

        PadStrideInfo conv_info;
        if(padding_valid)
        {
            conv_info = PadStrideInfo();
        }
        else
        {
            conv_info = calculate_same_pad(src_shape, weights_shape, PadStrideInfo(stride.width, stride.height), DataLayout::NCHW, dilation);
        }

        _target    = compute_target(src_shape, weights_shape, biases_shape, conv_info, dilation, depth_multiplier, data_type, data_layout, act_info, n0);
        _reference = compute_reference(src_shape, weights_shape, biases_shape, conv_info, dilation, depth_multiplier, data_type, act_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<half> distribution{ -1.0f, 1.0f };
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, TensorShape biases_shape, PadStrideInfo &conv_info, Size2D dilation,
                              unsigned int depth_multiplier, const DataType data_type, const DataLayout data_layout, const ActivationLayerInfo &act_info, unsigned int n0)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType biases  = create_tensor<TensorType>(biases_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType dst     = create_tensor<TensorType>(TensorShape(), data_type, 1, QuantizationInfo(), data_layout);

        DWCWeightsKernelInfo dwc_weights_info;
        dwc_weights_info.n0 = n0;

        DWCKernelInfo dwc_info;
        dwc_info.activation_info = act_info;

        // Create Depthwise Convolution configure function
        FunctionType dwc;
        dwc.configure(&src, &weights, &biases, &dst, dwc_weights_info, dwc_info, conv_info, depth_multiplier, dilation);

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

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &biases_shape, const PadStrideInfo &conv_info,
                                      const Size2D &dilation, unsigned int depth_multiplier, const DataType data_type, const ActivationLayerInfo &act_info)
    {
        SimpleTensor<T> src{ input_shape, data_type };
        SimpleTensor<T> weights{ weights_shape, data_type };
        SimpleTensor<T> biases{ biases_shape, data_type };

        fill(src, 0);
        fill(weights, 1);
        fill(biases, 2);

        const TensorShape dst_shape = compute_depthwise_convolution_shape(TensorInfo(input_shape, 1, data_type), TensorInfo(weights_shape, 1, data_type), conv_info,
                                                                          depth_multiplier, dilation);
        return reference::activation_layer(reference::depthwise_convolution(src, weights, biases, dst_shape, conv_info, depth_multiplier, dilation), act_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionLayerValidationQuantizedFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation, unsigned int depth_multiplier, DataType data_type,
               QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, DataLayout data_layout, ActivationLayerInfo act_info)
    {
        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(in_shape, kernel_size, pad_stride_info, dilation, depth_multiplier, data_type,
                                                                                                               data_type, input_quantization_info, input_quantization_info, output_quantization_info,
                                                                                                               data_layout, act_info);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class DepthwiseConvolutionLayerValidationQuantizedPerChannelFixture : public DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, TW>
{
public:
    template <typename...>
    void setup(TensorShape in_shape, Size2D kernel_size, PadStrideInfo pad_stride_info, Size2D dilation, unsigned int depth_multiplier, DataType input_data_type, DataType weights_data_type,
               QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, DataLayout data_layout, ActivationLayerInfo act_info)
    {
        const float out_scale = output_quantization_info.uniform().scale;
        const float in_scale  = input_quantization_info.uniform().scale;

        std::vector<float>               weights_scales{};
        std::mt19937                     gen(library->seed());
        std::uniform_real_distribution<> dis(0.01f, out_scale / in_scale);
        for(size_t i = 0; i < in_shape.z() * depth_multiplier; ++i)
        {
            weights_scales.push_back(dis(gen));
        }

        DepthwiseConvolutionLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, TW>::setup(in_shape, kernel_size, pad_stride_info, dilation, depth_multiplier,
                                                                                                                input_data_type, weights_data_type,
                                                                                                                input_quantization_info, QuantizationInfo(weights_scales), output_quantization_info,
                                                                                                                data_layout, act_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE */
