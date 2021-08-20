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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/fixtures/ConvolutionLayerFixture.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Permute.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolutionValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional < std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value, int32_t, T >::type;

    template <typename...>
    void setup(TensorShape input_shape, int stride_x, int stride_y, int pad_x, int pad_y, unsigned int kernel_size, unsigned int num_kernels,
               DataType data_type, QuantizationInfo quantization_info, ActivationLayerInfo act_info, DataLayout data_layout, bool mixed_layout = false)
    {
        _quantization_info = quantization_info;
        _data_type         = data_type;
        _mixed_layout      = mixed_layout;

        TensorShape         weights_shape(kernel_size, kernel_size, input_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo info(stride_x, stride_y, pad_x, pad_y, DimensionRoundingType::FLOOR);
        const DataType      bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

        TensorInfo input_info   = TensorInfo(input_shape, 1, data_type);
        TensorInfo weights_info = TensorInfo(weights_shape, 1, data_type);

        const TensorShape output_shape = compute_deep_convolution_shape(input_info, weights_info, info);

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, data_type, bias_data_type, quantization_info, act_info, data_layout);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, data_type, bias_data_type, quantization_info, act_info);
    }

    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation,
               DataType data_type, QuantizationInfo quantization_info, ActivationLayerInfo act_info, DataLayout data_layout)
    {
        ARM_COMPUTE_ERROR_ON(data_layout == DataLayout::UNKNOWN);
        ARM_COMPUTE_UNUSED(dilation);

        _quantization_info = quantization_info;
        _data_type         = data_type;

        const DataType bias_data_type = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, data_type, bias_data_type, quantization_info, act_info, data_layout);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, data_type, bias_data_type, quantization_info, act_info);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        DataLayout data_layout = src.info()->data_layout();
        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(data_layout);
        dst.info()->set_data_layout(data_layout);
    }

    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                std::uniform_int_distribution<uint8_t> distribution(0, 50);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QASYMM8_SIGNED:
            {
                // Use small input range to avoid all the test results being saturated at the end.
                std::uniform_int_distribution<int8_t> distribution(-25, 25);
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
                std::uniform_int_distribution<int32_t> distribution(-5, 5);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, const TensorShape &bias_shape, TensorShape output_shape, const PadStrideInfo &info,
                              DataType data_type, DataType bias_data_type, QuantizationInfo quantization_info, ActivationLayerInfo act_info, const DataLayout &data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, quantization_info, data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, data_type, 1, quantization_info, data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, bias_data_type, 1, quantization_info);
        TensorType dst     = create_tensor<TensorType>(output_shape, data_type, 1, quantization_info, data_layout);

        add_padding_x({ &src, &bias, &dst }, data_layout);
        add_padding_x({ &weights }, data_layout, input_shape[0] % 4 == 0); // Don't add left padding if cl image will be used

        // Create and configure function
        FunctionType conv;
        conv.configure(&src, &weights, &bias, &dst, info, act_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        bias.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(weights), 1);
        fill(AccessorType(bias), 2);

        if(_mixed_layout)
        {
            mix_layout(conv, src, dst);
        }
        else
        {
            // Compute Convolution function
            conv.run();
        }

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info,
                                      DataType data_type, DataType bias_data_type, QuantizationInfo quantization_info, ActivationLayerInfo act_info)
    {
        // Create reference
        SimpleTensor<T>     src{ input_shape, data_type, 1, quantization_info };
        SimpleTensor<T>     weights{ weights_shape, data_type, 1, quantization_info };
        SimpleTensor<TBias> bias{ bias_shape, bias_data_type, 1, quantization_info };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        SimpleTensor<T> dst = reference::convolution_layer<T>(src, weights, bias, output_shape, info);
        return (act_info.enabled()) ? reference::activation_layer<T>(dst, act_info) : dst;
    }
    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    QuantizationInfo _quantization_info{};
    DataType         _data_type{};
    bool             _mixed_layout{ false };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class DirectConvolutionValidationFixture : public DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, int stride_x, int stride_y, int pad_x, int pad_y, unsigned int kernel_size, unsigned int num_kernels, DataType data_type, ActivationLayerInfo act_info,
               DataLayout data_layout)
    {
        DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, stride_x, stride_y, pad_x, pad_y, kernel_size, num_kernels, data_type, QuantizationInfo(),
                                                                                                    act_info, data_layout, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class DirectConvolutionValidationQuantizedFixture : public DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, int stride_x, int stride_y, int pad_x, int pad_y, unsigned int kernel_size, unsigned int num_kernels, DataType data_type, QuantizationInfo quantization_info,
               ActivationLayerInfo act_info, DataLayout data_layout)
    {
        DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, stride_x, stride_y, pad_x, pad_y, kernel_size, num_kernels, data_type, quantization_info,
                                                                                                    act_info, data_layout, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolutionValidationWithTensorShapesQuantizedFixture : public DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation,
               DataType data_type, QuantizationInfo quantization_info, ActivationLayerInfo act_info, DataLayout data_layout)
    {
        DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation, data_type, quantization_info,
                                                                                                    act_info, data_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DirectConvolutionValidationWithTensorShapesFixture : public DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation,
               DataType data_type, ActivationLayerInfo act_info)
    {
        DirectConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation, data_type, QuantizationInfo(),
                                                                                                    act_info, DataLayout::NCHW);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
