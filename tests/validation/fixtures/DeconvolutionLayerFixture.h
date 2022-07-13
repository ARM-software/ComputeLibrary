/*
 * Copyright (c) 2017-2022 Arm Limited.
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
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/DeconvolutionLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class DeconvolutionLayerFixtureBase : public framework::Fixture
{
public:
    using TBias = typename std::conditional < std::is_same<typename std::decay<T>::type, uint8_t>::value || std::is_same<typename std::decay<T>::type, int8_t>::value, int32_t, T >::type;

public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info,
               DataType data_type, DataType weights_data_type, DataLayout data_layout,
               QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, QuantizationInfo weights_quantization_info, bool add_bias)
    {
        _data_type                 = data_type;
        _weights_data_type         = weights_data_type;
        _bias_data_type            = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;
        _data_layout               = data_layout;
        _input_quantization_info   = input_quantization_info;
        _output_quantization_info  = output_quantization_info;
        _weights_quantization_info = weights_quantization_info;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, add_bias);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, add_bias);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                std::pair<int, int> bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<uint32_t> distribution(bounds.first, bounds.second);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QASYMM8_SIGNED:
            {
                std::pair<int, int> bounds = get_quantized_qasymm8_signed_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<int32_t> distribution(bounds.first, bounds.second);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QSYMM8_PER_CHANNEL:
            {
                int min_bound = 128;
                int max_bound = -127;
                for(size_t i = 0; i < _input_quantization_info.scale().size(); i++)
                {
                    std::pair<int, int> bounds = get_symm_quantized_per_channel_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                    if(bounds.first < min_bound)
                    {
                        min_bound = bounds.first;
                    }
                    if(bounds.second > max_bound)
                    {
                        max_bound = bounds.second;
                    }
                }
                std::uniform_int_distribution<int32_t> distribution(min_bound, max_bound);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            {
                std::uniform_int_distribution<int32_t> distribution(-100, 100);
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
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    template <typename U>
    void fill_zeros(U &&tensor)
    {
        switch(tensor.data_type())
        {
            case DataType::S32:
            {
                library->fill_tensor_value(tensor, 0);
                break;
            }
            case DataType::F16:
                library->fill_tensor_value(tensor, static_cast<half>(0.0f));
                break;
            case DataType::F32:
                library->fill_tensor_value(tensor, static_cast<float>(0.0f));
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported");
        }
    }

    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, const TensorShape bias_shape, TensorShape output_shape,
                              const PadStrideInfo &info, bool add_bias)
    {
        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, _input_quantization_info, _data_layout);
        TensorType weights = create_tensor<TensorType>(weights_shape, _weights_data_type, 1, _weights_quantization_info, _data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _bias_data_type, 1, _input_quantization_info, _data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, _data_type, 1, _output_quantization_info, _data_layout);

        // Create and configure function
        FunctionType conv;
        conv.configure(&src, &weights, add_bias ? &bias : nullptr, &dst, info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        }
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        if(add_bias)
        {
            bias.allocator()->allocate();
        }
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!weights.info()->is_resizable());
        if(add_bias)
        {
            ARM_COMPUTE_ASSERT(!bias.info()->is_resizable());
        }
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(weights), 1);
        if(add_bias)
        {
            fill(AccessorType(bias), 2);
        }

        // Compute DeconvolutionLayer function
        conv.run();
        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape,
                                      const PadStrideInfo &info, bool add_bias)
    {
        // Create reference
        SimpleTensor<T>     src{ input_shape, _data_type, 1, _input_quantization_info };
        SimpleTensor<TW>    weights{ weights_shape, _weights_data_type, 1, _weights_quantization_info };
        SimpleTensor<TBias> bias{ bias_shape, _bias_data_type, 1, _input_quantization_info };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);

        if(add_bias)
        {
            fill(bias, 2);
        }
        else
        {
            fill_zeros(bias);
        }
        return reference::deconvolution_layer<T, TW>(src, weights, bias, output_shape, info, _output_quantization_info);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type{};
    DataType         _weights_data_type{};
    DataType         _bias_data_type{};
    DataLayout       _data_layout{};
    QuantizationInfo _input_quantization_info{};
    QuantizationInfo _output_quantization_info{};
    QuantizationInfo _weights_quantization_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, unsigned int kernel_size_x, unsigned int kernel_size_y>
class DeconvolutionValidationFixture : public DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, unsigned int sx, unsigned int sy, unsigned int padx, unsigned int pady,
               unsigned int num_kernels, DataType data_type, DataLayout data_layout, bool add_bias)
    {
        ARM_COMPUTE_ERROR_ON_MSG(kernel_size_x != kernel_size_y, "Only square kernels supported");
        const TensorShape   weights_shape(kernel_size_x, kernel_size_y, input_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo info(sx, sy, padx, pady, DimensionRoundingType::CEIL);
        auto                out_dim = deconvolution_output_dimensions(input_shape.x(), input_shape.y(), kernel_size_x, kernel_size_y, info);
        TensorInfo          input_info(input_shape, 1, data_type);
        TensorInfo          weights_info(weights_shape, 1, data_type);
        TensorShape         output_shape = compute_deconvolution_output_shape(out_dim, input_info, weights_info);
        DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, data_type, data_type, data_layout, QuantizationInfo(),
                                                                                           QuantizationInfo(), QuantizationInfo(), add_bias);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, unsigned int kernel_size_x, unsigned int kernel_size_y>
class DeconvolutionValidationAsymmFixture : public DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, unsigned int sx, unsigned int sy, unsigned int pad_left, unsigned int pad_right, unsigned int pad_top,
               unsigned int pad_bottom, unsigned int num_kernels, DataType data_type, DataLayout data_layout, bool add_bias)
    {
        ARM_COMPUTE_ERROR_ON_MSG(kernel_size_x != kernel_size_y, "Only square kernels supported");
        const TensorShape   weights_shape(kernel_size_x, kernel_size_y, input_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo info(sx, sy, pad_left, pad_right, pad_top, pad_bottom, DimensionRoundingType::CEIL);
        auto                out_dim = deconvolution_output_dimensions(input_shape.x(), input_shape.y(), kernel_size_x, kernel_size_y, info);
        TensorInfo          input_info(input_shape, 1, data_type);
        TensorInfo          weights_info(weights_shape, 1, data_type);
        TensorShape         output_shape = compute_deconvolution_output_shape(out_dim, input_info, weights_info);
        DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, data_type, data_type, data_layout, QuantizationInfo(),
                                                                                           QuantizationInfo(), QuantizationInfo(), add_bias);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, unsigned int kernel_size_x, unsigned int kernel_size_y>
class DeconvolutionValidationQuantizedFixture : public DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, unsigned int sx, unsigned int sy, unsigned int padx, unsigned int pady,
               unsigned int num_kernels, DataType data_type, DataLayout data_layout, QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, bool add_bias)
    {
        ARM_COMPUTE_ERROR_ON_MSG(kernel_size_x != kernel_size_y, "Only square kernels supported");
        const TensorShape   weights_shape(kernel_size_x, kernel_size_y, input_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo info(sx, sy, padx, pady, DimensionRoundingType::CEIL);
        auto                out_dim = deconvolution_output_dimensions(input_shape.x(), input_shape.y(), kernel_size_x, kernel_size_y, info);
        TensorInfo          input_info(input_shape, 1, data_type, input_quantization_info);
        TensorInfo          weights_info(weights_shape, 1, data_type, input_quantization_info);
        TensorShape         output_shape = compute_deconvolution_output_shape(out_dim, input_info, weights_info);
        DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, data_type, data_type, data_layout,
                                                                                           input_quantization_info,
                                                                                           output_quantization_info, input_quantization_info, add_bias);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW, unsigned int kernel_size_x, unsigned int kernel_size_y>
class DeconvolutionValidationQuantizedPerChannelFixture : public DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, TW>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, unsigned int sx, unsigned int sy, unsigned int padx, unsigned int pady,
               unsigned int num_kernels, DataType data_type, DataLayout data_layout, QuantizationInfo input_quantization_info, QuantizationInfo output_quantization_info, bool add_bias,
               DataType weights_data_type)
    {
        ARM_COMPUTE_ERROR_ON_MSG(kernel_size_x != kernel_size_y, "Only square kernels supported");
        const TensorShape   weights_shape(kernel_size_x, kernel_size_y, input_shape.z(), num_kernels);
        const TensorShape   bias_shape(num_kernels);
        const PadStrideInfo info(sx, sy, padx, pady, DimensionRoundingType::CEIL);
        auto                out_dim = deconvolution_output_dimensions(input_shape.x(), input_shape.y(), kernel_size_x, kernel_size_y, info);
        TensorInfo          input_info(input_shape, 1, data_type, input_quantization_info);
        TensorInfo          weights_info(weights_shape, 1, weights_data_type, input_quantization_info);
        TensorShape         output_shape = compute_deconvolution_output_shape(out_dim, input_info, weights_info);

        std::vector<float>                    weights_scales{};
        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> dis(0.01f, 1.f);
        for(size_t i = 0; i < output_shape[2]; ++i)
        {
            weights_scales.push_back(dis(gen));
        }
        DeconvolutionLayerFixtureBase<TensorType, AccessorType, FunctionType, T, TW>::setup(input_shape, weights_shape, bias_shape, output_shape, info, data_type, weights_data_type, data_layout,
                                                                                            input_quantization_info,
                                                                                            output_quantization_info, QuantizationInfo(weights_scales), add_bias);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
