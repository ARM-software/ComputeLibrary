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
#ifndef ARM_COMPUTE_TEST_CONVOLUTION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_CONVOLUTION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/graph/mutators/MutatorUtils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/PadLayer.h"
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace detail
{
template <typename ConvolutionFunction, typename TensorType>
void configure_conv_function(ConvolutionFunction &func,
                             TensorType *src, const TensorType *weights, const TensorType *bias, TensorType *dst,
                             const PadStrideInfo &info, const WeightsInfo &weights_info,
                             const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    func.configure(src, weights, bias, dst, info, weights_info, dilation, act_info, num_groups);
}
} // namespace detail

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class ConvolutionValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional < std::is_same<typename std::decay<T>::type, uint8_t>::value
                  || std::is_same<typename std::decay<T>::type, int8_t>::value,
                  int32_t, T >::type;

public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights,
               DataType data_type, DataType weights_data_type, DataLayout data_layout, QuantizationInfo quantization_info, QuantizationInfo weight_quantization_info, ActivationLayerInfo act_info,
               bool mixed_layout = false, PaddingList pre_pad_layer = PaddingList({}))
    {
        _mixed_layout             = mixed_layout;
        _data_type                = data_type;
        _weights_data_type        = weights_data_type;
        _is_quantized             = is_data_type_quantized_asymmetric(data_type);
        _is_bfloat16              = data_type == DataType::BFLOAT16;
        _bias_data_type           = _is_quantized ? DataType::S32 : (_is_bfloat16 ? DataType::F32 : data_type);
        _output_data_type         = _is_bfloat16 ? DataType::F32 : data_type;
        _quantization_info        = quantization_info;
        _weight_quantization_info = weight_quantization_info;
        _data_layout              = data_layout;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, reshape_weights, dilation, act_info, pre_pad_layer);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, dilation, act_info, pre_pad_layer);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        // Test Multi DataLayout graph cases, when the data layout changes after configure
        src.info()->set_data_layout(_data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);
        dst.info()->set_data_layout(_data_layout == DataLayout::NCHW ? DataLayout::NHWC : DataLayout::NCHW);

        // Compute Convolution function
        layer.run();

        // Reinstating original data layout for the test suite to properly check the values
        src.info()->set_data_layout(_data_layout);
        dst.info()->set_data_layout(_data_layout);
    }

    void regularize_values(void *values, size_t size)
    {
        float *fvalues = static_cast<float *>(values);
        for(size_t i = 0; i < size; ++i)
        {
            fvalues[i] = float(bfloat16(fvalues[i]));
        }
    }

    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                std::pair<int, int> bounds = get_quantized_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<uint8_t> distribution(bounds.first, bounds.second);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QASYMM8_SIGNED:
            {
                std::pair<int, int> bounds = get_quantized_qasymm8_signed_bounds(tensor.quantization_info(), -1.0f, 1.0f);
                std::uniform_int_distribution<int8_t> distribution(bounds.first, bounds.second);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::QSYMM8_PER_CHANNEL:
            {
                int min_bound = 128;
                int max_bound = -127;
                for(size_t i = 0; i < _weight_quantization_info.scale().size(); i++)
                {
                    std::pair<int, int> bounds = get_symm_quantized_per_channel_bounds(tensor.quantization_info(), -1.0f, 1.0f, i);
                    if(bounds.first < min_bound)
                    {
                        min_bound = bounds.first;
                    }
                    if(bounds.second > max_bound)
                    {
                        max_bound = bounds.second;
                    }
                }
                std::uniform_int_distribution<int8_t> distribution(min_bound, max_bound);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::S32:
            {
                std::uniform_int_distribution<int32_t> distribution(-100, 100);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::BFLOAT16:
            {
                arm_compute::utils::uniform_real_distribution_16bit<bfloat16> distribution{ -1.0f, 1.0f };
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

    // given input is IN nchw format
    TensorType compute_target(TensorShape input_shape, TensorShape weights_shape, const TensorShape &bias_shape, TensorShape output_shape, const PadStrideInfo &info,
                              bool reshape_weights, const Size2D &dilation, const ActivationLayerInfo act_info, PaddingList pre_pad_layer = PaddingList({}))
    {
        ARM_COMPUTE_ERROR_ON((input_shape[2] % weights_shape[2]) != 0);

        const unsigned int num_groups = input_shape[2] / weights_shape[2];

        if(_data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(weights_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));

            if(pre_pad_layer.size() > 0)
            {
                // make sure paddings exist for each c,h,w dimensions
                for(unsigned int i = 0; i < 3 - pre_pad_layer.size(); ++i)
                {
                    pre_pad_layer.push_back({ 0, 0 });
                }

                // rotate padding info from nchw to nhwc
                std::rotate(pre_pad_layer.begin(), pre_pad_layer.begin() + 2, pre_pad_layer.begin() + 3);
            }
        }

        const int idx_width  = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);

        WeightsInfo weights_info(!reshape_weights, weights_shape[idx_width], weights_shape[idx_height], weights_shape[3]);
        TensorShape reshaped_weights_shape(weights_shape);

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, _quantization_info, _data_layout);
        TensorType weights = create_tensor<TensorType>(reshaped_weights_shape, _weights_data_type, 1, _weight_quantization_info, _data_layout);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _bias_data_type, 1, _quantization_info, _data_layout);
        TensorType dst     = create_tensor<TensorType>(output_shape, _output_data_type, 1, _quantization_info, _data_layout);

        // Create and configure function
        FunctionType conv;

        const unsigned int height_index = arm_compute::graph::get_dimension_idx(_data_layout, DataLayoutDimension::HEIGHT);
        const unsigned int width_index  = arm_compute::graph::get_dimension_idx(_data_layout, DataLayoutDimension::WIDTH);

        const PaddingInfo pad_w = width_index < pre_pad_layer.size() ? pre_pad_layer[width_index] : PaddingInfo(0, 0);
        const PaddingInfo pad_h = height_index < pre_pad_layer.size() ? pre_pad_layer[height_index] : PaddingInfo(0, 0);

        if(pre_pad_layer.size() > 0 && arm_compute::graph::is_padding_in_height_or_width(_data_layout, pre_pad_layer))
        {
            // this is the logic implemented in NodeFusionMutator -> fuse_pad_with_convolution
            const PadStrideInfo new_conv_info(
                info.stride().first,
                info.stride().second,
                info.pad_left() + pad_w.first,
                info.pad_right() + pad_w.second,
                info.pad_top() + pad_h.first,
                info.pad_bottom() + pad_h.second,
                info.round());
            detail::configure_conv_function(conv, &src, &weights, &bias, &dst, new_conv_info, weights_info, dilation, act_info, num_groups);
        }
        else
        {
            detail::configure_conv_function(conv, &src, &weights, &bias, &dst, info, weights_info, dilation, act_info, num_groups);
        }

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &weights, &bias, &dst }, _data_layout);

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
                                      const Size2D &dilation, const ActivationLayerInfo act_info, PaddingList pre_pad_layer = PaddingList({}))
    {
        ARM_COMPUTE_ERROR_ON((input_shape[2] % weights_shape[2]) != 0);

        const unsigned int num_groups = input_shape[2] / weights_shape[2];

        // Setup reference data types
        const DataType src_dt     = _is_bfloat16 ? DataType::F32 : _data_type;
        const DataType weights_dt = _is_bfloat16 ? DataType::F32 : _weights_data_type;
        const DataType bias_dt    = _is_bfloat16 ? DataType::F32 : _bias_data_type;

        // Create reference
        SimpleTensor<T>     src{ input_shape, src_dt, 1, _quantization_info };
        SimpleTensor<TW>    weights{ weights_shape, weights_dt, 1, _weight_quantization_info };
        SimpleTensor<TBias> bias{ bias_shape, bias_dt, 1, _quantization_info };

        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        // Fill with bfloat16 to perform the conversion and reduce the mismatches in the output
        if(_is_bfloat16)
        {
            regularize_values(static_cast<void *>(src.data()), src.num_elements());
            regularize_values(static_cast<void *>(weights.data()), weights.num_elements());
        }

        if(pre_pad_layer.size() > 0)
        {
            src = reference::pad_layer<T>(src, pre_pad_layer, PixelValue(0), PaddingMode::CONSTANT);
        }

        return (act_info.enabled()) ? reference::activation_layer<T>(reference::convolution_layer<T>(src, weights, bias, output_shape, info, dilation, num_groups),
                                                                     act_info) :
               reference::convolution_layer<T>(src, weights, bias, output_shape, info, dilation, num_groups);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type{};
    DataType         _weights_data_type{};
    DataType         _bias_data_type{};
    DataType         _output_data_type{};
    DataLayout       _data_layout{};
    QuantizationInfo _quantization_info{};
    QuantizationInfo _weight_quantization_info{};
    bool             _is_quantized = false;
    bool             _is_bfloat16  = false;
    bool             _mixed_layout = false;
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class ConvolutionValidationFixture : public ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights, DataType data_type,
               DataLayout data_layout, ActivationLayerInfo act_info)
    {
        ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation, reshape_weights,
                                                                                                 data_type, data_type, data_layout,
                                                                                                 QuantizationInfo(), QuantizationInfo(), act_info, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class ConvolutionValidationWithPaddingFixture : public ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights, DataType data_type,
               DataLayout data_layout, ActivationLayerInfo act_info, PaddingList pre_pad_layer = PaddingList({}))
    {
        ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation, reshape_weights,
                                                                                                 data_type, data_type, data_layout,
                                                                                                 QuantizationInfo(), QuantizationInfo(), act_info, mixed_layout, pre_pad_layer);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class ConvolutionValidationQuantizedFixture : public ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights, DataType data_type,
               DataLayout data_layout, QuantizationInfo quantization_info, ActivationLayerInfo act_info)
    {
        ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation, reshape_weights,
                                                                                                 data_type, data_type, data_layout, quantization_info, quantization_info, act_info, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename TW>
class ConvolutionValidationQuantizedPerChannelFixture : public ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, TW>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, Size2D dilation, bool reshape_weights, DataType data_type,
               DataLayout data_layout, QuantizationInfo quantization_info, ActivationLayerInfo act_info, DataType weights_data_type)
    {
        std::vector<float>                    weights_scales{};
        std::mt19937                          gen(library->seed());
        std::uniform_real_distribution<float> dis(0.01f, 1.f);
        for(size_t i = 0; i < output_shape[2]; ++i)
        {
            weights_scales.push_back(dis(gen));
        }
        ConvolutionValidationGenericFixture<TensorType, AccessorType, FunctionType, T, TW>::setup(input_shape, weights_shape, bias_shape, output_shape, info, dilation,
                                                                                                  reshape_weights, data_type, weights_data_type, data_layout,
                                                                                                  quantization_info, QuantizationInfo(weights_scales), act_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CONVOLUTION_LAYER_FIXTURE */
