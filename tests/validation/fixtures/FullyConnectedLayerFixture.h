/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_FULLYCONNECTEDLAYERFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_FULLYCONNECTEDLAYERFIXTURE_H

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Utils.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/RawTensor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/Validation.h"
#include "tests/validation/reference/ActivationLayer.h"
#include "tests/validation/reference/FullyConnectedLayer.h"
#include "tests/validation/reference/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FullyConnectedLayerValidationGenericFixture : public framework::Fixture
{
public:
    using TDecay = typename std::decay<T>::type;
    using TBias  = typename std::conditional < (std::is_same<TDecay, uint8_t>::value || std::is_same<TDecay, int8_t>::value), int32_t, T >::type;

public:
    void setup_quantization(TensorShape weights_shape, TensorShape output_shape, QuantizationInfo &input_q_info, QuantizationInfo &weights_q_info, DataType data_type)
    {
        _hash = weights_shape[0] + weights_shape[1] + output_shape[0] + output_shape[1];
        const int32_t t_max = static_cast<int32_t>(std::numeric_limits<T>::max());
        const int32_t t_min = static_cast<int32_t>(std::numeric_limits<T>::min());

        std::mt19937                           generator(library->seed() + _hash);
        std::uniform_real_distribution<float>  distribution_float(-5.0f, 3.0f);
        std::uniform_int_distribution<int32_t> distribution_t(t_min, t_max);

        const float scale_lhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]
        const float scale_rhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]
        const int32_t offset_lhs = distribution_t(generator);
        const int32_t offset_rhs = distribution_t(generator);

        input_q_info = QuantizationInfo(scale_lhs, offset_lhs);
        weights_q_info = QuantizationInfo(scale_rhs, offset_rhs);


        const int k = weights_shape.x();
        QuantizationHint q_hint = suggest_mac_dst_q_info_and_bias(input_q_info, weights_q_info, k, data_type, 0.1f /* bias_fraction */, 4 /* number of standard deviations*/);

        _dst_q_info = q_hint.q_info;
        _min_bias = q_hint.bias_min;
        _max_bias = q_hint.bias_max;

        // Do not change here as these limits are the natural limits of the associated data types and
        // are embedded in the computation of the dst quantization info.
        _min_u8 = 0;
        _max_u8 = 255;
        _min_s8 = -128;
        _max_s8 = 127;
    }

    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights,
               DataType data_type, QuantizationInfo quantization_info, ActivationLayerInfo activation_info, bool mixed_layout = false)
    {
        ARM_COMPUTE_UNUSED(weights_shape);
        ARM_COMPUTE_UNUSED(bias_shape);

        _mixed_layout      = mixed_layout;
        _data_type         = data_type;
        _bias_data_type    = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;

        // Note : Quantization Info parameter from setup function is only used when quant datatype and activation function is not enabled or is identity.
        if(is_data_type_quantized(data_type) && (!activation_info.enabled() || activation_info.activation() == ActivationFunction::IDENTITY))
        {
            // Initialises quantization info with appropriate scale and offset for given input shapes.
            setup_quantization(weights_shape, output_shape,_input_q_info, _weight_q_info, data_type);
        }
        else
        {
            _input_q_info = quantization_info;
            _weight_q_info = quantization_info;
            _dst_q_info = quantization_info;
        }

        _activation_info   = activation_info;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, transpose_weights, reshape_weights);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape);
    }

protected:
    void mix_layout(FunctionType &layer, TensorType &src, TensorType &dst)
    {
        const DataLayout data_layout = src.info()->data_layout();
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
        if(_data_type == DataType::QASYMM8)
        {
            std::uniform_int_distribution<uint32_t> distribution(_min_u8, _max_u8);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::QASYMM8_SIGNED)
        {
            std::uniform_int_distribution<int32_t> distribution(_min_s8, _max_s8);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::S32)
        {
            std::uniform_int_distribution<int32_t> distribution(_min_bias, _max_bias);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, bool transpose_weights,
                              bool reshape_weights)
    {
        TensorShape reshaped_weights_shape(weights_shape);

        // Test actions depending on the target settings
        //
        //            | reshape   | !reshape
        // -----------+-----------+---------------------------
        //  transpose |           | ***
        // -----------+-----------+---------------------------
        // !transpose | transpose | transpose
        //            |           |
        //
        // ***: That combination is invalid. But we can ignore the transpose flag and handle all !reshape the same
        if(!reshape_weights || !transpose_weights)
        {
            const size_t shape_x = reshaped_weights_shape.x();
            reshaped_weights_shape.set(0, reshaped_weights_shape.y());
            reshaped_weights_shape.set(1, shape_x);
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, _input_q_info);
        TensorType weights = create_tensor<TensorType>(reshaped_weights_shape, _data_type, 1, _weight_q_info);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _bias_data_type, 1);
        TensorType dst     = create_tensor<TensorType>(output_shape, _data_type, 1, _dst_q_info);

        // Create Fully Connected layer info
        FullyConnectedLayerInfo fc_info;
        fc_info.transpose_weights    = transpose_weights;
        fc_info.are_weights_reshaped = !reshape_weights;
        fc_info.activation_info      = _activation_info;

        // Create and configure function.
        FunctionType fc;
        fc.configure(&src, &weights, &bias, &dst, fc_info);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(weights.info()->is_resizable());
        ARM_COMPUTE_ASSERT(bias.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        add_padding_x({ &src, &weights, &bias, &dst });

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
        fill(AccessorType(src), 0 + _hash);
        fill(AccessorType(bias), 2 + _hash);

        if(!reshape_weights || !transpose_weights)
        {
            TensorShape tmp_shape(weights_shape);
            RawTensor   tmp(tmp_shape, _data_type, 1);

            // Fill with original shape
            fill(tmp, 1 + _hash);

            // Transpose elementwise
            tmp = transpose(tmp);

            AccessorType weights_accessor(weights);

            for(int i = 0; i < tmp.num_elements(); ++i)
            {
                Coordinates coord = index2coord(tmp.shape(), i);
                std::copy_n(static_cast<const RawTensor::value_type *>(tmp(coord)),
                            tmp.element_size(),
                            static_cast<RawTensor::value_type *>(weights_accessor(coord)));
            }
        }
        else
        {
            fill(AccessorType(weights), 1 + _hash);
        }

        if(_mixed_layout)
        {
            mix_layout(fc, src, dst);
        }
        else
        {
            // Compute NEFullyConnectedLayer function
            fc.run();
        }

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape)
    {
        // Create reference
        SimpleTensor<T>     src{ input_shape, _data_type, 1, _input_q_info };
        SimpleTensor<T>     weights{ weights_shape, _data_type, 1, _weight_q_info };
        SimpleTensor<TBias> bias{ bias_shape, _bias_data_type, 1, QuantizationInfo() };

        // Fill reference
        fill(src, 0 + _hash);
        fill(weights, 1 + _hash);
        fill(bias, 2 + _hash);

        return reference::activation_layer(reference::fully_connected_layer<T>(src, weights, bias, output_shape, _dst_q_info), _activation_info, _dst_q_info);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    DataType            _data_type{};
    DataType            _bias_data_type{};
    bool                _mixed_layout{ false };
    QuantizationInfo    _input_q_info{};
    QuantizationInfo    _weight_q_info{};
    QuantizationInfo    _dst_q_info{};
    ActivationLayerInfo _activation_info{};

    // Random initialization limits
    // Default values are previously handcrafted limits
    // that sould be used when we don't use dynamic quantization
    int32_t _min_bias{-50};
    int32_t _max_bias{50};

    int32_t _min_u8{0};
    int32_t _max_u8{30};
    int32_t _min_s8{-15};
    int32_t _max_s8{15};
    int    _hash{0};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class FullyConnectedLayerValidationFixture : public FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights, DataType data_type,
               ActivationLayerInfo activation_info)
    {
        FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, transpose_weights,
                                                                                                      reshape_weights, data_type,
                                                                                                      QuantizationInfo(), activation_info, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class FullyConnectedLayerValidationQuantizedFixture : public FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights, DataType data_type,
               QuantizationInfo quantization_info, ActivationLayerInfo activation_info)
    {
        FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, transpose_weights,
                                                                                                      reshape_weights, data_type,
                                                                                                      quantization_info, activation_info, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FullyConnectedWithDynamicTensorsFixture : public framework::Fixture
{
private:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        if(_data_type == DataType::F16)
        {
            arm_compute::utils::uniform_real_distribution_16bit<half> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::F32)
        {
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::QASYMM8)
        {
            std::uniform_int_distribution<uint32_t> distribution(_min_u8, _max_u8);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::QASYMM8_SIGNED)
        {
            std::uniform_int_distribution<int32_t> distribution(_min_s8, _max_s8);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::S32)
        {
            std::uniform_int_distribution<int32_t> distribution(_min_bias, _max_bias);
            library->fill(tensor, distribution, i);
        }
        else
        {
            library->fill_tensor_uniform(tensor, i);
        }
    }

    void fill_transposed_weights(TensorType &weights, TensorShape weights_shape, int seed)
    {
        RawTensor tmp(weights_shape, _data_type, 1);

        // Fill with original shape
        fill(tmp, seed);

        // Transpose elementwise
        tmp = transpose(tmp);

        AccessorType weights_accessor(weights);

        for(int i = 0; i < tmp.num_elements(); ++i)
        {
            Coordinates coord = index2coord(tmp.shape(), i);
            std::copy_n(static_cast<const RawTensor::value_type *>(tmp(coord)),
                        tmp.element_size(),
                        static_cast<RawTensor::value_type *>(weights_accessor(coord)));
        }
    }

    void validate_with_tolerance(TensorType &target, SimpleTensor<float> &ref)
    {
        constexpr RelativeTolerance<float> rel_tolerance_f32(0.01f);
        constexpr AbsoluteTolerance<float> abs_tolerance_f32(0.001f);
        validate(AccessorType(target), ref, rel_tolerance_f32, 0, abs_tolerance_f32);
    }

    void validate_with_tolerance(TensorType &target, SimpleTensor<half_float::half> &ref)
    {
        constexpr AbsoluteTolerance<float>        abs_tolerance_f16(0.3f);
        const RelativeTolerance<half_float::half> rel_tolerance_f16(half_float::half(0.2f));
        constexpr float                           tolerance_num_f16 = 0.07f;

        validate(AccessorType(target), ref, rel_tolerance_f16, tolerance_num_f16, abs_tolerance_f16);
    }

    void validate_with_tolerance(TensorType &target, SimpleTensor<uint8_t> &ref)
    {
        constexpr AbsoluteTolerance<uint32_t> tolerance_qasymm8(1);
        validate(AccessorType(target), ref, tolerance_qasymm8);
    }

    void validate_with_tolerance(TensorType &target, SimpleTensor<int8_t> &ref)
    {
        constexpr AbsoluteTolerance<uint32_t> tolerance_qasymm8_signed(1);
        validate(AccessorType(target), ref, tolerance_qasymm8_signed);
    }

    void setup_quantization(TensorShape weights_shape, TensorShape output_shape, QuantizationInfo &input_q_info, QuantizationInfo &weights_q_info, DataType data_type)
    {
        _hash = weights_shape[0] + weights_shape[1] + output_shape[0] + output_shape[1];

        const int32_t t_max = static_cast<int32_t>(std::numeric_limits<T>::max());
        const int32_t t_min = static_cast<int32_t>(std::numeric_limits<T>::min());

        std::mt19937                           generator(library->seed() + _hash);
        std::uniform_real_distribution<float>  distribution_float(-5.0f, 3.0f);
        std::uniform_int_distribution<int32_t> distribution_t(t_min, t_max);

        const float scale_lhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]
        const float scale_rhs = pow(2, distribution_float(generator)); // [2^-5, 2^3]
        const int32_t offset_lhs = distribution_t(generator);
        const int32_t offset_rhs = distribution_t(generator);

        input_q_info = QuantizationInfo(scale_lhs, offset_lhs);
        weights_q_info = QuantizationInfo(scale_rhs, offset_rhs);

        const int k = weights_shape.x();
        QuantizationHint q_hint = suggest_mac_dst_q_info_and_bias(input_q_info, weights_q_info, k, data_type, 0.1f /* bias_fraction */, 4 /* number of standard deviations*/);

        _dst_q_info = q_hint.q_info;
        _min_bias = q_hint.bias_min;
        _max_bias = q_hint.bias_max;

        // Do not change here as these limits are the natural limits of the associated data types and
        // are embedded in the computation of the dst quantization info.
        _min_u8 = 0;
        _max_u8 = 255;
        _min_s8 = -128;
        _max_s8 = 127;
    }

public:
    using TDecay = typename std::decay<T>::type;
    using TBias  = typename std::conditional < (std::is_same<TDecay, uint8_t>::value || std::is_same<TDecay, int8_t>::value), int32_t, T >::type;

    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape dst_shape,
               DataType data_type, ActivationLayerInfo activation_info, bool constant_weights, bool constant_bias, bool weights_reshaped, bool remove_bias = false)
    {
        _data_type = data_type;

        const bool     is_quantized   = is_data_type_quantized(data_type);
        const DataType bias_data_type = (is_quantized) ? DataType::S32 : data_type;

        if (is_quantized && (!activation_info.enabled() || activation_info.activation() == ActivationFunction::IDENTITY))
        {
            setup_quantization(weights_shape, dst_shape, _src_q_info, _weights_q_info, data_type);
        }
        else
        {
            _src_q_info = QuantizationInfo(0.1f, 10);
            _dst_q_info = QuantizationInfo(0.3f, 20);
            _weights_q_info = QuantizationInfo(0.2f, 5);
        }

        // Configure TensorInfo Objects
        const TensorInfo src_info(src_shape, 1, data_type, _src_q_info);
        const TensorInfo dst_info(dst_shape, 1, data_type, _dst_q_info);
        TensorInfo       bias_info(bias_shape, 1, bias_data_type);
        TensorInfo       wei_info(weights_shape, 1, data_type, _weights_q_info);

        if(!constant_weights && weights_reshaped)
        {
            const TensorShape tr_weights_shape{ weights_shape[1], weights_shape[0] };
            wei_info.set_tensor_shape(tr_weights_shape);
        }
        wei_info.set_are_values_constant(constant_weights);
        bias_info.set_are_values_constant(constant_bias);

        // Initialise Tensors
        _src.allocator()->init(src_info);
        _weights.allocator()->init(wei_info);
        if(!remove_bias)
            _bias.allocator()->init(bias_info);
        _dst.allocator()->init(dst_info);

        // Configure FC layer and mark the weights as non constant
        FullyConnectedLayerInfo fc_info;
        fc_info.activation_info = activation_info;
        if(!constant_weights)
        {
            fc_info.are_weights_reshaped = weights_reshaped;
            fc_info.transpose_weights    = !weights_reshaped;
        }
        FunctionType fc;
        fc.configure(&_src, &_weights, (remove_bias) ? nullptr : &_bias, &_dst, fc_info);

        // Allocate all the tensors
        _src.allocator()->allocate();
        _weights.allocator()->allocate();
        if(!remove_bias)
            _bias.allocator()->allocate();
        _dst.allocator()->allocate();

        // Run multiple iterations with different inputs
        constexpr int num_iterations    = 5;
        int           randomizer_offset = 0;

        // Create reference tensors
        SimpleTensor<T>     src{ src_shape, data_type, 1, _src_q_info };
        SimpleTensor<T>     weights{ weights_shape, data_type, 1, _weights_q_info };
        SimpleTensor<TBias> bias{ bias_shape, bias_data_type };

        // Fill weights and/or bias if they remain constant
        if(constant_weights)
        {
            fill(AccessorType(_weights), 1 + _hash);
            fill(weights, 1 + _hash);
        }
        if(constant_bias && !remove_bias)
        {
            fill(AccessorType(_bias), 2 + _hash);
            fill(bias, 2 + _hash);
        }
        // To remove bias, fill with 0
        if(remove_bias && is_quantized)
        {
            library->fill_tensor_value(bias, 0);
        }
        else if(remove_bias)
        {
            library->fill_tensor_value(bias, (float)0.0);
        }

        for(int i = 0; i < num_iterations; ++i)
        {
            // Run target
            {
                fill(AccessorType(_src), randomizer_offset);
                if(!constant_weights)
                {
                    if(weights_reshaped)
                    {
                        fill_transposed_weights(_weights, weights_shape, randomizer_offset + 1 + _hash);
                    }
                    else
                    {
                        fill(AccessorType(_weights), randomizer_offset + 1 +_hash);
                    }
                }
                if(!constant_bias && !remove_bias)
                {
                    fill(AccessorType(_bias), randomizer_offset + 2 + _hash);
                }

                fc.run();
            }

            // Run reference and compare
            {
                // Fill reference
                fill(src, randomizer_offset);
                if(!constant_weights)
                {
                    fill(weights, randomizer_offset + 1 + _hash);
                }
                if(!constant_bias && !remove_bias)
                {
                    fill(bias, randomizer_offset + 2 + _hash);
                }

                auto dst = reference::activation_layer(reference::fully_connected_layer<T>(src, weights, bias, dst_shape, _dst_q_info), activation_info, _dst_q_info);

                // Validate
                validate_with_tolerance(_dst, dst);
            }

            randomizer_offset += 100;
        }
    }

private:
    TensorType _src{}, _weights{}, _bias{}, _dst{};
    DataType   _data_type{ DataType::UNKNOWN };

    QuantizationInfo _src_q_info{};
    QuantizationInfo _weights_q_info{};
    QuantizationInfo _dst_q_info{};

    // Random initialization limits
    // Default values are previously handcrafted limits
    // that sould be used when we don't use dynamic quantization
    int32_t _min_bias{-50};
    int32_t _max_bias{50};

    int32_t _min_u8{0};
    int32_t _max_u8{30};
    int32_t _min_s8{-15};
    int32_t _max_s8{15};
    int     _hash{0};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FullyConnectedWithDynamicWeightsFixture : public FullyConnectedWithDynamicTensorsFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape dst_shape,
               DataType data_type, ActivationLayerInfo activation_info, bool weights_reshaped)
    {
        FullyConnectedWithDynamicTensorsFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, weights_shape, bias_shape,
                                                                                                  dst_shape, data_type, activation_info, false, true, weights_reshaped, false);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FullyConnectedDynamicNoBiasFixture : public FullyConnectedWithDynamicTensorsFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape dst_shape,
               DataType data_type, ActivationLayerInfo activation_info, bool weights_reshaped)
    {
        FullyConnectedWithDynamicTensorsFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, weights_shape, bias_shape,
                                                                                                  dst_shape, data_type, activation_info, false, true, weights_reshaped, true);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FullyConnectedWithDynamicBiasFixture : public FullyConnectedWithDynamicTensorsFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape dst_shape,
               DataType data_type, ActivationLayerInfo activation_info)
    {
        FullyConnectedWithDynamicTensorsFixture<TensorType, AccessorType, FunctionType, T>::setup(src_shape, weights_shape, bias_shape,
                                                                                                  dst_shape, data_type, activation_info, true, false, false, false);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_FULLYCONNECTEDLAYERFIXTURE_H
