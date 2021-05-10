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
#ifndef ARM_COMPUTE_TEST_FULLY_CONNECTED_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_FULLY_CONNECTED_LAYER_FIXTURE

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
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights,
               DataType data_type, QuantizationInfo quantization_info, ActivationLayerInfo activation_info, bool mixed_layout = false)
    {
        ARM_COMPUTE_UNUSED(weights_shape);
        ARM_COMPUTE_UNUSED(bias_shape);

        _mixed_layout      = mixed_layout;
        _data_type         = data_type;
        _bias_data_type    = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;
        _quantization_info = quantization_info;
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
            std::uniform_int_distribution<uint8_t> distribution(0, 30);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::QASYMM8_SIGNED)
        {
            std::uniform_int_distribution<int8_t> distribution(-15, 15);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::S32)
        {
            std::uniform_int_distribution<int32_t> distribution(-50, 50);
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
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, _quantization_info);
        TensorType weights = create_tensor<TensorType>(reshaped_weights_shape, _data_type, 1, _quantization_info);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _bias_data_type, 1, _quantization_info);
        TensorType dst     = create_tensor<TensorType>(output_shape, _data_type, 1, _quantization_info);

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
        fill(AccessorType(src), 0);
        fill(AccessorType(bias), 2);

        if(!reshape_weights || !transpose_weights)
        {
            TensorShape tmp_shape(weights_shape);
            RawTensor   tmp(tmp_shape, _data_type, 1);

            // Fill with original shape
            fill(tmp, 1);

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
            fill(AccessorType(weights), 1);
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
        SimpleTensor<T>     src{ input_shape, _data_type, 1, _quantization_info };
        SimpleTensor<T>     weights{ weights_shape, _data_type, 1, _quantization_info };
        SimpleTensor<TBias> bias{ bias_shape, _bias_data_type, 1, _quantization_info };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        return reference::activation_layer(reference::fully_connected_layer<T>(src, weights, bias, output_shape), _activation_info, _quantization_info);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    DataType            _data_type{};
    DataType            _bias_data_type{};
    bool                _mixed_layout{ false };
    QuantizationInfo    _quantization_info{};
    ActivationLayerInfo _activation_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool mixed_layout = false>
class FullyConnectedLayerValidationFixture : public FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
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
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights, DataType data_type,
               QuantizationInfo quantization_info, ActivationLayerInfo activation_info)
    {
        FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, transpose_weights,
                                                                                                      reshape_weights, data_type,
                                                                                                      quantization_info, activation_info, mixed_layout);
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class FullyConnectedWithDynamicWeightsFixture : public framework::Fixture
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

    void validate_with_tolerance(TensorType &target, SimpleTensor<T> &ref)
    {
        if(_data_type == DataType::F32)
        {
            constexpr RelativeTolerance<float> rel_tolerance_f32(0.05f);
            constexpr AbsoluteTolerance<float> abs_tolerance_f32(0.0001f);
            validate(AccessorType(target), ref, rel_tolerance_f32, 0, abs_tolerance_f32);
        }
        else
        {
            validate(AccessorType(target), ref);
        }
    }

public:
    template <typename...>
    void setup(TensorShape src_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape dst_shape,
               DataType data_type, ActivationLayerInfo activation_info)
    {
        _data_type = data_type;

        // Setup tensor meta-data
        TensorInfo src_info(src_shape, 1, data_type);
        _src.allocator()->init(src_info);

        TensorShape tr_weights_shape{ weights_shape[1], weights_shape[0] };
        TensorInfo  wei_info(tr_weights_shape, 1, data_type);
        _weights.allocator()->init(wei_info);

        TensorInfo bias_info(bias_shape, 1, data_type);
        _bias.allocator()->init(bias_info);

        TensorInfo dst_info(dst_shape, 1, data_type);
        _dst.allocator()->init(dst_info);

        // Configure FC layer and mark the weights as non constant
        FullyConnectedLayerInfo fc_info;
        fc_info.activation_info      = activation_info;
        fc_info.are_weights_reshaped = true;
        fc_info.constant_weights     = false;
        FunctionType fc;
        fc.configure(&_src, &_weights, &_bias, &_dst, fc_info);

        // Allocate all the tensors
        _src.allocator()->allocate();
        _weights.allocator()->allocate();
        _bias.allocator()->allocate();
        _dst.allocator()->allocate();

        // Run multiple iterations with different inputs
        constexpr int num_iterations    = 5;
        int           randomizer_offset = 0;
        for(int i = 0; i < num_iterations; ++i)
        {
            // Run target
            {
                fill(AccessorType(_src), randomizer_offset);
                fill_transposed_weights(_weights, weights_shape, randomizer_offset + 1);
                fill(AccessorType(_bias), randomizer_offset + 2);

                fc.run();
            }

            // Run reference and compare
            {
                SimpleTensor<T> src{ src_shape, data_type };
                SimpleTensor<T> weights{ weights_shape, data_type };
                SimpleTensor<T> bias{ bias_shape, data_type };

                // Fill reference
                fill(src, randomizer_offset);
                fill(weights, randomizer_offset + 1);
                fill(bias, randomizer_offset + 2);

                auto dst = reference::activation_layer(reference::fully_connected_layer<T>(src, weights, bias, dst_shape), activation_info);

                // Validate
                validate_with_tolerance(_dst, dst);
            }

            randomizer_offset += 100;
        }
    }

private:
    TensorType _src{}, _weights{}, _bias{}, _dst{};
    DataType   _data_type{ DataType::UNKNOWN };
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FULLY_CONNECTED_LAYER_FIXTURE */
