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
#include "tests/validation/reference/FullyConnectedLayer.h"
#include "tests/validation/reference/Utils.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool run_interleave>
class FullyConnectedLayerValidationGenericFixture : public framework::Fixture
{
public:
    using TBias = typename std::conditional<std::is_same<typename std::decay<T>::type, uint8_t>::value, int32_t, T>::type;

public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights,
               DataType data_type, int fractional_bits, QuantizationInfo quantization_info)
    {
        ARM_COMPUTE_UNUSED(weights_shape);
        ARM_COMPUTE_UNUSED(bias_shape);

        _data_type         = data_type;
        _bias_data_type    = is_data_type_quantized_asymmetric(data_type) ? DataType::S32 : data_type;
        _fractional_bits   = fractional_bits;
        _quantization_info = quantization_info;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, transpose_weights, reshape_weights);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, transpose_weights, reshape_weights);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        if(is_data_type_quantized_asymmetric(_data_type))
        {
            std::uniform_int_distribution<uint8_t> distribution(0, 30);
            library->fill(tensor, distribution, i);
        }
        else if(_data_type == DataType::S32)
        {
            std::uniform_int_distribution<int32_t> distribution(-50, 50);
            library->fill(tensor, distribution, i);
        }
        else if(is_data_type_float(_data_type))
        {
            std::uniform_real_distribution<> distribution(0.5f, 1.f);
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
        // !transpose | transpose | transpose &
        //            |           | transpose1xW (if required)
        //
        // ***: That combination is invalid. But we can ignore the transpose flag and handle all !reshape the same
        if(!reshape_weights || !transpose_weights)
        {
            const size_t shape_x = reshaped_weights_shape.x();
            reshaped_weights_shape.set(0, reshaped_weights_shape.y());
            reshaped_weights_shape.set(1, shape_x);

            // Weights have to be passed reshaped
            // Transpose 1xW for batched version
            if(!reshape_weights && output_shape.y() > 1 && run_interleave)
            {
                const int   transpose_width = 16 / data_size_from_type(_data_type);
                const float shape_x         = reshaped_weights_shape.x();
                reshaped_weights_shape.set(0, reshaped_weights_shape.y() * transpose_width);
                reshaped_weights_shape.set(1, static_cast<unsigned int>(std::ceil(shape_x / transpose_width)));
            }
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, _data_type, 1, _fractional_bits, _quantization_info);
        TensorType weights = create_tensor<TensorType>(reshaped_weights_shape, _data_type, 1, _fractional_bits, _quantization_info);
        TensorType bias    = create_tensor<TensorType>(bias_shape, _bias_data_type, 1, _fractional_bits, _quantization_info);
        TensorType dst     = create_tensor<TensorType>(output_shape, _data_type, 1, _fractional_bits, _quantization_info);

        // Create and configure function.
        FunctionType fc;
        fc.configure(&src, &weights, &bias, &dst, transpose_weights, !reshape_weights);

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
        fill(AccessorType(bias), 2);

        if(!reshape_weights || !transpose_weights)
        {
            TensorShape tmp_shape(weights_shape);
            RawTensor   tmp(tmp_shape, _data_type, 1, _fractional_bits);

            // Fill with original shape
            fill(tmp, 1);

            // Transpose elementwise
            tmp = transpose(tmp);

            // Reshape weights for batched runs
            if(!reshape_weights && output_shape.y() > 1 && run_interleave)
            {
                // Transpose with interleave
                const int interleave_size = 16 / tmp.element_size();
                tmp                       = transpose(tmp, interleave_size);
            }

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

        // Compute NEFullyConnectedLayer function
        fc.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, bool transpose_weights,
                                      bool reshape_weights)
    {
        // Create reference
        SimpleTensor<T>     src{ input_shape, _data_type, 1, _fractional_bits, _quantization_info };
        SimpleTensor<T>     weights{ weights_shape, _data_type, 1, _fractional_bits, _quantization_info };
        SimpleTensor<TBias> bias{ bias_shape, _bias_data_type, 1, _fractional_bits, _quantization_info };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        return reference::fully_connected_layer<T>(src, weights, bias, output_shape);
    }

    TensorType       _target{};
    SimpleTensor<T>  _reference{};
    DataType         _data_type{};
    DataType         _bias_data_type{};
    int              _fractional_bits{};
    QuantizationInfo _quantization_info{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool run_interleave>
class FullyConnectedLayerValidationFixture : public FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, run_interleave>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights, DataType data_type)
    {
        FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, run_interleave>::setup(input_shape, weights_shape, bias_shape, output_shape, transpose_weights,
                                                                                                                      reshape_weights, data_type,
                                                                                                                      0, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool run_interleave>
class FullyConnectedLayerValidationFixedPointFixture : public FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, run_interleave>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights, DataType data_type, int fractional_bits)
    {
        FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, run_interleave>::setup(input_shape, weights_shape, bias_shape, output_shape, transpose_weights,
                                                                                                                      reshape_weights, data_type,
                                                                                                                      fractional_bits, QuantizationInfo());
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool run_interleave>
class FullyConnectedLayerValidationQuantizedFixture : public FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, run_interleave>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, bool transpose_weights, bool reshape_weights, DataType data_type,
               QuantizationInfo quantization_info)
    {
        FullyConnectedLayerValidationGenericFixture<TensorType, AccessorType, FunctionType, T, run_interleave>::setup(input_shape, weights_shape, bias_shape, output_shape, transpose_weights,
                                                                                                                      reshape_weights, data_type,
                                                                                                                      0, quantization_info);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FULLY_CONNECTED_LAYER_FIXTURE */
