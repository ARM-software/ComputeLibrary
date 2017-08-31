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
#ifndef ARM_COMPUTE_TEST_CONVOLUTION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_CONVOLUTION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/CPP/ConvolutionLayer.h"
#include "tests/validation/CPP/Utils.h"
#include "tests/validation/Helpers.h"

#include <random>

namespace arm_compute
{
class NEConvolutionLayer;

namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvolutionValidationFixedPointFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, bool reshape_weights, DataType data_type, int fractional_bits)
    {
        _fractional_bits = fractional_bits;
        _data_type       = data_type;

        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info, reshape_weights, data_type, fractional_bits);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info, data_type, fractional_bits);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
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

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info,
                              bool reshape_weights, DataType data_type, int fixed_point_position)
    {
        WeightsInfo weights_info(!reshape_weights, weights_shape.x(), weights_shape.y(), weights_shape[3]);
        TensorShape reshaped_weights_shape(weights_shape);

        if(!reshape_weights)
        {
            // Check if its a "fully connected" convolution
            const bool is_fully_connected_convolution = (output_shape.x() == 1 && output_shape.y() == 1);
            const bool is_optimised                   = std::is_same<FunctionType, NEConvolutionLayer>::value && NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV8 && data_type == DataType::F32;

            reshaped_weights_shape.collapse(3);

            if(bias_shape.total_size() > 0)
            {
                reshaped_weights_shape.set(0, reshaped_weights_shape.x() + 1);
            }

            if(is_fully_connected_convolution || is_optimised)
            {
                const size_t shape_x = reshaped_weights_shape.x();
                reshaped_weights_shape.set(0, reshaped_weights_shape.y());
                reshaped_weights_shape.set(1, shape_x);
            }
            else
            {
                const int interleave_width = 16 / data_size_from_type(data_type);
                reshaped_weights_shape.set(0, reshaped_weights_shape.x() * interleave_width);
                reshaped_weights_shape.set(1, static_cast<unsigned int>(std::ceil(reshaped_weights_shape.y() / static_cast<float>(interleave_width))));
            }
        }

        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, data_type, 1, fixed_point_position);
        TensorType weights = create_tensor<TensorType>(reshaped_weights_shape, data_type, 1, fixed_point_position);
        TensorType bias    = create_tensor<TensorType>(bias_shape, data_type, 1, fixed_point_position);
        TensorType dst     = create_tensor<TensorType>(output_shape, data_type, 1, fixed_point_position);

        // Create and configure function
        FunctionType conv;
        conv.configure(&src, &weights, &bias, &dst, info, weights_info);

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

        if(!reshape_weights)
        {
            const bool is_fully_connected_convolution = (output_shape.x() == 1 && output_shape.y() == 1);
            const bool is_optimised                   = std::is_same<FunctionType, NEConvolutionLayer>::value && NEScheduler::get().cpu_info().CPU >= CPUTarget::ARMV8 && data_type == DataType::F32;

            TensorShape     tmp_weights_shape(weights_shape);
            SimpleTensor<T> tmp_weights(tmp_weights_shape, data_type, 1, fixed_point_position);
            SimpleTensor<T> tmp_bias(bias_shape, data_type, 1, fixed_point_position);

            // Fill with original shape
            fill(tmp_weights, 1);
            fill(tmp_bias, 2);

            tmp_weights = linearise_weights(tmp_weights, &tmp_bias);

            if(!is_fully_connected_convolution && !is_optimised)
            {
                // Transpose with interleave
                const int interleave_size = 16 / tmp_weights.element_size();
                tmp_weights               = transpose(std::move(tmp_weights), interleave_size);
            }

            AccessorType weights_accessor(weights);

            for(int i = 0; i < tmp_weights.num_elements(); ++i)
            {
                Coordinates coord = index2coord(tmp_weights.shape(), i);
                std::copy_n(static_cast<const T *>(tmp_weights(coord)), 1, static_cast<T *>(weights_accessor(coord)));
            }
        }
        else
        {
            fill(AccessorType(weights), 1);
            fill(AccessorType(bias), 2);
        }

        // Compute NEConvolutionLayer function
        conv.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info,
                                      DataType data_type, int fixed_point_position)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type, 1, fixed_point_position };
        SimpleTensor<T> weights{ weights_shape, data_type, 1, fixed_point_position };
        SimpleTensor<T> bias{ bias_shape, data_type, 1, fixed_point_position };

        // Fill reference
        fill(src, 0);
        fill(weights, 1);
        fill(bias, 2);

        return reference::convolution_layer<T>(src, weights, bias, output_shape, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    int             _fractional_bits{};
    DataType        _data_type{};

private:
    template <typename U>
    SimpleTensor<U> linearise_weights(const SimpleTensor<U> &weights, const SimpleTensor<U> *biases = nullptr)
    {
        TensorShape dst_shape(weights.shape());
        dst_shape.collapse(3);

        if(biases != nullptr)
        {
            dst_shape.set(0, dst_shape.x() + 1);
        }

        const size_t shape_x = dst_shape.x();
        dst_shape.set(0, dst_shape.y());
        dst_shape.set(1, shape_x);

        SimpleTensor<U> dst(dst_shape, weights.data_type());

        // Don't iterate over biases yet
        for(int weights_idx = 0; weights_idx < weights.num_elements(); ++weights_idx)
        {
            Coordinates weights_coord = index2coord(weights.shape(), weights_idx);
            const int   dst_row       = weights_idx % weights.shape().total_size_lower(3);
            Coordinates dst_coord{ weights_coord[3], dst_row, weights_coord[4] };
            const int   dst_idx = coord2index(dst.shape(), dst_coord);

            dst[dst_idx] = weights[weights_idx];
        }

        if(biases != nullptr)
        {
            // Fill last row with biases
            for(int bias_idx = 0; bias_idx < biases->num_elements(); ++bias_idx)
            {
                Coordinates bias_coord = index2coord(biases->shape(), bias_idx);
                Coordinates dst_coord{ bias_coord.x(), static_cast<int>(dst.shape().y()) - 1, bias_coord.y() };
                int         dst_idx = coord2index(dst.shape(), dst_coord);

                dst[dst_idx] = (*biases)[bias_idx];
            }
        }

        return dst;
    }
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvolutionValidationFixture : public ConvolutionValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info, bool reshape_weights, DataType data_type)
    {
        ConvolutionValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>::setup(input_shape, weights_shape, bias_shape, output_shape, info, reshape_weights, data_type, 0);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CONVOLUTION_LAYER_FIXTURE */
