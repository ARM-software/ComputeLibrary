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
#ifndef ARM_COMPUTE_TEST_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/DepthwiseSeparableConvolutionLayer.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseSeparableConvolutionValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape in_shape, TensorShape depthwise_weights_shape, TensorShape depthwise_biases_shape, TensorShape depthwise_out_shape, TensorShape pointwise_weights_shape,
               TensorShape pointwise_biases_shape, TensorShape output_shape,
               PadStrideInfo pad_stride_depthwise_info, PadStrideInfo pad_stride_pointwise_info)
    {
        _target = compute_target(in_shape, depthwise_weights_shape, depthwise_biases_shape, depthwise_out_shape, pointwise_weights_shape, pointwise_biases_shape, output_shape, pad_stride_depthwise_info,
                                 pad_stride_pointwise_info);
        _reference = compute_reference(in_shape, depthwise_weights_shape, depthwise_biases_shape, depthwise_out_shape, pointwise_weights_shape, pointwise_biases_shape, output_shape, pad_stride_depthwise_info,
                                       pad_stride_pointwise_info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, bool zero_fill = false)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution((zero_fill) ? 0.f : -1.0f, (zero_fill) ? 0.f : 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &depthwise_weights_shape, const TensorShape &depthwise_biases_shape, const TensorShape &depthwise_out_shape,
                              const TensorShape &pointwise_weights_shape, const TensorShape &pointwise_biases_shape, const TensorShape &output_shape,
                              const PadStrideInfo &pad_stride_depthwise_info, const PadStrideInfo &pad_stride_pointwise_info)
    {
        // Create tensors
        TensorType src               = create_tensor<TensorType>(input_shape, DataType::F32);
        TensorType depthwise_weights = create_tensor<TensorType>(depthwise_weights_shape, DataType::F32);
        TensorType depthwise_biases  = create_tensor<TensorType>(depthwise_biases_shape, DataType::F32);
        TensorType depthwise_out     = create_tensor<TensorType>(depthwise_out_shape, DataType::F32);
        TensorType pointwise_weights = create_tensor<TensorType>(pointwise_weights_shape, DataType::F32);
        TensorType pointwise_biases  = create_tensor<TensorType>(pointwise_biases_shape, DataType::F32);
        TensorType dst               = create_tensor<TensorType>(output_shape, DataType::F32);

        // Create Depthwise Separable Convolution Layer configure function
        FunctionType depthwise_separable_convolution_layer;
        depthwise_separable_convolution_layer.configure(&src, &depthwise_weights, &depthwise_biases, &depthwise_out, &pointwise_weights, &pointwise_biases, &dst, pad_stride_depthwise_info,
                                                        pad_stride_pointwise_info);

        // Allocate tensors
        src.allocator()->allocate();
        depthwise_weights.allocator()->allocate();
        depthwise_biases.allocator()->allocate();
        depthwise_out.allocator()->allocate();
        pointwise_weights.allocator()->allocate();
        pointwise_biases.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!depthwise_weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!depthwise_biases.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!depthwise_out.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!pointwise_weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!pointwise_biases.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(depthwise_weights), 1);
        fill(AccessorType(depthwise_biases), 2, true);
        fill(AccessorType(pointwise_weights), 3);
        fill(AccessorType(pointwise_biases), 4);

        // Compute function
        depthwise_separable_convolution_layer.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &in_shape, const TensorShape &depthwise_weights_shape, const TensorShape &depthwise_biases_shape, const TensorShape &depthwise_out_shape,
                                      const TensorShape &pointwise_weights_shape, const TensorShape &pointwise_biases_shape, const TensorShape &dst_shape,
                                      const PadStrideInfo &pad_stride_depthwise_info, const PadStrideInfo &pad_stride_pointwise_info)
    {
        SimpleTensor<T> src(in_shape, DataType::F32);
        SimpleTensor<T> depthwise_weights(depthwise_weights_shape, DataType::F32);
        SimpleTensor<T> depthwise_biases(depthwise_biases_shape, DataType::F32);
        SimpleTensor<T> pointwise_weights(pointwise_weights_shape, DataType::F32);
        SimpleTensor<T> pointwise_biases(pointwise_biases_shape, DataType::F32);

        fill(src, 0);
        fill(depthwise_weights, 1);
        fill(depthwise_biases, 2, true);
        fill(pointwise_weights, 3);
        fill(pointwise_biases, 4);

        return reference::depthwise_separable_convolution_layer(src,
                                                                depthwise_weights, depthwise_biases, depthwise_out_shape,
                                                                pointwise_weights, pointwise_biases,
                                                                dst_shape,
                                                                pad_stride_depthwise_info, pad_stride_pointwise_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_SEPARABLE_CONVOLUTION_LAYER_FIXTURE */
