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
#ifndef ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE
#define ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "framework/Asserts.h"
#include "framework/Fixture.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/validation_new/CPP/DepthwiseConvolution.h"
#include "tests/validation_new/Helpers.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class DepthwiseConvolutionValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape in_shape, TensorShape weights_shape, TensorShape out_shape, PadStrideInfo pad_stride_info)
    {
        _target    = compute_target(in_shape, weights_shape, out_shape, pad_stride_info);
        _reference = compute_reference(in_shape, weights_shape, out_shape, pad_stride_info);
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

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &output_shape, PadStrideInfo &pad_stride_info)
    {
        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, DataType::F32);
        TensorType weights = create_tensor<TensorType>(weights_shape, DataType::F32);
        TensorType dst     = create_tensor<TensorType>(output_shape, DataType::F32);

        // Create Depthwise Convolution configure function
        CLDepthwiseConvolution depthwise_convolution;
        depthwise_convolution.configure(&src, &dst, &weights, pad_stride_info);

        // Allocate tensors
        src.allocator()->allocate();
        weights.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!weights.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(weights), 1);

        // Compute function
        depthwise_convolution.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &in_shape, const TensorShape &weights_shape, const TensorShape &out_shape, const PadStrideInfo &pad_stride_info)
    {
        SimpleTensor<T> src(in_shape, DataType::F32);
        SimpleTensor<T> weights(weights_shape, DataType::F32);

        fill(src, 0);
        fill(weights, 1);

        return reference::depthwise_convolution(src, weights, out_shape, pad_stride_info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTHWISE_CONVOLUTION_FIXTURE */
