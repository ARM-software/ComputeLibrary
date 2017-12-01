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
#ifndef ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ConvolutionLayer.h"
#include "tests/validation/reference/Utils.h"

#include <random>

namespace arm_compute
{
class NEWinogradLayer;

namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class WinogradLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape weights_shape, TensorShape bias_shape, TensorShape output_shape, PadStrideInfo info)
    {
        _target    = compute_target(input_shape, weights_shape, bias_shape, output_shape, info);
        _reference = compute_reference(input_shape, weights_shape, bias_shape, output_shape, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, float min, float max)
    {
        switch(tensor.data_type())
        {
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(min, max);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
            {
                ARM_COMPUTE_ERROR("Not supported");
                library->fill_tensor_uniform(tensor, i);
                break;
            }
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info)
    {
        // Create tensors
        TensorType src     = create_tensor<TensorType>(input_shape, DataType::F32, 1);
        TensorType weights = create_tensor<TensorType>(weights_shape, DataType::F32, 1);
        TensorType bias    = create_tensor<TensorType>(bias_shape, DataType::F32, 1);
        TensorType dst     = create_tensor<TensorType>(output_shape, DataType::F32, 1);

        // Create and configure function
        FunctionType conv;
        conv.configure(&src, &weights, nullptr, &dst, info);

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
        fill(AccessorType(src), 0, -1.f, 1.f);
        fill(AccessorType(weights), 1, -1.f, 1.f);
        fill(AccessorType(bias), 2, 0.f, 0.f);
        fill(AccessorType(dst), 3, -1.f, 1.f);

        // Compute NEWinogradLayer function
        conv.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const TensorShape &bias_shape, const TensorShape &output_shape, const PadStrideInfo &info)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, DataType::F32, 1 };
        SimpleTensor<T> weights{ weights_shape, DataType::F32, 1 };
        SimpleTensor<T> bias{ bias_shape, DataType::F32, 1 };

        // Fill reference
        fill(src, 0, -1.f, 1.f);
        fill(weights, 1, -1.f, 1.f);
        fill(bias, 2, 0.f, 0.f);

        return reference::convolution_layer<T>(src, weights, bias, output_shape, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    int             _fractional_bits{};
    DataType        _data_type{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WINOGRAD_LAYER_FIXTURE */
