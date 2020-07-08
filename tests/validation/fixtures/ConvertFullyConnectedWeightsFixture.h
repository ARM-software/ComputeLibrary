/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_CONVERT_FULLY_CONNECTED_WEIGHTS_FIXTURE
#define ARM_COMPUTE_TEST_CONVERT_FULLY_CONNECTED_WEIGHTS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/ConvertFullyConnectedWeights.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ConvertFullyConnectedWeightsValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, unsigned int weights_w, DataLayout training_data_layout, DataType data_type)
    {
        const unsigned int height = input_shape.x() * input_shape.y() * input_shape.z();
        const TensorShape  weights_shape(weights_w, height);

        _target    = compute_target(input_shape, weights_shape, training_data_layout, data_type);
        _reference = compute_reference(input_shape, weights_shape, training_data_layout, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::QASYMM8:
            {
                std::uniform_int_distribution<uint8_t> distribution(0, 10);
                library->fill(tensor, distribution, i);
                break;
            }
            case DataType::F32:
            case DataType::F16:
            {
                std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
        }
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &weights_shape, const DataLayout training_data_layout, const DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(weights_shape, data_type);
        TensorType dst = create_tensor<TensorType>(weights_shape, data_type);

        // Create and configure function
        FunctionType convert_weights;

        convert_weights.configure(&src, &dst, input_shape, training_data_layout);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute function
        convert_weights.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &weights_shape, const DataLayout training_data_layout, const DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ weights_shape, data_type };

        // Fill reference
        fill(src, 0);

        return reference::convert_fully_connected_weights(src, input_shape, training_data_layout);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_CONVERT_FULLY_CONNECTED_WEIGHTS_FIXTURE */
