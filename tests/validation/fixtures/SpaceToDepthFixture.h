/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SPACE_TO_DEPTH_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_SPACE_TO_DEPTH_LAYER_FIXTURE

#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/SpaceToDepth.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SpaceToDepthLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape output_shape, const int block_shape, DataType data_type, DataLayout data_layout)
    {
        _target    = compute_target(input_shape, output_shape, block_shape, data_type, data_layout);
        _reference = compute_reference(input_shape, output_shape, block_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }
    TensorType compute_target(TensorShape input_shape, TensorShape output_shape, const int block_shape,
                              DataType data_type, DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType input  = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType output = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType space_to_depth;
        space_to_depth.configure(&input, &output, block_shape);

        ARM_COMPUTE_EXPECT(input.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(output.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        input.allocator()->allocate();
        output.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!input.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!output.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(input), 0);
        // Compute function
        space_to_depth.run();

        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape,
                                      const int block_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> input{ input_shape, data_type };

        // Fill reference
        fill(input, 0);

        // Compute reference
        return reference::space_to_depth(input, output_shape, block_shape);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SPACE_TO_DEPTH_LAYER_FIXTURE */
