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
#ifndef ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_FIXTURE
#define ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_FIXTURE

#include "tests/Globals.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/BatchToSpaceLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class BatchToSpaceLayerValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape block_shape_shape, TensorShape output_shape, DataType data_type, DataLayout data_layout)
    {
        _target    = compute_target(input_shape, block_shape_shape, output_shape, data_type, data_layout);
        _reference = compute_reference(input_shape, block_shape_shape, output_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        library->fill(tensor, distribution, i);
    }
    TensorType compute_target(TensorShape input_shape, TensorShape block_shape_shape, TensorShape output_shape,
                              DataType data_type, DataLayout data_layout)
    {
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType input       = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType block_shape = create_tensor<TensorType>(block_shape_shape, DataType::S32);
        TensorType output      = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType batch_to_space;
        batch_to_space.configure(&input, &block_shape, &output);

        ARM_COMPUTE_EXPECT(input.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(block_shape.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(output.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        input.allocator()->allocate();
        block_shape.allocator()->allocate();
        output.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!input.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!block_shape.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!output.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(input), 0);
        {
            auto block_shape_data = AccessorType(block_shape);
            const int idx_width       = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
            for(unsigned int i = 0; i < block_shape_shape.x(); ++i)
            {
                static_cast<int32_t *>(block_shape_data.data())[i] = output_shape[i + idx_width] / input_shape[i + idx_width];
            }
        }
        // Compute function
        batch_to_space.run();

        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &block_shape_shape,
                                      const TensorShape &output_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T>       input{ input_shape, data_type };
        SimpleTensor<int32_t> block_shape{ block_shape_shape, DataType::S32 };

        // Fill reference
        fill(input, 0);
        for(unsigned int i = 0; i < block_shape_shape.x(); ++i)
        {
            block_shape[i] = output_shape[i] / input_shape[i];
        }

        // Compute reference
        return reference::batch_to_space(input, block_shape, output_shape);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_FIXTURE */
