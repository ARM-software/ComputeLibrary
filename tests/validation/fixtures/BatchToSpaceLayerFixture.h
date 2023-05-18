/*
 * Copyright (c) 2018-2021, 2023 Arm Limited.
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

#include "arm_compute/core/Helpers.h"
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
    void setup(const TensorShape &input_shape, const std::vector<int32_t> &block_shape, const CropInfo &crop_info, const TensorShape &output_shape, DataType data_type, DataLayout data_layout)
    {
        _target    = compute_target(input_shape, block_shape, crop_info, output_shape, data_type, data_layout);
        _reference = compute_reference(input_shape, block_shape, crop_info, output_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        static_assert(std::is_floating_point<T>::value || std::is_same<T, half>::value, "Only floating point data types supported.");
        using DistributionType = typename std::conditional<std::is_same<T, half>::value, arm_compute::utils::uniform_real_distribution_16bit<T>, std::uniform_real_distribution<T>>::type;

        DistributionType distribution{ T(-1.0f), T(1.0f) };
        library->fill(tensor, distribution, i);
    }
    TensorType compute_target(TensorShape input_shape, const std::vector<int32_t> &block_shape, const CropInfo &crop_info, TensorShape output_shape,
                              DataType data_type, DataLayout data_layout)
    {
        ARM_COMPUTE_ERROR_ON(block_shape.size() != 2U); // Only support batch to 2D space (x, y) for now
        if(data_layout == DataLayout::NHWC)
        {
            permute(input_shape, PermutationVector(2U, 0U, 1U));
            permute(output_shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType input  = create_tensor<TensorType>(input_shape, data_type, 1, QuantizationInfo(), data_layout);
        TensorType output = create_tensor<TensorType>(output_shape, data_type, 1, QuantizationInfo(), data_layout);

        // Create and configure function
        FunctionType batch_to_space;
        batch_to_space.configure(&input, block_shape.at(0), block_shape.at(1), &output, crop_info);

        ARM_COMPUTE_ASSERT(input.info()->is_resizable());
        ARM_COMPUTE_ASSERT(output.info()->is_resizable());

        // Allocate tensors
        input.allocator()->allocate();
        output.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!input.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!output.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(input), 0);
        // Compute function
        batch_to_space.run();

        return output;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const std::vector<int32_t> &block_shape,
                                      const CropInfo &crop_info, const TensorShape &output_shape, DataType data_type)
    {
        ARM_COMPUTE_ERROR_ON(block_shape.size() != 2U); // Only support batch to 2D space (x, y) for now
        // Create reference
        SimpleTensor<T> input{ input_shape, data_type };

        // Fill reference
        fill(input, 0);

        // Compute reference
        return reference::batch_to_space(input, block_shape, crop_info, output_shape);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_BATCH_TO_SPACE_LAYER_FIXTURE */
