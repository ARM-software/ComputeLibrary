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
#ifndef ARM_COMPUTE_TEST_NON_MAX_SUPPRESSION_FIXTURE
#define ARM_COMPUTE_TEST_NON_MAX_SUPPRESSION_FIXTURE

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/NonMaxSuppression.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType>

class NMSValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, unsigned int max_output_size, float score_threshold, float nms_threshold)
    {
        ARM_COMPUTE_ERROR_ON(max_output_size == 0);
        ARM_COMPUTE_ERROR_ON(input_shape.num_dimensions() != 2);
        const TensorShape output_shape(max_output_size);
        const TensorShape scores_shape(input_shape[1]);
        _target    = compute_target(input_shape, scores_shape, output_shape, max_output_size, score_threshold, nms_threshold);
        _reference = compute_reference(input_shape, scores_shape, output_shape, max_output_size, score_threshold, nms_threshold);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, int lo, int hi)
    {
        std::uniform_real_distribution<> distribution(lo, hi);
        library->fill_boxes(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape input_shape, const TensorShape scores_shape, const TensorShape output_shape,
                              unsigned int max_output_size, float score_threshold, float nms_threshold)
    {
        // Create tensors
        TensorType bboxes  = create_tensor<TensorType>(input_shape, DataType::F32);
        TensorType scores  = create_tensor<TensorType>(scores_shape, DataType::F32);
        TensorType indices = create_tensor<TensorType>(output_shape, DataType::S32);

        // Create and configure function
        FunctionType nms_func;
        nms_func.configure(&bboxes, &scores, &indices, max_output_size, score_threshold, nms_threshold);

        ARM_COMPUTE_EXPECT(bboxes.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(indices.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(scores.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        bboxes.allocator()->allocate();
        indices.allocator()->allocate();
        scores.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!bboxes.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!indices.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!scores.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(bboxes), 0, 0.f, 1.f);
        fill(AccessorType(scores), 1, 0.f, 1.f);

        // Compute function
        nms_func.run();
        return indices;
    }

    SimpleTensor<int> compute_reference(const TensorShape input_shape, const TensorShape scores_shape, const TensorShape output_shape,
                                        unsigned int max_output_size, float score_threshold, float nms_threshold)
    {
        // Create reference
        SimpleTensor<float> bboxes{ input_shape, DataType::F32 };
        SimpleTensor<float> scores{ scores_shape, DataType::F32 };
        SimpleTensor<int>   indices{ output_shape, DataType::S32 };

        // Fill reference
        fill(bboxes, 0, 0.f, 1.f);
        fill(scores, 1, 0.f, 1.f);

        return reference::non_max_suppression(bboxes, scores, indices, max_output_size, score_threshold, nms_threshold);
    }

    TensorType        _target{};
    SimpleTensor<int> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_NON_MAX_SUPPRESSION_FIXTURE */
