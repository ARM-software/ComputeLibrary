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
#ifndef ARM_COMPUTE_TEST_COMPUTEALLANCHORS_FIXTURE
#define ARM_COMPUTE_TEST_COMPUTEALLANCHORS_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/ComputeAllAnchors.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ComputeAllAnchorsFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t num_anchors, const ComputeAnchorsInfo &info, DataType data_type)
    {
        _target    = compute_target(num_anchors, data_type, info);
        _reference = compute_reference(num_anchors, data_type, info);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0, T(0), T(100));
    }

    TensorType compute_target(size_t num_anchors, DataType data_type, const ComputeAnchorsInfo &info)
    {
        // Create tensors
        TensorShape anchors_shape(4, num_anchors);
        TensorType  anchors = create_tensor<TensorType>(anchors_shape, data_type);
        TensorType  all_anchors;

        // Create and configure function
        FunctionType compute_all_anchors;
        compute_all_anchors.configure(&anchors, &all_anchors, info);

        ARM_COMPUTE_EXPECT(all_anchors.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        all_anchors.allocator()->allocate();
        anchors.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!all_anchors.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(CLAccessor(anchors));

        // Compute function
        compute_all_anchors.run();

        return all_anchors;
    }

    SimpleTensor<T> compute_reference(size_t                    num_anchors,
                                      DataType                  data_type,
                                      const ComputeAnchorsInfo &info)
    {
        // Create reference tensor
        SimpleTensor<T> anchors(TensorShape(4, num_anchors), data_type);

        // Fill reference tensor
        fill(anchors);
        return reference::compute_all_anchors(anchors, info);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COMPUTEALLANCHORS_FIXTURE */
