/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_COL2IM_FIXTURE
#define ARM_COMPUTE_TEST_COL2IM_FIXTURE

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Col2Im.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
using namespace arm_compute::misc::shape_calculator;

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, bool batch_size_on_z>
class Col2ImValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, const unsigned int convolved_width, unsigned int convolved_height, unsigned int num_groups, DataType data_type)
    {
        const Size2D convolved_dims(convolved_width, convolved_height);

        const TensorShape output_shape = compute_col2im_shape(TensorInfo(input_shape, 1, data_type), convolved_dims, batch_size_on_z, num_groups);

        _target    = compute_target(input_shape, output_shape, convolved_dims, num_groups, data_type);
        _reference = compute_reference(input_shape, output_shape, num_groups, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, const int seed)
    {
        library->fill_tensor_uniform(tensor, seed);
    }

    TensorType compute_target(const TensorShape &input_shape, const TensorShape &output_shape, const Size2D &convolved_dims, unsigned int num_groups, DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, data_type);
        TensorType dst = create_tensor<TensorType>(output_shape, data_type);

        // Create and configure function
        FunctionType col2im_func;
        col2im_func.configure(&src, &dst, convolved_dims, num_groups);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute function
        col2im_func.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape, const TensorShape &output_shape, unsigned int num_groups, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ input_shape, data_type };

        // Fill reference
        fill(src, 0);

        return reference::col2im<T>(src, output_shape, num_groups);
    }
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_COL2IM_FIXTURE */
