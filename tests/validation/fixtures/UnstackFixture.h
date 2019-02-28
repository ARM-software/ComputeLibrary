/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_UNSTACK_FIXTURE
#define ARM_COMPUTE_TEST_UNSTACK_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Unstack.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename ITensorType, typename AccessorType, typename FunctionType, typename T>
class UnstackValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, int axis, int num, DataType data_type)
    {
        _target    = compute_target(input_shape, axis, num, data_type);
        _reference = compute_reference(input_shape, axis, num, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    std::vector<TensorType> compute_target(TensorShape input_shape, int axis, unsigned int num, DataType data_type)
    {
        TensorType                 input_tensor = create_tensor<TensorType>(input_shape, data_type);
        const unsigned int         axis_u       = wrap_around(axis, static_cast<int>(input_shape.num_dimensions()));
        const unsigned int         axis_size    = input_shape[axis_u];
        const unsigned int         num_slices   = std::min(num, axis_size);
        std::vector<TensorType>    output_slices(num_slices);
        std::vector<ITensorType *> output_ptrs(num_slices);
        for(size_t k = 0; k < num_slices; ++k)
        {
            output_ptrs[k] = &output_slices[k];
        }
        // Create and configure function
        FunctionType unstack;
        unstack.configure(&input_tensor, output_ptrs, axis);
        // Allocate tensors
        for(auto &out : output_slices)
        {
            out.allocator()->allocate();
            ARM_COMPUTE_EXPECT(!out.info()->is_resizable(), framework::LogLevel::ERRORS);
        }
        input_tensor.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!input_tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
        fill(AccessorType(input_tensor), 0);
        // Compute function
        unstack.run();
        return output_slices;
    }

    std::vector<SimpleTensor<T>> compute_reference(TensorShape input_shape, int axis, unsigned int num, DataType data_type)
    {
        const unsigned int axis_u             = wrap_around(axis, static_cast<int>(input_shape.num_dimensions()));
        const unsigned int axis_size          = input_shape[axis_u];
        const unsigned int num_output_tensors = (num == 0) ? axis_size : std::min(axis_size, num);
        // create and fill input tensor
        SimpleTensor<T> input_tensor{ input_shape, data_type };
        fill(input_tensor, 0);
        // create output tensors
        const TensorShape            slice_shape = arm_compute::misc::shape_calculator::calculate_unstack_shape(input_shape, axis_u);
        std::vector<SimpleTensor<T>> output_tensors(num_output_tensors);
        for(size_t k = 0; k < num_output_tensors; ++k)
        {
            output_tensors[k] = SimpleTensor<T>(slice_shape, data_type);
        }

        return reference::unstack<T>(input_tensor, output_tensors, axis);
    }

    std::vector<TensorType>      _target{};
    std::vector<SimpleTensor<T>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_UNSTACK_FIXTURE */
