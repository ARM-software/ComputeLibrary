/*
 * Copyright (c) 2018-2019 ARM Limited.
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

#ifndef ARM_COMPUTE_TEST_GATHER_FIXTURE
#define ARM_COMPUTE_TEST_GATHER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"

#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/Gather.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GatherFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, TensorShape indices_shape, int axis, DataType data_type)
    {
        _target    = compute_target(input_shape, data_type, axis, indices_shape);
        _reference = compute_reference(input_shape, data_type, axis, indices_shape);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    template <typename U>
    void generate_indices(U &&indices, const TensorShape &input_shape, uint32_t actual_axis, TensorShape indices_shape)
    {
        std::mt19937 gen(library->seed());
        uint32_t    *indices_ptr = static_cast<uint32_t *>(indices.data());

        std::uniform_int_distribution<uint32_t> dist_index(0, input_shape[actual_axis] - 1);
        //Let's consider 1D indices
        for(unsigned int ind = 0; ind < indices_shape[0]; ind++)
        {
            indices_ptr[ind] = dist_index(gen);
        }
    }

    TensorType compute_target(const TensorShape &input_shape,
                              DataType           data_type,
                              int                axis,
                              const TensorShape  indices_shape)
    {
        // Create tensors
        TensorType     src            = create_tensor<TensorType>(input_shape, data_type);
        TensorType     indices_tensor = create_tensor<TensorType>(indices_shape, DataType::U32);
        const uint32_t actual_axis    = wrap_around(axis, static_cast<int>(input_shape.num_dimensions()));
        TensorShape    output_shape   = arm_compute::misc::shape_calculator::compute_gather_shape(input_shape, indices_shape, actual_axis);
        TensorType     dst            = create_tensor<TensorType>(output_shape, data_type);

        // Create and configure function
        FunctionType gather;
        gather.configure(&src, &indices_tensor, &dst, axis);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(indices_tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        indices_tensor.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!indices_tensor.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));
        generate_indices(AccessorType(indices_tensor), input_shape, actual_axis, indices_shape);

        // Compute function
        gather.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &input_shape,
                                      DataType           data_type,
                                      int                axis,
                                      const TensorShape  indices_shape)
    {
        // Create reference tensor
        SimpleTensor<T>        src{ input_shape, data_type };
        SimpleTensor<uint32_t> indices_tensor{ indices_shape, DataType::U32 };
        const uint32_t         actual_axis = wrap_around(axis, static_cast<int>(input_shape.num_dimensions()));

        // Fill reference tensor
        fill(src);
        generate_indices(indices_tensor, input_shape, actual_axis, indices_shape);

        return reference::gather(src, indices_tensor, actual_axis);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute

#endif /* ARM_COMPUTE_TEST_GATHER_FIXTURE */
