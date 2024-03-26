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
#ifndef ACL_TESTS_VALIDATION_FIXTURES_REVERSEFIXTURE_H
#define ACL_TESTS_VALIDATION_FIXTURES_REVERSEFIXTURE_H

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Reverse.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ReverseValidationFixture : public framework::Fixture
{
public:
    void setup(TensorShape shape, TensorShape axis_shape, DataType data_type, bool use_negative_axis = false, bool use_inverted_axis = false)
    {
        _num_dims  = shape.num_dimensions();
        _target    = compute_target(shape, axis_shape, data_type, use_negative_axis, use_inverted_axis);
        _reference = compute_reference(shape, axis_shape, data_type, use_negative_axis, use_inverted_axis);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }
    std::vector<int32_t> generate_random_axis(bool use_negative = false)
    {
        std::vector<int32_t> axis_v;
        if(use_negative)
        {
            axis_v = { -1, -2, -3, -4 };
        }
        else
        {
            axis_v = { 0, 1, 2, 3 };
        }
        axis_v = std::vector<int32_t>(axis_v.begin(), axis_v.begin() + _num_dims);
        std::mt19937 g(library->seed());
        std::shuffle(axis_v.begin(), axis_v.end(), g);

        return axis_v;
    }

    TensorType compute_target(const TensorShape &shape, const TensorShape &axis_shape, DataType data_type, bool use_negative_axis, bool use_inverted_axis = false)
    {
        // Create tensors
        TensorType src  = create_tensor<TensorType>(shape, data_type, 1);
        TensorType axis = create_tensor<TensorType>(axis_shape, DataType::U32, 1);
        TensorType dst;

        // Create and configure function
        FunctionType reverse_func;
        reverse_func.configure(&src, &dst, &axis, use_inverted_axis);

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(axis.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        axis.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!axis.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));
        {
            auto axis_data = AccessorType(axis);
            auto axis_v    = generate_random_axis(use_negative_axis);
            std::copy(axis_v.begin(), axis_v.begin() + axis_shape.total_size(), static_cast<int32_t *>(axis_data.data()));
        }

        // Compute function
        reverse_func.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const TensorShape &axis_shape, DataType data_type, bool use_negative_axis, bool use_inverted_axis = false)
    {
        // Create reference
        SimpleTensor<T>       src{ shape, data_type };
        SimpleTensor<int32_t> axis{ axis_shape, DataType::S32 };

        // Fill reference
        fill(src);
        auto axis_v = generate_random_axis(use_negative_axis);
        std::copy(axis_v.begin(), axis_v.begin() + axis_shape.total_size(), axis.data());

        return reference::reverse<T>(src, axis, use_inverted_axis);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    unsigned int    _num_dims{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif // ACL_TESTS_VALIDATION_FIXTURES_REVERSEFIXTURE_H
