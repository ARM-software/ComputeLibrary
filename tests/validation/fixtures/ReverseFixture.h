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
#ifndef ARM_COMPUTE_TEST_REVERSE_FIXTURE
#define ARM_COMPUTE_TEST_REVERSE_FIXTURE

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
    template <typename...>
    void setup(TensorShape shape, TensorShape axis_shape, DataType data_type)
    {
        _target    = compute_target(shape, axis_shape, data_type);
        _reference = compute_reference(shape, axis_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }
    std::vector<int> generate_random_axis()
    {
        std::vector<int> axis_v = { 0, 1, 2, 3 };
        std::mt19937     g(0);
        std::shuffle(axis_v.begin(), axis_v.end(), g);

        return axis_v;
    }

    TensorType compute_target(const TensorShape &shape, const TensorShape &axis_shape, DataType data_type)
    {
        // Create tensors
        TensorType src  = create_tensor<TensorType>(shape, data_type, 1);
        TensorType axis = create_tensor<TensorType>(axis_shape, DataType::U32, 1);
        TensorType dst;

        // Create and configure function
        FunctionType reverse_func;
        reverse_func.configure(&src, &dst, &axis);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(axis.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        axis.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!axis.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));
        {
            auto axis_data = AccessorType(axis);
            auto axis_v    = generate_random_axis();
            std::copy(axis_v.begin(), axis_v.begin() + axis_shape.x(), static_cast<int32_t *>(axis_data.data()));
        }

        // Compute function
        reverse_func.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const TensorShape &axis_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T>        src{ shape, data_type };
        SimpleTensor<uint32_t> axis{ axis_shape, DataType::U32 };

        // Fill reference
        fill(src);
        auto axis_v = generate_random_axis();
        std::copy(axis_v.begin(), axis_v.begin() + axis_shape.x(), axis.data());

        return reference::reverse<T>(src, axis);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REVERSE_FIXTURE */
