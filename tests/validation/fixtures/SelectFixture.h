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
#ifndef ARM_COMPUTE_TEST_SELECT_FIXTURE
#define ARM_COMPUTE_TEST_SELECT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Select.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
namespace detail
{
/** Get the expected shape of the condition tensor
 *
 * @param shape               Shape of the other input tensor to select
 * @param has_same_same_rank  Boolean that flags if the condition has the same rank with the other tensors
 *
 * @return the expected condition shape
 */
inline TensorShape select_condition_shape(TensorShape shape, bool has_same_same_rank)
{
    TensorShape condition_shape = shape;
    if(!has_same_same_rank)
    {
        condition_shape = TensorShape(shape[shape.num_dimensions() - 1]);
    }
    return condition_shape;
}
} // namespace detail

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class SelectValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, bool has_same_same_rank, DataType data_type)
    {
        TensorShape condition_shape = detail::select_condition_shape(shape, has_same_same_rank);

        _target    = compute_target(shape, condition_shape, data_type);
        _reference = compute_reference(shape, condition_shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }
    template <typename U>
    void fill_bool(U &&tensor, int i)
    {
        ARM_COMPUTE_ERROR_ON(tensor.data_type() != DataType::U8);
        library->fill_tensor_uniform(tensor, i, static_cast<uint8_t>(0), static_cast<uint8_t>(1));
    }

    TensorType compute_target(const TensorShape &shape, const TensorShape &condition_shape, DataType data_type)
    {
        // Create tensors
        TensorType c_t   = create_tensor<TensorType>(condition_shape, DataType::U8);
        TensorType x_t   = create_tensor<TensorType>(shape, data_type);
        TensorType y_t   = create_tensor<TensorType>(shape, data_type);
        TensorType dst_t = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType select;
        select.configure(&c_t, &x_t, &y_t, &dst_t);

        ARM_COMPUTE_EXPECT(c_t.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(x_t.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(y_t.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst_t.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        c_t.allocator()->allocate();
        x_t.allocator()->allocate();
        y_t.allocator()->allocate();
        dst_t.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!c_t.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!x_t.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!y_t.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst_t.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill_bool(AccessorType(c_t), 0);
        fill(AccessorType(x_t), 1);
        fill(AccessorType(y_t), 2);

        // Compute function
        select.run();

        return dst_t;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const TensorShape &condition_shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<uint8_t> ref_c{ condition_shape, DataType::U8 };
        SimpleTensor<T>       ref_x{ shape, data_type };
        SimpleTensor<T>       ref_y{ shape, data_type };

        // Fill reference
        fill_bool(ref_c, 0);
        fill(ref_x, 1);
        fill(ref_y, 2);

        return reference::select<T>(ref_c, ref_x, ref_y);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SELECT_FIXTURE */
