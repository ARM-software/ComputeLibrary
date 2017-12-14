/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MIN_MAX_LOCATION_FIXTURE
#define ARM_COMPUTE_TEST_MIN_MAX_LOCATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/Types.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/MinMaxLocation.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename ArrayType, typename ArrayAccessorType, typename FunctionType, typename T>
class MinMaxLocationValidationFixture : public framework::Fixture
{
public:
    using target_type = typename std::conditional<std::is_integral<T>::value, int32_t, float>::type;

    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        _target    = compute_target(shape, data_type);
        _reference = compute_reference(shape, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    MinMaxLocationValues<target_type> compute_target(const TensorShape &shape, DataType data_type)
    {
        MinMaxLocationValues<target_type> target;

        ArrayType min_loc(shape.total_size());
        ArrayType max_loc(shape.total_size());

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType min_max_loc;
        min_max_loc.configure(&src, &target.min, &target.max, &min_loc, &max_loc);

        // Allocate tensors
        src.allocator()->allocate();
        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        min_max_loc.run();

        // Create accessor objects for mapping operations
        ArrayAccessorType min_loc_accessor(min_loc);
        ArrayAccessorType max_loc_accessor(max_loc);

        // Move min Coordinates2D values from ArrayType to vector
        for(size_t i = 0; i < min_loc.num_values(); ++i)
        {
            target.min_loc.push_back(std::move(min_loc_accessor.at(i)));
        }

        // Move max Coordinates2D values from ArrayType to vector
        for(size_t i = 0; i < max_loc.num_values(); ++i)
        {
            target.max_loc.push_back(std::move(max_loc_accessor.at(i)));
        }

        return target;
    }

    MinMaxLocationValues<T> compute_reference(const TensorShape &shape, DataType data_type)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        return reference::min_max_location<T>(src);
    }

    MinMaxLocationValues<target_type> _target{};
    MinMaxLocationValues<T>           _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MIN_MAX_LOCATION_FIXTURE */
