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
#ifndef ARM_COMPUTE_TEST_MIN_MAX_LOCATION_FIXTURE
#define ARM_COMPUTE_TEST_MIN_MAX_LOCATION_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Types.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType, typename Function, typename Accessor, typename ArrayType>
class MinMaxLocationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const TensorShape &shape, DataType data_type)
    {
        // Create tensors
        src = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        min_max_location_func.configure(&src, &min, &max, &min_loc, &max_loc);

        // Allocate tensors
        src.allocator()->allocate();
    }

    void run()
    {
        min_max_location_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

    void teardown()
    {
        src.allocator()->free();
    }

private:
    TensorType src{};
    int32_t    min{ 0 };
    int32_t    max{ 0 };
    ArrayType  min_loc{ 20000 };
    ArrayType  max_loc{ 20000 };
    Function   min_max_location_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MIN_MAX_LOCATION_FIXTURE */
