/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_FLOORFIXTURE
#define ARM_COMPUTE_TEST_FLOORFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class FloorFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type)
    {
        // Create tensors
        src = create_tensor<TensorType>(shape, data_type);
        dst = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        floor_func.configure(&src, &dst);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        floor_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst{};
    Function   floor_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_FLOORFIXTURE */
