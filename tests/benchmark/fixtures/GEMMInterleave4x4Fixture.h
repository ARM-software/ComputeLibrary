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
#ifndef ARM_COMPUTE_TEST_GEMM_INTERLEAVE4X4_FIXTURE
#define ARM_COMPUTE_TEST_GEMM_INTERLEAVE4X4_FIXTURE

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
class GEMMInterleave4x4Fixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t x, size_t y, DataType data_type)
    {
        constexpr int fixed_point_position = 4;

        const TensorShape shape_a(x, y);
        const TensorShape shape_b(static_cast<size_t>(x * 4.f), static_cast<size_t>(std::ceil(y / 4.f)));

        // Create tensors
        a = create_tensor<TensorType>(shape_a, data_type, 1, fixed_point_position);
        b = create_tensor<TensorType>(shape_b, data_type, 1, fixed_point_position);

        // Create and configure function
        gemm.configure(&a, &b);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
    }

    void run()
    {
        gemm.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(b);
    }

    void teardown()
    {
        a.allocator()->free();
        b.allocator()->free();
    }

private:
    TensorType a{};
    TensorType b{};
    Function   gemm{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMM_INTERLEAVE4X4_FIXTURE */
