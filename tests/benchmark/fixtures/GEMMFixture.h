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
#ifndef ARM_COMPUTE_TEST_GEMMFIXTURE
#define ARM_COMPUTE_TEST_GEMMFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class GEMMFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape_a, TensorShape shape_b, TensorShape shape_c, TensorShape shape_dst, float alpha, float beta, DataType data_type)
    {
        constexpr int fixed_point_position = 4;

        // Create tensors
        a   = create_tensor<TensorType>(shape_a, data_type, 1, fixed_point_position);
        b   = create_tensor<TensorType>(shape_b, data_type, 1, fixed_point_position);
        c   = create_tensor<TensorType>(shape_c, data_type, 1, fixed_point_position);
        dst = create_tensor<TensorType>(shape_dst, data_type, 1, fixed_point_position);

        // Create and configure function
        gemm.configure(&a, &b, &c, &dst, alpha, beta);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        gemm.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        a.allocator()->free();
        b.allocator()->free();
        c.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType a{};
    TensorType b{};
    TensorType c{};
    TensorType dst{};
    Function   gemm{};
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMFIXTURE */
