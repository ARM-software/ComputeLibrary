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
template <typename TensorType, typename Function, typename Accessor, bool Transposed = false>
class GEMMInterleaveBlockedFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t x, size_t y, int int_by, int block)
    {
        const float       interleave_by_f32 = int_by;
        const TensorShape shape_a(x, y);
        const TensorShape shape_b(static_cast<size_t>(x * interleave_by_f32), static_cast<size_t>(std::ceil(y / interleave_by_f32)));
        // Create tensors
        a = create_tensor<TensorType>(shape_a, DataType::U8, 1);
        b = create_tensor<TensorType>(shape_b, DataType::U8, 1);

        // Create and configure function
        f.configure(&a, &b, int_by, block, Transposed);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
    }
    void run()
    {
        f.run();
    }

    void teardown()
    {
        a.allocator()->free();
        b.allocator()->free();
    }

private:
    TensorType a{};
    TensorType b{};
    Function   f{};
};

/** Fixture that can be used for NEON and CL */
template <typename TensorType, typename Function, typename Accessor>
class GEMMLowpFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t m, size_t n, size_t k)
    {
        const TensorShape shape_a(k, m);
        const TensorShape shape_b(n, k);
        const TensorShape shape_c(n, m);
        // Create tensors
        a = create_tensor<TensorType>(shape_a, DataType::U8, 1);
        b = create_tensor<TensorType>(shape_b, DataType::U8, 1);
        c = create_tensor<TensorType>(shape_c, DataType::U32, 1);

        // Create and configure function
        gemmlowp.configure(&a, &b, &c);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();

        // Fill tensors
        library->fill_tensor_uniform(Accessor(a), 0);
        library->fill_tensor_uniform(Accessor(b), 1);
        library->fill_tensor_uniform(Accessor(c), 2);
    }
    void run()
    {
        gemmlowp.run();
    }

    void teardown()
    {
        a.allocator()->free();
        b.allocator()->free();
        c.allocator()->free();
    }

private:
    TensorType a{};
    TensorType b{};
    TensorType c{};
    Function   gemmlowp{};
};

} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMFIXTURE */
