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
#ifndef ARM_COMPUTE_TEST_GEMMLOWP_ASSEMBLY_FIXTURE
#define ARM_COMPUTE_TEST_GEMMLOWP_ASSEMBLY_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/CPP/GEMMLowp.h"
#include "tests/validation/Helpers.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType>
class GEMMLowpAssemblyFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t m, size_t n, size_t k)
    {
        const TensorShape shape_a(k, m);
        const TensorShape shape_b(n, k);
        const TensorShape shape_c(n, m);
        _target    = compute_target(shape_a, shape_b, shape_c);
        _reference = compute_reference(shape_a, shape_b, shape_c);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i, int lo, int hi)
    {
        std::uniform_int_distribution<> distribution(lo, hi);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c)
    {
        // Create tensors
        TensorType a = create_tensor<TensorType>(shape_a, DataType::S8, 1);
        TensorType b = create_tensor<TensorType>(shape_b, DataType::S8, 1);
        TensorType c = create_tensor<TensorType>(shape_c, DataType::S32, 1);

        // Create and configure function
        FunctionType gemmlowp;
        gemmlowp.configure(&a, &b, &c);

        ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(c.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();
        c.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!b.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!c.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(a), 0, -128, 127);
        fill(AccessorType(b), 1, -128, 127);
        fill(AccessorType(c), 2, 0, 0);

        // Compute GEMM function
        gemmlowp.run();
        return c;
    }

    SimpleTensor<int32_t> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c)
    {
        // Create reference
        SimpleTensor<int8_t> a{ shape_a, DataType::S8, 1 };
        SimpleTensor<int8_t> b{ shape_b, DataType::S8, 1 };

        // Fill reference
        fill(a, 0, -128, 127);
        fill(b, 1, -128, 127);

        return reference::gemmlowp(a, b);
    }

    TensorType            _target{};
    SimpleTensor<int32_t> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE */
