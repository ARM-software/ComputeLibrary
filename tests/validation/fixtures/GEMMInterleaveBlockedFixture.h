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
#ifndef ARM_COMPUTE_TEST_GEMM_INTERLEAVE_BLOCKED_FIXTURE
#define ARM_COMPUTE_TEST_GEMM_INTERLEAVE_BLOCKED_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/GEMMInterleaveBlocked.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, bool Transposed = false>
class GEMMInterleaveBlockedValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t x, size_t y, int int_by, int block)
    {
        const float       interleave_by_f32 = int_by;
        const TensorShape shape_a(x, y);
        const TensorShape shape_b(static_cast<size_t>(x * interleave_by_f32), static_cast<size_t>(std::ceil(y / interleave_by_f32)));
        _target    = compute_target(shape_a, shape_b, int_by, block);
        _reference = compute_reference(shape_a, shape_b, int_by, block);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        ARM_COMPUTE_ERROR_ON(tensor.data_type() != DataType::U8);
        std::uniform_int_distribution<> distribution(0, 255);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, int int_by, int block)
    {
        // Create tensors
        TensorType a = create_tensor<TensorType>(shape_a, DataType::U8, 1);
        TensorType b = create_tensor<TensorType>(shape_b, DataType::U8, 1);

        // Create and configure function
        FunctionType f;
        f.configure(&a, &b, int_by, block, Transposed);

        ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!b.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(a), 0);

        // Compute GEMM function
        f.run();
        return b;
    }

    SimpleTensor<uint8_t> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, int int_by, int block)
    {
        // Create reference
        SimpleTensor<uint8_t> a{ shape_a, DataType::U8, 1 };
        SimpleTensor<uint8_t> b{ shape_b, DataType::U8, 1 };

        // Fill reference
        fill(a, 0);
        return reference::gemm_interleave_blocked<uint8_t>(a, b, int_by, block, Transposed);
    }

    TensorType            _target{};
    SimpleTensor<uint8_t> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMM_INTERLEAVE_BLOCKED_FIXTURE */
