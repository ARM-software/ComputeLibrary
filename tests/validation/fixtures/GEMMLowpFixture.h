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
#ifndef ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE
#define ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE

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
class GEMMLowpOffsetValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t m, size_t n, size_t k, int32_t a_offset, int32_t b_offset, int32_t c_offset, int32_t c_mult_int, int32_t out_shift)
    {
        const TensorShape shape_a(k, m);
        const TensorShape shape_b(n, k);
        const TensorShape shape_c(n, m);
        _target    = compute_target(shape_a, shape_b, shape_c, a_offset, b_offset, c_offset, c_mult_int, out_shift);
        _reference = compute_reference(shape_a, shape_b, shape_c, a_offset, b_offset, c_offset, c_mult_int, out_shift);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        ARM_COMPUTE_ERROR_ON(tensor.data_type() != DataType::U8);
        std::uniform_int_distribution<> distribution(0, 3);
        library->fill(tensor, distribution, i);
    }

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c,
                              int32_t a_offset, int32_t b_offset, int32_t c_offset, int32_t c_mult_int, int32_t out_shift)
    {
        // Create tensors
        TensorType a = create_tensor<TensorType>(shape_a, DataType::U8, 1);
        TensorType b = create_tensor<TensorType>(shape_b, DataType::U8, 1);
        TensorType c = create_tensor<TensorType>(shape_c, DataType::U8, 1);

        // Create and configure function
        FunctionType gemmlowp;
        gemmlowp.configure(&a, &b, &c, a_offset, b_offset, c_offset, c_mult_int, out_shift);

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
        fill(AccessorType(a), 0);
        fill(AccessorType(b), 1);
        fill(AccessorType(c), 2);

        // Compute GEMM function
        gemmlowp.run();
        return c;
    }

    SimpleTensor<uint8_t> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, const TensorShape &shape_c,
                                            int32_t a_offset, int32_t b_offset, int32_t c_offset, int32_t c_mult_int, int32_t out_shift)
    {
        // Create reference
        SimpleTensor<uint8_t> a{ shape_a, DataType::U8, 1 };
        SimpleTensor<uint8_t> b{ shape_b, DataType::U8, 1 };
        SimpleTensor<uint8_t> c{ shape_c, DataType::U8, 1 };

        // Fill reference
        fill(a, 0);
        fill(b, 1);
        fill(c, 2);

        return reference::gemmlowp<uint8_t>(a, b, c, a_offset, b_offset, c_offset, c_mult_int, out_shift);
    }

    TensorType            _target{};
    SimpleTensor<uint8_t> _reference{};
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMMLOWP_FIXTURE */
