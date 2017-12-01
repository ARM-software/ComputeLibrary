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
#ifndef ARM_COMPUTE_TEST_GEMM_INTERLEAVE_4X4_FIXTURE
#define ARM_COMPUTE_TEST_GEMM_INTERLEAVE_4X4_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/GEMMInterleave4x4.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GEMMInterleave4x4ValidationFixedPointFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(size_t x, size_t y, DataType data_type, int fractional_bits)
    {
        _fractional_bits = fractional_bits;
        _data_type       = data_type;
        const TensorShape shape_a(x, y);
        const TensorShape shape_b(static_cast<size_t>(x * 4.f), static_cast<size_t>(std::ceil(y / 4.f)));
        _target    = compute_target(shape_a, shape_b, data_type, fractional_bits);
        _reference = compute_reference(shape_a, shape_b, data_type, fractional_bits);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        switch(tensor.data_type())
        {
            case DataType::F16:
            case DataType::F32:
            {
                std::uniform_real_distribution<> distribution(-1.f, 1.f);
                library->fill(tensor, distribution, i);
                break;
            }
            default:
                library->fill_tensor_uniform(tensor, i);
                break;
        }
    }

    TensorType compute_target(const TensorShape &shape_a, const TensorShape &shape_b, DataType data_type, int fixed_point_position)
    {
        // Create tensors
        TensorType a = create_tensor<TensorType>(shape_a, data_type, 1, fixed_point_position);
        TensorType b = create_tensor<TensorType>(shape_b, data_type, 1, fixed_point_position);

        // Create and configure function
        FunctionType f;
        f.configure(&a, &b);

        ARM_COMPUTE_EXPECT(a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(b.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        a.allocator()->allocate();
        b.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!a.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!b.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(a), 0);
        fill(AccessorType(b), 0);

        // Compute GEMM function
        f.run();
        return b;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape_a, const TensorShape &shape_b, DataType data_type, int fixed_point_position)
    {
        // Create reference
        SimpleTensor<T> a{ shape_a, data_type, 1, fixed_point_position };
        SimpleTensor<T> b{ shape_b, data_type, 1, fixed_point_position };

        // Fill reference
        fill(a, 0);
        fill(b, 0);

        return reference::gemm_interleave_4x4<T>(a, b);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    int             _fractional_bits{};
    DataType        _data_type{};
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class GEMMInterleave4x4ValidationFixture : public GEMMInterleave4x4ValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(size_t x, size_t y, DataType data_type)
    {
        GEMMInterleave4x4ValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T>::setup(x, y, data_type, 0);
    }
};

} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GEMM_INTERLEAVE_4X4_FIXTURE */
