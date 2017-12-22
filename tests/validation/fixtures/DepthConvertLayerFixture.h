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
#ifndef ARM_COMPUTE_TEST_DEPTH_CONVERT_FIXTURE
#define ARM_COMPUTE_TEST_DEPTH_CONVERT_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/DepthConvertLayer.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DepthConvertLayerValidationFixedPointFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, uint32_t fractional_bits)
    {
        _shift           = shift;
        _fractional_bits = fractional_bits;
        _target          = compute_target(shape, dt_in, dt_out, policy, shift, fractional_bits);
        _reference       = compute_reference(shape, dt_in, dt_out, policy, shift, fractional_bits);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, uint32_t fixed_point_position)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, dt_in, 1, static_cast<int>(fixed_point_position));
        TensorType dst = create_tensor<TensorType>(shape, dt_out, 1, static_cast<int>(fixed_point_position));

        // Create and configure function
        FunctionType depth_convert;
        depth_convert.configure(&src, &dst, policy, shift);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);

        // Compute function
        depth_convert.run();

        return dst;
    }

    SimpleTensor<T2> compute_reference(const TensorShape &shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift, uint32_t fixed_point_position)
    {
        // Create reference
        SimpleTensor<T1> src{ shape, dt_in, 1, static_cast<int>(fixed_point_position) };

        // Fill reference
        fill(src, 0);

        return reference::depth_convert<T1, T2>(src, dt_out, policy, shift);
    }

    TensorType       _target{};
    SimpleTensor<T2> _reference{};
    int              _fractional_bits{};
    int              _shift{};
};
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DepthConvertLayerValidationFixture : public DepthConvertLayerValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t shift)
    {
        DepthConvertLayerValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, dt_in, dt_out, policy, shift, 0);
    }
};
template <typename TensorType, typename AccessorType, typename FunctionType, typename T1, typename T2>
class DepthConvertLayerValidationFractionalBitsFixture : public DepthConvertLayerValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T1, T2>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType dt_in, DataType dt_out, ConvertPolicy policy, uint32_t fractional_bits)
    {
        DepthConvertLayerValidationFixedPointFixture<TensorType, AccessorType, FunctionType, T1, T2>::setup(shape, dt_in, dt_out, policy, 0, fractional_bits);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DEPTH_CONVERT_FIXTURE */
