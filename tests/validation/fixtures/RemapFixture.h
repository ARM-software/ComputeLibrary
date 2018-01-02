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
#ifndef ARM_COMPUTE_TEST_REMAP_FIXTURE
#define ARM_COMPUTE_TEST_REMAP_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Remap.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class RemapValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        T                                      constant_border_value = static_cast<T>(distribution(gen));

        _target    = compute_target(shape, policy, data_type, border_mode, constant_border_value);
        _reference = compute_reference(shape, policy, data_type, border_mode, constant_border_value);
    }

protected:
    template <typename U>
    void fill(U &&tensor, int i)
    {
        library->fill_tensor_uniform(tensor, i);
    }

    TensorType compute_target(const TensorShape &shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode, T constant_border_value)
    {
        // Create tensors
        TensorType src   = create_tensor<TensorType>(shape, data_type);
        TensorType map_x = create_tensor<TensorType>(shape, DataType::F32);
        TensorType map_y = create_tensor<TensorType>(shape, DataType::F32);
        TensorType dst   = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType remap;
        remap.configure(&src, &map_x, &map_y, &dst, policy, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(map_x.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(map_y.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        map_x.allocator()->allocate();
        map_y.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!map_x.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!map_y.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src), 0);
        fill(AccessorType(map_x), 1);
        fill(AccessorType(map_y), 2);

        // Compute function
        remap.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, InterpolationPolicy policy, DataType data_type, BorderMode border_mode, T constant_border_value)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::U8);

        // Create reference
        SimpleTensor<T>     src{ shape, data_type };
        SimpleTensor<float> map_x{ shape, DataType::F32 };
        SimpleTensor<float> map_y{ shape, DataType::F32 };

        // Create the valid mask Tensor
        _valid_mask = SimpleTensor<T> { shape, data_type };

        // Fill reference
        fill(src, 0);
        fill(map_x, 1);
        fill(map_y, 2);

        // Compute reference
        return reference::remap<T>(src, map_x, map_y, _valid_mask, policy, border_mode, constant_border_value);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    SimpleTensor<T> _valid_mask{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_REMAP_FIXTURE */
