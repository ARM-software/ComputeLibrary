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
#ifndef ARM_COMPUTE_TEST_WARP_PERSPECTIVE_FIXTURE
#define ARM_COMPUTE_TEST_WARP_PERSPECTIVE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Utils.h"
#include "tests/validation/reference/WarpPerspective.h"

#include <random>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class WarpPerspectiveValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, DataType data_type, InterpolationPolicy policy, BorderMode border_mode)
    {
        uint8_t constant_border_value = 0;
        // Generate a random constant value if border_mode is constant
        if(border_mode == BorderMode::CONSTANT)
        {
            std::mt19937                           gen(library->seed());
            std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
            constant_border_value = distribution_u8(gen);
        }

        const TensorShape vmask_shape(input_shape);

        // Create the matrix
        std::array<float, 9> matrix = { { 0 } };
        fill_warp_matrix<9>(matrix);

        _target    = compute_target(input_shape, vmask_shape, matrix.data(), policy, border_mode, constant_border_value, data_type);
        _reference = compute_reference(input_shape, vmask_shape, matrix.data(), policy, border_mode, constant_border_value, data_type);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, const TensorShape &vmask_shape, const float *matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value,
                              DataType data_type)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType warp_perspective;
        warp_perspective.configure(&src, &dst, matrix, policy, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        warp_perspective.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const TensorShape &vmask_shape, const float *matrix, InterpolationPolicy policy, BorderMode border_mode, uint8_t constant_border_value,
                                      DataType data_type)
    {
        ARM_COMPUTE_ERROR_ON(data_type != DataType::U8);

        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Create the valid mask Tensor
        _valid_mask = SimpleTensor<T>(shape, data_type);

        // Fill reference
        fill(src);

        // Compute reference
        return reference::warp_perspective<T>(src, _valid_mask, matrix, policy, border_mode, constant_border_value);
    }

    TensorType      _target{};
    SimpleTensor<T> _reference{};
    BorderMode      _border_mode{};
    SimpleTensor<T> _valid_mask{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_WARP_PERSPECTIVE_FIXTURE */
