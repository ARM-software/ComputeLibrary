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
#ifndef ARM_COMPUTE_TEST_NONLINEAR_FILTER_FIXTURE
#define ARM_COMPUTE_TEST_NONLINEAR_FILTER_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/Helpers.h"
#include "tests/validation/reference/NonLinearFilter.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class NonLinearFilterValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, BorderMode border_mode, DataType data_type)
    {
        std::mt19937                           generator(library->seed());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        const uint8_t                          constant_border_value = distribution_u8(generator);

        // Create the mask
        std::vector<uint8_t> mask(mask_size * mask_size);
        fill_mask_from_pattern(mask.data(), mask_size, mask_size, pattern);

        _border_size = BorderSize(static_cast<int>(mask_size / 2));
        _target      = compute_target(shape, data_type, function, mask_size, pattern, mask.data(), border_mode, constant_border_value);
        _reference   = compute_reference(shape, data_type, function, mask_size, pattern, mask.data(), border_mode, constant_border_value);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, DataType data_type, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask, BorderMode border_mode,
                              uint8_t constant_border_value)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, data_type);
        TensorType dst = create_tensor<TensorType>(shape, data_type);

        // Create and configure function
        FunctionType non_linear_filter;
        non_linear_filter.configure(&src, &dst, function, mask_size, pattern, mask, border_mode, constant_border_value);

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
        non_linear_filter.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, DataType data_type, NonLinearFilterFunction function, unsigned int mask_size, MatrixPattern pattern, const uint8_t *mask,
                                      BorderMode border_mode, uint8_t constant_border_value)
    {
        // Create reference
        SimpleTensor<T> src{ shape, data_type };

        // Fill reference
        fill(src);

        return reference::non_linear_filter<T>(src, function, mask_size, pattern, mask, border_mode, constant_border_value);
    }

    BorderMode      _border_mode{};
    BorderSize      _border_size{};
    TensorType      _target{};
    SimpleTensor<T> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_NONLINEAR_FILTER_FIXTURE */
