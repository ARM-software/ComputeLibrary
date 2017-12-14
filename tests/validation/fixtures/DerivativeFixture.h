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
#ifndef ARM_COMPUTE_TEST_DERIVATIVE_FIXTURE
#define ARM_COMPUTE_TEST_DERIVATIVE_FIXTURE

#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/Types.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Derivative.h"

#include <memory>

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename U>
class DerivativeValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, BorderMode border_mode, Format format, GradientDimension gradient_dimension)
    {
        // Generate a random constant value
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        const uint8_t                          constant_border_value = int_dist(gen);

        _border_mode = border_mode;
        _target      = compute_target(shape, border_mode, format, constant_border_value, gradient_dimension);
        _reference   = compute_reference(shape, border_mode, format, constant_border_value, gradient_dimension);
    }

protected:
    template <typename V>
    void fill(V &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    template <typename V>
    void fill_zero(V &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0, static_cast<U>(0), static_cast<U>(0));
    }

    std::pair<TensorType, TensorType> compute_target(const TensorShape &shape, BorderMode border_mode, Format format, uint8_t constant_border_value, GradientDimension gradient_dimension)
    {
        // Create tensors
        TensorType src   = create_tensor<TensorType>(shape, data_type_from_format(format));
        TensorType dst_x = create_tensor<TensorType>(shape, data_type_from_format(Format::S16));
        TensorType dst_y = create_tensor<TensorType>(shape, data_type_from_format(Format::S16));

        src.info()->set_format(format);
        dst_x.info()->set_format(Format::S16);
        dst_y.info()->set_format(Format::S16);

        FunctionType derivative;

        switch(gradient_dimension)
        {
            case GradientDimension::GRAD_X:
                derivative.configure(&src, &dst_x, nullptr, border_mode, constant_border_value);
                break;
            case GradientDimension::GRAD_Y:
                derivative.configure(&src, nullptr, &dst_y, border_mode, constant_border_value);
                break;
            case GradientDimension::GRAD_XY:
                derivative.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Gradient dimension not supported");
        }

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst_x.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst_y.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst_x.allocator()->allocate();
        dst_y.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst_x.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!dst_y.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Fill tensors
        fill(AccessorType(src));
        fill_zero(AccessorType(dst_x));
        fill_zero(AccessorType(dst_y));

        // Compute function
        derivative.run();

        return std::make_pair(std::move(dst_x), std::move(dst_y));
    }

    std::pair<SimpleTensor<U>, SimpleTensor<U>> compute_reference(const TensorShape &shape, BorderMode border_mode, Format format, uint8_t constant_border_value, GradientDimension gradient_dimension)
    {
        // Create reference
        SimpleTensor<T> src{ shape, format };

        // Fill reference
        fill(src);

        return reference::derivative<U>(src, border_mode, constant_border_value, gradient_dimension);
    }

    BorderMode _border_mode{ BorderMode::UNDEFINED };
    std::pair<TensorType, TensorType>           _target{};
    std::pair<SimpleTensor<U>, SimpleTensor<U>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_DERIVATIVE_FIXTURE */
