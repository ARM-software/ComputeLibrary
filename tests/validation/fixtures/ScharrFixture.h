/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_SCHARR_FIXTURE
#define ARM_COMPUTE_TEST_SCHARR_FIXTURE

#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/Types.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Scharr.h"

#include <memory>

namespace arm_compute
{
class CLScharr3x3;
class NEScharr3x3;

namespace test
{
namespace validation
{
namespace
{
template <typename Function>
struct info;

template <>
struct info<NEScharr3x3>
{
    static const Format dst_format  = Format::S16;
    static const int    filter_size = 3;
};

template <>
struct info<CLScharr3x3>
{
    static const Format dst_format  = Format::S16;
    static const int    filter_size = 3;
};
} // namespace

template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename U>
class ScharrValidationFixture : public framework::Fixture
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
        _reference   = compute_reference(shape, info<FunctionType>::filter_size, border_mode, format, constant_border_value, gradient_dimension);
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
        TensorType dst_x = create_tensor<TensorType>(shape, data_type_from_format(info<FunctionType>::dst_format));
        TensorType dst_y = create_tensor<TensorType>(shape, data_type_from_format(info<FunctionType>::dst_format));

        src.info()->set_format(format);
        dst_x.info()->set_format(info<FunctionType>::dst_format);
        dst_y.info()->set_format(info<FunctionType>::dst_format);

        FunctionType scharr;

        switch(gradient_dimension)
        {
            case GradientDimension::GRAD_X:
                scharr.configure(&src, &dst_x, nullptr, border_mode, constant_border_value);
                break;
            case GradientDimension::GRAD_Y:
                scharr.configure(&src, nullptr, &dst_y, border_mode, constant_border_value);
                break;
            case GradientDimension::GRAD_XY:
                scharr.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Gradient dimension not supported");
        }

        ARM_COMPUTE_ASSERT(src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst_x.info()->is_resizable());
        ARM_COMPUTE_ASSERT(dst_y.info()->is_resizable());

        // Allocate tensors
        src.allocator()->allocate();
        dst_x.allocator()->allocate();
        dst_y.allocator()->allocate();

        ARM_COMPUTE_ASSERT(!src.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst_x.info()->is_resizable());
        ARM_COMPUTE_ASSERT(!dst_y.info()->is_resizable());

        // Fill tensors
        fill(AccessorType(src));
        fill_zero(AccessorType(dst_x));
        fill_zero(AccessorType(dst_y));

        // Compute function
        scharr.run();

        return std::make_pair(std::move(dst_x), std::move(dst_y));
    }

    std::pair<SimpleTensor<U>, SimpleTensor<U>> compute_reference(const TensorShape &shape, int filter_size, BorderMode border_mode, Format format, uint8_t constant_border_value,
                                                                  GradientDimension gradient_dimension)
    {
        // Create reference
        SimpleTensor<T> src{ shape, format };

        // Fill reference
        fill(src);

        return reference::scharr<U>(src, filter_size, border_mode, constant_border_value, gradient_dimension);
    }

    BorderMode _border_mode{ BorderMode::UNDEFINED };
    std::pair<TensorType, TensorType>           _target{};
    std::pair<SimpleTensor<U>, SimpleTensor<U>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCHARR_FIXTURE */
