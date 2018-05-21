/*
 * Copyright (c) 2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SOBEL_FIXTURE
#define ARM_COMPUTE_TEST_SOBEL_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Types.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
class CLSobel3x3;
class CLSobel5x5;
class CLSobel7x7;
class NESobel3x3;
class NESobel5x5;
class NESobel7x7;

namespace test
{
namespace benchmark
{
namespace
{
template <typename Function>
struct info;

template <>
struct info<NESobel3x3>
{
    static const Format dst_format = Format::S16;
};

template <>
struct info<CLSobel3x3>
{
    static const Format dst_format = Format::S16;
};

template <>
struct info<NESobel5x5>
{
    static const Format dst_format = Format::S16;
};

template <>
struct info<CLSobel5x5>
{
    static const Format dst_format = Format::S16;
};

template <>
struct info<NESobel7x7>
{
    static const Format dst_format = Format::S32;
};

template <>
struct info<CLSobel7x7>
{
    static const Format dst_format = Format::S32;
};
} // namespace

template <typename TensorType, typename Function, typename Accessor>
class SobelFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(const TensorShape &input_shape, BorderMode border_mode, GradientDimension gradient_dimension, Format input_format)
    {
        // Generate a random constant value
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> int_dist(0, 255);
        const uint8_t                          constant_border_value = int_dist(gen);

        // Create source tensor
        src = create_tensor<TensorType>(input_shape, input_format);

        // Create destination tensors and configure function
        switch(gradient_dimension)
        {
            case GradientDimension::GRAD_X:
                dst_x = create_tensor<TensorType>(input_shape, info<Function>::dst_format);
                sobel_func.configure(&src, &dst_x, nullptr, border_mode, constant_border_value);
                break;
            case GradientDimension::GRAD_Y:
                dst_y = create_tensor<TensorType>(input_shape, info<Function>::dst_format);
                sobel_func.configure(&src, nullptr, &dst_y, border_mode, constant_border_value);
                break;
            case GradientDimension::GRAD_XY:
                dst_x = create_tensor<TensorType>(input_shape, info<Function>::dst_format);
                dst_y = create_tensor<TensorType>(input_shape, info<Function>::dst_format);
                sobel_func.configure(&src, &dst_x, &dst_y, border_mode, constant_border_value);
                break;
            default:
                ARM_COMPUTE_ERROR("Gradient dimension not supported");
        }

        // Allocate tensors
        src.allocator()->allocate();
        dst_x.allocator()->allocate();
        dst_y.allocator()->allocate();
    }

    void run()
    {
        sobel_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst_x);
        sync_tensor_if_necessary<TensorType>(dst_y);
    }

    void teardown()
    {
        src.allocator()->free();
        dst_x.allocator()->free();
        dst_y.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst_x{};
    TensorType dst_y{};
    Function   sobel_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SOBEL_FIXTURE */
