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
#ifndef ARM_COMPUTE_TEST_GAUSSIAN_PYRAMID_HALF_FIXTURE
#define ARM_COMPUTE_TEST_GAUSSIAN_PYRAMID_HALF_FIXTURE

#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/GaussianPyramidHalf.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename PyramidType>
class GaussianPyramidHalfValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, BorderMode border_mode, size_t num_levels)
    {
        std::mt19937                           gen(library->seed());
        std::uniform_int_distribution<uint8_t> distribution(0, 255);
        const uint8_t                          constant_border_value = distribution(gen);

        _border_mode = border_mode;

        // Compute target and reference
        compute_target(shape, border_mode, constant_border_value, num_levels);
        compute_reference(shape, border_mode, constant_border_value, num_levels);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    void compute_target(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value, size_t num_levels)
    {
        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, DataType::U8);

        PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, shape, Format::U8);
        _target.init(pyramid_info);

        // Create and configure function
        FunctionType gaussian_pyramid;

        gaussian_pyramid.configure(&src, &_target, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        for(size_t i = 0; i < pyramid_info.num_levels(); ++i)
        {
            ARM_COMPUTE_EXPECT(_target.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Allocate input tensor
        src.allocator()->allocate();

        // Allocate pyramid
        _target.allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        for(size_t i = 0; i < pyramid_info.num_levels(); ++i)
        {
            ARM_COMPUTE_EXPECT(!_target.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        gaussian_pyramid.run();
    }

    void compute_reference(const TensorShape &shape, BorderMode border_mode, uint8_t constant_border_value, size_t num_levels)
    {
        // Create reference
        SimpleTensor<T> src{ shape, DataType::U8 };

        // Fill reference
        fill(src);

        _reference = reference::gaussian_pyramid_half<T>(src, border_mode, constant_border_value, num_levels);
    }

    PyramidType                  _target{};
    std::vector<SimpleTensor<T>> _reference{};
    BorderMode                   _border_mode{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_GAUSSIAN_PYRAMID_HALF_FIXTURE */