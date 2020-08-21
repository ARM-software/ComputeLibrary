/*
 * Copyright (c) 2018 Arm Limited.
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
#ifndef ARM_COMPUTE_TEST_LAPLACIAN_PYRAMID_FIXTURE
#define ARM_COMPUTE_TEST_LAPLACIAN_PYRAMID_FIXTURE

#include "arm_compute/core/IPyramid.h"
#include "arm_compute/core/PyramidInfo.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/LaplacianPyramid.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T, typename U, typename PyramidType>
class LaplacianPyramidValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape input_shape, BorderMode border_mode, size_t num_levels, Format format_in, Format format_out)
    {
        std::mt19937                     generator(library->seed());
        std::uniform_int_distribution<T> distribution_u8(0, 255);
        const T                          constant_border_value = distribution_u8(generator);

        _pyramid_levels = num_levels;
        _border_mode    = border_mode;

        _target    = compute_target(input_shape, border_mode, constant_border_value, format_in, format_out);
        _reference = compute_reference(input_shape, border_mode, constant_border_value, format_in, format_out);
    }

protected:
    template <typename V>
    void fill(V &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    PyramidType compute_target(const TensorShape &input_shape, BorderMode border_mode, T constant_border_value,
                               Format format_in, Format format_out)
    {
        // Create pyramid
        PyramidType pyramid{};

        // Create Pyramid Info
        PyramidInfo pyramid_info(_pyramid_levels, SCALE_PYRAMID_HALF, input_shape, format_out);

        // Use conservative padding strategy to fit all subsequent kernels
        pyramid.init_auto_padding(pyramid_info);

        // Create tensors
        TensorType src = create_tensor<TensorType>(input_shape, format_in);

        // The first two dimensions of the output tensor must match the first
        // two dimensions of the tensor in the last level of the pyramid
        TensorShape dst_shape(input_shape);
        dst_shape.set(0, pyramid.get_pyramid_level(_pyramid_levels - 1)->info()->dimension(0));
        dst_shape.set(1, pyramid.get_pyramid_level(_pyramid_levels - 1)->info()->dimension(1));

        // The lowest resolution tensor necessary to reconstruct the input
        // tensor from the pyramid.
        _dst_target = create_tensor<TensorType>(dst_shape, format_out);

        // Create and configure function
        FunctionType laplacian_pyramid;
        laplacian_pyramid.configure(&src, &pyramid, &_dst_target, border_mode, constant_border_value);

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(_dst_target.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        _dst_target.allocator()->allocate();

        ARM_COMPUTE_EXPECT(!src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(!_dst_target.info()->is_resizable(), framework::LogLevel::ERRORS);

        pyramid.allocate();

        for(size_t i = 0; i < pyramid_info.num_levels(); ++i)
        {
            ARM_COMPUTE_EXPECT(!pyramid.get_pyramid_level(i)->info()->is_resizable(), framework::LogLevel::ERRORS);
        }

        // Fill tensors
        fill(AccessorType(src));

        // Compute function
        laplacian_pyramid.run();

        return pyramid;
    }

    std::vector<SimpleTensor<U>> compute_reference(const TensorShape &shape, BorderMode border_mode, T constant_border_value,
                                                   Format format_in, Format format_out)
    {
        // Create reference
        SimpleTensor<T> src{ shape, format_in };

        // The first two dimensions of the output tensor must match the first
        // two dimensions of the tensor in the last level of the pyramid
        TensorShape dst_shape(shape);
        dst_shape.set(0, static_cast<float>(shape[0] + 1) / static_cast<float>(std::pow(2, _pyramid_levels - 1)));
        dst_shape.set(1, static_cast<float>(shape[1] + 1) / static_cast<float>(std::pow(2, _pyramid_levels - 1)));

        _dst_reference = SimpleTensor<U>(dst_shape, format_out);

        // Fill reference
        fill(src);

        return reference::laplacian_pyramid<T, U>(src, _dst_reference, _pyramid_levels, border_mode, constant_border_value);
    }

    size_t                       _pyramid_levels{};
    BorderMode                   _border_mode{};
    SimpleTensor<U>              _dst_reference{};
    TensorType                   _dst_target{};
    PyramidType                  _target{};
    std::vector<SimpleTensor<U>> _reference{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_LAPLACIAN_PYRAMID_FIXTURE */
