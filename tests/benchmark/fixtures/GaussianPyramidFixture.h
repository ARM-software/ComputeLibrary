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
#ifndef ARM_COMPUTE_GAUSSIAN_PYRAMID_HALF_FIXTURE
#define ARM_COMPUTE_GAUSSIAN_PYRAMID_HALF_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
namespace benchmark
{
template <typename TensorType, typename Function, typename Accessor, typename PyramidType>
class GaussianPyramidHalfFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, BorderMode border_mode, size_t num_levels)
    {
        const uint8_t constant_border_value = 0;

        // Create tensors
        src = create_tensor<TensorType>(shape, DataType::U8);
        PyramidInfo pyramid_info(num_levels, SCALE_PYRAMID_HALF, shape, Format::U8);
        pyramid.init(pyramid_info);

        // Configure function
        gaussian_pyramid_func.configure(&src, &pyramid, border_mode, constant_border_value);

        // Allocate input tensor and pyramid
        src.allocator()->allocate();
        pyramid.allocate();

        // Fill tensor
        library->fill_tensor_uniform(Accessor(src), 0);
    }

    void run()
    {
        gaussian_pyramid_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
    }

private:
    TensorType  src{};
    PyramidType pyramid{};
    Function    gaussian_pyramid_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_GAUSSIAN_PYRAMID_HALF_FIXTURE */
