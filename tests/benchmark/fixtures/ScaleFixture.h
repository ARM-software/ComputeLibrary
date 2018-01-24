/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_SCALE_FIXTURE
#define ARM_COMPUTE_TEST_SCALE_FIXTURE

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
template <typename TensorType, typename Function, typename Accessor>
class ScaleFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy)
    {
        constexpr float max_width  = 8192.0f;
        constexpr float max_height = 6384.0f;

        std::mt19937                          generator(library->seed());
        std::uniform_real_distribution<float> distribution_float(0.25f, 3.0f);
        float                                 scale_x = distribution_float(generator);
        float                                 scale_y = distribution_float(generator);

        scale_x = ((shape.x() * scale_x) > max_width) ? (max_width / shape.x()) : scale_x;
        scale_y = ((shape.y() * scale_y) > max_height) ? (max_height / shape.y()) : scale_y;

        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        uint8_t                                constant_border_value = static_cast<uint8_t>(distribution_u8(generator));

        TensorShape shape_scaled(shape);
        shape_scaled.set(0, shape[0] * scale_x);
        shape_scaled.set(1, shape[1] * scale_y);

        // Create tensors
        src = create_tensor<TensorType>(shape, data_type);
        dst = create_tensor<TensorType>(shape_scaled, data_type);

        // Create and configure function
        scale_func.configure(&src, &dst, policy, border_mode, constant_border_value, sampling_policy);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        scale_func.run();
    }

    void sync()
    {
        sync_if_necessary<TensorType>();
        sync_tensor_if_necessary<TensorType>(dst);
    }

    void teardown()
    {
        src.allocator()->free();
        dst.allocator()->free();
    }

private:
    TensorType src{};
    TensorType dst{};
    Function   scale_func{};
};
} // namespace benchmark
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_FIXTURE */
