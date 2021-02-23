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
#ifndef ARM_COMPUTE_TEST_SCALELAYERFIXTURE
#define ARM_COMPUTE_TEST_SCALELAYERFIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/Globals.h"
#include "tests/Utils.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"

namespace arm_compute
{
namespace test
{
/** Fixture that can be used for Neon, CL and OpenGL ES */
template <typename TensorType, typename Function, typename Accessor, typename T>
class ScaleLayerFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy, float sx, float sy, DataType data_type)
    {
        constexpr float max_width  = 8192.0f;
        constexpr float max_height = 6384.0f;

        std::mt19937 generator(library->seed());

        float scale_x = ((shape.x() * sx) > max_width) ? (max_width / shape.x()) : sx;
        float scale_y = ((shape.y() * sy) > max_height) ? (max_height / shape.y()) : sy;

        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        T                                      constant_border_value = static_cast<T>(distribution_u8(generator));

        // Create tensors
        src = create_tensor<TensorType>(shape, data_type);
        TensorShape shape_scaled(shape);
        shape_scaled.set(0, shape[0] * scale_x);
        shape_scaled.set(1, shape[1] * scale_y);
        dst = create_tensor<TensorType>(shape_scaled, data_type);

        scale_layer.configure(&src, &dst, ScaleKernelInfo{ policy, border_mode, constant_border_value, sampling_policy });

        ARM_COMPUTE_EXPECT(src.info()->is_resizable(), framework::LogLevel::ERRORS);
        ARM_COMPUTE_EXPECT(dst.info()->is_resizable(), framework::LogLevel::ERRORS);

        // Allocate tensors
        src.allocator()->allocate();
        dst.allocator()->allocate();
    }

    void run()
    {
        scale_layer.run();
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
    Function   scale_layer{};
};
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALELAYERFIXTURE */
