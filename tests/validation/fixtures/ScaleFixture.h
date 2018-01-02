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
#ifndef ARM_COMPUTE_TEST_SCALE_FIXTURE
#define ARM_COMPUTE_TEST_SCALE_FIXTURE

#include "arm_compute/core/TensorShape.h"
#include "arm_compute/core/Types.h"
#include "tests/AssetsLibrary.h"
#include "tests/Globals.h"
#include "tests/IAccessor.h"
#include "tests/framework/Asserts.h"
#include "tests/framework/Fixture.h"
#include "tests/validation/reference/Scale.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScaleValidationFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy)
    {
        constexpr float max_width  = 8192.0f;
        constexpr float max_height = 6384.0f;

        _shape           = shape;
        _policy          = policy;
        _border_mode     = border_mode;
        _sampling_policy = sampling_policy;
        _data_type       = data_type;

        std::mt19937                          generator(library->seed());
        std::uniform_real_distribution<float> distribution_float(0.25, 3);
        float                                 scale_x = distribution_float(generator);
        float                                 scale_y = distribution_float(generator);

        scale_x = ((shape.x() * scale_x) > max_width) ? (max_width / shape.x()) : scale_x;
        scale_y = ((shape.y() * scale_y) > max_height) ? (max_height / shape.y()) : scale_y;

        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        T                                      constant_border_value = static_cast<T>(distribution_u8(generator));

        _target    = compute_target(shape, scale_x, scale_y, policy, border_mode, constant_border_value, sampling_policy);
        _reference = compute_reference(shape, scale_x, scale_y, policy, border_mode, constant_border_value, sampling_policy);
    }

protected:
    template <typename U>
    void fill(U &&tensor)
    {
        library->fill_tensor_uniform(tensor, 0);
    }

    TensorType compute_target(const TensorShape &shape, const float scale_x, const float scale_y,
                              InterpolationPolicy policy, BorderMode border_mode, T constant_border_value, SamplingPolicy sampling_policy)
    {
        // Create tensors
        TensorType  src = create_tensor<TensorType>(shape, _data_type);
        TensorShape shape_scaled(shape);
        shape_scaled.set(0, shape[0] * scale_x);
        shape_scaled.set(1, shape[1] * scale_y);
        TensorType dst = create_tensor<TensorType>(shape_scaled, _data_type);

        // Create and configure function
        FunctionType scale;

        scale.configure(&src, &dst, policy, border_mode, constant_border_value, sampling_policy);

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
        scale.run();

        return dst;
    }

    SimpleTensor<T> compute_reference(const TensorShape &shape, const float scale_x, const float scale_y,
                                      InterpolationPolicy policy, BorderMode border_mode, T constant_border_value, SamplingPolicy sampling_policy)
    {
        // Create reference
        SimpleTensor<T> src{ shape, _data_type };

        // Fill reference
        fill(src);

        return reference::scale<T>(src, scale_x, scale_y, policy, border_mode, constant_border_value, sampling_policy);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    TensorShape         _shape{};
    InterpolationPolicy _policy{};
    BorderMode          _border_mode{};
    SamplingPolicy      _sampling_policy{};
    DataType            _data_type{};
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_FIXTURE */
