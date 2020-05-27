/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "tests/validation/reference/Permute.h"
#include "tests/validation/reference/Scale.h"

namespace arm_compute
{
namespace test
{
namespace validation
{
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScaleValidationGenericFixture : public framework::Fixture
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, QuantizationInfo quantization_info, DataLayout data_layout, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy,
               bool align_corners)
    {
        _shape             = shape;
        _policy            = policy;
        _border_mode       = border_mode;
        _sampling_policy   = sampling_policy;
        _data_type         = data_type;
        _quantization_info = quantization_info;
        _align_corners     = align_corners && _policy == InterpolationPolicy::BILINEAR && _sampling_policy == SamplingPolicy::TOP_LEFT;

        generate_scale(shape);

        std::mt19937                           generator(library->seed());
        std::uniform_int_distribution<uint8_t> distribution_u8(0, 255);
        _constant_border_value = static_cast<T>(distribution_u8(generator));

        _target    = compute_target(shape, data_layout);
        _reference = compute_reference(shape);
    }

protected:
    void generate_scale(const TensorShape &shape)
    {
        static constexpr float _min_scale{ 0.25f };
        static constexpr float _max_scale{ 3.f };

        constexpr float max_width{ 8192.0f };
        constexpr float max_height{ 6384.0f };

        const float min_width  = _align_corners ? 2.f : 1.f;
        const float min_height = _align_corners ? 2.f : 1.f;

        std::mt19937                          generator(library->seed());
        std::uniform_real_distribution<float> distribution_float(_min_scale, _max_scale);

        auto generate = [&](size_t input_size, float min_output, float max_output) -> float
        {
            const float generated_scale = distribution_float(generator);
            const float output_size     = utility::clamp(static_cast<float>(input_size) * generated_scale, min_output, max_output);
            return output_size / input_size;
        };

        // Input shape is always given in NCHW layout. NHWC is dealt by permute in compute_target()
        const int idx_width  = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(DataLayout::NCHW, DataLayoutDimension::HEIGHT);

        _scale_x = generate(shape[idx_width], min_width, max_width);
        _scale_y = generate(shape[idx_height], min_height, max_height);
    }

    template <typename U>
    void fill(U &&tensor)
    {
        if(is_data_type_float(_data_type))
        {
            library->fill_tensor_uniform(tensor, 0);
        }
        else if(is_data_type_quantized(tensor.data_type()))
        {
            std::uniform_int_distribution<> distribution(0, 100);
            library->fill(tensor, distribution, 0);
        }
        else
        {
            // Restrict range for float to avoid any floating point issues
            std::uniform_real_distribution<> distribution(-5.0f, 5.0f);
            library->fill(tensor, distribution, 0);
        }
    }

    TensorType compute_target(TensorShape shape, DataLayout data_layout)
    {
        // Change shape in case of NHWC.
        if(data_layout == DataLayout::NHWC)
        {
            permute(shape, PermutationVector(2U, 0U, 1U));
        }

        // Create tensors
        TensorType src = create_tensor<TensorType>(shape, _data_type, 1, _quantization_info, data_layout);

        const int idx_width  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
        const int idx_height = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

        TensorShape shape_scaled(shape);
        shape_scaled.set(idx_width, shape[idx_width] * _scale_x);
        shape_scaled.set(idx_height, shape[idx_height] * _scale_y);
        TensorType dst = create_tensor<TensorType>(shape_scaled, _data_type, 1, _quantization_info, data_layout);

        // Create and configure function
        FunctionType scale;

        scale.configure(&src, &dst, _policy, _border_mode, _constant_border_value, _sampling_policy, /* use_padding */ true, _align_corners);

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

    SimpleTensor<T> compute_reference(const TensorShape &shape)
    {
        // Create reference
        SimpleTensor<T> src{ shape, _data_type, 1, _quantization_info };

        // Fill reference
        fill(src);

        return reference::scale<T>(src, _scale_x, _scale_y, _policy, _border_mode, _constant_border_value, _sampling_policy, /* ceil_policy_scale */ false, _align_corners);
    }

    TensorType          _target{};
    SimpleTensor<T>     _reference{};
    TensorShape         _shape{};
    InterpolationPolicy _policy{};
    BorderMode          _border_mode{};
    T                   _constant_border_value{};
    SamplingPolicy      _sampling_policy{};
    DataType            _data_type{};
    QuantizationInfo    _quantization_info{};
    bool                _align_corners{ false };
    float               _scale_x{ 1.f };
    float               _scale_y{ 1.f };
};

template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScaleValidationQuantizedFixture : public ScaleValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, QuantizationInfo quantization_info, DataLayout data_layout, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy,
               bool align_corners)
    {
        ScaleValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                        data_type,
                                                                                        quantization_info,
                                                                                        data_layout,
                                                                                        policy,
                                                                                        border_mode,
                                                                                        sampling_policy,
                                                                                        align_corners);
    }
};
template <typename TensorType, typename AccessorType, typename FunctionType, typename T>
class ScaleValidationFixture : public ScaleValidationGenericFixture<TensorType, AccessorType, FunctionType, T>
{
public:
    template <typename...>
    void setup(TensorShape shape, DataType data_type, DataLayout data_layout, InterpolationPolicy policy, BorderMode border_mode, SamplingPolicy sampling_policy, bool align_corners)
    {
        ScaleValidationGenericFixture<TensorType, AccessorType, FunctionType, T>::setup(shape,
                                                                                        data_type,
                                                                                        QuantizationInfo(),
                                                                                        data_layout,
                                                                                        policy,
                                                                                        border_mode,
                                                                                        sampling_policy,
                                                                                        align_corners);
    }
};
} // namespace validation
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_SCALE_FIXTURE */
