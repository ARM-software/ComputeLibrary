/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEInstanceNormalizationLayerKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/KernelDescriptors.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
template <typename InputType, typename AccType = InputType>
void vector_float_sum(AccType &result, AccType &result_square, const InputType &inputs)
{
    result        = wrapper::vadd(result, inputs);
    result_square = wrapper::vadd(result_square, wrapper::vmul(inputs, inputs));
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline void vector_float_sum(float32x4_t &result, float32x4_t &result_square, const float16x8_t &inputs)
{
    vector_float_sum(result, result_square, wrapper::vcvt<float>(wrapper::vgetlow(inputs)));
    vector_float_sum(result, result_square, wrapper::vcvt<float>(wrapper::vgethigh(inputs)));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename InputType, typename AccType = InputType>
InputType vector_float_norm(const InputType &inputs, const AccType &vec_mean, const AccType &vec_multip, const AccType &vec_beta)
{
    return wrapper::vadd(wrapper::vmul(wrapper::vsub(inputs, vec_mean), vec_multip), vec_beta);
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
inline float16x8_t vector_float_norm(const float16x8_t &inputs, const float32x4_t &vec_mean, const float32x4_t &vec_multip, const float32x4_t &vec_beta)
{
    const auto  input_low   = wrapper::vcvt<float>(wrapper::vgetlow(inputs));
    const auto  input_high  = wrapper::vcvt<float>(wrapper::vgethigh(inputs));
    const auto  result_low  = wrapper::vcvt<float16_t>(vector_float_norm(input_low, vec_mean, vec_multip, vec_beta));
    const auto  result_high = wrapper::vcvt<float16_t>(vector_float_norm(input_high, vec_mean, vec_multip, vec_beta));
    float16x8_t result      = wrapper::vcombine(result_low, result_high);

    return result;
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <typename T, typename AccType = T>
void instance_normalization_nchw(ITensor *input, ITensor *output, float gamma, float beta, float epsilon, const Window &window)
{
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    // Clear X/Y dimensions on execution window as we handle the planes manually
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));
    win.set(Window::DimY, Window::Dimension(0, 1, 1));

    constexpr int      window_step_x  = 16 / sizeof(T);
    const unsigned int elements_plane = input->info()->dimension(0) * output->info()->dimension(1);

    Iterator input_it(input, win);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        Window win_plane = window;
        win_plane.set(Window::DimX, Window::Dimension(0, 1, 1));
        win_plane.set(Window::DimZ, Window::Dimension(id[2], id[2] + 1, 1));
        win_plane.set(3, Window::Dimension(id[3], id[3] + 1, 1));

        Iterator input_plane_it(input, win_plane);
        Iterator output_plane_it(output, win_plane);

        auto sum_h_w         = static_cast<AccType>(0.f);
        auto sum_squares_h_w = static_cast<AccType>(0.f);

        execute_window_loop(win_plane, [&](const Coordinates &)
        {
            const auto input_ptr = reinterpret_cast<const T *>(input_plane_it.ptr());

            auto vec_sum_h_w         = wrapper::vdup_n(static_cast<AccType>(0.f), ExactTagType{});
            auto vec_sum_squares_h_w = wrapper::vdup_n(static_cast<AccType>(0.f), ExactTagType{});

            // Compute S elements per iteration
            int x = window.x().start();
            for(; x <= (window.x().end() - window_step_x); x += window_step_x)
            {
                auto vec_input_val = wrapper::vloadq(input_ptr + x);
                vector_float_sum(vec_sum_h_w, vec_sum_squares_h_w, vec_input_val);
            }

            auto vec2_sum_h_w         = wrapper::vpadd(wrapper::vgethigh(vec_sum_h_w), wrapper::vgetlow(vec_sum_h_w));
            auto vec2_sum_squares_h_w = wrapper::vpadd(wrapper::vgethigh(vec_sum_squares_h_w), wrapper::vgetlow(vec_sum_squares_h_w));

            vec2_sum_h_w         = wrapper::vpadd(vec2_sum_h_w, vec2_sum_h_w);
            vec2_sum_squares_h_w = wrapper::vpadd(vec2_sum_squares_h_w, vec2_sum_squares_h_w);

            sum_h_w += wrapper::vgetlane(vec2_sum_h_w, 0);
            sum_squares_h_w += wrapper::vgetlane(vec2_sum_squares_h_w, 0);

            // Compute left-over elements
            for(; x < window.x().end(); ++x)
            {
                const auto value = static_cast<AccType>(*(input_ptr + x));
                sum_h_w += value;
                sum_squares_h_w += value * value;
            }
        },
        input_plane_it, output_plane_it);

        const auto mean_h_w = sum_h_w / elements_plane;
        const auto var_h_w  = sum_squares_h_w / elements_plane - mean_h_w * mean_h_w;

        const auto multip_h_w     = gamma / std::sqrt(var_h_w + epsilon);
        const auto vec_mean_h_w   = wrapper::vdup_n(static_cast<AccType>(mean_h_w), ExactTagType{});
        const auto vec_multip_h_w = wrapper::vdup_n(static_cast<AccType>(multip_h_w), ExactTagType{});
        const auto vec_beta       = wrapper::vdup_n(static_cast<AccType>(beta), ExactTagType{});

        execute_window_loop(win_plane, [&](const Coordinates &)
        {
            auto input_ptr  = reinterpret_cast<T *>(input_plane_it.ptr());
            auto output_ptr = reinterpret_cast<T *>(output_plane_it.ptr());

            // Compute S elements per iteration
            int x = window.x().start();
            //auto vec_val = wrapper::vdup_n(static_cast<T>(0.0f), ExactTagType{});
            for(; x <= (window.x().end() - window_step_x); x += window_step_x)
            {
                const auto vec_val        = wrapper::vloadq(input_ptr + x);
                const auto normalized_vec = vector_float_norm(vec_val, vec_mean_h_w, vec_multip_h_w, vec_beta);
                wrapper::vstore(output_ptr + x, normalized_vec);
            }

            // Compute left-over elements
            for(; x < window.x().end(); ++x)
            {
                const auto val    = static_cast<AccType>(*(input_ptr + x));
                *(output_ptr + x) = static_cast<T>((val - mean_h_w) * multip_h_w + beta);
            }
        },
        input_plane_it, output_plane_it);
    },
    input_it);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, float gamma, float beta, float epsilon)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_UNUSED(gamma);
    ARM_COMPUTE_UNUSED(beta);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(epsilon == 0.f, "Epsilon must be different than 0");

    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_layout() == DataLayout::NHWC, "NHWC data layout is not supported by the kernel directly");

    if(output != nullptr && output->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_channels() != output->num_channels(), "Input and output have different number of channels");
    }
    return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
    // We handle the planes manually
    Window win = calculate_max_window(*input, Steps(1));

    // Output auto initialization if not yet initialized
    auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

    // NEInstanceNormalizationLayerKernel doesn't need padding so update_window_and_padding() can be skipped
    Coordinates coord;
    coord.set_num_dimensions(output->num_dimensions());
    output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    return std::make_pair(Status{}, win);
}
} // namespace

NEInstanceNormalizationLayerKernel::NEInstanceNormalizationLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _gamma(1), _beta(0), _epsilon(1e-12)
{
}

void NEInstanceNormalizationLayerKernel::configure(ITensor *input, ITensor *output, const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _input               = input;
    _output              = output == nullptr ? input : output;
    _gamma               = info.gamma;
    _beta                = info.beta;
    _epsilon             = info.epsilon;
    _use_mixed_precision = info.use_mixed_precision;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(_input->info(), _output->info(), _gamma, _beta, _epsilon));

    if(_input->info()->data_type() == DataType::F32)
    {
        _func = &instance_normalization_nchw<float>;
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if(_input->info()->data_type() == DataType::F16)
    {
        if(_use_mixed_precision)
        {
            _func = &instance_normalization_nchw<float16_t, float>;
        }
        else
        {
            _func = &instance_normalization_nchw<float16_t>;
        }
    }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else
    {
        ARM_COMPUTE_ERROR("Unsupported data type");
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(_input->info(), _output->info());
    ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

    INEKernel::configure(std::get<1>(win_config));
}

Status NEInstanceNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const InstanceNormalizationLayerKernelInfo &info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, info.gamma, info.beta, info.epsilon));
    ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(input->clone().get(), (output == nullptr ? input->clone().get() : output->clone().get()))));
    return Status{};
}

void NEInstanceNormalizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    (*_func)(_input, _output, _gamma, _beta, _epsilon, window);
}
} // namespace arm_compute
