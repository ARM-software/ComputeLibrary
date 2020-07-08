/*
 * Copyright (c) 2019 Arm Limited.
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
#include "arm_compute/core/NEON/kernels/NEInstanceNormalizationLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
template <typename T>
void instance_normalization_nchw(ITensor *input, ITensor *output, float gamma, float beta, float epsilon, const Window &window)
{
    /** NEON vector tag type. */
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

        auto sum_h_w         = static_cast<T>(0.f);
        auto sum_squares_h_w = static_cast<T>(0.f);

        execute_window_loop(win_plane, [&](const Coordinates &)
        {
            const auto input_ptr = reinterpret_cast<const T *>(input_plane_it.ptr());

            auto vec_sum_h_w         = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
            auto vec_sum_squares_h_w = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

            // Compute S elements per iteration
            int x = window.x().start();
            for(; x <= (window.x().end() - window_step_x); x += window_step_x)
            {
                auto vec_input_val  = wrapper::vloadq(input_ptr + x);
                vec_sum_h_w         = wrapper::vadd(vec_sum_h_w, vec_input_val);
                vec_sum_squares_h_w = wrapper::vadd(vec_sum_squares_h_w, wrapper::vmul(vec_input_val, vec_input_val));
            }

            auto vec2_sum_h_w         = wrapper::vpadd(wrapper::vgethigh(vec_sum_h_w), wrapper::vgetlow(vec_sum_h_w));
            auto vec2_sum_squares_h_w = wrapper::vpadd(wrapper::vgethigh(vec_sum_squares_h_w), wrapper::vgetlow(vec_sum_squares_h_w));
            for(int i = 0; i < window_step_x / 4; ++i)
            {
                vec2_sum_h_w         = wrapper::vpadd(vec2_sum_h_w, vec2_sum_h_w);
                vec2_sum_squares_h_w = wrapper::vpadd(vec2_sum_squares_h_w, vec2_sum_squares_h_w);
            }
            sum_h_w += wrapper::vgetlane(vec2_sum_h_w, 0);
            sum_squares_h_w += wrapper::vgetlane(vec2_sum_squares_h_w, 0);

            // Compute left-over elements
            for(; x < window.x().end(); ++x)
            {
                const auto value = *(input_ptr + x);
                sum_h_w += value;
                sum_squares_h_w += value * value;
            }
        },
        input_plane_it, output_plane_it);

        const auto mean_h_w = sum_h_w / elements_plane;
        const auto var_h_w  = sum_squares_h_w / elements_plane - mean_h_w * mean_h_w;

        const auto multip_h_w     = gamma / std::sqrt(var_h_w + epsilon);
        const auto vec_mean_h_w   = wrapper::vdup_n(static_cast<T>(mean_h_w), ExactTagType{});
        const auto vec_multip_h_w = wrapper::vdup_n(static_cast<T>(multip_h_w), ExactTagType{});
        const auto vec_beta       = wrapper::vdup_n(static_cast<T>(beta), ExactTagType{});

        execute_window_loop(win_plane, [&](const Coordinates &)
        {
            auto input_ptr  = reinterpret_cast<T *>(input_plane_it.ptr());
            auto output_ptr = reinterpret_cast<T *>(output_plane_it.ptr());

            // Compute S elements per iteration
            int  x       = window.x().start();
            auto vec_val = wrapper::vdup_n(static_cast<T>(0.0f), ExactTagType{});
            for(; x <= (window.x().end() - window_step_x); x += window_step_x)
            {
                vec_val = wrapper::vloadq(input_ptr + x);
                vec_val = wrapper::vadd(wrapper::vmul(wrapper::vsub(vec_val, vec_mean_h_w), vec_multip_h_w), vec_beta);
                wrapper::vstore(output_ptr + x, vec_val);
            }

            // Compute left-over elements
            for(; x < window.x().end(); ++x)
            {
                *(output_ptr + x) = ((*(input_ptr + x)) - mean_h_w) * multip_h_w + beta;
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

void NEInstanceNormalizationLayerKernel::configure(ITensor *input, ITensor *output, float gamma, float beta, float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input);

    _input   = input;
    _output  = output == nullptr ? input : output;
    _gamma   = gamma;
    _beta    = beta;
    _epsilon = epsilon;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(_input->info(), _output->info(), gamma, beta, epsilon));

    if(_input->info()->data_type() == DataType::F32)
    {
        _func = &instance_normalization_nchw<float>;
    }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if(_input->info()->data_type() == DataType::F16)
    {
        _func = &instance_normalization_nchw<float16_t>;
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

Status NEInstanceNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, float gamma, float beta, float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, gamma, beta, epsilon));
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
