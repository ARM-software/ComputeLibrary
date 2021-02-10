/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensorPack.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/core/common/Validate.h"

#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace arm_compute
{
namespace
{
using BatchNomalizationPtr = void (*)(ITensor *src, ITensor *dst, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma,
                                      float epsilon, ActivationLayerInfo &act_info, const Window &window);

template <typename T>
void batch_normalization(ITensor *src, ITensor *dst, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma,
                         float epsilon, ActivationLayerInfo &act_info, const Window &window)
{
    /** Neon vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<float, wrapper::traits::BitWidth::W128>;

    const int  window_step_x  = 4;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(src, win_collapsed);
    Iterator output(dst, win_collapsed);

    const auto input_mean  = reinterpret_cast<const float *>(mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float *>(var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (gamma != nullptr) ? reinterpret_cast<const float *>(gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (beta != nullptr) ? reinterpret_cast<const float *>(beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    T activation_functor(act_info);

    const auto epsilon_vec = wrapper::vdup_n(static_cast<float>(epsilon), ExactTagType{});
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const float *>(input.ptr());
        const auto output_ptr = reinterpret_cast<float *>(output.ptr());

        // Perform core calculations using vector operations
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Conctruct vectors
            const auto mean_vec  = wrapper::vloadq(input_mean + x);
            const auto var_vec   = wrapper::vloadq(input_var + x);
            const auto gamma_vec = (input_gamma != nullptr) ? wrapper::vloadq(input_gamma + x) : wrapper::vdup_n(static_cast<float>(1.f), ExactTagType{});
            const auto beta_vec  = (input_beta != nullptr) ? wrapper::vloadq(input_beta + x) : wrapper::vdup_n(static_cast<float>(0.f), ExactTagType{});

            // Calculate denominator
            const auto denominator = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));

            // Calculate x bar
            const auto numerator = wrapper::vsub(wrapper::vloadq(input_ptr + x), mean_vec);
            const auto x_bar     = wrapper::vmul(numerator, denominator);
            auto       res       = wrapper::vmla(beta_vec, x_bar, gamma_vec);

            // Perform fused activation
            if(act_info.enabled())
            {
                activation_functor(res);
            }

            // Store results
            wrapper::vstore(output_ptr + x, res);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            // Conctruct vectors
            const float gamma = (input_gamma != nullptr) ? input_gamma[x] : 1.f;
            const float beta  = (input_beta != nullptr) ? input_beta[x] : 0.f;

            const float denominator = sqrt(input_var[x] + epsilon);
            const float numerator   = input_ptr[x] - input_mean[x];
            const float x_bar       = numerator / denominator;
            float       res         = beta + x_bar * gamma;

            // Perform fused activation
            if(act_info.enabled())
            {
                activation_functor(res);
            }

            // Store results
            *reinterpret_cast<float *>(output_ptr + x) = res;
        }
    },
    input, output);
}

// Fused Batched Normalization with activation functions
static std::map<ActivationLayerInfo::ActivationFunction, BatchNomalizationPtr> fused_map =
{
    { ActivationLayerInfo::ActivationFunction::RELU, &batch_normalization<detail::relu<float, 4>> },
    { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &batch_normalization<detail::brelu<float, 4>> },
    { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &batch_normalization<detail::lubrelu<float, 4>> }
};
}
namespace cpu
{
void fp32_neon_batch_normalization(ITensor *src, ITensor *dst, const ITensor *mean, const ITensor *var, const ITensor *beta, const ITensor *gamma,
                                   float epsilon, ActivationLayerInfo &act_info, const Window &window)
{
    if(act_info.enabled())
    {
        fused_map[act_info.activation()](src, dst, mean, var, beta, gamma, epsilon, act_info, window);
    }
    else
    {
        batch_normalization<detail::dummy<float, 4>>(src, dst, mean, var, beta, gamma, epsilon, act_info, window);
    }
}
} // namespace cpu
} // namespace arm_compute
