/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#ifndef SRC_CORE_NEON_KERNELS_FUSE_BATCH_NORMALIZATION_GENERIC_IMPL_H
#define SRC_CORE_NEON_KERNELS_FUSE_BATCH_NORMALIZATION_GENERIC_IMPL_H

#include "arm_compute/core/Helpers.h"

#include "src/core/NEON/wrapper/wrapper.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void fused_batch_normalization_conv(const ITensor *conv_weights,
                                    const ITensor *conv_bias,
                                    ITensor       *fused_weights,
                                    ITensor       *fused_bias,
                                    const ITensor *bn_mean,
                                    const ITensor *bn_var,
                                    const ITensor *bn_beta,
                                    const ITensor *bn_gamma,
                                    float          epsilon,
                                    const Window  &window)
{
    using ScalarType   = T;
    const int size     = 16 / conv_weights->info()->element_size();
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    const bool run_in_place_weights = (fused_weights == nullptr) || (fused_weights == conv_weights);
    const bool run_in_place_bias    = (fused_bias == nullptr) || (conv_bias != nullptr && fused_bias == conv_bias);

    // Set build options
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x  = size;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Iterator conv_w_in(conv_weights, win);
    Iterator conv_w_out(run_in_place_weights ? conv_weights : fused_weights, win);

    const auto conv_bias_in =
        (conv_bias != nullptr ? reinterpret_cast<ScalarType *>(conv_bias->ptr_to_element(Coordinates(0, 0))) : nullptr);
    auto conv_bias_out =
        (run_in_place_bias ? conv_bias_in
                           : reinterpret_cast<ScalarType *>(fused_bias->ptr_to_element(Coordinates(0, 0))));

    const auto input_mean  = reinterpret_cast<const ScalarType *>(bn_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const ScalarType *>(bn_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (bn_gamma != nullptr)
                                 ? reinterpret_cast<const ScalarType *>(bn_gamma->ptr_to_element(Coordinates(0, 0)))
                                 : nullptr;
    const auto input_beta  = (bn_beta != nullptr)
                                 ? reinterpret_cast<const ScalarType *>(bn_beta->ptr_to_element(Coordinates(0, 0)))
                                 : nullptr;

    auto       mean_vec    = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       var_vec     = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       gamma_vec   = wrapper::vdup_n(ScalarType(1), ExactTagType{});
    auto       beta_vec    = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       rvar_vec    = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    const auto epsilon_vec = wrapper::vdup_n(ScalarType(epsilon), ExactTagType{});

    auto mean                = ScalarType(0.0);
    auto var                 = ScalarType(0.0);
    auto gamma               = ScalarType(1.0);
    auto beta                = ScalarType(0.0);
    auto conv_bias_in_scalar = ScalarType(0.0);
    execute_window_loop(
        win,
        [&](const Coordinates &id)
        {
            var = input_var[id[3]];
            if (input_gamma != nullptr)
            {
                gamma = input_gamma[id[3]];
            }

            if ((id[0] == 0) && (id[1] == 0) && (id[2] == 0))
            {
                if (input_beta != nullptr)
                {
                    beta     = input_beta[id[3]];
                    beta_vec = wrapper::vdup_n(beta, ExactTagType{});
                }

                // Construct vectors
                mean     = input_mean[id[3]];
                mean_vec = wrapper::vdup_n(mean, ExactTagType{});

                if (conv_bias_in != nullptr)
                {
                    conv_bias_in_scalar = conv_bias_in[id[3]];
                }
                auto conv_bias_tmp_scalar = (conv_bias_in_scalar - mean) / std::sqrt(var + ScalarType(epsilon));
                conv_bias_out[id[3]]      = (conv_bias_tmp_scalar * gamma) + beta;
            }

            int  x              = window_start_x;
            auto conv_w_in_ptr  = reinterpret_cast<const ScalarType *>(conv_w_in.ptr());
            auto conv_w_out_ptr = reinterpret_cast<ScalarType *>(conv_w_out.ptr());
            var_vec             = wrapper::vdup_n(var, ExactTagType{});
            gamma_vec           = wrapper::vdup_n(gamma, ExactTagType{});
            rvar_vec            = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));

            for (; x <= (window_end_x - window_step_x); x += window_step_x)
            {
                auto wn = wrapper::vloadq(conv_w_in_ptr + x);
                wn      = wrapper::vmul(wn, rvar_vec);
                wn      = wrapper::vmul(wn, gamma_vec);

                // Store results
                wrapper::vstore(conv_w_out_ptr + x, wn);
            }

            // Compute left-over elements
            for (; x < window_end_x; ++x)
            {
                *(conv_w_out_ptr + x) = *(conv_w_in_ptr + x) / std::sqrt(var + ScalarType(epsilon)) * gamma;
            }
        },
        conv_w_in, conv_w_out);
}
} // namespace cpu
} // namespace arm_compute
#endif //SRC_CORE_NEON_KERNELS_FUSE_BATCH_NORMALIZATION_GENERIC_IMPL_H
