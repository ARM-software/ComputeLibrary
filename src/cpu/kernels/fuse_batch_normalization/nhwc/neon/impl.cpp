/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "src/cpu/kernels/fuse_batch_normalization/nhwc/neon/impl.h"

namespace arm_compute
{
namespace cpu
{
template <typename T>
void fused_batch_normalization_dwc_nhwc(const ITensor *dwc_weights, const ITensor *dwc_bias, ITensor *fused_weights, ITensor *fused_bias,
                                        const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window)
{
    using ScalarType   = T;
    const int size     = 16 / dwc_weights->info()->element_size();
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    const bool run_in_place_weights = (fused_weights == nullptr) || (fused_weights == dwc_weights);
    const bool run_in_place_bias    = (fused_bias == nullptr) || (dwc_bias != nullptr && fused_bias == dwc_bias);

    // Set build options
    Window win = window;
    win.set(Window::DimX, Window::Dimension(0, 1, 1));

    const int  window_step_x  = size;
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Iterator dwc_w_in(dwc_weights, win);
    Iterator dwc_w_out(run_in_place_weights ? dwc_weights : fused_weights, win);

    const auto dwc_bias_in  = (dwc_bias != nullptr ? reinterpret_cast<ScalarType *>(dwc_bias->ptr_to_element(Coordinates(0, 0))) : nullptr);
    auto       dwc_bias_out = (run_in_place_bias ? dwc_bias_in : reinterpret_cast<ScalarType *>(fused_bias->ptr_to_element(Coordinates(0, 0))));

    const auto input_mean  = reinterpret_cast<const ScalarType *>(bn_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const ScalarType *>(bn_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (bn_gamma != nullptr) ? reinterpret_cast<const ScalarType *>(bn_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (bn_beta != nullptr) ? reinterpret_cast<const ScalarType *>(bn_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    auto       mean_vec     = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       var_vec      = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       gamma_vec    = wrapper::vdup_n(ScalarType(1), ExactTagType{});
    auto       beta_vec     = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       rvar_vec     = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       dwc_bias_vec = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    const auto epsilon_vec  = wrapper::vdup_n(ScalarType(epsilon), ExactTagType{});

    auto gamma              = ScalarType(1.0);
    auto beta               = ScalarType(0.0);
    auto dwc_bias_in_scalar = ScalarType(0);

    execute_window_loop(win, [&](const Coordinates & id)
    {
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            var_vec = wrapper::vloadq(input_var + x);
            if(input_gamma != nullptr)
            {
                gamma_vec = wrapper::vloadq(input_gamma + x);
            }

            if((id[2] == 0) && (id[1] == 0))
            {
                mean_vec = wrapper::vloadq(input_mean + x);

                // Construct vectors
                if(input_beta != nullptr)
                {
                    beta_vec = wrapper::vloadq(input_beta + x);
                }

                if(dwc_bias_in != nullptr)
                {
                    dwc_bias_vec = wrapper::vloadq(dwc_bias_in + x);
                }

                auto dwc_bias_tmp_vec = wrapper::vmul(wrapper::vsub(dwc_bias_vec, mean_vec), wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec)));
                dwc_bias_tmp_vec      = wrapper::vadd(wrapper::vmul(dwc_bias_tmp_vec, gamma_vec), beta_vec);
                wrapper::vstore(dwc_bias_out + x, dwc_bias_tmp_vec);
            }

            auto dwc_w_in_ptr  = reinterpret_cast<const ScalarType *>(dwc_w_in.ptr());
            auto dwc_w_out_ptr = reinterpret_cast<ScalarType *>(dwc_w_out.ptr());

            auto wn  = wrapper::vloadq(dwc_w_in_ptr + x);
            rvar_vec = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));
            wn       = wrapper::vmul(wn, rvar_vec);
            wn       = wrapper::vmul(wn, gamma_vec);

            // Store results
            wrapper::vstore(dwc_w_out_ptr + x, wn);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            auto var = input_var[x];
            if(input_gamma != nullptr)
            {
                gamma = input_gamma[x];
            }

            if(id[2] == 0 && id[1] == 0)
            {
                auto mean = input_mean[x];
                if(input_beta != nullptr)
                {
                    beta = input_beta[x];
                }
                if(dwc_bias_in != nullptr)
                {
                    dwc_bias_in_scalar = dwc_bias_in[x];
                }

                auto dwc_bias_tmp_scalar = (dwc_bias_in_scalar - mean) / std::sqrt(var + ScalarType(epsilon));
                dwc_bias_out[x]          = (dwc_bias_tmp_scalar * gamma) + beta;
            }

            const auto dwc_w_in_ptr  = reinterpret_cast<const ScalarType *>(dwc_w_in.ptr());
            auto       dwc_w_out_ptr = reinterpret_cast<ScalarType *>(dwc_w_out.ptr());

            *(dwc_w_out_ptr + x) = *(dwc_w_in_ptr + x) / std::sqrt(var + ScalarType(epsilon)) * gamma;
        }
    },
    dwc_w_in, dwc_w_out);
}

template void fused_batch_normalization_dwc_nhwc<float32_t>(const ITensor *dwc_weights, const ITensor *dwc_bias, ITensor *fused_weights, ITensor *fused_bias,
                                                            const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window);

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS)
template void fused_batch_normalization_dwc_nhwc<float16_t>(const ITensor *dwc_weights, const ITensor *dwc_bias, ITensor *fused_weights, ITensor *fused_bias,
                                                            const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window);
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(ENABLE_FP16_KERNELS) */

} // namespace cpu
} // namespace arm_compute
