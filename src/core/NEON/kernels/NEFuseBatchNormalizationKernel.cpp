/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEFuseBatchNormalizationKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "support/ToolchainSupport.h"

#include "utils/TypePrinter.h"
#include <map>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                          const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                          const ITensorInfo *input_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                          float epsilon, FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input_weights, bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON(input_bias == nullptr && fused_bias == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(bn_mean->num_dimensions() > 1);

    if(fbn_type == FuseBatchNormalizationType::CONVOLUTION)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input_weights->dimension(3) != bn_mean->dimension(0));
    }
    else
    {
        const size_t channel_idx = get_data_layout_dimension_index(input_weights->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(input_weights->dimension(channel_idx) != bn_mean->dimension(0));
    }
    // Validate bias
    if(input_bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, input_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, input_bias);
    }
    // Validate beta
    if(bn_beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_beta);
    }
    // Validate gamma
    if(bn_gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_gamma);
    }

    // Validate output weights
    if(fused_weights != nullptr && fused_weights->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, fused_weights);
    }
    // Validate output bias
    if(fused_bias != nullptr && fused_bias->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, fused_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, fused_bias);
    }

    return Status{};
}

template <typename VectorType>
void fused_batch_normalization_conv(const ITensor *conv_weights, const ITensor *conv_bias, ITensor *fused_weights, ITensor *fused_bias,
                                    const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window)
{
    using ScalarType   = typename VectorType::scalar_type;
    const int size     = 16 / conv_weights->info()->element_size();
    using ExactTagType = typename VectorType::tag_type;

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

    const auto conv_bias_in  = (conv_bias != nullptr ? reinterpret_cast<ScalarType *>(conv_bias->ptr_to_element(Coordinates(0, 0))) : nullptr);
    auto       conv_bias_out = (run_in_place_bias ? conv_bias_in : reinterpret_cast<ScalarType *>(fused_bias->ptr_to_element(Coordinates(0, 0))));

    const auto input_mean  = reinterpret_cast<const ScalarType *>(bn_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const ScalarType *>(bn_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (bn_gamma != nullptr) ? reinterpret_cast<const ScalarType *>(bn_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (bn_beta != nullptr) ? reinterpret_cast<const ScalarType *>(bn_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

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
    execute_window_loop(win, [&](const Coordinates & id)
    {
        var = input_var[id[3]];
        if(input_gamma != nullptr)
        {
            gamma = input_gamma[id[3]];
        }

        if((id[0] == 0) && (id[1] == 0) && (id[2] == 0))
        {
            if(input_beta != nullptr)
            {
                beta     = input_beta[id[3]];
                beta_vec = wrapper::vdup_n(beta, ExactTagType{});
            }

            // Construct vectors
            mean     = input_mean[id[3]];
            mean_vec = wrapper::vdup_n(mean, ExactTagType{});

            if(conv_bias_in != nullptr)
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

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            auto wn = wrapper::vloadq(conv_w_in_ptr + x);
            wn      = wrapper::vmul(wn, rvar_vec);
            wn      = wrapper::vmul(wn, gamma_vec);

            // Store results
            wrapper::vstore(conv_w_out_ptr + x, wn);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            *(conv_w_out_ptr + x) = *(conv_w_in_ptr + x) / std::sqrt(var + ScalarType(epsilon)) * gamma;
        }
    },
    conv_w_in, conv_w_out);
}

template <typename VectorType>
void fused_batch_normalization_dwc_nhwc(const ITensor *dwc_weights, const ITensor *dwc_bias, ITensor *fused_weights, ITensor *fused_bias,
                                        const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window)
{
    using ScalarType   = typename VectorType::scalar_type;
    const int size     = 16 / dwc_weights->info()->element_size();
    using ExactTagType = typename VectorType::tag_type;

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

template <typename VectorType>
void fused_batch_normalization_dwc_nchw(const ITensor *dwc_weights, const ITensor *dwc_bias, ITensor *fused_weights, ITensor *fused_bias,
                                        const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window)
{
    using ScalarType   = typename VectorType::scalar_type;
    const int size     = 16 / dwc_weights->info()->element_size();
    using ExactTagType = typename VectorType::tag_type;

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

    auto       mean_vec    = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       var_vec     = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       gamma_vec   = wrapper::vdup_n(ScalarType(1), ExactTagType{});
    auto       beta_vec    = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    auto       rvar_vec    = wrapper::vdup_n(ScalarType(0), ExactTagType{});
    const auto epsilon_vec = wrapper::vdup_n(ScalarType(epsilon), ExactTagType{});

    auto mean               = ScalarType(0.0);
    auto var                = ScalarType(0.0);
    auto gamma              = ScalarType(1.0);
    auto beta               = ScalarType(0.0);
    auto dwc_bias_in_scalar = ScalarType(0.0);
    execute_window_loop(win, [&](const Coordinates & id)
    {
        var = input_var[id[2]];
        if(input_gamma != nullptr)
        {
            gamma = input_gamma[id[2]];
        }

        if(id[1] == 0)
        {
            mean = input_mean[id[2]];

            // Construct vectors
            mean_vec = wrapper::vdup_n(mean, ExactTagType{});
            if(input_beta != nullptr)
            {
                beta     = input_beta[id[2]];
                beta_vec = wrapper::vdup_n(beta, ExactTagType{});
            }

            if(dwc_bias_in != nullptr)
            {
                dwc_bias_in_scalar = dwc_bias_in[id[2]];
            }

            auto dwc_bias_tmp_scalar = (dwc_bias_in_scalar - mean) / std::sqrt(var + ScalarType(epsilon));
            dwc_bias_out[id[2]]      = (dwc_bias_tmp_scalar * gamma) + beta;
        }

        int  x             = window_start_x;
        auto dwc_w_in_ptr  = reinterpret_cast<const ScalarType *>(dwc_w_in.ptr());
        auto dwc_w_out_ptr = reinterpret_cast<ScalarType *>(dwc_w_out.ptr());
        var_vec            = wrapper::vdup_n(var, ExactTagType{});
        gamma_vec          = wrapper::vdup_n(gamma, ExactTagType{});
        rvar_vec           = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));

        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            auto wn = wrapper::vloadq(dwc_w_in_ptr + x);
            wn      = wrapper::vmul(wn, rvar_vec);
            wn      = wrapper::vmul(wn, gamma_vec);

            // Store results
            wrapper::vstore(dwc_w_out_ptr + x, wn);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            *(dwc_w_out_ptr + x) = *(dwc_w_in_ptr + x) / std::sqrt(var + ScalarType(epsilon)) * gamma;
        }
    },
    dwc_w_in, dwc_w_out);
}

} // namespace

NEFuseBatchNormalizationKernel::NEFuseBatchNormalizationKernel()
    : _input_weights(nullptr), _input_bias(nullptr), _bn_mean(nullptr), _bn_var(nullptr), _bn_gamma(nullptr), _bn_beta(nullptr), _fused_weights(nullptr), _fused_bias(nullptr), _epsilon(),
      _run_in_place_weights(false), _run_in_place_bias(false), _func(nullptr)
{
}

void NEFuseBatchNormalizationKernel::configure(const ITensor *input_weights, const ITensor *bn_mean, const ITensor *bn_var,
                                               ITensor *fused_weights, ITensor *fused_bias,
                                               const ITensor *input_bias, const ITensor *bn_beta, const ITensor *bn_gamma,
                                               float epsilon, FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input_weights, bn_mean, bn_var);

    _input_weights = input_weights;
    _input_bias    = input_bias;
    _bn_mean       = bn_mean;
    _bn_var        = bn_var;
    _bn_beta       = bn_beta;
    _bn_gamma      = bn_gamma;
    _fused_weights = fused_weights;
    _fused_bias    = fused_bias;
    _epsilon       = epsilon;

    _run_in_place_weights = (fused_weights == nullptr) || (fused_weights == input_weights);
    _run_in_place_bias    = (fused_bias == nullptr) || (input_bias != nullptr && fused_bias == input_bias);

    // Auto initialize outputs
    if(_fused_weights != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_weights->info(), *_input_weights->info()->clone());
        fused_weights->info()->set_valid_region(input_weights->info()->valid_region());
    }
    if(_fused_bias != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_bias->info(), *_bn_mean->info()->clone());
        _fused_bias->info()->set_valid_region(bn_mean->info()->valid_region());
    }

    // Validate arguments
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input_weights->info(), bn_mean->info(), bn_var->info(),
                                                  (fused_weights != nullptr) ? fused_weights->info() : nullptr,
                                                  (fused_bias != nullptr) ? fused_bias->info() : nullptr,
                                                  (input_bias != nullptr) ? input_bias->info() : nullptr,
                                                  (bn_beta != nullptr) ? bn_beta->info() : nullptr,
                                                  (bn_gamma != nullptr) ? bn_gamma->info() : nullptr,
                                                  epsilon, fbn_type));

    // Configure kernel window
    Window win = calculate_max_window(*input_weights->info());
    INEKernel::configure(win);

    // Configure function
    static std::map<std::string, FuseBatchNormFunction *> map_function =
    {
        { "fused_batch_normalization_conv_NHWC_F32", &fused_batch_normalization_conv<wrapper::traits::neon_vector<float, 4>> },
        { "fused_batch_normalization_conv_NCHW_F32", &fused_batch_normalization_conv<wrapper::traits::neon_vector<float, 4>> },
        { "fused_batch_normalization_dwc_NHWC_F32", &fused_batch_normalization_dwc_nhwc<wrapper::traits::neon_vector<float, 4>> },
        { "fused_batch_normalization_dwc_NCHW_F32", &fused_batch_normalization_dwc_nchw<wrapper::traits::neon_vector<float, 4>> },
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        { "fused_batch_normalization_conv_NHWC_F16", &fused_batch_normalization_conv<wrapper::traits::neon_vector<float16_t, 8>> },
        { "fused_batch_normalization_conv_NCHW_F16", &fused_batch_normalization_conv<wrapper::traits::neon_vector<float16_t, 8>> },
        { "fused_batch_normalization_dwc_NHWC_F16", &fused_batch_normalization_dwc_nhwc<wrapper::traits::neon_vector<float16_t, 8>> },
        { "fused_batch_normalization_dwc_NCHW_F16", &fused_batch_normalization_dwc_nchw<wrapper::traits::neon_vector<float16_t, 8>> },
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    };

    std::string function_to_call("fused_batch_normalization_");
    function_to_call += fbn_type == FuseBatchNormalizationType::CONVOLUTION ? "conv_" : "dwc_";
    function_to_call += string_from_data_layout(_input_weights->info()->data_layout());
    function_to_call += "_";
    function_to_call += string_from_data_type(_input_weights->info()->data_type());

    auto it = map_function.find(function_to_call);

    if(it != map_function.end())
    {
        _func = it->second;
    }
}

Status NEFuseBatchNormalizationKernel::validate(const ITensorInfo *input_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                                                const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                                                const ITensorInfo *input_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                                                float epsilon, FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input_weights, bn_mean, bn_var, fused_weights, fused_bias, input_bias, bn_beta, bn_gamma, epsilon, fbn_type));
    return Status{};
}

void NEFuseBatchNormalizationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    (*_func)(_input_weights, _input_bias, _fused_weights, _fused_bias, _bn_mean, _bn_var, _bn_beta, _bn_gamma, _epsilon, window);
}
} // namespace arm_compute
