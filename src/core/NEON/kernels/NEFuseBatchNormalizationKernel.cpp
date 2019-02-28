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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/Window.h"

#include "support/ToolchainSupport.h"

#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "utils/TypePrinter.h"
namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *conv_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                          const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                          const ITensorInfo *conv_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                          float epsilon)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(conv_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(conv_weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, bn_mean, bn_var);

    unsigned int kernels_idx = get_data_layout_dimension_index(conv_weights->data_layout(), DataLayoutDimension::BATCHES);
    ARM_COMPUTE_RETURN_ERROR_ON(conv_weights->dimension(kernels_idx) != bn_mean->dimension(0));

    // Validate bias
    if(conv_bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, conv_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, conv_bias);
    }
    // Validate beta
    if(bn_beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, bn_beta);
    }
    // Validate gamma
    if(bn_gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, bn_gamma);
    }

    // Validate output weights
    if(fused_weights != nullptr && fused_weights->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(conv_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(conv_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, fused_weights);
    }
    // Validate output bias
    if(fused_bias != nullptr && fused_bias->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, fused_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(conv_weights, fused_bias);
    }

    return Status{};
}

template <typename ScalarType, int size>
void fused_batch_normmalization(const ITensor *conv_weights, const ITensor *conv_bias, ITensor *fused_weights, ITensor *fused_bias,
                                const ITensor *bn_mean, const ITensor *bn_var, const ITensor *bn_beta, const ITensor *bn_gamma, float epsilon, const Window &window)
{
    using ExactTagType = typename wrapper::traits::neon_vector<ScalarType, size>::tag_type;

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

    int slice = -1;

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
        if(slice != id[3])
        {
            slice = id[3];
            mean  = input_mean[slice];
            var   = input_var[slice];
            gamma = ScalarType(1.0);
            beta  = ScalarType(0.0);

            // Construct vectors
            mean_vec = wrapper::vdup_n(mean, ExactTagType{});
            var_vec  = wrapper::vdup_n(var, ExactTagType{});
            if(input_gamma != nullptr)
            {
                gamma     = input_gamma[slice];
                gamma_vec = wrapper::vdup_n(gamma, ExactTagType{});
            }
            if(input_beta != nullptr)
            {
                beta     = input_beta[slice];
                beta_vec = wrapper::vdup_n(beta, ExactTagType{});
            }
            if(conv_bias_in != nullptr)
            {
                conv_bias_in_scalar = conv_bias_in[slice];
            }
            else
            {
                conv_bias_in_scalar = ScalarType(0);
            }

            conv_bias_in_scalar  = (conv_bias_in_scalar - mean) / sqrt(var + ScalarType(epsilon));
            conv_bias_in_scalar  = (conv_bias_in_scalar * gamma) + beta;
            conv_bias_out[slice] = conv_bias_in_scalar;
            rvar_vec             = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));
        }

        int  x              = window_start_x;
        auto conv_w_in_ptr  = reinterpret_cast<const ScalarType *>(conv_w_in.ptr());
        auto conv_w_out_ptr = reinterpret_cast<ScalarType *>(conv_w_out.ptr());

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
            *(conv_w_out_ptr + x) = *(conv_w_in_ptr + x) / sqrt(var + ScalarType(epsilon)) * gamma;
        }
    },
    conv_w_in, conv_w_out);
}
} // namespace

NEFuseBatchNormalizationKernel::NEFuseBatchNormalizationKernel()
    : _conv_weights(nullptr), _conv_bias(nullptr), _bn_mean(nullptr), _bn_var(nullptr), _bn_gamma(nullptr), _bn_beta(nullptr), _fused_weights(nullptr), _fused_bias(nullptr), _epsilon(),
      _run_in_place_weights(false), _run_in_place_bias(false), _func(nullptr)
{
}

void NEFuseBatchNormalizationKernel::configure(const ITensor *conv_weights, const ITensor *bn_mean, const ITensor *bn_var,
                                               ITensor *fused_weights, ITensor *fused_bias,
                                               const ITensor *conv_bias, const ITensor *bn_beta, const ITensor *bn_gamma,
                                               float epsilon)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(conv_weights, bn_mean, bn_var);

    _conv_weights  = conv_weights;
    _conv_bias     = conv_bias;
    _bn_mean       = bn_mean;
    _bn_var        = bn_var;
    _bn_beta       = bn_beta;
    _bn_gamma      = bn_gamma;
    _fused_weights = fused_weights;
    _fused_bias    = fused_bias;
    _epsilon       = epsilon;

    _run_in_place_weights = (fused_weights == nullptr) || (fused_weights == conv_weights);
    _run_in_place_bias    = (fused_bias == nullptr) || (conv_bias != nullptr && fused_bias == conv_bias);

    // Auto initialize outputs
    if(_fused_weights != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_weights->info(), *_conv_weights->info()->clone());
        fused_weights->info()->set_valid_region(conv_weights->info()->valid_region());
    }
    if(_fused_bias != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_bias->info(), *_bn_mean->info()->clone());
        _fused_bias->info()->set_valid_region(bn_mean->info()->valid_region());
    }

    // Validate arguments
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(conv_weights->info(), bn_mean->info(), bn_var->info(),
                                                  (fused_weights != nullptr) ? fused_weights->info() : nullptr,
                                                  (fused_bias != nullptr) ? fused_bias->info() : nullptr,
                                                  (conv_bias != nullptr) ? conv_bias->info() : nullptr,
                                                  (bn_beta != nullptr) ? bn_beta->info() : nullptr,
                                                  (bn_gamma != nullptr) ? bn_gamma->info() : nullptr,
                                                  epsilon));

    // Configure kernel window
    Window win = calculate_max_window(*conv_weights->info());
    INEKernel::configure(win);

    // Configure function to run based on different data types
    const DataType data_type = _conv_weights->info()->data_type();
    switch(data_type)
    {
        case DataType::F32:
            _func = &fused_batch_normmalization<float, 4>;
            break;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &fused_batch_normmalization<float16_t, 8>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        default:
            ARM_COMPUTE_ERROR("Not Supported");
            break;
    }
}

Status NEFuseBatchNormalizationKernel::validate(const ITensorInfo *conv_weights, const ITensorInfo *bn_mean, const ITensorInfo *bn_var,
                                                const ITensorInfo *fused_weights, const ITensorInfo *fused_bias,
                                                const ITensorInfo *conv_bias, const ITensorInfo *bn_beta, const ITensorInfo *bn_gamma,
                                                float epsilon)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(conv_weights, bn_mean, bn_var, fused_weights, fused_bias, conv_bias, bn_beta, bn_gamma, epsilon));
    return Status{};
}

void NEFuseBatchNormalizationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);
    (*_func)(_conv_weights, _conv_bias, _fused_weights, _fused_bias, _bn_mean, _bn_var, _bn_beta, _bn_gamma, _epsilon, window);
}
} // namespace arm_compute
