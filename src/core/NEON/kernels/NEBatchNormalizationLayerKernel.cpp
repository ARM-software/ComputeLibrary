/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#include "arm_compute/core/NEON/kernels/NEBatchNormalizationLayerKernel.h"

#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/NEON/NEMath.h"
#include "arm_compute/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <map>

using namespace arm_compute;

namespace
{
Status
validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *var,
                   const ITensorInfo *beta, const ITensorInfo *gamma, float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);

    if(act_info.enabled())
    {
        ActivationLayerInfo::ActivationFunction act = act_info.activation();
        ARM_COMPUTE_RETURN_ERROR_ON(act != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::RELU
                                    && act != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::BOUNDED_RELU
                                    && act != ActivationLayerInfo::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU);
        ARM_COMPUTE_RETURN_ERROR_ON(act_info.b() > act_info.a());
    }

    if(nullptr != output)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, mean, var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, var);
    if(beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, beta);
    }
    if(gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(mean, gamma);
    }
    ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(get_data_layout_dimension_index(input->data_layout(), DataLayoutDimension::CHANNEL)) != mean->dimension(0));

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output, ITensorInfo *mean, ITensorInfo *var, ITensorInfo *gamma, ITensorInfo *beta)
{
    if(output != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*output, *input->clone());
    }

    unsigned int num_elems_processed_per_iteration = 16 / input->element_size();

    Window                 win = calculate_max_window(*input, Steps(num_elems_processed_per_iteration));
    AccessWindowHorizontal input_access(input, 0, num_elems_processed_per_iteration);
    bool                   window_changed = update_window_and_padding(win, input_access);

    if(output != nullptr)
    {
        AccessWindowHorizontal output_access(output, 0, num_elems_processed_per_iteration);
        window_changed |= update_window_and_padding(win, output_access);
        output_access.set_valid_region(win, input->valid_region());
    }

    // Mean, var, gamma and beta get parallelized for the NHWC case as they follow the channel dimension, which is along the first axis
    if(input->data_layout() == DataLayout::NHWC)
    {
        AccessWindowHorizontal mean_access(mean, 0, num_elems_processed_per_iteration);
        AccessWindowHorizontal var_access(var, 0, num_elems_processed_per_iteration);
        window_changed |= update_window_and_padding(win, mean_access, var_access);

        if(gamma != nullptr)
        {
            AccessWindowHorizontal gamma_access(gamma, 0, num_elems_processed_per_iteration);
            window_changed |= update_window_and_padding(win, gamma_access);
        }
        if(beta != nullptr)
        {
            AccessWindowHorizontal beta_access(beta, 0, num_elems_processed_per_iteration);
            window_changed |= update_window_and_padding(win, beta_access);
        }
    }

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, win);
}
} //namespace

template <bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_fp16_nchw(const Window &window)
{
    ARM_COMPUTE_UNUSED(window);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input(_input, window);
    Iterator output(_output, window);

    F activation_functor(_act_info);

    // Hold information about the current feature map we are iterating.
    // Only compute denominator and NEON vectors once per feature map.
    int slice = -1;

    const auto input_mean  = reinterpret_cast<const float16_t *>(_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float16_t *>(_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (_gamma != nullptr) ? reinterpret_cast<const float16_t *>(_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (_beta != nullptr) ? reinterpret_cast<const float16_t *>(_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    float16x8_t       mean_vec    = vdupq_n_f16(0.0);
    float16x8_t       var_vec     = vdupq_n_f16(0.0);
    float16x8_t       gamma_vec   = vdupq_n_f16(1.0);
    float16x8_t       beta_vec    = vdupq_n_f16(0.0);
    float16x8_t       denominator = vdupq_n_f16(0.0);
    const float16x8_t epsilon_vec = vdupq_n_f16(_epsilon);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(slice != id.z())
        {
            // Conctruct vectors
            mean_vec = vdupq_n_f16(*(input_mean + id.z()));
            var_vec  = vdupq_n_f16(*(input_var + id.z()));
            if(input_gamma != nullptr)
            {
                gamma_vec = vdupq_n_f16(*(input_gamma + id.z()));
            }
            if(input_beta != nullptr)
            {
                beta_vec = vdupq_n_f16(*(input_beta + id.z()));
            }

            // Calculate denominator
            denominator = vinvsqrtq_f16(vaddq_f16(var_vec, epsilon_vec));
            slice       = id.z();
        }

        // Calculate x bar and store results
        const float16x8_t numerator = vsubq_f16(vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr())), mean_vec);
        const float16x8_t x_bar     = vmulq_f16(numerator, denominator);
        float16x8_t       res       = vaddq_f16(beta_vec, vmulq_f16(x_bar, gamma_vec));

        // Perform fused activation
        if(fused_activation)
        {
            activation_functor(res);
        }

        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), res);
    },
    input, output);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_fp16_nhwc(const Window &window)
{
    ARM_COMPUTE_UNUSED(window);
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    Iterator input(_input, window);
    Iterator output(_output, window);

    F activation_functor(_act_info);

    const auto input_mean  = reinterpret_cast<const float16_t *>(_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float16_t *>(_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (_gamma != nullptr) ? reinterpret_cast<const float16_t *>(_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (_beta != nullptr) ? reinterpret_cast<const float16_t *>(_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    const float16x8_t epsilon_vec = vdupq_n_f16(_epsilon);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Conctruct vectors
        const float16x8_t mean_vec  = vld1q_f16(input_mean + id.x());
        const float16x8_t var_vec   = vld1q_f16(input_var + id.x());
        const float16x8_t gamma_vec = (input_gamma != nullptr) ? vld1q_f16(input_gamma + id.x()) : vdupq_n_f16(1.0);
        const float16x8_t beta_vec  = (input_beta != nullptr) ? vld1q_f16(input_beta + id.x()) : vdupq_n_f16(0.0);
        // Calculate denominator
        const float16x8_t denominator = vinvsqrtq_f16(vaddq_f16(var_vec, epsilon_vec));

        // Calculate x bar and store results
        const float16x8_t numerator = vsubq_f16(vld1q_f16(reinterpret_cast<const float16_t *>(input.ptr())), mean_vec);
        const float16x8_t x_bar     = vmulq_f16(numerator, denominator);
        float16x8_t       res       = vaddq_f16(beta_vec, vmulq_f16(x_bar, gamma_vec));

        // Perform fused activation
        if(fused_activation)
        {
            activation_functor(res);
        }

        vst1q_f16(reinterpret_cast<float16_t *>(output.ptr()), res);
    },
    input, output);
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
}

template <bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_fp32_nchw(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    F activation_functor(_act_info);

    // Hold information about the current feature map we are iterating.
    // Only compute denominator and NEON vectors once per feature map.
    int slice = -1;

    const auto input_mean  = reinterpret_cast<const float *>(_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float *>(_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (_gamma != nullptr) ? reinterpret_cast<const float *>(_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (_beta != nullptr) ? reinterpret_cast<const float *>(_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    float32x4_t       mean_vec    = vdupq_n_f32(0.0);
    float32x4_t       var_vec     = vdupq_n_f32(0.0);
    float32x4_t       gamma_vec   = vdupq_n_f32(1.0);
    float32x4_t       beta_vec    = vdupq_n_f32(0.0);
    float32x4_t       denominator = vdupq_n_f32(0.0);
    const float32x4_t epsilon_vec = vdupq_n_f32(_epsilon);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        if(slice != id.z())
        {
            // Conctruct vectors
            mean_vec = vdupq_n_f32(*(input_mean + id.z()));
            var_vec  = vdupq_n_f32(*(input_var + id.z()));
            if(input_gamma != nullptr)
            {
                gamma_vec = vdupq_n_f32(*(input_gamma + id.z()));
            }
            if(input_beta != nullptr)
            {
                beta_vec = vdupq_n_f32(*(input_beta + id.z()));
            }

            // Calculate denominator
            denominator = vinvsqrtq_f32(vaddq_f32(var_vec, epsilon_vec));
            slice       = id.z();
        }

        // Calculate x bar
        const float32x4_t numerator = vsubq_f32(vld1q_f32(reinterpret_cast<const float *>(input.ptr())), mean_vec);
        const float32x4_t x_bar     = vmulq_f32(numerator, denominator);
        float32x4_t       res       = vmlaq_f32(beta_vec, x_bar, gamma_vec);

        // Perform fused activation
        if(fused_activation)
        {
            activation_functor(res);
        }

        // Store results
        vst1q_f32(reinterpret_cast<float *>(output.ptr()), res);
    },
    input, output);
}

template <bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_fp32_nhwc(const Window &window)
{
    Iterator input(_input, window);
    Iterator output(_output, window);

    F activation_functor(_act_info);

    const auto input_mean  = reinterpret_cast<const float *>(_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const float *>(_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (_gamma != nullptr) ? reinterpret_cast<const float *>(_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (_beta != nullptr) ? reinterpret_cast<const float *>(_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    const float32x4_t epsilon_vec = vdupq_n_f32(_epsilon);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        // Conctruct vectors
        const float32x4_t mean_vec  = vld1q_f32(input_mean + id.x());
        const float32x4_t var_vec   = vld1q_f32(input_var + id.x());
        const float32x4_t gamma_vec = (input_gamma != nullptr) ? vld1q_f32(input_gamma + id.x()) : vdupq_n_f32(1.0);
        const float32x4_t beta_vec  = (input_beta != nullptr) ? vld1q_f32(input_beta + id.x()) : vdupq_n_f32(0.0);
        // Calculate denominator
        const float32x4_t denominator = vinvsqrtq_f32(vaddq_f32(var_vec, epsilon_vec));

        // Calculate x bar
        const float32x4_t numerator = vsubq_f32(vld1q_f32(reinterpret_cast<const float *>(input.ptr())), mean_vec);
        const float32x4_t x_bar     = vmulq_f32(numerator, denominator);
        float32x4_t       res       = vmlaq_f32(beta_vec, x_bar, gamma_vec);

        // Perform fused activation
        if(fused_activation)
        {
            activation_functor(res);
        }

        // Store results
        vst1q_f32(reinterpret_cast<float *>(output.ptr()), res);
    },
    input, output);
}

void NEBatchNormalizationLayerKernel::configure_non_fused()
{
    const bool is_nhwc = _input->info()->data_layout() == DataLayout::NHWC;
    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = (is_nhwc) ? &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nhwc<false, ::detail::dummy<float16_t, 8>> :
                    &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nchw<false, ::detail::dummy<float16_t, 8>>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = (is_nhwc) ? &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nhwc<false, ::detail::dummy<float, 4>> :
                    &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nchw<false, ::detail::dummy<float, 4>>;
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }
}

void NEBatchNormalizationLayerKernel::configure_fused()
{
    // NCHW Fused Batched Normalization with activation functions : FP32
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f32_nchw =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nchw<true, ::detail::relu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nchw<true, ::detail::brelu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nchw<true, ::detail::lubrelu<float, 4>> }
    };
    // NHWC Fused Batched Normalization with activation functions : FP32
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f32_nhwc =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nhwc<true, ::detail::relu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nhwc<true, ::detail::brelu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp32_nhwc<true, ::detail::lubrelu<float, 4>> }
    };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // NCHW Fused Batched Normalization with activation functions : FP16
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f16_nchw =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nchw<true, ::detail::relu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nchw<true, ::detail::brelu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nchw<true, ::detail::lubrelu<float16_t, 8>> }
    };
    // NHWC Fused Batched Normalization with activation functions : FP16
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f16_nhwc =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nhwc<true, ::detail::relu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nhwc<true, ::detail::brelu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_fp16_nhwc<true, ::detail::lubrelu<float16_t, 8>> }
    };
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = (_input->info()->data_layout() == DataLayout::NHWC) ? bn_fused_map_f16_nhwc[_act_info.activation()] : bn_fused_map_f16_nchw[_act_info.activation()];
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = (_input->info()->data_layout() == DataLayout::NHWC) ? bn_fused_map_f32_nhwc[_act_info.activation()] : bn_fused_map_f32_nchw[_act_info.activation()];
            break;
        default:
            ARM_COMPUTE_ERROR("Element size not supported");
            break;
    }
}

NEBatchNormalizationLayerKernel::NEBatchNormalizationLayerKernel()
    : _func(nullptr), _input(nullptr), _output(nullptr), _mean(nullptr), _var(nullptr), _gamma(nullptr), _beta(nullptr), _epsilon(), _act_info()
{
}

void NEBatchNormalizationLayerKernel::configure(ITensor *input, ITensor *output,
                                                const ITensor *mean, const ITensor *var,
                                                const ITensor *beta, const ITensor *gamma,
                                                float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, mean, var);

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), (output != nullptr) ? output->info() : nullptr,
                                                  mean->info(), var->info(),
                                                  (beta != nullptr) ? beta->info() : nullptr,
                                                  (gamma != nullptr) ? gamma->info() : nullptr,
                                                  epsilon, act_info));

    _input    = input;
    _output   = input;
    _mean     = mean;
    _var      = var;
    _gamma    = gamma;
    _beta     = beta;
    _epsilon  = epsilon;
    _act_info = act_info;

    const bool run_in_place = (output == nullptr) || (output == input);
    if(!run_in_place)
    {
        _output = output;
    }

    // Configure activation function to run
    if(_act_info.enabled())
    {
        configure_fused();
    }
    else
    {
        configure_non_fused();
    }

    // Configure kernel window
    auto win_config = validate_and_configure_window(input->info(), (run_in_place) ? nullptr : output->info(), mean->info(), var->info(), (gamma != nullptr) ? gamma->info() : nullptr,
                                                    (beta != nullptr) ? beta->info() : nullptr);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}

Status NEBatchNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                 const ITensorInfo *mean, const ITensorInfo *var,
                                                 const ITensorInfo *beta, const ITensorInfo *gamma,
                                                 float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mean, var, beta, gamma, epsilon, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output ? output->clone().get() : nullptr, mean->clone().get(), var->clone().get(),
                                                              (gamma != nullptr) ? gamma->clone().get() : nullptr, (beta != nullptr) ? beta->clone().get() : nullptr)
                                .first);

    return Status{};
}

void NEBatchNormalizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr);

    (this->*_func)(window);
}
