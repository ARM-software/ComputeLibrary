/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"

#include "src/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include <map>

namespace arm_compute
{
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
    ARM_COMPUTE_UNUSED(mean, var, gamma, beta);

    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    if(output != nullptr)
    {
        // Output auto initialization if not yet initialized
        auto_init_if_empty(*output, *input->clone());

        // NEBatchNormalizationLayerKernel doesn't need padding so update_window_and_padding() can be skipped
        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} //namespace

template <typename T, bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_nchw(const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    const int  window_step_x  = 16 / sizeof(T);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_to_use = window;
    win_to_use.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win_to_use);
    Iterator output(_output, win_to_use);

    F activation_functor(_act_info);

    // Hold information about the current feature map we are iterating.
    // Only compute denominator and NEON vectors once per feature map.
    int slice = -1;

    const auto input_mean  = reinterpret_cast<const T *>(_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const T *>(_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (_gamma != nullptr) ? reinterpret_cast<const T *>(_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (_beta != nullptr) ? reinterpret_cast<const T *>(_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    T mean        = static_cast<T>(0);
    T var         = static_cast<T>(0);
    T gamma       = static_cast<T>(1);
    T beta        = static_cast<T>(0);
    T denominator = static_cast<T>(0);

    auto       mean_vec        = wrapper::vdup_n(mean, ExactTagType{});
    auto       var_vec         = wrapper::vdup_n(var, ExactTagType{});
    auto       gamma_vec       = wrapper::vdup_n(gamma, ExactTagType{});
    auto       beta_vec        = wrapper::vdup_n(beta, ExactTagType{});
    auto       denominator_vec = wrapper::vdup_n(denominator, ExactTagType{});
    const auto epsilon_vec     = wrapper::vdup_n(static_cast<T>(_epsilon), ExactTagType{});
    execute_window_loop(win_to_use, [&](const Coordinates & id)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto output_ptr = reinterpret_cast<T *>(output.ptr());

        if(slice != id.z())
        {
            mean     = input_mean[id.z()];
            var      = input_var[id.z()];
            mean_vec = wrapper::vdup_n(mean, ExactTagType{});
            var_vec  = wrapper::vdup_n(var, ExactTagType{});
            if(input_gamma != nullptr)
            {
                gamma     = input_gamma[id.z()];
                gamma_vec = wrapper::vdup_n(gamma, ExactTagType{});
            }
            if(input_beta != nullptr)
            {
                beta     = input_beta[id.z()];
                beta_vec = wrapper::vdup_n(beta, ExactTagType{});
            }

            // Calculate denominator
            denominator_vec = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));
            denominator     = wrapper::vgetlane(denominator_vec, 0);
            slice           = id.z();
        }

        // Perform core calculations using vector operations
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Calculate x bar
            const auto numerator = wrapper::vsub(wrapper::vloadq(input_ptr + x), mean_vec);
            const auto x_bar     = wrapper::vmul(numerator, denominator_vec);
            auto       res       = wrapper::vmla(beta_vec, x_bar, gamma_vec);

            // Perform fused activation
            if(fused_activation)
            {
                activation_functor(res);
            }

            // Store results
            wrapper::vstore(output_ptr + x, res);
        }

        // Compute left-over elements
        for(; x < window_end_x; ++x)
        {
            const T numerator = input_ptr[x] - mean;
            const T x_bar     = numerator * denominator;
            T       res       = beta + x_bar * gamma;

            // Perform fused activation
            if(fused_activation)
            {
                activation_functor(res);
            }

            // Store results
            *(output_ptr + x) = res;
        }
    },
    input, output);
}

template <typename T, bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_nhwc(const Window &window)
{
    /** NEON vector tag type. */
    using ExactTagType = typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

    const int  window_step_x  = 16 / sizeof(T);
    const auto window_start_x = static_cast<int>(window.x().start());
    const auto window_end_x   = static_cast<int>(window.x().end());

    Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
    win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

    Iterator input(_input, win_collapsed);
    Iterator output(_output, win_collapsed);

    F activation_functor(_act_info);

    const auto input_mean  = reinterpret_cast<const T *>(_mean->ptr_to_element(Coordinates(0, 0)));
    const auto input_var   = reinterpret_cast<const T *>(_var->ptr_to_element(Coordinates(0, 0)));
    const auto input_gamma = (_gamma != nullptr) ? reinterpret_cast<const T *>(_gamma->ptr_to_element(Coordinates(0, 0))) : nullptr;
    const auto input_beta  = (_beta != nullptr) ? reinterpret_cast<const T *>(_beta->ptr_to_element(Coordinates(0, 0))) : nullptr;

    const auto epsilon_vec = wrapper::vdup_n(static_cast<T>(_epsilon), ExactTagType{});
    execute_window_loop(win_collapsed, [&](const Coordinates &)
    {
        const auto input_ptr  = reinterpret_cast<const T *>(input.ptr());
        const auto output_ptr = reinterpret_cast<T *>(output.ptr());

        // Perform core calculations using vector operations
        int x = window_start_x;
        for(; x <= (window_end_x - window_step_x); x += window_step_x)
        {
            // Conctruct vectors
            const auto mean_vec  = wrapper::vloadq(input_mean + x);
            const auto var_vec   = wrapper::vloadq(input_var + x);
            const auto gamma_vec = (input_gamma != nullptr) ? wrapper::vloadq(input_gamma + x) : wrapper::vdup_n(static_cast<T>(1.f), ExactTagType{});
            const auto beta_vec  = (input_beta != nullptr) ? wrapper::vloadq(input_beta + x) : wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

            // Calculate denominator
            const auto denominator = wrapper::vinvsqrt(wrapper::vadd(var_vec, epsilon_vec));

            // Calculate x bar
            const auto numerator = wrapper::vsub(wrapper::vloadq(input_ptr + x), mean_vec);
            const auto x_bar     = wrapper::vmul(numerator, denominator);
            auto       res       = wrapper::vmla(beta_vec, x_bar, gamma_vec);

            // Perform fused activation
            if(fused_activation)
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
            const T gamma = (input_gamma != nullptr) ? input_gamma[x] : 1.f;
            const T beta  = (input_beta != nullptr) ? input_beta[x] : 0.f;

            const T denominator = sqrt(input_var[x] + _epsilon);
            const T numerator   = input_ptr[x] - input_mean[x];
            const T x_bar       = numerator / denominator;
            T       res         = beta + x_bar * gamma;

            // Perform fused activation
            if(fused_activation)
            {
                activation_functor(res);
            }

            // Store results
            *reinterpret_cast<T *>(output_ptr + x) = res;
        }
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
            _func = (is_nhwc) ? &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float16_t, false, detail::dummy<float16_t, 8>> :
                    &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, false, detail::dummy<float16_t, 8>>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = (is_nhwc) ? &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float, false, detail::dummy<float, 4>> :
                    &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float, false, detail::dummy<float, 4>>;
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
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float, true, detail::relu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float, true, detail::brelu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float, true, detail::lubrelu<float, 4>> }
    };
    // NHWC Fused Batched Normalization with activation functions : FP32
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f32_nhwc =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float, true, detail::relu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float, true, detail::brelu<float, 4>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float, true, detail::lubrelu<float, 4>> }
    };
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // NCHW Fused Batched Normalization with activation functions : FP16
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f16_nchw =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, true, detail::relu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, true, detail::brelu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, true, detail::lubrelu<float16_t, 8>> }
    };
    // NHWC Fused Batched Normalization with activation functions : FP16
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f16_nhwc =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float16_t, true, detail::relu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float16_t, true, detail::brelu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nhwc<float16_t, true, detail::lubrelu<float16_t, 8>> }
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
} // namespace arm_compute
