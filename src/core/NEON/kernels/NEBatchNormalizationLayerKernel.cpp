/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#include "src/core/NEON/kernels/NEBatchNormalizationLayerKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/NEFixedPoint.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/NEON/kernels/detail/NEActivationFunctionDetail.h"
#include "src/core/NEON/wrapper/wrapper.h"

#include "src/core/NEON/kernels/batchnormalization/impl/list.h"
#include "src/core/common/Registrars.h"

#include <map>

namespace arm_compute
{
namespace
{
struct BatchNormalizationSelectorData
{
    DataType dt;
};
using BatchNormalizationSelectorPtr = std::add_pointer<bool(const BatchNormalizationSelectorData &data)>::type;
using BatchNormalizationKernelPtr   = std::add_pointer<void(ITensor *, ITensor *, const ITensor *, const ITensor *, const ITensor *, const ITensor *,
                                                            float, ActivationLayerInfo &, const Window &)>::type;

struct BatchNormalizationKernel
{
    const char                         *name;
    const BatchNormalizationSelectorPtr is_selected;
    BatchNormalizationKernelPtr         ukernel;
};

static const BatchNormalizationKernel available_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "fp16_sve_batch_normalization",
        [](const BatchNormalizationSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_SVE(arm_compute::cpu::fp16_sve_batch_normalization)
    },
    {
        "f32_sve_batch_normalization",
        [](const BatchNormalizationSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_SVE(arm_compute::cpu::fp32_sve_batch_normalization)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "fp16_neon_batch_normalization",
        [](const BatchNormalizationSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::fp16_neon_batch_normalization)
    },
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    {
        "f32_neon_batch_normalization",
        [](const BatchNormalizationSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::fp32_neon_batch_normalization)
    },
#endif /* !defined(__ARM_FEATURE_SVE) */
};

const BatchNormalizationKernel *get_implementation(const BatchNormalizationSelectorData &data)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status
validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ITensorInfo *mean, const ITensorInfo *var,
                   const ITensorInfo *beta, const ITensorInfo *gamma, float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_UNUSED(epsilon);

    const auto *uk = get_implementation(BatchNormalizationSelectorData{ input->data_type() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

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
} //namespace

template <typename T, bool fused_activation, typename F>
void NEBatchNormalizationLayerKernel::batch_normalization_nchw(const Window &window)
{
    /** Neon vector tag type. */
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
    // Only compute denominator and Neon vectors once per feature map.
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

void NEBatchNormalizationLayerKernel::configure_non_fused()
{
    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, false, detail::dummy<float16_t, 8>>;
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float, false, detail::dummy<float, 4>>;
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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    // NCHW Fused Batched Normalization with activation functions : FP16
    static std::map<ActivationLayerInfo::ActivationFunction, BatchNormFunctionPtr> bn_fused_map_f16_nchw =
    {
        { ActivationLayerInfo::ActivationFunction::RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, true, detail::relu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, true, detail::brelu<float16_t, 8>> },
        { ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, &NEBatchNormalizationLayerKernel::batch_normalization_nchw<float16_t, true, detail::lubrelu<float16_t, 8>> }
    };
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

    switch(_input->info()->data_type())
    {
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F16:
            _func = bn_fused_map_f16_nchw[_act_info.activation()];
            break;
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        case DataType::F32:
            _func = bn_fused_map_f32_nchw[_act_info.activation()];
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
    const bool is_nchw = _input->info()->data_layout() == DataLayout::NCHW;
    if(is_nchw)
    {
        if(_act_info.enabled())
        {
            configure_fused();
        }
        else
        {
            configure_non_fused();
        }
    }

    // Configure kernel window
    Window win = calculate_max_window(*input->info(), Steps());
    INEKernel::configure(win);

    if(output != nullptr)
    {
        // Output auto initialization if not yet initialized
        auto_init_if_empty(*output->info(), *input->info()->clone());

        Coordinates coord;
        coord.set_num_dimensions(output->info()->num_dimensions());
        output->info()->set_valid_region(ValidRegion(coord, output->info()->tensor_shape()));
    }
}

Status NEBatchNormalizationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                                                 const ITensorInfo *mean, const ITensorInfo *var,
                                                 const ITensorInfo *beta, const ITensorInfo *gamma,
                                                 float epsilon, ActivationLayerInfo act_info)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, mean, var, beta, gamma, epsilon, act_info));

    return Status{};
}

void NEBatchNormalizationLayerKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_func == nullptr && _input->info()->data_layout() == DataLayout::NCHW);

    const bool is_nchw = _input->info()->data_layout() == DataLayout::NCHW;
    if(is_nchw)
    {
        (this->*_func)(window);
    }
    else
    {
        const auto *uk = get_implementation(BatchNormalizationSelectorData{ _input->info()->data_type() });
        uk->ukernel(_input, _output, _mean, _var, _beta, _gamma, _epsilon, _act_info, window);
    }
}
} // namespace arm_compute
