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
#include "src/core/NEON/kernels/NEFuseBatchNormalizationKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/common/cpuinfo/CpuIsaInfo.h"
#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/NEON/wrapper/wrapper.h"
#include "src/cpu/kernels/fuse_batch_normalization/list.h"

#include <map>

namespace arm_compute
{
namespace
{
struct FuseBatchNormalizeSelectorData
{
    DataType                   dt;
    DataLayout                 dl;
    FuseBatchNormalizationType fbn_type;
    cpuinfo::CpuIsaInfo        isa;
};

using FBNSelectorPtr = std::add_pointer<bool(const FuseBatchNormalizeSelectorData &data)>::type;
using FBNUKernelPtr  = std::add_pointer<void(const ITensor *,
                                            const ITensor *,
                                            ITensor *,
                                            ITensor *,
                                            const ITensor *,
                                            const ITensor *,
                                            const ITensor *,
                                            const ITensor *,
                                            float,
                                            const Window &)>::type;

struct FBNUKernel
{
    const char          *name;
    const FBNSelectorPtr is_selected;
    FBNUKernelPtr        ukernel;
};

static const FBNUKernel available_kernels[] = {
    {"fused_batch_normalization_conv_NHWC_F16",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F16 && data.dl == DataLayout::NHWC && data.isa.fp16 &&
                data.fbn_type == FuseBatchNormalizationType::CONVOLUTION;
     },
     REGISTER_FP16_NEON(arm_compute::cpu::fused_batch_normalization_conv_f16)},
    {"fused_batch_normalization_conv_NCHW_F16",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F16 && data.dl == DataLayout::NCHW && data.isa.fp16 &&
                data.fbn_type == FuseBatchNormalizationType::CONVOLUTION;
     },
     REGISTER_FP16_NEON(arm_compute::cpu::fused_batch_normalization_conv_f16)},
    {"fused_batch_normalization_dwc_NHWC_F16",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F16 && data.dl == DataLayout::NHWC && data.isa.fp16 &&
                data.fbn_type == FuseBatchNormalizationType::DEPTHWISECONVOLUTION;
     },
     REGISTER_FP16_NEON(arm_compute::cpu::fused_batch_normalization_dwc_nhwc_f16)},
    {"fused_batch_normalization_dwc_NCHW_F16",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F16 && data.dl == DataLayout::NCHW && data.isa.fp16 &&
                data.fbn_type == FuseBatchNormalizationType::DEPTHWISECONVOLUTION;
     },
     REGISTER_FP16_NEON(arm_compute::cpu::fused_batch_normalization_dwc_nchw_f16)},
    {"fused_batch_normalization_conv_NHWC_F32",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F32 && data.dl == DataLayout::NHWC &&
                data.fbn_type == FuseBatchNormalizationType::CONVOLUTION;
     },
     REGISTER_FP32_NEON(arm_compute::cpu::fused_batch_normalization_conv_f32)},
    {"fused_batch_normalization_conv_NCHW_F32",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F32 && data.dl == DataLayout::NCHW &&
                data.fbn_type == FuseBatchNormalizationType::CONVOLUTION;
     },
     REGISTER_FP32_NEON(arm_compute::cpu::fused_batch_normalization_conv_f32)},
    {"fused_batch_normalization_dwc_NHWC_F32",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F32 && data.dl == DataLayout::NHWC &&
                data.fbn_type == FuseBatchNormalizationType::DEPTHWISECONVOLUTION;
     },
     REGISTER_FP32_NEON(arm_compute::cpu::fused_batch_normalization_dwc_nhwc_f32)},
    {"fused_batch_normalization_dwc_NCHW_F32",
     [](const FuseBatchNormalizeSelectorData &data)
     {
         return data.dt == DataType::F32 && data.dl == DataLayout::NCHW &&
                data.fbn_type == FuseBatchNormalizationType::DEPTHWISECONVOLUTION;
     },
     REGISTER_FP32_NEON(arm_compute::cpu::fused_batch_normalization_dwc_nchw_f32)}};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @param[in]
 *
 * @return A matching micro-kernel else nullptr
 */
const FBNUKernel *get_implementation(const FuseBatchNormalizeSelectorData &data)
{
    for (const auto &uk : available_kernels)
    {
        if (uk.is_selected(data))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo         *input_weights,
                          const ITensorInfo         *bn_mean,
                          const ITensorInfo         *bn_var,
                          const ITensorInfo         *fused_weights,
                          const ITensorInfo         *fused_bias,
                          const ITensorInfo         *input_bias,
                          const ITensorInfo         *bn_beta,
                          const ITensorInfo         *bn_gamma,
                          float                      epsilon,
                          FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_UNUSED(epsilon);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input_weights, bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input_weights);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input_weights, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_mean, bn_var);
    ARM_COMPUTE_RETURN_ERROR_ON(input_bias == nullptr && fused_bias == nullptr);
    ARM_COMPUTE_RETURN_ERROR_ON(bn_mean->num_dimensions() > 1);

    if (fbn_type == FuseBatchNormalizationType::CONVOLUTION)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(input_weights->dimension(3) != bn_mean->dimension(0));
    }
    else
    {
        const size_t channel_idx =
            get_data_layout_dimension_index(input_weights->data_layout(), DataLayoutDimension::CHANNEL);
        ARM_COMPUTE_RETURN_ERROR_ON(input_weights->dimension(channel_idx) != bn_mean->dimension(0));
    }
    // Validate bias
    if (input_bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, input_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, input_bias);
    }
    // Validate beta
    if (bn_beta != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_beta);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_beta);
    }
    // Validate gamma
    if (bn_gamma != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, bn_gamma);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, bn_gamma);
    }

    // Validate output weights
    if (fused_weights != nullptr && fused_weights->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input_weights, fused_weights);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, fused_weights);
    }
    // Validate output bias
    if (fused_bias != nullptr && fused_bias->total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(bn_mean, fused_bias);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input_weights, fused_bias);
    }

    return Status{};
}

} // namespace

NEFuseBatchNormalizationKernel::NEFuseBatchNormalizationKernel()
    : _input_weights(nullptr),
      _input_bias(nullptr),
      _bn_mean(nullptr),
      _bn_var(nullptr),
      _bn_gamma(nullptr),
      _bn_beta(nullptr),
      _fused_weights(nullptr),
      _fused_bias(nullptr),
      _epsilon(),
      _run_in_place_weights(false),
      _run_in_place_bias(false),
      _func(nullptr)
{
}

void NEFuseBatchNormalizationKernel::configure(const ITensor             *input_weights,
                                               const ITensor             *bn_mean,
                                               const ITensor             *bn_var,
                                               ITensor                   *fused_weights,
                                               ITensor                   *fused_bias,
                                               const ITensor             *input_bias,
                                               const ITensor             *bn_beta,
                                               const ITensor             *bn_gamma,
                                               float                      epsilon,
                                               FuseBatchNormalizationType fbn_type)
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
    if (_fused_weights != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_weights->info(), *_input_weights->info()->clone());
    }
    if (_fused_bias != nullptr)
    {
        // Output tensor auto initialization if not yet initialized
        auto_init_if_empty(*_fused_bias->info(), *_bn_mean->info()->clone());
    }

    // Validate arguments
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(
        input_weights->info(), bn_mean->info(), bn_var->info(),
        (fused_weights != nullptr) ? fused_weights->info() : nullptr,
        (fused_bias != nullptr) ? fused_bias->info() : nullptr, (input_bias != nullptr) ? input_bias->info() : nullptr,
        (bn_beta != nullptr) ? bn_beta->info() : nullptr, (bn_gamma != nullptr) ? bn_gamma->info() : nullptr, epsilon,
        fbn_type));

    const auto *uk = get_implementation(FuseBatchNormalizeSelectorData{
        input_weights->info()->data_type(), input_weights->info()->data_layout(), fbn_type, CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);
    ARM_COMPUTE_ERROR_ON(uk->ukernel == nullptr);
    _func = uk->ukernel;

    // Configure kernel window
    Window win = calculate_max_window(*input_weights->info());
    INEKernel::configure(win);
}

Status NEFuseBatchNormalizationKernel::validate(const ITensorInfo         *input_weights,
                                                const ITensorInfo         *bn_mean,
                                                const ITensorInfo         *bn_var,
                                                const ITensorInfo         *fused_weights,
                                                const ITensorInfo         *fused_bias,
                                                const ITensorInfo         *input_bias,
                                                const ITensorInfo         *bn_beta,
                                                const ITensorInfo         *bn_gamma,
                                                float                      epsilon,
                                                FuseBatchNormalizationType fbn_type)
{
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input_weights, bn_mean, bn_var, fused_weights, fused_bias,
                                                   input_bias, bn_beta, bn_gamma, epsilon, fbn_type));
    return Status{};
}

void NEFuseBatchNormalizationKernel::run(const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(_func == nullptr);
    (*_func)(_input_weights, _input_bias, _fused_weights, _fused_bias, _bn_mean, _bn_var, _bn_beta, _bn_gamma, _epsilon,
             window);
}
} // namespace arm_compute
