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
#include "src/core/cpu/kernels/CpuActivationKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/common/Registrars.h"
#include "src/core/cpu/kernels/activation/list.h"

#include <array>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct ActivationSelectorData
{
    DataType       dt;
    const CPUInfo &ci;
};

using ActivationSelectorPtr = std::add_pointer<bool(const ActivationSelectorData &data)>::type;
using ActivationKernelPtr   = std::add_pointer<void(const ITensor *, ITensor *, const ActivationLayerInfo &, const Window &)>::type;

struct ActivationKernel
{
    const char                 *name;
    const ActivationSelectorPtr is_selected;
    ActivationKernelPtr         ukernel;
};

static const ActivationKernel available_kernels[] =
{
#if defined(ARM_COMPUTE_ENABLE_SVE)
    {
        "fp16_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::F16 && data.ci.has_sve(); },
        REGISTER_FP16_SVE(arm_compute::cpu::fp16_sve_activation)
    },
    {
        "fp32_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::F32 && data.ci.has_sve(); },
        REGISTER_FP32_SVE(arm_compute::cpu::fp32_sve_activation)
    },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE)  */
#if defined(ARM_COMPUTE_ENABLE_NEON)
    {
        "fp16_neon_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_NEON(arm_compute::cpu::fp16_neon_activation)
    },
    {
        "fp32_neon_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::fp32_neon_activation)
    },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON)  */
#if defined(ARM_COMPUTE_ENABLE_SVE2)
    {
        "qasymm8_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QASYMM8 && data.ci.has_sve2(); },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::qasymm8_sve_activation)
    },
    {
        "qasymm8_signed_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED && data.ci.has_sve2(); },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::qasymm8_signed_sve_activation)
    },
    {
        "qsymm16_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QSYMM16 && data.ci.has_sve2(); },
        REGISTER_QSYMM16_SVE(arm_compute::cpu::qsymm16_sve_activation)
    },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
    {
        "qasymm8_neon_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QASYMM8; },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::qasymm8_neon_activation)
    },
    {
        "qasymm8_signed_neon_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::qasymm8_signed_neon_activation)
    },
    {
        "qsymm16_neon_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QSYMM16; },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::qsymm16_neon_activation)
    },
};

const ActivationKernel *get_implementation(const ActivationSelectorData &data)
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

/* Supported activation in the 8-bit integer domain */
static const std::array<ActivationLayerInfo::ActivationFunction, 7> qasymm8_activations =
{
    ActivationLayerInfo::ActivationFunction::RELU,
    ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
    ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
    ActivationLayerInfo::ActivationFunction::LOGISTIC,
    ActivationLayerInfo::ActivationFunction::TANH,
    ActivationLayerInfo::ActivationFunction::HARD_SWISH,
    ActivationLayerInfo::ActivationFunction::LEAKY_RELU,
};
/* Supported activation in the 16-bit integer domain */
static const std::array<ActivationLayerInfo::ActivationFunction, 3> qsymm16_activations =
{
    ActivationLayerInfo::ActivationFunction::LOGISTIC,
    ActivationLayerInfo::ActivationFunction::TANH,
    ActivationLayerInfo::ActivationFunction::HARD_SWISH
};

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &activation_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::QSYMM16, DataType::F16, DataType::F32);

    const auto *uk = get_implementation(ActivationSelectorData{ src->data_type(), CPUInfo::get() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    const DataType                                data_type = src->data_type();
    const QuantizationInfo                       &oq_info   = (dst != nullptr) ? dst->quantization_info() : src->quantization_info();
    const ActivationLayerInfo::ActivationFunction f_act     = activation_info.activation();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized_asymmetric(data_type) && (std::find(std::begin(qasymm8_activations), std::end(qasymm8_activations), f_act) == std::end(qasymm8_activations)),
                                    "For QASYMM8 only hard swish, leaky relu, tanh, logistic, relu and lower/upper bounded relu are supported");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized_symmetric(data_type) && (std::find(std::begin(qsymm16_activations), std::end(qsymm16_activations), f_act) == std::end(qsymm16_activations)),
                                    "For QSYMM16 only tanh and logistic are supported");
    ARM_COMPUTE_RETURN_ERROR_ON((data_type == DataType::QASYMM8 || data_type == DataType::QASYMM16) && (f_act == ActivationLayerInfo::ActivationFunction::TANH)
                                && (oq_info != QuantizationInfo(1.f / 128.f, 128)));
    ARM_COMPUTE_RETURN_ERROR_ON((data_type == DataType::QASYMM8 || data_type == DataType::QASYMM16) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
                                && (oq_info != QuantizationInfo(1.f / 256.f, 0)));

    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8_SIGNED && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 128.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8_SIGNED && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 256.f, -128)));

    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));

    // Checks performed when dst is configured
    if((dst != nullptr) && (dst->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src, dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, dst);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *src, ITensorInfo *dst)
{
    // Configure kernel window
    Window win = calculate_max_window(*src, Steps());

    if(dst != nullptr)
    {
        // dst auto inizialitation if not yet initialized
        auto_init_if_empty(*dst, *src->clone());
    }

    return std::make_pair(Status{}, win);
}
} // namespace

void CpuActivationKernel::configure(const ITensorInfo *src, ITensorInfo *dst, ActivationLayerInfo activation_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src);

    _act_info = activation_info;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, dst, activation_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status CpuActivationKernel::validate(const ITensorInfo *src, const ITensorInfo *dst, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, dst, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src->clone().get(), (dst != nullptr) ? dst->clone().get() : nullptr).first);

    return Status{};
}

void CpuActivationKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    // Early exit on disabled activation
    if(!_act_info.enabled())
    {
        return;
    }

    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(IKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());

    const ITensor *src = tensors.get_const_tensor(TensorType::ACL_SRC);
    ITensor       *dst = tensors.get_tensor(TensorType::ACL_DST);

    const auto *uk = get_implementation(ActivationSelectorData{ src->info()->data_type(), CPUInfo::get() });

    uk->ukernel(src, dst, _act_info, window);
}

const char *CpuActivationKernel::name() const
{
    return "CpuActivationKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
