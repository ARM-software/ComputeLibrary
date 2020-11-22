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
#include "src/core/NEON/kernels/NEActivationLayerKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/NEON/kernels/activation/impl/list.h"
#include "src/core/common/Registrars.h"

#include <set>

namespace arm_compute
{
namespace
{
struct ActivationSelectorData
{
    DataType dt;
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
#if defined(__ARM_FEATURE_SVE)
    {
        "fp16_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::F16; },
        REGISTER_FP16_SVE(arm_compute::cpu::fp16_sve_activation)
    },
    {
        "fp32_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_SVE(arm_compute::cpu::fp32_sve_activation)
    },
#else  /* !defined(__ARM_FEATURE_SVE) */
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
#endif /* defined(__ARM_FEATURE_SVE)  */

#if defined(__ARM_FEATURE_SVE2) /* defined(__ARM_FEATURE_SVE2) */
    {
        "qasymm8_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QASYMM8; },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::qasymm8_sve_activation)
    },
    {
        "qasymm8_signed_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::qasymm8_signed_sve_activation)
    },
    {
        "qsymm16_sve_activation",
        [](const ActivationSelectorData & data) { return data.dt == DataType::QSYMM16; },
        REGISTER_QSYMM16_SVE(arm_compute::cpu::qsymm16_sve_activation)
    },
#else  /* !defined(__ARM_FEATURE_SVE2) */
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
#endif /* defined(__ARM_FEATURE_SVE2) */
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

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &activation_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8_SIGNED, DataType::QASYMM8, DataType::QSYMM16, DataType::F16, DataType::F32);

    const auto *uk = get_implementation(ActivationSelectorData{ input->data_type() });
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    const static std::set<ActivationLayerInfo::ActivationFunction> qasymm8_supported_activations =
    {
        ActivationLayerInfo::ActivationFunction::RELU,
        ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::LOGISTIC,
        ActivationLayerInfo::ActivationFunction::TANH,
        ActivationLayerInfo::ActivationFunction::HARD_SWISH,
        ActivationLayerInfo::ActivationFunction::LEAKY_RELU,
    };
    const static std::set<ActivationLayerInfo::ActivationFunction> qsymm16_supported_activations =
    {
        ActivationLayerInfo::ActivationFunction::LOGISTIC,
        ActivationLayerInfo::ActivationFunction::TANH,
        ActivationLayerInfo::ActivationFunction::HARD_SWISH
    };
    const DataType                                data_type = input->data_type();
    const QuantizationInfo                       &oq_info   = (output != nullptr) ? output->quantization_info() : input->quantization_info();
    const ActivationLayerInfo::ActivationFunction f_act     = activation_info.activation();

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized_asymmetric(data_type) && (qasymm8_supported_activations.count(f_act) == 0),
                                    "For QASYMM8 only hard swish, leaky relu, tanh, logistic, relu and lower/upper bounded relu are supported");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized_symmetric(data_type) && (qsymm16_supported_activations.count(f_act) == 0),
                                    "For QSYMM16 only tanh and logistic are supported");
    ARM_COMPUTE_RETURN_ERROR_ON((data_type == DataType::QASYMM8 || data_type == DataType::QASYMM16) && (f_act == ActivationLayerInfo::ActivationFunction::TANH)
                                && (oq_info != QuantizationInfo(1.f / 128.f, 128)));
    ARM_COMPUTE_RETURN_ERROR_ON((data_type == DataType::QASYMM8 || data_type == DataType::QASYMM16) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC)
                                && (oq_info != QuantizationInfo(1.f / 256.f, 0)));

    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8_SIGNED && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 128.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(data_type == DataType::QASYMM8_SIGNED && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 256.f, -128)));

    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::TANH) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));
    ARM_COMPUTE_RETURN_ERROR_ON(is_data_type_quantized_symmetric(data_type) && (f_act == ActivationLayerInfo::ActivationFunction::LOGISTIC) && (oq_info != QuantizationInfo(1.f / 32768.f, 0)));

    // Checks performed when output is configured
    if((output != nullptr) && (output->total_size() != 0))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    }

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *input, ITensorInfo *output)
{
    // Configure kernel window
    Window win = calculate_max_window(*input, Steps());

    if(output != nullptr)
    {
        // Output auto inizialitation if not yet initialized
        auto_init_if_empty(*output, *input->clone());

        Coordinates coord;
        coord.set_num_dimensions(output->num_dimensions());
        output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
    }

    return std::make_pair(Status{}, win);
}
} // namespace

NEActivationLayerKernel::NEActivationLayerKernel()
    : _act_info()
{
}

void NEActivationLayerKernel::configure(const ITensorInfo *input, ITensorInfo *output, ActivationLayerInfo activation_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);

    _act_info = activation_info;

    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input, output, activation_info));

    // Configure kernel window
    auto win_config = validate_and_configure_window(input, output);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICPPKernel::configure(win_config.second);
}

Status NEActivationLayerKernel::validate(const ITensorInfo *input, const ITensorInfo *output, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, act_info));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), (output != nullptr) ? output->clone().get() : nullptr).first);

    return Status{};
}

void NEActivationLayerKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
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

    const auto *uk = get_implementation(ActivationSelectorData{ src->info()->data_type() });

    uk->ukernel(src, dst, _act_info, window);
}
} // namespace arm_compute
