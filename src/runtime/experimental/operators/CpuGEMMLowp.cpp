/*
 * Copyright (c) 2017-2021, 2023-2025 Arm Limited.
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
#include "arm_compute/runtime/experimental/operators/CpuGEMMLowp.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"

#include "src/core/utils/quantization/AsymmHelpers.h"
#include "src/cpu/operators/CpuGemmLowpMatrixMultiplyCore.h"

#include <set>

namespace arm_compute
{
namespace experimental
{
namespace op
{
struct CpuGEMMLowp::Impl
{
    std::unique_ptr<arm_compute::cpu::CpuGemmLowpMatrixMultiplyCore> op{nullptr};
    ActivationLayerInfo                                              act_info{};
    bool                                                             is_prepared{false};
};

CpuGEMMLowp::CpuGEMMLowp() : _impl(std::make_unique<Impl>())
{
    _impl->op = std::make_unique<cpu::CpuGemmLowpMatrixMultiplyCore>();
}
CpuGEMMLowp::~CpuGEMMLowp() = default;

experimental::MemoryRequirements CpuGEMMLowp::workspace() const
{
    return _impl->op->workspace();
}

void CpuGEMMLowp::configure(
    const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *output, const GEMMInfo &gemm_info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, output);

    // Make the B matrix dynamic values.
    auto b_info_to_use = b->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_info_to_use->set_are_values_constant(false);
    }

    _impl->act_info    = gemm_info.activation_info();
    _impl->is_prepared = false;
    _impl->op->configure(a, b_info_to_use.get(), (c != nullptr ? c : nullptr), output, gemm_info);
}

Status CpuGEMMLowp::validate(const ITensorInfo *a,
                             const ITensorInfo *b,
                             const ITensorInfo *c,
                             const ITensorInfo *output,
                             const GEMMInfo    &gemm_info)
{
    // Make the B matrix dynamic values.
    auto b_info_to_use = b->clone();
    if (!gemm_info.reshape_b_only_on_first_run())
    {
        b_info_to_use->set_are_values_constant(false);
    }

    return cpu::CpuGemmLowpMatrixMultiplyCore::validate(a, b_info_to_use.get(), c, output, gemm_info);
}

void CpuGEMMLowp::update_quantization_parameters(const QuantizationInfo &a,
                                                 const QuantizationInfo &b,
                                                 const QuantizationInfo &c,
                                                 const DataType          data_type,
                                                 const bool              is_prepared,
                                                 const bool              negated_offsets)
{
    // Supported activations in GEMM
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = {
        ActivationLayerInfo::ActivationFunction::RELU, ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
        ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU};

    PixelValue type_min{};
    PixelValue type_max{};
    std::tie(type_min, type_max) = get_min_max(data_type);
    int32_t min_activation       = type_min.get<int32_t>();
    int32_t max_activation       = type_max.get<int32_t>();

    const UniformQuantizationInfo uoqinfo = c.uniform();
    if (supported_acts.find(_impl->act_info.activation()) != supported_acts.end())
    {
        std::tie(min_activation, max_activation) =
            get_quantized_activation_min_max(_impl->act_info, data_type, uoqinfo);
    }

    GEMMLowpOutputStageInfo output_info;
    output_info.type                     = GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT;
    output_info.gemmlowp_offset          = uoqinfo.offset;
    output_info.gemmlowp_min_bound       = min_activation;
    output_info.gemmlowp_max_bound       = max_activation;
    output_info.is_quantized_per_channel = false;
    output_info.output_data_type         = data_type;
    const Status status                  = quantization::calculate_quantized_multipliers(a, b, c, output_info);
    ARM_COMPUTE_ERROR_ON(!bool(status));

    _impl->op->update_quantization_parameters(output_info, a, b, is_prepared, negated_offsets);
}

void CpuGEMMLowp::run(ITensorPack &tensors)
{
    prepare(tensors);
    _impl->op->run(tensors);
}

void CpuGEMMLowp::prepare(ITensorPack &tensors)
{
    if (!_impl->is_prepared)
    {
        _impl->op->prepare(tensors);

        auto aux_mem_req = _impl->op->workspace();

        auto has_reshape =
            std::find_if(aux_mem_req.begin(), aux_mem_req.end(),
                         [](const MemoryInfo &m) -> bool { return m.lifetime == MemoryLifetime::Persistent; });

        if (has_reshape != std::end(aux_mem_req))
        {
            auto b = tensors.get_tensor(TensorType::ACL_SRC_1);
            b->mark_as_unused();
        }

        _impl->is_prepared = true;
    }
}
} // namespace op
} // namespace experimental
} // namespace arm_compute
