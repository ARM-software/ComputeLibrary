/*
 * Copyright (c) 2017-2024 Arm Limited.
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

#include "src/cpu/kernels/activation/heuristics/CpuActivationKernelHeuristics.h"

#include "src/core/common/Registrars.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/activation/list.h"
#include "src/cpu/kernels/logistic/list.h"

#include <map>
#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{
namespace
{

bool is_fp16_lut_supported(ActivationLayerInfo::ActivationFunction func)
{
    return func == ActivationLayerInfo::ActivationFunction::LOGISTIC ||
           func == ActivationLayerInfo::ActivationFunction::TANH;
}

using KernelList = std::vector<CpuActivationKernelHeuristics::ActivationKernel>;
using KernelMap  = std::map<DataType, KernelList>;

static const KernelList fp32_kernels = {
    {"sme2_fp32_logistic",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.f == ActivationLayerInfo::ActivationFunction::LOGISTIC && data.isa.sme2; },
     REGISTER_FP32_SME2(arm_compute::cpu::sme2_fp32_logistic)},
    {"sve_fp32_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_FP32_SVE(arm_compute::cpu::sve_fp32_activation)},
    {"neon_fp32_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_activation)},
};

static const KernelList fp16_kernels = {
    {"sve_fp16_activation_lut",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.fp16 && data.isa.sve && is_fp16_lut_supported(data.f); },
     REGISTER_FP16_SVE(arm_compute::cpu::sve_fp16_activation_lut)},
    {"sve_fp16_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve && data.isa.fp16 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_FP16_SVE(arm_compute::cpu::sve_fp16_activation)},
    {"neon_fp16_activation", [](const ActivationDataTypeISASelectorData &data) { return data.isa.fp16; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_activation)},
};

static const KernelList qasymm8_kernels = {
    {"sve2_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data) {
         return data.cpumodel == CPUModel::A510 && data.isa.sve2 &&
                data.f != ActivationLayerInfo::ActivationFunction::RELU;
     },
     REGISTER_QASYMM8_SVE2(arm_compute::cpu::sve2_q8_activation_lut)},
#ifdef __aarch64__
    {// Neon LUT implementantion takes precedence
     "neon_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.f != ActivationLayerInfo::ActivationFunction::RELU; },
     REGISTER_Q8_NEON(arm_compute::cpu::neon_q8_activation_lut)},
#endif // __aarch64__
    {"sve2_qu8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve2 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_QASYMM8_SVE2(arm_compute::cpu::sve2_qasymm8_activation)},
    {"neon_qu8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qasymm8_activation)},
};

static const KernelList qasymm8_signed_kernels = {
    {"sve2_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data) {
         return data.cpumodel == CPUModel::A510 && data.isa.sve2 &&
                data.f != ActivationLayerInfo::ActivationFunction::RELU;
     },
     REGISTER_QASYMM8_SIGNED_SVE2(arm_compute::cpu::sve2_q8_activation_lut)},
#ifdef __aarch64__
    {// Neon LUT implementantion takes precedence
     "neon_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.f != ActivationLayerInfo::ActivationFunction::RELU; },
     REGISTER_Q8_NEON(arm_compute::cpu::neon_q8_activation_lut)},
#endif // __aarch64__
    {"sve2_qs8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve2 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_QASYMM8_SIGNED_SVE2(arm_compute::cpu::sve2_qasymm8_signed_activation)},
    {"neon_qs8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_qasymm8_signed_activation)},
};

static const KernelList qsymm16_kernels = {
    {"sve2_qs16_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve2 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_QSYMM16_SVE2(arm_compute::cpu::sve2_qsymm16_activation)},
    {"neon_qs16_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_QSYMM16_NEON(arm_compute::cpu::neon_qsymm16_activation)},
};

static const KernelMap kernels = {{DataType::F32, fp32_kernels},
                                  {DataType::F16, fp16_kernels},
                                  {DataType::QASYMM8, qasymm8_kernels},
                                  {DataType::QASYMM8_SIGNED, qasymm8_signed_kernels},
                                  {DataType::QSYMM16, qsymm16_kernels}};

} // namespace

void CpuActivationKernelHeuristics::choose_kernel(ActivationDataTypeISASelectorData &selector)
{
    const auto &klist = kernels.find(selector.dt);
    if (klist == kernels.end())
    {
        return;
    }

    for (const auto &uk : klist->second)
    {
        if (uk.is_selected(selector) && uk.ukernel != nullptr)
        {
            _kernel = &uk;
            return;
        }
    }
}

CpuActivationKernelHeuristics::CpuActivationKernelHeuristics(const ITensorInfo         *src,
                                                             const ITensorInfo         *dst,
                                                             const ActivationLayerInfo &activation_info)
{
    ARM_COMPUTE_UNUSED(dst);

    // Set kernel
    const DataType                    dtype = src->data_type();
    ActivationDataTypeISASelectorData selector{dtype, CPUInfo::get().get_cpu_model(), CPUInfo::get().get_isa(),
                                               activation_info.activation()};
    choose_kernel(selector);

    // Set window and scheduling hint
    int split_dim;
    std::tie(_window, split_dim) = calculate_squashed_or_max_window(*src);

    // Collapse window with SME kernels in Y-Dim
    if (std::string(_kernel->name) == "sme2_fp32_logistic")
    {
        _window = _window.collapse(_window, Window::DimY);
    }

    _hint = IScheduler::Hints(split_dim);

    // Set minimum workload size
    if (split_dim == Window::DimX)
    {
        // Don't split the work load too small if the tensor has been reinterpreted as 1D.
        // This number is loosely chosen as threading overhead in each platform varies wildly.
        _mws = 1536;
    }
}

/** Return minimum workload size
 *
 * @return Minimum workload size for requested configuration.
 */
size_t CpuActivationKernelHeuristics::mws() const
{
    return _mws;
}

/** Return kernel's execution window
 *
 * @return The execution window
 */
const Window &CpuActivationKernelHeuristics::window() const
{
    return _window;
}

/** Return the kernel to run
 *
 * @return The function pointer to the chosen kernel
 */
const CpuActivationKernelHeuristics::ActivationKernel *CpuActivationKernelHeuristics::kernel()
{
    return _kernel;
}

/** Return the scheduling hint e.g. dimension(s) to split
 *
 * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
 */
const IScheduler::Hints &CpuActivationKernelHeuristics::scheduler_hint() const
{
    return _hint;
}
} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
