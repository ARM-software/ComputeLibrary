/*
 * Copyright (c) 2026 Arm Limited.
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
#include "src/cpu/kernels/CpuTopKVKernel.h"

#include "arm_compute/core/Validate.h"

#include "src/common/utils/profile/acl_profile.h"
#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/topkv/list.h"

namespace arm_compute
{
namespace cpu
{

namespace kernels
{
namespace
{

static const std::vector<CpuTopKVKernel::TopKVKernel> available_kernels = {
    {"neon_s32_topkv", [](const CpuTopKVKernelDataTypeISASelectorData &data) { return (data.dt == DataType::S32); },
     REGISTER_INTEGER_NEON(arm_compute::cpu::topkv_s32_neon)},
    {"neon_fp32_topkv", [](const CpuTopKVKernelDataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(arm_compute::cpu::topkv_fp32_neon)},
    {"neon_fp16_topkv",
     [](const CpuTopKVKernelDataTypeISASelectorData &data) { return (data.dt == DataType::F16) && data.isa.fp16; },
     REGISTER_FP16_NEON(arm_compute::cpu::topkv_fp16_neon)},
    {"neon_qu8_topkv", [](const CpuTopKVKernelDataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::topkv_qasymm8_neon)},
    {"neon_qs8_topkv",
     [](const CpuTopKVKernelDataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::topkv_qasymm8_signed_neon)}};

Status
validate_arguments(const ITensorInfo &predictions, const ITensorInfo &targets, const ITensorInfo &dst, uint32_t k)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&predictions);

    // predictions (logical shape [C, N], where N defaults to 1 if dimension 1 is absent)
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&predictions, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::S32, DataType::F16, DataType::F32);

    // targets (class indices)
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&targets, 1, DataType::U32);

    const unsigned int C = predictions.tensor_shape()[0]; // classes
    const unsigned int N = predictions.tensor_shape()[1]; // batch (defaults to 1 if not present)

    // k constraints
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(k == 0, "k must be > 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(C == 0, "predictions classes dimension must be > 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(N == 0, "predictions batch dimension must be > 0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(k > C, "k must be <= number of classes (C)");

    // targets must match batch
    // targets is expected to contain N elements (shape [N])
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(targets.tensor_shape()[0] != N,
                                    "targets dimension must match predictions batch dimension (N)");

    ARM_COMPUTE_RETURN_ERROR_ON(predictions.num_dimensions() > 2);
    ARM_COMPUTE_RETURN_ERROR_ON(targets.num_dimensions() > 1);

    // Output is one byte per batch element: shape [N]
    const TensorShape out_shape(N);

    // If dst is already configured, validate it
    if (dst.total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, DataType::U8);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(dst.tensor_shape() != out_shape, "dst shape must be [N]");
    }

    const auto uk = CpuTopKVKernel::get_implementation<CpuTopKVKernelDataTypeISASelectorData>(
        CpuTopKVKernelDataTypeISASelectorData{predictions.data_type(), CPUInfo::get().get_isa()});

    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    return Status{};
}

} // namespace

void CpuTopKVKernel::configure(const ITensorInfo *predictions, const ITensorInfo *targets, ITensorInfo *dst, uint32_t k)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTopKVKernel::configure");
    ARM_COMPUTE_UNUSED(targets); // workaround for a compiler bug about a false positive -Wunused-parameter
    ARM_COMPUTE_ERROR_ON_NULLPTR(predictions, targets, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*predictions, *targets, *dst, k));

    const auto uk = CpuTopKVKernel::get_implementation<CpuTopKVKernelDataTypeISASelectorData>(
        CpuTopKVKernelDataTypeISASelectorData{predictions->data_type(), CPUInfo::get().get_isa()});

    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _run_method = uk->ukernel;
    _name       = std::string("CpuTopKVKernel").append("/").append(uk->name);
    _k          = k;
    // Auto initialize dst if not initialized
    auto_init_if_empty(*dst, TensorShape(predictions->dimension(1)), 1U, DataType::U8);
    // Configure kernel window
    Window win = calculate_max_window(*dst, Steps());
    ICpuKernel::configure(win);
}
size_t CpuTopKVKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);
    ARM_COMPUTE_UNUSED(platform);

    return 1024u;
}

Status
CpuTopKVKernel::validate(const ITensorInfo *predictions, const ITensorInfo *targets, const ITensorInfo *dst, uint32_t k)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTopKVKernel::validate");
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(predictions, targets, dst);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*predictions, *targets, *dst, k));

    return Status{};
}

void CpuTopKVKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_TRACE_EVENT(ARM_COMPUTE_PROF_CAT_CPU, ARM_COMPUTE_PROF_LVL_CPU, "CpuTopKVKernel::run_op");
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *predictions = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *targets     = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *output      = tensors.get_tensor(TensorType::ACL_DST);
    _run_method(predictions, targets, output, _k, window);
}

const char *CpuTopKVKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuTopKVKernel::TopKVKernel> &CpuTopKVKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
