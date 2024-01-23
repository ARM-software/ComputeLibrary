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
#include "src/cpu/kernels/CpuSoftmaxKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/common/Registrars.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/Utils.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/softmax/list.h"

#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
/* Softmax */
static const std::vector<typename CpuSoftmaxKernel::SoftmaxKernel> available_kernels = {
    {"neon_fp32_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data) { return (!data.is_log && data.dt == DataType::F32); },
     REGISTER_FP32_NEON(neon_fp32_softmax<false>)},
    {"neon_fp16_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data)
     { return (!data.is_log && data.dt == DataType::F16) && data.isa.fp16; },
     REGISTER_FP16_NEON(neon_fp16_softmax<false>)},
    {"neon_qu8_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data) { return (!data.is_log && data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qasymm8_softmax<false>)},
    {"neon_qs8_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data)
     { return (!data.is_log && data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_qasymm8_signed_softmax<false>)},
    {"neon_fp32_log_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data) { return (data.is_log && data.dt == DataType::F32); },
     REGISTER_FP32_NEON(neon_fp32_softmax<true>)},
    {"neon_fp16_log_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data)
     { return (data.is_log && data.dt == DataType::F16) && data.isa.fp16; },
     REGISTER_FP16_NEON(neon_fp16_softmax<true>)},
    {"neon_qu8_log_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data) { return (data.is_log && data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qasymm8_softmax<true>)},
    {"neon_qs8_log_softmax",
     [](const SoftmaxKernelDataTypeISASelectorData &data)
     { return (data.is_log && data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_qasymm8_signed_softmax<true>)},
};

Status validate_arguments_softmax(
    const ITensorInfo &src, const ITensorInfo &dst, float beta, const ITensorInfo &tmp, bool is_log)
{
    ARM_COMPUTE_UNUSED(beta);
    // Check input
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src.data_type());

    // Check output if configured
    if (dst.total_size() != 0)
    {
        const QuantizationInfo output_quantization =
            is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(src.data_type(), is_log)
                                    : dst.quantization_info();
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON(dst.quantization_info() != output_quantization);
    }

    // Check tmp if configured
    if (tmp.total_size() != 0)
    {
        // We have temporary storage only if src data type is quantized.
        // Therefore, tmp data type must be F32
        ARM_COMPUTE_RETURN_ERROR_ON(tmp.data_type() != DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON(!is_quantized_asymmetric);

        // We could potentially reduce tmp memory if we could predict or make an assumption
        // on the maximum number of threads that will run in parallel.
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &tmp);
    }

    return Status{};
}
} // namespace

const std::vector<typename CpuSoftmaxKernel::SoftmaxKernel> &CpuSoftmaxKernel::get_available_kernels()
{
    return available_kernels;
}

void CpuSoftmaxKernel::configure(const ITensorInfo *src, ITensorInfo *dst, float beta, bool is_log, ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst, tmp);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_softmax(*src, *dst, beta, *tmp, is_log));

    // Configure kernel window
    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src->data_type());

    // Output auto initialization if not yet initialized
    const QuantizationInfo output_quantization =
        is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(src->data_type(), is_log)
                                : dst->quantization_info();
    auto_init_if_empty(*dst, TensorInfo(*src).set_quantization_info(output_quantization).reset_padding());

    // Tmp auto initialization if not yet initialized and src is quantized
    if (is_quantized_asymmetric)
    {
        auto_init_if_empty(*tmp, TensorInfo(*src).set_data_type(DataType::F32).reset_padding());
    }

    const auto *uk = CpuSoftmaxKernel::get_implementation(
        SoftmaxKernelDataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa(), is_log});
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    std::string kernel_name = is_log ? std::string("CpuLogSoftmaxKernel") : std::string("CpuSoftmaxKernel");

    _beta       = beta;
    _run_method = uk->ukernel;
    _name       = kernel_name.append("/").append(uk->name);

    Window win = calculate_max_window(*dst, Steps());

    /// TODO: Check dimensions > 0 for holes only. For this, we need
    /// a utility function checking if there are holes after some dimension.
    if (!has_holes(*dst, dst->num_dimensions() - 1))
    {
        win = win.collapse(win, Window::DimY);
    }

    win.set(Window::DimX, Window::Dimension(0, 1, 1)); // First dimension is the reduction axis

    ICpuKernel<CpuSoftmaxKernel>::configure(win);
}

Status CpuSoftmaxKernel::validate(
    const ITensorInfo *src, const ITensorInfo *dst, float beta, bool is_log, const ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst, tmp);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_softmax(*src, *dst, beta, *tmp, is_log));

    return Status{};
}

void CpuSoftmaxKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel<CpuSoftmaxKernel>::window(), window);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST_0);

    if (is_data_type_quantized_asymmetric(src->info()->data_type()))
    {
        auto tmp = tensors.get_tensor(TensorType::ACL_DST_1);

        const unsigned int num_elems_processed_per_iteration = src->info()->valid_region().shape.x();
        const unsigned int tmp_size_for_thread = tmp->info()->element_size() * num_elems_processed_per_iteration;

        ARM_COMPUTE_ERROR_ON(tmp->info()->total_size() < (info.num_threads * tmp_size_for_thread));

        void *tmp_for_thread = tmp->buffer() + (info.thread_id * tmp_size_for_thread);
        _run_method(src, tmp_for_thread, dst, _beta, window);
    }
    else
    {
        _run_method(src, nullptr, dst, _beta, window);
    }
}

const char *CpuSoftmaxKernel::name() const
{
    return _name.c_str();
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
