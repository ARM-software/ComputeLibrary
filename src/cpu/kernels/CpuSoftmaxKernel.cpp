/*
 * Copyright (c) 2017-2023 Arm Limited.
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
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/softmax/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
/* Softmax Logits 1D Max - identifying the max value of 1D Logits  */
static const std::vector<CpuLogits1DMaxKernel::SoftmaxLogits1DMaxKernel> available_kernels_max_logits = {
    {"sve_fp32_logits_1d_max",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32) && data.isa.sve; },
     REGISTER_FP32_SVE(sve_fp32_logits)},
    {"sve_fp16_logits_1d_max",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16) && data.isa.sve && data.isa.fp16; },
     REGISTER_FP16_SVE(sve_fp16_logits)},
    {"sve_qu8_logits_1d_max",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8) && data.isa.sve; },
     REGISTER_QASYMM8_SVE(sve_qasymm8_logits)},
    {"sve_qs8_logits_1d_max",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED) && data.isa.sve; },
     REGISTER_QASYMM8_SIGNED_SVE(sve_qasymm8_signed_logits)},
    {"neon_fp32_logits_1d_max", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(neon_fp32_logits)},
    {"neon_fp16_logits_1d_max",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16) && data.isa.fp16; },
     REGISTER_FP16_NEON(neon_fp16_logits)},
    {"neon_qu8_logits_1d_max", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(neon_qasymm8_logits)},
    {"neon_qs8_logits_1d_max",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_singed_logits)},
};

Status validate_arguments_logits_1d_max(const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);

    // Validate in case of configured output
    if (output.total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output.tensor_shape(),
                                                           TensorShape(input.tensor_shape()).set(0, 1));
    }

    return Status{};
}
} //namespace
const std::vector<CpuLogits1DMaxKernel::SoftmaxLogits1DMaxKernel> &CpuLogits1DMaxKernel::get_available_kernels()
{
    return available_kernels_max_logits;
}

void CpuLogits1DMaxKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_max(*src, *dst));

    // Softmax across the x dimension
    const TensorShape output_shape = TensorShape(src->tensor_shape()).set(0, 1);
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, src->data_type(), src->quantization_info());

    const auto *uk = get_implementation(DataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    _run_method = uk->ukernel;
    _name       = std::string("CpuLogits1DMaxKernel").append("/").append(uk->name);

    Window win = calculate_max_window(*src, Steps());
    ICpuKernel::configure(win);
}

Status CpuLogits1DMaxKernel::validate(const ITensorInfo *src, const ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_1d_max(*src, *dst));

    return Status{};
}

void CpuLogits1DMaxKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src, dst, window);
}

const char *CpuLogits1DMaxKernel::name() const
{
    return _name.c_str();
}

/* Softmax Logits 1D  - computation for QASYMM8 with pre-computed max.  */
template <bool IS_LOG>
static const std::vector<typename CpuLogits1DSoftmaxKernel<IS_LOG>::SoftmaxLogits1DKernel> available_kernels_logits = {
    {"sve2_qu8_softmax_logits_1d",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8) && data.isa.sve2; },
     REGISTER_QASYMM8_SVE2(sve2_qasymm8_softmax)},
    {"sve2_qs8_softmax_logits_1d",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED) && data.isa.sve2; },
     REGISTER_QASYMM8_SIGNED_SVE2(sve2_qasymm8_signed_softmax)},
    {"sve_fp32_softmax_logits_1d",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32) && data.isa.sve; },
     REGISTER_FP32_SVE(sve_fp32_softmax)},
    {"sve_fp16_softmax_logits_1d",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16) && data.isa.sve && data.isa.fp16; },
     REGISTER_FP16_SVE(sve_fp16_softmax)},

    {"neon_fp32_softmax_logits_1d", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F32); },
     REGISTER_FP32_NEON(neon_fp32_softmax)},
    {"neon_fp16_softmax_logits_1d",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::F16) && data.isa.fp16; },
     REGISTER_FP16_NEON(neon_fp16_softmax)},
    {"neon_qu8_softmax_logits_1d", [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8); },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qasymm8_softmax)},
    {"neon_qs8_softmax_logits_1d",
     [](const DataTypeISASelectorData &data) { return (data.dt == DataType::QASYMM8_SIGNED); },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_qasymm8_signed_softmax)},
};
namespace
{
Status validate_arguments_logits_softmax(const ITensorInfo &src,
                                         const ITensorInfo &max,
                                         const ITensorInfo &dst,
                                         const float        beta,
                                         const ITensorInfo &tmp,
                                         bool               is_log)
{
    ARM_COMPUTE_UNUSED(beta);
    // Check input
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::F16, DataType::F32);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src.data_type());

    // Check max
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &max);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(TensorShape(src.tensor_shape()).set(0, 1), max.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&src, &max);

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
        const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : src.data_type();
        ARM_COMPUTE_RETURN_ERROR_ON(tmp.data_type() != tmp_data_type);
        // We could potentially reduce tmp memory if we could predict or make an assumption
        // on the maximum number of threads that will run in parallel.
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &tmp);
    }

    return Status{};
}
} // namespace

template <bool IS_LOG>
const std::vector<typename CpuLogits1DSoftmaxKernel<IS_LOG>::SoftmaxLogits1DKernel> &
CpuLogits1DSoftmaxKernel<IS_LOG>::get_available_kernels()
{
    return available_kernels_logits<IS_LOG>;
}

template <bool IS_LOG>
void CpuLogits1DSoftmaxKernel<IS_LOG>::configure(
    const ITensorInfo *src, const ITensorInfo *max, ITensorInfo *dst, const float beta, ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, max, dst, tmp);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_softmax(*src, *max, *dst, beta, *tmp, IS_LOG));

    // Configure kernel window
    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src->data_type());

    // Output auto initialization if not yet initialized
    const QuantizationInfo output_quantization =
        is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(src->data_type(), IS_LOG)
                                : dst->quantization_info();
    auto_init_if_empty(*dst, TensorInfo(*src).set_quantization_info(output_quantization).reset_padding());

    // Tmp auto initialization if not yet initialized
    const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : src->data_type();
    auto_init_if_empty(*tmp, TensorInfo(*src).set_data_type(tmp_data_type).reset_padding());

    const auto *uk = CpuLogits1DSoftmaxKernel<IS_LOG>::get_implementation(
        DataTypeISASelectorData{src->data_type(), CPUInfo::get().get_isa()});
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    std::string kernel_name =
        IS_LOG ? std::string("CpuLogits1DLogSoftmaxKernel") : std::string("CpuLogits1DSoftmaxKernel");

    _beta       = beta;
    _run_method = uk->ukernel;
    _name       = kernel_name.append("/").append(uk->name);

    // Configure kernel window
    Window win = calculate_max_window(*max, Steps());

    ICpuKernel<CpuLogits1DSoftmaxKernel<IS_LOG>>::configure(win);
}

template <bool IS_LOG>
Status CpuLogits1DSoftmaxKernel<IS_LOG>::validate(
    const ITensorInfo *src, const ITensorInfo *max, const ITensorInfo *dst, const float beta, const ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, max, dst, tmp);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments_logits_softmax(*src, *max, *dst, beta, *tmp, IS_LOG));

    return Status{};
}

template <bool IS_LOG>
void CpuLogits1DSoftmaxKernel<IS_LOG>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel<CpuLogits1DSoftmaxKernel<IS_LOG>>::window(), window);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto       max = tensors.get_tensor(TensorType::ACL_SRC_1);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST_0);
    auto       tmp = tensors.get_tensor(TensorType::ACL_DST_1);

    const unsigned int num_elems_processed_per_iteration = src->info()->valid_region().shape.x();
    const unsigned int tmp_size_for_thread = tmp->info()->element_size() * num_elems_processed_per_iteration;

    ARM_COMPUTE_ERROR_ON(tmp->info()->total_size() < (info.num_threads * tmp_size_for_thread));

    void *tmp_for_thread = tmp->buffer() + (info.thread_id * tmp_size_for_thread);
    _run_method(src, max, tmp_for_thread, dst, _beta, IS_LOG, window);
}

template <bool IS_LOG>
const char *CpuLogits1DSoftmaxKernel<IS_LOG>::name() const
{
    return _name.c_str();
}

template class CpuLogits1DSoftmaxKernel<true>;
template class CpuLogits1DSoftmaxKernel<false>;

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
