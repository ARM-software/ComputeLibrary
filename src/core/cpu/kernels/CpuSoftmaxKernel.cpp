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
#include "src/core/cpu/kernels/CpuSoftmaxKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "src/core/CPP/Validate.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "src/core/common/Registrars.h"
#include "src/core/cpu/kernels/softmax/impl/NEON/list.h"
#include "src/core/cpu/kernels/softmax/impl/SVE/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct SoftmaxSelectorData
{
    DataType dt;
};
using SoftmaxSelectorPtr          = std::add_pointer<bool(const SoftmaxSelectorData &data)>::type;
using SoftmaxLogits1DMaxKernelPtr = std::add_pointer<void(const ITensor *, ITensor *, const Window &)>::type;
using SoftmaxLogits1DKernelPtr    = std::add_pointer<void(const ITensor *, const ITensor *, void *const, ITensor *, float, bool, const Window &)>::type;

struct SoftmaxLogits1DKernel
{
    const char              *name;
    const SoftmaxSelectorPtr is_selected;
    SoftmaxLogits1DKernelPtr ukernel;
};

struct SoftmaxLogits1DMaxKernel
{
    const char                 *name;
    const SoftmaxSelectorPtr    is_selected;
    SoftmaxLogits1DMaxKernelPtr ukernel;
};

static const SoftmaxLogits1DKernel available_logits_1d_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "sve_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_SVE(arm_compute::cpu::sve_softmax_logits_1d_float<float>)
    },
    {
        "sve_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_SVE(arm_compute::cpu::sve_softmax_logits_1d_float<float16_t>)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
    {
        "neon_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_NEON(arm_compute::cpu::neon_softmax_logits_1d_float<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_softmax_logits_1d_float",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_softmax_logits_1d_float<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
#endif /* defined(__ARM_FEATURE_SVE) */

#if defined(__ARM_FEATURE_SVE2)
    {
        "sve_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::sve_softmax_logits_1d_quantized<qasymm8_t>)
    },
    {
        "sve_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::sve_softmax_logits_1d_quantized<qasymm8_signed_t>)
    },
#else  /* !defined(__ARM_FEATURE_SVE2) */
    {
        "neon_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_softmax_logits_1d_quantized<qasymm8_t>)
    },
    {
        "neon_softmax_logits_1d_quantized",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_softmax_logits_1d_quantized<qasymm8_signed_t>)
    },
#endif /* defined(__ARM_FEATURE_SVE2) */

};

static const SoftmaxLogits1DMaxKernel available_logits_1d_max_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_SVE(arm_compute::cpu::sve_logits_1d_max<float>)
    },
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_SVE(arm_compute::cpu::sve_logits_1d_max<float16_t>)
    },
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::sve_logits_1d_max<qasymm8_t>)
    },
    {
        "sve_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::sve_logits_1d_max<qasymm8_signed_t>)
    },
#else /* !defined(__ARM_FEATURE_SVE) */
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_NEON(arm_compute::cpu::neon_logits_1d_max<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::F16); },
        REGISTER_FP16_NEON(arm_compute::cpu::neon_logits_1d_max<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_logits_1d_max<qasymm8_t>)
    },
    {
        "neon_logits_1d_max",
        [](const SoftmaxSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_logits_1d_max<qasymm8_signed_t>)
    },
#endif /* defined(__ARM_FEATURE_SVE) */
};

const SoftmaxLogits1DKernel *get_implementation_logits(const SoftmaxSelectorData &data)
{
    for(const auto &uk : available_logits_1d_kernels)
    {
        if(uk.is_selected({ data.dt }))
        {
            return &uk;
        }
    }
    return nullptr;
}

const SoftmaxLogits1DMaxKernel *get_implementation_logits_max(const SoftmaxSelectorData &data)
{
    for(const auto &uk : available_logits_1d_max_kernels)
    {
        if(uk.is_selected({ data.dt }))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments_logits_1d_max(const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);

    // Validate in case of configured output
    if(output.total_size() != 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&input, &output);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(output.tensor_shape(), TensorShape(input.tensor_shape()).set(0, 1));
    }

    return Status{};
}

} // namespace

CpuLogits1DMaxKernel::CpuLogits1DMaxKernel()
{
}

void CpuLogits1DMaxKernel::configure(const ITensorInfo *src, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, dst);

    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_1d_max(*src, *dst));

    // Softmax across the x dimension
    const TensorShape output_shape = TensorShape(src->tensor_shape()).set(0, 1);
    // Output auto initialization if not yet initialized
    auto_init_if_empty(*dst, output_shape, 1, src->data_type(), src->quantization_info());

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

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST);

    const auto *uk = get_implementation_logits_max(SoftmaxSelectorData{ src->info()->data_type() });
    uk->ukernel(src, dst, window);
}

const char *CpuLogits1DMaxKernel::name() const
{
    return "CpuLogits1DMaxKernel";
}

namespace
{
Status validate_arguments_logits_softmax(const ITensorInfo &src, const ITensorInfo &max,
                                         const ITensorInfo &dst, const float beta, const ITensorInfo &tmp, bool is_log)
{
    ARM_COMPUTE_UNUSED(beta);
    // Check input
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::F16, DataType::F32);

    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src.data_type());

    // Check max
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &max);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DIMENSIONS(TensorShape(src.tensor_shape()).set(0, 1), max.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_QUANTIZATION_INFO(&src, &max);

    // Check output if configured
    if(dst.total_size() != 0)
    {
        const QuantizationInfo output_quantization = is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(src.data_type(), is_log) : dst.quantization_info();
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(&src, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON(dst.quantization_info() != output_quantization);
    }

    // Check tmp if configured
    if(tmp.total_size() != 0)
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
CpuLogits1DSoftmaxKernel<IS_LOG>::CpuLogits1DSoftmaxKernel()
    : _beta(1.0f)
{
}

template <bool IS_LOG>
void CpuLogits1DSoftmaxKernel<IS_LOG>::configure(const ITensorInfo *src, const ITensorInfo *max, ITensorInfo *dst, const float beta, ITensorInfo *tmp)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, max, dst, tmp);
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, max, dst, tmp);
    // Perform validation step
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments_logits_softmax(*src, *max, *dst, beta, *tmp, IS_LOG));

    _beta = beta;

    // Configure kernel window
    const bool is_quantized_asymmetric = is_data_type_quantized_asymmetric(src->data_type());

    // Output auto initialization if not yet initialized
    const QuantizationInfo output_quantization = is_quantized_asymmetric ? arm_compute::get_softmax_output_quantization_info(src->data_type(), IS_LOG) : dst->quantization_info();
    auto_init_if_empty(*dst, TensorInfo(*src).set_quantization_info(output_quantization).reset_padding());

    // Tmp auto initialization if not yet initialized
    const DataType tmp_data_type = is_quantized_asymmetric ? DataType::F32 : src->data_type();
    auto_init_if_empty(*tmp, TensorInfo(*src).set_data_type(tmp_data_type).reset_padding());

    // Configure kernel window
    Window win = calculate_max_window(*max, Steps());

    ICpuKernel::configure(win);
}

template <bool IS_LOG>
Status CpuLogits1DSoftmaxKernel<IS_LOG>::validate(const ITensorInfo *src, const ITensorInfo *max,
                                                  const ITensorInfo *dst, const float beta, const ITensorInfo *tmp)
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
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const auto src = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto       max = tensors.get_tensor(TensorType::ACL_SRC_1);
    auto       dst = tensors.get_tensor(TensorType::ACL_DST_0);
    auto       tmp = tensors.get_tensor(TensorType::ACL_DST_1);

    const unsigned int num_elems_processed_per_iteration = src->info()->valid_region().shape.x();
    const unsigned int tmp_size_for_thread               = tmp->info()->element_size() * num_elems_processed_per_iteration;

    ARM_COMPUTE_ERROR_ON(tmp->info()->total_size() < (info.num_threads * tmp_size_for_thread));

    void *tmp_for_thread = tmp->buffer() + (info.thread_id * tmp_size_for_thread);

    const auto *uk = get_implementation_logits(SoftmaxSelectorData{ src->info()->data_type() });
    uk->ukernel(src, max, tmp_for_thread, dst, _beta, IS_LOG, window);
}

template <bool IS_LOG>
const char    *CpuLogits1DSoftmaxKernel<IS_LOG>::name() const
{
    if(IS_LOG)
    {
        return "CpuLogits1DSoftmaxKernel";
    }
    else
    {
        return "CpuLogits1DLogSoftmaxKernel";
    }
}

template class CpuLogits1DSoftmaxKernel<true>;
template class CpuLogits1DSoftmaxKernel<false>;

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
