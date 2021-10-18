/*
 * Copyright (c) 2021 Arm Limited.
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
#include "src/cpu/kernels/CpuAddKernel.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/add/neon/list.h"
#include "src/cpu/kernels/add/sve/list.h"

#include <array>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct AddSelectorData
{
    DataType       dt;
    const CPUInfo &ci;
};

using AddSelectorPtr = std::add_pointer<bool(const AddSelectorData &data)>::type;
using AddKernelPtr   = std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const ConvertPolicy &, const Window &)>::type;
struct AddKernel
{
    const char          *name;
    const AddSelectorPtr is_selected;
    AddKernelPtr         ukernel;
};

static const AddKernel available_kernels[] =
{
#if defined(ARM_COMPUTE_ENABLE_SVE2)
    {
        "sve2_qu8_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::QASYMM8) && data.ci.has_sve2();
        },
        REGISTER_QASYMM8_SVE(arm_compute::cpu::add_qasymm8_sve)
    },
    {
        "sve2_qs8_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::QASYMM8_SIGNED) && data.ci.has_sve2();
        },
        REGISTER_QASYMM8_SIGNED_SVE(arm_compute::cpu::add_qasymm8_signed_sve)
    },
    {
        "sve2_qs16_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::QSYMM16) && data.ci.has_sve2();
        },
        REGISTER_QSYMM16_SVE(arm_compute::cpu::add_qsymm16_sve)
    },
#endif /* !defined(ARM_COMPUTE_ENABLE_SVE2) */
#if defined(ARM_COMPUTE_ENABLE_SVE)
    {
        "sve_fp32_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::F32) && data.ci.has_sve();
        },
        REGISTER_FP32_SVE(arm_compute::cpu::add_same_sve<float>)
    },
    {
        "sve_fp16_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::F16) && data.ci.has_sve();
        },
        REGISTER_FP16_SVE(arm_compute::cpu::add_same_sve<float16_t>)
    },
    {
        "sve_u8_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::U8) && data.ci.has_sve();
        },
        REGISTER_INTEGER_SVE(arm_compute::cpu::add_same_sve<uint8_t>)
    },
    {
        "sve_s16_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::S16) && data.ci.has_sve();
        },
        REGISTER_INTEGER_SVE(arm_compute::cpu::add_same_sve<int16_t>)
    },
    {
        "sve_s32_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::S32) && data.ci.has_sve();
        },
        REGISTER_INTEGER_SVE(arm_compute::cpu::add_same_sve<int32_t>)
    },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */
#if defined(ARM_COMPUTE_ENABLE_NEON)
    {
        "neon_fp32_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_NEON(arm_compute::cpu::add_same_neon<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_fp16_add",
        [](const AddSelectorData & data)
        {
            return (data.dt == DataType::F16) && data.ci.has_fp16();
        },
        REGISTER_FP16_NEON(arm_compute::cpu::add_same_neon<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "neon_u8_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::U8); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::add_same_neon<uint8_t>)
    },
    {
        "neon_s16_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::S16); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::add_same_neon<int16_t>)
    },
    {
        "neon_s32_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::S32); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::add_same_neon<int32_t>)
    },
#endif /*  defined(ARM_COMPUTE_ENABLE_NEON) */
#if defined(ARM_COMPUTE_ENABLE_NEON) || defined(ARM_COMPUTE_ENABLE_SVE)
    {
        "neon_qu8_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::add_qasymm8_neon)
    },
    {
        "neon_qs8_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::add_qasymm8_signed_neon)
    },
    {
        "neon_qs16_add",
        [](const AddSelectorData & data) { return (data.dt == DataType::QSYMM16); },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::add_qsymm16_neon)
    },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) || defined(ARM_COMPUTE_ENABLE_SVE) */
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const AddKernel *get_implementation(const CPUInfo &cpuinfo, DataType dt)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected({ dt, cpuinfo }))
        {
            return &uk;
        }
    }
    return nullptr;
}

Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);

    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED,
                                                         DataType::S16, DataType::QSYMM16, DataType::F16,
                                                         DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &src1);

    const TensorShape out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((src0.tensor_shape().x() != src1.tensor_shape().x()) && ((src0.data_type() != src1.data_type()) || (src0.data_type() != dst.data_type())
                                                                                             || (src1.data_type() != dst.data_type())),
                                    "Broadcasting across width is supported on configurations where all tensors have the same data type");

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &dst);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0),
                                        "Wrong shape for dst");
    }

    const auto *uk = get_implementation(CPUInfo::get(), src0.data_type());
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo &src0, const ITensorInfo &src1, ITensorInfo &dst)
{
    const TensorShape &out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape());

    // Auto initialize dst if not initialized
    set_shape_if_empty(dst, out_shape);
    set_data_type_if_unknown(dst, src0.data_type());

    Window win = calculate_max_window(out_shape, Steps());

    // CpuAddKernel doesn't need padding so update_window_and_padding() can be skipped
    return std::make_pair(Status{}, win);
}
} // namespace

void CpuAddKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst, policy));

    const auto uk = get_implementation(CPUInfo::get(), src0->data_type());
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _policy     = policy;
    _run_method = uk->ukernel;
    _name       = std::string("CpuAddKernel").append("/").append(uk->name);

    // Configure kernel window
    auto win_config = validate_and_configure_window(*src0, *src1, *dst);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICpuKernel::configure(win_config.second);
}

Status CpuAddKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);

    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst, policy));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(*src0->clone(), *src1->clone(), *dst->clone()).first);

    return Status{};
}

void CpuAddKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    ARM_COMPUTE_ERROR_ON(tensors.empty());
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst  = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, dst, _policy, window);
}

const char *CpuAddKernel::name() const
{
    return _name.c_str();
}

size_t CpuAddKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(platform, thread_count);

    return ICPPKernel::small_network_mws;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
