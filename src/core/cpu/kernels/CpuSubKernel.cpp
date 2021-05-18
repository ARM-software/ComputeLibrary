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
#include "src/core/cpu/kernels/CpuSubKernel.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/cpu/kernels/sub/neon/list.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct SubSelectorData
{
    DataType dt1;
    DataType dt2;
    DataType dt3;
};

using SubSelectorPtr = std::add_pointer<bool(const SubSelectorData &data)>::type;
using SubKernelPtr   = std::add_pointer<void(const ITensor *, const ITensor *, ITensor *, const ConvertPolicy &, const Window &)>::type;

struct SubKernel
{
    const char          *name;
    const SubSelectorPtr is_selected;
    SubKernelPtr         ukernel;
};

static const SubKernel available_kernels[] =
{
    {
        "sub_same_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::F32)); },
        REGISTER_FP32_NEON(arm_compute::cpu::sub_same_neon<float>)
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "sub_same_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::F16)); },
        REGISTER_FP16_NEON(arm_compute::cpu::sub_same_neon<float16_t>)
    },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    {
        "sub_same_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::U8)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_same_neon<uint8_t>)
    },
    {
        "sub_same_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::S16)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_same_neon<int16_t>)
    },
    {
        "sub_same_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == data.dt3) && (data.dt1 == DataType::S32)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_same_neon<int32_t>)
    },
    {
        "sub_u8_s16_s16_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == DataType::U8) && (data.dt2 == DataType::S16)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_u8_s16_s16_neon)
    },
    {
        "sub_s16_u8_s16_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == DataType::S16) && (data.dt2 == DataType::U8)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_s16_u8_s16_neon)
    },
    {
        "sub_u8_u8_s16_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt3 == DataType::S16)); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_u8_u8_s16_neon)
    },
    {
        "sub_qasymm8_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QASYMM8)); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::sub_qasymm8_neon)
    },
    {
        "sub_qasymm8_signed_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QASYMM8_SIGNED)); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::sub_qasymm8_signed_neon)
    },
    {
        "sub_qsymm16_neon",
        [](const SubSelectorData & data) { return ((data.dt1 == data.dt2) && (data.dt1 == DataType::QSYMM16)); },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::sub_qsymm16_neon)
    },
};

/** Micro-kernel selector
 *
 * @param[in] data Selection data passed to help pick the appropriate micro-kernel
 *
 * @return A matching micro-kernel else nullptr
 */
const SubKernel *get_implementation(DataType dt1, DataType dt2, DataType dt3)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected({ dt1, dt2, dt3 }))
        {
            return &uk;
        }
    }
    return nullptr;
}

inline Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::S32, DataType::F16,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::S32, DataType::F16,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::S32, DataType::F16,
                                                         DataType::F32);

    const auto *uk = get_implementation(src0.data_type(), src1.data_type(), dst.data_type());
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    const TensorShape out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        !(src0.data_type() == DataType::U8 && src1.data_type() == DataType::U8)
        && !(src0.data_type() == DataType::QASYMM8 && src1.data_type() == DataType::QASYMM8)
        && !(src0.data_type() == DataType::QASYMM8_SIGNED && src1.data_type() == DataType::QASYMM8_SIGNED)
        && !(src0.data_type() == DataType::QSYMM16 && src1.data_type() == DataType::QSYMM16)
        && !(src0.data_type() == DataType::U8 && src1.data_type() == DataType::U8)
        && !(src0.data_type() == DataType::U8 && src1.data_type() == DataType::S16)
        && !(src0.data_type() == DataType::S16 && src1.data_type() == DataType::U8)
        && !(src0.data_type() == DataType::S16 && src1.data_type() == DataType::S16)
        && !(src0.data_type() == DataType::S32 && src1.data_type() == DataType::S32)
        && !(src0.data_type() == DataType::F32 && src1.data_type() == DataType::F32)
        && !(src0.data_type() == DataType::F16 && src1.data_type() == DataType::F16),
        "You called subtract with the wrong image formats");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        (src0.data_type() == DataType::QASYMM8_SIGNED && src1.data_type() == DataType::QASYMM8_SIGNED && policy == ConvertPolicy::WRAP)
        || (src0.data_type() == DataType::QASYMM8 && src1.data_type() == DataType::QASYMM8 && policy == ConvertPolicy::WRAP)
        || (src0.data_type() == DataType::QSYMM16 && src1.data_type() == DataType::QSYMM16 && policy == ConvertPolicy::WRAP),
        "Convert policy cannot be WRAP if datatype is quantized");

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            !(src0.data_type() == DataType::U8 && src1.data_type() == DataType::U8 && dst.data_type() == DataType::U8)
            && !(src0.data_type() == DataType::QASYMM8 && src1.data_type() == DataType::QASYMM8 && dst.data_type() == DataType::QASYMM8)
            && !(src0.data_type() == DataType::QASYMM8_SIGNED && src1.data_type() == DataType::QASYMM8_SIGNED && dst.data_type() == DataType::QASYMM8_SIGNED)
            && !(src0.data_type() == DataType::QSYMM16 && src1.data_type() == DataType::QSYMM16 && dst.data_type() == DataType::QSYMM16)
            && !(src0.data_type() == DataType::U8 && src1.data_type() == DataType::U8 && dst.data_type() == DataType::S16)
            && !(src0.data_type() == DataType::U8 && src1.data_type() == DataType::S16 && dst.data_type() == DataType::S16)
            && !(src0.data_type() == DataType::S16 && src1.data_type() == DataType::U8 && dst.data_type() == DataType::S16)
            && !(src0.data_type() == DataType::S16 && src1.data_type() == DataType::S16 && dst.data_type() == DataType::S16)
            && !(src0.data_type() == DataType::S32 && src1.data_type() == DataType::S32 && dst.data_type() == DataType::S32)
            && !(src0.data_type() == DataType::F32 && src1.data_type() == DataType::F32 && dst.data_type() == DataType::F32)
            && !(src0.data_type() == DataType::F16 && src1.data_type() == DataType::F16 && dst.data_type() == DataType::F16),
            "You called subtract with the wrong image formats");

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0),
                                        "Wrong shape for dst");
    }
    return Status{};
}
} // namespace

void CpuSubKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst, policy));

    const TensorShape &out_shape = TensorShape::broadcast_shape(src0->tensor_shape(), src1->tensor_shape());

    // Auto initialize dst if not initialized
    set_shape_if_empty(*dst, out_shape);

    _policy = policy;

    // CpuSubKernel doesn't need padding so update_window_and_padding() can be skipped
    Window win = calculate_max_window(out_shape, Steps());

    ICpuKernel::configure(win);
}

Status CpuSubKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ConvertPolicy policy)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst, policy));

    return Status{};
}

void CpuSubKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    const ITensor *src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst  = tensors.get_tensor(TensorType::ACL_DST);

    // Dispatch kernel
    const auto *uk = get_implementation(src0->info()->data_type(), src1->info()->data_type(), dst->info()->data_type());
    uk->ukernel(src0, src1, dst, _policy, window);
}

const char *CpuSubKernel::name() const
{
    return "CpuSubKernel";
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
