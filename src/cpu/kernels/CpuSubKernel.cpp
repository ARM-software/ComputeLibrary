/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "src/cpu/kernels/CpuSubKernel.h"

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/sub/neon/list.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
static const std::vector<CpuSubKernel::SubKernel> available_kernels =
{
    {
        "neon_fp32_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::F32); },
        REGISTER_FP32_NEON(arm_compute::cpu::sub_same_neon<float>)
    },
    {
        "neon_fp16_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::F16) && data.isa.fp16; },
        REGISTER_FP16_NEON(arm_compute::cpu::sub_same_neon<float16_t>)
    },
    {
        "neon_u8_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::U8); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_same_neon<uint8_t>)
    },
    {
        "neon_s16_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::S16); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_same_neon<int16_t>)
    },
    {
        "neon_s32_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::S32); },
        REGISTER_INTEGER_NEON(arm_compute::cpu::sub_same_neon<int32_t>)
    },
    {
        "neon_qu8_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::QASYMM8); },
        REGISTER_QASYMM8_NEON(arm_compute::cpu::sub_qasymm8_neon)
    },
    {
        "neon_qs8_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::QASYMM8_SIGNED); },
        REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::sub_qasymm8_signed_neon)
    },
    {
        "neon_qs16_sub",
        [](const DataTypeISASelectorData & data) { return (data.dt == DataType::QSYMM16); },
        REGISTER_QSYMM16_NEON(arm_compute::cpu::sub_qsymm16_neon)
    },
};

inline Status validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst, ConvertPolicy policy)
{
    ARM_COMPUTE_UNUSED(policy);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM16, DataType::S16, DataType::S32, DataType::F16,
                                                         DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &src1);

    const auto *uk = CpuSubKernel::get_implementation(DataTypeISASelectorData{ src0.data_type(), CPUInfo::get().get_isa() });

    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    const TensorShape out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape());
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(is_data_type_quantized(src0.data_type()) && (policy == ConvertPolicy::WRAP),
                                    "Convert policy cannot be WRAP if datatype is quantized");

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &dst);
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
    set_data_type_if_unknown(*dst, src0->data_type());

    const auto *uk = CpuSubKernel::get_implementation(DataTypeISASelectorData{ src0->data_type(), CPUInfo::get().get_isa() });
    ARM_COMPUTE_ERROR_ON_NULLPTR(uk);

    _policy     = policy;
    _run_method = uk->ukernel;
    _name       = std::string("CpuSubKernel").append("/").append(uk->name);

    // CpuSubKernel doesn't need padding so update_window_and_padding() can be skipped
    Window win;
    std::tie(win, _split_dimension) = calculate_squashed_or_max_window(*src0, *src1);

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
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    const ITensor *src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const ITensor *src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    ITensor       *dst  = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, dst, _policy, window);
}

const char *CpuSubKernel::name() const
{
    return _name.c_str();
}

const std::vector<CpuSubKernel::SubKernel> &CpuSubKernel::get_available_kernels()
{
    return available_kernels;
}

} // namespace kernels
} // namespace cpu
} // namespace arm_compute
