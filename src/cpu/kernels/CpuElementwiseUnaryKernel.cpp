/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#include "src/cpu/kernels/CpuElementwiseUnaryKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/elementwise/neon/elementwise_unary_list.h"
#include "src/cpu/kernels/elementwise/sve/elementwise_unary_list.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct ElementwiseUnarySelectorData
{
    DataType       dt;
    const CPUInfo &ci;
};
using ElementwiseUnarySelector = std::add_pointer<bool(const ElementwiseUnarySelectorData &)>::type;

struct ElementwiseUnaryKernel
{
    const char                                           *name;
    const ElementwiseUnarySelector                        is_selected;
    CpuElementwiseUnaryKernel::ElementwiseUnaryUkernelPtr ukernel;
};

static const ElementwiseUnaryKernel available_kernels[] =
{
#if defined(ARM_COMPUTE_ENABLE_SVE)
    {
        "sve_fp32_elementwise_unary",
        [](const ElementwiseUnarySelectorData & data) { return data.dt == DataType::F32 && data.ci.has_sve(); },
        REGISTER_FP32_SVE(arm_compute::cpu::elementwise_sve_op<float>),
    },
    {
        "sve_fp16_elementwise_unary",
        [](const ElementwiseUnarySelectorData & data) { return data.dt == DataType::F16 && data.ci.has_sve(); },
        REGISTER_FP16_SVE(arm_compute::cpu::elementwise_sve_op<__fp16>),
    },
    {
        "sve_s32_elementwise_unary",
        [](const ElementwiseUnarySelectorData & data) { return data.dt == DataType::S32 && data.ci.has_sve(); },
        REGISTER_INTEGER_SVE(arm_compute::cpu::elementwise_sve_op<int32_t>),
    },
#endif // defined(ARM_COMPUTE_ENABLE_SVE)
#if defined(ARM_COMPUTE_ENABLE_NEON)
    {
        "neon_fp32_elementwise_unary",
        [](const ElementwiseUnarySelectorData & data) { return data.dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::elementwise_op<float>),
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_fp16_elementwise_unary",
        [](const ElementwiseUnarySelectorData & data) { return data.dt == DataType::F16 && data.ci.has_fp16(); },
        REGISTER_FP32_NEON(arm_compute::cpu::elementwise_op<__fp16>),
    },
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "neon_s32_elementwise_unary",
        [](const ElementwiseUnarySelectorData & data) { return data.dt == DataType::S32; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::elementwise_op<int32_t>),
    },
#endif // defined(ARM_COMPUTE_ENABLE_NEON)
};

const ElementwiseUnaryKernel *get_implementation(DataType dt)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected({ dt, CPUInfo::get() }))
        {
            return &uk;
        }
    }
    return nullptr;
}
} // namespace

void CpuElementwiseUnaryKernel::configure(ElementWiseUnary op, const ITensorInfo &src, ITensorInfo &dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(op, src, dst));
    const auto uk = get_implementation(src.data_type());
    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    _op         = op;
    _run_method = uk->ukernel;
    _name       = std::string("CpuElementwiseUnaryKernel").append("/").append(uk->name);

    // If input shape is dynamic, expect a configured window and dst at run-time.
    if(src.is_dynamic())
    {
        return;
    }

    auto shape_and_window = compute_output_shape_and_window(src.tensor_shape());
    auto_init_if_empty(dst, shape_and_window.first, 1, src.data_type());
    ICpuKernel::configure(shape_and_window.second);
}

Status CpuElementwiseUnaryKernel::validate(ElementWiseUnary op, const ITensorInfo &src, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src);

    const auto *uk = get_implementation(src.data_type());
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    switch(op)
    {
        case ElementWiseUnary::EXP:
        case ElementWiseUnary::RSQRT:
        case ElementWiseUnary::LOG:
        case ElementWiseUnary::ROUND:
        case ElementWiseUnary::SIN:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32);
            break;
        case ElementWiseUnary::NEG:
        case ElementWiseUnary::ABS:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src, 1, DataType::F16, DataType::F32, DataType::S32);
            break;
        default:
            ARM_COMPUTE_ERROR("ElementWiseUnary operation not supported");
    }
    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src, &dst);
    }

    return Status{};
}

void CpuElementwiseUnaryKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);

    auto src = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src, dst, window, _op);
}

const char *CpuElementwiseUnaryKernel::name() const
{
    return _name.c_str();
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
