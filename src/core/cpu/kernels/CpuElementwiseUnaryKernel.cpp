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
#include "src/core/cpu/kernels/CpuElementwiseUnaryKernel.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Validate.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/cpu/kernels/elementwise/neon/elementwise_unary_list.h"
#include "src/core/cpu/kernels/elementwise/sve/elementwise_unary_list.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
using ElementwiseUnarySelector = std::add_pointer<bool(DataType)>::type;

struct ElementwiseUnaryKernel
{
    const char                                           *name;
    const ElementwiseUnarySelector                        is_selected;
    CpuElementwiseUnaryKernel::ElementwiseUnaryUkernelPtr ukernel;
};

static const ElementwiseUnaryKernel available_kernels[] =
{
#if defined(__ARM_FEATURE_SVE)
    {
        "fp32_sve_elementwise_unary",
        [](DataType dt) { return dt == DataType::F32; },
        REGISTER_FP32_SVE(arm_compute::cpu::elementwise_sve_op<float>),
    },
    {
        "fp16_sve_elementwise_unary",
        [](DataType dt) { return dt == DataType::F16; },
        REGISTER_FP16_SVE(arm_compute::cpu::elementwise_sve_op<__fp16>),
    },
    {
        "s32_sve_elementwise_unary",
        [](DataType dt) { return dt == DataType::S32; },
        REGISTER_INTEGER_SVE(arm_compute::cpu::elementwise_sve_op<int32_t>),
    },
#endif // defined(__ARM_FEATURE_SVE)
    {
        "fp32_neon_elementwise_unary",
        [](DataType dt) { return dt == DataType::F32; },
        REGISTER_FP32_NEON(arm_compute::cpu::elementwise_op<float>),
    },
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "fp16_neon_elementwise_unary",
        [](DataType dt) { return dt == DataType::F16; },
        REGISTER_FP32_NEON(arm_compute::cpu::elementwise_op<__fp16>),
    },
#endif // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    {
        "s32_neon_elementwise_unary",
        [](DataType dt) { return dt == DataType::S32; },
        REGISTER_INTEGER_NEON(arm_compute::cpu::elementwise_op<int32_t>),
    },
};

const ElementwiseUnaryKernel *get_implementation(DataType dt)
{
    for(const auto &uk : available_kernels)
    {
        if(uk.is_selected(dt))
        {
            return &uk;
        }
    }
    return nullptr;
}
} // namespace

CpuElementwiseUnaryKernel::CpuElementwiseUnaryKernel()
    : _op()
{
}

void CpuElementwiseUnaryKernel::configure(ElementWiseUnary op, const ITensorInfo &input, ITensorInfo &output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate(op, input, output));

    // Configure kernel window
    const TensorShape &out_shape = TensorShape::broadcast_shape(input.tensor_shape());

    // Auto initialize output if not initialized
    auto_init_if_empty(output, out_shape, 1, input.data_type());

    Window win = calculate_max_window(out_shape);

    _op = op;

    ICpuKernel::configure(win);
}

Status CpuElementwiseUnaryKernel::validate(ElementWiseUnary op, const ITensorInfo &input, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input);

    const auto *uk = get_implementation(input.data_type());
    ARM_COMPUTE_RETURN_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    switch(op)
    {
        case ElementWiseUnary::EXP:
        case ElementWiseUnary::RSQRT:
        case ElementWiseUnary::LOG:
        case ElementWiseUnary::ROUND:
        case ElementWiseUnary::SIN:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::F16, DataType::F32);
            break;
        case ElementWiseUnary::NEG:
        case ElementWiseUnary::ABS:
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input, 1, DataType::F16, DataType::F32, DataType::S32);
            break;
        default:
            ARM_COMPUTE_ERROR("ElementWiseUnary operation not supported");
    }
    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input, &output);
    }

    return Status{};
}

void CpuElementwiseUnaryKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src  = tensors.get_const_tensor(TensorType::ACL_SRC);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);
    auto func = get_implementation(src->info()->data_type())->ukernel;
    ARM_COMPUTE_ERROR_ON(func == nullptr);
    func(src, dst, window, _op);
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
