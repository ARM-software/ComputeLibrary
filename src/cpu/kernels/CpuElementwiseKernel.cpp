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
#include "src/cpu/kernels/CpuElementwiseKernel.h"

#include "arm_compute/core/Helpers.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/elementwise/neon/elementwise_list.h"
#include "src/cpu/kernels/elementwise/neon/elementwise_quantized_list.h"
#include "src/cpu/kernels/elementwise/sve/elementwise_list.h"
#include "src/cpu/kernels/elementwise/sve/elementwise_quantized_list.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
struct ElementwiseSelectorData
{
    DataType       dt;
    const CPUInfo &ci;
};

using ElementwiseSelector = std::add_pointer<bool(const ElementwiseSelectorData &)>::type;
using UKernelType         = CpuElementwiseKernel::ElementwiseFunction;
struct ElementwiseKernel
{
    const char               *name;
    const ElementwiseSelector is_selected;
    UKernelType              *ukernel;
};

template <ArithmeticOperation     op>
CpuElementwiseKernel::UKernelInfo configure_arithm_func(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src1, dst);
    static ElementwiseKernel kernels[] =
    {
#if defined(ARM_COMPUTE_ENABLE_SVE)
        {
            "sve_fp32_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F32 && data.ci.has_sve(); },
            REGISTER_FP32_SVE((arm_compute::cpu::elementwise_arithmetic_op<op, float32_t>))
        },
        {
            "sve_s32_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S32 && data.ci.has_sve(); },
            REGISTER_INTEGER_SVE((arm_compute::cpu::elementwise_arithmetic_op<op, int32_t>))
        },
        {
            "sve_s16_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S16 && data.ci.has_sve(); },
            REGISTER_INTEGER_SVE((arm_compute::cpu::elementwise_arithmetic_op<op, int16_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */
#if defined(ARM_COMPUTE_ENABLE_NEON)
        {
            "neon_fp32_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F32; },
            REGISTER_FP32_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float, 4>>))
        },
        {
            "neon_s32_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S32; },
            REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int32_t, 4>>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) */
#if defined(ARM_COMPUTE_ENABLE_SVE2)
        {
            "sve2_qu8_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8 && data.ci.has_sve2(); },
            REGISTER_QASYMM8_SVE((arm_compute::cpu::elementwise_arithmetic_quantized_op<op, uint8_t>))
        },
        {
            "sve2_qs8_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED && data.ci.has_sve2(); },
            REGISTER_QASYMM8_SIGNED_SVE((arm_compute::cpu::elementwise_arithmetic_quantized_op<op, int8_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
#if defined(ARM_COMPUTE_ENABLE_NEON) || defined(ARM_COMPUTE_ENABLE_SVE)
        {
            "neon_qu8_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8; },
            REGISTER_QASYMM8_NEON((arm_compute::cpu::elementwise_arithm_op_quantized<op>))
        },
        {
            "neon_qs8_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
            REGISTER_QASYMM8_SIGNED_NEON((arm_compute::cpu::elementwise_arithm_op_quantized_signed<op>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON)  || defined(ARM_COMPUTE_ENABLE_SVE) */
#if defined(ARM_COMPUTE_ENABLE_SVE)
        {
            "sve_fp16_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F16 && data.ci.has_sve(); },
            REGISTER_FP16_SVE((arm_compute::cpu::elementwise_arithmetic_op<op, float16_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */
#if defined(ARM_COMPUTE_ENABLE_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        {
            "neon_fp16_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F16 && data.ci.has_fp16(); },
            REGISTER_FP16_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float16_t, 8>>))
        },
#endif /* defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
        {
            "neon_s16_elementwise",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S16; },
            REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int16_t, 8>>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) */
    };

    for(const auto &uk : kernels)
    {
        if(uk.is_selected({ src0->data_type(), CPUInfo::get() }))
        {
            return { uk.name, uk.ukernel };
        }
    }

    return { "", nullptr };
}

template <ComparisonOperation     op>
CpuElementwiseKernel::UKernelInfo configure_comp_func(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(src1, dst);
    static ElementwiseKernel kernels[] =
    {
#if defined(ARM_COMPUTE_ENABLE_SVE)
        {
            "sve_u8_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::U8 && data.ci.has_sve(); },
            REGISTER_INTEGER_SVE((arm_compute::cpu::elementwise_comparison_op<op, uint8_t>))
        },
        {
            "sve_fp32_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F32 && data.ci.has_sve(); },
            REGISTER_FP32_SVE((arm_compute::cpu::elementwise_comparison_op<op, float>))
        },
        {
            "sve_s16_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S16 && data.ci.has_sve(); },
            REGISTER_INTEGER_SVE((arm_compute::cpu::elementwise_comparison_op<op, int16_t>))
        },
        {
            "sve_s32_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S32 && data.ci.has_sve(); },
            REGISTER_INTEGER_SVE((arm_compute::cpu::elementwise_comparison_op<op, int32_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE) */
#if defined(ARM_COMPUTE_ENABLE_NEON)
        {
            "neon_u8_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::U8; },
            REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_comp_op_8<op, uint8_t, uint8x16_t>))
        },
        {
            "neon_fp32_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F32; },
            REGISTER_FP32_NEON((arm_compute::cpu::elementwise_comp_op_32<op, float, float32x4_t>))
        },
        {
            "neon_s16_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S16; },
            REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_comp_op_16<op, int16_t, int16x8_t>))
        },
        {
            "neon_s32_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::S32; },
            REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_comp_op_32<op, int32_t, int32x4_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) */
#if defined(ARM_COMPUTE_ENABLE_SVE2)
        {
            "sve2_qu8_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8 && data.ci.has_sve2(); },
            REGISTER_QASYMM8_SVE((arm_compute::cpu::elementwise_comparison_quantized_op<op, uint8_t>))
        },
        {
            "sve2_qs8_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED && data.ci.has_sve2(); },
            REGISTER_QASYMM8_SIGNED_SVE((arm_compute::cpu::elementwise_comparison_quantized_op<op, int8_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE2) */
#if defined(ARM_COMPUTE_ENABLE_NEON) || defined(ARM_COMPUTE_ENABLE_SVE)
        {
            "neon_qu8_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8; },
            REGISTER_QASYMM8_NEON((arm_compute::cpu::elementwise_comp_op_quantized<op>))
        },
        {
            "neon_qs8_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::QASYMM8_SIGNED; },
            REGISTER_QASYMM8_SIGNED_NEON((arm_compute::cpu::elementwise_comp_op_quantized_signed<op>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON)  || defined(ARM_COMPUTE_ENABLE_SVE) */
#if defined(ARM_COMPUTE_ENABLE_SVE)
        {
            "sve_fp16_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F16 && data.ci.has_sve(); },
            REGISTER_FP16_SVE((arm_compute::cpu::elementwise_comparison_op<op, float16_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_SVE)  */
#if defined(ARM_COMPUTE_ENABLE_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        {
            "neon_fp16_comparison",
            [](const ElementwiseSelectorData & data) { return data.dt == DataType::F16 && data.ci.has_fp16(); },
            REGISTER_FP16_NEON((arm_compute::cpu::elementwise_comp_op_16<op, float16_t, float16x8_t>))
        },
#endif /* defined(ARM_COMPUTE_ENABLE_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) */
    };

    for(const auto &uk : kernels)
    {
        if(uk.is_selected({ src0->data_type(), CPUInfo::get() }))
        {
            return { uk.name, uk.ukernel };
        }
    }

    return { "", nullptr };
}
} // namespace

Status CpuElementwiseKernel::validate_arguments_common(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&src0);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &src1);

    const TensorShape out_shape = TensorShape::broadcast_shape(src0.tensor_shape(), src1.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, dst.tensor_shape(), 0),
                                        "Wrong shape for output");
    }

    return Status{};
}

void CpuElementwiseKernel::configure_common(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    const auto uk = get_implementation(src0, src1, dst);

    _run_method = uk.ukernel;
    _name       = std::string("CpuElementwiseKernel").append("/").append(uk.name);

    // If any of shapes is dynamic, expect a configured window and dst at run-time.
    if(src0->is_dynamic() || src1->is_dynamic())
    {
        return;
    }

    auto shape_and_window = compute_output_shape_and_window(src0->tensor_shape(), src1->tensor_shape());
    auto_init_if_empty(*dst, shape_and_window.first, 1, src0->data_type());
    ICpuKernel::configure(shape_and_window.second);
}

void CpuElementwiseKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    auto src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, dst, window);
}

const char *CpuElementwiseKernel::name() const
{
    return _name.c_str();
}

/** Arithmetic operators (min, max, squared_diff) */
void CpuArithmeticKernel::configure(ArithmeticOperation op, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
    _op = op;
    configure_common(src0, src1, dst);
}

Status CpuArithmeticKernel::validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::F16, DataType::S32, DataType::F32);
    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&src0, &dst);
    }
    return validate_arguments_common(src0, src1, dst);
}

Status CpuArithmeticKernel::validate(ArithmeticOperation op, const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
    return Status{};
}

CpuElementwiseKernel::UKernelInfo CpuArithmeticKernel::get_implementation(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    switch(_op)
    {
        case ArithmeticOperation::MAX:
            return configure_arithm_func<ArithmeticOperation::MAX>(src0, src1, dst);
        case ArithmeticOperation::MIN:
            return configure_arithm_func<ArithmeticOperation::MIN>(src0, src1, dst);
        case ArithmeticOperation::SQUARED_DIFF:
            return configure_arithm_func<ArithmeticOperation::SQUARED_DIFF>(src0, src1, dst);
        case ArithmeticOperation::PRELU:
            return configure_arithm_func<ArithmeticOperation::PRELU>(src0, src1, dst);
        case ArithmeticOperation::DIV:
            return configure_arithm_func<ArithmeticOperation::DIV>(src0, src1, dst);
        case ArithmeticOperation::POWER:
            return configure_arithm_func<ArithmeticOperation::POWER>(src0, src1, dst);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return { "", nullptr };
}

/** The division operator */

void CpuDivisionKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
    _op = ArithmeticOperation::DIV;
    configure_common(src0, src1, dst);
}

Status CpuDivisionKernel::validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::S32, DataType::F16, DataType::F32);
    return CpuArithmeticKernel::validate_arguments(src0, src1, dst);
}

Status CpuDivisionKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
    return Status{};
}

/** The power operator */
void CpuPowerKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
    _op = ArithmeticOperation::POWER;
    configure_common(src0, src1, dst);
}

Status CpuPowerKernel::validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::F16, DataType::F32);
    return CpuArithmeticKernel::validate_arguments(src0, src1, dst);
}

Status CpuPowerKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
    return Status{};
}

/** Comparison operators (equal, not equal, less than, greater than, less than or equal, greater than or equal) */
void CpuComparisonKernel::configure(ComparisonOperation op, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
    _op = op;
    configure_common(src0, src1, dst);
}

Status CpuComparisonKernel::validate_arguments(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&src0, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::F16, DataType::S32, DataType::F32);
    // Validate in case of configured dst
    if(dst.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&dst, 1, DataType::U8);
    }
    return validate_arguments_common(src0, src1, dst);
}

Status CpuComparisonKernel::validate(ComparisonOperation op, const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*src0, *src1, *dst));
    return Status{};
}

CpuElementwiseKernel::UKernelInfo CpuComparisonKernel::get_implementation(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    switch(_op)
    {
        case ComparisonOperation::Equal:
            return configure_comp_func<ComparisonOperation::Equal>(src0, src1, dst);
        case ComparisonOperation::NotEqual:
            return configure_comp_func<ComparisonOperation::NotEqual>(src0, src1, dst);
        case ComparisonOperation::Greater:
            return configure_comp_func<ComparisonOperation::Greater>(src0, src1, dst);
        case ComparisonOperation::GreaterEqual:
            return configure_comp_func<ComparisonOperation::GreaterEqual>(src0, src1, dst);
        case ComparisonOperation::Less:
            return configure_comp_func<ComparisonOperation::Less>(src0, src1, dst);
        case ComparisonOperation::LessEqual:
            return configure_comp_func<ComparisonOperation::LessEqual>(src0, src1, dst);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return { "", nullptr };
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
