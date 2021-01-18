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
#include "src/core/cpu/kernels/CpuElementwiseKernel.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "src/core/CPP/Validate.h"
#include "src/core/common/Registrars.h"
#include "src/core/cpu/kernels/elementwise/neon/elementwise_list.h"
#include "src/core/cpu/kernels/elementwise/neon/elementwise_quantized_list.h"
#include "src/core/cpu/kernels/elementwise/sve/elementwise_list.h"
#include "src/core/cpu/kernels/elementwise/sve/elementwise_quantized_list.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
using ElementwiseSelector = std::add_pointer<bool(DataType)>::type;
using UKernelType         = CpuElementwiseKernel::ElementwiseFunction;
struct ElementwiseKernel
{
    const char               *name;
    const ElementwiseSelector is_selected;
    UKernelType              *ukernel;
};

template <DataType dt>
inline bool is_selected(DataType data_type)
{
    return dt == data_type;
}

template <DataType input_data_type, DataType output_data_type = input_data_type>
static ElementwiseKernel generate_kernel(UKernelType *ukernel)
{
    std::string kernel_name("op_");
    kernel_name += string_from_data_type(input_data_type) + "_";
    kernel_name += string_from_data_type(input_data_type) + "_";
    kernel_name += string_from_data_type(output_data_type);

    return { kernel_name.c_str(), is_selected<input_data_type>, ukernel };
}

template <ArithmeticOperation op>
std::function<void(const ITensor *, const ITensor *, ITensor *, const Window &)>
configure_arithm_func(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(input2, output);
    static ElementwiseKernel kernels[] =
    {
#if defined(__ARM_FEATURE_SVE)
        generate_kernel<DataType::F32>(REGISTER_FP32_SVE((arm_compute::cpu::sve::elementwise_arithmetic_op<op, float32_t>))),
        generate_kernel<DataType::S32>(REGISTER_INTEGER_SVE((arm_compute::cpu::sve::elementwise_arithmetic_op<op, int32_t>))),
#else  /* defined(__ARM_FEATURE_SVE) */
        generate_kernel<DataType::F32>(REGISTER_FP32_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float, 4>>))),
        generate_kernel<DataType::S32>(REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int32_t, 4>>))),
#endif /* defined(__ARM_FEATURE_SVE) */
#if defined(__ARM_FEATURE_SVE2)
        generate_kernel<DataType::QASYMM8>(REGISTER_QASYMM8_SVE((arm_compute::cpu::sve::elementwise_arithmetic_quantized_op<op, uint8_t>))),
        generate_kernel<DataType::QASYMM8_SIGNED>(REGISTER_QASYMM8_SIGNED_SVE((arm_compute::cpu::sve::elementwise_arithmetic_quantized_op<op, int8_t>))),
#else  /* defined(__ARM_FEATURE_SVE2) */
        generate_kernel<DataType::QASYMM8>(REGISTER_QASYMM8_NEON((arm_compute::cpu::elementwise_arithm_op_quantized<op>))),
        generate_kernel<DataType::QASYMM8_SIGNED>(REGISTER_QASYMM8_SIGNED_NEON((arm_compute::cpu::elementwise_arithm_op_quantized_signed<op>))),
#endif /* defined(__ARM_FEATURE_SVE2) */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#if defined(__ARM_FEATURE_SVE)
        generate_kernel<DataType::F16>(REGISTER_FP16_SVE((arm_compute::cpu::sve::elementwise_arithmetic_op<op, float16_t>))),
#else  /* defined(__ARM_FEATURE_SVE) */
        generate_kernel<DataType::F16>(REGISTER_FP16_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<float16_t, 8>>))),
#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
        generate_kernel<DataType::S16>(REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_arithm_op<op, typename wrapper::traits::neon_vector<int16_t, 8>>))),
    };

    for(const auto &uk : kernels)
    {
        if(uk.is_selected(input1->data_type()))
        {
            return uk.ukernel;
        }
    }

    return nullptr;
}

template <ComparisonOperation op>
std::function<void(const ITensor *input1, const ITensor *input2, ITensor *output, const Window &window)>
configure_comp_func(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(input2, output);
    static ElementwiseKernel kernels[] =
    {
#if defined(__ARM_FEATURE_SVE)
        generate_kernel<DataType::U8, DataType::U8>(REGISTER_INTEGER_SVE((arm_compute::cpu::sve::elementwise_comparison_op<op, uint8_t>))),
        generate_kernel<DataType::F32, DataType::U8>(REGISTER_FP32_SVE((arm_compute::cpu::sve::elementwise_comparison_op<op, float>))),
        generate_kernel<DataType::S16, DataType::U8>(REGISTER_INTEGER_SVE((arm_compute::cpu::sve::elementwise_comparison_op<op, int16_t>))),
        generate_kernel<DataType::S32, DataType::U8>(REGISTER_INTEGER_SVE((arm_compute::cpu::sve::elementwise_comparison_op<op, int32_t>))),
#else  /* defined(__ARM_FEATURE_SVE) */
        generate_kernel<DataType::U8, DataType::U8>(REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_comp_op_8<op, uint8_t, uint8x16_t>))),
        generate_kernel<DataType::F32, DataType::U8>(REGISTER_FP32_NEON((arm_compute::cpu::elementwise_comp_op_32<op, float, float32x4_t>))),
        generate_kernel<DataType::S16, DataType::U8>(REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_comp_op_16<op, int16_t, int16x8_t>))),
        generate_kernel<DataType::S32, DataType::U8>(REGISTER_INTEGER_NEON((arm_compute::cpu::elementwise_comp_op_32<op, int32_t, int32x4_t>))),
#endif /* defined(__ARM_FEATURE_SVE) */
#if defined(__ARM_FEATURE_SVE2)
        generate_kernel<DataType::QASYMM8_SIGNED, DataType::U8>(REGISTER_QASYMM8_SIGNED_SVE((arm_compute::cpu::sve::elementwise_comparison_quantized_op<op, int8_t>))),
        generate_kernel<DataType::QASYMM8, DataType::U8>(REGISTER_QASYMM8_SVE((arm_compute::cpu::sve::elementwise_comparison_quantized_op<op, uint8_t>))),
#else  /* defined(__ARM_FEATURE_SVE2) */
        generate_kernel<DataType::QASYMM8_SIGNED, DataType::U8>(REGISTER_QASYMM8_SIGNED_NEON((arm_compute::cpu::elementwise_comp_op_quantized_signed<op>))),
        generate_kernel<DataType::QASYMM8, DataType::U8>(REGISTER_QASYMM8_NEON((arm_compute::cpu::elementwise_comp_op_quantized<op>))),
#endif /* defined(__ARM_FEATURE_SVE2) */
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#if defined(__ARM_FEATURE_SVE)
        generate_kernel<DataType::F16, DataType::U8>(REGISTER_FP16_SVE((arm_compute::cpu::sve::elementwise_comparison_op<op, float16_t>))),
#else  /* defined(__ARM_FEATURE_SVE) */
        generate_kernel<DataType::F16, DataType::U8>(REGISTER_FP16_NEON((arm_compute::cpu::elementwise_comp_op_16<op, float16_t, float16x8_t>))),
#endif /* defined(__ARM_FEATURE_SVE) */
#endif /* __ARM_FEATURE_FP16_VECTOR_ARITHMETIC */
    };

    for(const auto &uk : kernels)
    {
        if(uk.is_selected(input1->data_type()))
        {
            return uk.ukernel;
        }
    }

    return nullptr;
}
} // namespace

Status CpuElementwiseKernel::validate_arguments_common(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(&input1);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &input2);

    const TensorShape out_shape = TensorShape::broadcast_shape(input1.tensor_shape(), input2.tensor_shape());

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(out_shape.total_size() == 0, "Inputs are not broadcast compatible");

    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(detail::have_different_dimensions(out_shape, output.tensor_shape(), 0),
                                        "Wrong shape for output");
    }

    return Status{};
}

void CpuElementwiseKernel::configure_common(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input1, input2, output);

    // Configure kernel window
    const std::pair<TensorShape, ValidRegion> broadcast_pair = ITensorInfo::broadcast_shape_and_valid_region(*input1, *input2);
    const TensorShape &out_shape    = broadcast_pair.first;
    const ValidRegion &valid_region = broadcast_pair.second;

    // Auto initialize output if not initialized
    auto_init_if_empty(*output, out_shape, 1, input1->data_type());

    Window win = calculate_max_window(valid_region);

    ICpuKernel::configure(win);
}

void CpuElementwiseKernel::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info, window);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICpuKernel::window(), window);

    auto src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    auto function = get_implementation(src0->info(), src1->info(), dst->info());
    ARM_COMPUTE_ERROR_ON(function == nullptr);
    function(src0, src1, dst, window);
}

/** Arithmetic operators (min, max, squared_diff) */
void CpuArithmeticKernel::configure(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    _op = op;
}

Status CpuArithmeticKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::F16, DataType::S32, DataType::F32);
    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(&input1, &output);
    }
    return validate_arguments_common(input1, input2, output);
}

Status CpuArithmeticKernel::validate(ArithmeticOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

std::function<CpuElementwiseKernel::ElementwiseFunction>
CpuArithmeticKernel::get_implementation(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    switch(_op)
    {
        case ArithmeticOperation::MAX:
            return configure_arithm_func<ArithmeticOperation::MAX>(input1, input2, output);
        case ArithmeticOperation::MIN:
            return configure_arithm_func<ArithmeticOperation::MIN>(input1, input2, output);
        case ArithmeticOperation::SQUARED_DIFF:
            return configure_arithm_func<ArithmeticOperation::SQUARED_DIFF>(input1, input2, output);
        case ArithmeticOperation::PRELU:
            return configure_arithm_func<ArithmeticOperation::PRELU>(input1, input2, output);
        case ArithmeticOperation::DIV:
            return configure_arithm_func<ArithmeticOperation::DIV>(input1, input2, output);
        case ArithmeticOperation::POWER:
            return configure_arithm_func<ArithmeticOperation::POWER>(input1, input2, output);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return nullptr;
}

/** The division operator */

void CpuDivisionKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    _op = ArithmeticOperation::DIV;
}

Status CpuDivisionKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::S32, DataType::F16, DataType::F32);
    return CpuArithmeticKernel::validate_arguments(input1, input2, output);
}

Status CpuDivisionKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

/** The power operator */
void CpuPowerKernel::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    _op = ArithmeticOperation::POWER;
}

Status CpuPowerKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::F16, DataType::F32);
    return CpuArithmeticKernel::validate_arguments(input1, input2, output);
}

Status CpuPowerKernel::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

/** Comparison operators (equal, not equal, less than, greater than, less than or equal, greater than or equal) */
void CpuComparisonKernel::configure(ComparisonOperation op, const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*input1, *input2, *output));
    configure_common(input1, input2, output);
    _op = op;
}

Status CpuComparisonKernel::validate_arguments(const ITensorInfo &input1, const ITensorInfo &input2, const ITensorInfo &output)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&input1, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::S16, DataType::F16, DataType::S32, DataType::F32);
    // Validate in case of configured output
    if(output.total_size() > 0)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(&output, 1, DataType::U8);
    }
    return validate_arguments_common(input1, input2, output);
}

Status CpuComparisonKernel::validate(ComparisonOperation op, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    ARM_COMPUTE_UNUSED(op);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input1, input2, output);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(*input1, *input2, *output));
    return Status{};
}

std::function<CpuElementwiseKernel::ElementwiseFunction>
CpuComparisonKernel::get_implementation(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    switch(_op)
    {
        case ComparisonOperation::Equal:
            return configure_comp_func<ComparisonOperation::Equal>(input1, input2, output);
        case ComparisonOperation::NotEqual:
            return configure_comp_func<ComparisonOperation::NotEqual>(input1, input2, output);
        case ComparisonOperation::Greater:
            return configure_comp_func<ComparisonOperation::Greater>(input1, input2, output);
        case ComparisonOperation::GreaterEqual:
            return configure_comp_func<ComparisonOperation::GreaterEqual>(input1, input2, output);
        case ComparisonOperation::Less:
            return configure_comp_func<ComparisonOperation::Less>(input1, input2, output);
        case ComparisonOperation::LessEqual:
            return configure_comp_func<ComparisonOperation::LessEqual>(input1, input2, output);
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }
    return nullptr;
}
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
