/*
 * Copyright (c) 2018-2022 Arm Limited.
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
#include "src/cpu/kernels/elementwise_binary/list.h"

#include <arm_neon.h>

#if defined(ENABLE_FP32_KERNELS)
namespace
{
    static constexpr size_t default_min_max_mws_N1_fp32_neon = 25308;
    static constexpr size_t default_min_max_mws_V1_fp32_neon = 34772;
    static constexpr size_t default_div_mws_N1_fp32_neon = 19043;
    static constexpr size_t default_div_mws_V1_fp32_neon = 25511;
}
#endif /* ENABLE_FP32_KERNELS */

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace
{
template <ArithmeticOperation                                                   op>
const std::vector<CpuElementwiseKernel<CpuArithmeticKernel>::ElementwiseKernel> available_kernels_arithmetic =
{
    {
        "sve2_qu8_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8 && data.isa.sve2 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_SVE2(sve2_qasymm8_elementwise_binary<op>)
    },
    {
        "sve2_qs8_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8_SIGNED && data.isa.sve2 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_SIGNED_SVE2(sve2_qasymm8_signed_elementwise_binary<op>)
    },
    {
        "sve_fp32_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F32 && data.isa.sve && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_FP32_SVE(sve_fp32_elementwise_binary<op>)
    },
    {
        "sve_s32_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S32 && data.isa.sve && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_INTEGER_SVE(sve_s32_elementwise_binary<op>)
    },
    {
        "sve_s16_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S16 && data.isa.sve && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_INTEGER_SVE(sve_s16_elementwise_binary<op>)
    },
    {
        "sve_fp16_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F16 && data.isa.sve && data.isa.fp16 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_FP16_SVE(sve_fp16_elementwise_binary<op>)
    },
    {
        "neon_fp32_arithmetic",

        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F32 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_FP32_NEON(neon_fp32_elementwise_binary<op>)
    },
    {
        "neon_s32_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S32 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_INTEGER_NEON(neon_s32_elementwise_binary<op>)
    },
    {
        "neon_fp16_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F16 && data.isa.fp16 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_FP16_NEON(neon_fp16_elementwise_binary<op>)
    },
    {
        "neon_s16_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S16 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_INTEGER_NEON(neon_s16_elementwise_binary<op>)
    },
    {
        "neon_qu8_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8 && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_NEON(neon_qasymm8_elementwise_binary<op>)
    },
    {
        "neon_qs8_arithmetic",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8_SIGNED && static_cast<ArithmeticOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_elementwise_binary<op>)
    },
};
template <ComparisonOperation                                                   op>
const std::vector<CpuElementwiseKernel<CpuComparisonKernel>::ElementwiseKernel> available_kernels_comperison =
{
    {
        "sve2_qu8_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8 && data.isa.sve2 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_SVE2(sve2_qasymm8_comparison_elementwise_binary<op>)
    },
    {
        "sve2_qs8_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8_SIGNED && data.isa.sve2 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_SIGNED_SVE2(sve2_qasymm8_signed_comparison_elementwise_binary<op>)
    },
    {
        "sve_u8_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::U8 && data.isa.sve && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_INTEGER_SVE(sve_u8_comparison_elementwise_binary<op>)
    },
    {
        "sve_fp32_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F32 && data.isa.sve && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_FP32_SVE(sve_fp32_comparison_elementwise_binary<op>)
    },
    {
        "sve_s16_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S16 && data.isa.sve && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_INTEGER_SVE(sve_s16_comparison_elementwise_binary<op>)
    },
    {
        "sve_s32_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S32 && data.isa.sve && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_INTEGER_SVE(sve_s32_comparison_elementwise_binary<op>)
    },
    {
        "sve_fp16_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F16 && data.isa.sve && data.isa.fp16 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_FP16_SVE(sve_fp16_comparison_elementwise_binary<op>)
    },
    {
        "neon_u8_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::U8 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_INTEGER_NEON(neon_u8_comparison_elementwise_binary<op>)
    },
    {
        "neon_fp32_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F32 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_FP32_NEON(neon_fp32_comparison_elementwise_binary<op>)
    },
    {
        "neon_s16_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S16 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_INTEGER_NEON(neon_s16_comparison_elementwise_binary<op>)
    },
    {
        "neon_s32_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::S32 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_INTEGER_NEON(neon_s32_comparison_elementwise_binary<op>)
    },
    {
        "neon_qu8_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_NEON(neon_qasymm8_comparison_elementwise_binary<op>)
    },
    {
        "neon_qs8_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::QASYMM8_SIGNED && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_QASYMM8_SIGNED_NEON(neon_qasymm8_signed_comparison_elementwise_binary<op>)
    },
    {
        "neon_fp16_comparison",
        [](const ElementwiseDataTypeISASelectorData & data)
        {
            return data.dt == DataType::F16 && data.isa.fp16 && static_cast<ComparisonOperation>(data.op) == op;
        },
        REGISTER_FP16_NEON(neon_fp16_comparison_elementwise_binary<op>)
    },
};
} // namespace

const std::vector<CpuElementwiseKernel<CpuArithmeticKernel>::ElementwiseKernel> &CpuArithmeticKernel::get_available_kernels()
{
    static std::vector<CpuElementwiseKernel<CpuArithmeticKernel>::ElementwiseKernel> available_kernels;
    std::move(available_kernels_arithmetic<ArithmeticOperation::ADD>.begin(), available_kernels_arithmetic<ArithmeticOperation::ADD>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::SUB>.begin(), available_kernels_arithmetic<ArithmeticOperation::SUB>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::DIV>.begin(), available_kernels_arithmetic<ArithmeticOperation::DIV>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::MIN>.begin(), available_kernels_arithmetic<ArithmeticOperation::MIN>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::MAX>.begin(), available_kernels_arithmetic<ArithmeticOperation::MAX>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::SQUARED_DIFF>.begin(), available_kernels_arithmetic<ArithmeticOperation::SQUARED_DIFF>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::POWER>.begin(), available_kernels_arithmetic<ArithmeticOperation::POWER>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_arithmetic<ArithmeticOperation::PRELU>.begin(), available_kernels_arithmetic<ArithmeticOperation::PRELU>.end(), std::back_inserter(available_kernels));

    return available_kernels;
}

const std::vector<CpuElementwiseKernel<CpuComparisonKernel>::ElementwiseKernel> &CpuComparisonKernel::get_available_kernels()
{
    static std::vector<CpuElementwiseKernel<CpuComparisonKernel>::ElementwiseKernel> available_kernels;
    std::move(available_kernels_comperison<ComparisonOperation::Equal>.begin(), available_kernels_comperison<ComparisonOperation::Equal>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_comperison<ComparisonOperation::NotEqual>.begin(), available_kernels_comperison<ComparisonOperation::NotEqual>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_comperison<ComparisonOperation::Greater>.begin(), available_kernels_comperison<ComparisonOperation::Greater>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_comperison<ComparisonOperation::GreaterEqual>.begin(), available_kernels_comperison<ComparisonOperation::GreaterEqual>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_comperison<ComparisonOperation::Less>.begin(), available_kernels_comperison<ComparisonOperation::Less>.end(), std::back_inserter(available_kernels));
    std::move(available_kernels_comperison<ComparisonOperation::LessEqual>.begin(), available_kernels_comperison<ComparisonOperation::LessEqual>.end(), std::back_inserter(available_kernels));

    return available_kernels;
}

template <class Derived>
Status CpuElementwiseKernel<Derived>::validate_arguments_common(const ITensorInfo &src0, const ITensorInfo &src1, const ITensorInfo &dst)
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

void CpuArithmeticKernel::configure_common(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    const auto *uk = CpuArithmeticKernel::get_implementation(ElementwiseDataTypeISASelectorData{ src0->data_type(), CPUInfo::get().get_isa(), static_cast<int>(_op) });

    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    _run_method = uk->ukernel;
    _name       = std::string("CpuArithmeticKernel").append("/").append(uk->name);

    // If any of shapes is dynamic, expect a configured window and dst at run-time.
    if(src0->is_dynamic() || src1->is_dynamic())
    {
        return;
    }

    auto shape_and_window = compute_output_shape_and_window(src0->tensor_shape(), src1->tensor_shape());
    auto_init_if_empty(*dst, shape_and_window.first, 1, src0->data_type());
    ICpuKernel::configure(shape_and_window.second);
}

void CpuComparisonKernel::configure_common(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);

    const auto *uk = CpuComparisonKernel::get_implementation(ElementwiseDataTypeISASelectorData{ src0->data_type(), CPUInfo::get().get_isa(), static_cast<int>(_op) });

    ARM_COMPUTE_ERROR_ON(uk == nullptr || uk->ukernel == nullptr);

    _run_method = uk->ukernel;
    _name       = std::string("CpuComparisonKernel").append("/").append(uk->name);

    // If any of shapes is dynamic, expect a configured window and dst at run-time.
    if(src0->is_dynamic() || src1->is_dynamic())
    {
        return;
    }

    auto shape_and_window = compute_output_shape_and_window(src0->tensor_shape(), src1->tensor_shape());
    auto_init_if_empty(*dst, shape_and_window.first, 1, src0->data_type());
    ICpuKernel::configure(shape_and_window.second);
}

template <class Derived>
void CpuElementwiseKernel<Derived>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info)
{
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON(_run_method == nullptr);

    auto src0 = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto src1 = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto dst  = tensors.get_tensor(TensorType::ACL_DST);

    _run_method(src0, src1, dst, window);
}
template void CpuElementwiseKernel<CpuArithmeticKernel>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info);
template void CpuElementwiseKernel<CpuComparisonKernel>::run_op(ITensorPack &tensors, const Window &window, const ThreadInfo &info);

template <class Derived>
const char *CpuElementwiseKernel<Derived>::name() const
{
    return _name.c_str();
}
template const char *CpuElementwiseKernel<CpuArithmeticKernel>::name() const;
template const char *CpuElementwiseKernel<CpuComparisonKernel>::name() const;

/** Arithmetic operators (min, max, squared_diff) */
void CpuArithmeticKernel::configure(ArithmeticOperation op, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
    _op = op;
    CpuArithmeticKernel::configure_common(src0, src1, dst);
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

size_t CpuArithmeticKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
    if(this->_run_method == &neon_fp32_elementwise_binary<ArithmeticOperation::MIN>
    || this->_run_method == &neon_fp32_elementwise_binary<ArithmeticOperation::MAX>)
    {
        size_t mws = ICPPKernel::default_mws;
        if(platform.get_cpu_model() == CPUModel::N1)
        {
            mws = default_min_max_mws_N1_fp32_neon;
        }
        else if(platform.get_cpu_model() == CPUModel::V1)
        {
            mws = default_min_max_mws_V1_fp32_neon;
        }
        else
        {
            return ICPPKernel::default_mws;
        }

        // tensor is 1D or was re-interpreted as 1D
        if(this->window().shape().num_dimensions() == 1)
        {
            return mws;
        }
        else
        {
            // scale mws down by the number of elements along all the dimensions (x, z, w, etc) except the one
            // that we parallelize along (the y dimension). This allows for parallelization when the Y_SIZE is small
            // but the other sizes are large, which boosts performance.
            mws = static_cast<size_t>(mws / (this->window().num_iterations_total() / this->window().num_iterations(1)));
            return std::max(static_cast<size_t>(1), mws);
        }
    }
#else /* ENABLE_FP32_KERNELS */
    ARM_COMPUTE_UNUSED(platform);
#endif /* ENABLE_FP32_KERNELS */
    return ICPPKernel::default_mws;
}

/** The division operator */

void CpuDivisionKernel::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(*src0, *src1, *dst));
    _op = ArithmeticOperation::DIV;
    CpuArithmeticKernel::configure_common(src0, src1, dst);
}

size_t CpuDivisionKernel::get_mws(const CPUInfo &platform, size_t thread_count) const
{
    ARM_COMPUTE_UNUSED(thread_count);

#if defined(ENABLE_FP32_KERNELS)
    if(this->_run_method == &neon_fp32_elementwise_binary<ArithmeticOperation::DIV>)
    {
        size_t mws = ICPPKernel::default_mws;
        if(platform.get_cpu_model() == CPUModel::N1)
        {
            mws = default_div_mws_N1_fp32_neon;
        }
        else if(platform.get_cpu_model() == CPUModel::V1)
        {
            mws = default_div_mws_V1_fp32_neon;
        }
        else
        {
            return ICPPKernel::default_mws;
        }

        // tensor is 1D or was re-interpreted as 1D
        if(this->window().shape().num_dimensions() == 1)
        {
            return mws;
        }
        else
        {
            // scale mws down by the number of elements along all the dimensions (x, z, w, etc) except the one
            // that we parallelize along (the y dimension). This allows for parallelization when the Y_SIZE is small
            // but the other sizes are large, which boosts performance.
            mws = static_cast<size_t>(mws / (this->window().num_iterations_total() / this->window().num_iterations(1)));
            return std::max(static_cast<size_t>(1), mws);
        }
    }
#else /* ENABLE_FP32_KERNELS */
    ARM_COMPUTE_UNUSED(platform);
#endif /* ENABLE_FP32_KERNELS */
    return ICPPKernel::default_mws;
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
    CpuArithmeticKernel::configure_common(src0, src1, dst);
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
    CpuComparisonKernel::configure_common(src0, src1, dst);
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
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
