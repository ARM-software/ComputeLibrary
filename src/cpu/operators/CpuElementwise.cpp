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
#include "src/cpu/operators/CpuElementwise.h"
#include "src/common/utils/Log.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/CpuElementwiseKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuElementwiseBase::run(ITensorPack &tensors)
{
    // If the kernel has been configured, use the window from the kernel.
    if(_kernel->is_window_configured())
    {
        ICpuOperator::run(tensors);
        return;
    }

    auto src0_info        = tensors.get_const_tensor(TensorType::ACL_SRC_0)->info();
    auto src1_info        = tensors.get_const_tensor(TensorType::ACL_SRC_1)->info();
    auto shape_and_window = compute_output_shape_and_window(src0_info->tensor_shape(), src1_info->tensor_shape());
    ICpuOperator::run(tensors, shape_and_window.second);
}

template <ArithmeticOperation op>
void CpuElementwiseArithmetic<op>::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src0, src1, dst);
    auto k = std::make_unique<kernels::CpuArithmeticKernel>();
    k->configure(op, src0, src1, dst);
    _kernel = std::move(k);
}

template <ArithmeticOperation op>
Status CpuElementwiseArithmetic<op>::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return kernels::CpuArithmeticKernel::validate(op, src0, src1, dst);
}

template class CpuElementwiseArithmetic<ArithmeticOperation::MAX>;
template class CpuElementwiseArithmetic<ArithmeticOperation::MIN>;
template class CpuElementwiseArithmetic<ArithmeticOperation::SQUARED_DIFF>;
template class CpuElementwiseArithmetic<ArithmeticOperation::PRELU>;

void CpuElementwiseDivision::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src0, src1, dst);
    auto k = std::make_unique<kernels::CpuDivisionKernel>();
    k->configure(src0, src1, dst);
    _kernel = std::move(k);
}

Status CpuElementwiseDivision::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return kernels::CpuDivisionKernel::validate(src0, src1, dst);
}

void CpuElementwisePower::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src0, src1, dst);
    auto k = std::make_unique<kernels::CpuPowerKernel>();
    k->configure(src0, src1, dst);
    _kernel = std::move(k);
}

Status CpuElementwisePower::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return kernels::CpuPowerKernel::validate(src0, src1, dst);
}

template <ComparisonOperation COP>
void CpuElementwiseComparisonStatic<COP>::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst)
{
    ARM_COMPUTE_LOG_PARAMS(src0, src1, dst);
    auto k = std::make_unique<kernels::CpuComparisonKernel>();
    k->configure(COP, src0, src1, dst);
    _kernel = std::move(k);
}

template <ComparisonOperation COP>
Status CpuElementwiseComparisonStatic<COP>::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst)
{
    return kernels::CpuComparisonKernel::validate(COP, src0, src1, dst);
}

void CpuElementwiseComparison::configure(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, ComparisonOperation op)
{
    ARM_COMPUTE_LOG_PARAMS(src0, src1, dst);
    auto k = std::make_unique<kernels::CpuComparisonKernel>();
    k->configure(op, src0, src1, dst);
    _kernel = std::move(k);
}

Status CpuElementwiseComparison::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, ComparisonOperation op)
{
    return kernels::CpuComparisonKernel::validate(op, src0, src1, dst);
}

// Supported Specializations
template class CpuElementwiseComparisonStatic<ComparisonOperation::Equal>;
template class CpuElementwiseComparisonStatic<ComparisonOperation::NotEqual>;
template class CpuElementwiseComparisonStatic<ComparisonOperation::Greater>;
template class CpuElementwiseComparisonStatic<ComparisonOperation::GreaterEqual>;
template class CpuElementwiseComparisonStatic<ComparisonOperation::Less>;
template class CpuElementwiseComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace cpu
} // namespace arm_compute