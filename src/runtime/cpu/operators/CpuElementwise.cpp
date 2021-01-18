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
#include "src/runtime/cpu/operators/CpuElementwise.h"
#include "src/core/cpu/kernels/CpuElementwiseKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuElementwiseMax::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<kernels::CpuArithmeticKernel>();
    k->configure(ArithmeticOperation::MAX, input1, input2, output);
    _kernel = std::move(k);
}

Status CpuElementwiseMax::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::CpuArithmeticKernel::validate(ArithmeticOperation::MAX, input1, input2, output);
}

void CpuElementwiseMin::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<kernels::CpuArithmeticKernel>();
    k->configure(ArithmeticOperation::MIN, input1, input2, output);
    _kernel = std::move(k);
}

Status CpuElementwiseMin::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::CpuArithmeticKernel::validate(ArithmeticOperation::MIN, input1, input2, output);
}

void CpuElementwiseSquaredDiff::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<kernels::CpuArithmeticKernel>();
    k->configure(ArithmeticOperation::SQUARED_DIFF, input1, input2, output);
    _kernel = std::move(k);
}

Status CpuElementwiseSquaredDiff::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::CpuArithmeticKernel::validate(ArithmeticOperation::SQUARED_DIFF, input1, input2, output);
}

void CpuElementwiseDivision::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<kernels::CpuDivisionKernel>();
    k->configure(input1, input2, output);
    _kernel = std::move(k);
}

Status CpuElementwiseDivision::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::CpuDivisionKernel::validate(input1, input2, output);
}

void CpuElementwisePower::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<kernels::CpuPowerKernel>();
    k->configure(input1, input2, output);
    _kernel = std::move(k);
}

Status CpuElementwisePower::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::CpuPowerKernel::validate(input1, input2, output);
}

template <ComparisonOperation COP>
void CpuElementwiseComparisonStatic<COP>::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output)
{
    auto k = std::make_unique<kernels::CpuComparisonKernel>();
    k->configure(COP, input1, input2, output);
    _kernel = std::move(k);
}

template <ComparisonOperation COP>
Status CpuElementwiseComparisonStatic<COP>::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return kernels::CpuComparisonKernel::validate(COP, input1, input2, output);
}

void CpuElementwiseComparison::configure(const ITensorInfo *input1, const ITensorInfo *input2, ITensorInfo *output, ComparisonOperation op)
{
    auto k = std::make_unique<kernels::CpuComparisonKernel>();
    k->configure(op, input1, input2, output);
    _kernel = std::move(k);
}

Status CpuElementwiseComparison::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation op)
{
    return kernels::CpuComparisonKernel::validate(op, input1, input2, output);
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