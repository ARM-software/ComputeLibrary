/*
 * Copyright (c) 2018-2019 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLComparison.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/kernels/CLComparisonKernel.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
void CLComparison::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output, ComparisonOperation operation)
{
    auto k = arm_compute::support::cpp14::make_unique<CLComparisonKernel>();
    k->configure(input1, input2, output, operation);
    _kernel = std::move(k);

    if(output->info()->dimension(0) > 1)
    {
        ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            _border_handler.configure(broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

Status CLComparison::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ComparisonOperation operation)
{
    return CLComparisonKernel::validate(input1, input2, output, operation);
}

template <ComparisonOperation COP>
void CLComparisonStatic<COP>::configure(ICLTensor *input1, ICLTensor *input2, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLComparisonKernel>();
    k->configure(input1, input2, output, COP);
    _kernel = std::move(k);

    if(output->info()->dimension(0) > 1)
    {
        ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            _border_handler.configure(broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

template <ComparisonOperation COP>
Status CLComparisonStatic<COP>::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output)
{
    return CLComparisonKernel::validate(input1, input2, output, COP);
}

// Supported Specializations
template class CLComparisonStatic<ComparisonOperation::Equal>;
template class CLComparisonStatic<ComparisonOperation::NotEqual>;
template class CLComparisonStatic<ComparisonOperation::Greater>;
template class CLComparisonStatic<ComparisonOperation::GreaterEqual>;
template class CLComparisonStatic<ComparisonOperation::Less>;
template class CLComparisonStatic<ComparisonOperation::LessEqual>;
} // namespace arm_compute
