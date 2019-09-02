/*
 * Copyright (c) 2019 ARM Limited.
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
#include "arm_compute/core/CL/kernels/CLElementwiseOperationKernel.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/runtime/CL/functions/CLPReluLayer.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
namespace
{
void configure_border_handler(CLFillBorderKernel &border_handler, BorderSize border_size, ICLTensor *input1, ICLTensor *input2, const ICLTensor *output)
{
    if(output->info()->dimension(0) > 1)
    {
        ICLTensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            border_handler.configure(broadcasted_info, border_size, BorderMode::REPLICATE);
        }
    }
}
} // namespace

void CLPReluLayer::configure(ICLTensor *input, ICLTensor *alpha, ICLTensor *output)
{
    auto k = arm_compute::support::cpp14::make_unique<CLArithmeticOperationKernel>();
    k->configure(ArithmeticOperation::PRELU, input, alpha, output);
    _kernel = std::move(k);
    configure_border_handler(_border_handler, _kernel->border_size(), input, alpha, output);
}

Status CLPReluLayer::validate(const ITensorInfo *input, const ITensorInfo *alpha, const ITensorInfo *output)
{
    return CLArithmeticOperationKernel::validate(ArithmeticOperation::PRELU, input, alpha, output);
}
} // namespace arm_compute
