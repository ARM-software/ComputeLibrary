/*
 * Copyright (c) 2017-2020 ARM Limited.
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
#include "arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h"

#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/kernels/NEArithmeticSubtractionKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void NEArithmeticSubtraction::configure(ITensor *input1, ITensor *input2, ITensor *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_UNUSED(act_info);
    auto k = arm_compute::support::cpp14::make_unique<NEArithmeticSubtractionKernel>();
    k->configure(input1, input2, output, policy);
    _kernel = std::move(k);

    if(output->info()->dimension(0) > 1)
    {
        ITensor *broadcasted_info = (input1->info()->dimension(0) == 1) ? input1 : input2;

        if(broadcasted_info->info()->dimension(0) == 1)
        {
            _border_handler.configure(broadcasted_info, _kernel->border_size(), BorderMode::REPLICATE);
        }
    }
}

Status NEArithmeticSubtraction::validate(const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, ConvertPolicy policy, const ActivationLayerInfo &act_info)
{
    ARM_COMPUTE_RETURN_ERROR_ON(act_info.enabled());
    return NEArithmeticSubtractionKernel::validate(input1, input2, output, policy);
}
} // namespace arm_compute
