/*
 * Copyright (c) 2025 Arm Limited.
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

#include "src/cpu/operators/CpuMeanStdDevNormalization.h"

#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/common/utils/Log.h"
#include "src/cpu/kernels/CpuMeanStdDevNormalizationKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuMeanStdDevNormalization::configure(ITensorInfo *input, ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_LOG_PARAMS(input, output, epsilon);

    auto k = std::make_unique<kernels::CpuMeanStdDevNormalizationKernel>();
    k->configure(input, output, epsilon);
    _kernel = std::move(k);
}
Status CpuMeanStdDevNormalization::validate(const ITensorInfo *input, const ITensorInfo *output, float epsilon)
{
    ARM_COMPUTE_RETURN_ERROR_ON_DYNAMIC_SHAPE(input, output);
    return arm_compute::cpu::kernels::CpuMeanStdDevNormalizationKernel::validate(input, output, epsilon);
}

void CpuMeanStdDevNormalization::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON_MSG(tensors.empty(), "No inputs provided");
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimY, _kernel->window(), tensors);
}
} // namespace cpu
} // namespace arm_compute
