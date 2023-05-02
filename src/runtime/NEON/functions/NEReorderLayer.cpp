/*
 * Copyright (c) 2023 Arm Limited.
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
#if defined(__aarch64__)

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEReorderLayer.h"

namespace arm_compute
{

NEReorderLayer::NEReorderLayer()
    : _reorder_kernel(std::make_unique<NEReorderKernel>())
{
}

void NEReorderLayer::configure(const ITensor *input, ITensor *output, arm_compute::WeightFormat input_wf, arm_compute::WeightFormat output_wf)
{
    auto k = std::make_unique<NEReorderKernel>();
    k->configure(input, output, input_wf, output_wf);
    _reorder_kernel = std::move(k);
}

void NEReorderLayer::run()
{
    // Run Reorder
    NEScheduler::get().schedule(_reorder_kernel.get(), Window::DimX);
}

Status NEReorderLayer::validate(const ITensorInfo *input, const ITensorInfo *output, arm_compute::WeightFormat input_wf, arm_compute::WeightFormat output_wf)
{
    return NEReorderKernel::validate(input, output, input_wf, output_wf);
}

} // namespace arm_compute

#endif  // defined(__aarch64__)