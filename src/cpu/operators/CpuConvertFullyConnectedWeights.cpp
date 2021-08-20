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
#include "src/cpu/operators/CpuConvertFullyConnectedWeights.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/cpu/kernels/CpuConvertFullyConnectedWeightsKernel.h"

namespace arm_compute
{
namespace cpu
{
void CpuConvertFullyConnectedWeights::configure(const ITensorInfo *src, ITensorInfo *dst, const TensorShape &original_src_shape, DataLayout data_layout)
{
    auto k = std::make_unique<kernels::CpuConvertFullyConnectedWeightsKernel>();
    k->configure(src, dst, original_src_shape, data_layout);
    _kernel = std::move(k);
}

Status CpuConvertFullyConnectedWeights::validate(const ITensorInfo *src, const ITensorInfo *dst, const TensorShape &original_src_shape, DataLayout data_layout)
{
    return kernels::CpuConvertFullyConnectedWeightsKernel::validate(src, dst, original_src_shape, data_layout);
}

void CpuConvertFullyConnectedWeights::run(ITensorPack &tensors)
{
    NEScheduler::get().schedule_op(_kernel.get(), Window::DimZ, _kernel->window(), tensors);
}
} // namesapce cpu
} // namespace arm_compute
