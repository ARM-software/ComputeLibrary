/*
 * Copyright (c) 2018-2020 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLNormalizePlanarYUVLayer.h"

#include "arm_compute/core/CL/kernels/CLNormalizePlanarYUVLayerKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void CLNormalizePlanarYUVLayer::configure(const ICLTensor *input, ICLTensor *output, const ICLTensor *mean, const ICLTensor *std)
{
    auto k = arm_compute::support::cpp14::make_unique<CLNormalizePlanarYUVLayerKernel>();
    k->configure(input, output, mean, std);
    _kernel = std::move(k);
}

Status CLNormalizePlanarYUVLayer::validate(const ITensorInfo *input, const ITensorInfo *output,
                                           const ITensorInfo *mean, const ITensorInfo *std)
{
    return CLNormalizePlanarYUVLayerKernel::validate(input, output, mean, std);
}
} // namespace arm_compute
