/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/runtime/NEON/functions/NEGather.h"

#include "src/core/NEON/kernels/NEGatherKernel.h"
#include "support/MemorySupport.h"

#include <utility>

namespace arm_compute
{
void NEGather::configure(const ITensor *input, const ITensor *indices, ITensor *output, int axis)
{
    auto k = arm_compute::support::cpp14::make_unique<NEGatherKernel>();
    k->configure(input, indices, output, axis);
    _kernel = std::move(k);
}

Status NEGather::validate(const ITensorInfo *input, const ITensorInfo *indices, const ITensorInfo *output, int axis)
{
    return NEGatherKernel::validate(input, indices, output, axis);
}
} // namespace arm_compute
