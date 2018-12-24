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
#include "arm_compute/runtime/NEON/functions/NEStridedSlice.h"

#include "arm_compute/core/NEON/kernels/NEStridedSliceKernel.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
void NEStridedSlice::configure(const ITensor *input, ITensor *output,
                               const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                               int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    auto k = arm_compute::support::cpp14::make_unique<NEStridedSliceKernel>();
    k->configure(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
    _kernel = std::move(k);
}

Status NEStridedSlice::validate(const ITensorInfo *input, const ITensorInfo *output,
                                const Coordinates &starts, const Coordinates &ends, const BiStrides &strides,
                                int32_t begin_mask, int32_t end_mask, int32_t shrink_axis_mask)
{
    return NEStridedSliceKernel::validate(input, output, starts, ends, strides, begin_mask, end_mask, shrink_axis_mask);
}
} // namespace arm_compute
