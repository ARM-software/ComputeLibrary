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
#include "arm_compute/runtime/CL/functions/CLPadLayer.h"

#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/Types.h"
#include "support/ToolchainSupport.h"

namespace arm_compute
{
CLPadLayer::CLPadLayer()
    : _copy_kernel(), _fillborder_kernel(), _memset_kernel()
{
}

void CLPadLayer::configure(ICLTensor *input, ICLTensor *output, const PaddingList &padding, PixelValue constant_value)
{
    // Copy the input to the output
    _copy_kernel.configure(input, output, padding);

    // Set the pages of the output to zero
    _memset_kernel.configure(output, constant_value);

    // Fill padding on the first two dimensions with zeros
    _fillborder_kernel.configure(input, input->info()->padding(), BorderMode::CONSTANT, constant_value);
}

Status CLPadLayer::validate(const ITensorInfo *input, const ITensorInfo *output, const PaddingList &padding, PixelValue constant_value)
{
    ARM_COMPUTE_RETURN_ON_ERROR(CLMemsetKernel::validate(input, constant_value));
    ARM_COMPUTE_RETURN_ON_ERROR(CLCopyKernel::validate(input, output, padding));

    return Status{};
}

void CLPadLayer::run()
{
    CLScheduler::get().enqueue(_memset_kernel, false);
    CLScheduler::get().enqueue(_fillborder_kernel, false);
    CLScheduler::get().enqueue(_copy_kernel, true);
}
} // namespace arm_compute
