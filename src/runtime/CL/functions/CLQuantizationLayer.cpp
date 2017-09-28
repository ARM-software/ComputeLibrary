/*
 * Copyright (c) 2017 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLQuantizationLayer.h"

#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLQuantizationLayer::CLQuantizationLayer()
    : _quantize_kernel(), _min_max_kernel(), _min_max()
{
}

void CLQuantizationLayer::configure(const ICLTensor *input, ICLTensor *output)
{
    // Configure min-max kernel. _min_max tensor will be auto-configured within the kernel.
    _min_max_kernel.configure(input, &_min_max);

    // Configure quantize kernel
    _quantize_kernel.configure(input, output, &_min_max);

    // Allocate min_max tensor
    _min_max.allocator()->allocate();
}

void CLQuantizationLayer::run()
{
    cl::CommandQueue q = CLScheduler::get().queue();

    // Reset min and max
    _min_max_kernel.reset(q);

    // Run min-max kernel
    CLScheduler::get().enqueue(_min_max_kernel, false);

    // Run quantize kernel
    CLScheduler::get().enqueue(_quantize_kernel, false);
}
