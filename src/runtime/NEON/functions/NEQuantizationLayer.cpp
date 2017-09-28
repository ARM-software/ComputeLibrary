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

#include "arm_compute/runtime/NEON/functions/NEQuantizationLayer.h"

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;

NEQuantizationLayer::NEQuantizationLayer()
    : _quantize_kernel(), _min_max_kernel(), _min_max()
{
}

void NEQuantizationLayer::configure(const ITensor *input, ITensor *output)
{
    // Configure min-max kernel. _min_max tensor will be auto-configured within the kernel
    _min_max_kernel.configure(input, &_min_max);

    // Configure quantize kernel
    _quantize_kernel.configure(input, output, &_min_max);

    // Allocate min_max tensor
    _min_max.allocator()->allocate();
}

void NEQuantizationLayer::run()
{
    // Reset min and max
    _min_max_kernel.reset();

    // Run min and max kernel
    NEScheduler::get().schedule(&_min_max_kernel, Window::DimY);

    // Run quantize kernel
    NEScheduler::get().schedule(&_quantize_kernel, Window::DimY);
}
