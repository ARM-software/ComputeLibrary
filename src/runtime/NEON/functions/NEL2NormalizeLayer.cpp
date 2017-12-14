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
#include "arm_compute/runtime/NEON/functions/NEL2NormalizeLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

using namespace arm_compute;

NEL2NormalizeLayer::NEL2NormalizeLayer(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _reduce_func(), _normalize_kernel(), _sumsq()
{
}

void NEL2NormalizeLayer::configure(ITensor *input, ITensor *output, unsigned int axis, float epsilon)
{
    // Manage intermediate buffers
    _memory_group.manage(&_sumsq);

    // Configure Kernels
    _reduce_func.configure(input, &_sumsq, axis, ReductionOperation::SUM_SQUARE);
    _normalize_kernel.configure(input, &_sumsq, output, axis, epsilon);

    // Allocate intermediate tensors
    _sumsq.allocator()->allocate();
}

void NEL2NormalizeLayer::run()
{
    _memory_group.acquire();

    _reduce_func.run();
    NEScheduler::get().schedule(&_normalize_kernel, Window::DimY);

    _memory_group.release();
}
