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
#ifndef __ARM_COMPUTE_CLBUFFERALLOCATOR_H__
#define __ARM_COMPUTE_CLBUFFERALLOCATOR_H__

#include "arm_compute/runtime/IAllocator.h"

#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cstddef>

namespace arm_compute
{
/** Default OpenCL cl buffer allocator implementation */
class CLBufferAllocator : public IAllocator
{
public:
    /** Default constructor */
    explicit CLBufferAllocator(cl::Context context = CLScheduler::get().context());

    // Inherited methods overridden:
    void *allocate(size_t size, size_t alignment) override;
    void free(void *ptr) override;

private:
    cl::Context _context;
};
} // arm_compute
#endif /*__ARM_COMPUTE_CLBUFFERALLOCATOR_H__ */
