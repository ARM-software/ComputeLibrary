/*
 * Copyright (c) 2017-2020 Arm Limited.
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
#include "arm_compute/runtime/CL/CLBufferAllocator.h"

#include "arm_compute/core/CL/CLCoreRuntimeContext.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLMemoryRegion.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

#include <cstddef>

namespace arm_compute
{
CLBufferAllocator::CLBufferAllocator(CLCoreRuntimeContext *ctx)
    : _ctx(ctx)
{
}

void *CLBufferAllocator::allocate(size_t size, size_t alignment)
{
    ARM_COMPUTE_UNUSED(alignment);
    cl_mem buf;
    if(_ctx == nullptr)
    {
        buf = clCreateBuffer(CLScheduler::get().context().get(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size, nullptr, nullptr);
    }
    else
    {
        buf = clCreateBuffer(_ctx->context().get(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size, nullptr, nullptr);
    }
    return static_cast<void *>(buf);
}

void CLBufferAllocator::free(void *ptr)
{
    ARM_COMPUTE_ERROR_ON(ptr == nullptr);
    clReleaseMemObject(static_cast<cl_mem>(ptr));
}

std::unique_ptr<IMemoryRegion> CLBufferAllocator::make_region(size_t size, size_t alignment)
{
    ARM_COMPUTE_UNUSED(alignment);
    return std::make_unique<CLBufferMemoryRegion>(_ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE, size);
}
} // namespace arm_compute
