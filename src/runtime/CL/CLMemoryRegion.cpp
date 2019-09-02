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
#include "arm_compute/runtime/CL/CLMemoryRegion.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{
ICLMemoryRegion::ICLMemoryRegion(cl::Context ctx, size_t size)
    : IMemoryRegion(size), _ctx(std::move(ctx)), _mapping(nullptr), _mem()
{
}

const cl::Buffer &ICLMemoryRegion::cl_data() const
{
    return _mem;
}

void *ICLMemoryRegion::buffer()
{
    return _mapping;
}

void *ICLMemoryRegion::buffer() const
{
    return _mapping;
}

std::unique_ptr<IMemoryRegion> ICLMemoryRegion::extract_subregion(size_t offset, size_t size)
{
    ARM_COMPUTE_UNUSED(offset, size);
    return nullptr;
}

CLBufferMemoryRegion::CLBufferMemoryRegion(cl::Context ctx, cl_mem_flags flags, size_t size)
    : ICLMemoryRegion(std::move(ctx), size)
{
    if(_size != 0)
    {
        _mem = cl::Buffer(_ctx, flags, _size);
    }
}

CLBufferMemoryRegion::CLBufferMemoryRegion(const cl::Buffer &buffer)
    : ICLMemoryRegion(buffer.getInfo<CL_MEM_CONTEXT>(), buffer.getInfo<CL_MEM_SIZE>())
{
    _mem = buffer;
}

void *CLBufferMemoryRegion::ptr()
{
    return nullptr;
}

void *CLBufferMemoryRegion::map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_mem.get() == nullptr);
    _mapping = q.enqueueMapBuffer(_mem, blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, 0, _size);
    return _mapping;
}

void CLBufferMemoryRegion::unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_ERROR_ON(_mem.get() == nullptr);
    q.enqueueUnmapMemObject(_mem, _mapping);
    _mapping = nullptr;
}

ICLSVMMemoryRegion::ICLSVMMemoryRegion(cl::Context ctx, cl_mem_flags flags, size_t size, size_t alignment)
    : ICLMemoryRegion(std::move(ctx), size), _ptr(nullptr)
{
    if(size != 0)
    {
        _ptr = clSVMAlloc(_ctx.get(), flags, size, alignment);
        if(_ptr != nullptr)
        {
            _mem = cl::Buffer(_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, _size, _ptr);
        }
    }
}

ICLSVMMemoryRegion::~ICLSVMMemoryRegion()
{
    if(_ptr != nullptr)
    {
        try
        {
            clFinish(CLScheduler::get().queue().get());
            _mem = cl::Buffer();
            clSVMFree(_ctx.get(), _ptr);
        }
        catch(...)
        {
        }
    }
}

void *ICLSVMMemoryRegion::ptr()
{
    return _ptr;
}

CLCoarseSVMMemoryRegion::CLCoarseSVMMemoryRegion(cl::Context ctx, cl_mem_flags flags, size_t size, size_t alignment)
    : ICLSVMMemoryRegion(std::move(ctx), flags, size, alignment)
{
}

void *CLCoarseSVMMemoryRegion::map(cl::CommandQueue &q, bool blocking)
{
    ARM_COMPUTE_ERROR_ON(_ptr == nullptr);
    clEnqueueSVMMap(q.get(), blocking ? CL_TRUE : CL_FALSE, CL_MAP_READ | CL_MAP_WRITE, _ptr, _size, 0, nullptr, nullptr);
    _mapping = _ptr;
    return _mapping;
}

void CLCoarseSVMMemoryRegion::unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_ERROR_ON(_ptr == nullptr);
    clEnqueueSVMUnmap(q.get(), _ptr, 0, nullptr, nullptr);
    _mapping = nullptr;
}

CLFineSVMMemoryRegion::CLFineSVMMemoryRegion(cl::Context ctx, cl_mem_flags flags, size_t size, size_t alignment)
    : ICLSVMMemoryRegion(std::move(ctx), flags, size, alignment)
{
}

void *CLFineSVMMemoryRegion::map(cl::CommandQueue &q, bool blocking)
{
    if(blocking)
    {
        clFinish(q.get());
    }
    _mapping = _ptr;
    return _mapping;
}

void CLFineSVMMemoryRegion::unmap(cl::CommandQueue &q)
{
    ARM_COMPUTE_UNUSED(q);
    _mapping = nullptr;
}
} // namespace arm_compute