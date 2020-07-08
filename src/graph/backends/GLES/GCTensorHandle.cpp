/*
 * Copyright (c) 2018-2019 Arm Limited.
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
#include "arm_compute/graph/backends/GLES/GCTensorHandle.h"

#include "arm_compute/runtime/IMemoryGroup.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
GCTensorHandle::GCTensorHandle(const ITensorInfo &info)
    : _tensor()
{
    _tensor.allocator()->init(info);
}

void GCTensorHandle::allocate()
{
    _tensor.allocator()->allocate();
}

void GCTensorHandle::free()
{
    _tensor.allocator()->free();
}

void GCTensorHandle::manage(IMemoryGroup *mg)
{
    if(mg != nullptr)
    {
        mg->manage(&_tensor);
    }
}

void GCTensorHandle::map(bool blocking)
{
    _tensor.map(blocking);
}

void GCTensorHandle::unmap()
{
    _tensor.unmap();
}

void GCTensorHandle::release_if_unused()
{
    // TODO (geopin01): Release tensor only if all sub-tensors are marked as not used
    if(!_tensor.is_used())
    {
        _tensor.allocator()->free();
    }
}

const arm_compute::ITensor &GCTensorHandle::tensor() const
{
    return _tensor;
}

arm_compute::ITensor &GCTensorHandle::tensor()
{
    return _tensor;
}

ITensorHandle *GCTensorHandle::parent_handle()
{
    return this;
}

bool GCTensorHandle::is_subtensor() const
{
    return false;
}

Target GCTensorHandle::target() const
{
    return Target::GC;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute