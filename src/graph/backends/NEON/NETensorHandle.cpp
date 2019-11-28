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
#include "arm_compute/graph/backends/NEON/NETensorHandle.h"

#include "arm_compute/core/utils/misc/Cast.h"
#include "arm_compute/runtime/MemoryGroup.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
NETensorHandle::NETensorHandle(const ITensorInfo &info)
    : _tensor()
{
    _tensor.allocator()->init(info);
}

void NETensorHandle::allocate()
{
    _tensor.allocator()->allocate();
}

void NETensorHandle::free()
{
    _tensor.allocator()->free();
}

void NETensorHandle::manage(IMemoryGroup *mg)
{
    if(mg != nullptr)
    {
        mg->manage(&_tensor);
    }
}

void NETensorHandle::map(bool blocking)
{
    ARM_COMPUTE_UNUSED(blocking);
}

void NETensorHandle::unmap()
{
}

void NETensorHandle::release_if_unused()
{
    // TODO (geopin01): Release tensor only if all sub-tensors are marked as not used
    if(!_tensor.is_used())
    {
        _tensor.allocator()->free();
    }
}

const arm_compute::ITensor &NETensorHandle::tensor() const
{
    return _tensor;
}

arm_compute::ITensor &NETensorHandle::tensor()
{
    return _tensor;
}

ITensorHandle *NETensorHandle::parent_handle()
{
    return this;
}

bool NETensorHandle::is_subtensor() const
{
    return false;
}

Target NETensorHandle::target() const
{
    return Target::NEON;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute