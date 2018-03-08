/*
 * Copyright (c) 2018 ARM Limited.
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
#include "arm_compute/graph2/backends/CL/CLTensorHandle.h"

namespace arm_compute
{
namespace graph2
{
namespace backends
{
CLTensorHandle::CLTensorHandle(const ITensorInfo &info)
    : _tensor()
{
    _tensor.allocator()->init(info);
}

void CLTensorHandle::allocate()
{
    _tensor.allocator()->allocate();
}

const arm_compute::ITensor &CLTensorHandle::tensor() const
{
    return _tensor;
}

arm_compute::ITensor &CLTensorHandle::tensor()
{
    return _tensor;
}

void CLTensorHandle::map(bool blocking)
{
    _tensor.map(blocking);
}

void CLTensorHandle::unmap()
{
    _tensor.unmap();
}

void CLTensorHandle::release_if_unused()
{
    // TODO (geopin01): Release tensor only if all sub-tensors are marked as not used
    if(!_tensor.is_used())
    {
        _tensor.allocator()->free();
    }
}

bool CLTensorHandle::is_subtensor() const
{
    return false;
}
} // namespace backends
} // namespace graph2
} // namespace arm_compute