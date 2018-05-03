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
#include "arm_compute/graph/backends/CL/CLSubTensorHandle.h"

#include "arm_compute/core/utils/misc/Cast.h"

namespace arm_compute
{
namespace graph
{
namespace backends
{
CLSubTensorHandle::CLSubTensorHandle(ITensorHandle *parent_handle, const TensorShape &shape, const Coordinates &coords, bool extend_parent)
    : _sub_tensor(), _parent_handle(nullptr)
{
    ARM_COMPUTE_ERROR_ON(!parent_handle);
    auto parent_tensor = arm_compute::utils::cast::polymorphic_downcast<ICLTensor *>(&parent_handle->tensor());
    _sub_tensor        = arm_compute::CLSubTensor(parent_tensor, shape, coords, extend_parent);
    _parent_handle     = parent_handle;
}

void CLSubTensorHandle::allocate()
{
    // noop
}

void CLSubTensorHandle::free()
{
    // noop
}

void CLSubTensorHandle::manage(IMemoryGroup *mg)
{
    ARM_COMPUTE_UNUSED(mg);
    // noop
}

void CLSubTensorHandle::map(bool blocking)
{
    _sub_tensor.map(blocking);
}

void CLSubTensorHandle::unmap()
{
    _sub_tensor.unmap();
}

void CLSubTensorHandle::release_if_unused()
{
    // noop
}

const arm_compute::ITensor &CLSubTensorHandle::tensor() const
{
    return _sub_tensor;
}

arm_compute::ITensor &CLSubTensorHandle::tensor()
{
    return _sub_tensor;
}

ITensorHandle *CLSubTensorHandle::parent_handle()
{
    ARM_COMPUTE_ERROR_ON(_parent_handle == nullptr);
    return _parent_handle->parent_handle();
}

bool CLSubTensorHandle::is_subtensor() const
{
    return true;
}

Target CLSubTensorHandle::target() const
{
    return Target::CL;
}
} // namespace backends
} // namespace graph
} // namespace arm_compute