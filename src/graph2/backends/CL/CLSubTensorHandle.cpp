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
#include "arm_compute/graph2/backends/CL/CLSubTensorHandle.h"

#include "arm_compute/core/utils/misc/Cast.h"

namespace arm_compute
{
namespace graph2
{
namespace backends
{
CLSubTensorHandle::CLSubTensorHandle(ITensorHandle *parent_handle, const TensorShape &shape, const Coordinates &coords)
    : _sub_tensor()
{
    ARM_COMPUTE_ERROR_ON(!parent_handle);
    auto parent_tensor = arm_compute::utils::cast::polymorphic_downcast<ICLTensor *>(&parent_handle->tensor());
    _sub_tensor        = arm_compute::CLSubTensor(parent_tensor, shape, coords);
}

void CLSubTensorHandle::allocate()
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

void CLSubTensorHandle::map(bool blocking)
{
    _sub_tensor.map(blocking);
}

void CLSubTensorHandle::unmap()
{
    _sub_tensor.unmap();
}

bool CLSubTensorHandle::is_subtensor() const
{
    return true;
}
} // namespace backends
} // namespace graph2
} // namespace arm_compute