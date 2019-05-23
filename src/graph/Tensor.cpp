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
#include "arm_compute/graph/Tensor.h"

namespace arm_compute
{
namespace graph
{
Tensor::Tensor(TensorID id, TensorDescriptor desc)
    : _id(id), _desc(std::move(desc)), _handle(nullptr), _accessor(nullptr), _bound_edges()
{
}

TensorID Tensor::id() const
{
    return _id;
}

TensorDescriptor &Tensor::desc()
{
    return _desc;
}

const TensorDescriptor &Tensor::desc() const
{
    return _desc;
}

void Tensor::set_handle(std::unique_ptr<ITensorHandle> backend_tensor)
{
    _handle = std::move(backend_tensor);
}

ITensorHandle *Tensor::handle()
{
    return _handle.get();
}

void Tensor::set_accessor(std::unique_ptr<ITensorAccessor> accessor)
{
    _accessor = std::move(accessor);
}

ITensorAccessor *Tensor::accessor()
{
    return _accessor.get();
}

std::unique_ptr<ITensorAccessor> Tensor::extract_accessor()
{
    return std::move(_accessor);
}

bool Tensor::call_accessor()
{
    // Early exit guard
    if(!_accessor || !_handle)
    {
        return false;
    }

    // Map tensor
    _handle->map(true);

    // Return in case of null backend buffer
    if(_handle->tensor().buffer() == nullptr)
    {
        return false;
    }

    // Call accessor
    bool retval = _accessor->access_tensor(_handle->tensor());

    // Unmap tensor
    _handle->unmap();

    return retval;
}

void Tensor::bind_edge(EdgeID eid)
{
    _bound_edges.insert(eid);
}

void Tensor::unbind_edge(EdgeID eid)
{
    _bound_edges.erase(eid);
}

std::set<EdgeID> Tensor::bound_edges() const
{
    return _bound_edges;
}
} // namespace graph
} // namespace arm_compute
