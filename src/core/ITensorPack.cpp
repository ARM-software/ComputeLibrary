/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#include "arm_compute/core/ITensorPack.h"

#include "arm_compute/core/ITensor.h"

namespace arm_compute
{
ITensorPack::ITensorPack(std::initializer_list<PackElement> l)
    : _pack()
{
    for(auto &e : l)
    {
        _pack[e.id] = e;
    }
}

void ITensorPack::add_tensor(int id, ITensor *tensor)
{
    _pack[id] = PackElement(id, tensor);
}

void ITensorPack::add_tensor(int id, const ITensor *tensor)
{
    _pack[id] = PackElement(id, tensor);
}

void ITensorPack::add_const_tensor(int id, const ITensor *tensor)
{
    add_tensor(id, tensor);
}

const ITensor *ITensorPack::get_const_tensor(int id) const
{
    auto it = _pack.find(id);
    if(it != _pack.end())
    {
        return it->second.ctensor != nullptr ? it->second.ctensor : it->second.tensor;
    }
    return nullptr;
}

ITensor *ITensorPack::get_tensor(int id)
{
    auto it = _pack.find(id);
    return it != _pack.end() ? it->second.tensor : nullptr;
}

void ITensorPack::remove_tensor(int id)
{
    _pack.erase(id);
}

size_t ITensorPack::size() const
{
    return _pack.size();
}

bool ITensorPack::empty() const
{
    return _pack.empty();
}
} // namespace arm_compute