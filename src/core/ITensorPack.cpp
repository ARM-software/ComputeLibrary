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
ITensorPack::PackElement *ITensorPack::find(int id)
{
    for (size_t i = 0; i < _inline_size; ++i)
    {
        if (_inline[i].id == id)
        {
            return &_inline[i];
        }
    }
    for (auto &e : _overflow)
    {
        if (e.id == id)
        {
            return &e;
        }
    }
    return nullptr;
}

void ITensorPack::set(const PackElement &e)
{
    if (PackElement *existing = find(e.id))
    {
        *existing = e;
    }
    else if (_inline_size < _inline_capacity)
    {
        _inline[_inline_size++] = e;
    }
    else
    {
        _overflow.push_back(e);
    }
}

ITensorPack::ITensorPack(std::initializer_list<PackElement> l)
{
    for (auto &e : l)
    {
        set(e);
    }
}

void ITensorPack::add_tensor(int id, ITensor *tensor)
{
    set(PackElement(id, tensor));
}

void ITensorPack::add_tensor(int id, const ITensor *tensor)
{
    set(PackElement(id, tensor));
}

void ITensorPack::add_const_tensor(int id, const ITensor *tensor)
{
    add_tensor(id, tensor);
}

const ITensor *ITensorPack::get_const_tensor(int id) const
{
    // find() is non-const only to share one implementation; it does not modify the pack
    const PackElement *e = const_cast<ITensorPack *>(this)->find(id);
    if (e != nullptr)
    {
        return e->ctensor != nullptr ? e->ctensor : e->tensor;
    }
    return nullptr;
}

ITensor *ITensorPack::get_tensor(int id)
{
    const PackElement *e = find(id);
    return e != nullptr ? e->tensor : nullptr;
}

void ITensorPack::remove_tensor(int id)
{
    for (size_t i = 0; i < _inline_size; ++i)
    {
        if (_inline[i].id == id)
        {
            // Order is not observable (there is no iteration API), so fill the hole from the back
            _inline[i] = _inline[_inline_size - 1];
            --_inline_size;
            if (!_overflow.empty())
            {
                _inline[_inline_size++] = _overflow.back();
                _overflow.pop_back();
            }
            return;
        }
    }
    for (size_t i = 0; i < _overflow.size(); ++i)
    {
        if (_overflow[i].id == id)
        {
            _overflow[i] = _overflow.back();
            _overflow.pop_back();
            return;
        }
    }
}

size_t ITensorPack::size() const
{
    return _inline_size + _overflow.size();
}

bool ITensorPack::empty() const
{
    return size() == 0;
}
} // namespace arm_compute
