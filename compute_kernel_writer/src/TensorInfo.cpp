/*
 * Copyright (c) 2023 Arm Limited.
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

#include "ckw/TensorInfo.h"

namespace ckw
{
TensorInfo::TensorInfo(DataType dt, const TensorShape &shape, TensorDataLayout dl, int32_t id)
    : _shape(shape), _dt(dt), _dl(dl), _id(id)
{
}

TensorInfo &TensorInfo::shape(const TensorShape &shape)
{
    _shape = shape;
    return *this;
}

TensorShape TensorInfo::shape() const
{
    return _shape;
}

TensorInfo &TensorInfo::data_type(DataType dt)
{
    _dt = dt;
    return *this;
}

DataType TensorInfo::data_type() const
{
    return _dt;
}

TensorInfo &TensorInfo::data_layout(TensorDataLayout dl)
{
    _dl = dl;
    return *this;
}

TensorDataLayout TensorInfo::data_layout() const
{
    return _dl;
}

TensorInfo &TensorInfo::id(int32_t id)
{
    _id = id;
    return *this;
}

int32_t TensorInfo::id() const
{
    return _id;
}
} // namespace ckw
