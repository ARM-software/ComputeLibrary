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

#include "arm_compute/dynamic_fusion/sketch/attributes/Pool2dAttributes.h"

#include "arm_compute/core/Size2D.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
PoolingType Pool2dAttributes::pool_type() const
{
    return _pool_type;
}

Pool2dAttributes Pool2dAttributes::pool_type(PoolingType pool_type)
{
    _pool_type = pool_type;
    return *this;
}

Padding2D Pool2dAttributes::pad() const
{
    return _pad;
}

Pool2dAttributes Pool2dAttributes::pad(const Padding2D &pad)
{
    _pad = pad;
    return *this;
}

Size2D Pool2dAttributes::pool_size() const
{
    return _pool_size;
}

Pool2dAttributes Pool2dAttributes::pool_size(const Size2D &pool_size)
{
    _pool_size = pool_size;
    return *this;
}

Size2D Pool2dAttributes::stride() const
{
    return _stride;
}

Pool2dAttributes Pool2dAttributes::stride(const Size2D &stride)
{
    _stride = stride;
    return *this;
}

bool Pool2dAttributes::exclude_padding() const
{
    return _exclude_padding;
}

Pool2dAttributes Pool2dAttributes::exclude_padding(bool exclude_padding)
{
    _exclude_padding = exclude_padding;
    return *this;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
