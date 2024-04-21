/*
 * Copyright (c) 2022-2023 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/attributes/Conv2dAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Conv2dAttributes &Conv2dAttributes::pad(const Padding2D &pad)
{
    _pad = pad;
    return *this;
}
Padding2D Conv2dAttributes::pad() const
{
    return _pad;
}
Conv2dAttributes &Conv2dAttributes::stride(const Size2D &stride)
{
    _stride = stride;
    return *this;
}
Size2D Conv2dAttributes::stride() const
{
    return _stride;
}
Conv2dAttributes &Conv2dAttributes::dilation(const Size2D &dilation)
{
    _dilation = dilation;
    return *this;
}
Size2D Conv2dAttributes::dilation() const
{
    return _dilation;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
