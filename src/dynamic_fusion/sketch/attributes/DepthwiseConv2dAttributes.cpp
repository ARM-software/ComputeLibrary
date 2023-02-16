/*
 * Copyright (c) 2022 Arm Limited.
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

#include "arm_compute/dynamic_fusion/sketch/attributes/DepthwiseConv2dAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
DepthwiseConv2dAttributes &DepthwiseConv2dAttributes::pad(const Padding2D &pad)
{
    _pad = pad;
    return *this;
}
Padding2D DepthwiseConv2dAttributes::pad() const
{
    return _pad;
}
DepthwiseConv2dAttributes &DepthwiseConv2dAttributes::stride(const Size2D &stride)
{
    _stride = stride;
    return *this;
}
Size2D DepthwiseConv2dAttributes::stride() const
{
    return _stride;
}
DepthwiseConv2dAttributes &DepthwiseConv2dAttributes::dilation(const Size2D &dilation)
{
    _dilation = dilation;
    return *this;
}
Size2D DepthwiseConv2dAttributes::dilation() const
{
    return _dilation;
}

DepthwiseConv2dAttributes &DepthwiseConv2dAttributes::depth_multiplier(const uint32_t &depth_multiplier)
{
    _depth_multiplier = depth_multiplier;
    return *this;
}

uint32_t DepthwiseConv2dAttributes::depth_multiplier() const
{
    return _depth_multiplier;
}

DepthwiseConv2dAttributes &DepthwiseConv2dAttributes::dimension_rounding_type(const DimensionRoundingType &dimension_rounding_type)
{
    _dimension_rounding_type = dimension_rounding_type;
    return *this;
}

DimensionRoundingType DepthwiseConv2dAttributes::dimension_rounding_type() const
{
    return _dimension_rounding_type;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
