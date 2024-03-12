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

#include "arm_compute/dynamic_fusion/sketch/attributes/ResizeAttributes.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
ResizeAttributes &ResizeAttributes::output_width(int32_t output_width)
{
    _output_width = output_width;
    return *this;
}

int32_t ResizeAttributes::output_width() const
{
    return _output_width;
}

ResizeAttributes &ResizeAttributes::output_height(int32_t output_height)
{
    _output_height = output_height;
    return *this;
}

int32_t ResizeAttributes::output_height() const
{
    return _output_height;
}

ResizeAttributes &ResizeAttributes::interpolation_policy(InterpolationPolicy interpolation_policy)
{
    _interpolation_policy = interpolation_policy;
    return *this;
}

InterpolationPolicy ResizeAttributes::interpolation_policy() const
{
    return _interpolation_policy;
}

ResizeAttributes &ResizeAttributes::sampling_policy(SamplingPolicy sampling_policy)
{
    _sampling_policy = sampling_policy;
    return *this;
}

SamplingPolicy ResizeAttributes::sampling_policy() const
{
    return _sampling_policy;
}

ResizeAttributes &ResizeAttributes::align_corners(bool align_corners)
{
    _align_corners = align_corners;
    return *this;
}

bool ResizeAttributes::align_corners() const
{
    return _align_corners;
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
