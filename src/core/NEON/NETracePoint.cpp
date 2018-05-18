/*
 * Copyright (c) 2020 ARM Limited.
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
#include "arm_compute/core/TracePoint.h"

#include "arm_compute/core/NEON/kernels/NELKTrackerKernel.h"
#include "arm_compute/core/NEON/kernels/assembly/INEGEMMWrapperKernel.h"
#include "arm_compute/core/NEON/kernels/convolution/common/convolution.hpp"
#include "utils/TypePrinter.h"

#include <memory>

namespace arm_compute
{
std::string to_string(const PaddingType &arg)
{
    return ((arg == PADDING_SAME) ? "PADDING_SAME" : "PADDING_VALID");
}

TRACE_TO_STRING(INELKInternalKeypointArray)
TRACE_TO_STRING(std::unique_ptr<INEGEMMWrapperKernel>)

CONST_PTR_CLASS(INELKInternalKeypointArray)
CONST_PTR_CLASS(std::unique_ptr<INEGEMMWrapperKernel>)

template <>
TracePoint::Args &&operator<<(TracePoint::Args &&tp, const PaddingType &arg)
{
    tp.args.push_back("PaddingType(" + to_string(arg) + ")");
    return std::move(tp);
}

} // namespace arm_compute
