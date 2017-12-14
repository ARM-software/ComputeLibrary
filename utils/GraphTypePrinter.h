/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_GRAPH_TYPE_PRINTER_H__
#define __ARM_COMPUTE_TEST_GRAPH_TYPE_PRINTER_H__

#include "arm_compute/graph/Types.h"

#include <ostream>
#include <sstream>
#include <string>

namespace arm_compute
{
namespace graph
{
/** Formatted output of the @ref ConvolutionMethodHint type. */
inline ::std::ostream &operator<<(::std::ostream &os, const ConvolutionMethodHint &conv_method)
{
    switch(conv_method)
    {
        case ConvolutionMethodHint::DIRECT:
            os << "DIRECT";
            break;
        case ConvolutionMethodHint::GEMM:
            os << "GEMM";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const ConvolutionMethodHint &conv_method)
{
    std::stringstream str;
    str << conv_method;
    return str.str();
}

/** Formatted output of the @ref TargetHint type. */
inline ::std::ostream &operator<<(::std::ostream &os, const TargetHint &target_hint)
{
    switch(target_hint)
    {
        case TargetHint::NEON:
            os << "NEON";
            break;
        case TargetHint::OPENCL:
            os << "OPENCL";
            break;
        default:
            ARM_COMPUTE_ERROR("NOT_SUPPORTED!");
    }

    return os;
}

inline std::string to_string(const TargetHint &target_hint)
{
    std::stringstream str;
    str << target_hint;
    return str.str();
}
} // namespace graph
} // namespace arm_compute
#endif /* __ARM_COMPUTE_TEST_GRAPH_TYPE_PRINTER_H__ */
