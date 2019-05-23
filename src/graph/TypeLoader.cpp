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
#include "arm_compute/graph/TypeLoader.h"

#include "arm_compute/core/utils/misc/Utility.h"

#include <map>

namespace arm_compute
{
arm_compute::DataType data_type_from_name(const std::string &name)
{
    static const std::map<std::string, arm_compute::DataType> data_types =
    {
        { "f16", DataType::F16 },
        { "f32", DataType::F32 },
        { "qasymm8", DataType::QASYMM8 },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return data_types.at(arm_compute::utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

arm_compute::DataLayout data_layout_from_name(const std::string &name)
{
    static const std::map<std::string, arm_compute::DataLayout> data_layouts =
    {
        { "nhwc", DataLayout::NHWC },
        { "nchw", DataLayout::NCHW },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return data_layouts.at(arm_compute::utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}
namespace graph
{
Target target_from_name(const std::string &name)
{
    static const std::map<std::string, Target> targets =
    {
        { "neon", Target::NEON },
        { "cl", Target::CL },
        { "gc", Target::GC },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return targets.at(arm_compute::utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

ConvolutionMethod Convolution_method_from_name(const std::string &name)
{
    static const std::map<std::string, ConvolutionMethod> methods =
    {
        { "default", ConvolutionMethod::Default },
        { "direct", ConvolutionMethod::Direct },
        { "gemm", ConvolutionMethod::GEMM },
        { "winograd", ConvolutionMethod::Winograd },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return methods.at(arm_compute::utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

DepthwiseConvolutionMethod depthwise_convolution_method_from_name(const std::string &name)
{
    static const std::map<std::string, DepthwiseConvolutionMethod> methods =
    {
        { "default", DepthwiseConvolutionMethod::Default },
        { "gemv", DepthwiseConvolutionMethod::GEMV },
        { "optimized3x3", DepthwiseConvolutionMethod::Optimized3x3 },
    };

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    try
    {
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
        return methods.at(arm_compute::utility::tolower(name));

#ifndef ARM_COMPUTE_EXCEPTIONS_DISABLED
    }
    catch(const std::out_of_range &)
    {
        throw std::invalid_argument(name);
    }
#endif /* ARM_COMPUTE_EXCEPTIONS_DISABLED */
}

} // namespace graph
} // namespace arm_compute
