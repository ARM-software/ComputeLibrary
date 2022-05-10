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
#ifdef ENABLE_EXPERIMENTAL_DYNAMIC_FUSION

#ifndef ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_UTILS
#define ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_UTILS

#include "src/core/experimental/dynamic_fusion/ClKernelBuildingAPI.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
inline ::std::ostream &operator<<(::std::ostream &os, const CLBuildOptions::StringSet &build_opts)
{
    for(const auto &opt : build_opts)
    {
        os << opt << ",";
    }
    return os;
}
inline ::std::ostream &operator<<(::std::ostream &os, const CLBuildOptions &cl_build_opts)
{
    os << cl_build_opts.options();
    return os;
}

inline std::string to_string(const CLBuildOptions &cl_build_opts)
{
    std::stringstream str;
    str << cl_build_opts;
    return str.str();
}
inline ::std::ostream &operator<<(::std::ostream &os, const ClKernelCode &code)
{
    os << "name: " << code.name << std::endl;
    os << "code: " << code.code << std::endl;
    os << "build_opts: " << code.build_options << std::endl;
    return os;
}
inline std::string to_string(const ClKernelCode &code)
{
    std::stringstream str;
    str << code;
    return str.str();
}

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif //ARM_COMPUTE_EXPERIMENTAL_DYNAMICFUSION_IMPL_UTILS
#endif /* ENABLE_EXPERIMENTAL_DYNAMIC_FUSION */