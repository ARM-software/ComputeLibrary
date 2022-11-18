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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELSOURCECODE
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELSOURCECODE

#include "arm_compute/core/CL/CLCompileContext.h"
#include "arm_compute/core/Window.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"

#include <map>
#include <string>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** The argument list of a @ref GpuKernelSourceCode */
using GpuKernelArgumentList = std::map<ITensorInfo::Id, GpuKernelArgument>;

/** Container of kernel code to be compiled and run in a @ref GpuUnitWorkload
 */
class GpuKernelSourceCode
{
public:
    /** Set kernel name */
    GpuKernelSourceCode &name(const std::string &n)
    {
        _name = n;
        return *this;
    }
    /** Set kernel code */
    GpuKernelSourceCode &code(const std::string &c)
    {
        _code = c;
        return *this;
    }
    /** Set kernel config id string */
    GpuKernelSourceCode &config_id(const std::string &c_id)
    {
        _config_id = c_id;
        return *this;
    }
    /** Set kernel build options */
    GpuKernelSourceCode &build_options(const CLBuildOptions &b_options)
    {
        _build_options = b_options;
        return *this;
    }
    /** Set kernel execution window */
    GpuKernelSourceCode &window(const Window &window)
    {
        _window = window;
        return *this;
    }
    /** Set kernel argument list */
    GpuKernelSourceCode &arguments(const GpuKernelArgumentList &arguments)
    {
        _arguments = arguments;
        return *this;
    }
    /** Get kernel name */
    std::string name() const
    {
        return _name;
    }
    /** Get kernel code */
    std::string code() const
    {
        return _code;
    }
    /** Get kernel config id string */
    std::string config_id() const
    {
        return _config_id;
    }
    /** Get kernel build options */
    const CLBuildOptions &build_options() const
    {
        return _build_options;
    }
    /** Get kernel execution window */
    const Window &window() const
    {
        return _window;
    }
    /** Get kernel argument list */
    const GpuKernelArgumentList &arguments() const
    {
        return _arguments;
    }

private:
    std::string           _name{};
    std::string           _code{};
    std::string           _config_id{};
    CLBuildOptions        _build_options{};
    Window                _window{};
    GpuKernelArgumentList _arguments{};
};
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_GPUKERNELSOURCECODE */
