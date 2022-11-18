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
#ifndef SRC_DYNAMIC_FUSION_SKETCH_GPU_IGPUKERNELWRITER
#define SRC_DYNAMIC_FUSION_SKETCH_GPU_IGPUKERNELWRITER

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
/** An interface that can write a gpu kernel
 */
class IGpuKernelWriter
{
public:
    /** Destructor */
    virtual ~IGpuKernelWriter()
    {
    }
    /** Generate kernel name */
    virtual std::string get_name() = 0;
    /** Generate kernel code */
    virtual std::string get_code() = 0;
    /** Generate build options */
    virtual CLBuildOptions get_build_options() = 0;
    /** Generate config id string of the entire kernel. This is used for tuning */
    virtual std::string get_config_id() = 0;
    /** Generate execution window */
    virtual Window get_window() const = 0;
    /** Get the kernel argument lists of the kernel*/
    virtual std::map<ITensorInfo::Id, GpuKernelArgument> get_tensors() = 0;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif /* SRC_DYNAMIC_FUSION_SKETCH_GPU_IGPUKERNELWRITER */
