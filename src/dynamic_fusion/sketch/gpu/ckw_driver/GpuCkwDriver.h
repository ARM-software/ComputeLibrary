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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWDRIVER
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWDRIVER

#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"
#include "src/dynamic_fusion/sketch/gpu/GpuKernelComponentGroup.h"
#include "src/dynamic_fusion/sketch/gpu/IGpuKernelWriter.h"

#include "ckw/Kernel.h"

#include <map>
#include <string>

namespace arm_compute
{
/** Forward declarations */
class Window;

namespace experimental
{
namespace dynamic_fusion
{
/** Use Kernel Writer to write kernel code
 *  Used by dynamic_fusion module
 */
class GpuCkwDriver : public IGpuKernelWriter
{
public:
    /** Default constructor */
    GpuCkwDriver() = default;
    /** Constructor
     *
     * @param[in] components Kernel component group from which the kernel will be generated
     */
    GpuCkwDriver(const GpuKernelComponentGroup &components);
    /** Destructor */
    ~GpuCkwDriver() override = default;
    /** Generate kernel name */
    std::string get_name() override;
    /** Generate kernel code */
    std::string get_code() override;
    /** Generate config id string of the entire kernel. This is used for tuning */
    std::string get_config_id() override;
    /** Generate execution window */
    Window get_window() const override;
    /** Get the flat list of arguments of the kernel*/
    GpuKernelArgumentList get_kernel_arguments() override;

private:
    GpuKernelComponentGroup _components{};
    ckw::Kernel             _kernel;
    std::string             _code;
};

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute

#endif /* ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_GPUCKWDRIVER */
