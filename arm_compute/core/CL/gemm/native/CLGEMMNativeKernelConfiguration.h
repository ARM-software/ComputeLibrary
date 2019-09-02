/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMNATIVEKERNELCONFIGURATION_H__
#define __ARM_COMPUTE_CLGEMMNATIVEKERNELCONFIGURATION_H__

#include "arm_compute/core/CL/ICLGEMMKernelConfiguration.h"
#include "arm_compute/core/CL/gemm/native/CLGEMMNativeKernelConfigurationBifrost.h"

#include <memory>

namespace arm_compute
{
namespace cl_gemm
{
/** CLGEMMNative factory class */
class CLGEMMNativeKernelConfigurationFactory final
{
public:
    /** Static method to construct CLGEMMNative kernel object accordingly with the GPU architecture
     *
     * @param[in] arch GPU target
     *
     * @return CLGEMMNative kernel configuration class
     */
    static std::unique_ptr<ICLGEMMKernelConfiguration> create(GPUTarget arch)
    {
        // Note: At the moment we only support Bifrost architecture. However, we should have a dedicated path for each GPU architecture
        // using get_arch_from_target(arch)
        return support::cpp14::make_unique<CLGEMMNativeKernelConfigurationBifrost>(arch);
    }
};
} // namespace cl_gemm
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMNATIVEKERNELCONFIGURATION_H__ */
