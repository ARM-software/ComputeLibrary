/*
 * Copyright (c) 2019-2020 ARM Limited.
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
#ifndef ARM_COMPUTE_CLGEMMNATIVEKERNELCONFIGURATION_H
#define ARM_COMPUTE_CLGEMMNATIVEKERNELCONFIGURATION_H

#include "arm_compute/core/CL/ICLGEMMKernelConfiguration.h"
#include "arm_compute/core/CL/gemm/native/CLGEMMNativeKernelConfigurationBifrost.h"
#include "arm_compute/core/CL/gemm/native/CLGEMMNativeKernelConfigurationValhall.h"

#include <memory>

namespace arm_compute
{
namespace cl_gemm
{
/** CLGEMMNative factory class */
class CLGEMMNativeKernelConfigurationFactory final
{
public:
    /** Static method to construct CLGEMMNative kernel object accordingly with the GPU target
     *
     * @param[in] gpu GPU target
     *
     * @return CLGEMMNative kernel configuration class
     */
    static std::unique_ptr<ICLGEMMKernelConfiguration> create(GPUTarget gpu)
    {
        switch(get_arch_from_target(gpu))
        {
            case GPUTarget::MIDGARD:
            case GPUTarget::BIFROST:
                return support::cpp14::make_unique<CLGEMMNativeKernelConfigurationBifrost>(gpu);
                break;
            case GPUTarget::VALHALL:
                return support::cpp14::make_unique<CLGEMMNativeKernelConfigurationValhall>(gpu);
                break;
            default:
                ARM_COMPUTE_ERROR("Not supported GPU target");
        }
    }
};
} // namespace cl_gemm
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLGEMMNATIVEKERNELCONFIGURATION_H */
