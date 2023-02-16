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
#ifndef SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEKERNELCONFIG
#define SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEKERNELCONFIG

#include "src/runtime/heuristics/dwc_native/ClDWCNativeDefaultConfigBifrost.h"
#include "src/runtime/heuristics/dwc_native/ClDWCNativeDefaultConfigValhall.h"
#include "src/runtime/heuristics/dwc_native/IClDWCNativeKernelConfig.h"

#include <memory>

namespace arm_compute
{
namespace cl_dwc
{
/** ClDWCNativeKernelConfigurationFactory factory class */
class ClDWCNativeKernelConfigurationFactory final
{
public:
    /** Static method to call the ClDWCNative kernel configuration class accordingly with the GPU target
     *
     * @param[in] gpu GPU target
     *
     * @return IClDWCNativeKernelConfig
     */
    static std::unique_ptr<IClDWCNativeKernelConfig> create(GPUTarget gpu)
    {
        switch(get_arch_from_target(gpu))
        {
            case GPUTarget::MIDGARD:
                // The heuristic for Midgard is the same as the one used for Arm Mali-G71
                return std::make_unique<ClDWCNativeDefaultConfigBifrost>(GPUTarget::G71);
            case GPUTarget::BIFROST:
                return std::make_unique<ClDWCNativeDefaultConfigBifrost>(gpu);
            case GPUTarget::VALHALL:
                return std::make_unique<ClDWCNativeDefaultConfigValhall>(gpu);
            default:
                ARM_COMPUTE_ERROR("Not supported GPU target");
        }
    }
};
} // namespace cl_dwc
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEKERNELCONFIG */
