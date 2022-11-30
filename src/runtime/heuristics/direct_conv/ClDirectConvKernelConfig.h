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
#ifndef SRC_RUNTIME_HEURISTICS_DIRECT_CONV_CLDIRECTCONVKERNELCONFIG
#define SRC_RUNTIME_HEURISTICS_DIRECT_CONV_CLDIRECTCONVKERNELCONFIG

#include "src/runtime/heuristics/direct_conv/ClDirectConvDefaultConfigBifrost.h"
#include "src/runtime/heuristics/direct_conv/ClDirectConvDefaultConfigValhall.h"
#include "src/runtime/heuristics/direct_conv/IClDirectConvKernelConfig.h"

#include <memory>

namespace arm_compute
{
namespace cl_direct_conv
{
/** ClDirectConvolution factory class */
class ClDirectConvKernelConfigurationFactory final
{
public:
    /** Static method to call the ClDirectConvolution kernel configuration class accordingly with the GPU target
     *
     * @param[in] gpu GPU target
     *
     * @return IClDirectConvKernelConfig
     */
    static std::unique_ptr<IClDirectConvKernelConfig> create(GPUTarget gpu)
    {
        switch(get_arch_from_target(gpu))
        {
            case GPUTarget::MIDGARD:
                return std::make_unique<ClDirectConvDefaultConfigBifrost>(GPUTarget::G71);
            case GPUTarget::BIFROST:
                return std::make_unique<ClDirectConvDefaultConfigBifrost>(gpu);
            case GPUTarget::VALHALL:
                return std::make_unique<ClDirectConvDefaultConfigValhall>(gpu);
            default:
                ARM_COMPUTE_ERROR("Not supported GPU target");
        }
    }
};
} // namespace opencl
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_DIRECT_CONV_CLDIRECTCONVKERNELCONFIG */
