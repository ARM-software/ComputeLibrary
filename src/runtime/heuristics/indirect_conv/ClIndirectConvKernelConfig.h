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
#ifndef SRC_RUNTIME_HEURISTICS_INDIRECT_CONV_CLINDIRECTCONVKERNELCONFIG
#define SRC_RUNTIME_HEURISTICS_INDIRECT_CONV_CLINDIRECTCONVKERNELCONFIG

#include "src/runtime/heuristics/indirect_conv/ClIndirectConvDefaultConfigValhall.h"
#include "src/runtime/heuristics/indirect_conv/IClIndirectConvKernelConfig.h"

#include <memory>

namespace arm_compute
{
namespace cl_indirect_conv
{
/** ClIndirectConvolution factory class */
class ClIndirectConvKernelConfigurationFactory final
{
public:
    /** Static method to call the ClIndirectConvolution kernel configuration class accordingly with the GPU target
     *
     * @param[in] gpu GPU target
     *
     * @return IClIndirectConvKernelConfig
     */
    static std::unique_ptr<IClIndirectConvKernelConfig> create(GPUTarget gpu)
    {
        switch(get_arch_from_target(gpu))
        {
            case GPUTarget::MIDGARD:
            case GPUTarget::BIFROST:
            case GPUTarget::VALHALL:
                return std::make_unique<ClIndirectConvDefaultConfigValhall>(gpu);
            default:
                ARM_COMPUTE_ERROR("Not supported GPU target");
        }
    }
};
} // namespace cl_indirect_conv
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_INDIRECT_CONV_CLINDIRECTCONVKERNELCONFIG */
