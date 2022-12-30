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
#ifndef SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEDEFAULTCONFIGBIFROST
#define SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEDEFAULTCONFIGBIFROST

#include "src/runtime/heuristics/dwc_native/IClDWCNativeKernelConfig.h"

namespace arm_compute
{
namespace cl_dwc
{
/** Bifrost based OpenCL depthwise convolution configuration */
class ClDWCNativeDefaultConfigBifrost final : public IClDWCNativeKernelConfig
{
public:
    /** Constructor
     *
     * @param[in] gpu GPU target
     */
    ClDWCNativeDefaultConfigBifrost(GPUTarget gpu);

    // Inherited overridden method
    DWCComputeKernelInfo configure(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                   unsigned int depth_multiplier) override;

private:
    DWCComputeKernelInfo configure_G71_f32(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                           unsigned int depth_multiplier);
    DWCComputeKernelInfo configure_G71_f16(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                           unsigned int depth_multiplier);
    DWCComputeKernelInfo configure_G7x_f32(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                           unsigned int depth_multiplier);
    DWCComputeKernelInfo configure_G7x_f16(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                           unsigned int depth_multiplier);
    DWCComputeKernelInfo configure_G7x_u8(const ITensorInfo *src, const ITensorInfo *wei, const PadStrideInfo &conv_info, const Size2D &dilation,
                                          unsigned int depth_multiplier);
};
} // namespace cl_dwc
} // namespace arm_compute
#endif /* SRC_RUNTIME_HEURISTICS_DWC_NATIVE_CLDWCNATIVEDEFAULTCONFIGBIFROST */
