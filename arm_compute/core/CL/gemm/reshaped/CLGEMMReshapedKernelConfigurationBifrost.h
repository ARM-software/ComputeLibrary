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
#ifndef __ARM_COMPUTE_CLGEMMRESHAPEDKERNELCONFIGURATIONBIFROST_H__
#define __ARM_COMPUTE_CLGEMMRESHAPEDKERNELCONFIGURATIONBIFROST_H__

#include "arm_compute/core/CL/ICLGEMMKernelConfiguration.h"

namespace arm_compute
{
namespace cl_gemm
{
/** Bifrost based OpenCL GEMMReshaped configuration */
class CLGEMMReshapedKernelConfigurationBifrost final : public ICLGEMMKernelConfiguration
{
public:
    /** Constructor
     *
     * @param[in] arch GPU target
     */
    CLGEMMReshapedKernelConfigurationBifrost(GPUTarget arch);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMReshapedKernelConfigurationBifrost(const CLGEMMReshapedKernelConfigurationBifrost &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMReshapedKernelConfigurationBifrost &operator=(const CLGEMMReshapedKernelConfigurationBifrost &) = delete;
    /** Default Move Constructor. */
    CLGEMMReshapedKernelConfigurationBifrost(CLGEMMReshapedKernelConfigurationBifrost &&) = default;
    /** Default move assignment operator */
    CLGEMMReshapedKernelConfigurationBifrost &operator=(CLGEMMReshapedKernelConfigurationBifrost &&) = default;

    // Inherited overridden method
    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure(unsigned int m, unsigned int n, unsigned int k, unsigned int b, DataType data_type) override;

private:
    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_G7x_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b);
    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_G76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b);
    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_G7x_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b);
    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_G76_u8(unsigned int m, unsigned int n, unsigned int k, unsigned int b);
};
} // namespace cl_gemm
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMRESHAPEDKERNELCONFIGURATIONBIFROST_H__ */
