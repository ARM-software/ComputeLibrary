/*
 * Copyright (c) 2020 Arm Limited.
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
#ifndef SRC_CLGEMMKERNELSELECTION_H
#define SRC_CLGEMMKERNELSELECTION_H

#include "arm_compute/runtime/CL/ICLGEMMKernelSelection.h"
#include "src/runtime/CL/gemm/CLGEMMDefaultTypeBifrost.h"
#include "src/runtime/CL/gemm/CLGEMMDefaultTypeMidgard.h"
#include "src/runtime/CL/gemm/CLGEMMDefaultTypeValhall.h"

namespace arm_compute
{
namespace cl_gemm
{
/** CLGEMMKernelSelection factory class */
class CLGEMMKernelSelectionFactory final
{
public:
    /** Static method to select the GEMM kernel accordingly with the GPU target and GEMM's dimensionality
     *
     * @param[in] gpu GPU target
     *
     * @return CLGEMMKernelSelection class
     */
    static std::unique_ptr<ICLGEMMKernelSelection> create(GPUTarget gpu)
    {
        switch(get_arch_from_target(gpu))
        {
            case GPUTarget::MIDGARD:
                return std::make_unique<CLGEMMDefaultTypeMidgard>(gpu);
            case GPUTarget::BIFROST:
                return std::make_unique<CLGEMMDefaultTypeBifrost>(gpu);
            case GPUTarget::VALHALL:
                return std::make_unique<CLGEMMDefaultTypeValhall>(gpu);
            default:
                ARM_COMPUTE_ERROR("Not supported GPU target");
        }
    }
};
} // namespace cl_gemm
} // namespace arm_compute
#endif /* SRC_CLGEMMKERNELSELECTION_H */
