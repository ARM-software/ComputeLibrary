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
#ifndef ARM_COMPUTE_ICLGEMMKERNELSELECTION_H
#define ARM_COMPUTE_ICLGEMMKERNELSELECTION_H

#include "arm_compute/core/GPUTarget.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLTypes.h"

namespace arm_compute
{
namespace cl_gemm
{
/** Basic interface for the GEMM kernel selection */
class ICLGEMMKernelSelection
{
public:
    /** Constructor
     *
     * @param[in] arch GPU target
     */
    ICLGEMMKernelSelection(GPUTarget arch)
        : _target(arch)
    {
    }
    /** Default Move Constructor. */
    ICLGEMMKernelSelection(ICLGEMMKernelSelection &&) = default;
    /** Default move assignment operator */
    ICLGEMMKernelSelection &operator=(ICLGEMMKernelSelection &&) = default;
    /** Virtual destructor */
    virtual ~ICLGEMMKernelSelection() = default;
    /** Given the input parameters passed through @ref CLGEMMKernelSelectionParams, this method returns the @ref CLGEMMKernelType to use
     *
     * @param[in] params Input parameters used by the function to return the OpenCL GEMM's kernel
     *
     * @return @ref CLGEMMKernelType
     */
    virtual CLGEMMKernelType select_kernel(const CLGEMMKernelSelectionParams &params) = 0;

protected:
    GPUTarget _target; /**< GPU target could be used to call a dedicated heuristic for each GPU IP for a given GPU architecture */
};
} // namespace cl_gemm
} // namespace arm_compute
#endif /*ARM_COMPUTE_ICLGEMMKERNELSELECTION_H */
