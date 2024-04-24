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
#ifndef SRC_CLGEMMDEFAULTTYPEBIFROST_H
#define SRC_CLGEMMDEFAULTTYPEBIFROST_H

#include "arm_compute/runtime/CL/ICLGEMMKernelSelection.h"

namespace arm_compute
{
namespace cl_gemm
{
/** Bifrost based OpenCL GEMMKernel selection */
class CLGEMMDefaultTypeBifrost final : public ICLGEMMKernelSelection
{
public:
    /** Constructor
     *
     * @param[in] gpu GPU target
     */
    CLGEMMDefaultTypeBifrost(GPUTarget gpu);

    // Inherited overridden method
    CLGEMMKernelType select_kernel(const CLGEMMKernelSelectionParams &params) override;

private:
    CLGEMMKernelType g52_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType g76_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType g76_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType g52_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType g71_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType default_f32(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType default_f16(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
    CLGEMMKernelType default_q8(unsigned int m, unsigned int n, unsigned int k, unsigned int b, bool is_rhs_constant);
};
} // namespace cl_gemm
} // namespace arm_compute
#endif /* SRC_CLGEMMDEFAULTTYPEBIFROST_H */
