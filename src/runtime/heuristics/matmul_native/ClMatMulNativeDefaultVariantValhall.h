/*
 * Copyright (c) 2023 Arm Limited.
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
#ifndef ACL_SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_CLMATMULNATIVEDEFAULTVARIANTVALHALL_H
#define ACL_SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_CLMATMULNATIVEDEFAULTVARIANTVALHALL_H

#include "src/runtime/heuristics/matmul_native/IClMatMulNativeKernelVariant.h"

namespace arm_compute
{
namespace cl_matmul
{
/** Valhall based OpenCL matmul configuration */
class ClMatMulNativeDefaultVariantValhall final : public IClMatMulNativeKernelVariant
{
public:
    /** Constructor
     *
     * @param[in] gpu GPU target
     */
    ClMatMulNativeDefaultVariantValhall(GPUTarget gpu);

    // Inherited overridden method
    MatMulKernelType select_kernel(const ITensorInfo         *lhs,
                                   const ITensorInfo         *rhs,
                                   const MatMulInfo          &info,
                                   const ActivationLayerInfo &act_info) override;

private:
    MatMulKernelType configure_G715_float(int k, bool act_enabled);
    MatMulKernelType configure_G715_quantized(int k, bool act_enabled);
    MatMulKernelType configure_default_float(int k, bool act_enabled);
    MatMulKernelType configure_default_quantized(int k, bool act_enabled);
};
} // namespace cl_matmul
} // namespace arm_compute
#endif // ACL_SRC_RUNTIME_HEURISTICS_MATMUL_NATIVE_CLMATMULNATIVEDEFAULTVARIANTVALHALL_H
