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
#ifndef ACL_SRC_GPU_CL_KERNELS_CLMATMULLOWPNATIVEMMULKERNEL_H
#define ACL_SRC_GPU_CL_KERNELS_CLMATMULLOWPNATIVEMMULKERNEL_H

#include "arm_compute/function_info/ActivationLayerInfo.h"

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
// Forward declerations
struct MatMulKernelInfo;
namespace opencl
{
namespace kernels
{
class ClMatMulLowpNativeMMULKernel : public IClKernel
{
public:
    ClMatMulLowpNativeMMULKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClMatMulLowpNativeMMULKernel);

    /** Initialise the kernel's input and output.
     *
     * Similar to @ref ClMatMulLowpNativeKernel::configure()
     *
     */
    void configure(const ClCompileContext    &compile_context,
                   ITensorInfo               *lhs,
                   ITensorInfo               *rhs,
                   ITensorInfo               *bias,
                   ITensorInfo               *dst,
                   const MatMulKernelInfo    &matmul_kernel_info,
                   const ActivationLayerInfo &act_info = ActivationLayerInfo());
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClMatMulLowpNativeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo         *lhs,
                           const ITensorInfo         *rhs,
                           const ITensorInfo         *bias,
                           const ITensorInfo         *dst,
                           const MatMulKernelInfo    &matmul_kernel_info,
                           const ActivationLayerInfo &act_info = ActivationLayerInfo());

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif // ACL_SRC_GPU_CL_KERNELS_CLMATMULLOWPNATIVEMMULKERNEL_H
