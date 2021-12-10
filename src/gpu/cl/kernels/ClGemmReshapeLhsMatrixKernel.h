/*
 * Copyright (c) 2018-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_GEMM_RESHAPE_LHS_MATRIX_KERNEL_H
#define ARM_COMPUTE_CL_GEMM_RESHAPE_LHS_MATRIX_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel to reshape the LHS matrix when performing the matrix multiplication.
 *  In particular, this function splits the src matrix in blocks of size M0xK0 (defined through GEMMLHSInfo) and
 *  stores each one in the dst matrix unrolling the values
 */
class ClGemmReshapeLhsMatrixKernel : public ICLKernel
{
public:
    ClGemmReshapeLhsMatrixKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmReshapeLhsMatrixKernel);
    /** Initialise the kernel's input and output.
     *
     * @param[in]  compile_context       The compile context to be used.
     * @param[in]  src                   Input tensor. Data types supported: All
     * @param[out] dst                   Output tensor. Data type supported: same as @p src
     * @param[in]  lhs_info              LHS matrix information to be used for reshaping. This object contains all the necessary
     *                                   information to reshape the src tensor. Only the following values are supported:
     *                                   lhs_info.m0: 2,3,4,5,6,7,8
     *                                   lhs_info.k0: 2,3,4,8,16
     *                                   lhs_info.v0: greater than 0
     *                                   lhs_info.transpose: true, false
     *                                   lhs_info.interleave: true, false
     * @param[in]  reinterpret_src_as_3d (Optional) True if the src has to be reinterpreted as 3D tensor
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src, ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_src_as_3d = false);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmReshapeLhsMatrixKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src, const ITensorInfo *dst, const GEMMLHSMatrixInfo &lhs_info, bool reinterpret_src_as_3d);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMM_RESHAPE_LHS_MATRIX_KERNEL_H */