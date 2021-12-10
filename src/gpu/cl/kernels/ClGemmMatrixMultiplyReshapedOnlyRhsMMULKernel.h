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
#ifndef ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_RESHAPED_ONLY_RHS_MMUL_KERNEL_H
#define ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_RESHAPED_ONLY_RHS_MMUL_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel to multiply matrices using MMUL when only the input matrix RHS (src1) has been reshaped */
class ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel : public IClKernel
{
public:
    ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel);
    /** Initialize the kernel's input and dst.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src0            Input tensor for the LHS matrix. Data type supported: F16/F32.
     * @param[in]  src1            Input tensor containing the RHS reshaped matrix. Data type supported: same as @p src0.
     * @param[in]  src2            Input tensor containing the bias matrix. Data type supported: same as @p src0.
     * @param[out] dst             dst tensor info. Data type supported: same as @p src0
     * @param[in]  alpha           Weight of the matrix product
     * @param[in]  beta            Weight of the matrix bias
     * @param[in]  lhs_info        LHS matrix information used to retrieve the number of rows and accumulations to be processed by each thread. Only the following values are supported:
     *                             lhs_info.m0 > 0
     *                             lhs_info.k0: 1
     * @param[in]  rhs_info        RHS matrix information used to retrieve the number of columns and accumulations to be processed by each thread. Only the following values are supported:
     *                             rhs_info.n0: 1,2,3,4,8,16
     *                             rhs_info.k0: same of lhs_info.k0
     *                             rhs_info.transpose: false
     * @param[in]  gemm_info       GEMM information used to retrieve the original dimensions of the input matrices
     */
    void configure(const ClCompileContext &compile_context, ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *src2, ITensorInfo *dst, float alpha, float beta,
                   const GEMMLHSMatrixInfo &lhs_info,
                   const GEMMRHSMatrixInfo &rhs_info,
                   const GEMMKernelInfo    &gemm_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmMatrixMultiplyReshapedOnlyRhsMMULKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                           const GEMMRHSMatrixInfo &rhs_info,
                           const GEMMKernelInfo    &gemm_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    bool       _add_bias{ false };
    bool       _export_to_cl_image{ false };
    signed int _m{ 1 };
    signed int _n{ 1 };
    signed int _k{ 1 };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_RESHAPED_ONLY_RHS_MMUL_KERNEL_H */
