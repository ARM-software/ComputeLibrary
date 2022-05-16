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
#ifndef ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_RESHAPED_ONLY_RHS_MMUL_KERNEL_H
#define ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_RESHAPED_ONLY_RHS_MMUL_KERNEL_H

#include "arm_compute/core/KernelDescriptors.h"
#include "src/core/common/Macros.h"
#include "src/gpu/cl/IClKernel.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel to multiply matrices with QASYMM8/QASYMM8_SIGNED data types when only the input matrix RHS (src1) has been reshaped using the MMUL instruction
 *
 * @note The input matrix src1 must be reshaped through @ref opencl::kernels::ClGemmReshapeRhsMatrixKernel
 * @note For fused output stage, only GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT type is supported
 */
class ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel : public IClKernel
{
public:
    ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel);
    /** Initialise the kernel's source and destination.
     *
     * @param[in]  compile_context    The compile context to be used.
     * @param[in]  src0               Input tensor containing the LHS matrix. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  src1               Input tensor containing the RHS reshaped matrix. Data type supported: same as @p src0
     * @param[out] dst                Destination tensor. Data type supported: QASYMM8/QASYMM8_SIGNED/S32.
     * @param[in]  gemm_info          GEMM information used to retrieve the original dimensions of the input matrices, output stage information and RHS/LHS info.
     *                                lhs_info.m0: 1,2,4
     *                                Only the following values are supported for RHS info:
     *                                rhs_info.n0: 1,4,8
     *                                rhs_info.k0: same as lhs_info.k0: 4
     *                                rhs_info.transpose: true
     * @param[in]  vector_sum_col     (Optional) Input row-vector of sums of all the entries in each column of matrix B.
     *                                Note: vector_sum_col can be a nullptr in case a_offset = 0. Data type supported: S32
     * @param[in]  vector_sum_row     (Optional) Input row-vector of sums of all the entries in each row of matrix A.
     *                                Note: vector_sum_row can be a nullptr in case b_offset = 0. Data type supported: S32
     * @param[in]  bias               (Optional) Biases tensor. Can be a nullptr if the addition of biases is not required.
     *                                Biases are 1D tensor with dimensions [OFM] or same dimensionality as dst if gemm_info.broadcast_bias is false. Data type supported: S32.
     * @param[in]  output_multipliers (Optional) Output multipliers tensor. Supported data types: S32.
     * @param[in]  output_shifts      (Optional) Output shifts tensor. Supported data types: S32.
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, const GEMMKernelInfo &gemm_info,
                   ITensorInfo *vector_sum_col = nullptr, const ITensorInfo *vector_sum_row = nullptr, ITensorInfo *bias = nullptr,
                   ITensorInfo *output_multipliers = nullptr, ITensorInfo *output_shifts = nullptr);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmLowpMatrixMultiplyReshapedOnlyRhsMMULKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, const GEMMKernelInfo &gemm_info,
                           const ITensorInfo *vector_sum_col = nullptr, const ITensorInfo *vector_sum_row = nullptr, const ITensorInfo *bias = nullptr,
                           const ITensorInfo *output_multipliers = nullptr, const ITensorInfo *output_shifts = nullptr);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    bool       _fuse_output_stage{ false };
    signed int _m{ 1 };
    signed int _n{ 1 };
    signed int _k{ 1 };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_RESHAPED_ONLY_RHS_MMULKERNEL_H */