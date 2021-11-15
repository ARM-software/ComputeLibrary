/*
 * Copyright (c) 2019-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_NATIVE_KERNEL_H
#define ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_NATIVE_KERNEL_H

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
/** OpenCL kernel to multiply matrices with QASYMM8/QASYMM8_SIGNED data type */
class ClGemmLowpMatrixMultiplyNativeKernel : public IClKernel
{
public:
    ClGemmLowpMatrixMultiplyNativeKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmLowpMatrixMultiplyNativeKernel);
    /** Initialise the kernel's input and dst.
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src0            Source tensor containing the LHS matrix. Data type supported: QASYMM8/QASYMM8_SIGNED
     * @param[in]  src1            Source tensor containing the RHS matrix. Data type supported: same as @p src0
     * @param[out] dst             Destination tensor to store the result of matrix multiplication. Data type supported: S32
     * @param[in]  lhs_info        LHS matrix information used to retrieve the number of rows to be processed by each thread
     *                             lhs_info.m0: 2,3,4,5,6,7,8
     *                             lhs_info.k0: 2,3,4,8,16
     * @param[in]  rhs_info        RHS matrix information used to retrieve the number of columns to be processed by each thread
     *                             rhs_info.n0: 2,3,4,8,16
     *                             rhs_info.k0: same as lhs_info.k0
     * @param[in]  gemm_info       GEMM information used to retrieve the original dimensions of the input matrices
     */
    void configure(const CLCompileContext &compile_context, const ITensorInfo *src0, ITensorInfo *src1, ITensorInfo *dst,
                   const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const GEMMReshapeInfo &gemm_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmLowpMatrixMultiplyNativeKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst,
                           const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const GEMMReshapeInfo &gemm_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    bool _slide_matrix_b{ true };
    bool _reinterpret_input_as_3d{ false };
    bool _reinterpret_output_as_3d{ false };
    bool _use_dummy_work_items{ false };
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMMLOWP_MATRIXMULTIPLY_NATIVE_KERNEL_H */
