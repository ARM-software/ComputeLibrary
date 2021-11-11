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
#ifndef ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_RESHAPED_KERNEL_H
#define ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_RESHAPED_KERNEL_H

#include "src/core/common/Macros.h"
#include "src/gpu/cl/ClCompileContext.h"
#include "src/gpu/cl/IClKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
/** OpenCL kernel to multiply matrices when both the input matrices LHS (src0) and RHS (src1) have been reshaped
 *
 * @note The input matrices @p src0 and @p src1 must be reshaped through:
 *  - @ref ClGemmReshapeLhsMatrixKernel
 *  - @ref ClGemmReshapeRhsMatrixKernel
 */
class ClGemmMatrixMultiplyReshapedKernel : public IClKernel
{
public:
    ClGemmMatrixMultiplyReshapedKernel();
    ARM_COMPUTE_DISALLOW_COPY_ALLOW_MOVE(ClGemmMatrixMultiplyReshapedKernel);
    /** Initialise the kernel's input and output.
     *
     * @note The F16 computation also supports mixed precision through the gemm_info.fp_mixed_precision flag.
     *       Mixed precision combines different floating precisions during the computation, in particular, F32 for the accumulations and F16 for the
     *       multiplications. i.e. float c = (half)a * (half)b
     *
     * @note If rhs_info.export_to_cl_image = true, this OpenCL kernel will fetch the RHS data using the OpenCL read_image built-in function.
     *       Reading from the OpenCL image object can increase the performance. However, since the OpenCL image object is created importing the OpenCL buffer,
     *       the following conditions are required:
     *       -# rhs_info.n0 can only be 4, 8 and 16
     *       -# rhs_info.k0 can only be 4, 8 and 16
     *       -# Data type can only be F32
     *       -# The platform should support the OpenCL cl_khr_image2d_from_buffer extension
     *       -# The stride Y for the src1 should satisfy the OpenCL pitch alignment requirement
     *       -# src1 width should be less or equal to (CL_DEVICE_IMAGE2D_MAX_WIDTH * 4)
     *       -# src1 (height * depth) should be less or equal to CL_DEVICE_IMAGE2D_MAX_HEIGHT
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  src0            Input tensor containing the LHS reshaped matrix. Data type supported: F16/F32  (only F32 if rhs_info.export_to_cl_image = true). The number of dimensions for the LHS matrix must be less or equal than 4
     * @param[in]  src1            Input tensor containing the RHS reshaped matrix. Data type supported: same as @p src0. The number of dimensions for the RHS matrix must be less or equal than 3
     * @param[in]  src2            Input tensor containing the bias matrix. Data type supported: same as @p src0.
     * @param[out] dst             dst tensor to store the result of matrix multiplication. Data type supported: same as @p src0
     * @param[in]  alpha           Weight of the matrix product
     * @param[in]  beta            Weight of the matrix bias
     * @param[in]  lhs_info        LHS matrix information used for reshaping the src0 tensor.  Only the following values are supported:
     *                             lhs_info.m0: 2,3,4,5,6,7,8
     *                             lhs_info.k0: 2,3,4,8,16
     *                             lhs_info.transpose: false
     * @param[in]  rhs_info        RHS matrix information used for reshaping the src1 tensor.  Only the following values are supported:
     *                             rhs_info.n0: 2,3,4,8,16 (only 4, 8 and 16 if rhs_info.export_to_cl_image = true)
     *                             rhs_info.k0: 2,3,4,8,16 (only 4, 8 and 16 if rhs_info.export_to_cl_image = true)
     *                             rhs_info.transpose: true
     * @param[in]  gemm_info       GEMM information used to retrieve the original dimensions of the input matrices
     *
     * @note lhs_info.k0 must be equal to rhs_info.k0
     */
    void configure(const ClCompileContext &compile_context,
                   const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, ITensorInfo *dst, float alpha, float beta,
                   const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const GEMMKernelInfo &gemm_info);
    /** Static function to check if given info will lead to a valid configuration
     *
     * Similar to @ref ClGemmMatrixMultiplyReshapedKernel::configure()
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *src2, const ITensorInfo *dst, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                           const GEMMRHSMatrixInfo &rhs_info,
                           const GEMMKernelInfo    &gemm_info);

    // Inherited methods overridden:
    void run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue) override;

private:
    bool         _slide_matrix_b{ true };
    bool         _reinterpret_output_as_3d{ false };
    bool         _use_dummy_work_items{ false };
    bool         _add_bias{ false };
    bool         _export_to_cl_image{ false };
    signed int   _m{ 1 };
    signed int   _n{ 1 };
    signed int   _k{ 1 };
    unsigned int _num_post_op_args{ 0 }; // (EXPERIMENTAL_POST_OPS) total number of post op arguments
};
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
#endif /* ARM_COMPUTE_CL_GEMM_MATRIXMULTIPLY_RESHAPED_KERNEL_H */