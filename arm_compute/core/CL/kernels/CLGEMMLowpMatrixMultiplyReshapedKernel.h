/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYRESHAPEDKERNEL_H__
#define __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYRESHAPEDKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to multiply matrices when both the input matrices LHS (input0) and RHS (input1) have been reshaped
 *
 * @note The input matrices @p input0 and @p input1 must be reshaped through @ref CLGEMMReshapeLHSMatrixKernel and  @ref CLGEMMReshapeRHSMatrixKernel
 */
class CLGEMMLowpMatrixMultiplyReshapedKernel : public ICLKernel
{
public:
    /** Default Constructor */
    CLGEMMLowpMatrixMultiplyReshapedKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyReshapedKernel(const CLGEMMLowpMatrixMultiplyReshapedKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyReshapedKernel &operator=(const CLGEMMLowpMatrixMultiplyReshapedKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMLowpMatrixMultiplyReshapedKernel(CLGEMMLowpMatrixMultiplyReshapedKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMLowpMatrixMultiplyReshapedKernel &operator=(CLGEMMLowpMatrixMultiplyReshapedKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input0    Input tensor containing the LHS reshaped matrix. Data type supported: QASYMM8. The number of dimensions for the LHS matrix must be less or equal than 4.
     * @param[in]  input1    Input tensor containing the RHS reshaped matrix. Data type supported: same as @p input0. The number of dimensions for the RHS matrix must be less or equal than 3.
     * @param[out] output    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in]  lhs_info  LHS matrix information used for reshaping the input0 tensor.  Only the following values are supported:
     *                       lhs_info.m0: 2,3,4,5,6,7,8
     *                       lhs_info.k0: 2,3,4,8,16
     *                       lhs_info.transpose: false
     * @param[in]  rhs_info  RHS matrix information used for reshaping the input1 tensor.  Only the following values are supported:
     *                       rhs_info.n0: 2,3,4,8,16
     *                       rhs_info.k0: 2,3,4,8,16
     *                       rhs_info.transpose: true
     * @param[in]  gemm_info GEMM information used to retrieve the original dimensions of the input matrices
     *
     * @note lhs_info.k0 must be equal to rhs_info.k0
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info, const GEMMReshapeInfo &gemm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixMultiplyReshapedKernel
     *
     * @param[in] input0    Input tensor info containing the LHS reshaped matrix. Data type supported: QASYMM8. The number of dimensions for the LHS matrix must be less or equal than 4.
     * @param[in] input1    Input tensor info containing the RHS reshaped matrix. Data type supported: same as @p input0. The number of dimensions for the RHS matrix must be less or equal than 3.
     * @param[in] output    Output tensor info. Data type supported: same as @p input0
     * @param[in] lhs_info  LHS matrix information used for reshaping the input0 tensor.  Only the following values are supported:
     *                      lhs_info.m0: 2,3,4,5,6,7,8
     *                      lhs_info.k0: 2,3,4,8,16
     *                      lhs_info.transpose: false
     * @param[in] rhs_info  RHS matrix information used for reshaping the input1 tensor.  Only the following values are supported:
     *                      rhs_info.n0: 2,3,4,8,16
     *                      rhs_info.k0: 2,3,4,8,16
     *                      rhs_info.transpose: true
     * @param[in] gemm_info GEMM information used to retrieve the original dimensions of the input matrices
     *
     * @note lhs_info.k0 must be equal to rhs_info.k0
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, const GEMMLHSMatrixInfo &lhs_info, const GEMMRHSMatrixInfo &rhs_info,
                           const GEMMReshapeInfo &gemm_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    ICLTensor       *_output;
    bool             _slide_matrix_b;
    bool             _reinterpret_output_as_3d;
    unsigned int     _k;
    bool             _use_dummy_work_items;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYRESHAPEDKERNEL_H__*/
