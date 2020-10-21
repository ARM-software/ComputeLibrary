/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#ifndef ARM_COMPUTE_CLGEMMMATRIXMULTIPLYRESHAPEDONLYRHSKERNEL_H
#define ARM_COMPUTE_CLGEMMMATRIXMULTIPLYRESHAPEDONLYRHSKERNEL_H

#include "src/core/CL/ICLKernel.h"

#include "arm_compute/core/KernelDescriptors.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to multiply matrices when only the input matrix RHS (input1) has been reshaped
 *
 * @note The input matrix input1 must be reshaped through @ref CLGEMMReshapeRHSMatrixKernel
 */
class CLGEMMMatrixMultiplyReshapedOnlyRHSKernel : public ICLKernel
{
public:
    /** Default Constructor */
    CLGEMMMatrixMultiplyReshapedOnlyRHSKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixMultiplyReshapedOnlyRHSKernel(const CLGEMMMatrixMultiplyReshapedOnlyRHSKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixMultiplyReshapedOnlyRHSKernel &operator=(const CLGEMMMatrixMultiplyReshapedOnlyRHSKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixMultiplyReshapedOnlyRHSKernel(CLGEMMMatrixMultiplyReshapedOnlyRHSKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixMultiplyReshapedOnlyRHSKernel &operator=(CLGEMMMatrixMultiplyReshapedOnlyRHSKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @note If rhs_info.export_to_cl_image = true, this OpenCL kernel will fetch the RHS data using the OpenCL read_image built-in function.
     *       Reading from the OpenCL image object can increase the performance. However, since the OpenCL image object is created importing the OpenCL buffer,
     *       the following conditions are required:
     *       -# rhs_info.n0 can only be 4, 8 and 16
     *       -# rhs_info.k0 can only be 4, 8 and 16
     *       -# Data type can only be F32
     *       -# The platform should support the OpenCL cl_khr_image2d_from_buffer extension
     *       -# The stride Y for the input1 should satisfy the OpenCL pitch alignment requirement
     *       -# input1 width should be less or equal to (CL_DEVICE_IMAGE2D_MAX_WIDTH * 4)
     *       -# input1 (height * depth) should be less or equal to CL_DEVICE_IMAGE2D_MAX_HEIGHT
     *
     * @param[in]  input0    Input tensor containing the LHS matrix. Data type supported: F16/F32 (only F32 if rhs_info.export_to_cl_image = true).
     *                       The number of dimensions for the LHS matrix must be less or equal than 4.
     * @param[in]  input1    Input tensor containing the RHS reshaped matrix. Data type supported: same as @p input0. The number of dimensions for the RHS matrix must be less or equal than 3.
     * @param[in]  input2    Input tensor containing the bias matrix. Data type supported: same as @p input0.
     * @param[out] output    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in]  alpha     Weight of the matrix product
     * @param[in]  beta      Weight of the matrix bias
     * @param[in]  lhs_info  LHS matrix information used to retrieve the number of rows to be processed by each thread. Only the following values are supported:
     *                       lhs_info.m0: 1,2,3,4,5,6,7,8
     * @param[in]  rhs_info  RHS matrix information used for reshaping the input1 tensor.  Only the following values are supported:
     *                       rhs_info.k0: 2,3,4,8,16
     *                       rhs_info.n0: 2,3,4,8,16
     *                       rhs_info.transpose: true,false
     * @param[in]  gemm_info GEMM information used to retrieve the original dimensions of the input matrices
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                   const GEMMRHSMatrixInfo &rhs_info,
                   const GEMMKernelInfo    &gemm_info);
    /** Initialise the kernel's input and output.
     *
     * @note If rhs_info.export_to_cl_image = true, this OpenCL kernel will fetch the RHS data using the OpenCL read_image built-in function.
     *       Reading from the OpenCL image object can increase the performance. However, since the OpenCL image object is created importing the OpenCL buffer,
     *       the following conditions are required:
     *       -# rhs_info.n0 can only be 4, 8 and 16
     *       -# rhs_info.k0 can only be 4, 8 and 16
     *       -# Data type can only be F32
     *       -# The platform should support the OpenCL cl_khr_image2d_from_buffer extension
     *       -# The stride Y for the input1 should satisfy the OpenCL pitch alignment requirement
     *       -# input1 width should be less or equal to (CL_DEVICE_IMAGE2D_MAX_WIDTH * 4)
     *       -# input1 (height * depth) should be less or equal to CL_DEVICE_IMAGE2D_MAX_HEIGHT
     *
     * @param[in]  compile_context The compile context to be used.
     * @param[in]  input0          Input tensor containing the LHS matrix. Data type supported: F16/F32 (only F32 if rhs_info.export_to_cl_image = true).
     *                             The number of dimensions for the LHS matrix must be less or equal than 4.
     * @param[in]  input1          Input tensor containing the RHS reshaped matrix. Data type supported: same as @p input0. The number of dimensions for the RHS matrix must be less or equal than 3.
     * @param[in]  input2          Input tensor containing the bias matrix. Data type supported: same as @p input0.
     * @param[out] output          Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in]  alpha           Weight of the matrix product
     * @param[in]  beta            Weight of the matrix bias
     * @param[in]  lhs_info        LHS matrix information used to retrieve the number of rows to be processed by each thread. Only the following values are supported:
     *                             lhs_info.m0: 1,2,3,4,5,6,7,8
     * @param[in]  rhs_info        RHS matrix information used for reshaping the input1 tensor.  Only the following values are supported:
     *                             rhs_info.k0: 2,3,4,8,16
     *                             rhs_info.n0: 2,3,4,8,16
     *                             rhs_info.transpose: true,false
     * @param[in]  gemm_info       GEMM information used to retrieve the original dimensions of the input matrices
     */
    void configure(const CLCompileContext &compile_context, const ICLTensor *input0, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, float alpha, float beta,
                   const GEMMLHSMatrixInfo &lhs_info,
                   const GEMMRHSMatrixInfo &rhs_info,
                   const GEMMKernelInfo    &gemm_info);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMMatrixMultiplyReshapedOnlyRHSKernel
     *
     * @note If rhs_info.export_to_cl_image = true, this OpenCL kernel will fetch the RHS data using the OpenCL read_image built-in function.
     *       Reading from the OpenCL image object can increase the performance. However, since the OpenCL image object is created importing the OpenCL buffer,
     *       the following conditions are required:
     *       -# rhs_info.n0 can only be 4, 8 and 16
     *       -# rhs_info.k0 can only be 4, 8 and 16
     *       -# Data type can only be F32
     *       -# The platform should support the OpenCL cl_khr_image2d_from_buffer extension
     *       -# The stride Y for the input1 should satisfy the OpenCL pitch alignment requirement
     *       -# input1 width should be less or equal to (CL_DEVICE_IMAGE2D_MAX_WIDTH * 4)
     *       -# input1 (height * depth) should be less or equal to CL_DEVICE_IMAGE2D_MAX_HEIGHT
     *
     * @param[in] input0    Input tensor info for the LHS matrix. Data type supported: F16/F32 (only F32 if rhs_info.export_to_cl_image = true).
     *                      The number of dimensions for the LHS matrix must be less or equal than 4.
     * @param[in] input1    Input tensor info for the RHS reshaped matrix. Data type supported: same as @p input0. The number of dimensions for the RHS matrix must be less or equal than 3.
     * @param[in] input2    Input tensor info containing the bias matrix. Data type supported: same as @p input0.
     * @param[in] output    Output tensor info. Data type supported: same as @p input0
     * @param[in] alpha     Weight of the matrix product
     * @param[in] beta      Weight of the matrix bias
     * @param[in] lhs_info  LHS matrix information used to retrieve the number of rows to be processed by each thread. Only the following values are supported:
     *                      lhs_info.m0: 1,2,3,4,5,6,7,8
     * @param[in] rhs_info  RHS matrix information used for reshaping the input1 tensor.  Only the following values are supported:
     *                      rhs_info.k0: 2,3,4,8,16
     *                      rhs_info.n0: 2,3,4,8,16
     *                      rhs_info.transpose: true,false
     * @param[in] gemm_info GEMM information used to retrieve the original dimensions of the input matrices
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float alpha, float beta, const GEMMLHSMatrixInfo &lhs_info,
                           const GEMMRHSMatrixInfo &rhs_info,
                           const GEMMKernelInfo    &gemm_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    const ICLTensor *_input2;
    ICLTensor       *_output;
    bool             _slide_matrix_b;
    bool             _reinterpret_input_as_3d;
    bool             _reinterpret_output_as_3d;
    bool             _use_dummy_work_items;
    bool             _add_bias;
    bool             _broadcast_bias;
    bool             _export_to_cl_image;
    bool             _has_pad_y;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLGEMMMATRIXMULTIPLYRESHAPEDONLYRHSKERNEL_H*/
