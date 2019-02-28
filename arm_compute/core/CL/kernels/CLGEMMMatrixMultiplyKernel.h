/*
 * Copyright (c) 2017-2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMMATRIXMULTIPLYKERNEL_H__
#define __ARM_COMPUTE_CLGEMMMATRIXMULTIPLYKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to multiply two input matrices "A" and "B" and add a vector "C" if provided. All elements of the output matrix will be multiplied by alpha. In case vector C is passed, it will be added to the previous result (a broadcast addition will be performed).
 *
 * @note If the input tensors @p input0 and @p input1 have been reshaped respectively with @ref CLGEMMReshapeLHSMatrixKernel" and @ref CLGEMMReshapeRHSMatrixKernel,
 *       the flag @p is_interleaved_transposed must be set to true
 *
 * @attention Vector C (@p input2) must be 1D. A broadcast addition is performed.
 *
 * @attention @p input1 tensor must have at least 2 dimensions (matrix)
 *
 */
class CLGEMMMatrixMultiplyKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLGEMMMatrixMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixMultiplyKernel(const CLGEMMMatrixMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixMultiplyKernel &operator=(const CLGEMMMatrixMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixMultiplyKernel(CLGEMMMatrixMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixMultiplyKernel &operator=(CLGEMMMatrixMultiplyKernel &&) = default;
    /** Initialise the kernel's input, output and alpha
     *
     * @param[in]  input0                    Input tensor containing the Matrix A. Data types supported: F16/F32
     * @param[in]  input1                    Input tensor containing the Matrix B. Data type supported: same as @p input0
     * @param[in]  input2                    Input tensor containing the Vector C. Can be nullptr. Data type supported: same as @p input0
     * @param[out] output                    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in]  alpha                     Weight of the matrix product
     * @param[in]  beta                      (Optional) Weight of vector C. Default value is 0. Only beta = 1 is currently supported.
     * @param[in]  is_interleaved_transposed (Optional) True if input0 and input1 have been reshaped respectively using @ref CLGEMMReshapeLHSMatrixKernel and @ref CLGEMMReshapeRHSMatrixKernel
     * @param[in]  reshape_info              (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     * @param[in]  fp_mixed_precision        (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy
     *
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, const ICLTensor *input2, ICLTensor *output, float alpha, float beta = 0.f,
                   bool is_interleaved_transposed = true, const GEMMReshapeInfo &reshape_info = GEMMReshapeInfo(), bool fp_mixed_precision = false);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMMatrixMultiplyKernel
     *
     * @param[in] input0                    Input tensor containing the Matrix A info. Data types supported: F16/F32
     * @param[in] input1                    Input tensor containing the Matrix B info. Data type supported: same as @p input0
     * @param[in] input2                    Input tensor containing the Vector C info. Can be nullptr. Data type supported: same as @p input0
     * @param[in] output                    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in] alpha                     Weight of the matrix product
     * @param[in] beta                      Weight of vector C. Default value is 0. Only beta = 1 is currently supported.
     * @param[in] is_interleaved_transposed True if input0 and input1 have been reshaped respectively using @ref CLGEMMReshapeLHSMatrixKernel and @ref CLGEMMReshapeRHSMatrixKernel
     * @param[in] reshape_info              GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     * @param[in] gpu_target                GPU Target
     * @param[in] fp_mixed_precision        (Optional) Use wider accumulators (32 bit instead of 16 for FP16) to improve accuracy
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *input2, const ITensorInfo *output, float alpha, float beta,
                           bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info, GPUTarget gpu_target, bool fp_mixed_precision = false);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

public:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    const ICLTensor *_input2;
    ICLTensor       *_output;
    bool             _slide_matrix_b;
    bool             _reinterpret_input_as_3d;
    bool             _reinterpret_output_as_3d;
    bool             _has_vec_c;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMMATRIXMULTIPLYKERNEL_H__ */
