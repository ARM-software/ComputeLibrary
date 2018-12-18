/*
 * Copyright (c) 2017-2018 ARM Limited.
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
#ifndef __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYKERNEL_H__
#define __ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to multiply matrices
 *
 * @note @ref CLGEMMLowpMatrixMultiplyKernel low precision matrix product kernel
 *  This kernel performs the following computation:
 *
 *  -# Convert a values from int8 to int32
 *  -# Convert b values from int8 to int32
 *  -# Compute the int32 matrix product of the resulting a * b and store the result as int32
 *
 */
class CLGEMMLowpMatrixMultiplyKernel : public ICLKernel
{
public:
    /** Default Constructor */
    CLGEMMLowpMatrixMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyKernel(const CLGEMMLowpMatrixMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMLowpMatrixMultiplyKernel &operator=(const CLGEMMLowpMatrixMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMLowpMatrixMultiplyKernel(CLGEMMLowpMatrixMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMLowpMatrixMultiplyKernel &operator=(CLGEMMLowpMatrixMultiplyKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @param[in]  input0                    Input tensor containing the interleaved Matrix A. Data type supported: QASYMM8
     * @param[in]  input1                    Input tensor containing the transposed1xW Matrix B. Data type supported: same as @p input0
     * @param[out] output                    Output tensor to store the result of matrix multiplication. Data type supported: S32
     * @param[in]  is_interleaved_transposed (Optional) True if input0 and input1 have been reshaped respectively using @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMReshapeRHSMatrixKernel
     * @param[in]  reshape_info              (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, bool is_interleaved_transposed = true, const GEMMReshapeInfo &reshape_info = GEMMReshapeInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMLowpMatrixMultiplyKernel
     *
     * @param[in] input0                    Input tensor info containing the interleaved Matrix A. Data type supported: QASYMM8
     * @param[in] input1                    Input tensor info containing the transposed Matrix B. Data type supported: same as @p input0
     * @param[in] output                    Output tensor info to store the result of matrix multiplication. Data type supported: S32
     * @param[in] is_interleaved_transposed True if input0 and input1 have been reshaped respectively using @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMReshapeRHSMatrixKernel
     * @param[in] reshape_info              GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, bool is_interleaved_transposed, const GEMMReshapeInfo &reshape_info);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    ICLTensor       *_output;
    bool             _slide_matrix_b;
    bool             _reinterpret_input_as_3d;
    bool             _reinterpret_output_as_3d;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_CLGEMMLOWPMATRIXMULTIPLYKERNEL_H__*/
