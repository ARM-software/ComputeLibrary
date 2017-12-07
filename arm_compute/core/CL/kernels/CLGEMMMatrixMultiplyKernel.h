/*
 * Copyright (c) 2017 ARM Limited.
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

/** OpenCL kernel to multiply two input matrices "A" and "B" . All elements of the output matrix will be multiplied by alpha
 *
 * @note If the input tensors @p input0 and @p input1 have been reshaped respectively with @ref CLGEMMInterleave4x4Kernel" and @ref CLGEMMTranspose1xWKernel,
 *       the flag @p is_interleaved_transposed must be set to true
 *
 * @attention The second input tensor must have at least 2 dimensions (matrix)
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
     * @param[in]  input0                    Input tensor containing the Matrix A. Data types supported: QS8/QS16/F16/F32
     * @param[in]  input1                    Input tensor containing the Matrix B. Data type supported: same as @p input0
     * @param[out] output                    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in]  alpha                     Weight of the matrix product
     * @param[in]  is_interleaved_transposed (Optional) True if input0 and input1 have been reshaped respectively using @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel
     */
    void configure(const ICLTensor *input0, const ICLTensor *input1, ICLTensor *output, float alpha, bool is_interleaved_transposed = true);
    /** Static function to check if given info will lead to a valid configuration of @ref CLGEMMMatrixMultiplyKernel
     *
     * @param[in] input0                    Input tensor containing the Matrix A. Data types supported: QS8/QS16/F16/F32
     * @param[in] input1                    Input tensor containing the Matrix B. Data type supported: same as @p input0
     * @param[in] output                    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in] alpha                     Weight of the matrix product
     * @param[in] is_interleaved_transposed True if input0 and input1 have been reshaped respectively using @ref CLGEMMInterleave4x4Kernel and @ref CLGEMMTranspose1xWKernel
     * @param[in] gpu_target                GPU Target
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved_transposed, GPUTarget gpu_target);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input0;
    const ICLTensor *_input1;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMMATRIXMULTIPLYKERNEL_H__ */
