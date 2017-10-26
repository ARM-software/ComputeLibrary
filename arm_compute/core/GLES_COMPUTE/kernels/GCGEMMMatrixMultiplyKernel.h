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
#ifndef __ARM_COMPUTE_GCGEMMMATRIXMULTIPLYKERNEL_H__
#define __ARM_COMPUTE_GCGEMMMATRIXMULTIPLYKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** GLES Compute kernel to multiply two input matrices "A" and "B" or to multiply a vector "A" by a matrix "B". All elements of the output matrix/vector will be multiplied by alpha
 *
 * @note If the output tensor is a matrix, the implementation assumes that the input tensors @p input0 and @p input1 are both matrices and reshaped respectively with @ref GCGEMMInterleave4x4Kernel" and @ref GCGEMMTranspose1xWKernel
 * @note If the output tensor is a vector and the data type is F32, the implementation assumes that the first input tensor @p input0 is a vector and the second input tensor @p input1 a matrix. The implementation also assumes that both tensors have not been reshaped
 *
 * @attention The second input tensor must have at least 2 dimensions (matrix)
 *
 */
class GCGEMMMatrixMultiplyKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCGEMMMatrixMultiplyKernel();

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMMatrixMultiplyKernel(const GCGEMMMatrixMultiplyKernel &) = delete;

    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMMatrixMultiplyKernel &operator=(const GCGEMMMatrixMultiplyKernel &) = delete;

    /** Allow instances of this class to be moved */
    GCGEMMMatrixMultiplyKernel(GCGEMMMatrixMultiplyKernel &&) = default;

    /** Allow instances of this class to be moved */
    GCGEMMMatrixMultiplyKernel &operator=(GCGEMMMatrixMultiplyKernel &&) = default;

    /** Initialise the kernel's input, output and alpha
     *
     * @param[in]  input0                    Input tensor containing the interleaved Matrix A or the vector A. Data types supported: F16/F32
     * @param[in]  input1                    Input tensor containing the transposed Matrix B if the first input tensor A is not a vector.
     *                                       If the output tensor is a vector, input1 must contain the matrix B not reshaped. Data type supported: same as @p input0
     * @param[out] output                    Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0
     * @param[in]  alpha                     Weight of the matrix product
     * @param[in]  is_interleaved_transposed (Optional) True if input0 and input1 have been reshaped respectively using @ref GCGEMMInterleave4x4Kernel and @ref GCGEMMTranspose1xWKernel
     */
    void configure(const IGCTensor *input0, const IGCTensor *input1, IGCTensor *output, float alpha, bool is_interleaved_transposed = true);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input0;
    const IGCTensor *_input1;
    IGCTensor       *_output;
};
}
#endif /* __ARM_COMPUTE_GCGEMMMATRIXMULTIPLYKERNEL_H__ */
