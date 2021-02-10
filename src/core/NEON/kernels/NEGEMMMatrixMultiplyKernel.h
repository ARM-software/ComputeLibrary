/*
 * Copyright (c) 2017-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_NEGEMMMATRIXMULTIPLYKERNEL_H
#define ARM_COMPUTE_NEGEMMMATRIXMULTIPLYKERNEL_H

#include "src/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** Neon kernel to multiply two input matrices "A" and "B". All elements of the output matrix/vector will be multiplied by alpha after the matrix multiplication
 *
 * @note If the output tensor is a matrix, the implementation assumes that the input tensors @p input0 and @p input1 are both matrices and reshaped respectively with @ref NEGEMMInterleave4x4Kernel" and @ref NEGEMMTranspose1xWKernel
 * @note If the output tensor is a vector and the data type is F32, the implementation assumes that the first input tensor @p input0 is a vector and the second input tensor @p input1 a matrix. The implementation also assumes that both tensors have not been reshaped
 *
 */
class NEGEMMMatrixMultiplyKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEGEMMMatrixMultiplyKernel";
    }
    /** Constructor */
    NEGEMMMatrixMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMMatrixMultiplyKernel(const NEGEMMMatrixMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEGEMMMatrixMultiplyKernel &operator=(const NEGEMMMatrixMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixMultiplyKernel(NEGEMMMatrixMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixMultiplyKernel &operator=(NEGEMMMatrixMultiplyKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @note If the output tensor is a matrix, the input matrices @p input0 and @p input1 should be the output of the kernels: @ref NEGEMMInterleave4x4Kernel and @ref NEGEMMTranspose1xWKernel
     *       These two kernels change the layout of the original matrices to be more cache-friendly.
     *
     * @param[in]  input0         Input tensor containing the interleaved Matrix A or the vector A. Data types supported: F16/F32
     * @param[in]  input1         Input tensor containing the transposed Matrix B if the first input tensor A is not a vector.
     *                            If the output tensor is a vector, input1 must contain the matrix B not reshaped. Data type supported: same as @p input0
     * @param[out] output         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in]  alpha          Weight of the matrix product
     * @param[in]  is_interleaved (Optional) True if input0 and input1 have been reshaped respectively using @ref NEGEMMInterleave4x4Kernel and @ref NEGEMMTranspose1xWKernel
     * @param[in]  reshape_info   (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info = GEMMReshapeInfo());
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMMatrixMultiplyKernel
     *
     * @param[in] input0         Input tensor containing the interleaved Matrix A or the vector A. Data types supported: F16/F32
     * @param[in] input1         Input tensor containing the transposed Matrix B if the first input tensor A is not a vector.
     *                           If the output tensor is a vector, input1 must contain the matrix B not reshaped. Data type supported: same as @p input0
     * @param[in] output         Output tensor to store the result of matrix multiplication. Data type supported: same as @p input0.
     * @param[in] alpha          Weight of the matrix product
     * @param[in] is_interleaved (Optional) True if input0 and input1 have been reshaped respectively using @ref NEGEMMInterleave4x4Kernel and @ref NEGEMMTranspose1xWKernel
     * @param[in] reshape_info   (Optional) GEMM reshape info. If is_interleaved_transposed = true, this object must contain the information to understand how the matrix A and matrix B have been reshaped
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output, float alpha, bool is_interleaved, const GEMMReshapeInfo &reshape_info);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input0;
    const ITensor *_input1;
    ITensor       *_output;
    float          _alpha;
};
} // namespace arm_compute
#endif /*ARM_COMPUTE_NEGEMMMATRIXMULTIPLYKERNEL_H*/
