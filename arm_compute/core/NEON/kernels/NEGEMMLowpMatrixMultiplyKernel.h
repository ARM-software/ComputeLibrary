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
#ifndef __ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYKERNEL_H__
#define __ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYKERNEL_H__

#include "arm_compute/core/NEON/INEKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to multiply matrices
 *
 * @note @ref NEGEMMLowpMatrixMultiplyKernel low precision matrix product kernel
 *  This kernel performs the following computation:
 *
 *  -# Convert a values from int8 to int32
 *  -# Convert b values from int8 to int32
 *  -# Compute the int32 matrix product of the resulting a * b and store the result as int32
 *
 */
class NEGEMMLowpMatrixMultiplyKernel : public INEKernel
{
public:
    /** Constructor */
    NEGEMMLowpMatrixMultiplyKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpMatrixMultiplyKernel(const NEGEMMLowpMatrixMultiplyKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers)*/
    NEGEMMLowpMatrixMultiplyKernel &operator=(const NEGEMMLowpMatrixMultiplyKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMLowpMatrixMultiplyKernel(NEGEMMLowpMatrixMultiplyKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMLowpMatrixMultiplyKernel &operator=(NEGEMMLowpMatrixMultiplyKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * The input matrices @p input0 and @p input1 must be the output of the kernels: @ref NEGEMMInterleave4x4Kernel and @ref NEGEMMTranspose1xWKernel. These two
     * kernels change the layout of the original matrices to be more cache-friendly.
     *
     * @param[in]  input0 Input tensor containing the interleaved Matrix A. Data type supported: QASYMM8
     * @param[in]  input1 Input tensor containing the transposed1xW Matrix B. Data type supported: same as @p input0
     * @param[out] output Output tensor to store the result of matrix multiplication. Data type supported: S32
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output);
    /** Static function to check if given info will lead to a valid configuration of @ref NEGEMMLowpMatrixMultiplyKernel
     *
     * @param[in] input0 Input tensor info containing the interleaved Matrix A. Data type supported: QASYMM8
     * @param[in] input1 Input tensor info containing the transposed Matrix B. Data type supported: same as @p input0
     * @param[in] output Output tensor info to store the result of matrix multiplication. Data type supported: S32
     *
     * @return a status
     */
    static Status validate(const ITensorInfo *input0, const ITensorInfo *input1, const ITensorInfo *output);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input0;
    const ITensor *_input1;
    ITensor       *_output;
    bool           _slide_matrix_b;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYKERNEL_H__*/
