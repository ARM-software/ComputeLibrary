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
 *  -# Convert a values from uint8 to int32 and add a_offset to each of them.
 *  -# Convert b values from uint8 to int32 and add b_offset to each of them.
 *  -# Compute the int32 matrix product of the resulting a * b.
 *  -# Add output_offset to each entry of the result.
 *  -# Multiply each entry of the result and round to the nearest integer
 *  -# Clamp the resulting int32 values to the [0..255] range and cast to uint8.
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
     * @param[in]  input0          Input tensor containing the interleaved Matrix A. Data type supported: U8
     * @param[in]  input1          Input tensor containing the transposed Matrix B. Data type supported: same as @p input0
     * @param[out] output          Output tensor to store the result of matrix multiplication, Data type supported: same as @p input0
     * @param[in]  a_offset        Offset to be added to each element of the matrix A.
     * @param[in]  b_offset        Offset to be added to each element of the matrix B.
     * @param[in]  output_offset   Offset to be added to each element of the output matrix
     * @param[in]  output_mult_int Value to be multipied to each entry of the result.
     * @param[in]  shift           Number of bits to shift right the result.
     */
    void configure(const ITensor *input0, const ITensor *input1, ITensor *output, int32_t a_offset, int32_t b_offset, int32_t output_offset, int32_t output_mult_int, int32_t shift);
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    const ITensor *_input0;
    const ITensor *_input1;
    ITensor       *_output;
    int32_t        _a_offset;
    int32_t        _b_offset;
    int32_t        _output_offset;
    int32_t        _output_mult_int;
    int32_t        _shift;
};
} // namespace arm_compute
#endif /*__ARM_COMPUTE_NEGEMMLOWPMATRIXMULTIPLYKERNEL_H__*/