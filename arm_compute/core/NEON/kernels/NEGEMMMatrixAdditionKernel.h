/*
 * Copyright (c) 2016, 2017 ARM Limited.
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
#ifndef __ARM_COMPUTE_NEGEMMMATRIXADDITIONKERNEL_H__
#define __ARM_COMPUTE_NEGEMMMATRIXADDITIONKERNEL_H__

#include "arm_compute/core/NEON/INESimpleKernel.h"

namespace arm_compute
{
class ITensor;

/** NEON kernel to perform the in-place matrix addition between 2 matrices taking into account that the second matrix might be weighted by a scalar value beta:
 *
 * @note [ MTX_OUT = MTX_0 + beta * MTX_1 ] with MTX_0 and MTX_1 of the same size
 *
 * @note This stage is used to finalize the GEMM result and it is computed if and only if beta != 0.0. In case this kernel is used for finalizing GEMM result, we have:
 *        - MTX_0 = A * B * alpha, where MTX_0 is the output of @ref NEGEMMMatrixMultiplyKernel
 *        - MTX_1 = C
 */
class NEGEMMMatrixAdditionKernel : public INESimpleKernel
{
public:
    /** Constructor */
    NEGEMMMatrixAdditionKernel();
    /** Prevent instances of this class from being copied */
    NEGEMMMatrixAdditionKernel(const NEGEMMMatrixAdditionKernel &) = delete;
    /** Prevent instances of this class from being copied */
    NEGEMMMatrixAdditionKernel &operator=(const NEGEMMMatrixAdditionKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixAdditionKernel(NEGEMMMatrixAdditionKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEGEMMMatrixAdditionKernel &operator=(NEGEMMMatrixAdditionKernel &&) = default;
    /** Initialise the kernel's input and output.
     *
     * @note The input and output tensor must have the same dimensions
     *
     * @param[in]      input  Input tensor (Matrix C). Data types supported: QS8/QS16/F16/F32
     * @param[in, out] output Output tensor. If this kernel is used to finalize the GEMM result, output contains the result obtained by the kernel @ref NEGEMMMatrixMultiplyKernel. Data type supported: the same as @p input.
     * @param[in]      beta   Weight of matrix C
     */
    void configure(const ITensor *input, ITensor *output, float beta);

    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;

private:
    /** Common signature for all the matrix addition functions
     *
     * @param[in]  input  An input tensor. Data types supported: QS8/QS16/F16/F32
     * @param[out] output The output tensor. Data type supported: same as @p input
     * @param[in]  window Region on which to execute the kernel.
     * @param[in]  beta   Weight of matrix C
     */
    using MatrixAdditionFunction = void(const ITensor *input, ITensor *output, const Window &window, float beta);
    /** Matrix addition function to use for the particular tensor types passed to configure() */
    MatrixAdditionFunction *_func;
    float                   _beta;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_NEGEMMMATRIXADDITIONKERNEL_H__ */
