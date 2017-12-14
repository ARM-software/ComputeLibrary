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
#ifndef __ARM_COMPUTE_GCGEMMMATRIXADDITIONKERNEL_H__
#define __ARM_COMPUTE_GCGEMMMATRIXADDITIONKERNEL_H__

#include "arm_compute/core/GLES_COMPUTE/IGCKernel.h"

namespace arm_compute
{
class IGCTensor;

/** OpenGL ES kernel to perform the in-place matrix addition between 2 matrices, taking into account that the second matrix might be weighted by a scalar value beta.
 *  The matrices must have the same dimensions
 *
 * @note This kernel is computed if and only if beta != 0.0.
 */
class GCGEMMMatrixAdditionKernel : public IGCKernel
{
public:
    /** Default constructor */
    GCGEMMMatrixAdditionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMMatrixAdditionKernel(const GCGEMMMatrixAdditionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    GCGEMMMatrixAdditionKernel &operator=(const GCGEMMMatrixAdditionKernel &) = delete;
    /** Allow instances of this class to be moved */
    GCGEMMMatrixAdditionKernel(GCGEMMMatrixAdditionKernel &&) = default;
    /** Allow instances of this class to be moved */
    GCGEMMMatrixAdditionKernel &operator=(GCGEMMMatrixAdditionKernel &&) = default;
    /** Initialise the kernel's input, output and beta value
     *
     * @note The input and output tensors must have the same dimensions
     *
     * @param[in]      input  Input tensor (Matrix C). Data types supported: F32
     * @param[in, out] output Output tensor. If this kernel is used to finalize the GEMM result (alpha * AB + beta * C), output must contain the result obtained by @ref GCGEMMMatrixMultiplyKernel. Data type supported: same as @p input
     * @param[in]      beta   Weight of matrix C
     */
    void configure(const IGCTensor *input, IGCTensor *output, float beta);

    // Inherited methods overridden:
    void run(const Window &window) override;

private:
    const IGCTensor *_input;
    IGCTensor       *_output;
};
}

#endif /* __ARM_COMPUTE_GCGEMMMATRIXADDITIONKERNEL_H__ */
