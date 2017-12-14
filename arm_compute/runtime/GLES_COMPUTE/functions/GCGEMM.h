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

#ifndef __ARM_COMPUTE_GCGEMM_H__
#define __ARM_COMPUTE_GCGEMM_H__

#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMInterleave4x4Kernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixAdditionKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMMatrixMultiplyKernel.h"
#include "arm_compute/core/GLES_COMPUTE/kernels/GCGEMMTranspose1xWKernel.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#include "arm_compute/runtime/IFunction.h"

namespace arm_compute
{
class IGCTensor;

/** Basic function to execute GEMM on OpenGLES Compute. This function calls the following kernels:
 *
 *  -# @ref GCGEMMInterleave4x4Kernel (if the output tensor is a matrix)
 *  -# @ref GCGEMMTranspose1xWKernel (if the output tensor is a matrix)
 *  -# @ref GCGEMMMatrixMultiplyKernel
 *  -# @ref GCGEMMMatrixAdditionKernel (if c != nullptr and beta != 0.0)
 *
 */
class GCGEMM : public IFunction
{
public:
    /** Default constructor. */
    GCGEMM();

    /** Initialise the kernel's inputs and output
     *
     * @note GEMM: General Matrix Multiply - [alpha * A * B + beta * C].
     *
     * @note All tensors must have the same data type.
     *
     * @note Whilst the first input tensor can be a vector, the second input tensor must be at least a matrix
     *
     * @param[in]  a      First input tensor  (Matrix or Vector A). Data types supported: F32
     * @param[in]  b      Second input tensor (Matrix B). Data type supported: same as @p a.
     * @param[in]  c      Third input tensor  (Matrix C). It can be a nullptr if just the multiplication between @p a and @p b is needed. Data type supported: same as @p a.
     * @param[out] output Output tensor. Data type supported: same as @p a
     * @param[in]  alpha  Weight of the matrix product
     * @param[in]  beta   Weight of matrix C
     */
    void configure(const IGCTensor *a, const IGCTensor *b, const IGCTensor *c, IGCTensor *output, float alpha, float beta);

    // Inherited methods overridden:
    void run() override;

private:
    GCGEMMInterleave4x4Kernel  _interleave_kernel;
    GCGEMMTranspose1xWKernel   _transpose_kernel;
    GCGEMMMatrixMultiplyKernel _mm_kernel;
    GCGEMMMatrixAdditionKernel _ma_kernel;
    GCTensor                   _tmp_a;
    GCTensor                   _tmp_b;
    bool                       _is_interleaved_transposed;
    bool                       _run_addition;
};
}

#endif /* __ARM_COMPUTE_GCGEMM_H__ */
