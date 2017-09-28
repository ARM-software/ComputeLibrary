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
#ifndef __ARM_COMPUTE_CLGEMMMATRIXADDITIONKERNEL_H__
#define __ARM_COMPUTE_CLGEMMMATRIXADDITIONKERNEL_H__

#include "arm_compute/core/CL/ICLKernel.h"

namespace arm_compute
{
class ICLTensor;

/** OpenCL kernel to perform the in-place matrix addition between 2 matrices, taking into account that the second matrix might be weighted by a scalar value beta.
 *  The matrices must have the same dimensions
 *
 * @note This kernel is computed if and only if beta != 0.0.
 */
class CLGEMMMatrixAdditionKernel : public ICLKernel
{
public:
    /** Default constructor */
    CLGEMMMatrixAdditionKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixAdditionKernel(const CLGEMMMatrixAdditionKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    CLGEMMMatrixAdditionKernel &operator=(const CLGEMMMatrixAdditionKernel &) = delete;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixAdditionKernel(CLGEMMMatrixAdditionKernel &&) = default;
    /** Allow instances of this class to be moved */
    CLGEMMMatrixAdditionKernel &operator=(CLGEMMMatrixAdditionKernel &&) = default;
    /** Initialise the kernel's input, output and beta value
     *
     * @note The input and output tensors must have the same dimensions
     *
     * @param[in]      input  Input tensor (Matrix C). Data types supported: QS8/QS16/F16/F32
     * @param[in, out] output Output tensor. If this kernel is used to finalize the GEMM result (alpha * AB + beta * C), output must contain the result obtained by @ref CLGEMMMatrixMultiplyKernel. Data type supported: same as @p input
     * @param[in]      beta   Weight of matrix C
     */
    void configure(const ICLTensor *input, ICLTensor *output, float beta);

    // Inherited methods overridden:
    void run(const Window &window, cl::CommandQueue &queue) override;

private:
    const ICLTensor *_input;
    ICLTensor       *_output;
};
} // namespace arm_compute
#endif /* __ARM_COMPUTE_CLGEMMMATRIXADDITIONKERNEL_H__ */
